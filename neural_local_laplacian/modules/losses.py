import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class LossConfig:
    """Configuration for a loss module with its associated weight.

    If weight is None, the loss is computed and logged but not included in backprop.
    """
    loss_module: nn.Module
    weight: Optional[float]

    def __post_init__(self):
        """Validate the loss configuration after initialization."""
        if not isinstance(self.loss_module, nn.Module):
            raise ValueError(f"loss_module must be a nn.Module, got {type(self.loss_module)}")
        if self.weight is not None and (not isinstance(self.weight, (int, float)) or self.weight < 0):
            raise ValueError(f"weight must be None or a non-negative number, got {self.weight}")


# =============================================================================
# Base class and utilities
# =============================================================================

class ThreeHeadLossBase(nn.Module):
    """
    Base class for 3-head architecture losses.

    All losses share the same forward() signature to enable easy iteration.
    Each loss uses only the inputs it needs.
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def _compute_weighted_position_sum(self,
                                       normal_weights: torch.Tensor,
                                       attention_mask: torch.Tensor,
                                       batch_sizes: torch.Tensor,
                                       positions: torch.Tensor) -> torch.Tensor:
        """
        Compute sum_j w_ij * p_j for each patch.

        Args:
            normal_weights: (batch_size, max_k)
            attention_mask: (batch_size, max_k)
            batch_sizes: (batch_size,)
            positions: (total_points, 3)

        Returns:
            weighted_sum: (batch_size, 3)
        """
        batch_size = normal_weights.shape[0]
        max_k = normal_weights.shape[1]
        device = normal_weights.device
        dtype = positions.dtype

        # Apply attention mask
        masked_weights = normal_weights.masked_fill(~attention_mask, 0.0)
        weights_flat = masked_weights.flatten()

        # Create batch indices for flattened weights
        batch_indices_weights = torch.arange(batch_size, device=device).repeat_interleave(max_k)

        # Compute batch starts
        batch_cumsum = torch.cumsum(batch_sizes, dim=0)
        batch_starts = torch.cat([torch.zeros(1, device=device, dtype=batch_cumsum.dtype), batch_cumsum[:-1]])

        # Position indices for each weight
        position_indices = torch.arange(max_k, device=device).repeat(batch_size)

        # Filter valid indices
        valid_mask = position_indices < batch_sizes.repeat_interleave(max_k)
        valid_weights = weights_flat[valid_mask]
        valid_batch_indices = batch_indices_weights[valid_mask]
        valid_position_indices = position_indices[valid_mask]

        # Get actual position indices
        actual_position_indices = batch_starts[valid_batch_indices] + valid_position_indices
        valid_positions = positions[actual_position_indices.long()]

        # Compute weighted positions
        weighted_positions = valid_weights.unsqueeze(-1) * valid_positions

        # Sum for each batch
        weighted_sum = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        weighted_sum.scatter_add_(0, valid_batch_indices.unsqueeze(-1).expand(-1, 3), weighted_positions)

        return weighted_sum

    def _compute_predicted_normal_unnormalized(self,
                                               normal_weights: torch.Tensor,
                                               areas: torch.Tensor,
                                               positions: torch.Tensor,
                                               attention_mask: torch.Tensor,
                                               batch_sizes: torch.Tensor) -> torch.Tensor:
        """
        Compute the unnormalized predicted normal: (sum_j w_ij * p_j) / A_i

        This should be unit norm if training is successful.

        Returns:
            pred_normal_unnorm: (batch_size, 3)
        """
        weighted_sum = self._compute_weighted_position_sum(
            normal_weights, attention_mask, batch_sizes, positions
        )
        return weighted_sum / (areas.unsqueeze(-1) + self.eps)

    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def forward(self,
                normal_weights: torch.Tensor,
                areas: torch.Tensor,
                mean_curvatures: torch.Tensor,
                positions: torch.Tensor,
                attention_mask: torch.Tensor,
                batch_sizes: torch.Tensor,
                target_normals: torch.Tensor,
                target_curvatures: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.

        All losses have the same signature for easy iteration.
        Each loss uses only the inputs it needs.

        Args:
            normal_weights: (batch_size, max_k) - learned weights w_ij
            areas: (batch_size,) - learned areas A_i
            mean_curvatures: (batch_size,) - learned mean curvatures H_i
            positions: (total_points, 3) - neighbor positions (centered at origin)
            attention_mask: (batch_size, max_k) - True for real tokens
            batch_sizes: (batch_size,) - actual number of points per patch
            target_normals: (batch_size, 3) - GT unit normals
            target_curvatures: (batch_size,) - GT mean curvatures H

        Returns:
            Loss value (scalar if reduction='mean'/'sum', else (batch_size,))
        """
        raise NotImplementedError("Subclasses must implement forward()")


# =============================================================================
# Normal Direction Losses
# =============================================================================

class NormalDirectionCosineLoss(ThreeHeadLossBase):
    """
    Cosine similarity loss for normal direction.

    Computes:
        pred = normalize((sum_j w_ij * p_j) / A_i)
        L = 1 - cos(pred, n_GT)

    Only supervises direction, not magnitude.
    Loss = 0 when directions match, Loss = 2 when opposite.
    """

    def forward(self,
                normal_weights: torch.Tensor,
                areas: torch.Tensor,
                mean_curvatures: torch.Tensor,
                positions: torch.Tensor,
                attention_mask: torch.Tensor,
                batch_sizes: torch.Tensor,
                target_normals: torch.Tensor,
                target_curvatures: torch.Tensor) -> torch.Tensor:
        # Compute unnormalized prediction
        pred_unnorm = self._compute_predicted_normal_unnormalized(
            normal_weights, areas, positions, attention_mask, batch_sizes
        )

        # Normalize for direction comparison
        pred_norm = F.normalize(pred_unnorm, p=2, dim=1, eps=self.eps)
        target_norm = F.normalize(target_normals, p=2, dim=1, eps=self.eps)

        # Cosine similarity loss
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        loss = 1.0 - cos_sim

        return self._apply_reduction(loss)


class NormalMagnitudeLoss(ThreeHeadLossBase):
    """
    Magnitude loss to enforce unit norm constraint.

    Computes:
        pred = (sum_j w_ij * p_j) / A_i
        L = (||pred|| - 1)^2

    Ensures the predicted normal has unit magnitude.
    """

    def forward(self,
                normal_weights: torch.Tensor,
                areas: torch.Tensor,
                mean_curvatures: torch.Tensor,
                positions: torch.Tensor,
                attention_mask: torch.Tensor,
                batch_sizes: torch.Tensor,
                target_normals: torch.Tensor,
                target_curvatures: torch.Tensor) -> torch.Tensor:
        # Compute unnormalized prediction
        pred_unnorm = self._compute_predicted_normal_unnormalized(
            normal_weights, areas, positions, attention_mask, batch_sizes
        )

        # Magnitude should be 1
        pred_magnitude = torch.norm(pred_unnorm, p=2, dim=1)
        loss = (pred_magnitude - 1.0) ** 2

        return self._apply_reduction(loss)


class NormalMSELoss(ThreeHeadLossBase):
    """
    MSE loss between predicted and target normal.

    Computes:
        pred = (sum_j w_ij * p_j) / A_i
        L = ||pred - n_GT||^2

    Since n_GT is unit norm, this implicitly enforces both:
    - Correct direction (minimizes angle error)
    - Unit magnitude (pred should match unit-norm target)

    This is often the simplest and most effective choice.
    """

    def forward(self,
                normal_weights: torch.Tensor,
                areas: torch.Tensor,
                mean_curvatures: torch.Tensor,
                positions: torch.Tensor,
                attention_mask: torch.Tensor,
                batch_sizes: torch.Tensor,
                target_normals: torch.Tensor,
                target_curvatures: torch.Tensor) -> torch.Tensor:
        # Compute unnormalized prediction (should be unit norm)
        pred_unnorm = self._compute_predicted_normal_unnormalized(
            normal_weights, areas, positions, attention_mask, batch_sizes
        )

        # MSE against unit-norm target
        loss = F.mse_loss(pred_unnorm, target_normals, reduction='none').sum(dim=1)

        return self._apply_reduction(loss)


class NormalL1Loss(ThreeHeadLossBase):
    """
    L1 loss between predicted and target normal.

    More robust to outliers than MSE.

    Computes:
        pred = (sum_j w_ij * p_j) / A_i
        L = ||pred - n_GT||_1
    """

    def forward(self,
                normal_weights: torch.Tensor,
                areas: torch.Tensor,
                mean_curvatures: torch.Tensor,
                positions: torch.Tensor,
                attention_mask: torch.Tensor,
                batch_sizes: torch.Tensor,
                target_normals: torch.Tensor,
                target_curvatures: torch.Tensor) -> torch.Tensor:
        pred_unnorm = self._compute_predicted_normal_unnormalized(
            normal_weights, areas, positions, attention_mask, batch_sizes
        )

        loss = F.l1_loss(pred_unnorm, target_normals, reduction='none').sum(dim=1)

        return self._apply_reduction(loss)


# =============================================================================
# Mean Curvature Losses
# =============================================================================

class MeanCurvatureMSELoss(ThreeHeadLossBase):
    """
    MSE loss for mean curvature.

    Computes: L = (H_pred - H_GT)^2
    """

    def forward(self,
                normal_weights: torch.Tensor,
                areas: torch.Tensor,
                mean_curvatures: torch.Tensor,
                positions: torch.Tensor,
                attention_mask: torch.Tensor,
                batch_sizes: torch.Tensor,
                target_normals: torch.Tensor,
                target_curvatures: torch.Tensor) -> torch.Tensor:
        loss = (mean_curvatures - target_curvatures) ** 2
        return self._apply_reduction(loss)


class MeanCurvatureL1Loss(ThreeHeadLossBase):
    """
    L1 loss for mean curvature.

    More robust to outliers than MSE.
    Computes: L = |H_pred - H_GT|
    """

    def forward(self,
                normal_weights: torch.Tensor,
                areas: torch.Tensor,
                mean_curvatures: torch.Tensor,
                positions: torch.Tensor,
                attention_mask: torch.Tensor,
                batch_sizes: torch.Tensor,
                target_normals: torch.Tensor,
                target_curvatures: torch.Tensor) -> torch.Tensor:
        loss = torch.abs(mean_curvatures - target_curvatures)
        return self._apply_reduction(loss)


class MeanCurvatureHuberLoss(ThreeHeadLossBase):
    """
    Huber loss for mean curvature.

    Smooth combination of MSE (for small errors) and L1 (for large errors).
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8, delta: float = 1.0):
        super().__init__(reduction=reduction, eps=eps)
        self.delta = delta

    def forward(self,
                normal_weights: torch.Tensor,
                areas: torch.Tensor,
                mean_curvatures: torch.Tensor,
                positions: torch.Tensor,
                attention_mask: torch.Tensor,
                batch_sizes: torch.Tensor,
                target_normals: torch.Tensor,
                target_curvatures: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(mean_curvatures, target_curvatures,
                            reduction=self.reduction, delta=self.delta)


class MeanCurvatureRelativeLoss(ThreeHeadLossBase):
    """
    Relative error loss for mean curvature.

    Useful when curvatures span multiple orders of magnitude.
    Computes: L = |H_pred - H_GT| / (|H_GT| + eps)
    """

    def forward(self,
                normal_weights: torch.Tensor,
                areas: torch.Tensor,
                mean_curvatures: torch.Tensor,
                positions: torch.Tensor,
                attention_mask: torch.Tensor,
                batch_sizes: torch.Tensor,
                target_normals: torch.Tensor,
                target_curvatures: torch.Tensor) -> torch.Tensor:
        relative_error = torch.abs(mean_curvatures - target_curvatures) / (torch.abs(target_curvatures) + self.eps)
        return self._apply_reduction(relative_error)


# =============================================================================
# Legacy Vector Losses (for backward compatibility)
# These do NOT follow the unified signature - kept for old code
# =============================================================================

class VectorMSELoss(nn.Module):
    """Legacy: Standard MSE loss between two vectors."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predicted, target, reduction=self.reduction)


class DirectionMSELoss(nn.Module):
    """Legacy: MSE loss between normalized vectors."""

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted_norm = F.normalize(predicted, p=2, dim=1, eps=self.eps)
        target_norm = F.normalize(target, p=2, dim=1, eps=self.eps)
        return F.mse_loss(predicted_norm, target_norm, reduction=self.reduction)


class DirectionCosineLoss(nn.Module):
    """Legacy: Cosine similarity loss between vectors."""

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted_norm = F.normalize(predicted, p=2, dim=1, eps=self.eps)
        target_norm = F.normalize(target, p=2, dim=1, eps=self.eps)
        cosine_similarity = torch.sum(predicted_norm * target_norm, dim=1)
        cosine_loss = 1.0 - cosine_similarity

        if self.reduction == 'mean':
            return torch.mean(cosine_loss)
        elif self.reduction == 'sum':
            return torch.sum(cosine_loss)
        return cosine_loss


class MagnitudeMSELoss(nn.Module):
    """Legacy: MSE loss between vector magnitudes."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted_magnitude = torch.norm(predicted, p=2, dim=1)
        target_magnitude = torch.norm(target, p=2, dim=1)
        return F.mse_loss(predicted_magnitude, target_magnitude, reduction=self.reduction)