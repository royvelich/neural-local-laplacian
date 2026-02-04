import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class LossContext:
    """
    Bundles all tensors that any loss function might need.

    Built once per training step and passed to every loss module.
    Each loss reads the fields it needs and ignores the rest.

    Fields:
        predicted_mcv: Predicted mean curvature vectors (batch_size, 3)
        target_mcv: Target mean curvature vectors (batch_size, 3)
        grad_coeffs: Learned gradient coefficients (batch_size, max_k, 3) — gradient mode only
        positions: Neighbor positions, batched (batch_size, max_k, 3)
        normals: Surface normals at patch centers (batch_size, 3)
        attention_mask: Valid token mask (batch_size, max_k)
    """
    predicted_mcv: torch.Tensor
    target_mcv: torch.Tensor
    grad_coeffs: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    normals: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


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


class VectorMSELoss(nn.Module):
    """
    Standard MSE loss between two vectors.
    This is equivalent to the loss currently used in SurfaceTransformerModule.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initialize the VectorMSELoss.

        Args:
            reduction: Specifies the reduction to apply to the output.
                      'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target mean curvature vectors.

        Args:
            ctx: LossContext containing predicted_mcv and target_mcv of shape (batch_size, 3)

        Returns:
            MSE loss between the vectors
        """
        return F.mse_loss(ctx.predicted_mcv, ctx.target_mcv, reduction=self.reduction)


class DirectionMSELoss(nn.Module):
    """
    MSE loss between normalized vectors (direction only, ignoring magnitude).
    Computes loss between unit vectors to focus only on directional alignment.
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        """
        Initialize the DirectionMSELoss.

        Args:
            reduction: Specifies the reduction to apply to the output.
                      'mean' | 'sum' | 'none'
            eps: Small epsilon value to avoid division by zero during normalization
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute MSE loss between normalized vectors (direction only).

        Args:
            ctx: LossContext containing predicted_mcv and target_mcv of shape (batch_size, 3)

        Returns:
            MSE loss between normalized vectors
        """
        # Normalize both vectors to unit length
        predicted_norm = F.normalize(ctx.predicted_mcv, p=2, dim=1, eps=self.eps)
        target_norm = F.normalize(ctx.target_mcv, p=2, dim=1, eps=self.eps)

        return F.mse_loss(predicted_norm, target_norm, reduction=self.reduction)


class DirectionCosineLoss(nn.Module):
    """
    Cosine similarity loss between normalized vectors (direction only, ignoring magnitude).
    Loss = 1 - cosine_similarity, so loss is 0 when vectors point in same direction,
    and loss is 2 when vectors point in opposite directions.
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        """
        Initialize the DirectionCosineLoss.

        Args:
            reduction: Specifies the reduction to apply to the output.
                      'mean' | 'sum' | 'none'
            eps: Small epsilon value to avoid division by zero during normalization
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute cosine similarity loss between normalized vectors (direction only).

        Args:
            ctx: LossContext containing predicted_mcv and target_mcv of shape (batch_size, 3)

        Returns:
            Cosine similarity loss: 1 - cosine_similarity
            - Loss = 0 when vectors point in same direction (cos = 1)
            - Loss = 1 when vectors are orthogonal (cos = 0)
            - Loss = 2 when vectors point in opposite directions (cos = -1)
        """
        # Normalize both vectors to unit length
        predicted_norm = F.normalize(ctx.predicted_mcv, p=2, dim=1, eps=self.eps)
        target_norm = F.normalize(ctx.target_mcv, p=2, dim=1, eps=self.eps)

        # Compute cosine similarity: dot product of normalized vectors
        cosine_similarity = torch.sum(predicted_norm * target_norm, dim=1)  # Shape: (batch_size,)

        # Convert to loss: 1 - cosine_similarity
        # This gives us:
        # - Loss = 0 when cosine_similarity = 1 (same direction)
        # - Loss = 1 when cosine_similarity = 0 (orthogonal)
        # - Loss = 2 when cosine_similarity = -1 (opposite direction)
        cosine_loss = 1.0 - cosine_similarity  # Shape: (batch_size,)

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(cosine_loss)
        elif self.reduction == 'sum':
            return torch.sum(cosine_loss)
        elif self.reduction == 'none':
            return cosine_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class MagnitudeMSELoss(nn.Module):
    """
    MSE loss between vector magnitudes (scale only, ignoring direction).
    Computes loss between the L2 norms of the vectors.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initialize the MagnitudeMSELoss.

        Args:
            reduction: Specifies the reduction to apply to the output.
                      'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute MSE loss between vector magnitudes.

        Args:
            ctx: LossContext containing predicted_mcv and target_mcv of shape (batch_size, 3)

        Returns:
            MSE loss between vector magnitudes
        """
        # Compute L2 norms (magnitudes) of both vectors
        predicted_magnitude = torch.norm(ctx.predicted_mcv, p=2, dim=1)  # (batch_size,)
        target_magnitude = torch.norm(ctx.target_mcv, p=2, dim=1)  # (batch_size,)

        return F.mse_loss(predicted_magnitude, target_magnitude, reduction=self.reduction)


class RelativeMagnitudeLoss(nn.Module):
    """
    Relative MSE loss between vector magnitudes.
    Computes ((||pred|| - ||target||) / ||target||)^2.

    This ensures that samples with small magnitudes (low curvature regions)
    contribute equally to the loss when they have the same relative error
    as samples with large magnitudes.
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        """
        Initialize the RelativeMagnitudeLoss.

        Args:
            reduction: Specifies the reduction to apply to the output.
                      'mean' | 'sum' | 'none'
            eps: Small epsilon to avoid division by zero for near-zero targets
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute relative MSE loss between vector magnitudes.

        Args:
            ctx: LossContext containing predicted_mcv and target_mcv of shape (batch_size, 3)

        Returns:
            Relative MSE loss: mean/sum of ((||pred|| - ||target||) / ||target||)^2
        """
        predicted_magnitude = torch.norm(ctx.predicted_mcv, p=2, dim=1)  # (batch_size,)
        target_magnitude = torch.norm(ctx.target_mcv, p=2, dim=1)  # (batch_size,)

        # Relative error: (pred - target) / target
        relative_error = (predicted_magnitude - target_magnitude) / (target_magnitude + self.eps)
        relative_error_sq = relative_error ** 2

        if self.reduction == 'mean':
            return torch.mean(relative_error_sq)
        elif self.reduction == 'sum':
            return torch.sum(relative_error_sq)
        elif self.reduction == 'none':
            return relative_error_sq
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class LogMagnitudeLoss(nn.Module):
    """
    Log-space MSE loss between vector magnitudes.
    Computes (log(||pred||) - log(||target||))^2 = (log(||pred|| / ||target||))^2.

    This is scale-invariant and symmetric: a 2x overestimate and 2x underestimate
    produce the same loss. Particularly useful when magnitudes span multiple
    orders of magnitude (e.g., curvature values).
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        """
        Initialize the LogMagnitudeLoss.

        Args:
            reduction: Specifies the reduction to apply to the output.
                      'mean' | 'sum' | 'none'
            eps: Small epsilon to avoid log(0) for near-zero magnitudes
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute log-space MSE loss between vector magnitudes.

        Args:
            ctx: LossContext containing predicted_mcv and target_mcv of shape (batch_size, 3)

        Returns:
            Log-space MSE loss: mean/sum of (log(||pred||) - log(||target||))^2
        """
        predicted_magnitude = torch.norm(ctx.predicted_mcv, p=2, dim=1)  # (batch_size,)
        target_magnitude = torch.norm(ctx.target_mcv, p=2, dim=1)  # (batch_size,)

        # Log-space difference: log(pred) - log(target) = log(pred/target)
        log_pred = torch.log(predicted_magnitude + self.eps)
        log_target = torch.log(target_magnitude + self.eps)
        log_error_sq = (log_pred - log_target) ** 2

        if self.reduction == 'mean':
            return torch.mean(log_error_sq)
        elif self.reduction == 'sum':
            return torch.sum(log_error_sq)
        elif self.reduction == 'none':
            return log_error_sq
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class TangentPlaneProjectorLoss(nn.Module):
    """
    Gradient supervision loss via the tangent plane projector.

    For a surface with normal n̂ at vertex i, the surface gradient of coordinate
    functions x, y, z gives the tangent plane projector P = I - n̂n̂^T.

    The predicted gradient of coordinates is:
        predicted_P[d, c] = Σ_j g_ij[d] * p_j[c]

    where g_ij ∈ ℝ³ are the learned gradient coefficients and p_j ∈ ℝ³ are
    neighbor positions (relative to the patch center).

    This gives 9 constraints (6 independent due to symmetry of P) on 3k unknowns
    per patch. The system is underdetermined — the model has freedom to arrange
    coefficients optimally while satisfying the tangent plane constraint.

    Args:
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute MSE between predicted and target tangent plane projectors.

        Reads from ctx: grad_coeffs, positions, normals, attention_mask.

        Args:
            ctx: LossContext with gradient-mode fields populated

        Returns:
            Scalar loss (or per-sample if reduction='none')
        """
        # Mask gradient coefficients at padded positions
        mask_3d = ctx.attention_mask.unsqueeze(-1).float()  # (batch_size, max_k, 1)
        grad_masked = ctx.grad_coeffs * mask_3d

        # Predicted projector: P_pred[b, d, c] = Σ_k g[b,k,d] * pos[b,k,c]
        predicted_P = torch.einsum('bkd,bkc->bdc', grad_masked, ctx.positions)  # (batch_size, 3, 3)

        # Target projector: P = I - n̂n̂^T
        normals = F.normalize(ctx.normals, p=2, dim=1)  # (batch_size, 3)
        I = torch.eye(3, device=normals.device, dtype=normals.dtype).unsqueeze(0)  # (1, 3, 3)
        target_P = I - torch.einsum('bi,bj->bij', normals, normals)  # (batch_size, 3, 3)

        return F.mse_loss(predicted_P, target_P, reduction=self.reduction)