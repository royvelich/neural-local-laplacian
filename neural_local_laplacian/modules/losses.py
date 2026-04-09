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
        grad_coeffs: Learned gradient coefficients (batch_size, max_k, 3) â€” gradient mode only
        positions: Neighbor positions, batched (batch_size, max_k, 3)
        normals: Surface normals at patch centers (batch_size, 3)
        attention_mask: Valid token mask (batch_size, max_k)
    """
    predicted_mcv: torch.Tensor
    target_mcv: torch.Tensor
    predicted_raw_mcv: Optional[torch.Tensor] = None  # areas * predicted_mcv (before area division)
    grad_coeffs: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    normals: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    areas: Optional[torch.Tensor] = None
    stiffness_weights: Optional[torch.Tensor] = None
    gt_vertex_areas: Optional[torch.Tensor] = None


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


class AreaWeightedMSELoss(nn.Module):
    """
    Area-weighted MSE loss between predicted and target mean curvature vectors.

    Loss:
        L = (1/N) Σ_i  A_i · || pred_mcv_i - target_mcv_i ||^2

    Weights each vertex's MCV error by its area, so vertices representing
    more surface contribute proportionally more. Uses gt_vertex_areas when
    available (synthetic data), falls back to predicted areas otherwise.
    """

    def __init__(self, use_gt_areas: bool = True):
        """
        Args:
            use_gt_areas: If True, use gt_vertex_areas when available.
                         If False, always use predicted areas.
        """
        super().__init__()
        self.use_gt_areas = use_gt_areas

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.areas is None:
            raise ValueError("AreaWeightedMSELoss requires ctx.areas to be set")

        # Choose area weights
        if self.use_gt_areas and ctx.gt_vertex_areas is not None:
            weights = ctx.gt_vertex_areas
        else:
            weights = ctx.areas

        # Per-vertex squared error: ||pred - target||^2, shape (batch_size,)
        sq_error = ((ctx.predicted_mcv - ctx.target_mcv) ** 2).sum(dim=1)

        # Area-weighted mean
        return (weights * sq_error).mean()


class StiffnessActionLoss(nn.Module):
    """
    Direct supervision of the stiffness matrix action on positions.

    Compares the raw Laplacian action (before area division):
        pred: A_i^pred * MCV_i^pred  =  sum_j w_ij (p_j - p_i)
        gt:   A_i^gt   * MCV_i^gt

    Loss:
        L = mean_i || A_i^pred * MCV_i^pred  -  A_i^gt * MCV_i^gt ||^2

    This directly supervises the stiffness weights' action on positions,
    decoupling from area head accuracy. Requires gt_vertex_areas.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.predicted_raw_mcv is None:
            raise ValueError("StiffnessActionLoss requires ctx.predicted_raw_mcv")
        if ctx.gt_vertex_areas is None:
            raise ValueError("StiffnessActionLoss requires ctx.gt_vertex_areas")

        # Target: A_gt * MCV_gt
        target_raw = ctx.gt_vertex_areas[:, None] * ctx.target_mcv

        sq_error = ((ctx.predicted_raw_mcv - target_raw) ** 2).sum(dim=1)

        if self.reduction == 'mean':
            return torch.mean(sq_error)
        elif self.reduction == 'sum':
            return torch.sum(sq_error)
        elif self.reduction == 'none':
            return sq_error
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


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


class AreaOnlyLogMagnitudeLoss(nn.Module):
    """
    Log-magnitude loss that backpropagates ONLY through the area head.

    The MCV magnitude is ||mcv|| = ||Σ w_ij p_j|| / a_i.
    Standard LogMagnitudeLoss backprops through both w_ij (stiffness/gradient)
    and a_i (area head), causing magnitude and direction losses to conflict.

    This loss detaches the stiffness contribution (numerator), so gradients
    flow only through a_i. The area head learns to scale M correctly without
    disturbing the weight ratios that DirectionCosineLoss carefully trained.

    Mathematically:
        loss = (log(||detach(Σ w_ij p_j)|| / a_i) - log(||target_mcv||))^2
             = (log(||detach(Σ w_ij p_j)||) - log(a_i) - log(||target_mcv||))^2

    Only d/d(a_i) is non-zero.
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute log-magnitude loss with gradients only through areas.

        Requires ctx.areas to be set (predicted areas from the area head).
        """
        if ctx.areas is None:
            raise ValueError("AreaOnlyLogMagnitudeLoss requires ctx.areas to be set")

        # Reconstruct stiffness_sum = mcv * areas, then detach
        # This removes gradient flow through g_ij / stiffness_weights
        stiffness_sum = ctx.predicted_mcv * ctx.areas.unsqueeze(-1)  # (batch_size, 3)
        stiffness_sum_detached = stiffness_sum.detach()  # no grad through weights

        # Recompute magnitude with detached numerator: ||detach(sum)|| / areas
        numerator_magnitude = torch.norm(stiffness_sum_detached, p=2, dim=1)  # (batch_size,)
        predicted_magnitude = numerator_magnitude / (ctx.areas + self.eps)  # (batch_size,)

        target_magnitude = torch.norm(ctx.target_mcv, p=2, dim=1)  # (batch_size,)

        # Log-space MSE (same as LogMagnitudeLoss)
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


class MeanCurvatureOnlyLogMagnitudeLoss(nn.Module):
    """
    Log-magnitude loss that backpropagates ONLY through the gradient head.

    The MCV magnitude is ||mcv|| = ||Σ w_ij p_j|| / a_i.
    This loss detaches the area contribution (denominator), so gradients
    flow only through w_ij (gradient coefficients). The gradient head learns
    to produce the correct stiffness action magnitude without disturbing
    the area head.

    Mathematically:
        loss = (log(||Σ w_ij p_j|| / detach(a_i)) - log(||target_mcv||))^2

    Only d/d(w_ij) is non-zero.
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.areas is None:
            raise ValueError("MeanCurvatureOnlyLogMagnitudeLoss requires ctx.areas")
        if ctx.predicted_raw_mcv is None:
            raise ValueError("MeanCurvatureOnlyLogMagnitudeLoss requires ctx.predicted_raw_mcv")

        # predicted_raw_mcv = Σ w_ij (p_j - p_i), no areas in the graph
        # Divide by detached areas so only gradient head gets gradients
        areas_detached = ctx.areas.detach()
        predicted_magnitude = torch.norm(ctx.predicted_raw_mcv, p=2, dim=1) / (areas_detached + self.eps)

        target_magnitude = torch.norm(ctx.target_mcv, p=2, dim=1)

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


class AreaSupervisionLoss(nn.Module):
    """
    Direct area supervision using MSE against GT barycentric areas.

    Loss:
        L = mean_i (a_i^pred - a_i^gt)^2

    Requires gt_vertex_areas in the LossContext (set by synthetic datasets).
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.gt_vertex_areas is None:
            raise ValueError("AreaSupervisionLoss requires ctx.gt_vertex_areas to be set")
        if ctx.areas is None:
            raise ValueError("AreaSupervisionLoss requires ctx.areas to be set")

        error_sq = (ctx.areas - ctx.gt_vertex_areas) ** 2

        if self.reduction == 'mean':
            return torch.mean(error_sq)
        elif self.reduction == 'sum':
            return torch.sum(error_sq)
        elif self.reduction == 'none':
            return error_sq
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class LogAreaSupervisionLoss(nn.Module):
    """
    Direct area supervision using log-space MSE against GT barycentric areas.

    Loss:
        L = mean_i (log(a_i^pred) - log(a_i^gt))^2

    Log space ensures equal relative-error penalty across vertices
    regardless of absolute area magnitude.

    Requires gt_vertex_areas in the LossContext (set by synthetic datasets).
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.gt_vertex_areas is None:
            raise ValueError("LogAreaSupervisionLoss requires ctx.gt_vertex_areas to be set")
        if ctx.areas is None:
            raise ValueError("LogAreaSupervisionLoss requires ctx.areas to be set")

        log_pred = torch.log(ctx.areas + self.eps)
        log_gt = torch.log(ctx.gt_vertex_areas + self.eps)
        log_error_sq = (log_pred - log_gt) ** 2

        if self.reduction == 'mean':
            return torch.mean(log_error_sq)
        elif self.reduction == 'sum':
            return torch.sum(log_error_sq)
        elif self.reduction == 'none':
            return log_error_sq
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class GradientL2RegularizationLoss(nn.Module):
    """
    L2 regularization on gradient coefficients to prevent scale drift.

    Without this, g_ij can grow unboundedly because cosine loss is
    scale-invariant and AreaOnlyLogMagnitudeLoss detaches the numerator.
    This anchors ||g_ij|| to a reasonable magnitude.

    Loss:
        L = mean_i (1/k) sum_j ||g_ij||^2
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.grad_coeffs is None:
            raise ValueError("GradientL2RegularizationLoss requires ctx.grad_coeffs")

        # ||g_ij||^2 averaged over all (i, j) pairs
        # grad_coeffs: (batch_size, max_k, 3)
        sq_norms = (ctx.grad_coeffs ** 2).sum(dim=2)  # (batch_size, max_k)

        if ctx.attention_mask is not None:
            sq_norms = sq_norms.masked_fill(~ctx.attention_mask, 0.0)
            n_valid = ctx.attention_mask.sum()
            if n_valid > 0:
                return sq_norms.sum() / n_valid
            return sq_norms.sum() * 0.0  # no valid entries

        if self.reduction == 'mean':
            return sq_norms.mean()
        elif self.reduction == 'sum':
            return sq_norms.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class TangentPlaneProjectorLoss(nn.Module):
    """
    Gradient supervision loss via the tangent plane projector.

    For a surface with normal nÌ‚ at vertex i, the surface gradient of coordinate
    functions x, y, z gives the tangent plane projector P = I - nÌ‚nÌ‚^T.

    The predicted gradient of coordinates is:
        predicted_P[d, c] = Î£_j g_ij[d] * p_j[c]

    where g_ij âˆˆ â„Â³ are the learned gradient coefficients and p_j âˆˆ â„Â³ are
    neighbor positions (relative to the patch center).

    This gives 9 constraints (6 independent due to symmetry of P) on 3k unknowns
    per patch. The system is underdetermined â€” the model has freedom to arrange
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

        # Predicted projector: P_pred[b, d, c] = Î£_k g[b,k,d] * pos[b,k,c]
        predicted_P = torch.einsum('bkd,bkc->bdc', grad_masked, ctx.positions)  # (batch_size, 3, 3)

        # Target projector: P = I - nÌ‚nÌ‚^T
        normals = F.normalize(ctx.normals, p=2, dim=1)  # (batch_size, 3)
        I = torch.eye(3, device=normals.device, dtype=normals.dtype).unsqueeze(0)  # (1, 3, 3)
        target_P = I - torch.einsum('bi,bj->bij', normals, normals)  # (batch_size, 3, 3)

        return F.mse_loss(predicted_P, target_P, reduction=self.reduction)


class DirichletEnergyConsistencyLoss(nn.Module):
    """
    Per-patch Dirichlet energy density consistency between S-path and G-path.

    For a probe function f with per-neighbor differences δf_j = f_j - f_i,
    the energy density at patch i can be computed two ways:

        S-path:  E_S = Σ_j s_ij · δf_j²  =  Σ_j ||g_ij||² · δf_j²
        G-path:  E_G = a_i · ||Σ_j g_ij · δf_j||²

    These agree when:
      (1) the off-diagonal Gram terms g_ij · g_ik are small, and
      (2) the area prediction a_i is consistent with the gradient scale.

    The loss penalizes the relative discrepancy across probe functions:

        L = (1/BP) Σ_i Σ_p (E_S - E_G)² / (E_S² + ε)

    Probe functions:
      - 'coordinates': uses neighbor positions (δf = p_j) — 3 probes (x, y, z).
        Couples with MCV supervision since Δx = 2Hn.
      - 'random': samples δf ~ N(0,1) — tests consistency on arbitrary functions.
        Gives genuinely new signal beyond MCV/TPP.

    Gradient-mode only. Requires: grad_coeffs, attention_mask, areas.
    Additionally requires positions when probe_mode='coordinates'.

    Args:
        probe_mode: 'random' or 'coordinates'
        num_random_probes: Number of random probe functions (only for probe_mode='random')
        reduction: 'mean' | 'sum' | 'none'
        eps: Small constant for numerical stability in denominator
    """

    def __init__(self, probe_mode: str = 'random', num_random_probes: int = 8,
                 reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        if probe_mode not in ('random', 'coordinates'):
            raise ValueError(f"probe_mode must be 'random' or 'coordinates', got '{probe_mode}'")
        self.probe_mode = probe_mode
        self.num_random_probes = num_random_probes
        self.reduction = reduction
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute Dirichlet energy consistency loss.

        Args:
            ctx: LossContext with grad_coeffs, attention_mask, areas, and
                 positions (if probe_mode='coordinates')

        Returns:
            Scalar loss (or per-sample if reduction='none')
        """
        grad_coeffs = ctx.grad_coeffs       # (B, K, 3)
        mask = ctx.attention_mask            # (B, K)
        areas = ctx.areas                    # (B,)

        mask_float = mask.float()            # (B, K)
        mask_3d = mask_float.unsqueeze(-1)   # (B, K, 1)

        # Masked gradient coefficients
        g = grad_coeffs * mask_3d            # (B, K, 3)

        # Stiffness weights: s_ij = ||g_ij||²
        s = (g ** 2).sum(dim=-1)             # (B, K)

        # Build probe delta-f values: (B, K, P)
        if self.probe_mode == 'coordinates':
            # δf_j = p_j (positions relative to center) — 3 probes
            delta_f = ctx.positions * mask_3d  # (B, K, 3) — P=3
        else:
            # Random probes: δf ~ N(0, 1)
            P = self.num_random_probes
            delta_f = torch.randn(
                grad_coeffs.shape[0], grad_coeffs.shape[1], P,
                device=grad_coeffs.device, dtype=grad_coeffs.dtype
            ) * mask_float.unsqueeze(-1)       # (B, K, P)

        # S-path energy: E_S = Σ_j s_ij · δf_j²  per probe
        # s: (B, K), delta_f²: (B, K, P)
        E_S = (s.unsqueeze(-1) * delta_f ** 2).sum(dim=1)  # (B, P)

        # G-path energy: E_G = a_i · ||Σ_j g_ij · δf_j||²  per probe
        # g: (B, K, 3), delta_f: (B, K, P)
        # grad_f = Σ_j g_ij * δf_j → (B, 3, P) via einsum
        grad_f = torch.einsum('bkd,bkp->bdp', g, delta_f)  # (B, 3, P)
        E_G = areas.unsqueeze(-1) * (grad_f ** 2).sum(dim=1)  # (B, P)

        # Relative squared error: (E_S - E_G)² / (E_S² + ε)
        error = (E_S - E_G) ** 2 / (E_S ** 2 + self.eps)  # (B, P)

        # Reduce over probes first, then over batch
        per_patch = error.mean(dim=-1)  # (B,)

        if self.reduction == 'mean':
            return per_patch.mean()
        elif self.reduction == 'sum':
            return per_patch.sum()
        elif self.reduction == 'none':
            return per_patch
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class GramOffDiagonalLoss(nn.Module):
    """
    Penalizes off-diagonal entries in the local Gram matrix of gradient coefficients.

    The local Gram matrix at patch i is:

        G_i ∈ R^{K×K},   (G_i)_{jk} = g_ij · g_ik

    The diagonal entries are the stiffness weights: (G_i)_{jj} = ||g_ij||² = s_ij.
    The off-diagonal entries (G_i)_{jk} for j≠k represent inter-edge coupling.

    The model derives s_ij = ||g_ij||², which corresponds to taking only the
    diagonal of the Gram matrix. This is exact when the off-diagonals are zero,
    i.e., when the gradient coefficient vectors are mutually orthogonal.

    The loss penalizes the (scale-invariant) off-diagonal energy:

        L = (1/B) Σ_i  Σ_{j≠k} m_j m_k (g_ij · g_ik)² / (Σ_j m_j ||g_ij||²)² + ε)

    Note: perfect orthogonality is impossible when K > 3 (can't have more than 3
    mutually orthogonal vectors in R³), but encouraging small cross-terms still
    makes the diagonal approximation more faithful.

    Gradient-mode only. Requires: grad_coeffs, attention_mask.

    Args:
        reduction: 'mean' | 'sum' | 'none'
        eps: Small constant for numerical stability in denominator
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute Gram off-diagonal regularization loss.

        Args:
            ctx: LossContext with grad_coeffs and attention_mask

        Returns:
            Scalar loss (or per-sample if reduction='none')
        """
        grad_coeffs = ctx.grad_coeffs       # (B, K, 3)
        mask = ctx.attention_mask            # (B, K)

        mask_float = mask.float()            # (B, K)
        mask_3d = mask_float.unsqueeze(-1)   # (B, K, 1)

        # Masked gradient coefficients
        g = grad_coeffs * mask_3d            # (B, K, 3)

        # Full Gram matrix: (G_i)_{jk} = g_ij · g_ik
        gram = torch.einsum('bjd,bkd->bjk', g, g)  # (B, K, K)

        # Mask: only count valid pairs (both j and k are real neighbors)
        pair_mask = mask_float.unsqueeze(-1) * mask_float.unsqueeze(-2)  # (B, K, K)

        # Zero out diagonal to get off-diagonal entries only
        diag_mask = torch.eye(gram.shape[1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
        off_diag_mask = pair_mask * (1.0 - diag_mask)  # (B, K, K)

        # Sum of squared off-diagonal Gram entries per patch
        off_diag_sq = (gram ** 2 * off_diag_mask).sum(dim=(1, 2))  # (B,)

        # Normalization: (Σ_j m_j ||g_ij||²)²
        stiffness_sum = (mask_float * (g ** 2).sum(dim=-1)).sum(dim=1)  # (B,)
        denom = stiffness_sum ** 2 + self.eps  # (B,)

        per_patch = off_diag_sq / denom  # (B,)

        if self.reduction == 'mean':
            return per_patch.mean()
        elif self.reduction == 'sum':
            return per_patch.sum()
        elif self.reduction == 'none':
            return per_patch
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

class WeightEntropyLoss(nn.Module):
    """
    Sparsity-inducing loss that penalizes high entropy in stiffness weight distributions.

    Normalizes the per-patch stiffness weights into a probability distribution
    and computes the Shannon entropy. High entropy means weights are spread
    uniformly across all k neighbors (dense Laplacian, slow downstream solves).
    Low entropy means weight is concentrated on a few important neighbors
    (sparse Laplacian, enabling effective top-k pruning).

    The loss is the normalized entropy: H / log(k), so it's in [0, 1] regardless of k.
    At entropy=0, all weight is on one neighbor. At entropy=1, weights are perfectly uniform.

    Typical values before training with this loss: ~0.97 (nearly uniform).
    Target after training: ~0.5-0.7 (concentrated but not degenerate).
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-12):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute normalized entropy of stiffness weight distribution.

        Args:
            ctx: LossContext with stiffness_weights and attention_mask

        Returns:
            Scalar loss in [0, 1] (normalized entropy)
        """
        w = ctx.stiffness_weights  # (B, k), positive
        mask = ctx.attention_mask   # (B, k)

        # Mask invalid entries
        w = w.masked_fill(~mask, 0.0)

        # Normalize to probability distribution per patch
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        p = w / w_sum  # (B, k)

        # Shannon entropy: H = -sum(p * log(p))
        log_p = torch.where(p > self.eps, p.log(), torch.zeros_like(p))
        entropy = -(p * log_p).sum(dim=-1)  # (B,)

        # Normalize by max entropy log(k) so loss is in [0, 1]
        k = mask.sum(dim=-1).float().clamp(min=1.0)  # actual k per patch
        max_entropy = k.log().clamp(min=self.eps)
        normalized_entropy = entropy / max_entropy  # (B,)

        if self.reduction == 'mean':
            return normalized_entropy.mean()
        elif self.reduction == 'sum':
            return normalized_entropy.sum()
        elif self.reduction == 'none':
            return normalized_entropy
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class WeightL1Loss(nn.Module):
    """
    L1 sparsity loss on stiffness weights.

    Penalizes the mean absolute value of weights, encouraging small weights
    to go to zero. Simpler than entropy but applies uniform pressure on all
    weights rather than encouraging a peaked distribution.

    The loss is the coefficient of variation (std/mean), which is
    scale-invariant and measures relative spread. For uniform weights CV ~ 0,
    for peaked weights CV >> 0. We return 1/(1+CV) so the loss is in [0, 1]
    and lower is sparser.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, ctx: LossContext) -> torch.Tensor:
        """
        Compute inverse coefficient of variation of weights per patch.

        Args:
            ctx: LossContext with stiffness_weights and attention_mask

        Returns:
            Scalar loss in [0, 1] (lower = more peaked weights)
        """
        w = ctx.stiffness_weights  # (B, k), positive
        mask = ctx.attention_mask   # (B, k)

        # Mask invalid entries
        w = w.masked_fill(~mask, 0.0)
        k = mask.sum(dim=-1).float().clamp(min=1.0)  # (B,)

        # Per-patch mean and std
        w_sum = w.sum(dim=-1)
        w_mean = w_sum / k  # (B,)
        w_sq_sum = (w ** 2).sum(dim=-1)
        w_var = w_sq_sum / k - w_mean ** 2  # (B,)
        w_std = w_var.clamp(min=0).sqrt()  # (B,)

        # Coefficient of variation: std/mean
        cv = w_std / w_mean.clamp(min=1e-12)  # (B,)

        # Inverse: 1/(1+CV), so lower = more spread, higher = more peaked
        # We want to MINIMIZE this (penalize low CV = uniform weights)
        inv_cv = 1.0 / (1.0 + cv)  # (B,), in [0, 1]

        if self.reduction == 'mean':
            return inv_cv.mean()
        elif self.reduction == 'sum':
            return inv_cv.sum()
        elif self.reduction == 'none':
            return inv_cv
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")