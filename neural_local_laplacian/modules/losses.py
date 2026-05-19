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
    # Test function probe data (from TestFunctionSampler) — patch-level
    test_func_deltas: Optional[torch.Tensor] = None      # (B, max_k, P)
    test_func_laplacians: Optional[torch.Tensor] = None   # (B, P)
    test_func_gradients: Optional[torch.Tensor] = None    # (B, P, 3)
    # Surface-level fields (variational training pipeline)
    knn: Optional[torch.Tensor] = None                              # (n, k)
    vertex_pos: Optional[torch.Tensor] = None                       # (n, 3)
    vertex_normals: Optional[torch.Tensor] = None                   # (n, 3)
    test_func_values: Optional[torch.Tensor] = None                 # (n, P)
    test_func_continuous_bilinear: Optional[torch.Tensor] = None    # (P, P)
    test_func_continuous_energy: Optional[torch.Tensor] = None      # (P,)
    test_func_gradients_at_vertices: Optional[torch.Tensor] = None  # (n, P, 3)
    test_func_laplacians_at_vertices: Optional[torch.Tensor] = None  # (n, P)
    # Heat-method geodesic supervision (variational pipeline)
    geodesic_sources: Optional[torch.Tensor] = None                 # (S,) vertex idx
    geodesic_distances: Optional[torch.Tensor] = None               # (S, n)
    # Heat-kernel supervision (variational pipeline)
    heat_kernel_times: Optional[torch.Tensor] = None                # (T,)
    heat_kernel_gt: Optional[torch.Tensor] = None                   # (T, n) or (T, n, n)
    heat_kernel_diagonal_only: Optional[torch.Tensor] = None        # 0-d bool


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

        Requires ctx.areas and ctx.predicted_raw_mcv.
        """
        if ctx.areas is None:
            raise ValueError("AreaOnlyLogMagnitudeLoss requires ctx.areas")
        if ctx.predicted_raw_mcv is None:
            raise ValueError("AreaOnlyLogMagnitudeLoss requires ctx.predicted_raw_mcv")

        # Detach stiffness action so gradients only flow through areas
        numerator_magnitude = torch.norm(ctx.predicted_raw_mcv.detach(), p=2, dim=1)
        predicted_magnitude = numerator_magnitude / (ctx.areas + self.eps)

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


class AreaEntropyRegularizer(nn.Module):
    """
    Regularizer that penalizes areas deviating from uniform distribution.

    Without ground-truth area supervision, the area head can drift to become
    a curvature-compensating scalar (absorbing the gradient head's unconstrained
    absolute scale). This regularizer softly encourages areas to stay close to
    a uniform distribution via KL divergence: KL(p || uniform), where
    p_i = A_i / sum(A).

    The KL divergence is zero when all areas are equal, and increases as
    areas become non-uniform. This prevents pathological drift while still
    allowing the area head to learn moderate per-vertex variation.

    Mathematically:
        KL(p || u) = sum_i p_i * log(p_i * N)

    where p_i = A_i / sum(A) and u_i = 1/N.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.areas is None:
            raise ValueError("AreaEntropyRegularizer requires ctx.areas")

        areas = ctx.areas
        n = len(areas)

        # Normalize areas to a probability distribution
        p = areas / (areas.sum() + self.eps)

        # KL(p || uniform) = sum(p_i * log(p_i * N))
        return (p * torch.log(p * n + self.eps)).sum()


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
        if probe_mode not in ('random', 'coordinates', 'test_functions'):
            raise ValueError(f"probe_mode must be 'random', 'coordinates', or 'test_functions', got '{probe_mode}'")
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
        elif self.probe_mode == 'test_functions':
            # Use precomputed test function deltas from dataset
            if ctx.test_func_deltas is None:
                raise ValueError("probe_mode='test_functions' requires ctx.test_func_deltas")
            delta_f = ctx.test_func_deltas * mask_3d  # (B, K, P)
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


class StiffnessNormalAlignmentLoss(nn.Module):
    """
    Self-supervised loss enforcing that the stiffness action points along the normal.

    Uses the predicted tangent plane projector P = Σ_j g_ij ⊗ p_jᵀ to check
    that the stiffness action Σ_j s_ij p_j has zero tangent component:

        L = mean_i || P_i · (Σ_j s_ij p_j) ||²

    If P is a valid projector (tangent plane) and the stiffness action points
    along the normal, then P · action = 0.

    No GT normal or MCV target needed — purely self-supervised from the
    learned g_ij vectors. Should be combined with TangentPlaneProjectorLoss
    (or ProjectorIdempotencyLoss) to ensure P is a valid projector, otherwise
    the model can trivially satisfy this by collapsing P to zero.

    Requires: grad_coeffs, stiffness_weights, positions, attention_mask.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.grad_coeffs is None:
            raise ValueError("StiffnessNormalAlignmentLoss requires ctx.grad_coeffs")
        if ctx.stiffness_weights is None:
            raise ValueError("StiffnessNormalAlignmentLoss requires ctx.stiffness_weights")
        if ctx.positions is None:
            raise ValueError("StiffnessNormalAlignmentLoss requires ctx.positions")

        mask_3d = ctx.attention_mask.unsqueeze(-1).float()  # (B, K, 1)
        grad_masked = ctx.grad_coeffs * mask_3d              # (B, K, 3)
        pos_masked = ctx.positions * mask_3d                  # (B, K, 3)

        # Predicted tangent plane projector: P[d, c] = Σ_j g_ij[d] * p_j[c]
        P_pred = torch.einsum('bkd,bkc->bdc', grad_masked, pos_masked)  # (B, 3, 3)

        # Stiffness action: Σ_j s_ij * p_j
        weights_masked = ctx.stiffness_weights * ctx.attention_mask.float()  # (B, K)
        stiffness_action = (weights_masked.unsqueeze(-1) * pos_masked).sum(dim=1)  # (B, 3)

        # Tangent component: P · stiffness_action — should be zero
        tangent_component = torch.einsum('bdc,bc->bd', P_pred, stiffness_action)  # (B, 3)

        per_patch = (tangent_component ** 2).sum(dim=-1)  # (B,)

        if self.reduction == 'mean':
            return per_patch.mean()
        elif self.reduction == 'sum':
            return per_patch.sum()
        elif self.reduction == 'none':
            return per_patch
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class ProjectorRegularizationLoss(nn.Module):
    """
    Self-supervised regularizer enforcing that the predicted gradient projector
    P = Σ_j g_ij ⊗ p_jᵀ is a valid rank-2 orthogonal projector (tangent plane).

    Combines three properties of the tangent plane projector P = I - n̂n̂ᵀ:

        1. Symmetry:    ‖P - Pᵀ‖²         (orthogonal, not oblique)
        2. Idempotency: ‖P² - P‖²         (projecting twice = projecting once)
        3. Trace:       (tr(P) - 2)²       (rank 2 = tangent plane, not collapse)

    No GT normal needed. Should be combined with StiffnessNormalAlignmentLoss
    to jointly enforce: (a) gradient operator recovers tangent plane, and
    (b) stiffness action points along the normal.

    Args:
        w_symmetry: Weight for symmetry term (default: 1.0)
        w_idempotency: Weight for idempotency term (default: 1.0)
        w_trace: Weight for trace term (default: 1.0)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, w_symmetry: float = 1.0, w_idempotency: float = 1.0,
                 w_trace: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.w_symmetry = w_symmetry
        self.w_idempotency = w_idempotency
        self.w_trace = w_trace
        self.reduction = reduction

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.grad_coeffs is None:
            raise ValueError("ProjectorRegularizationLoss requires ctx.grad_coeffs")
        if ctx.positions is None:
            raise ValueError("ProjectorRegularizationLoss requires ctx.positions")

        mask_3d = ctx.attention_mask.unsqueeze(-1).float()  # (B, K, 1)
        grad_masked = ctx.grad_coeffs * mask_3d              # (B, K, 3)
        pos_masked = ctx.positions * mask_3d                  # (B, K, 3)

        # Predicted projector: P[d, c] = Σ_j g_ij[d] * p_j[c]
        P = torch.einsum('bkd,bkc->bdc', grad_masked, pos_masked)  # (B, 3, 3)

        loss = torch.zeros(P.shape[0], device=P.device, dtype=P.dtype)  # (B,)

        # 1. Symmetry: ‖P - Pᵀ‖²_F
        if self.w_symmetry > 0:
            P_t = P.transpose(-1, -2)
            sym_err = ((P - P_t) ** 2).sum(dim=(-1, -2))  # (B,)
            loss = loss + self.w_symmetry * sym_err

        # 2. Idempotency: ‖P² - P‖²_F
        if self.w_idempotency > 0:
            P_sq = torch.bmm(P, P)
            idem_err = ((P_sq - P) ** 2).sum(dim=(-1, -2))  # (B,)
            loss = loss + self.w_idempotency * idem_err

        # 3. Trace: (tr(P) - 2)²
        if self.w_trace > 0:
            trace = P.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (B,)
            trace_err = (trace - 2.0) ** 2  # (B,)
            loss = loss + self.w_trace * trace_err

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


# =============================================
# Shared helpers for test-function losses
# =============================================

def _masked_grad_and_stiffness(ctx: LossContext):
    """Extract masked grad_coeffs, stiffness_weights, and positions.

    Returns:
        grad_masked: (B, K, 3)
        weights_masked: (B, K)
        pos_masked: (B, K, 3)
        mask_3d: (B, K, 1)
    """
    mask_3d = ctx.attention_mask.unsqueeze(-1).float()
    grad_masked = ctx.grad_coeffs * mask_3d
    weights_masked = ctx.stiffness_weights * ctx.attention_mask.float()
    pos_masked = ctx.positions * mask_3d
    return grad_masked, weights_masked, pos_masked, mask_3d


def _signed_log(x: torch.Tensor) -> torch.Tensor:
    """``sign(x) · log(|x| + 1)`` — smooth, monotonic, handles negatives."""
    return x.sign() * torch.log(x.abs() + 1.0)


def _scalar_error_per_element(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mode: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-element error metric on matched scalar tensors.

    Modes:
      ``'mse'``         (pred - target)²
      ``'relative'``    ((pred - target) / (|target| + eps))²
      ``'signed_log'``  (slog(pred) - slog(target))²
    """
    if mode == 'mse':
        return (predicted - target) ** 2
    if mode == 'relative':
        return ((predicted - target) / (target.abs() + eps)) ** 2
    if mode == 'signed_log':
        return (_signed_log(predicted) - _signed_log(target)) ** 2
    raise ValueError(
        f"loss_mode must be 'mse' / 'relative' / 'signed_log', got '{mode}'")


def _compute_discrete_gradient(grad_masked, deltas):
    """Compute discrete surface gradient ∇h = Σ_j g_ij δh_j.

    Args:
        grad_masked: (B, K, 3) masked gradient coefficients.
        deltas: (B, K, P) function differences per probe.

    Returns:
        (B, P, 3) discrete gradient vectors per probe.
    """
    # (B, K, 3) x (B, K, P) → (B, 3, P) → transpose → (B, P, 3)
    return torch.einsum('bkd,bkp->bpd', grad_masked, deltas)


# =============================================
# Generalized Laplacian test loss
# =============================================

class GeneralizedLaplacianTestLoss(nn.Module):
    """Supervise the discrete Laplacian on multiple test functions.

    For each test function h_p, compares the discrete Laplacian action:

        predicted_p = (Σ_j s_ij · δh_p_j) / A_i

    against the analytic ground-truth:

        target_p = Δ_LB(h_p)

    This generalises MCV supervision (which tests h = x, y, z only) to
    arbitrary smooth functions, reducing the null-space ambiguity in the
    learned stiffness weights.

    Requires test_func_deltas and test_func_laplacians in the LossContext.

    Args:
        loss_mode: Error metric to use:
            'mse': (predicted - target)²  — absolute MSE.
            'relative': ((predicted - target) / (|target| + eps))²  — relative error.
            'signed_log': (slog(predicted) - slog(target))²  where
                slog(x) = sign(x)·log(|x|+1).  Compresses large values,
                handles both signs, scale-insensitive.
        reduction: 'mean' | 'sum' | 'none'
        eps: Small constant for numerical stability (relative mode).
    """

    def __init__(self, loss_mode: str = 'mse', reduction: str = 'mean',
                 eps: float = 1e-8):
        super().__init__()
        if loss_mode not in ('mse', 'relative', 'signed_log'):
            raise ValueError(f"loss_mode must be 'mse', 'relative', or 'signed_log', got '{loss_mode}'")
        self.loss_mode = loss_mode
        self.reduction = reduction
        self.eps = eps

    @staticmethod
    def _signed_log(x: torch.Tensor) -> torch.Tensor:
        """Backwards-compatible alias for the module-level helper."""
        return _signed_log(x)

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.test_func_deltas is None:
            raise ValueError("GeneralizedLaplacianTestLoss requires ctx.test_func_deltas")
        if ctx.test_func_laplacians is None:
            raise ValueError("GeneralizedLaplacianTestLoss requires ctx.test_func_laplacians")

        # s_ij: (B, K),  deltas: (B, K, P),  areas: (B,)
        weights = ctx.stiffness_weights * ctx.attention_mask.float()  # (B, K)
        deltas = ctx.test_func_deltas  # (B, K, P)

        # Discrete Laplacian: (Σ_j s_ij · δh_j) / A_i  →  (B, P)
        numerator = (weights.unsqueeze(-1) * deltas).sum(dim=1)  # (B, P)
        predicted = numerator / ctx.areas.unsqueeze(-1)           # (B, P)

        target = ctx.test_func_laplacians  # (B, P)

        per_func_error = _scalar_error_per_element(
            predicted, target, self.loss_mode, eps=self.eps)
        per_patch = per_func_error.mean(dim=-1)  # (B,)

        if self.reduction == 'mean':
            return per_patch.mean()
        elif self.reduction == 'sum':
            return per_patch.sum()
        elif self.reduction == 'none':
            return per_patch
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


# =============================================
# Gradient tangent-plane losses
# =============================================

class GradientTangentPlaneLoss(nn.Module):
    """Supervised loss: discrete gradient of test functions should be tangential.

    Checks that (n̂ · ∇h_p)² = 0 for every test function h_p, using the
    ground-truth surface normal n̂.

    This generalises TangentPlaneProjectorLoss (which only tests coordinate
    functions) to arbitrary test functions — a stronger constraint on the
    gradient operator.

    Requires test_func_deltas and normals in the LossContext.

    Args:
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.test_func_deltas is None:
            raise ValueError("GradientTangentPlaneLoss requires ctx.test_func_deltas")
        if ctx.normals is None:
            raise ValueError("GradientTangentPlaneLoss requires ctx.normals")

        grad_masked, _, _, mask_3d = _masked_grad_and_stiffness(ctx)
        deltas = ctx.test_func_deltas * mask_3d  # (B, K, P) masked

        # Discrete gradient: ∇h_p = Σ_j g_ij δh_p_j  →  (B, P, 3)
        grad_h = _compute_discrete_gradient(grad_masked, deltas)

        # Normal component: (n̂ · ∇h_p)²  →  (B, P)
        normals = F.normalize(ctx.normals, p=2, dim=1)            # (B, 3)
        normal_component = torch.einsum('bd,bpd->bp', normals, grad_h)  # (B, P)

        per_patch = (normal_component ** 2).mean(dim=-1)  # (B,)

        if self.reduction == 'mean':
            return per_patch.mean()
        elif self.reduction == 'sum':
            return per_patch.sum()
        elif self.reduction == 'none':
            return per_patch
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class GeneralizedGradientSupervisionLoss(nn.Module):
    """Supervise the discrete gradient action on multiple test functions.

    For each test function h_p, compares the discrete gradient action:

        (∇_S h_p)_pred = Σ_j g_ij · δh_p_j     (B, P, 3)

    against the analytic ground-truth surface gradient:

        (∇_S h_p)_GT                              (B, P, 3)

    This is a much stronger constraint than GradientTangentPlaneLoss (which
    only checks zero normal component).  It supervises **direction and
    magnitude** of the gradient across P test functions — P vector equations
    constraining the full 3×K gradient coefficient matrix.

    Requires test_func_deltas and test_func_gradients in the LossContext.

    Args:
        loss_mode: Error metric:
            'mse': ‖pred - target‖²
            'relative': ‖pred - target‖² / (‖target‖² + eps)
            'cosine': 1 - cos(pred, target)  (direction only)
            'log_mse': ‖slog(pred) - slog(target)‖²  where
                slog(v) = v̂ · log(‖v‖ + 1).  Compresses magnitude range
                (log scale) while preserving direction.  Not scale-invariant.
        reduction: 'mean' | 'sum' | 'none'
        eps: Small constant for numerical stability.
    """

    def __init__(self, loss_mode: str = 'mse', reduction: str = 'mean',
                 eps: float = 1e-8):
        super().__init__()
        if loss_mode not in ('mse', 'relative', 'cosine', 'log_mse'):
            raise ValueError(f"loss_mode must be 'mse', 'relative', 'cosine', "
                             f"or 'log_mse', got '{loss_mode}'")
        self.loss_mode = loss_mode
        self.reduction = reduction
        self.eps = eps

    @staticmethod
    def _slog_vec(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Map vector to log-magnitude space: v̂ · log(‖v‖ + 1).

        Preserves direction, compresses magnitude.  slog(0) = 0.
        """
        norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
        return v * (torch.log(norm + 1.0) / norm)

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.test_func_deltas is None:
            raise ValueError("GeneralizedGradientSupervisionLoss requires ctx.test_func_deltas")
        if ctx.test_func_gradients is None:
            raise ValueError("GeneralizedGradientSupervisionLoss requires ctx.test_func_gradients")

        grad_masked, _, _, mask_3d = _masked_grad_and_stiffness(ctx)
        deltas = ctx.test_func_deltas * mask_3d  # (B, K, P) masked

        # Predicted gradient: Σ_j g_ij δh_p_j  →  (B, P, 3)
        predicted = _compute_discrete_gradient(grad_masked, deltas)

        # GT gradient  →  (B, P, 3)
        target = ctx.test_func_gradients

        if self.loss_mode == 'mse':
            # ‖pred - target‖² per function  →  (B, P)
            per_func = ((predicted - target) ** 2).sum(dim=-1)
        elif self.loss_mode == 'relative':
            # ‖pred - target‖² / (‖target‖² + eps)  →  (B, P)
            diff_sq = ((predicted - target) ** 2).sum(dim=-1)
            target_sq = (target ** 2).sum(dim=-1) + self.eps
            per_func = diff_sq / target_sq
        elif self.loss_mode == 'cosine':
            # 1 - cos(pred, target)  →  (B, P)
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)  # (B, P)
            per_func = 1.0 - cos_sim
        elif self.loss_mode == 'log_mse':
            # ‖slog(pred) - slog(target)‖²  →  (B, P)
            pred_log = self._slog_vec(predicted, self.eps)
            target_log = self._slog_vec(target, self.eps)
            per_func = ((pred_log - target_log) ** 2).sum(dim=-1)

        per_patch = per_func.mean(dim=-1)  # (B,)

        if self.reduction == 'mean':
            return per_patch.mean()
        elif self.reduction == 'sum':
            return per_patch.sum()
        elif self.reduction == 'none':
            return per_patch
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class SelfSupervisedGradientTangentPlaneLoss(nn.Module):
    """Unsupervised loss: discrete gradient of test functions should be tangential.

    Uses the predicted tangent-plane projector P = Σ_j g_ij ⊗ p_jᵀ to
    extract the normal direction as (I - P), then checks:

        ‖(I - P) · ∇h_p‖² = 0

    No ground-truth normal needed.  Should be combined with
    ProjectorRegularizationLoss to prevent P from collapsing.

    Requires test_func_deltas, grad_coeffs, positions, attention_mask.

    Args:
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.test_func_deltas is None:
            raise ValueError("SelfSupervisedGradientTangentPlaneLoss requires ctx.test_func_deltas")

        grad_masked, _, pos_masked, mask_3d = _masked_grad_and_stiffness(ctx)
        deltas = ctx.test_func_deltas * mask_3d  # (B, K, P) masked

        # Predicted projector: P = Σ_j g_ij ⊗ p_jᵀ  →  (B, 3, 3)
        P = torch.einsum('bkd,bkc->bdc', grad_masked, pos_masked)

        # Normal projector: I - P  →  (B, 3, 3)
        I = torch.eye(3, device=P.device, dtype=P.dtype).unsqueeze(0)
        N_proj = I - P

        # Discrete gradient: ∇h_p = Σ_j g_ij δh_p_j  →  (B, P, 3)
        grad_h = _compute_discrete_gradient(grad_masked, deltas)

        # Normal component: (I - P) · ∇h_p  →  (B, P, 3)
        normal_component = torch.einsum('bdc,bpc->bpd', N_proj, grad_h)

        per_patch = (normal_component ** 2).sum(dim=-1).mean(dim=-1)  # (B,)

        if self.reduction == 'mean':
            return per_patch.mean()
        elif self.reduction == 'sum':
            return per_patch.sum()
        elif self.reduction == 'none':
            return per_patch
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

# =============================================
# Variational / Dirichlet-energy losses
# (surface-level: read per-vertex fields from LossContext)
# =============================================

class _VariationalTestLoss(nn.Module):
    """Private base for surface-level variational losses.

    Subclasses use the discrete Dirichlet-form helpers in
    ``neural_local_laplacian.utils.dirichlet`` to compute predicted
    quantities from ``ctx.grad_coeffs``, ``ctx.areas``, ``ctx.knn``, and
    ``ctx.test_func_values`` (all populated by
    ``MongeSurfaceVariationalDataset``), and compare them against the
    analytic surface-level GT carried on the same ``LossContext``.

    Args:
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                f"reduction must be 'mean' / 'sum' / 'none', got '{reduction}'")
        self.reduction = reduction

    @staticmethod
    def _check_required(ctx: LossContext, names) -> None:
        for name in names:
            if getattr(ctx, name, None) is None:
                raise ValueError(
                    f"Variational loss is missing required LossContext field "
                    f"'{name}'. The context must come from a "
                    f"_VariationalSurfaceData item with the matching test-function "
                    f"sampler flags enabled.")

    @staticmethod
    def _compute_pred_grads(ctx: LossContext) -> torch.Tensor:
        """Compute per-vertex per-probe predicted gradients ``(n, P, 3)``."""
        from neural_local_laplacian.utils.dirichlet import predicted_gradient_per_vertex
        return predicted_gradient_per_vertex(
            ctx.grad_coeffs, ctx.knn, ctx.test_func_values)

    def _reduce(self, per_unit: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return per_unit.mean()
        if self.reduction == 'sum':
            return per_unit.sum()
        return per_unit


class DirichletEnergyTestLoss(_VariationalTestLoss):
    r"""Match discrete and continuous Dirichlet energies per probe.

    Predicted: ``Ê(f_ℓ) = ½ Σ_i A_i ‖ĝ_i^{(ℓ)}‖²`` where
    ``ĝ_i^{(ℓ)} = Σ_j g_ij · (f_ℓ(v_j) − f_ℓ(v_i))``.
    Target:    ``E(f_ℓ) = ½ ∫_U ‖∇_g f̃_ℓ‖² √det g  du dv``  (Gauss-Legendre).

    Reads from ``ctx``: ``grad_coeffs``, ``areas``, ``knn``,
    ``test_func_values`` (n, P), and ``test_func_continuous_energy`` (P,).
    """

    def __init__(self, loss_mode: str = 'mse', reduction: str = 'mean',
                 eps: float = 1e-8):
        super().__init__(reduction=reduction)
        if loss_mode not in ('mse', 'relative', 'signed_log'):
            raise ValueError(
                f"loss_mode must be 'mse' / 'relative' / 'signed_log', got '{loss_mode}'")
        self.loss_mode = loss_mode
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        from neural_local_laplacian.utils.dirichlet import discrete_dirichlet_energy
        self._check_required(ctx, ['grad_coeffs', 'areas', 'knn',
                                    'test_func_values',
                                    'test_func_continuous_energy'])
        pred_grads = self._compute_pred_grads(ctx)                 # (n, P, 3)
        pred_energy = discrete_dirichlet_energy(pred_grads, ctx.areas)   # (P,)
        per_probe = _scalar_error_per_element(
            pred_energy, ctx.test_func_continuous_energy,
            self.loss_mode, eps=self.eps)
        return self._reduce(per_probe)


class BilinearFormTestLoss(_VariationalTestLoss):
    r"""Match the discrete and continuous Dirichlet bilinear forms.

    Predicted: ``Ê(f_ℓ, f_p) = Σ_i A_i ⟨ĝ_i^{(ℓ)}, ĝ_i^{(p)}⟩``  ((P, P) symmetric).
    Target:    ``E(f_ℓ, f_p) = ∫_U ⟨∇_g f̃_ℓ, ∇_g f̃_p⟩_g √det g  du dv``.

    Strictly stronger than :class:`DirichletEnergyTestLoss` (its diagonal):
    bilinear pins down per-vertex gradient inner products, leaving only
    per-vertex frame rotations as a gauge group, while energy alone is
    invariant under per-function per-vertex rotations.

    Compares only the upper triangle (including diagonal) — both matrices
    are symmetric, so off-diagonals are not double-counted.

    Reads from ``ctx``: ``grad_coeffs``, ``areas``, ``knn``,
    ``test_func_values`` (n, P), and ``test_func_continuous_bilinear`` (P, P).
    """

    def __init__(self, loss_mode: str = 'mse', reduction: str = 'mean',
                 eps: float = 1e-8):
        super().__init__(reduction=reduction)
        if loss_mode not in ('mse', 'relative', 'signed_log'):
            raise ValueError(
                f"loss_mode must be 'mse' / 'relative' / 'signed_log', got '{loss_mode}'")
        self.loss_mode = loss_mode
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        from neural_local_laplacian.utils.dirichlet import discrete_dirichlet_bilinear
        self._check_required(ctx, ['grad_coeffs', 'areas', 'knn',
                                    'test_func_values',
                                    'test_func_continuous_bilinear'])
        pred_grads = self._compute_pred_grads(ctx)                  # (n, P, 3)
        pred_B = discrete_dirichlet_bilinear(pred_grads, ctx.areas)  # (P, P)
        target_B = ctx.test_func_continuous_bilinear                # (P, P)

        P = pred_B.shape[0]
        iu = torch.triu_indices(P, P, offset=0, device=pred_B.device)
        pred_flat = pred_B[iu[0], iu[1]]
        target_flat = target_B[iu[0], iu[1]]
        per_pair = _scalar_error_per_element(
            pred_flat, target_flat, self.loss_mode, eps=self.eps)
        return self._reduce(per_pair)


class GradientTangencyTestLoss(_VariationalTestLoss):
    r"""Penalise non-tangential components of predicted surface gradients.

    Loss:  ``L = Σ_ℓ Σ_i A_i · ⟨ĝ_i^{(ℓ)}, n̂_i⟩²``  (then reduced).

    Required gauge fix when training with energy or bilinear-form
    supervision alone: both are invariant under per-vertex rotations of
    the predicted gradients, so without a tangency penalty the model can
    learn geometrically wrong frames that nonetheless match the form.

    Reads from ``ctx``: ``grad_coeffs``, ``areas``, ``knn``,
    ``test_func_values``, and ``vertex_normals`` (n, 3).
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction=reduction)

    def forward(self, ctx: LossContext) -> torch.Tensor:
        self._check_required(ctx, ['grad_coeffs', 'areas', 'knn',
                                    'test_func_values', 'vertex_normals'])
        pred_grads = self._compute_pred_grads(ctx)                  # (n, P, 3)
        normals = F.normalize(ctx.vertex_normals, p=2, dim=-1)      # (n, 3)
        normal_component = torch.einsum('npd,nd->np', pred_grads, normals)  # (n, P)
        weighted = ctx.areas.unsqueeze(-1) * (normal_component ** 2)         # (n, P)
        return self._reduce(weighted)


class GradientVectorTestLoss(_VariationalTestLoss):
    r"""Pointwise vector match: ``ĝ_i^{(ℓ)} ≈ ∇_g f_ℓ(v_i)``.

    The strongest signal in the §21 hierarchy of the LBO conversation:
    pins the predicted per-vertex per-probe surface gradients to their
    analytic targets in **direction, magnitude, and tangency
    simultaneously**.  Removes every gauge freedom that energy and
    bilinear-form supervision leave open — including the per-vertex
    frame rotation that bilinear can't see.

    Reads from ``ctx``:
        grad_coeffs                        (n, k, 3)
        knn                                 (n, k)
        test_func_values                    (n, P)
        test_func_gradients_at_vertices     (n, P, 3)   — analytic GT, on Σ
        areas                               (n,)        — only when ``area_weighted=True``

    The dataset must be configured with
    ``test_func_sampler.compute_gradients_all_points: true`` so the GT
    field is populated.

    Args:
        loss_mode: vector-error metric.
            ``'mse'``        — ``‖pred - target‖²``
            ``'relative'``   — ``‖pred - target‖² / (‖target‖² + ε)``
            ``'cosine'``     — ``1 - cos(pred, target)``  (direction only)
            ``'log_mse'``    — ``‖slog(pred) - slog(target)‖²``  with
                ``slog(v) = v̂ · log(‖v‖+1)``;  log-compresses magnitude
                while preserving direction.
        reduction: ``'mean'`` / ``'sum'`` / ``'none'`` over (vertex × probe).
        eps: numerical floor for relative / cosine / log_mse.
        area_weighted: if True, weight each (i, ℓ) entry by ``A_i`` before
            reducing; matches the variational pipeline's integration-style
            convention but is cosmetic for the gradient match.
    """

    def __init__(self, loss_mode: str = 'mse', reduction: str = 'mean',
                 eps: float = 1e-8, area_weighted: bool = False):
        super().__init__(reduction=reduction)
        if loss_mode not in ('mse', 'relative', 'cosine', 'log_mse'):
            raise ValueError(
                f"loss_mode must be 'mse' / 'relative' / 'cosine' / 'log_mse', "
                f"got '{loss_mode}'")
        self.loss_mode = loss_mode
        self.eps = eps
        self.area_weighted = area_weighted

    @staticmethod
    def _slog_vec(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Map ``v`` to log-magnitude space: ``v̂ · log(‖v‖ + 1)``.

        Preserves direction, compresses magnitude.  ``slog(0) = 0``.
        """
        norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
        return v * (torch.log(norm + 1.0) / norm)

    def forward(self, ctx: LossContext) -> torch.Tensor:
        required = ['grad_coeffs', 'knn', 'test_func_values',
                    'test_func_gradients_at_vertices']
        if self.area_weighted:
            required.append('areas')
        self._check_required(ctx, required)

        pred_grads = self._compute_pred_grads(ctx)            # (n, P, 3)
        target = ctx.test_func_gradients_at_vertices          # (n, P, 3)

        if self.loss_mode == 'mse':
            per = ((pred_grads - target) ** 2).sum(dim=-1)    # (n, P)
        elif self.loss_mode == 'relative':
            diff_sq = ((pred_grads - target) ** 2).sum(dim=-1)
            target_sq = (target ** 2).sum(dim=-1) + self.eps
            per = diff_sq / target_sq
        elif self.loss_mode == 'cosine':
            cos_sim = F.cosine_similarity(pred_grads, target, dim=-1)
            per = 1.0 - cos_sim
        else:  # 'log_mse'
            pred_log = self._slog_vec(pred_grads, self.eps)
            target_log = self._slog_vec(target, self.eps)
            per = ((pred_log - target_log) ** 2).sum(dim=-1)

        if self.area_weighted:
            per = ctx.areas.unsqueeze(-1) * per

        return self._reduce(per)


class LaplacianActionTestLoss(_VariationalTestLoss):
    r"""Per-vertex match of the discrete Laplacian action on test functions.

    The variational analog of :class:`GeneralizedLaplacianTestLoss`
    (which is per-patch, at the origin only).  For each vertex i and
    probe ℓ, compare the discrete Laplacian action against the analytic
    target evaluated at v_i:

        predicted_i^{(ℓ)}  =  (Σ_j  s_ij · (h_ℓ(v_j) − h_ℓ(v_i)))  /  A_i
        target_i^{(ℓ)}     =  Δ_g h_ℓ (v_i)

    Routes gradient through the **stiffness head** ``s_ij`` and the
    **area head** ``A_i``, so this is the primary signal you'd use to
    train ``stiffness_mode='learned'`` / ``'learned_positive'`` in the
    variational pipeline.  None of the other variational losses
    currently consume ``stiffness_weights`` — without one of these the
    learned stiffness head receives no gradient at all.

    Reads from ``ctx``:
        stiffness_weights                  (n, k)
        areas                              (n,)
        knn                                (n, k)
        test_func_values                   (n, P)
        test_func_laplacians_at_vertices   (n, P)        — analytic GT

    The dataset must be configured with
    ``test_func_sampler.compute_lb_all_points: true`` so the analytic
    per-vertex Laplacian targets are populated.

    Args:
        loss_mode: per-element scalar metric — ``'mse'`` / ``'relative'``
            / ``'signed_log'`` — *or* ``'cosine'``, which behaves
            differently from the scalar modes:

              * ``'mse'`` / ``'relative'`` / ``'signed_log'``: per-element
                error on the ``(n, P)`` tensors, reduced over both axes.
              * ``'cosine'``: per-vertex cosine similarity *across the
                probe axis*.  Each vertex ``i`` contributes
                ``1 - cos(pred[i, :], target[i, :])``, a scalar; the
                resulting ``(n,)`` tensor is then optionally area-weighted
                and reduced.  Scale-invariant under per-vertex scalar
                rescaling — i.e. the model can rescale ``s_ij / A_i`` at
                each vertex independently and still get cosine = 1.
                Measures whether the *pattern* of LB values across probes
                at each vertex matches GT, ignoring local magnitude.
        reduction: ``'mean'`` / ``'sum'`` / ``'none'`` over (vertex × probe)
            for the scalar modes, or over vertices for cosine.
        eps: numerical floor for relative / cosine.
        area_weighted: if True, weight each entry by ``A_i`` before
            reducing.  Off by default — the per-vertex match is already
            normalised by ``A_i`` inside the predicted Laplacian.  In
            ``cosine`` mode this multiplies the per-vertex ``(n,)``
            tensor by ``ctx.areas``.
    """

    def __init__(self, loss_mode: str = 'mse', reduction: str = 'mean',
                 eps: float = 1e-8, area_weighted: bool = False):
        super().__init__(reduction=reduction)
        if loss_mode not in ('mse', 'relative', 'signed_log', 'cosine'):
            raise ValueError(
                f"loss_mode must be 'mse' / 'relative' / 'signed_log' / "
                f"'cosine', got '{loss_mode}'")
        self.loss_mode = loss_mode
        self.eps = eps
        self.area_weighted = area_weighted

    def forward(self, ctx: LossContext) -> torch.Tensor:
        self._check_required(ctx, [
            'stiffness_weights', 'areas', 'knn',
            'test_func_values', 'test_func_laplacians_at_vertices'])

        # Per-vertex per-probe deltas via knn gather.
        # values shape (n, P); knn shape (n, k);
        # values[knn] → (n, k, P).
        neighbour_values = ctx.test_func_values[ctx.knn]            # (n, k, P)
        center_values    = ctx.test_func_values.unsqueeze(1)        # (n, 1, P)
        deltas           = neighbour_values - center_values          # (n, k, P)

        # Discrete Laplacian-Beltrami action:
        #   Δ̂h_i^{(ℓ)}  =  (Σ_j  s_ij · δh_ji^{(ℓ)})  /  A_i
        s = ctx.stiffness_weights                                   # (n, k)
        numerator = (s.unsqueeze(-1) * deltas).sum(dim=1)           # (n, P)
        predicted = numerator / ctx.areas.unsqueeze(-1)             # (n, P)

        target = ctx.test_func_laplacians_at_vertices               # (n, P)

        if self.loss_mode == 'cosine':
            # Per-vertex cosine similarity across the P probes.
            cos_sim = F.cosine_similarity(
                predicted, target, dim=-1, eps=self.eps)            # (n,)
            per = 1.0 - cos_sim                                      # (n,)
            if self.area_weighted:
                per = ctx.areas * per                                # (n,)
            return self._reduce(per)

        per = _scalar_error_per_element(
            predicted, target, self.loss_mode, eps=self.eps)        # (n, P)

        if self.area_weighted:
            per = ctx.areas.unsqueeze(-1) * per

        return self._reduce(per)


class MeanCurvatureDirectionCosineTestLoss(_VariationalTestLoss):
    r"""Cosine-similarity direction loss on the discrete mean curvature vector.

    Variational analog of :class:`DirectionCosineLoss`.  Computes the
    per-vertex MCV from the **learned stiffness head** acting on vertex
    positions and compares its direction to the analytic surface normal:

        MCV_i  =  (Σ_j s_ij · (p_{knn[i,j]} - p_i)) / A_i
        L_i    =  1 − |⟨MCV_i / ‖MCV_i‖,  n̂_i⟩|       (signed=False, default)
        L_i    =  1 −  ⟨MCV_i / ‖MCV_i‖,  n̂_i⟩         (signed=True)

    The sign-agnostic form is the default since :class:`MongeSurfaceVariationalDataset`
    only carries the (unsigned) surface normal, not the signed mean
    curvature.  Use ``signed=True`` only when you know the dataset's
    normal convention matches the MCV sign convention you expect.

    Reads from ``ctx``:
        stiffness_weights  (n, k)
        areas              (n,)
        knn                (n, k)
        vertex_pos         (n, 3)
        vertex_normals     (n, 3)

    Args:
        signed: if True, use ``1 - cos`` (assumes consistent sign convention);
            otherwise use ``1 - |cos|`` (direction-only, sign-agnostic).
        reduction: ``'mean'`` / ``'sum'`` / ``'none'`` over vertices.
        eps: numerical floor for normalisation.
        area_weighted: if True, weight each vertex by ``A_i`` before reducing.
    """

    def __init__(self, signed: bool = False, reduction: str = 'mean',
                 eps: float = 1e-8, area_weighted: bool = False):
        super().__init__(reduction=reduction)
        self.signed = bool(signed)
        self.eps = float(eps)
        self.area_weighted = bool(area_weighted)

    def forward(self, ctx: LossContext) -> torch.Tensor:
        self._check_required(ctx, [
            'stiffness_weights', 'areas', 'knn',
            'vertex_pos', 'vertex_normals'])

        p = ctx.vertex_pos                                  # (n, 3)
        neighbour_pos = p[ctx.knn]                          # (n, k, 3)
        deltas = neighbour_pos - p.unsqueeze(1)             # (n, k, 3)

        s = ctx.stiffness_weights                           # (n, k)
        numerator = (s.unsqueeze(-1) * deltas).sum(dim=1)   # (n, 3)
        mcv = numerator / ctx.areas.unsqueeze(-1)           # (n, 3)

        mcv_unit = F.normalize(mcv, p=2, dim=-1, eps=self.eps)
        normal_unit = F.normalize(ctx.vertex_normals, p=2, dim=-1, eps=self.eps)
        cos = (mcv_unit * normal_unit).sum(dim=-1)          # (n,)
        if not self.signed:
            cos = cos.abs()
        per_vertex = 1.0 - cos                              # (n,)

        if self.area_weighted:
            per_vertex = ctx.areas * per_vertex

        return self._reduce(per_vertex)


class HeatMethodGeodesicLoss(_VariationalTestLoss):
    r"""Match heat-method geodesic distances against analytic GT.

    Implements the discrete Crane-Weischedel-Wardetzky heat method on the
    operators emitted by the model and compares the recovered distance
    field to the GT geodesic distances attached by
    :class:`MongeSurfaceVariationalDataset` (computed via exact MMP on a
    fine reference triangulation of the same Monge patch).

    Pipeline per source ``s``:

      1.  Heat-diffuse: ``(M + t·L) u = e_s``.
      2.  Vertex-wise gradient: ``∇u_i = Σ_j g_ij · (u_j − u_i)``  ∈ ℝ³.
      3.  Normalize: ``X_i = −∇u_i / ‖∇u_i‖``.
      4.  Recover distance: ``(L + ε·M) φ = Gᵀ M₃ X_flat``.
      5.  Gauge fix ``φ ← φ − φ(s)``.

    Routes gradient through **all three heads**:
        - ``areas``           — into M (mass) and into L (when ``L = α·s_ij``).
        - ``stiffness_weights`` — into L.
        - ``grad_coeffs``     — into G (gradient + divergence).

    Because steps 1 and 4 both solve linear systems involving L, and step
    2 reads G while step 4 reads Gᵀ, this loss implicitly couples the two
    heads: it is only consistent when ``L ≈ Gᵀ M_3 G`` (the discrete
    Stokes identity).  Useful side-effect when training with
    ``stiffness_mode='learned[_positive]'``, where the two heads can
    otherwise drift apart silently.

    Args:
        t: heat diffusion time (positive scalar).  Crane et al. recommend
            ``t ≈ mean_edge_length²``.  Default uses a normalised area
            scaling that adapts to the patch size: ``t = t_scale · A_total / n``.
        t_scale: multiplier when ``t`` is set to ``None`` (auto mode).
        epsilon_reg: small regularizer added as ``ε·M`` to the Poisson
            stiffness to handle the constant nullspace.  Default 1e-6.
        loss_mode: scalar error metric — ``'mse'`` / ``'relative'`` / ``'signed_log'``.
        normalize_by_diameter: divide the per-vertex error by the source's
            GT maximum (the patch diameter from that source) before
            reducing.  Makes the loss scale-invariant across patches and
            sources — useful when patch sizes vary.
        reduction: ``'mean'`` / ``'sum'`` / ``'none'`` over (source × vertex).
        eps: numerical floor for relative mode.
        L_assembly: which assembly to use for L.  ``'from_stiffness'``
            (default) routes through the stiffness head — pair with
            ``stiffness_mode='learned[_positive]'``.  ``'diagonal_gram'``
            uses ‖g_ij‖² — useful when stiffness is gram-derived.
    """

    def __init__(self,
                 t: Optional[float] = None,
                 t_scale: float = 1.0,
                 epsilon_reg: float = 1e-6,
                 loss_mode: str = 'mse',
                 normalize_by_diameter: bool = True,
                 reduction: str = 'mean',
                 eps: float = 1e-8,
                 L_assembly: str = 'from_stiffness'):
        super().__init__(reduction=reduction)
        if loss_mode not in ('mse', 'relative', 'signed_log'):
            raise ValueError(
                f"loss_mode must be 'mse' / 'relative' / 'signed_log', got '{loss_mode}'")
        if L_assembly not in ('from_stiffness', 'diagonal_gram'):
            raise ValueError(
                f"L_assembly must be 'from_stiffness' / 'diagonal_gram', got '{L_assembly}'")
        self.t = t
        self.t_scale = float(t_scale)
        self.epsilon_reg = float(epsilon_reg)
        self.loss_mode = loss_mode
        self.normalize_by_diameter = normalize_by_diameter
        self.eps = eps
        self.L_assembly = L_assembly

    def _assemble_L(self, ctx: LossContext) -> torch.Tensor:
        from neural_local_laplacian.utils.laplacian_assembly import (
            assemble_from_stiffness_weights,
            assemble_diagonal_gram_laplacian,
        )
        if self.L_assembly == 'from_stiffness':
            return assemble_from_stiffness_weights(
                ctx.stiffness_weights, ctx.knn, areas=None)
        return assemble_diagonal_gram_laplacian(
            ctx.grad_coeffs, ctx.knn, areas=None)

    @staticmethod
    def _per_vertex_gradient(
        grad_coeffs: torch.Tensor,
        knn: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``(∇u)_i = Σ_j g_ij · (u_j − u_i)`` → (n, 3)."""
        # u: (n,); knn: (n, k); grad_coeffs: (n, k, 3)
        u_neigh = u[knn]                              # (n, k)
        u_center = u.unsqueeze(1)                     # (n, 1)
        delta = (u_neigh - u_center).unsqueeze(-1)    # (n, k, 1)
        return (grad_coeffs * delta).sum(dim=1)       # (n, 3)

    @staticmethod
    def _per_vertex_divergence(
        grad_coeffs: torch.Tensor,
        knn: torch.Tensor,
        X: torch.Tensor,
        areas: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the FEM-form RHS ``b = Gᵀ M_3 X_flat`` → (n,).

        Using the gradient operator ``(G u)_i = Σ_j g_ij (u_j − u_i)``,
        its transpose acts on vector fields ``X : ℝ^{3n}`` as
        ``(Gᵀ Y)_c = Σ_i  G[3i+d, c] · Y_{i,d}`` (summed over d).  Because
        ``G[3i+d, c] = g_ij[d]`` when c is the j-th neighbour of i, and
        ``G[3i+d, i] = −Σ_j g_ij[d]``, we get
            (Gᵀ Y)_c =  Σ_{i: c ∈ knn(i)}  g_{ic}[d] · Y_{i,d}
                       − Σ_d (Σ_j g_cj[d]) · Y_{c,d}
        and the FEM RHS is ``b_c = A_c · (Gᵀ Y)_c`` after folding the mass
        matrix ``M_3 = diag(repeat(A, 3))``.  We use scatter-add for the
        first term.
        """
        n, k, _ = grad_coeffs.shape
        # Center contribution: −(Σ_j g_cj) · A_c X_c
        g_sum = grad_coeffs.sum(dim=1)                                # (n, 3)
        center_term = -(g_sum * X).sum(dim=-1) * areas                # (n,)
        # Neighbour contribution: for each (i, j_local) the c = knn[i, j_local]
        # accumulates g_ij · A_i · X_i.
        per_edge = (grad_coeffs * (areas.unsqueeze(-1) * X).unsqueeze(1)
                    ).sum(dim=-1)                                      # (n, k)
        neighbour_term = grad_coeffs.new_zeros(n).index_add(
            0, knn.reshape(-1), per_edge.reshape(-1))                   # (n,)
        return neighbour_term + center_term

    def forward(self, ctx: LossContext) -> torch.Tensor:
        self._check_required(ctx, [
            'grad_coeffs', 'stiffness_weights', 'areas', 'knn',
            'geodesic_sources', 'geodesic_distances'])

        L = self._assemble_L(ctx)                                     # (n, n)
        M_diag = ctx.areas                                            # (n,)
        n = M_diag.shape[0]
        device = L.device
        dtype = L.dtype
        eye = torch.eye(n, device=device, dtype=dtype)
        M_full = M_diag.unsqueeze(-1) * eye                           # (n, n)

        if self.t is None:
            t = self.t_scale * float(M_diag.sum().item()) / n
        else:
            t = float(self.t)

        A_heat = M_full + t * L                                       # (n, n)
        A_poisson = L + self.epsilon_reg * M_full                     # (n, n)

        sources = ctx.geodesic_sources                                # (S,)
        gt = ctx.geodesic_distances                                   # (S, n)
        S = sources.shape[0]

        # Heat-method per source.
        per_source_errors = []
        for s_idx in range(S):
            s = int(sources[s_idx].item())
            b_heat = L.new_zeros(n)
            b_heat[s] = 1.0
            u = torch.linalg.solve(A_heat, b_heat)                    # (n,)
            grad_u = self._per_vertex_gradient(ctx.grad_coeffs, ctx.knn, u)  # (n, 3)
            X = -grad_u / (grad_u.norm(dim=-1, keepdim=True) + self.eps)     # (n, 3)
            rhs = self._per_vertex_divergence(
                ctx.grad_coeffs, ctx.knn, X, M_diag)                  # (n,)
            phi = torch.linalg.solve(A_poisson, rhs)                  # (n,)
            phi = phi - phi[s]                                        # gauge: φ(s) = 0
            per_vertex = _scalar_error_per_element(
                phi, gt[s_idx].to(device=device, dtype=dtype),
                self.loss_mode, eps=self.eps)                         # (n,)
            if self.normalize_by_diameter:
                # Source-specific diameter from the GT (≥ eps); makes
                # the loss invariant to patch scale.
                diam_sq = (gt[s_idx].to(device=device, dtype=dtype) ** 2).max() + self.eps
                per_vertex = per_vertex / diam_sq
            per_source_errors.append(per_vertex)

        per = torch.stack(per_source_errors, dim=0)                   # (S, n)
        return self._reduce(per)


class HeatKernelLoss(_VariationalTestLoss):
    r"""Match the discrete heat kernel against the analytic GT.

    The discrete heat kernel matrix at diffusion time ``t`` is

        K_t  =  exp(-t · M⁻¹ L)              ∈ ℝ^{n×n}

    Entrywise this approximates ``K_t(x_i, x_j)`` of the continuous heat
    semigroup ``exp(-tΔ_g)``.  Spectrally,

        K_t(x_i, x_j)  =  Σ_k  exp(-λ_k t) · φ_k(x_i) · φ_k(x_j)

    where ``(λ_k, φ_k)`` are M-orthonormal generalized eigenpairs of
    ``Lφ = λMφ`` (PSD convention).  The GT side computes this spectral
    form on a dense reference triangulation of the same Monge patch (see
    :func:`compute_heat_kernel_gt_on_monge_patch`).

    Routes gradient through:
      • the stiffness head (via ``L = assemble_from_stiffness_weights``)
      • the area head     (via ``M = diag(areas)``)

    Does NOT touch the gradient head — pair with a small
    :class:`GradientVectorTestLoss` if you want that head trained too.

    Args:
        diagonal_only: if True, supervise only the diagonal ``K_t(x_i, x_i)``
            (= HKS at each vertex), which is what fmap descriptors read.
            If False, supervise the full ``(n, n)`` kernel — strictly more
            information, ``n²`` memory per t.  Must match the
            ``heat_kernel_diagonal_only`` flag the dataset was built with.
        loss_mode: ``'mse'`` / ``'relative'`` / ``'signed_log'``.
        normalize_per_t: divide each t-channel's error by ``‖K_t_gt‖_F``
            (or ``‖K_t_gt‖_2`` in diagonal mode) before reducing.  Makes
            the loss scale-invariant per spectral band — useful because
            small-t kernels can have much larger entries than large-t.
        L_assembly: ``'from_stiffness'`` (default — pair with
            ``stiffness_mode='learned[_positive]'``) or ``'diagonal_gram'``.
        eps: numerical floor.
        reduction: ``'mean'`` / ``'sum'`` / ``'none'`` over the t-axis.
    """

    def __init__(self,
                 diagonal_only: bool = True,
                 loss_mode: str = 'mse',
                 normalize_per_t: bool = True,
                 L_assembly: str = 'from_stiffness',
                 eps: float = 1e-12,
                 reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        if loss_mode not in ('mse', 'relative', 'signed_log'):
            raise ValueError(
                f"loss_mode must be 'mse' / 'relative' / 'signed_log', got '{loss_mode}'")
        if L_assembly not in ('from_stiffness', 'diagonal_gram'):
            raise ValueError(
                f"L_assembly must be 'from_stiffness' / 'diagonal_gram', got '{L_assembly}'")
        self.diagonal_only = diagonal_only
        self.loss_mode = loss_mode
        self.normalize_per_t = normalize_per_t
        self.L_assembly = L_assembly
        self.eps = eps

    def _assemble_L(self, ctx: LossContext) -> torch.Tensor:
        from neural_local_laplacian.utils.laplacian_assembly import (
            assemble_from_stiffness_weights,
            assemble_diagonal_gram_laplacian,
        )
        if self.L_assembly == 'from_stiffness':
            return assemble_from_stiffness_weights(
                ctx.stiffness_weights, ctx.knn, areas=None)
        return assemble_diagonal_gram_laplacian(
            ctx.grad_coeffs, ctx.knn, areas=None)

    def forward(self, ctx: LossContext) -> torch.Tensor:
        self._check_required(ctx, ['stiffness_weights', 'areas', 'knn',
                                   'heat_kernel_times', 'heat_kernel_gt'])

        # Sanity-check the dataset convention matches the loss config.
        gt_diag_flag = ctx.heat_kernel_diagonal_only
        if gt_diag_flag is not None:
            gt_is_diag = bool(gt_diag_flag.item()
                              if torch.is_tensor(gt_diag_flag) else gt_diag_flag)
            if gt_is_diag != self.diagonal_only:
                raise ValueError(
                    f"HeatKernelLoss(diagonal_only={self.diagonal_only}) does "
                    f"not match the dataset (heat_kernel_diagonal_only="
                    f"{gt_is_diag}).  Either flip the loss config or rebuild "
                    f"the GT with the matching flag.")

        L = self._assemble_L(ctx)                                  # (n, n)
        n = L.shape[0]
        device = L.device
        dtype = L.dtype

        # M is diagonal — avoid materialising the full (n, n) inverse.
        # M⁻¹ L is computed by scaling each row of L by 1 / A_i.
        inv_areas = (1.0 / (ctx.areas + self.eps)).unsqueeze(-1)   # (n, 1)
        M_inv_L = inv_areas * L                                    # (n, n)

        times = ctx.heat_kernel_times.to(device=device, dtype=dtype)  # (T,)
        gt = ctx.heat_kernel_gt.to(device=device, dtype=dtype)        # (T, n) or (T, n, n)
        T = times.shape[0]

        per_t = []
        for t_idx in range(T):
            t = times[t_idx]
            K_pred = torch.matrix_exp(-t * M_inv_L)                # (n, n)
            if self.diagonal_only:
                K_pred = torch.diagonal(K_pred)                    # (n,)
            target = gt[t_idx]                                     # (n,) or (n, n)
            err = _scalar_error_per_element(
                K_pred, target, self.loss_mode, eps=self.eps)      # same shape
            if self.normalize_per_t:
                denom = (target ** 2).sum().clamp_min(self.eps).sqrt()
                err = err / denom
            per_t.append(err.mean() if self.reduction == 'mean'
                         else err.sum() if self.reduction == 'sum'
                         else err)
        if self.reduction == 'none':
            return torch.stack(per_t, dim=0)                       # (T, ...)
        return torch.stack(per_t).mean() if self.reduction == 'mean' \
            else torch.stack(per_t).sum()
