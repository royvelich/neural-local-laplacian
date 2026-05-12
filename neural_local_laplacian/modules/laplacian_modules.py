import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple

import wandb
from omegaconf import DictConfig, OmegaConf
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import lightning
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from neural_local_laplacian.utils.utils import (
    assemble_gradient_operator,
    compute_laplacian_eigendecomposition,
    build_patches_from_vertices,
)
from neural_local_laplacian.utils.laplacian_assembly import (
    LaplacianConfig,
    assemble_laplacian,
    to_scipy_sparse,
    mass_matrix_to_scipy,
)
from neural_local_laplacian.modules.losses import LossConfig, LossContext
from neural_local_laplacian.utils.features import FeatureExtractor
from neural_local_laplacian.utils.geodesic_utils import (
    compute_heat_geodesic_learned,
    compute_multisource_geodesic_metrics,
)

# Shared fmap evaluation utilities (used when fmap_val_cfg is set)
from fmaps_finetune.utils.fmap_eval_utils import (
    evaluate_pair as _fmap_evaluate_pair,
    build_gt_corr_from_pair as _fmap_build_gt_corr,
    build_geo_cache as _fmap_build_geo_cache,
    GeodesicCache as _FmapGeodesicCache,
    geo_cache_path as _fmap_geo_cache_path,
    precompute_geo_cache_worker as _fmap_precompute_geo_cache_worker,
    summarise_fmap_metrics as _fmap_summarise,
)
from fmaps_finetune.modules.evaluators import SpectralNNEvaluator, FunctionalMapEvaluator


def _eigh_full_gram(L: torch.Tensor, M_diag: torch.Tensor,
                    num_eigenvectors: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized eigenproblem L v = lambda M v on GPU via standard form.

    Converts to M^{-1/2} L M^{-1/2} w = lambda w, solves, transforms back.
    """
    M_inv_sqrt = 1.0 / M_diag.sqrt().clamp(min=1e-8)
    L_std = L * M_inv_sqrt[:, None] * M_inv_sqrt[None, :]
    L_std = 0.5 * (L_std + L_std.T)

    all_evals, all_evecs = torch.linalg.eigh(L_std)
    evals = all_evals[:num_eigenvectors].cpu().numpy()
    evecs = (M_inv_sqrt[:, None] * all_evecs[:, :num_eigenvectors]).cpu().numpy()
    return evals, evecs


class LaplacianModuleBase(lightning.pytorch.LightningModule):
    def __init__(self,
                 optimizer_cfg: DictConfig,
                 scheduler_cfg: Optional[DictConfig] = None,
                 **kwargs):
        super().__init__()
        self._optimizer_cfg = optimizer_cfg
        self._scheduler_cfg = scheduler_cfg

    def setup(self, stage):
        if self.trainer.global_rank == 0:
            from neural_local_laplacian.utils.utils import wandb_log_code
            wandb_log_code(self.logger)

            if wandb.run is not None and hasattr(self.trainer, 'cfg'):
                dict_cfg = OmegaConf.to_container(self.trainer.cfg, resolve=True)
                self.logger.experiment.config.update(dict_cfg)

    def configure_optimizers(self):
        """Configure optimizer and optionally scheduler."""
        if self._optimizer_cfg is None:
            raise ValueError("optimizer_cfg is required but was None")

        optimizer = self._optimizer_cfg(params=self.parameters())

        if self._scheduler_cfg is None:
            return optimizer

        scheduler = self._scheduler_cfg(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": 'epoch'},
        }


# ============================================================================
# Patch encoders — the pluggable per-element mixing stage
# ============================================================================
#
# All encoders share the same contract:
#   forward(sequences: (B, max_k, d_model), attention_mask: (B, max_k) bool)
#     -> encoded: (B, max_k, d_model)
#
# They must be permutation-equivariant in the k dimension: permuting the
# neighbors of a patch must produce the correspondingly-permuted output.
#
# These are the ONLY architecture-specific piece. Everything upstream
# (patch-feature normalization, feature extractor, input projection) and
# everything downstream (grad head, area head, 1/sqrt(k) scaling, loss
# computation, validation) is shared across encoder choices.
# ============================================================================


class PatchEncoder(nn.Module):
    """Abstract permutation-equivariant encoder over a padded patch sequence.

    Concrete subclasses define the mixing architecture (transformer,
    DeepSet, ...). The module that owns the encoder is responsible for
    building the padded representation and the mask.
    """

    def forward(self,
                sequences: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerPatchEncoder(PatchEncoder):
    """Multi-head self-attention encoder stack (the original architecture)."""

    def __init__(self,
                 d_model: int,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers,
        )

    def forward(self,
                sequences: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        # PyTorch's TransformerEncoder takes key_padding_mask where True = ignore.
        # Our attention_mask is True = valid, so invert.
        if attention_mask.all():
            # Fixed-k path — no mask needed, faster (matches original behaviour)
            return self.encoder(sequences, src_key_padding_mask=None)
        return self.encoder(sequences, src_key_padding_mask=~attention_mask)


class DeepSetPatchEncoder(PatchEncoder):
    """Equivariant DeepSet encoder stack.

    Each layer computes:
        h_elem    = MLP_elem(x)                           # per-element
        context   = masked_pool(h_elem, mask)             # permutation-invariant
        x <- x + MLP_combine(concat(h_elem, broadcast(context)))

    The residual connection keeps the overall map equivariant and lets
    information propagate through the layer without being bottlenecked
    by the pool.

    Pooling: 'mean' (default), 'sum', or 'max'. Mean is usually a good
    default for small, roughly-constant-sized patches.
    """

    def __init__(self,
                 d_model: int,
                 num_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 pool: str = 'mean',
                 activation: str = 'gelu'):
        super().__init__()
        if pool not in ('mean', 'sum', 'max'):
            raise ValueError(f"pool must be 'mean', 'sum' or 'max', got '{pool}'")
        self._pool = pool
        self._num_layers = num_layers

        act_cls = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                # Per-element transform: d_model -> dff -> d_model
                'elem': nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, dim_feedforward),
                    act_cls(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                ),
                # Combine [h_elem, context_broadcast] (2*d_model) -> d_model
                'combine': nn.Sequential(
                    nn.LayerNorm(2 * d_model),
                    nn.Linear(2 * d_model, dim_feedforward),
                    act_cls(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                ),
            })
            for _ in range(num_layers)
        ])

    def _pool_fn(self,
                 h: torch.Tensor,
                 mask_f: torch.Tensor) -> torch.Tensor:
        """Masked pool over the k dimension. h: (B, K, D), mask_f: (B, K, 1)."""
        if self._pool == 'mean':
            num = (h * mask_f).sum(dim=1)
            den = mask_f.sum(dim=1).clamp(min=1.0)
            return num / den
        if self._pool == 'sum':
            return (h * mask_f).sum(dim=1)
        # max: set invalid positions to -inf so they don't win
        neg_inf = torch.finfo(h.dtype).min
        h_masked = h.masked_fill(mask_f == 0, neg_inf)
        return h_masked.max(dim=1).values

    def forward(self,
                sequences: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        # (B, K, 1) float mask for arithmetic
        mask_f = attention_mask.unsqueeze(-1).to(sequences.dtype)

        x = sequences
        for layer in self.layers:
            h = layer['elem'](x)                                  # (B, K, D)
            h = h * mask_f                                         # zero invalid
            context = self._pool_fn(h, mask_f)                     # (B, D)
            context_bcast = context.unsqueeze(1).expand_as(h)      # (B, K, D)
            combined = torch.cat([h, context_bcast], dim=-1)       # (B, K, 2D)
            update = layer['combine'](combined) * mask_f           # (B, K, D)
            x = x + update                                         # residual
        return x


class LaplacianLocalModule(LaplacianModuleBase):
    """Neural local Laplacian module with a pluggable patch encoder.

    All architecture-agnostic logic lives here:
      - patch feature normalization, optional feature extractor, input
        projection, padding / masking, 1/sqrt(k) grad normalization,
        gradient and area heads, loss computation, eigendecomposition,
        geodesic / fmap validation.

    The only architecture-specific piece is ``patch_encoder``: any
    ``PatchEncoder`` subclass with the contract
        (B, max_k, d_model), (B, max_k) -> (B, max_k, d_model)
    plugs in without touching any other code.

    Concrete subclasses (``LaplacianTransformerModule``,
    ``LaplacianDeepSetModule``) just build the right encoder and forward
    the rest of the kwargs to this constructor.
    """

    def __init__(self,
                 patch_encoder: PatchEncoder,
                 d_model: int,
                 input_dim: Optional[int] = None,
                 loss_configs: Optional[List[LossConfig]] = None,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 dropout: float = 0.1,
                 num_eigenvalues: int = 10,
                 normalize_loss_weights: bool = True,
                 input_projection_hidden_dims: Optional[List[int]] = None,
                 output_projection_hidden_dims: Optional[List[int]] = None,
                 normalize_patch_features: bool = True,
                 scale_areas_by_patch_size: bool = True,
                 area_activation: str = 'softplus',
                 area_bound_C: float = 12.566370614359172,  # 4*pi
                 mcv_mode: str = 'diagonal_gram',
                 stiffness_mode: str = 'diagonal_gram',
                 normalize_grad_by_k: bool = False,
                 detach_area_head: bool = False,
                 use_uniform_mass: bool = False,
                 val_laplacian: Optional[Dict] = None,
                 fmap_val_cfg: Optional[Dict] = None,
                 enable_nan_diagnostics: bool = True,
                 nan_diag_log_every_n_steps: int = 50,
                 **kwargs):
        # **kwargs absorbs legacy hparams (operator_mode, patch_mcv_mode,
        # val_laplacian_mode) from old checkpoints.
        super().__init__(**{k: v for k, v in kwargs.items()
                           if k in ('optimizer_cfg', 'scheduler_cfg')})

        self.save_hyperparameters(ignore=['loss_configs', 'feature_extractor',
                                          'patch_encoder'])

        # Manually save loss configuration info for logging (serializable version)
        if loss_configs is not None:
            self.hparams['loss_info'] = {
                'num_losses': len(loss_configs),
                'loss_types': [type(config.loss_module).__name__ for config in loss_configs],
                'loss_weights': [config.weight for config in loss_configs],
                'normalize_loss_weights': normalize_loss_weights
            }
            if normalize_loss_weights:
                self.hparams['loss_info']['normalized_weights'] = [
                    config.weight for config in self._normalize_loss_weights(loss_configs)
                ]

        # Validate input_dim / feature_extractor
        if feature_extractor is not None:
            if isinstance(feature_extractor, nn.Module):
                self.feature_extractor = feature_extractor
            else:
                self.feature_extractor = None
                self._feature_extractor_fn = feature_extractor
            resolved_input_dim = feature_extractor.output_dim
            if input_dim is not None and input_dim != resolved_input_dim:
                raise ValueError(
                    f"input_dim={input_dim} conflicts with feature_extractor.output_dim={resolved_input_dim}. "
                    f"When a feature_extractor is provided, input_dim is inferred automatically."
                )
        else:
            self.feature_extractor = None
            if input_dim is None or input_dim <= 0:
                raise ValueError("input_dim must be a positive integer when no feature_extractor is provided.")
            resolved_input_dim = input_dim

        self._d_model = d_model
        self._input_dim = resolved_input_dim
        self._num_eigenvalues = num_eigenvalues
        self._normalize_patch_features = normalize_patch_features
        self._scale_areas_by_patch_size = scale_areas_by_patch_size
        self._area_activation = area_activation
        self._area_bound_C = area_bound_C
        self._mcv_mode = mcv_mode
        _stiffness_modes = ('diagonal_gram', 'full_gram', 'learned', 'learned_positive')
        if stiffness_mode not in _stiffness_modes:
            raise ValueError(
                f"stiffness_mode must be one of {_stiffness_modes}, got '{stiffness_mode}'")
        self._stiffness_mode = stiffness_mode
        self._normalize_grad_by_k = normalize_grad_by_k
        self._detach_area_head = detach_area_head
        self._use_uniform_mass = use_uniform_mass

        # [DIAG] NaN/Inf diagnostics — see training_step and on_before_optimizer_step
        self._enable_nan_diagnostics = enable_nan_diagnostics
        self._nan_diag_log_every = max(int(nan_diag_log_every_n_steps), 1)

        # Validation Laplacian config
        _val_lap = val_laplacian or {'assembly': 'diagonal_gram', 'pruning': 'none'}
        self._val_lap_config = LaplacianConfig(**_val_lap)

        # Fmap validation config (optional — enables pair-based correspondence eval)
        self._fmap_val_cfg = fmap_val_cfg
        self._fmap_val_evaluators: Optional[List] = None
        self._fmap_val_eval_lap_configs: Optional[List[LaplacianConfig]] = None
        self._fmap_val_geo_cache: Dict[str, Optional[_FmapGeodesicCache]] = {}
        self._fmap_val_outputs: Dict[int, List[Dict[str, float]]] = {}
        if fmap_val_cfg is not None:
            # Build evaluators
            self._fmap_val_evaluators = [SpectralNNEvaluator()]
            if fmap_val_cfg.get('use_geomfum_eval', False):
                self._fmap_val_evaluators.append(FunctionalMapEvaluator(
                    descriptors=fmap_val_cfg.get('geomfum_descriptors', ['hks', 'wks']),
                    use_zoomout=fmap_val_cfg.get('geomfum_zoomout', True),
                    zoomout_k_init=fmap_val_cfg.get('geomfum_zoomout_k_init', 20),
                    zoomout_k_final=fmap_val_cfg.get('geomfum_zoomout_k_final', 50),
                    zoomout_n_iters=fmap_val_cfg.get('geomfum_zoomout_n_iters', 10),
                    fmap_lmbda=fmap_val_cfg.get('fmap_lmbda', 1e3),
                    fmap_resolvent_gamma=fmap_val_cfg.get('fmap_resolvent_gamma', 1.0),
                ))
            # Build eval Laplacian configs
            _eval_laps = fmap_val_cfg.get('eval_laplacians',
                                          [{'assembly': 'diagonal_gram', 'pruning': 'none'}])
            self._fmap_val_eval_lap_configs = [LaplacianConfig(**d) for d in _eval_laps]

        # Store loss configs (optionally normalized)
        if normalize_loss_weights:
            self._loss_configs = self._normalize_loss_weights(loss_configs)
        else:
            self._loss_configs = loss_configs

        # Input projection
        self.input_projection = self._build_projection(resolved_input_dim, d_model, input_projection_hidden_dims)

        # Pluggable patch encoder — the only architecture-specific piece
        self.patch_encoder = patch_encoder

        # Output head: gradient coefficients g_ij in R^3
        self.grad_projection = self._build_projection(d_model, 3, output_projection_hidden_dims)

        # Output head: per-edge scalar stiffness s_ij.
        # Only created when stiffness_mode requires it (DDP-safe: no
        # unused params when stiffness is derived from grad_coeffs via
        # the 'diagonal_gram' / 'full_gram' modes).
        if stiffness_mode in ('learned', 'learned_positive'):
            self.stiffness_projection = self._build_projection(
                d_model, 1, output_projection_hidden_dims)

        # Area head: aggregated features -> scalar area A_i
        # Activation applied manually via _apply_area_activation
        # Omitted entirely when use_uniform_mass=True (DDP-safe: no unused params)
        if not use_uniform_mass:
            self.area_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

    def _apply_area_activation(self, areas_raw: torch.Tensor,
                                batch_sizes: torch.Tensor) -> torch.Tensor:
        """Apply area activation function to raw area head output.

        Args:
            areas_raw: (batch_size,) raw logits from area head.
            batch_sizes: (batch_size,) number of neighbors per patch (k).
                         Currently unused but kept for API compatibility.

        Returns:
            (batch_size,) positive area values.
        """
        if self._area_activation == 'bounded_sigmoid':
            # A_max = C (constant) — center vertex area on a unit-sphere patch
            # is O(1) regardless of k.  Do NOT divide by k: that would make
            # M_ii ~ 1/k while S_ii ~ 1 (with normalize_grad_by_k), causing
            # eigenvalues to grow as O(k).
            return self._area_bound_C * torch.sigmoid(areas_raw)
        else:
            # Default: unbounded softplus
            return F.softplus(areas_raw)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _diag_tensor_summary(name: str, t: torch.Tensor) -> str:
        """[DIAG] One-line summary of a tensor for NaN/Inf debug prints."""
        if t is None:
            return f"{name}: <None>"
        try:
            shape = tuple(t.shape)
            finite = torch.isfinite(t)
            finite_frac = finite.float().mean().item()
            if finite.any():
                t_finite = t[finite]
                stats = (f"min={t_finite.min().item():.3e} "
                         f"max={t_finite.max().item():.3e} "
                         f"mean={t_finite.mean().item():.3e}")
            else:
                stats = "ALL non-finite"
            return (f"{name}: shape={shape} finite={finite_frac:.4f} {stats}")
        except Exception as e:
            return f"{name}: <error inspecting tensor: {e}>"

    def _diag_dump_context(self, header: str, *,
                           batch_idx: Optional[int] = None,
                           extra: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """[DIAG] Print a multi-line context dump tagged with epoch/step/rank.

        Used right before raising on a detected NaN/Inf so the log captures
        enough context to diagnose which sample and which tensor went bad.
        """
        try:
            rank = self.global_rank
        except Exception:
            rank = -1
        print(f"[NAN-DETECT] {header} | "
              f"epoch={self.current_epoch} step={self.global_step} "
              f"rank={rank} batch_idx={batch_idx}", flush=True)
        if extra:
            for k, v in extra.items():
                print(f"  {self._diag_tensor_summary(k, v)}", flush=True)

    @staticmethod
    def _build_projection(in_dim: int, out_dim: int,
                          hidden_dims: Optional[List[int]] = None) -> nn.Module:
        """Build a linear or MLP projection."""
        if hidden_dims is None or len(hidden_dims) == 0:
            return nn.Linear(in_dim, out_dim)
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim),
                           nn.LayerNorm(hidden_dim), nn.GELU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)

    def _normalize_loss_weights(self, loss_configs: List[LossConfig]) -> List[LossConfig]:
        """Normalize loss weights so non-None weights sum to 1."""
        if not loss_configs:
            return loss_configs
        total_weight = sum(c.weight for c in loss_configs if c.weight is not None)
        if total_weight == 0:
            raise ValueError("Total loss weights cannot be zero")
        normalized = []
        for c in loss_configs:
            if c.weight is None:
                normalized.append(c)
            else:
                normalized.append(LossConfig(
                    loss_module=c.loss_module, weight=c.weight / total_weight))
        return normalized

    def _pad_sequences_vectorized(self, features: torch.Tensor,
                                  batch_indices: torch.Tensor,
                                  batch_size: int, max_k: int
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad variable-length sequences to (batch_size, max_k, d_model)."""
        device = features.device
        d_model = features.shape[1]

        sorted_indices = torch.argsort(batch_indices)
        sorted_batch = batch_indices[sorted_indices]
        batch_sizes = torch.bincount(batch_indices, minlength=batch_size)

        cumsum_sizes = torch.cumsum(batch_sizes, dim=0)
        starts = torch.cat([torch.tensor([0], device=device), cumsum_sizes[:-1]])

        total_points = batch_indices.shape[0]
        arange_full = torch.arange(total_points, device=device)
        batch_starts = starts[sorted_batch]

        positions = torch.zeros_like(batch_indices, dtype=torch.long)
        positions[sorted_indices] = arange_full[sorted_indices] - batch_starts

        valid_mask = positions < max_k
        flat_indices = batch_indices[valid_mask] * max_k + positions[valid_mask]

        sequences = torch.zeros(batch_size * max_k, d_model, device=device,
                                dtype=features.dtype)
        attention_mask = torch.zeros(batch_size * max_k, dtype=torch.bool,
                                     device=device)

        sequences.scatter_(0, flat_indices.unsqueeze(1).expand(-1, d_model),
                           features[valid_mask])
        attention_mask.scatter_(0, flat_indices, True)

        return (sequences.view(batch_size, max_k, d_model),
                attention_mask.view(batch_size, max_k))

    def _compute_mean_curvature_vectors(self, forward_result: Dict[str, torch.Tensor],
                                        batch_data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute predicted mean curvature vectors from gradient coefficients and areas.

        Returns:
            Tuple of (predicted_mcv, raw_stiffness_action):
            - predicted_mcv: MCV = (Σ w_ij (p_j - p_i)) / A_i, shape (batch_size, 3)
            - raw_stiffness_action: Σ w_ij (p_j - p_i), shape (batch_size, 3)
        """
        areas = forward_result['areas']
        attention_mask = forward_result['attention_mask']
        batch_sizes = forward_result['batch_sizes']
        batch_size = len(batch_sizes)

        if self._mcv_mode == 'full_gram':
            grad_coeffs = forward_result['grad_coeffs']
            gc_masked = grad_coeffs.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
            g_self = -gc_masked.sum(dim=1)
            weights = torch.einsum('bd,bkd->bk', g_self, gc_masked)
            weights = weights.masked_fill(~attention_mask, 0.0)
        else:
            # diagonal_gram: w_ij = s_ij = ||g_ij||^2
            weights = forward_result['stiffness_weights'].masked_fill(~attention_mask, 0.0)

        positions = batch_data.pos
        weights_flat = weights.flatten()
        batch_indices_w = torch.arange(batch_size, device=weights.device
                                       ).repeat_interleave(weights.shape[1])
        batch_cumsum = torch.cumsum(batch_sizes, dim=0)
        batch_starts = torch.cat([torch.zeros(1, device=batch_cumsum.device,
                                              dtype=batch_cumsum.dtype),
                                  batch_cumsum[:-1]])
        position_indices = torch.arange(weights.shape[1], device=weights.device
                                        ).repeat(batch_size)
        valid_mask = position_indices < batch_sizes.repeat_interleave(weights.shape[1])

        valid_weights = weights_flat[valid_mask]
        valid_batch_idx = batch_indices_w[valid_mask]
        actual_pos_idx = batch_starts[valid_batch_idx] + position_indices[valid_mask]
        weighted_pos = valid_weights.unsqueeze(-1) * positions[actual_pos_idx]

        stiffness_sum = torch.zeros(batch_size, 3, device=weights.device)
        stiffness_sum.scatter_add_(
            0, valid_batch_idx.unsqueeze(-1).expand(-1, 3), weighted_pos)

        predicted_mcv = stiffness_sum / areas.unsqueeze(-1)
        return predicted_mcv, stiffness_sum

    def _reshape_positions_to_batched(self, pos_flat: torch.Tensor,
                                      batch_sizes: torch.Tensor) -> torch.Tensor:
        """Reshape flat (total_points, D) to padded (batch_size, max_k, D)."""
        batch_size = len(batch_sizes)
        max_k = batch_sizes.max().item()
        D = pos_flat.shape[-1]
        if torch.all(batch_sizes == batch_sizes[0]):
            return pos_flat.view(batch_size, max_k, D)
        out = torch.zeros(batch_size, max_k, D, device=pos_flat.device,
                          dtype=pos_flat.dtype)
        offset = 0
        for i in range(batch_size):
            sz = batch_sizes[i].item()
            out[i, :sz] = pos_flat[offset:offset + sz]
            offset += sz
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass supporting both fixed-k and variable-sized patches.

        Automatically selects the optimized fixed-k path (no padding, no masking)
        when all patches have the same size.

        Returns:
            Dict with stiffness_weights, areas, attention_mask, batch_sizes,
            scale_factors, grad_coeffs.
        """
        features = batch.x
        positions = batch.pos
        batch_indices = getattr(batch, 'patch_idx', batch.batch)

        batch_sizes = batch_indices.bincount()
        batch_size = len(batch_sizes)
        max_k = batch_sizes.max().item()
        fixed_k = torch.all(batch_sizes == batch_sizes[0]).item()

        # ── Normalize features by per-patch scale ────────────────────
        if self._normalize_patch_features:
            distances = torch.norm(positions, dim=1)
            scale_factors = torch.zeros(batch_size, device=positions.device,
                                        dtype=positions.dtype)
            scale_factors.scatter_reduce_(
                0, batch_indices, distances, reduce='amax', include_self=True)
            scale_factors = torch.clamp(scale_factors, min=1e-8)
            features = features / scale_factors[batch_indices].unsqueeze(-1)
        else:
            scale_factors = torch.ones(batch_size, device=features.device,
                                        dtype=features.dtype)

        # ── Feature extractor ────────────────────────────────────────
        if self.feature_extractor is not None:
            if fixed_k:
                features = features.view(batch_size, max_k, -1)
                features = self.feature_extractor.extract_features(features)
                features = features.view(batch_size * max_k, -1)
            else:
                extracted = []
                start = 0
                for sz in batch_sizes:
                    sz = sz.item()
                    extracted.append(
                        self.feature_extractor.extract_features(
                            features[start:start + sz]))
                    start += sz
                features = torch.cat(extracted, dim=0)

        # ── Input projection ─────────────────────────────────────────
        features = self.input_projection(features)

        # ── Sequence padding + encoding ──────────────────────────────
        if fixed_k:
            sequences = features.view(batch_size, max_k, -1)
            attention_mask = torch.ones(batch_size, max_k, dtype=torch.bool,
                                        device=features.device)
        else:
            sequences, attention_mask = self._pad_sequences_vectorized(
                features, batch_indices, batch_size, max_k)
        encoded = self.patch_encoder(sequences, attention_mask)

        # ── Output heads ─────────────────────────────────────────────
        grad_coeffs = self.grad_projection(encoded)

        # Normalize grad_coeffs by 1/sqrt(k) so that the Laplacian's
        # eigenvalues stay stable as k changes.  ||g_ij/sqrt(k)||^2 = ||g_ij||^2/k
        # turns the neighbor sum into a mean.
        if self._normalize_grad_by_k:
            k_per_patch = batch_sizes.float()  # (batch_size,)
            grad_coeffs = grad_coeffs / k_per_patch[:, None, None].sqrt()

        if self._stiffness_mode == 'full_gram':
            # FEM-style per-edge stiffness from the predicted gradient operator:
            #   s_ij = -⟨g_ii, g_ij⟩    with    g_ii = -Σ_k g_ik
            # Equivalently:
            #   s_ij = ‖g_ij‖² + Σ_{k≠j} ⟨g_ik, g_ij⟩
            # so it equals the diagonal-gram value plus the cross-Gram terms
            # involving neighbour j.  Reduces to ‖g_ij‖² when the g_ij are
            # mutually orthogonal.  Can be negative when cross terms dominate.
            #
            # Note: this picks the sign convention so that diagonal-gram and
            # full-gram weights agree in the orthogonal limit.  The MCV path's
            # ``mcv_mode='full_gram'`` snippet uses ⟨g_ii, g_ij⟩ (no flip)
            # internally and is independent of this field; flipping the sign
            # convention there would break sign-symmetric behaviour learned
            # by existing checkpoints, so it is intentionally left alone.
            gc_masked = grad_coeffs.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
            g_self = -gc_masked.sum(dim=1, keepdim=True)              # (B, 1, 3) = g_ii
            inner = (g_self * gc_masked).sum(dim=-1)                   # (B, K) = ⟨g_ii, g_ij⟩
            stiffness_weights = -inner                                 # (B, K) = -⟨g_ii, g_ij⟩
        elif self._stiffness_mode in ('learned', 'learned_positive'):
            # Independent stiffness head: s_ij is a direct network
            # output, decoupled from ‖g_ij‖².  This breaks the
            # structural identity ``L = G^T M G`` (the assembled
            # Laplacian from g and the stiffness-action Laplacian from
            # s are no longer guaranteed to be the same operator) — pair
            # with a soft consistency loss
            # (DirichletEnergyConsistencyLoss) if you want them to track
            # each other.
            stiffness_weights = self.stiffness_projection(encoded).squeeze(-1)  # (B, K)
            if self._stiffness_mode == 'learned_positive':
                # PSD-by-construction: smooth, monotone, ≥ 0.
                stiffness_weights = F.softplus(stiffness_weights)
            # 'learned' (signed) lets the operator be indefinite — the
            # FEM convention on obtuse cells.
        else:
            # 'diagonal_gram' (default): s_ij = ‖g_ij‖²
            stiffness_weights = (grad_coeffs ** 2).sum(dim=-1)

        # ── Area prediction ──────────────────────────────────────────
        if self._use_uniform_mass:
            # Uniform mass: M_ii = scale² (or 1.0 if not scaling)
            if self._scale_areas_by_patch_size:
                areas = scale_factors ** 2
            else:
                areas = torch.ones(batch_size, device=grad_coeffs.device,
                                   dtype=grad_coeffs.dtype)
        else:
            if fixed_k:
                pooled = encoded.mean(dim=1)
            else:
                float_mask = attention_mask.float()
                num_tokens = float_mask.sum(dim=1, keepdim=True)
                pooled = (encoded * float_mask.unsqueeze(-1)).sum(dim=1) / num_tokens

            # Optionally detach: area head reads encoder features but doesn't
            # send gradients back — encoder is trained only by gradient/stiffness losses.
            if self._detach_area_head:
                pooled = pooled.detach()

            areas_raw = self.area_head(pooled).squeeze(-1)
            areas_normalized = self._apply_area_activation(areas_raw, batch_sizes)
            if self._scale_areas_by_patch_size:
                areas = areas_normalized * (scale_factors ** 2)
            else:
                areas = areas_normalized

        return {
            'stiffness_weights': stiffness_weights,
            'areas': areas,
            'attention_mask': attention_mask,
            'batch_sizes': batch_sizes,
            'scale_factors': scale_factors,
            'grad_coeffs': grad_coeffs,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step with variable-sized patch support."""
        batch_data = batch[0]
        forward_result = self.forward(batch_data)

        # [DET-DIAG] Hash model parameters and forward output at the first
        # few training steps. Compare these across two runs to localize where
        # determinism breaks:
        #   - param_hash differs at step 0    → model init non-deterministic
        #   - param_hash matches but pos_hash from data differs
        #                                      → DataLoader / sampler non-det
        #   - both match but fwd_hash differs → forward path non-deterministic
        #     (likely dropout fed by non-det torch global RNG)
        #   - all match at step 0 but diverge later → loss / backward / DDP
        if not hasattr(self, '_det_diag_step_count'):
            self._det_diag_step_count = 0
        if self._det_diag_step_count < 3:
            import hashlib
            ph = hashlib.md5()
            for p in self.parameters():
                ph.update(p.detach().cpu().numpy().tobytes())
            param_hash = ph.hexdigest()[:16]

            fh = hashlib.md5()
            for k in ('grad_coeffs', 'areas', 'stiffness_weights'):
                v = forward_result.get(k)
                if v is not None:
                    arr = v.detach().cpu().numpy()
                    fh.update(arr.tobytes())
                    fh.update(str(arr.shape).encode())
                    fh.update(k.encode())
            fwd_hash = fh.hexdigest()[:16]

            ih = hashlib.md5()
            if hasattr(batch_data, 'pos') and batch_data.pos is not None:
                ih.update(batch_data.pos.detach().cpu().numpy().tobytes())
            input_hash = ih.hexdigest()[:16]

            print(
                f"[DET-DIAG-STEP] rank={self.global_rank} epoch={self.current_epoch} "
                f"step={batch_idx} param_hash={param_hash} "
                f"input_pos_hash={input_hash} fwd_hash={fwd_hash}",
                flush=True,
            )
            self._det_diag_step_count += 1

        # ── [DIAG] Forward output sanity check + cheap stats ────────────
        if self._enable_nan_diagnostics:
            g_check = forward_result.get('grad_coeffs')
            A_check = forward_result.get('areas')
            S_check = forward_result.get('stiffness_weights')
            mask_check = forward_result.get('attention_mask')

            bad_forward = False
            if g_check is not None and not torch.isfinite(g_check).all():
                bad_forward = True
            if A_check is not None and not torch.isfinite(A_check).all():
                bad_forward = True
            if S_check is not None and not torch.isfinite(S_check).all():
                bad_forward = True

            if bad_forward:
                self._diag_dump_context(
                    "Non-finite tensor in forward output",
                    batch_idx=batch_idx,
                    extra={
                        'grad_coeffs': g_check,
                        'areas': A_check,
                        'stiffness_weights': S_check,
                        'attention_mask': (mask_check.float()
                                            if mask_check is not None else None),
                        'batch.pos': getattr(batch_data, 'pos', None),
                    },
                )
                raise RuntimeError(
                    "Non-finite values in forward output — see [NAN-DETECT] log above"
                )

            # Cheap continuous monitoring: log a few stats every N steps.
            # Lightning will route these to W&B via on_step=True.
            if self.global_step % self._nan_diag_log_every == 0:
                # Fraction of areas saturated near upper bound — relevant
                # when area_activation='bounded_sigmoid' and area_bound_C
                # was set high. Saturation -> vanishing gradient, then
                # AdamW eats accumulator state.
                if A_check is not None and A_check.numel() > 0:
                    sat_thresh = 0.9 * float(self._area_bound_C)
                    self.log('diag/A_max', A_check.max().item(),
                             on_step=True, on_epoch=False, logger=True,
                             rank_zero_only=True)
                    self.log('diag/A_min', A_check.min().item(),
                             on_step=True, on_epoch=False, logger=True,
                             rank_zero_only=True)
                    self.log('diag/A_at_upper_bound_frac',
                             (A_check > sat_thresh).float().mean().item(),
                             on_step=True, on_epoch=False, logger=True,
                             rank_zero_only=True)
                if g_check is not None and g_check.numel() > 0:
                    self.log('diag/g_max_abs', g_check.abs().max().item(),
                             on_step=True, on_epoch=False, logger=True,
                             rank_zero_only=True)
                    g_norm = g_check.norm(dim=-1)
                    self.log('diag/g_norm_max', g_norm.max().item(),
                             on_step=True, on_epoch=False, logger=True,
                             rank_zero_only=True)

        predicted_mcv, predicted_raw_mcv = self._compute_mean_curvature_vectors(forward_result, batch_data)

        batch_size = len(forward_result['batch_sizes'])
        normals = batch_data.normal
        mean_curvatures = batch_data.H
        target_mcv = 2.0 * mean_curvatures.unsqueeze(-1) * F.normalize(normals, p=2, dim=1)

        # Reshape test function deltas from flat (total_points, P) to (B, max_k, P)
        tf_deltas = None
        if hasattr(batch_data, 'test_func_deltas') and batch_data.test_func_deltas is not None:
            P = batch_data.test_func_deltas.shape[-1]
            tf_deltas = self._reshape_positions_to_batched(
                batch_data.test_func_deltas, forward_result['batch_sizes'])  # (B, max_k, P)

        loss_context = LossContext(
            predicted_mcv=predicted_mcv,
            target_mcv=target_mcv,
            predicted_raw_mcv=predicted_raw_mcv,
            grad_coeffs=forward_result.get('grad_coeffs'),
            positions=(self._reshape_positions_to_batched(
                           batch_data.pos, forward_result['batch_sizes'])
                       if forward_result.get('grad_coeffs') is not None else None),
            normals=getattr(batch_data, 'normal', None),
            attention_mask=forward_result['attention_mask'],
            areas=forward_result['areas'],
            stiffness_weights=forward_result['stiffness_weights'],
            gt_vertex_areas=getattr(batch_data, 'gt_vertex_areas', None),
            test_func_deltas=tf_deltas,
            test_func_laplacians=getattr(batch_data, 'test_func_laplacians', None),
            test_func_gradients=getattr(batch_data, 'test_func_gradients', None),
        )

        total_loss = 0.0
        loss_components_unweighted = {}

        for loss_config in self._loss_configs:
            unweighted_loss = loss_config.loss_module(loss_context)
            loss_name = loss_config.loss_module.__class__.__name__

            # ── [DIAG] Per-loss NaN/Inf check ─────────────────────────
            # Catches the SPECIFIC loss term that introduced non-finite
            # values, before they get summed into total_loss and lose
            # attribution. Fail fast: continuing with NaN poisons AdamW
            # accumulators and the next ~10000 steps before observable.
            if (self._enable_nan_diagnostics
                    and torch.is_tensor(unweighted_loss)
                    and not torch.isfinite(unweighted_loss).all()):
                weight = loss_config.weight if loss_config.weight is not None else 0.0
                self._diag_dump_context(
                    f"Loss {loss_name} (weight={weight}) returned non-finite value "
                    f"= {unweighted_loss.detach()}",
                    batch_idx=batch_idx,
                    extra={
                        'predicted_mcv': predicted_mcv,
                        'target_mcv': target_mcv,
                        'predicted_raw_mcv': predicted_raw_mcv,
                        'grad_coeffs': forward_result.get('grad_coeffs'),
                        'areas': forward_result['areas'],
                        'stiffness_weights': forward_result['stiffness_weights'],
                        'gt_vertex_areas': getattr(batch_data, 'gt_vertex_areas', None),
                        'batch.H': mean_curvatures,
                        'batch.normal': normals,
                    },
                )
                raise RuntimeError(
                    f"Loss {loss_name} produced non-finite value — "
                    f"see [NAN-DETECT] log above"
                )

            loss_components_unweighted[f"train/{loss_name}"] = unweighted_loss
            if loss_config.weight is not None:
                total_loss = total_loss + loss_config.weight * unweighted_loss

        if not isinstance(total_loss, torch.Tensor):
            raise ValueError("At least one loss must have a non-None weight for training")

        # ── [DIAG] Final aggregated-loss check (defense in depth) ──────
        # If a single component was finite but their weighted sum overflows
        # (extremely unlikely with sane weights, but possible), this catches it.
        if (self._enable_nan_diagnostics
                and not torch.isfinite(total_loss).all()):
            self._diag_dump_context(
                f"Aggregated total_loss is non-finite (={total_loss.detach()})",
                batch_idx=batch_idx,
                extra={f"loss/{n}": v for n, v in loss_components_unweighted.items()},
            )
            raise RuntimeError(
                "Aggregated total_loss is non-finite — see [NAN-DETECT] log above"
            )

        cosine_sim = F.cosine_similarity(predicted_mcv, target_mcv, dim=1).mean()
        areas = forward_result['areas']
        stiffness = forward_result['stiffness_weights']  # (B, max_k)
        mask = forward_result['attention_mask'].float()   # (B, max_k)

        self.log('train/loss', total_loss.item(), on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log('train/cosine_similarity', cosine_sim.item(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size,
                 sync_dist=True)
        self.log('train/area_mean', areas.mean().item(), on_step=False, on_epoch=True,
                 logger=True, batch_size=batch_size, sync_dist=True)
        self.log('train/area_std', areas.std().item(), on_step=False, on_epoch=True,
                 logger=True, batch_size=batch_size, sync_dist=True)

        # ── Diagnostic metrics ──────────────────────────────────────
        # Stiffness magnitude stats (masked)
        s_masked = stiffness * mask  # zero out padding
        s_valid = s_masked[mask.bool()]
        if s_valid.numel() > 0:
            self.log('train/stiffness_mean', s_valid.mean().item(), on_step=False,
                     on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
            self.log('train/stiffness_std', s_valid.std().item(), on_step=False,
                     on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
            self.log('train/stiffness_max', s_valid.max().item(), on_step=False,
                     on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)

        # Stiffness-to-area ratio: Σ s_ij / A_i per patch (eigenvalue scale)
        s_sum_per_patch = s_masked.sum(dim=-1)  # (B,)
        s_over_a = s_sum_per_patch / areas.clamp(min=1e-8)  # (B,)
        self.log('train/stiffness_area_ratio_mean', s_over_a.mean().item(), on_step=False,
                 on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log('train/stiffness_area_ratio_std', s_over_a.std().item(), on_step=False,
                 on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)

        # MCV magnitude
        mcv_mag = torch.norm(predicted_mcv, dim=1)  # (B,)
        self.log('train/mcv_magnitude_mean', mcv_mag.mean().item(), on_step=False,
                 on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log('train/mcv_magnitude_std', mcv_mag.std().item(), on_step=False,
                 on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)

        # Encoder weight norm (architecture-agnostic: works for any patch_encoder)
        enc_norm = sum(p.norm().item() ** 2 for p in self.patch_encoder.parameters()) ** 0.5
        self.log('train/encoder_weight_norm', enc_norm, on_step=False,
                 on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)

        for name, val in loss_components_unweighted.items():
            self.log(name, val, on_step=False, on_epoch=True, logger=True,
                     batch_size=batch_size, sync_dist=True)

        result = {"loss": total_loss}
        result.update(loss_components_unweighted)
        return result

    # ------------------------------------------------------------------
    # [DIAG] Gradient diagnostics — runs after backward, before optimizer.step
    # ------------------------------------------------------------------

    # Buckets for per-component gradient norm logging. Each parameter's
    # qualified name is matched against these substrings; first hit wins.
    _DIAG_GRAD_BUCKETS = (
        ('encoder',     'patch_encoder'),
        ('input_proj',  'input_projection'),
        ('grad_proj',   'grad_projection'),
        ('area_head',   'area_head'),
    )

    def on_before_optimizer_step(self, optimizer) -> None:
        """[DIAG] Inspect gradients before they're applied.

        Three things happen here:
        1. Detect any non-finite gradient → fail fast with the param name(s).
        2. Log overall gradient norm every step → cheap W&B series for
           after-the-fact debugging.
        3. Log per-component gradient norms occasionally → tells us which
           branch (encoder / area_head / etc.) is exploding when norms spike.
        """
        if not self._enable_nan_diagnostics:
            return

        bad_params: List[Tuple[str, Tuple[int, ...], float]] = []
        total_norm_sq = 0.0
        component_sq: Dict[str, float] = {b[0]: 0.0 for b in self._DIAG_GRAD_BUCKETS}
        component_sq['other'] = 0.0

        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            g = p.grad
            finite_mask = torch.isfinite(g)
            if not finite_mask.all():
                finite_frac = finite_mask.float().mean().item()
                bad_params.append((name, tuple(g.shape), finite_frac))
                # Don't accumulate this norm — it's NaN and would poison the sum.
                continue
            n_sq = float(g.norm().item() ** 2)
            total_norm_sq += n_sq

            # Bucket the parameter for per-component logging
            bucket_key = 'other'
            for key, substr in self._DIAG_GRAD_BUCKETS:
                if substr in name:
                    bucket_key = key
                    break
            component_sq[bucket_key] += n_sq

        # Fail fast on non-finite gradients — they indicate the cascade
        # has already started, even if total_loss happened to be finite
        # (e.g. NaN gradient through a saturated activation).
        if bad_params:
            try:
                rank = self.global_rank
            except Exception:
                rank = -1
            print(f"[NAN-DETECT] Non-finite gradients at "
                  f"epoch={self.current_epoch} step={self.global_step} rank={rank}:",
                  flush=True)
            for name, shape, frac in bad_params[:20]:  # cap output
                print(f"  {name}: shape={shape} finite_frac={frac:.4f}", flush=True)
            if len(bad_params) > 20:
                print(f"  ... and {len(bad_params) - 20} more", flush=True)
            raise RuntimeError(
                f"Non-finite gradients in {len(bad_params)} parameters — "
                f"see [NAN-DETECT] log above"
            )

        total_norm = total_norm_sq ** 0.5
        # Always log total grad norm (one scalar per step — negligible cost)
        self.log('diag/grad_norm', total_norm, on_step=True, on_epoch=False,
                 logger=True, rank_zero_only=True)

        # Per-component breakdown only every N steps (slightly more verbose log)
        if self.global_step % self._nan_diag_log_every == 0:
            for k, sq in component_sq.items():
                self.log(f'diag/grad_norm_{k}', sq ** 0.5,
                         on_step=True, on_epoch=False,
                         logger=True, rank_zero_only=True)

        # Soft warning on suspiciously large grad norm. Doesn't stop training
        # — gradient_clip_val in the trainer config (recommended: 1.0) handles
        # the actual clipping. This just makes the event visible in stdout.
        if total_norm > 100.0:
            print(f"[GRAD-WARN] Large grad norm {total_norm:.2f} at "
                  f"epoch={self.current_epoch} step={self.global_step}",
                  flush=True)

    # ------------------------------------------------------------------
    # Fmap validation: setup
    # ------------------------------------------------------------------

    def _fmap_val_enabled(self) -> bool:
        """Check if fmap pair validation is enabled."""
        return self._fmap_val_cfg is not None and self._fmap_val_evaluators is not None

    def _get_fmap_val_pairs_from_dm(self):
        """Collect fmap pair datasets from the data module's val specs.

        Fmap pair datasets are identified by having a collate_fn set on the
        DatasetSpecification (plain DataLoader) and items that are not PyG Data.
        Returns list of (ds_name, dataloader_idx, pairs).
        """
        dm = self.trainer.datamodule
        if dm is None or not hasattr(dm, '_val_dataset_specifications'):
            return []

        results = []
        for idx, spec in enumerate(dm._val_dataset_specifications):
            if spec.collate_fn is not None:
                # This is a plain (non-PyG) dataset — assumed to be fmap pairs
                ds = spec.dataset
                ds_name = getattr(ds, 'name', ds.__class__.__name__)
                pairs = [ds[i] for i in range(len(ds))]
                results.append((ds_name, idx, pairs))
        return results

    def on_fit_start(self) -> None:
        """Precompute geodesic caches for fmap validation pairs (if enabled)."""
        if not self._fmap_val_enabled():
            return

        import os
        import time

        cfg = self._fmap_val_cfg
        is_rank0 = self.trainer.is_global_zero
        fmap_datasets = self._get_fmap_val_pairs_from_dm()

        if not fmap_datasets:
            return

        all_pairs = [(name, p) for name, _, pairs in fmap_datasets for p in pairs]
        mv = cfg.get('max_vertices_val', 0)
        geo_cache_dir = cfg.get('geo_cache_dir', None)

        if geo_cache_dir:
            # Disk cache mode: lazy loading
            n_found = sum(
                1 for _, p in all_pairs
                if _fmap_geo_cache_path(geo_cache_dir, p.name, mv).exists()
            )
            if is_rank0:
                print(f"\n  [fmap val] Using disk geo cache: {n_found}/{len(all_pairs)} "
                      f"pairs found in {geo_cache_dir}", flush=True)
        else:
            # In-memory precomputation
            if is_rank0:
                print(f"\n  [fmap val] Precomputing geodesic caches "
                      f"({len(all_pairs)} pairs)...", flush=True)

            from fmaps_finetune.datasets.functional_map_dataset import subsample_pair, _stable_hash

            t0 = time.perf_counter()
            worker_args = []
            for _, p in all_pairs:
                if p.name in self._fmap_val_geo_cache:
                    continue
                sub_p = p
                if mv > 0:
                    sub_p = subsample_pair(p, mv, np.random.RandomState(_stable_hash(p.name)))
                if sub_p.faces_b is None:
                    continue
                gt_corr = _fmap_build_gt_corr(sub_p)
                verts_full = sub_p._verts_b_full if sub_p._verts_b_full is not None else sub_p.verts_b
                idx_b = sub_p._idx_b if sub_p._idx_b is not None else np.arange(len(sub_p.verts_b))
                unique_targets = np.unique(gt_corr) if gt_corr is not None else np.arange(len(sub_p.verts_b))
                worker_args.append((p.name, verts_full, sub_p.faces_b, idx_b, unique_targets))

            _n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
            max_workers = cfg.get('geo_cache_workers', None)
            n_workers = min(len(worker_args), max_workers or _n_cpus)

            if worker_args:
                if n_workers > 1:
                    import multiprocessing as _mp
                    if is_rank0:
                        print(f"    Parallel precomputation: {len(worker_args)} pairs, "
                              f"{n_workers} workers", flush=True)
                    with _mp.Pool(n_workers) as pool:
                        results = {}
                        for r in pool.imap_unordered(_fmap_precompute_geo_cache_worker, worker_args):
                            results[r[0]] = r[1:]
                else:
                    results = {}
                    for args in worker_args:
                        r = _fmap_precompute_geo_cache_worker(args)
                        results[r[0]] = r[1:]

                for name, (dist_cache, sqrt_area, idx_b) in results.items():
                    if dist_cache is not None:
                        self._fmap_val_geo_cache[name] = _FmapGeodesicCache.from_precomputed(
                            dist_cache, sqrt_area, idx_b)

            dt = time.perf_counter() - t0
            if is_rank0:
                print(f"    Done: {len(self._fmap_val_geo_cache)} caches in {dt:.1f}s",
                      flush=True)

    def _get_fmap_geo_cache(self, pair_name: str) -> Optional[_FmapGeodesicCache]:
        """Get geo cache for a pair: from memory or disk."""
        cfg = self._fmap_val_cfg
        geo_cache_dir = cfg.get('geo_cache_dir', None) if cfg else None

        if not geo_cache_dir:
            return self._fmap_val_geo_cache.get(pair_name)

        mv = cfg.get('max_vertices_val', 0)
        cache_path = _fmap_geo_cache_path(geo_cache_dir, pair_name, mv)
        if cache_path.exists():
            return _FmapGeodesicCache.load_from_disk(str(cache_path))
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        """Reset fmap validation output buffers."""
        self._fmap_val_outputs = {}

    def validation_step(self, batch, batch_idx: int,
                        dataloader_idx: int = 0) -> Optional[Dict[str, float]]:
        """Validate against GT eigendecomposition/geodesics (mesh batches),
        functional map correspondence (pair batches), or per-patch MCV
        on synthetic patches (synthetic batches)."""
        # Detect batch type: PyG Batch (mesh OR synthetic patch) vs list of PairSample
        if isinstance(batch, Batch):
            # Synthetic-patch validation: per-element analytic H and normal
            # attributes survive PyG concatenation as batch.H / batch.normal.
            # Mesh validation: items carry .raw_vertices and a .gt_eigen pair.
            has_curvature = getattr(batch, 'H', None) is not None
            has_normal = getattr(batch, 'normal', None) is not None
            has_mesh_vertices = getattr(batch, 'raw_vertices', None) is not None
            if has_curvature and has_normal and not has_mesh_vertices:
                return self._validation_step_patch_mcv(batch, batch_idx, dataloader_idx)
            return self._validation_step_mesh(batch, batch_idx, dataloader_idx)
        elif isinstance(batch, list):
            return self._validation_step_fmap(batch, batch_idx, dataloader_idx)
        else:
            return None

    def _validation_step_mesh(self, batch: Batch, batch_idx: int,
                              dataloader_idx: int) -> Dict[str, float]:
        """Original mesh validation: eigendecomposition + geodesics."""
        mesh_list = batch.to_data_list()
        all_metrics = []
        for mesh_data in mesh_list:
            all_metrics.append(self._validate_single_mesh(mesh_data))

        averaged_metrics = {}
        if all_metrics:
            for name in all_metrics[0].keys():
                values = [m[name] for m in all_metrics if name in m]
                if values:
                    averaged_metrics[name] = sum(values) / len(values)

        for name, val in averaged_metrics.items():
            self.log(f'val/{name}', val, on_step=False, on_epoch=True,
                     logger=True, batch_size=len(mesh_list), sync_dist=True)
        return averaged_metrics

    def _validation_step_patch_mcv(self, batch: Batch, batch_idx: int,
                                    dataloader_idx: int) -> Dict[str, float]:
        """Validate per-patch MCV against the analytic ``2 H n̂`` target.

        Used by both the patch-level and variational pipelines: the model
        produces ``predicted_mcv = (Σ_j s_ij p_j) / A_i`` via
        ``_compute_mean_curvature_vectors``; we compare it to the GT
        ``2 H n̂`` carried on the synthetic patch and log direction
        (cosine), magnitude ratio, and log-magnitude error.

        These are validation metrics only — the variational training
        loss does not depend on them, so they give a head-to-head MCV
        quality comparison vs. the patch-level MCV-trained baseline.
        """
        forward_result = self.forward(batch)
        predicted_mcv, _predicted_raw_mcv = self._compute_mean_curvature_vectors(
            forward_result, batch)

        normals = F.normalize(batch.normal, p=2, dim=1)            # (B, 3)
        target_mcv = 2.0 * batch.H.unsqueeze(-1) * normals         # (B, 3)

        eps = 1e-8
        pred_norm = predicted_mcv.norm(dim=-1).clamp(min=eps)      # (B,)
        target_norm = target_mcv.norm(dim=-1).clamp(min=eps)
        pred_unit = predicted_mcv / pred_norm.unsqueeze(-1)
        target_unit = target_mcv / target_norm.unsqueeze(-1)

        cosine = (pred_unit * target_unit).sum(dim=-1)             # (B,) in [-1, 1]
        magnitude_ratio = pred_norm / target_norm                  # (B,)
        log_mag_err = (pred_norm.log() - target_norm.log()) ** 2   # (B,)

        metrics = {
            'mcv_cosine_similarity': cosine.mean().item(),
            'mcv_magnitude_ratio': magnitude_ratio.mean().item(),
            'mcv_log_magnitude_error': log_mag_err.mean().item(),
        }
        for name, val in metrics.items():
            self.log(f'val/{name}', val, on_step=False, on_epoch=True,
                     logger=True, batch_size=int(batch.num_graphs)
                     if hasattr(batch, 'num_graphs') else len(batch),
                     sync_dist=True)
        return metrics

    def _validate_single_mesh(self, mesh_data: BaseData) -> Dict[str, float]:
        """Validate a single mesh: eigendecomposition + geodesics."""
        device = next(self.parameters()).device
        vertices = mesh_data.raw_vertices
        k = int(mesh_data.k_neighbors) if hasattr(mesh_data, 'k_neighbors') else 20

        patch_data = build_patches_from_vertices(vertices, k, device=device)
        mesh_batch = Batch.from_data_list([patch_data]).to(device)
        forward_result = self.forward(mesh_batch)

        batch_sizes = forward_result['batch_sizes']
        N = len(batch_sizes)
        k_val = batch_sizes[0].item()
        knn = mesh_batch.vertex_indices.reshape(N, k_val).to(device)
        areas = forward_result['areas'].detach()

        # Eigendecomposition: uses config.area_weighted to decide
        with torch.no_grad():
            L = assemble_laplacian(forward_result['grad_coeffs'], knn,
                                   self._val_lap_config, areas=areas)

        stiffness_matrix = to_scipy_sparse(L)
        mass_matrix = mass_matrix_to_scipy(areas)
        self._last_stiffness_matrix = stiffness_matrix
        self._last_mass_matrix = mass_matrix

        pred_evals, pred_evecs = _eigh_full_gram(L, areas, self._num_eigenvalues)
        gt_evals, gt_evecs = mesh_data.gt_eigen

        metrics = self._compute_spectral_comparison_metrics(
            pred_evals, pred_evecs, gt_evals, gt_evecs)

        metrics.update(self._compute_geodesic_validation_metrics(
            mesh_data, stiffness_matrix, mass_matrix, forward_result, mesh_batch))
        return metrics

    def _compute_geodesic_validation_metrics(
            self,
            mesh_data: BaseData,
            stiffness_matrix: scipy.sparse.spmatrix,
            mass_matrix: scipy.sparse.spmatrix,
            forward_result: Optional[Dict[str, torch.Tensor]] = None,
            mesh_batch: Optional[Batch] = None,
    ) -> Dict[str, float]:
        """Compute geodesic validation metrics using the Heat Method."""
        geodesic_data = getattr(mesh_data, 'geodesic_data', None)
        if geodesic_data is None or not geodesic_data.get('has_geodesic_data', False):
            return {}

        try:
            source_indices = geodesic_data['source_indices']
            exact_geodesics_dict = geodesic_data['exact_geodesics']
            n_vertices = stiffness_matrix.shape[0]

            batch_indices = getattr(mesh_batch, 'patch_idx', mesh_batch.batch)
            gradient_operator = assemble_gradient_operator(
                grad_coeffs=forward_result['grad_coeffs'],
                attention_mask=forward_result['attention_mask'],
                vertex_indices=mesh_batch.vertex_indices,
                center_indices=mesh_batch.center_indices,
                batch_indices=batch_indices,
            )

            def compute_pred(source_idx):
                return compute_heat_geodesic_learned(
                    S=stiffness_matrix, M=mass_matrix, G=gradient_operator,
                    source_idx=source_idx, n_vertices=n_vertices)

            def get_exact(source_idx):
                return exact_geodesics_dict.get(source_idx, None)

            return compute_multisource_geodesic_metrics(
                computed_func=compute_pred, exact_func=get_exact,
                source_indices=source_indices,
            ).to_dict(prefix="")

        except Exception:
            return {}

    def _compute_spectral_comparison_metrics(
            self,
            pred_eigenvalues: np.ndarray,
            pred_eigenvectors: np.ndarray,
            gt_eigenvalues: np.ndarray,
            gt_eigenvectors: np.ndarray,
    ) -> Dict[str, float]:
        """Compare predicted vs ground-truth eigendecomposition."""
        k = min(len(pred_eigenvalues), len(gt_eigenvalues))
        pred_eig = pred_eigenvalues[:k]
        gt_eig = gt_eigenvalues[:k]
        pred_vec = pred_eigenvectors[:, :k]
        gt_vec = gt_eigenvectors[:, :k]

        metrics = {}
        eps = 1e-6

        if k > 1:
            rel_errors_sq = ((pred_eig[1:] - gt_eig[1:]) / (gt_eig[1:] + eps)) ** 2
            metrics['eigenvalue_rel_mse'] = float(rel_errors_sq.mean())

        pred_gap = pred_eig[1] - pred_eig[0] if k > 1 else 0.0
        gt_gap = gt_eig[1] - gt_eig[0] if k > 1 else 1.0
        metrics['spectral_gap_ratio'] = float(pred_gap / (gt_gap + eps))

        if k > 2:
            corr = np.corrcoef(pred_eig, gt_eig)[0, 1]
            metrics['eigenvalue_correlation'] = float(corr) if not np.isnan(corr) else 0.0

        if k > 1:
            metrics['lambda1_ratio'] = float(pred_eig[1] / (gt_eig[1] + eps))

        cos_sims = []
        for i in range(k):
            pv = pred_vec[:, i] / (np.linalg.norm(pred_vec[:, i]) + 1e-8)
            gv = gt_vec[:, i] / (np.linalg.norm(gt_vec[:, i]) + 1e-8)
            cos_sims.append(np.abs(np.dot(pv, gv)))
        cos_sims = np.array(cos_sims)

        metrics['eigenvector_similarity_mean'] = float(cos_sims.mean())
        if k > 1:
            metrics['eigenvector_similarity_mean_skip0'] = float(cos_sims[1:].mean())
        for i in range(k):
            metrics[f'eigenvector_{i}_similarity'] = float(cos_sims[i])

        eig_err = float(np.mean(((pred_eig - gt_eig) / (gt_eig + eps)) ** 2)) if k > 0 else 0.0
        metrics['spectral_distance'] = eig_err + (1.0 - float(cos_sims.mean()))

        return metrics

    # ------------------------------------------------------------------
    # Fmap pair validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validation_step_fmap(self, batch: list, batch_idx: int,
                              dataloader_idx: int) -> None:
        """Run fmap correspondence evaluation on a shape pair batch."""
        if not self._fmap_val_enabled():
            return None

        cfg = self._fmap_val_cfg
        k = cfg.get('k', 20)
        num_eigenvectors = cfg.get('num_eigenvectors', 50)
        mv = cfg.get('max_vertices_val', 0)

        from fmaps_finetune.datasets.functional_map_dataset import (
            PairSample, subsample_pair, _stable_hash,
        )

        # [FMAP-DIAG] Log batch arrival per rank — helps detect if some
        # ranks are receiving empty or non-PairSample batches.
        n_pair = sum(1 for p in batch if isinstance(p, PairSample))
        n_other = len(batch) - n_pair
        print(
            f"[FMAP-DIAG] rank={self.global_rank} epoch={self.current_epoch} "
            f"step_fmap batch_idx={batch_idx} dl_idx={dataloader_idx} "
            f"batch_size={len(batch)} n_pairs={n_pair} n_skipped={n_other}",
            flush=True,
        )

        for pair_idx, pair in enumerate(batch):
            if not isinstance(pair, PairSample):
                # Non-PairSample item — record what type it is so we know
                # if filtering is the source of skew.
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dataloader_idx} "
                    f"SKIP non-PairSample at pair_idx={pair_idx}: "
                    f"type={type(pair).__name__}",
                    flush=True,
                )
                continue

            pair_name = getattr(pair, 'name', f'<unnamed_{pair_idx}>')

            if mv > 0:
                pair = subsample_pair(
                    pair, mv, np.random.RandomState(_stable_hash(pair.name)))

            # [FMAP-DIAG] Wrap evaluate_pair in try/except so failures become
            # loud (rather than silently dropping the pair from outputs and
            # creating per-rank count divergence).
            try:
                metrics = _fmap_evaluate_pair(
                    self, pair, k, num_eigenvectors, self.device,
                    laplacian_configs=self._fmap_val_eval_lap_configs,
                    evaluators=self._fmap_val_evaluators,
                    geo_cache=self._get_fmap_geo_cache(pair.name),
                    verbose_timing=(self.global_rank == 0),
                )
            except Exception as e:
                # Critical: do NOT swallow silently. Print loudly with full
                # context so the rank-specific failure is visible.
                import traceback
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dataloader_idx} "
                    f"FAILED pair={pair_name}: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                traceback.print_exc()
                # Re-raise so training fails fast rather than hangs later.
                # If you'd rather skip the pair, replace this with `continue`,
                # but be aware that creates per-rank count skew → deadlock.
                raise

            # [FMAP-DIAG] Inspect metrics dict shape — float_keys mismatch
            # across ranks would cause shape mismatch in all_gather even
            # with matched per-rank counts.
            float_keys = sorted(
                kk for kk, vv in metrics.items()
                if isinstance(vv, (int, float)) and not kk.startswith("_")
            )
            print(
                f"[FMAP-DIAG] rank={self.global_rank} "
                f"epoch={self.current_epoch} dl_idx={dataloader_idx} "
                f"OK pair={pair_name} n_keys={len(float_keys)} "
                f"keys_hash={hash(tuple(float_keys)) & 0xFFFFFFFF:08x}",
                flush=True,
            )

            if dataloader_idx not in self._fmap_val_outputs:
                self._fmap_val_outputs[dataloader_idx] = []
            self._fmap_val_outputs[dataloader_idx].append(metrics)

    def on_validation_epoch_end(self) -> None:
        """Aggregate and log fmap pair validation metrics across DDP ranks.

        DDP rank-skew safety:
            With distributed validation samplers, ranks may have different
            numbers of fmap pairs in ``self._fmap_val_outputs[dl_idx]`` —
            including some ranks with zero pairs. A naive ``self.all_gather``
            on the per-rank tensor would deadlock because tensor shapes
            differ across ranks.

            This implementation:
              1. Iterates dl_idx values derived from val_specs (identical
                 on every rank), not from per-rank ``_fmap_val_outputs``.
              2. Uses all_reduce to learn the global max per-rank count
                 and the lowest rank that holds data (for key broadcast).
              3. Broadcasts ``float_keys`` from a data-owning rank so
                 empty-output ranks know the column layout.
              4. Pads every rank's local tensor to ``(max_n, K)`` before
                 all_gather, then trims using per-rank counts.

            All ranks therefore make matching collective calls in matching
            order, satisfying NCCL's shape-agreement requirement.
        """
        if not self._fmap_val_enabled():
            return

        dm = self.trainer.datamodule
        val_specs = (dm._val_dataset_specifications
                     if hasattr(dm, '_val_dataset_specifications') else [])

        # Globally-known list of fmap dl_indices — all ranks see the same
        # val_specs, so this is identical across ranks and safe to iterate.
        fmap_dl_indices = sorted(
            idx for _, idx, _ in self._get_fmap_val_pairs_from_dm()
        )

        if not fmap_dl_indices:
            return

        world_size = self.trainer.world_size
        is_distributed = (
            world_size > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

        # [FMAP-DIAG] Snapshot of per-rank state at function entry.
        # If different ranks print different counts here, that's the
        # signature of the rank-skew bug (which the rest of this function
        # now handles correctly via padding).
        local_state = {
            int(idx): len(self._fmap_val_outputs.get(idx, []))
            for idx in fmap_dl_indices
        }
        print(
            f"[FMAP-DIAG] rank={self.global_rank} "
            f"epoch={self.current_epoch} "
            f"on_validation_epoch_end ENTER "
            f"world_size={world_size} dl_indices={fmap_dl_indices} "
            f"local_counts_per_dl_idx={local_state}",
            flush=True,
        )

        for dl_idx in fmap_dl_indices:
            outputs = self._fmap_val_outputs.get(dl_idx, [])
            local_n = len(outputs)

            print(
                f"[FMAP-DIAG] rank={self.global_rank} "
                f"epoch={self.current_epoch} dl_idx={dl_idx} "
                f"loop_iter local_n={local_n}",
                flush=True,
            )

            # Determine max count across ranks (so we know how to pad).
            if is_distributed:
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dl_idx} "
                    f"-> all_reduce(MAX) on local_n={local_n}",
                    flush=True,
                )
                local_n_t = torch.tensor(
                    [local_n], device=self.device, dtype=torch.long)
                max_n_t = local_n_t.clone()
                torch.distributed.all_reduce(
                    max_n_t, op=torch.distributed.ReduceOp.MAX)
                max_n = int(max_n_t.item())
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dl_idx} "
                    f"<- all_reduce(MAX) max_n={max_n}",
                    flush=True,
                )
            else:
                max_n = local_n

            if max_n == 0:
                # No rank produced any output for this dl_idx — nothing to do.
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dl_idx} "
                    f"SKIP (max_n=0 across all ranks)",
                    flush=True,
                )
                continue

            # Determine the lowest rank that owns data (for key broadcast).
            # Ranks with no data contribute world_size (i.e. effectively +inf
            # for MIN reduction); ranks with data contribute their rank id.
            if is_distributed:
                rank_with_data = (self.global_rank
                                  if local_n > 0 else world_size)
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dl_idx} "
                    f"-> all_reduce(MIN) rank_with_data={rank_with_data}",
                    flush=True,
                )
                rwd_t = torch.tensor(
                    [rank_with_data], device=self.device, dtype=torch.long)
                torch.distributed.all_reduce(
                    rwd_t, op=torch.distributed.ReduceOp.MIN)
                src_rank = int(rwd_t.item())
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dl_idx} "
                    f"<- all_reduce(MIN) src_rank={src_rank}",
                    flush=True,
                )
            else:
                src_rank = 0

            # Compute float_keys on the source rank, then broadcast to others.
            # broadcast_object_list works with picklable Python objects (here
            # a list of strings) and is safe to call on every rank.
            if local_n > 0 and self.global_rank == src_rank:
                float_keys = sorted(
                    k for k, v in outputs[0].items()
                    if isinstance(v, (int, float)) and not k.startswith("_")
                )
            else:
                float_keys = None

            if is_distributed:
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dl_idx} "
                    f"-> broadcast_object_list(src={src_rank}) "
                    f"local_keys={None if float_keys is None else len(float_keys)}",
                    flush=True,
                )
                obj_list = [float_keys]
                torch.distributed.broadcast_object_list(obj_list, src=src_rank)
                float_keys = obj_list[0]
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dl_idx} "
                    f"<- broadcast_object_list "
                    f"received_keys={None if float_keys is None else len(float_keys)}",
                    flush=True,
                )

            if not float_keys:
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dl_idx} "
                    f"SKIP (no float_keys agreed)",
                    flush=True,
                )
                continue

            K = len(float_keys)

            # Build the padded local tensor of shape (max_n, K).
            # Real entries fill the first `local_n` rows; rest is NaN.
            local_t = torch.full(
                (max_n, K), float("nan"),
                device=self.device, dtype=torch.float32,
            )
            for i, d in enumerate(outputs):
                for j, k in enumerate(float_keys):
                    local_t[i, j] = float(d.get(k, float("nan")))

            # Also exchange real per-rank counts so we can trim padding.
            if is_distributed:
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dl_idx} "
                    f"-> all_gather(counts)",
                    flush=True,
                )
                local_count_t = torch.tensor(
                    [local_n], device=self.device, dtype=torch.long)
                count_list = [torch.zeros_like(local_count_t)
                              for _ in range(world_size)]
                torch.distributed.all_gather(count_list, local_count_t)
                per_rank_counts = [int(c.item()) for c in count_list]
                print(
                    f"[FMAP-DIAG] rank={self.global_rank} "
                    f"epoch={self.current_epoch} dl_idx={dl_idx} "
                    f"<- all_gather(counts) per_rank={per_rank_counts}",
                    flush=True,
                )
            else:
                per_rank_counts = [local_n]

            # All ranks now have local_t of shape (max_n, K) — collective is
            # shape-consistent across ranks.
            print(
                f"[FMAP-DIAG] rank={self.global_rank} "
                f"epoch={self.current_epoch} dl_idx={dl_idx} "
                f"-> all_gather(tensor) shape=({max_n},{K})",
                flush=True,
            )
            gathered = self.all_gather(local_t)
            print(
                f"[FMAP-DIAG] rank={self.global_rank} "
                f"epoch={self.current_epoch} dl_idx={dl_idx} "
                f"<- all_gather(tensor) gathered.shape={tuple(gathered.shape)}",
                flush=True,
            )
            # Lightning returns (W, max_n, K) under DDP, or (max_n, K) when
            # not distributed. Normalise to (W, max_n, K).
            if gathered.dim() == 2:
                gathered = gathered.unsqueeze(0)

            # Reassemble outputs in rank order, dropping per-rank padding.
            all_metrics: List[Dict[str, float]] = []
            for r, n_real in enumerate(per_rank_counts):
                for i in range(n_real):
                    all_metrics.append({
                        k: gathered[r, i, j].item()
                        for j, k in enumerate(float_keys)
                    })

            # Trim to the true dataset size (the fmap dataset may have been
            # padded by the DistributedSampler so world_size divides cleanly).
            ds = val_specs[dl_idx].dataset if dl_idx < len(val_specs) else None
            ds_name = getattr(ds, 'name', f'fmap_val_{dl_idx}')
            true_size = (len(val_specs[dl_idx].dataset)
                         if dl_idx < len(val_specs) else len(all_metrics))
            all_metrics = all_metrics[:true_size]

            if not all_metrics:
                continue

            # Summarise and print (rank 0 only via the silent= flag).
            summary = _fmap_summarise(
                all_metrics, self._fmap_val_evaluators,
                f"Val epoch {self.current_epoch} [{ds_name}]",
                silent=not self.trainer.is_global_zero,
            )

            # Log to W&B / Lightning. sync_dist=True here means Lightning
            # itself does the cross-rank reduction of the scalar — every rank
            # must call self.log with the same key, which all do because we
            # iterate fmap_dl_indices globally and only skip when max_n == 0.
            prefix = f"val_fmap/{ds_name}"
            for mk, mv in summary.items():
                self.log(f"{prefix}/{mk}", mv, sync_dist=True,
                         on_step=False, on_epoch=True, add_dataloader_idx=False)

            # Log primary fmap metric to prog_bar.
            for ev in self._fmap_val_evaluators:
                geo_key = f"{ev.name}/geo_at_05pct"
                if geo_key in summary:
                    self.log(f"val_fmap/geo@5%", summary[geo_key],
                             prog_bar=True, sync_dist=True,
                             on_step=False, on_epoch=True, add_dataloader_idx=False)
                    break

        # [FMAP-DIAG] Loop over all dl_indices completed cleanly on this rank.
        # If you see ENTER prints from all ranks but EXIT prints from only some,
        # the deadlock is between those two points (likely a collective).
        print(
            f"[FMAP-DIAG] rank={self.global_rank} "
            f"epoch={self.current_epoch} "
            f"on_validation_epoch_end EXIT (all dl_indices processed)",
            flush=True,
        )

# ============================================================================
# Backward-compatible thin shims
# ============================================================================
#
# Existing Hydra configs reference ``LaplacianTransformerModule`` by its
# fully-qualified path. The class below preserves that entry point and the
# original constructor kwargs (``nhead``, ``num_encoder_layers``,
# ``dim_feedforward``), translating them into a ``TransformerPatchEncoder``.
#
# The new ``LaplacianDeepSetModule`` exposes an analogous interface for the
# DeepSet architecture.
#
# Both are ~15 lines each; all real logic lives in ``LaplacianLocalModule``.
# ============================================================================


class LaplacianTransformerModule(LaplacianLocalModule):
    """Transformer-based patch encoder. Preserves the original config API."""

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 **kwargs):
        encoder = TransformerPatchEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        super().__init__(patch_encoder=encoder, d_model=d_model,
                         dropout=dropout, **kwargs)


class LaplacianDeepSetModule(LaplacianLocalModule):
    """DeepSet-based patch encoder.

    Drop-in replacement for LaplacianTransformerModule. Config keys mirror
    the transformer variant where they make sense:

      - ``d_model``, ``num_encoder_layers``, ``dim_feedforward``, ``dropout``
        play the same roles as in the transformer.
      - ``pool`` selects the permutation-invariant aggregation (mean/sum/max).
        Defaults to 'mean', which is usually a good choice for small patches.
      - No ``nhead`` — there's no attention.

    Parameter count is roughly ``num_encoder_layers * 4 * d_model * dim_feedforward``
    (two MLPs per layer, each ~2*d_model*dff). At d_model=256, dff=512, L=4
    this is ~2.1M — comparable to the transformer at similar depth/width.
    """

    def __init__(self,
                 d_model: int = 256,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 pool: str = 'mean',
                 **kwargs):
        encoder = DeepSetPatchEncoder(
            d_model=d_model,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pool=pool,
        )
        super().__init__(patch_encoder=encoder, d_model=d_model,
                         dropout=dropout, **kwargs)