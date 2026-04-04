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


class LaplacianTransformerModule(LaplacianModuleBase):
    """Surface transformer module with support for variable-sized patches."""

    def __init__(self,
                 input_dim: Optional[int] = None,
                 loss_configs: Optional[List[LossConfig]] = None,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_eigenvalues: int = 10,
                 normalize_loss_weights: bool = True,
                 input_projection_hidden_dims: Optional[List[int]] = None,
                 output_projection_hidden_dims: Optional[List[int]] = None,
                 normalize_patch_features: bool = True,
                 scale_areas_by_patch_size: bool = True,
                 mcv_mode: str = 'diagonal_gram',
                 val_laplacian: Optional[Dict] = None,
                 **kwargs):
        # **kwargs absorbs legacy hparams (operator_mode, patch_mcv_mode,
        # val_laplacian_mode) from old checkpoints.
        super().__init__(**{k: v for k, v in kwargs.items()
                           if k in ('optimizer_cfg', 'scheduler_cfg')})

        self.save_hyperparameters(ignore=['loss_configs', 'feature_extractor'])

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
        self._mcv_mode = mcv_mode

        # Validation Laplacian config
        _val_lap = val_laplacian or {'assembly': 'diagonal_gram', 'pruning': 'none'}
        self._val_lap_config = LaplacianConfig(**_val_lap)

        # Store loss configs (optionally normalized)
        if normalize_loss_weights:
            self._loss_configs = self._normalize_loss_weights(loss_configs)
        else:
            self._loss_configs = loss_configs

        # Input and output projections
        self.input_projection = self._build_projection(resolved_input_dim, d_model, input_projection_hidden_dims)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers,
        )

        # Output head: gradient coefficients g_ij in R^3
        self.grad_projection = self._build_projection(d_model, 3, output_projection_hidden_dims)

        # Area head: aggregated features -> scalar area A_i
        self.area_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
                                        batch_data: Batch) -> torch.Tensor:
        """
        Compute predicted mean curvature vectors from gradient coefficients and areas.

        Mode determines how weights are computed:
        - 'diagonal_gram': w_ij = ||g_ij||^2  (always positive, isotropic)
        - 'full_gram':     w_ij = g_ii . g_ij  (can be negative, anisotropic)
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

        return stiffness_sum / areas.unsqueeze(-1)

    def _reshape_positions_to_batched(self, pos_flat: torch.Tensor,
                                      batch_sizes: torch.Tensor) -> torch.Tensor:
        """Reshape flat positions to padded (batch_size, max_k, 3)."""
        batch_size = len(batch_sizes)
        max_k = batch_sizes.max().item()
        if torch.all(batch_sizes == batch_sizes[0]):
            return pos_flat.view(batch_size, max_k, 3)
        out = torch.zeros(batch_size, max_k, 3, device=pos_flat.device,
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

        # ── Sequence padding + transformer ───────────────────────────
        if fixed_k:
            sequences = features.view(batch_size, max_k, -1)
            encoded = self.transformer_encoder(sequences,
                                               src_key_padding_mask=None)
            attention_mask = torch.ones(batch_size, max_k, dtype=torch.bool,
                                        device=features.device)
        else:
            sequences, attention_mask = self._pad_sequences_vectorized(
                features, batch_indices, batch_size, max_k)
            encoded = self.transformer_encoder(
                sequences, src_key_padding_mask=~attention_mask)

        # ── Output heads ─────────────────────────────────────────────
        grad_coeffs = self.grad_projection(encoded)
        stiffness_weights = (grad_coeffs ** 2).sum(dim=-1)

        # ── Area prediction ──────────────────────────────────────────
        if fixed_k:
            pooled = encoded.mean(dim=1)
        else:
            float_mask = attention_mask.float()
            num_tokens = float_mask.sum(dim=1, keepdim=True)
            pooled = (encoded * float_mask.unsqueeze(-1)).sum(dim=1) / num_tokens

        areas_normalized = self.area_head(pooled).squeeze(-1)
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
        predicted_mcv = self._compute_mean_curvature_vectors(forward_result, batch_data)

        batch_size = len(forward_result['batch_sizes'])
        normals = batch_data.normal
        mean_curvatures = batch_data.H
        target_mcv = 2.0 * mean_curvatures.unsqueeze(-1) * F.normalize(normals, p=2, dim=1)

        loss_context = LossContext(
            predicted_mcv=predicted_mcv,
            target_mcv=target_mcv,
            grad_coeffs=forward_result.get('grad_coeffs'),
            positions=(self._reshape_positions_to_batched(
                           batch_data.pos, forward_result['batch_sizes'])
                       if forward_result.get('grad_coeffs') is not None else None),
            normals=getattr(batch_data, 'normal', None),
            attention_mask=forward_result['attention_mask'],
            areas=forward_result['areas'],
            stiffness_weights=forward_result['stiffness_weights'],
        )

        total_loss = 0.0
        loss_components_weighted = {}
        loss_components_unweighted = {}

        for loss_config in self._loss_configs:
            unweighted_loss = loss_config.loss_module(loss_context)
            loss_name = loss_config.loss_module.__class__.__name__
            loss_components_unweighted[f"train/{loss_name}"] = unweighted_loss
            if loss_config.weight is not None:
                weighted_loss = loss_config.weight * unweighted_loss
                total_loss = total_loss + weighted_loss
                loss_components_weighted[f"train/{loss_name}_weighted"] = weighted_loss

        if not isinstance(total_loss, torch.Tensor):
            raise ValueError("At least one loss must have a non-None weight for training")

        cosine_sim = F.cosine_similarity(predicted_mcv, target_mcv, dim=1).mean()
        areas = forward_result['areas']

        self.log('train/loss', total_loss.item(), on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log('train/cosine_similarity', cosine_sim.item(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size,
                 sync_dist=True)
        self.log('train/area_mean', areas.mean().item(), on_step=False, on_epoch=True,
                 logger=True, batch_size=batch_size, sync_dist=True)
        self.log('train/area_std', areas.std().item(), on_step=False, on_epoch=True,
                 logger=True, batch_size=batch_size, sync_dist=True)

        for name, val in loss_components_unweighted.items():
            self.log(name, val, on_step=False, on_epoch=True, logger=True,
                     batch_size=batch_size, sync_dist=True)
        for name, val in loss_components_weighted.items():
            self.log(name, val, on_step=False, on_epoch=True, logger=True,
                     batch_size=batch_size, sync_dist=True)

        result = {"loss": total_loss}
        result.update(loss_components_weighted)
        result.update(loss_components_unweighted)
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, float]:
        """Validate against ground-truth eigendecomposition and geodesics."""
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

        with torch.no_grad():
            L = assemble_laplacian(forward_result['grad_coeffs'], knn,
                                   self._val_lap_config)

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