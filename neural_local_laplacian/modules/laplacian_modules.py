import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch, Data

import os
import pickle
import zipfile
import shutil
from pathlib import Path
from typing import List, Type, Callable, Optional, Dict, Tuple, Any
from dataclasses import dataclass

# wandb
import wandb

# omegaconf
from omegaconf import DictConfig, OmegaConf

# numpy
import numpy as np

# scipy
import scipy.sparse
import scipy.sparse.linalg

# lightning
import lightning
from pytorch_lightning.callbacks import Callback

# torch_geometric
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData

# trimesh for loading mesh vertices
import trimesh

# pyfm
from pyFM.mesh import TriMesh

# neural laplacian
from neural_local_laplacian.utils.utils import split_results_by_nodes, split_results_by_graphs, assemble_sparse_laplacian_variable, assemble_stiffness_and_mass_matrices, compute_laplacian_eigendecomposition
from neural_local_laplacian.modules.losses import LossConfig


class LaplacianModuleBase(lightning.pytorch.LightningModule):
    def __init__(self,
                 optimizer_cfg: DictConfig,
                 scheduler_cfg: Optional[DictConfig] = None,
                 **kwargs
                 ):
        super().__init__()
        self._optimizer_cfg = optimizer_cfg
        self._scheduler_cfg = scheduler_cfg

    def setup(self, stage):
        def exclude_fn(path: str):
            if 'lightning_logs' in path:
                return True
            if 'outputs' in path:
                return True
            if 'wandb' in path:
                return True
            if 'checkpoints' in path:
                return True
            if '.git' in path:
                return True
            if '.venvs' in path:
                return True

            return False

        def include_fn(path: str):
            return True if path.endswith('.py') or path.endswith('.yml') or path.endswith('.yaml') else False

        if self.trainer.global_rank == 0 and wandb.run is not None:
            self.logger.experiment.log_code(root=".", exclude_fn=exclude_fn, include_fn=include_fn)
            dict_cfg = OmegaConf.to_container(self.trainer.cfg, resolve=True)
            self.logger.experiment.config.update(dict_cfg)

    def configure_optimizers(self):
        """Configure optimizer and optionally scheduler."""
        if self._optimizer_cfg is None:
            raise ValueError("optimizer_cfg is required but was None")

        # Create optimizer
        optimizer = self._optimizer_cfg(params=self.parameters())

        # If no scheduler config, return just the optimizer
        if self._scheduler_cfg is None:
            return optimizer

        # Create scheduler
        scheduler = self._scheduler_cfg(optimizer=optimizer)

        # Return optimizer and scheduler configuration
        scheduler_config = {
            "scheduler": scheduler,
            "interval": 'epoch'
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }


class LaplacianTransformerModule(LaplacianModuleBase):
    """
    Surface transformer module with 3-head architecture for learning discrete Laplace-Beltrami operators.

    Architecture:
    - Normal weights head (w_ij): per-neighbor positive weights that produce unit normal after area division
    - Mean curvature head (H_i): per-patch scalar for mean curvature
    - Area head (A_i): per-patch positive scalar, implicitly supervised via normal loss

    Key relationships:
    - Normal prediction: sum_j w_ij * p_j / A_i = n_hat (unit normal)
    - Stiffness derivation: s_ij = 2 * H_i * w_ij
    - Laplacian: sum_j s_ij * p_j = 2 * A_i * H_i * n_hat

    This decouples normal prediction from curvature magnitude, solving the flat-patch problem
    where near-zero curvature caused unreliable normal predictions.
    """

    def __init__(self,
                 input_dim: int,
                 loss_configs: Optional[List[LossConfig]] = None,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_eigenvalues: int = 10,
                 normalize_loss_weights: bool = True,
                 input_projection_hidden_dims: Optional[List[int]] = None,
                 output_projection_hidden_dims: Optional[List[int]] = None,
                 **kwargs):
        super().__init__(**kwargs)

        # Save hyperparameters (exclude loss_configs - non-serializable)
        self.save_hyperparameters(ignore=['loss_configs'])

        # Manually save loss configuration info for logging
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

        # Validate input_dim
        if input_dim is None or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got: {input_dim}")

        self._d_model = d_model
        self._input_dim = input_dim
        self._num_eigenvalues = num_eigenvalues

        # Store loss configs (optionally normalized)
        if normalize_loss_weights and loss_configs is not None:
            self._loss_configs = self._normalize_loss_weights(loss_configs)
        else:
            self._loss_configs = loss_configs

        # Input projection
        self.input_projection = self._build_projection(input_dim, d_model, input_projection_hidden_dims)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # === HEAD 1: Normal weights (per-token) ===
        # Produces w_ij such that sum_j w_ij * p_j / A_i = n_hat
        self.normal_weights_projection = self._build_projection(d_model, 1, output_projection_hidden_dims)

        # === HEAD 2: Mean curvature (per-patch) ===
        # Produces H_i scalar, can be positive or negative
        self.mean_curvature_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # === HEAD 3: Area (per-patch) ===
        # Produces A_i > 0, implicitly supervised via normal loss
        self.area_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # Ensure positive area
        )

    @staticmethod
    def _build_projection(in_dim: int, out_dim: int, hidden_dims: Optional[List[int]] = None) -> nn.Module:
        """
        Build a projection module: either single linear or MLP with hidden layers.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden_dims: Optional list of hidden layer dimensions. If None or empty, uses single linear.

        Returns:
            nn.Linear or nn.Sequential module
        """
        if hidden_dims is None or len(hidden_dims) == 0:
            return nn.Linear(in_dim, out_dim)

        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)

    def _normalize_loss_weights(self, loss_configs: List[LossConfig]) -> List[LossConfig]:
        """
        Normalize loss weights so they sum to 1.

        Loss configs with weight=None are kept as-is (logged but not included in backprop).

        Args:
            loss_configs: List of LossConfig objects

        Returns:
            List of LossConfig objects with normalized weights (for non-None weights)
        """
        if not loss_configs:
            return loss_configs

        # Calculate total weight (only for non-None weights)
        total_weight = sum(config.weight for config in loss_configs if config.weight is not None)

        if total_weight == 0:
            raise ValueError("Total loss weights cannot be zero (at least one loss must have a non-None weight)")

        # Create new configs with normalized weights
        normalized_configs = []
        for config in loss_configs:
            if config.weight is None:
                normalized_configs.append(config)
            else:
                normalized_weight = config.weight / total_weight
                normalized_config = LossConfig(
                    loss_module=config.loss_module,
                    weight=normalized_weight
                )
                normalized_configs.append(normalized_config)

        return normalized_configs

    def _pad_sequences_vectorized(self, features: torch.Tensor, batch_indices: torch.Tensor,
                                  batch_size: int, max_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized padding of variable-length sequences to max_k length using scatter operations.

        Args:
            features: (total_points, d_model)
            batch_indices: (total_points,) - which batch each point belongs to
            batch_size: number of sequences
            max_k: maximum sequence length

        Returns:
            sequences: (batch_size, max_k, d_model) - padded sequences
            attention_mask: (batch_size, max_k) - True for real tokens, False for padding
        """
        device = features.device
        d_model = features.shape[1]

        # Sort indices to group by batch
        sorted_indices = torch.argsort(batch_indices)
        sorted_batch = batch_indices[sorted_indices]

        # Get batch sizes
        batch_sizes = torch.bincount(batch_indices, minlength=batch_size)

        # Calculate positions within each batch using fully vectorized operations
        positions = torch.zeros_like(batch_indices, dtype=torch.long)

        # Create cumulative positions for sorted indices
        cumsum_sizes = torch.cumsum(batch_sizes, dim=0)
        starts = torch.cat([torch.tensor([0], device=device), cumsum_sizes[:-1]])

        # Create position indices for each batch
        total_points = batch_indices.shape[0]
        arange_full = torch.arange(total_points, device=device)

        # Calculate relative positions within each batch
        batch_starts = starts[sorted_batch]
        relative_positions = arange_full[sorted_indices] - batch_starts

        # Assign positions back to original order
        positions[sorted_indices] = relative_positions

        # Filter out positions >= max_k (in case some patches are larger than max_k)
        valid_mask = positions < max_k
        valid_batch_indices = batch_indices[valid_mask]
        valid_positions = positions[valid_mask]
        valid_features = features[valid_mask]

        # Create flat indices for scatter
        flat_indices = valid_batch_indices * max_k + valid_positions

        # Create output tensors
        sequences = torch.zeros(batch_size * max_k, d_model, device=device, dtype=features.dtype)
        attention_mask = torch.zeros(batch_size * max_k, dtype=torch.bool, device=device)

        # Scatter features and create mask
        sequences.scatter_(0, flat_indices.unsqueeze(1).expand(-1, d_model), valid_features)
        attention_mask.scatter_(0, flat_indices, True)

        # Reshape to (batch_size, max_k, d_model) and (batch_size, max_k)
        sequences = sequences.view(batch_size, max_k, d_model)
        attention_mask = attention_mask.view(batch_size, max_k)

        return sequences, attention_mask

    def _compute_weighted_position_sum(self, normal_weights: torch.Tensor,
                                       attention_mask: torch.Tensor,
                                       batch_sizes: torch.Tensor,
                                       positions: torch.Tensor) -> torch.Tensor:
        """
        Compute sum_j w_ij * p_j for each patch using vectorized operations.

        Args:
            normal_weights: (batch_size, max_k) - weights per neighbor
            attention_mask: (batch_size, max_k) - True for real tokens
            batch_sizes: (batch_size,) - actual number of points per patch
            positions: (total_points, 3) - flattened positions

        Returns:
            weighted_sum: (batch_size, 3) - sum_j w_ij * p_j for each patch
        """
        batch_size = normal_weights.shape[0]
        max_k = normal_weights.shape[1]
        device = normal_weights.device

        # Apply attention mask to weights (zero out padded positions)
        masked_weights = normal_weights.masked_fill(~attention_mask, 0.0)  # (batch_size, max_k)

        # Flatten masked weights
        weights_flat = masked_weights.flatten()  # (batch_size * max_k,)

        # Create batch indices for flattened weights
        batch_indices_weights = torch.arange(batch_size, device=device).repeat_interleave(max_k)

        # Create position indices within each batch
        batch_cumsum = torch.cumsum(batch_sizes, dim=0)
        batch_starts = torch.cat([torch.zeros(1, device=device, dtype=batch_cumsum.dtype), batch_cumsum[:-1]])

        # Position indices for each weight
        position_indices = torch.arange(max_k, device=device).repeat(batch_size)

        # Filter out indices that exceed actual batch sizes
        valid_mask = position_indices < batch_sizes.repeat_interleave(max_k)

        # Get valid weights and their corresponding batch indices
        valid_weights = weights_flat[valid_mask]  # (num_valid,)
        valid_batch_indices = batch_indices_weights[valid_mask]  # (num_valid,)
        valid_position_indices = position_indices[valid_mask]  # (num_valid,)

        # Calculate actual position indices in the flattened positions array
        actual_position_indices = batch_starts[valid_batch_indices] + valid_position_indices

        # Get valid positions
        valid_positions = positions[actual_position_indices.long()]  # (num_valid, 3)

        # Compute weighted positions: w_ij * p_j
        weighted_positions = valid_weights.unsqueeze(-1) * valid_positions  # (num_valid, 3)

        # Sum weighted positions for each batch using scatter_add
        weighted_sum = torch.zeros(batch_size, 3, device=device, dtype=positions.dtype)
        weighted_sum.scatter_add_(0,
                                  valid_batch_indices.unsqueeze(-1).expand(-1, 3),
                                  weighted_positions)

        return weighted_sum

    def _forward_pass(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Internal forward pass method that can be called from both forward() and validation_step().

        Args:
            batch: PyTorch Geometric batch

        Returns:
            Dictionary containing forward pass results
        """
        return self.forward(batch)

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass with 3-head architecture.

        Predicts:
        - Normal weights w_ij (per neighbor, positive)
        - Mean curvature H_i (per patch, scalar)
        - Area A_i (per patch, positive scalar)

        Args:
            batch: PyTorch Geometric batch

        Returns:
            Dict containing:
            - normal_weights: (batch_size, max_k) - learned normal weights w_ij (padded)
            - mean_curvatures: (batch_size,) - learned mean curvature H_i for each patch
            - areas: (batch_size,) - learned area A_i for each patch center
            - attention_mask: (batch_size, max_k) - True for real tokens, False for padding
            - batch_sizes: (batch_size,) - actual number of points per patch
        """
        features = batch.x  # Shape: (total_points, feature_dim)

        # Use patch_idx if available (MeshDataset), otherwise use batch (synthetic)
        batch_indices = getattr(batch, 'patch_idx', batch.batch)

        # Get batch sizes (number of points per patch)
        batch_sizes = batch_indices.bincount()
        batch_size = len(batch_sizes)
        max_k = batch_sizes.max().item()

        # Project to model dimension
        features = self.input_projection(features)  # (total_points, d_model)

        # Pad sequences to max_k and create attention masks
        sequences, attention_mask = self._pad_sequences_vectorized(
            features, batch_indices, batch_size, max_k
        )

        # Pass through transformer with attention mask
        # Note: src_key_padding_mask expects True for positions to IGNORE
        encoded_features = self.transformer_encoder(sequences, src_key_padding_mask=~attention_mask)

        # === HEAD 1: Normal weights (per token) ===
        normal_weights = self.normal_weights_projection(encoded_features)  # (batch_size, max_k, 1)
        normal_weights = normal_weights.squeeze(-1)  # (batch_size, max_k)

        # Apply exp to ensure positive weights
        normal_weights = torch.exp(normal_weights)

        # Mask out padded positions
        normal_weights = normal_weights.masked_fill(~attention_mask, 0.0)

        # === Pooled features for per-patch heads ===
        float_mask = attention_mask.float()  # (batch_size, max_k)
        num_tokens = float_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)

        # Masked mean: sum of (features * mask) / num_tokens
        masked_features = encoded_features * float_mask.unsqueeze(-1)  # (batch_size, max_k, d_model)
        pooled_features = masked_features.sum(dim=1) / num_tokens  # (batch_size, d_model)

        # === HEAD 2: Mean curvature (per patch) ===
        mean_curvatures = self.mean_curvature_head(pooled_features).squeeze(-1)  # (batch_size,)

        # === HEAD 3: Area (per patch) ===
        areas = self.area_head(pooled_features).squeeze(-1)  # (batch_size,)

        return {
            'normal_weights': normal_weights,
            'mean_curvatures': mean_curvatures,
            'areas': areas,
            'attention_mask': attention_mask,
            'batch_sizes': batch_sizes
        }

    def compute_predictions(self, forward_result: Dict[str, torch.Tensor],
                            batch_data: Batch) -> Dict[str, torch.Tensor]:
        """
        Compute derived predictions from forward pass results.

        Args:
            forward_result: Output from forward()
            batch_data: Original batch data with positions

        Returns:
            Dict containing:
            - predicted_normals: (batch_size, 3) - unit normal predictions
            - predicted_mean_curvature_vectors: (batch_size, 3) - H * n_hat
            - stiffness_weights: (batch_size, max_k) - derived s_ij = 2 * H_i * w_ij
        """
        normal_weights = forward_result['normal_weights']
        mean_curvatures = forward_result['mean_curvatures']
        areas = forward_result['areas']
        attention_mask = forward_result['attention_mask']
        batch_sizes = forward_result['batch_sizes']

        positions = batch_data.pos  # (total_points, 3)

        # Compute sum_j w_ij * p_j
        weighted_sum = self._compute_weighted_position_sum(
            normal_weights, attention_mask, batch_sizes, positions
        )  # (batch_size, 3)

        # Predicted normal: (sum_j w_ij * p_j) / A_i, then normalize to unit
        # The division by area scales the weighted sum to unit normal
        raw_normal = weighted_sum / areas.unsqueeze(-1)  # (batch_size, 3)
        predicted_normals = F.normalize(raw_normal, p=2, dim=1)  # (batch_size, 3)

        # Mean curvature vector: H_i * n_hat
        predicted_mean_curvature_vectors = mean_curvatures.unsqueeze(-1) * predicted_normals  # (batch_size, 3)

        # Derive stiffness weights: s_ij = 2 * H_i * w_ij
        # This is used for validation (eigendecomposition)
        stiffness_weights = 2.0 * mean_curvatures.unsqueeze(-1) * normal_weights  # (batch_size, max_k)

        return {
            'predicted_normals': predicted_normals,
            'predicted_mean_curvature_vectors': predicted_mean_curvature_vectors,
            'stiffness_weights': stiffness_weights,
            'weighted_sum': weighted_sum,
            'raw_normal': raw_normal
        }

    def training_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step with 3-head architecture.

        Losses:
        1. Normal direction loss: (sum_j w_ij * p_j) / A_i vs n_hat_GT
        2. Mean curvature loss: H_pred vs H_GT

        Args:
            batch: List of PyTorch Geometric batches (synthetic data)
            batch_idx: Batch index

        Returns:
            Dictionary with loss and other metrics
        """
        # Take the first batch from the list
        batch_data = batch[0]
        forward_result = self.forward(batch_data)
        predictions = self.compute_predictions(forward_result, batch_data)

        # Get batch size for logging
        batch_size = len(forward_result['batch_sizes'])

        # In training mode, diff_geom_at_origin_only=True, so normals and H are per-surface
        gt_normals = batch_data.normal  # (batch_size, 3) - one normal per surface at origin
        gt_mean_curvatures = batch_data.H  # (batch_size,) - one curvature per surface at origin

        # Normalize GT normals (should already be unit, but ensure)
        gt_normals = F.normalize(gt_normals, p=2, dim=1)

        # === Loss 1: Normal direction ===
        # Target: unit normal direction
        # Prediction: (sum_j w_ij * p_j) / A_i (should be unit normal)
        predicted_normals = predictions['predicted_normals']

        # Cosine similarity loss for direction (1 - cos_sim)
        cos_sim = F.cosine_similarity(predicted_normals, gt_normals, dim=1)
        normal_direction_loss = (1 - cos_sim).mean()

        # === Loss 2: Mean curvature magnitude ===
        predicted_H = forward_result['mean_curvatures']
        curvature_loss = F.mse_loss(predicted_H, gt_mean_curvatures)

        # === Combined loss ===
        # Use loss_configs if available, otherwise default weighting
        if self._loss_configs:
            total_loss = 0.0
            loss_components_weighted = {}
            loss_components_unweighted = {}

            # Compute target mean curvature vector for compatibility with existing losses
            target_mcv = gt_mean_curvatures.unsqueeze(-1) * gt_normals
            predicted_mcv = predictions['predicted_mean_curvature_vectors']

            for loss_config in self._loss_configs:
                unweighted_loss = loss_config.loss_module(predicted_mcv, target_mcv)
                loss_name = f"{loss_config.loss_module.__class__.__name__}"
                loss_components_unweighted[f"train/{loss_name}"] = unweighted_loss

                if loss_config.weight is not None:
                    weighted_loss = loss_config.weight * unweighted_loss
                    total_loss = total_loss + weighted_loss
                    loss_components_weighted[f"train/{loss_name}_weighted"] = weighted_loss

            if not isinstance(total_loss, torch.Tensor):
                raise ValueError("At least one loss must have a non-None weight for training")
        else:
            # Default: equal weight for normal and curvature losses
            total_loss = 0.5 * normal_direction_loss + 0.5 * curvature_loss
            loss_components_unweighted = {}
            loss_components_weighted = {}

        # === Logging ===
        self.log('train/loss', total_loss.item(), on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size, sync_dist=True)

        # Normal direction metrics
        self.log('train/normal_cosine_similarity', cos_sim.mean().item(), on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log('train/normal_direction_loss', normal_direction_loss.item(), on_step=False, on_epoch=True,
                 logger=True, batch_size=batch_size, sync_dist=True)

        # Curvature metrics
        self.log('train/curvature_loss', curvature_loss.item(), on_step=False, on_epoch=True,
                 logger=True, batch_size=batch_size, sync_dist=True)
        self.log('train/curvature_mae', F.l1_loss(predicted_H, gt_mean_curvatures).item(),
                 on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)

        # Area statistics
        areas = forward_result['areas']
        self.log('train/area_mean', areas.mean().item(), on_step=False, on_epoch=True,
                 logger=True, batch_size=batch_size, sync_dist=True)
        self.log('train/area_std', areas.std().item(), on_step=False, on_epoch=True,
                 logger=True, batch_size=batch_size, sync_dist=True)

        # Normal weights statistics
        normal_weights = forward_result['normal_weights']
        attention_mask = forward_result['attention_mask']
        self.log('train/normal_weights_mean', normal_weights[attention_mask].mean().item(),
                 on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)
        self.log('train/normal_weights_std', normal_weights[attention_mask].std().item(),
                 on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)

        # Log individual loss components
        for loss_name, loss_value in loss_components_unweighted.items():
            self.log(loss_name, loss_value, on_step=False, on_epoch=True, logger=True,
                     batch_size=batch_size, sync_dist=True)

        for loss_name, loss_value in loss_components_weighted.items():
            self.log(loss_name, loss_value, on_step=False, on_epoch=True, logger=True,
                     batch_size=batch_size, sync_dist=True)

        result = {"loss": total_loss}
        result.update(loss_components_weighted)
        result.update(loss_components_unweighted)

        return result

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, float]:
        """
        Validation step - compare predicted Laplacian eigendecomposition with ground truth.

        Uses derived stiffness weights: s_ij = 2 * H_i * w_ij

        Handles PyG batching by splitting combined Batch back into individual meshes.

        Args:
            batch: PyG Batch object (combined meshes)
            batch_idx: Batch index

        Returns:
            Dictionary with averaged validation metrics
        """
        # Split batch back into individual Data objects
        mesh_list = batch.to_data_list()

        # Validate each mesh and collect metrics
        all_metrics = []
        for mesh_data in mesh_list:
            mesh_metrics = self._validate_single_mesh(mesh_data)
            all_metrics.append(mesh_metrics)

        # Average metrics across batch
        averaged_metrics = {}
        if all_metrics:
            metric_names = all_metrics[0].keys()
            for name in metric_names:
                values = [m[name] for m in all_metrics if name in m]
                if values:
                    averaged_metrics[name] = sum(values) / len(values)

        # Log validation metrics
        for metric_name, metric_value in averaged_metrics.items():
            self.log(f'val/{metric_name}', metric_value, on_step=False, on_epoch=True,
                     logger=True, batch_size=len(mesh_list), sync_dist=True)

        return averaged_metrics

    def _validate_single_mesh(self, mesh_data: BaseData) -> Dict[str, float]:
        """
        Validate a single mesh by comparing predicted vs ground-truth eigendecomposition.

        Uses derived stiffness weights: s_ij = 2 * H_i * w_ij
        Then solves generalized eigenvalue problem: S @ v = lambda * M @ v

        Args:
            mesh_data: PyTorch Geometric Data object for a single mesh

        Returns:
            Dictionary with validation metrics for this mesh
        """
        # Convert single Data to Batch for forward pass
        mesh_batch = Batch.from_data_list([mesh_data])
        forward_result = self.forward(mesh_batch)
        predictions = self.compute_predictions(forward_result, mesh_batch)

        # Use patch_idx if available (MeshDataset), otherwise use batch (synthetic)
        batch_indices = getattr(mesh_batch, 'patch_idx', mesh_batch.batch)

        # Get derived stiffness weights: s_ij = 2 * H_i * w_ij
        stiffness_weights = predictions['stiffness_weights']
        areas = forward_result['areas']
        attention_mask = forward_result['attention_mask']

        # Assemble separate stiffness and mass matrices
        stiffness_matrix, mass_matrix = assemble_stiffness_and_mass_matrices(
            stiffness_weights=stiffness_weights,
            areas=areas,
            attention_mask=attention_mask,
            vertex_indices=mesh_batch.vertex_indices,
            center_indices=mesh_batch.center_indices,
            batch_indices=batch_indices
        )

        # Store matrices for callback (last mesh in batch)
        self._last_stiffness_matrix = stiffness_matrix
        self._last_mass_matrix = mass_matrix

        # Compute predicted eigendecomposition using generalized eigenvalue problem
        pred_eigenvalues, pred_eigenvectors = compute_laplacian_eigendecomposition(
            stiffness_matrix, self._num_eigenvalues, mass_matrix=mass_matrix
        )

        # Get ground-truth eigendecomposition (stored as tuple to avoid PyG batching issues)
        gt_eigenvalues, gt_eigenvectors = mesh_data.gt_eigen

        # Compute comparison metrics
        return self._compute_spectral_comparison_metrics(
            pred_eigenvalues, pred_eigenvectors,
            gt_eigenvalues, gt_eigenvectors
        )

    def _compute_spectral_comparison_metrics(
            self,
            pred_eigenvalues: np.ndarray,
            pred_eigenvectors: np.ndarray,
            gt_eigenvalues: np.ndarray,
            gt_eigenvectors: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute metrics comparing predicted vs ground-truth eigendecomposition.

        All computations done in numpy to avoid GPU memory usage.

        Args:
            pred_eigenvalues: Predicted eigenvalues (k,)
            pred_eigenvectors: Predicted eigenvectors (N, k)
            gt_eigenvalues: Ground-truth eigenvalues (k,)
            gt_eigenvectors: Ground-truth eigenvectors (N, k)

        Returns:
            Dictionary of comparison metrics (float values)
        """
        # Ensure we compare same number of eigenvalues
        k = min(len(pred_eigenvalues), len(gt_eigenvalues))
        pred_eig = pred_eigenvalues[:k]
        gt_eig = gt_eigenvalues[:k]
        pred_vec = pred_eigenvectors[:, :k]
        gt_vec = gt_eigenvectors[:, :k]

        metrics = {}

        # === Eigenvalue metrics ===

        # 1. Relative MSE (skip first eigenvalue since it's ~0)
        if k > 1:
            eps = 1e-6
            rel_errors_sq = ((pred_eig[1:] - gt_eig[1:]) / (gt_eig[1:] + eps)) ** 2
            metrics['eigenvalue_rel_mse'] = float(rel_errors_sq.mean())

        # 2. Spectral gap ratio (lambda_1 - lambda_0)
        pred_gap = pred_eig[1] - pred_eig[0] if k > 1 else 0.0
        gt_gap = gt_eig[1] - gt_eig[0] if k > 1 else 1.0
        metrics['spectral_gap_ratio'] = float(pred_gap / (gt_gap + 1e-6))

        # 3. Eigenvalue correlation
        if k > 2:
            correlation = np.corrcoef(pred_eig, gt_eig)[0, 1]
            metrics['eigenvalue_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0

        # 4. First non-zero eigenvalue ratio
        if k > 1:
            metrics['lambda1_ratio'] = float(pred_eig[1] / (gt_eig[1] + 1e-6))

        # === Eigenvector metrics ===

        # Compute cosine similarity for each eigenvector pair (handle sign ambiguity)
        cos_similarities = []
        for i in range(k):
            pred_v = pred_vec[:, i]
            gt_v = gt_vec[:, i]

            # Normalize vectors
            pred_v_norm = pred_v / (np.linalg.norm(pred_v) + 1e-8)
            gt_v_norm = gt_v / (np.linalg.norm(gt_v) + 1e-8)

            # Cosine similarity (absolute value due to sign ambiguity)
            cos_sim = np.abs(np.dot(pred_v_norm, gt_v_norm))
            cos_similarities.append(cos_sim)

        cos_similarities = np.array(cos_similarities)

        # Mean eigenvector similarity (all eigenvectors)
        metrics['eigenvector_similarity_mean'] = float(cos_similarities.mean())

        # Eigenvector similarity excluding first (constant) eigenvector
        if k > 1:
            metrics['eigenvector_similarity_mean_skip0'] = float(cos_similarities[1:].mean())

        # Individual eigenvector similarities for first few
        for i in range(min(k, 5)):  # Limit to first 5 for logging
            metrics[f'eigenvector_{i}_similarity'] = float(cos_similarities[i])

        # Overall spectral distance (combines eigenvalue and eigenvector differences)
        eigenvalue_error = float(np.mean(((pred_eig - gt_eig) / (gt_eig + 1e-6)) ** 2)) if k > 0 else 0.0
        eigenvector_error = 1.0 - float(cos_similarities.mean())
        metrics['spectral_distance'] = eigenvalue_error + eigenvector_error

        return metrics