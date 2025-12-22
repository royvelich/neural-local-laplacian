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
from neural_local_laplacian.utils.utils import split_results_by_nodes, split_results_by_graphs, assemble_sparse_laplacian_variable, compute_laplacian_eigendecomposition
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
            if '.git' in path:
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
        # The format depends on what scheduler monitoring you want
        scheduler_config = {
            "scheduler": scheduler,
            "interval": 'epoch'
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }


class LaplacianTransformerModule(LaplacianModuleBase):
    """Surface transformer module with support for variable-sized patches."""

    def __init__(self,
                 input_dim: int,
                 loss_configs: Optional[List[LossConfig]] = None,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_eigenvalues: int = 10,
                 **kwargs):
        super().__init__(**kwargs)

        # This saves all the __init__ arguments automatically
        # Exclude loss_configs from hyperparameters since they contain non-serializable PyTorch modules
        self.save_hyperparameters(ignore=['loss_configs'])

        # Manually save loss configuration info for logging (serializable version)
        if loss_configs is not None:
            self.hparams['loss_info'] = {
                'num_losses': len(loss_configs),
                'loss_types': [type(config.loss_module).__name__ for config in loss_configs],
                'loss_weights': [config.weight for config in loss_configs],
                'normalized_weights': [config.weight for config in self._normalize_loss_weights(loss_configs)]
            }

        # Validate input_dim
        if input_dim is None or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got: {input_dim}")

        self._d_model = d_model
        self._input_dim = input_dim
        self._num_eigenvalues = num_eigenvalues

        # Normalize loss weights to sum to 1
        self._loss_configs = self._normalize_loss_weights(loss_configs)

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

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

        # Output projection to scalar weights
        self.output_projection = nn.Linear(d_model, 1)

    def _normalize_loss_weights(self, loss_configs: List['LossConfig']) -> List['LossConfig']:
        """
        Normalize loss weights so they sum to 1.

        Args:
            loss_configs: List of LossConfig objects

        Returns:
            List of LossConfig objects with normalized weights
        """
        if not loss_configs:
            return loss_configs

        # Calculate total weight
        total_weight = sum(config.weight for config in loss_configs)

        if total_weight == 0:
            raise ValueError("Total loss weights cannot be zero")

        # Create new configs with normalized weights
        normalized_configs = []
        for config in loss_configs:
            normalized_weight = config.weight / total_weight
            # Create a new LossConfig with normalized weight
            normalized_config = LossConfig(
                loss_module=config.loss_module,
                weight=normalized_weight
            )
            normalized_configs.append(normalized_config)

        # Verify weights sum to 1 (within numerical precision)
        total_normalized = sum(config.weight for config in normalized_configs)
        assert abs(total_normalized - 1.0) < 1e-6, f"Normalized weights sum to {total_normalized}, not 1.0"

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
        # This replaces the loop with a vectorized operation
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

    def _compute_mean_curvature_vectors_vectorized(self, forward_result: Dict[str, torch.Tensor],
                                                   batch_data: Batch) -> torch.Tensor:
        """
        Compute predicted mean curvature vectors from token weights using fully vectorized operations.

        This method computes Î”r = Î£áµ¢ wáµ¢ * páµ¢ for each patch, where:
        - wáµ¢ are the learned token weights
        - páµ¢ are the patch positions (centered at origin)

        Args:
            forward_result: Dictionary containing token_weights, attention_mask, and batch_sizes
            batch_data: PyTorch Geometric batch containing positions and batch indices

        Returns:
            predicted_mean_curvature_vectors: (batch_size, 3) tensor of predicted mean curvature vectors
        """
        # Extract results
        token_weights = forward_result['token_weights']  # (batch_size, max_k)
        attention_mask = forward_result['attention_mask']  # (batch_size, max_k)
        batch_sizes = forward_result['batch_sizes']  # (batch_size,)
        batch_size = len(batch_sizes)

        # Get positions and flatten token weights with attention mask
        positions = batch_data.pos  # (total_points, 3)

        # Apply attention mask to token weights (zero out padded positions)
        masked_weights = token_weights.masked_fill(~attention_mask, 0.0)  # (batch_size, max_k)

        # Flatten masked weights to match positions structure
        weights_flat = masked_weights.flatten()  # (batch_size * max_k,)

        # Create batch indices for flattened weights
        batch_indices_weights = torch.arange(batch_size, device=token_weights.device).repeat_interleave(token_weights.shape[1])

        # Create position indices within each batch for mapping
        # This maps from flattened weight indices to actual position indices
        batch_cumsum = torch.cumsum(batch_sizes, dim=0)
        batch_starts = torch.cat([torch.zeros(1, device=batch_cumsum.device, dtype=batch_cumsum.dtype), batch_cumsum[:-1]])

        # Create position indices for each weight
        position_indices = torch.arange(token_weights.shape[1], device=token_weights.device).repeat(batch_size)

        # Filter out indices that exceed actual batch sizes
        valid_mask = position_indices < batch_sizes.repeat_interleave(token_weights.shape[1])

        # Get valid weights and their corresponding batch indices
        valid_weights = weights_flat[valid_mask]  # (num_valid,)
        valid_batch_indices = batch_indices_weights[valid_mask]  # (num_valid,)
        valid_position_indices = position_indices[valid_mask]  # (num_valid,)

        # Calculate actual position indices in the flattened positions array
        actual_position_indices = batch_starts[valid_batch_indices] + valid_position_indices

        # Get valid positions
        valid_positions = positions[actual_position_indices]  # (num_valid, 3)

        # Compute weighted positions: wáµ¢ * páµ¢
        weighted_positions = valid_weights.unsqueeze(-1) * valid_positions  # (num_valid, 3)

        # Sum weighted positions for each batch using scatter_add
        predicted_mean_curvature_vectors = torch.zeros(batch_size, 3, device=token_weights.device)
        predicted_mean_curvature_vectors.scatter_add_(0,
                                                      valid_batch_indices.unsqueeze(-1).expand(-1, 3),
                                                      weighted_positions)

        return predicted_mean_curvature_vectors

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
        Forward pass supporting variable-sized patches.

        Args:
            batch: PyTorch Geometric batch

        Returns:
            Dict containing:
            - token_weights: (batch_size, max_k) - learned Laplacian weights (padded)
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

        # Output projection to get scalar weights per token
        token_weights = self.output_projection(encoded_features)  # (batch_size, max_k, 1)
        token_weights = token_weights.squeeze(-1)  # (batch_size, max_k)

        # Apply softplus to ensure positive weights
        token_weights = torch.exp(token_weights)

        # Mask out padded positions
        token_weights = token_weights.masked_fill(~attention_mask, 0.0)

        return {
            'token_weights': token_weights,
            'attention_mask': attention_mask,
            'batch_sizes': batch_sizes
        }

    def training_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step with variable-sized patch support.

        Args:
            batch: List of PyTorch Geometric batches (synthetic data)
            batch_idx: Batch index

        Returns:
            Dictionary with loss and other metrics
        """
        # Take the first batch from the list
        batch_data = batch[0]
        forward_result = self.forward(batch_data)

        # Compute mean curvature vectors using vectorized method
        predicted_mean_curvature_vectors = self._compute_mean_curvature_vectors_vectorized(forward_result, batch_data)

        # Get batch size for logging
        batch_size = len(forward_result['batch_sizes'])

        # In training mode, diff_geom_at_origin_only=True, so normals and H are per-surface
        normals = batch_data.normal  # (batch_size, 3) - one normal per surface at origin
        mean_curvatures = batch_data.H  # (batch_size,) - one curvature per surface at origin

        # Target: H * nÌ‚ (mean curvature times unit normal at origin)
        target_mean_curvature_vectors = mean_curvatures.unsqueeze(-1) * F.normalize(normals, p=2, dim=1)  # (batch_size, 3)

        # Compute weighted combination of losses
        total_loss = 0.0
        loss_components_weighted = {}
        loss_components_unweighted = {}

        for i, loss_config in enumerate(self._loss_configs):
            # Compute unweighted loss
            unweighted_loss = loss_config.loss_module(predicted_mean_curvature_vectors, target_mean_curvature_vectors)

            # Compute weighted loss
            weighted_loss = loss_config.weight * unweighted_loss
            total_loss += weighted_loss

            # Store both weighted and unweighted loss components for logging
            loss_name = f"{loss_config.loss_module.__class__.__name__}"
            loss_components_weighted[f"train_{loss_name}_weighted"] = weighted_loss
            loss_components_unweighted[f"train_{loss_name}"] = unweighted_loss

        # Log the total loss
        self.log('train_loss', float(total_loss.item()), on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size, sync_dist=True)

        # Log individual unweighted loss components (these are the main loss values to track)
        for loss_name, loss_value in loss_components_unweighted.items():
            self.log(loss_name, loss_value, on_step=False, on_epoch=True, logger=True,
                     batch_size=batch_size, sync_dist=True)

        # Log individual weighted loss components (for debugging the weighting)
        for loss_name, loss_value in loss_components_weighted.items():
            self.log(loss_name, loss_value, on_step=False, on_epoch=True, logger=True,
                     batch_size=batch_size, sync_dist=True)

        # Create return dictionary with all losses
        result = {"loss": total_loss}
        result.update(loss_components_weighted)
        result.update(loss_components_unweighted)

        return result

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, float]:
        """
        Validation step - compare predicted Laplacian eigendecomposition with ground truth.

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
            self.log(f'val_{metric_name}', metric_value, on_step=False, on_epoch=True,
                     logger=True, batch_size=len(mesh_list))

        return averaged_metrics

    def _validate_single_mesh(self, mesh_data: BaseData) -> Dict[str, float]:
        """
        Validate a single mesh by comparing predicted vs ground-truth eigendecomposition.

        Args:
            mesh_data: PyTorch Geometric Data object for a single mesh

        Returns:
            Dictionary with validation metrics for this mesh
        """
        # Convert single Data to Batch for forward pass
        mesh_batch = Batch.from_data_list([mesh_data])
        forward_result = self.forward(mesh_batch)

        # Use patch_idx if available (MeshDataset), otherwise use batch (synthetic)
        batch_indices = getattr(mesh_batch, 'patch_idx', mesh_batch.batch)

        # Assemble sparse Laplacian matrix from learned weights
        laplacian_matrix = assemble_sparse_laplacian_variable(
            weights=forward_result['token_weights'],
            attention_mask=forward_result['attention_mask'],
            vertex_indices=mesh_batch.vertex_indices,
            center_indices=mesh_batch.center_indices,
            batch_indices=batch_indices
        )

        # Store Laplacian matrix for callback (last mesh in batch)
        self._last_laplacian_matrix = laplacian_matrix

        # Compute predicted eigendecomposition
        pred_eigenvalues, pred_eigenvectors = compute_laplacian_eigendecomposition(
            laplacian_matrix, self._num_eigenvalues
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
        for i in range(min(k, 5)):
            metrics[f'eigenvector_{i}_similarity'] = float(cos_similarities[i])

        # Overall spectral distance (combines eigenvalue and eigenvector differences)
        # Lower is better
        eigenvalue_error = float(np.mean(((pred_eig - gt_eig) / (gt_eig + 1e-6)) ** 2)) if k > 0 else 0.0
        eigenvector_error = 1.0 - float(cos_similarities.mean())
        metrics['spectral_distance'] = eigenvalue_error + eigenvector_error

        return metrics