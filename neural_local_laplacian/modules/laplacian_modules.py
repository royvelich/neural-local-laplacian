import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch

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

# trimesh for loading mesh vertices
import trimesh

# pyfm
from pyFM.mesh import TriMesh

# neural laplacian
from neural_local_laplacian.utils.utils import split_results_by_nodes, split_results_by_graphs, assemble_sparse_laplacian_variable
from neural_local_laplacian.modules.losses import LossConfig


class LocalLaplacianModuleBase(lightning.pytorch.LightningModule):
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

    def _shared_step(self, batch: Batch, batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
        pass

    def training_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step logic."""
        return self._shared_step(batch, batch_idx, 'train')

    # def validation_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
    #     """Validation step logic."""
    #     return self._shared_step(batch, batch_idx, 'val')

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


class SurfaceTransformerModule(LocalLaplacianModuleBase):
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

        This method computes Δr = Σᵢ wᵢ * pᵢ for each patch, where:
        - wᵢ are the learned token weights
        - pᵢ are the patch positions (centered at origin)

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

        # Compute weighted positions: wᵢ * pᵢ
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

        # Fix batch indices for MeshDataset validation if needed
        if hasattr(batch, 'center_indices') and torch.all(batch.batch == 0):
            # This is likely MeshDataset where DataLoader reset batch indices
            num_patches = len(batch.center_indices)
            total_points = len(batch.pos)

            # Create proper batch indices based on center_indices
            batch_indices = torch.zeros(total_points, dtype=torch.long, device=batch.batch.device)

            # For MeshDataset, we need to reconstruct batch indices from center_indices
            # This assumes points are grouped by patch
            points_per_patch = total_points // num_patches
            for i in range(num_patches):
                start_idx = i * points_per_patch
                end_idx = (i + 1) * points_per_patch
                batch_indices[start_idx:end_idx] = i

            batch.batch = batch_indices
            print(f"Fixed batch indices for MeshDataset: {num_patches} patches")

        # Get batch sizes (number of points per patch)
        batch_sizes = batch.batch.bincount()
        batch_size = len(batch_sizes)
        max_k = batch_sizes.max().item()

        # Project to model dimension
        features = self.input_projection(features)  # (total_points, d_model)

        # Pad sequences to max_k and create attention masks
        sequences, attention_mask = self._pad_sequences_vectorized(
            features, batch.batch, batch_size, max_k
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

        # Target: H * n̂ (mean curvature times unit normal at origin)
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

    def validation_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step with variable-sized patch support - eigenanalysis on mesh data.

        Args:
            batch: List of PyTorch Geometric batches (mesh data)
            batch_idx: Batch index

        Returns:
            Dictionary with validation metrics
        """
        # Take the first batch from the list
        batch_data = batch[0]
        forward_result = self.forward(batch_data)

        # Assemble sparse Laplacian matrix from variable-sized learned weights
        laplacian_matrix = assemble_sparse_laplacian_variable(
            weights=forward_result['token_weights'],
            attention_mask=forward_result['attention_mask'],
            vertex_indices=batch_data.vertex_indices,
            center_indices=batch_data.center_indices,
            batch_indices=batch_data.batch
        )

        # CRITICAL ADDITION: Store Laplacian matrix for ValidationMeshUploader
        self._last_laplacian_matrix = laplacian_matrix
        print(f"📊 Stored predicted Laplacian matrix for validation: {laplacian_matrix.shape}")

        # Compute eigendecomposition and validation metrics
        eigenvalues, eigenvectors = self._compute_eigendecomposition(laplacian_matrix)

        # Store eigendecomposition results for callback
        self._last_eigenvalues = eigenvalues
        self._last_eigenvectors = eigenvectors

        # Simple validation metrics
        metrics = {
            'first_eigenvalue': torch.tensor(eigenvalues[0]),
            'spectral_gap': torch.tensor(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else torch.tensor(0.0),
            'eigenvalue_mean': torch.tensor(eigenvalues.mean())
        }

        # Log validation metrics
        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_step=False, on_epoch=True, logger=True, batch_size=1)

        return metrics

    def _compute_eigendecomposition(self, laplacian_matrix: scipy.sparse.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of the Laplacian matrix using shift-invert mode.

        Uses shift-invert mode with sigma=-0.01 to find eigenvalues closest to zero.
        This is efficient for Laplacian matrices since we're interested in the smallest
        eigenvalues, which encode the most important spectral properties of the graph.

        Args:
            laplacian_matrix: Sparse positive semi-definite Laplacian matrix

        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted in ascending order by eigenvalue.
            - eigenvalues: Array of shape (k,) with smallest k eigenvalues
            - eigenvectors: Array of shape (n, k) with corresponding eigenvectors
        """
        # Use shift-invert mode with sigma=-0.01 to find eigenvalues closest to 0
        # This is robust for positive semi-definite Laplacian matrices
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(laplacian_matrix, k=self._num_eigenvalues, sigma=-0.01)

        # Sort eigenvalues and eigenvectors in ascending order
        sort_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        return eigenvalues, eigenvectors