import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Batch

import os
import pickle
import zipfile
from pathlib import Path
from typing import List, Type, Callable, Optional, Dict, Tuple, Any
from dataclasses import dataclass

# wandb
import wandb

# omegaconf
from omegaconf import DictConfig, OmegaConf

# torch
import torch
import torch.nn.functional as F

# numpy
import numpy as np

# lightning
import pytorch_lightning as pl
from lightning.pytorch.callbacks import Callback

# torch_geometric
from torch_geometric.data import Batch
from torch_geometric.data import Data

# neural laplacian
from neural_local_laplacian.utils.utils import split_results_by_nodes, split_results_by_graphs
from neural_local_laplacian.modules.losses import LossConfig


class LocalLaplacianModuleBase(pl.LightningModule):
    def __init__(self,
                 optimizer_cfg: DictConfig
                 ):
        super().__init__()
        self._optimizer_cfg = optimizer_cfg

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self._optimizer_cfg(params=self.parameters())


class SurfaceTransformerModule(LocalLaplacianModuleBase):
    """Simple transformer module for processing surface point clouds."""

    def __init__(self,
                 input_dim: int,
                 loss_configs: List['LossConfig'],
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)

        # Validate input_dim
        if input_dim is None or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got: {input_dim}")

        self._d_model = d_model
        self._input_dim = input_dim
        self._loss_configs = loss_configs

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

    def _shared_step(self, batch: List[Batch], batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
        """
        Shared step for training and validation.

        Args:
            batch: List of PyTorch Geometric batches
            batch_idx: Batch index
            stage: 'train' or 'val'

        Returns:
            Dictionary with loss and other metrics
        """
        # Take the first batch from the list
        batch = batch[0]

        features = batch.x  # Shape: (total_points, feature_dim)

        # Project to model dimension
        features = self.input_projection(features)  # (total_points, d_model)

        # Reshape to sequences per graph (all surfaces have same number of points)
        batch_sizes = batch.batch.bincount()
        num_points_per_surface = batch_sizes[0].item()  # All surfaces have same size
        batch_size = len(batch_sizes)

        # Simple reshape - no padding needed!
        sequences = features.view(batch_size, num_points_per_surface, self._d_model)

        # Reshape batch.pos
        positions = batch.pos.view(batch_size, num_points_per_surface, 3)  # (batch_size, num_points, 3)

        # In training mode, diff_geom_at_origin_only=True, so normals and H are per-surface
        normals = batch.normal  # (batch_size, 3) - one normal per surface at origin
        mean_curvatures = batch.H  # (batch_size,) - one curvature per surface at origin

        # Pass through transformer (no masking needed!)
        encoded_features = self.transformer_encoder(sequences)

        # Output projection to get scalar weights per token
        token_weights = self.output_projection(encoded_features)  # (batch_size, num_points, 1)
        token_weights = token_weights.squeeze(-1)  # (batch_size, num_points)

        # Apply softplus to ensure positive weights
        token_weights = F.softplus(token_weights)  # (batch_size, num_points)

        # Compute Laplace-Beltrami operator: Δr = Σᵢ wᵢ * pᵢ
        # Since center point is at origin, we don't need to subtract center coordinates
        predicted_laplacian = torch.sum(token_weights.unsqueeze(-1) * positions, dim=1)  # (batch_size, 3)

        # Target: H * n̂ (mean curvature times unit normal at origin)
        target_laplacian = mean_curvatures.unsqueeze(-1) * F.normalize(normals, p=2, dim=1)  # (batch_size, 3)

        print('\n')
        print('-' * 100)
        print(predicted_laplacian[0])
        print('-' * 100)
        print(target_laplacian[0])
        print('-' * 100)
        print('\n')

        print('\n')
        print('Token Weights:')
        print(token_weights[0])
        print('-' * 100)
        print('\n')

        # Compute weighted combination of losses
        total_loss = 0.0
        loss_components = {}

        for i, loss_config in enumerate(self._loss_configs):
            component_loss = loss_config.loss_module(predicted_laplacian, target_laplacian)
            weighted_loss = loss_config.weight * component_loss
            total_loss += weighted_loss

            # Store individual loss components for logging
            loss_name = f"{stage}_{loss_config.loss_module.__class__.__name__}"
            loss_components[loss_name] = component_loss

        # Log the total loss and all components
        self.log(f'{stage}_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # Log individual loss components
        for loss_name, loss_value in loss_components.items():
            self.log(loss_name, loss_value, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)

        # Create return dictionary with all losses
        result = {"loss": total_loss}
        result.update(loss_components)

        return result