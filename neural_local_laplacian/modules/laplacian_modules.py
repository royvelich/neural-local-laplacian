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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# torch_geometric
from torch_geometric.data import Batch
from torch_geometric.data import Data

# trimesh for loading mesh vertices
import trimesh

# pyfm
from pyFM.mesh import TriMesh

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
                 num_eigenvalues: int = 10,
                 **kwargs):
        super().__init__(**kwargs)

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
            dropout=0.0,
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

    def _forward_pass(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Shared forward pass logic for both training and validation.
        Only contains the transformer processing, no task-specific computations.

        Args:
            batch: PyTorch Geometric batch

        Returns:
            Dict containing:
            - token_weights: (batch_size, num_points) - learned Laplacian weights
            - batch_size: int
            - num_points_per_surface: int
        """
        features = batch.x  # Shape: (total_points, feature_dim)

        # Fix batch indices for MeshDataset validation
        # When DataLoader processes a single MeshDataset Data object, it resets all batch indices to 0
        # We need to restore the proper batch indices to separate patches
        if hasattr(batch, 'center_indices') and torch.all(batch.batch == 0):
            # This is likely MeshDataset where DataLoader reset batch indices
            num_patches = len(batch.center_indices)
            k = len(batch.pos) // num_patches
            # Recreate proper batch indices: [0,0,0,...,1,1,1,...,2,2,2,...]
            proper_batch_indices = torch.repeat_interleave(torch.arange(num_patches, device=batch.batch.device), k)
            batch.batch = proper_batch_indices
            print(f"Fixed batch indices for MeshDataset: {num_patches} patches with {k} points each")

        # Project to model dimension
        features = self.input_projection(features)  # (total_points, d_model)

        # Reshape to sequences per graph (all surfaces have same number of points)
        batch_sizes = batch.batch.bincount()
        num_points_per_surface = batch_sizes[0].item()  # All surfaces have same size
        batch_size = len(batch_sizes)

        # Simple reshape - no padding needed!
        sequences = features.view(batch_size, num_points_per_surface, self._d_model)

        # Pass through transformer (no masking needed!)
        encoded_features = self.transformer_encoder(sequences)

        # Output projection to get scalar weights per token
        token_weights = self.output_projection(encoded_features)  # (batch_size, num_points, 1)
        token_weights = token_weights.squeeze(-1)  # (batch_size, num_points)

        # Apply softplus to ensure positive weights
        token_weights = torch.exp(token_weights)  # (batch_size, num_points)

        return {
            'token_weights': token_weights,
            'batch_size': batch_size,
            'num_points_per_surface': num_points_per_surface
        }

    def training_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step - compute Laplacian prediction and losses.

        Args:
            batch: List of PyTorch Geometric batches (synthetic data)
            batch_idx: Batch index

        Returns:
            Dictionary with loss and other metrics
        """
        # Take the first batch from the list
        batch_data = batch[0]
        forward_result = self._forward_pass(batch_data)

        # Extract results
        token_weights = forward_result['token_weights']
        batch_size = forward_result['batch_size']
        num_points_per_surface = forward_result['num_points_per_surface']

        # Synthetic-specific computations for loss calculation
        # Reshape batch.pos
        positions = batch_data.pos.view(batch_size, num_points_per_surface, 3)  # (batch_size, num_points, 3)

        # In training mode, diff_geom_at_origin_only=True, so normals and H are per-surface
        normals = batch_data.normal  # (batch_size, 3) - one normal per surface at origin
        mean_curvatures = batch_data.H  # (batch_size,) - one curvature per surface at origin

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
        loss_components_weighted = {}
        loss_components_unweighted = {}

        for i, loss_config in enumerate(self._loss_configs):
            # Compute unweighted loss
            unweighted_loss = loss_config.loss_module(predicted_laplacian, target_laplacian)

            # Compute weighted loss
            weighted_loss = loss_config.weight * unweighted_loss
            total_loss += weighted_loss

            # Store both weighted and unweighted loss components for logging
            loss_name = f"{loss_config.loss_module.__class__.__name__}"
            loss_components_weighted[f"train_{loss_name}_weighted"] = weighted_loss
            loss_components_unweighted[f"train_{loss_name}"] = unweighted_loss

        # Log the total loss
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # Log individual unweighted loss components (these are the main loss values to track)
        for loss_name, loss_value in loss_components_unweighted.items():
            self.log(loss_name, loss_value, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)

        # Log individual weighted loss components (for debugging the weighting)
        for loss_name, loss_value in loss_components_weighted.items():
            self.log(loss_name, loss_value, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)

        # Create return dictionary with all losses
        result = {"loss": total_loss}
        result.update(loss_components_weighted)
        result.update(loss_components_unweighted)

        return result

    def validation_step(self, batch: List[Batch], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step - eigenanalysis on mesh data.

        Args:
            batch: List of PyTorch Geometric batches (mesh data)
            batch_idx: Batch index

        Returns:
            Dictionary with validation metrics
        """
        # Take the first batch from the list
        batch_data = batch[0]
        forward_result = self._forward_pass(batch_data)

        # Assemble sparse Laplacian matrix from learned weights
        laplacian_matrix = self._assemble_sparse_laplacian(
            weights=forward_result['token_weights'],
            vertex_indices=batch_data.vertex_indices,
            center_indices=batch_data.center_indices,
            batch_indices=batch_data.batch
        )

        # Print first 5 rows of Laplacian matrix (non-zero elements only)
        print("\n" + "=" * 80)
        print("LAPLACIAN MATRIX - FIRST 5 ROWS (NON-ZERO ELEMENTS)")
        print("=" * 80)
        num_rows_to_show = min(5, laplacian_matrix.shape[0])
        for row_idx in range(num_rows_to_show):
            # Get the row as a sparse vector
            row = laplacian_matrix.getrow(row_idx)

            # Find non-zero elements
            row_coo = row.tocoo()
            col_indices = row_coo.col
            values = row_coo.data

            if len(col_indices) > 0:
                print(f"Row {row_idx:3d}: ", end="")
                for col_idx, value in zip(col_indices, values):
                    print(f"({row_idx},{col_idx:3d})={value:8.4f} ", end="")
                print()  # New line

                # Also show row sum to verify it's close to zero
                row_sum = values.sum()
                print(f"         Row sum = {row_sum:.6f}")
            else:
                print(f"Row {row_idx:3d}: (no non-zero elements)")
            print()
        print("=" * 80)

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

    def _assemble_sparse_laplacian(self, weights: torch.Tensor, vertex_indices: torch.Tensor, center_indices: torch.Tensor, batch_indices: torch.Tensor) -> scipy.sparse.csr_matrix:
        """
        Assemble sparse Laplacian matrix from patch weights using vectorized operations.

        Args:
            weights: Token weights of shape (batch_size, num_points)
            vertex_indices: Neighbor vertex indices of shape (total_points,)
            center_indices: Center vertex index for each patch, shape (num_patches,)
            batch_indices: Batch indices of shape (total_points,)

        Returns:
            Sparse Laplacian matrix
        """
        # Convert to numpy for scipy operations
        weights_np = weights.detach().cpu().numpy()
        vertex_indices_np = vertex_indices.detach().cpu().numpy()
        center_indices_np = center_indices.detach().cpu().numpy()
        batch_indices_np = batch_indices.detach().cpu().numpy()

        # Get dimensions
        num_patches = weights.shape[0]
        num_points_per_patch = weights.shape[1]
        num_vertices = max(vertex_indices_np.max(), center_indices_np.max()) + 1

        # Flatten weights to match vertex_indices structure
        weights_flat = weights_np.flatten()  # Shape: (total_points,)

        # Expand center indices to match the structure of vertex_indices
        # Each center index should be repeated k times (once per neighbor)
        center_vertices_expanded = np.repeat(center_indices_np, num_points_per_patch)

        # Now we have:
        # center_vertices_expanded[i] = center vertex for the i-th neighbor point
        # vertex_indices_np[i] = neighbor vertex index for the i-th neighbor point
        # weights_flat[i] = weight for connection from center to neighbor

        # Create off-diagonal entries (negative weights)
        # Connection: center[i] -> neighbor[i] with -weight[i]
        row_indices = center_vertices_expanded  # From center
        col_indices = vertex_indices_np  # To neighbor
        data_values = -weights_flat  # Negative weights for off-diagonal

        # Create symmetric connections: neighbor[i] -> center[i] with same weight
        row_indices_sym = vertex_indices_np  # From neighbor
        col_indices_sym = center_vertices_expanded  # To center
        data_values_sym = -weights_flat  # Same negative weights

        # Combine all off-diagonal connections
        all_row_indices = np.concatenate([row_indices, row_indices_sym])
        all_col_indices = np.concatenate([col_indices, col_indices_sym])
        all_data_values = np.concatenate([data_values, data_values_sym])

        # Create sparse matrix from coordinates (off-diagonal entries only)
        laplacian_coo = scipy.sparse.coo_matrix(
            (all_data_values, (all_row_indices, all_col_indices)),
            shape=(num_vertices, num_vertices)
        )

        # Sum duplicate entries and convert to CSR
        laplacian_csr = laplacian_coo.tocsr()
        laplacian_csr.sum_duplicates()

        # Vectorized diagonal computation: each diagonal entry = -sum of off-diagonal entries in that row
        # This ensures each row sums to zero (discrete Laplacian property)
        # Get the sum of each row (which currently contains only off-diagonal entries)
        row_sums = np.array(laplacian_csr.sum(axis=1)).flatten()  # Shape: (num_vertices,)

        # Diagonal entries should be the negative of the row sums
        diagonal_values = -row_sums  # Shape: (num_vertices,)

        # Set diagonal entries
        laplacian_csr.setdiag(diagonal_values)

        # Ensure numerical symmetry (should already be symmetric, but for safety)
        laplacian_csr = 0.5 * (laplacian_csr + laplacian_csr.T)

        return laplacian_csr

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


class ValidationMeshUploader(Callback):
    """
    Callback to upload mesh vertices, faces, and eigendecomposition results to Weights & Biases.

    This callback:
    1. Extracts mesh vertices and faces from the validation dataset
    2. Computes ground-truth Laplacian eigendecomposition using PyFM
    3. Collects eigendecomposition results during validation
    4. Uploads everything as W&B artifacts at the end of each validation epoch
    """

    def __init__(self, k: int = 50):
        """
        Initialize the ValidationMeshUploader.

        Args:
            k: Number of eigenvalues/eigenvectors to compute for ground-truth Laplacian
        """
        super().__init__()
        self._validation_results = []  # Store results during validation
        self._mesh_data = None  # Cache mesh data including ground-truth eigendecomposition
        self._k = k  # Number of eigenvalues to compute

    def on_validation_start(self, trainer, pl_module):
        """Extract mesh data and compute ground-truth eigendecomposition at the start of validation."""
        # Get the validation dataloader
        val_dataloaders = trainer.datamodule.val_dataloader()
        val_dataloader = val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders

        # Check if this is a MeshDataset
        if hasattr(val_dataloader.dataset, 'mesh_file_paths'):
            mesh_dataset = val_dataloader.dataset

            # For simplicity, use the first mesh file (you can extend this for multiple meshes)
            if len(mesh_dataset.mesh_file_paths) > 0:
                mesh_file_path = mesh_dataset.mesh_file_paths[0]

                try:
                    # Load mesh vertices and faces using trimesh
                    mesh = trimesh.load(str(mesh_file_path))
                    raw_vertices = np.array(mesh.vertices, dtype=np.float32)
                    faces = np.array(mesh.faces, dtype=np.int32)

                    # Apply the same normalization as MeshDataset
                    # Import utils with proper path
                    from neural_local_laplacian.utils import utils
                    vertices = utils.normalize_mesh_vertices(raw_vertices)

                    # Update mesh with normalized vertices for normal computation
                    mesh.vertices = vertices
                    vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)

                    print(f"Loaded mesh for validation upload: {mesh_file_path}")
                    print(f"Mesh has {len(vertices)} vertices and {len(faces)} faces")
                    print(f"Normalized vertices: center at origin, max distance = {np.linalg.norm(vertices, axis=1).max():.6f}")

                    # Compute ground-truth Laplacian eigendecomposition using PyFM
                    print("Computing ground-truth Laplacian eigendecomposition...")

                    # Create PyFM TriMesh object
                    pyfm_mesh = TriMesh(vertices, faces)

                    # Process the mesh and compute the Laplacian spectrum
                    # Set intrinsic=False for using standard cotangent Laplacian
                    pyfm_mesh.process(k=self._k, intrinsic=False, verbose=False)

                    # Retrieve eigenvalues, eigenfunctions, and vertex areas
                    gt_eigenvalues = pyfm_mesh.eigenvalues  # Shape: (k,)
                    gt_eigenvectors = pyfm_mesh.eigenvectors  # Shape: (num_vertices, k)
                    vertex_areas = pyfm_mesh.vertex_areas  # Shape: (num_vertices,)

                    print(f"Computed {len(gt_eigenvalues)} ground-truth eigenvalues")
                    print(f"Ground-truth eigenvalue range: [{gt_eigenvalues[0]:.2e}, {gt_eigenvalues[-1]:.6f}]")
                    print(f"First eigenvalue (should be ~0): {gt_eigenvalues[0]:.2e}")
                    if len(gt_eigenvalues) > 1:
                        print(f"Second eigenvalue (Fiedler): {gt_eigenvalues[1]:.6f}")

                    self._mesh_data = {
                        'vertices': vertices,
                        'vertex_normals': vertex_normals,
                        'faces': faces,
                        'mesh_file_path': str(mesh_file_path),
                        'num_vertices': len(vertices),
                        'num_faces': len(faces),

                        # Ground-truth Laplacian eigendecomposition
                        'gt_eigenvalues': gt_eigenvalues,
                        'gt_eigenvectors': gt_eigenvectors,
                        'vertex_areas': vertex_areas,
                        'gt_num_eigenvalues': len(gt_eigenvalues)
                    }

                    print("Successfully computed and cached ground-truth eigendecomposition")

                except Exception as e:
                    print(f"Warning: Failed to load mesh or compute eigendecomposition for validation upload: {e}")
                    import traceback
                    traceback.print_exc()
                    self._mesh_data = None
        else:
            print("Validation dataset is not a MeshDataset - no mesh data to upload")
            self._mesh_data = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect validation results after each batch."""
        if outputs is not None and self._mesh_data is not None:
            # Store the validation results from this batch
            # outputs should contain the eigendecomposition metrics
            batch_results = {
                'batch_idx': batch_idx,
                'epoch': pl_module.current_epoch,
                'metrics': {},
                'eigendata': {}
            }

            # Extract metrics from outputs
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    batch_results['metrics'][key] = value.detach().cpu().numpy()
                else:
                    batch_results['metrics'][key] = value

            # If the module stores eigendecomposition data, extract it
            if hasattr(pl_module, '_last_eigenvalues') and hasattr(pl_module, '_last_eigenvectors'):
                batch_results['eigendata'] = {
                    'predicted_eigenvalues': pl_module._last_eigenvalues,
                    'predicted_eigenvectors': pl_module._last_eigenvectors,
                    'num_eigenvalues': len(pl_module._last_eigenvalues),
                    'matrix_size': pl_module._last_eigenvectors.shape[0] if pl_module._last_eigenvectors is not None else 0
                }

            self._validation_results.append(batch_results)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Upload validation data as W&B artifacts at the end of validation epoch."""
        if not self._validation_results or self._mesh_data is None:
            print("No validation results or mesh data to upload")
            return

        # Create directory for artifacts
        artifacts_dir = Path("validation_artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        epoch = pl_module.current_epoch
        epoch_dir = artifacts_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)

        rank = trainer.global_rank

        # Combine everything into a single data structure with PyTorch tensors
        combined_data = {
            'epoch': epoch,
            'rank': rank,
            'num_batches': len(self._validation_results),

            # Mesh data as PyTorch tensors
            'mesh_vertices': torch.from_numpy(self._mesh_data['vertices']).float(),
            'mesh_vertex_normals': torch.from_numpy(self._mesh_data['vertex_normals']).float(),
            'mesh_faces': torch.from_numpy(self._mesh_data['faces']).long(),
            'mesh_file_path': self._mesh_data['mesh_file_path'],
            'num_vertices': self._mesh_data['num_vertices'],
            'num_faces': self._mesh_data['num_faces'],

            # Ground-truth Laplacian eigendecomposition
            'gt_eigenvalues': torch.from_numpy(self._mesh_data['gt_eigenvalues']).float(),
            'gt_eigenvectors': torch.from_numpy(self._mesh_data['gt_eigenvectors']).float(),
            'vertex_areas': torch.from_numpy(self._mesh_data['vertex_areas']).float(),
            'gt_num_eigenvalues': self._mesh_data['gt_num_eigenvalues'],

            # Validation results with tensors
            'validation_results': []
        }

        # Convert all validation results to PyTorch tensors
        for batch_result in self._validation_results:
            converted_result = {
                'batch_idx': batch_result['batch_idx'],
                'epoch': batch_result['epoch'],
                'metrics': {},
                'eigendata': {}
            }

            # Convert metrics to PyTorch tensors
            for key, value in batch_result['metrics'].items():
                if isinstance(value, np.ndarray):
                    converted_result['metrics'][key] = torch.from_numpy(value).float()
                elif isinstance(value, (int, float)):
                    converted_result['metrics'][key] = torch.tensor(value).float()
                else:
                    converted_result['metrics'][key] = value

            # Convert eigendata to PyTorch tensors
            if 'eigendata' in batch_result and batch_result['eigendata']:
                eigendata = batch_result['eigendata']
                if 'predicted_eigenvalues' in eigendata and eigendata['predicted_eigenvalues'] is not None:
                    converted_result['eigendata']['predicted_eigenvalues'] = torch.from_numpy(eigendata['predicted_eigenvalues']).float()
                if 'predicted_eigenvectors' in eigendata and eigendata['predicted_eigenvectors'] is not None:
                    converted_result['eigendata']['predicted_eigenvectors'] = torch.from_numpy(eigendata['predicted_eigenvectors']).float()
                converted_result['eigendata']['num_eigenvalues'] = eigendata.get('num_eigenvalues', 0)
                converted_result['eigendata']['matrix_size'] = eigendata.get('matrix_size', 0)

            combined_data['validation_results'].append(converted_result)

        # Save everything in a single pickle file
        combined_pickle_name = f"rank_{rank}_validation_data.pkl"
        combined_pickle_path = epoch_dir / combined_pickle_name

        with open(combined_pickle_path, 'wb') as f:
            pickle.dump(combined_data, f)

        print(f"Saved validation data to {combined_pickle_path}")
        print(f"Data includes:")
        print(f"  - Mesh: {combined_data['num_vertices']} vertices, {combined_data['num_faces']} faces")
        print(f"  - Ground-truth: {combined_data['gt_num_eigenvalues']} eigenvalues")
        print(f"  - Validation results: {len(combined_data['validation_results'])} batches")

        # Synchronize all processes if using distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Only rank 0 uploads to W&B
        if trainer.global_rank == 0 and wandb.run is not None:
            self._upload_to_wandb(epoch, epoch_dir, artifacts_dir)

        # Clear results for next epoch
        self._validation_results = []

    def _upload_to_wandb(self, epoch: int, epoch_dir: Path, artifacts_dir: Path):
        """Upload artifacts to Weights & Biases."""
        try:
            # Create zip file with all validation data
            zip_filename = f"validation_mesh_data_epoch_{epoch}.zip"
            zip_path = artifacts_dir / zip_filename

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in epoch_dir.glob('*.pkl'):
                    zipf.write(file_path, arcname=file_path.name)

            # Create and upload artifact to W&B
            artifact = wandb.Artifact(
                name=f"validation_mesh_data_epoch_{epoch}",
                type="validation_mesh_data",
                description=f"Mesh data with ground-truth and predicted eigendecomposition from validation epoch {epoch}"
            )
            artifact.add_file(str(zip_path))

            # Add metadata
            artifact.metadata = {
                'epoch': epoch,
                'num_validation_batches': len(self._validation_results),
                'mesh_file': self._mesh_data['mesh_file_path'] if self._mesh_data else None,
                'num_vertices': self._mesh_data['num_vertices'] if self._mesh_data else 0,
                'num_faces': self._mesh_data['num_faces'] if self._mesh_data else 0,
                'gt_num_eigenvalues': self._mesh_data['gt_num_eigenvalues'] if self._mesh_data else 0,
                'eigenvalue_computation_k': self._k
            }

            wandb.log_artifact(artifact)
            print(f"Uploaded validation mesh data artifact for epoch {epoch}")

            # Clean up files after upload
            shutil.rmtree(epoch_dir)
            if zip_path.exists():
                zip_path.unlink()

        except Exception as e:
            print(f"Error uploading validation data to W&B: {e}")
            import traceback
            traceback.print_exc()