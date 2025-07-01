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
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, sync_dist=True)

        # Log individual unweighted loss components (these are the main loss values to track)
        for loss_name, loss_value in loss_components_unweighted.items():
            self.log(loss_name, loss_value, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)

        # Log individual weighted loss components (for debugging the weighting)
        for loss_name, loss_value in loss_components_weighted.items():
            self.log(loss_name, loss_value, on_step=False, on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)

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

        # CRITICAL ADDITION: Store Laplacian matrix for ValidationMeshUploader
        self._last_laplacian_matrix = laplacian_matrix
        print(f"📊 Stored predicted Laplacian matrix for validation: {laplacian_matrix.shape}")

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
    3. Computes ground-truth mean curvature using libigl
    4. Collects eigendecomposition results during validation
    5. Uploads everything as W&B artifacts at the end of each validation epoch

    Fixed for multi-GPU settings to properly handle mesh data per validation result.
    """

    def __init__(self, k: int = 50):
        """
        Initialize the ValidationMeshUploader.

        Args:
            k: Number of eigenvalues/eigenvectors to compute for ground-truth Laplacian
        """
        super().__init__()
        self._validation_results = []  # Store results during validation
        self._mesh_cache = {}  # Cache mesh data by mesh file path
        self._k = k  # Number of eigenvalues to compute

    def _load_and_cache_mesh(self, mesh_file_path: Path) -> Dict[str, Any]:
        """
        Load mesh data and cache it by file path.

        Args:
            mesh_file_path: Path to the mesh file

        Returns:
            Dictionary containing mesh data and ground-truth eigendecomposition
        """
        mesh_file_str = str(mesh_file_path)

        # Return cached data if available
        if mesh_file_str in self._mesh_cache:
            return self._mesh_cache[mesh_file_str]

        try:
            # Load mesh vertices and faces using trimesh
            mesh = trimesh.load(mesh_file_str)
            raw_vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)

            # Apply the same normalization as MeshDataset
            from neural_local_laplacian.utils import utils
            vertices = utils.normalize_mesh_vertices(raw_vertices)

            # Update mesh with normalized vertices for normal computation
            mesh.vertices = vertices
            vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)

            print(f"Loading mesh for validation upload: {mesh_file_path}")
            print(f"Mesh has {len(vertices)} vertices and {len(faces)} faces")
            print(f"Normalized vertices: center at origin, max distance = {np.linalg.norm(vertices, axis=1).max():.6f}")

            # Compute GT mean curvature using libigl
            print("Computing GT mean curvature using libigl...")
            try:
                import igl

                # Convert to float64 for libigl (it's more stable with double precision)
                vertices_igl = vertices.astype(np.float64)
                faces_igl = faces.astype(np.int32)

                # Compute principal curvatures using libigl
                _, _, principal_curvature1, principal_curvature2, _ = igl.principal_curvature(
                    vertices_igl, faces_igl
                )

                # Mean curvature is the average of principal curvatures: H = (k1 + k2) / 2
                gt_mean_curvature = (principal_curvature1 + principal_curvature2) / 2.0

                # Convert back to float32 for consistency
                gt_mean_curvature = gt_mean_curvature.astype(np.float32)

                print(f"Computed GT mean curvature using libigl")
                print(f"Mean curvature range: [{gt_mean_curvature.min():.6f}, {gt_mean_curvature.max():.6f}]")
                print(f"Mean curvature mean: {gt_mean_curvature.mean():.6f}")

            except ImportError:
                print("Warning: libigl not available, GT mean curvature will not be computed")
                gt_mean_curvature = None
            except Exception as e:
                print(f"Warning: Failed to compute GT mean curvature with libigl: {e}")
                gt_mean_curvature = None

            # Compute ground-truth Laplacian eigendecomposition using PyFM
            print("Computing ground-truth Laplacian eigendecomposition...")

            # Create PyFM TriMesh object
            pyfm_mesh = TriMesh(vertices, faces)

            # Process the mesh and compute the Laplacian spectrum
            pyfm_mesh.process(k=self._k, intrinsic=False, verbose=False)

            # Retrieve eigenvalues, eigenfunctions, and vertex areas
            gt_eigenvalues = pyfm_mesh.eigenvalues  # Shape: (k,)
            gt_eigenvectors = pyfm_mesh.eigenvectors  # Shape: (num_vertices, k)
            vertex_areas = pyfm_mesh.vertex_areas  # Shape: (num_vertices,)

            print(f"Computed {len(gt_eigenvalues)} ground-truth eigenvalues")
            print(f"Ground-truth eigenvalue range: [{gt_eigenvalues[0]:.2e}, {gt_eigenvalues[-1]:.6f}]")

            mesh_data = {
                'vertices': vertices,
                'vertex_normals': vertex_normals,
                'faces': faces,
                'mesh_file_path': mesh_file_str,
                'num_vertices': len(vertices),
                'num_faces': len(faces),

                # Ground-truth Laplacian eigendecomposition
                'gt_eigenvalues': gt_eigenvalues,
                'gt_eigenvectors': gt_eigenvectors,
                'vertex_areas': vertex_areas,
                'gt_num_eigenvalues': len(gt_eigenvalues),

                # Ground-truth mean curvature computed with libigl
                'gt_mean_curvature': gt_mean_curvature
            }

            # Cache the mesh data
            self._mesh_cache[mesh_file_str] = mesh_data
            print(f"Successfully computed and cached ground-truth eigendecomposition and mean curvature for {mesh_file_str}")

            return mesh_data

        except Exception as e:
            print(f"Warning: Failed to load mesh or compute eigendecomposition for {mesh_file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_mesh_info_from_batch(self, batch) -> Dict[str, Any]:
        """
        Extract mesh information directly from the validation batch.

        Args:
            batch: Current validation batch (list containing Data objects)

        Returns:
            Dictionary containing mesh metadata extracted from batch
        """
        # Get the batch data (list containing single Data object for MeshDataset)
        batch_data = batch[0] if isinstance(batch, list) else batch

        print(f"  Debug: batch_data type = {type(batch_data)}")

        def _extract_value_from_batched_attribute(attr_value):
            """Extract single value from potentially batched attribute."""
            if isinstance(attr_value, list):
                if len(attr_value) == 0:
                    return None
                return attr_value[0]  # Take first element
            elif torch.is_tensor(attr_value):
                if attr_value.numel() == 1:
                    return attr_value.item()  # Extract scalar from tensor
                elif attr_value.numel() > 1:
                    return attr_value[0].item()  # Take first element and extract scalar
                else:
                    return None
            else:
                return attr_value

        # Try to extract metadata using individual attributes (primary method)
        try:
            mesh_file_path_raw = getattr(batch_data, 'mesh_file_path', None)
            original_num_vertices_raw = getattr(batch_data, 'original_num_vertices', None)
            original_num_faces_raw = getattr(batch_data, 'original_num_faces', None)
            mesh_idx_raw = getattr(batch_data, 'mesh_idx', None)
            k_neighbors_raw = getattr(batch_data, 'k_neighbors', 0)

            if mesh_file_path_raw is not None:
                # Extract values from batched attributes
                mesh_file_path = _extract_value_from_batched_attribute(mesh_file_path_raw)
                original_num_vertices = _extract_value_from_batched_attribute(original_num_vertices_raw)
                original_num_faces = _extract_value_from_batched_attribute(original_num_faces_raw)
                mesh_idx = _extract_value_from_batched_attribute(mesh_idx_raw)
                k_neighbors = _extract_value_from_batched_attribute(k_neighbors_raw)

                print(f"  ✅ Found mesh metadata as individual attributes")
                print(f"    mesh_file_path: {mesh_file_path}")
                print(f"    original_num_vertices: {original_num_vertices}")

                # Calculate processed vertices from the actual batch data
                processed_num_vertices = len(torch.unique(batch_data.center_indices)) if hasattr(batch_data, 'center_indices') else 0
                total_points = len(batch_data.pos) if hasattr(batch_data, 'pos') else 0

                return {
                    'mesh_file_path': Path(mesh_file_path),
                    'original_num_vertices': original_num_vertices,
                    'original_num_faces': original_num_faces,
                    'mesh_idx': mesh_idx,
                    'processed_num_vertices': processed_num_vertices,
                    'total_points': total_points,
                    'k_neighbors': k_neighbors or 0
                }
        except Exception as e:
            print(f"  ⚠️  Failed to extract via individual attributes: {e}")

        # Fallback: Try mesh_metadata dict approach
        if hasattr(batch_data, 'mesh_metadata'):
            mesh_metadata = batch_data.mesh_metadata
            print(f"  Fallback: mesh_metadata type = {type(mesh_metadata)}")

            # Handle case where mesh_metadata might be a list due to batching
            if isinstance(mesh_metadata, list):
                if len(mesh_metadata) == 0:
                    raise ValueError("mesh_metadata list is empty")
                # Take the first metadata (assuming all patches from same mesh)
                mesh_metadata = mesh_metadata[0]
                print(f"  Note: mesh_metadata was a list, using first element")

            # Ensure mesh_metadata is a dictionary
            if not isinstance(mesh_metadata, dict):
                raise ValueError(f"mesh_metadata should be a dict, got {type(mesh_metadata)}: {mesh_metadata}")

            # Extract and clean values from metadata dict
            mesh_file_path = _extract_value_from_batched_attribute(mesh_metadata['mesh_file_path'])
            original_num_vertices = _extract_value_from_batched_attribute(mesh_metadata['original_num_vertices'])
            original_num_faces = _extract_value_from_batched_attribute(mesh_metadata['original_num_faces'])
            mesh_idx = _extract_value_from_batched_attribute(mesh_metadata['mesh_idx'])
            k_neighbors = _extract_value_from_batched_attribute(mesh_metadata.get('k_neighbors', 0))

            # Calculate processed vertices from the actual batch data
            processed_num_vertices = len(torch.unique(batch_data.center_indices)) if hasattr(batch_data, 'center_indices') else 0
            total_points = len(batch_data.pos) if hasattr(batch_data, 'pos') else 0

            print(f"  ✅ Using mesh_metadata dict fallback")
            print(f"    mesh_file_path: {mesh_file_path}")
            print(f"    original_num_vertices: {original_num_vertices}")

            return {
                'mesh_file_path': Path(mesh_file_path),
                'original_num_vertices': original_num_vertices,
                'original_num_faces': original_num_faces,
                'mesh_idx': mesh_idx,
                'processed_num_vertices': processed_num_vertices,
                'total_points': total_points,
                'k_neighbors': k_neighbors or 0
            }

        # If we get here, no metadata was found
        raise ValueError("No mesh metadata found in batch data. MeshDataset needs to be updated to include metadata.")

    def _validate_mesh_eigendata_consistency(self, mesh_info: Dict, eigendata: Dict) -> bool:
        """
        Validate that mesh and eigendata are consistent.

        Args:
            mesh_info: Mesh information extracted from batch
            eigendata: Eigendecomposition data from model

        Returns:
            True if consistent, False otherwise
        """
        if not eigendata:
            return True  # No eigendata to validate

        expected_vertices = mesh_info['original_num_vertices']

        if 'predicted_eigenvectors' in eigendata and eigendata['predicted_eigenvectors'] is not None:
            actual_vertices = eigendata['predicted_eigenvectors'].shape[0]

            if expected_vertices != actual_vertices:
                print(f"❌ ERROR: Mesh-eigendata mismatch detected!")
                print(f"  Expected vertices (from mesh): {expected_vertices}")
                print(f"  Actual vertices (from eigendata): {actual_vertices}")
                print(f"  Mesh file: {mesh_info['mesh_file_path']}")
                print(f"  Mesh index: {mesh_info['mesh_idx']}")
                print(f"  Processed vertices: {mesh_info['processed_num_vertices']}")
                return False

        if 'predicted_eigenvalues' in eigendata and eigendata['predicted_eigenvalues'] is not None:
            num_eigenvals = len(eigendata['predicted_eigenvalues'])
            print(f"✅ Consistency check passed:")
            print(f"  Mesh vertices: {expected_vertices}")
            print(f"  Eigendata vertices: {eigendata.get('matrix_size', 'unknown')}")
            print(f"  Eigenvalues computed: {num_eigenvals}")
            print(f"  Mesh file: {mesh_info['mesh_file_path']}")

        return True

    def on_validation_start(self, trainer, pl_module):
        """Clear validation results at the start of validation epoch."""
        self._validation_results = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect validation results after each batch with corresponding mesh data."""
        if outputs is None:
            return

        try:
            # STEP 1: Extract mesh info directly from batch data (CRITICAL FIX)
            mesh_info = self._extract_mesh_info_from_batch(batch)
            mesh_file_path = mesh_info['mesh_file_path']

            print(f"Processing batch {batch_idx}: {mesh_file_path.name}")
            print(f"  Original vertices: {mesh_info['original_num_vertices']}")
            print(f"  Processed vertices: {mesh_info['processed_num_vertices']}")

        except Exception as e:
            print(f"❌ Error extracting mesh info from batch {batch_idx}: {e}")
            return

        # STEP 2: Load and cache mesh data
        mesh_data = self._load_and_cache_mesh(mesh_file_path)
        if mesh_data is None:
            print(f"❌ Could not load mesh data for batch {batch_idx}")
            return

        # STEP 3: Extract eigendata AND predicted Laplacian matrix from model
        eigendata = {}
        predicted_laplacian = None

        if hasattr(pl_module, '_last_eigenvalues') and hasattr(pl_module, '_last_eigenvectors'):
            eigendata = {
                'predicted_eigenvalues': pl_module._last_eigenvalues,
                'predicted_eigenvectors': pl_module._last_eigenvectors,
                'num_eigenvalues': len(pl_module._last_eigenvalues) if pl_module._last_eigenvalues is not None else 0,
                'matrix_size': pl_module._last_eigenvectors.shape[0] if pl_module._last_eigenvectors is not None else 0
            }

        # CRITICAL ADDITION: Extract predicted Laplacian matrix
        if hasattr(pl_module, '_last_laplacian_matrix'):
            predicted_laplacian = pl_module._last_laplacian_matrix
            print(f"  📊 Found predicted Laplacian matrix: {predicted_laplacian.shape} sparse matrix")
            print(f"      Non-zero entries: {predicted_laplacian.nnz}")
            print(f"      Sparsity: {(1 - predicted_laplacian.nnz / (predicted_laplacian.shape[0] * predicted_laplacian.shape[1])) * 100:.2f}%")
        else:
            print(f"  ⚠️  No predicted Laplacian matrix found in model")

        # STEP 4: CRITICAL VALIDATION - Check mesh-eigendata consistency
        if not self._validate_mesh_eigendata_consistency(mesh_info, eigendata):
            print(f"⚠️  Skipping validation result for batch {batch_idx} due to mesh-eigendata inconsistency")
            return

        # STEP 5: Store validated result with enhanced metadata
        batch_results = {
            'batch_idx': batch_idx,
            'epoch': pl_module.current_epoch,
            'rank': trainer.global_rank,
            'metrics': {},
            'eigendata': eigendata,

            # Enhanced mesh data with validation metadata
            'mesh_data': {
                'mesh_file_path': str(mesh_file_path),
                'num_vertices': mesh_data['num_vertices'],
                'num_faces': mesh_data['num_faces'],
                'vertices': torch.from_numpy(mesh_data['vertices']).float(),
                'vertex_normals': torch.from_numpy(mesh_data['vertex_normals']).float(),
                'faces': torch.from_numpy(mesh_data['faces']).long(),

                # Ground-truth eigendecomposition for this specific mesh
                'gt_eigenvalues': torch.from_numpy(mesh_data['gt_eigenvalues']).float(),
                'gt_eigenvectors': torch.from_numpy(mesh_data['gt_eigenvectors']).float(),
                'vertex_areas': torch.from_numpy(mesh_data['vertex_areas']).float(),
                'gt_num_eigenvalues': mesh_data['gt_num_eigenvalues'],

                # Ground-truth mean curvature computed with libigl
                'gt_mean_curvature': torch.from_numpy(mesh_data['gt_mean_curvature']).float() if mesh_data['gt_mean_curvature'] is not None else None,

                # VALIDATION METADATA (NEW)
                'mesh_idx': mesh_info['mesh_idx'],
                'processed_vertices': mesh_info['processed_num_vertices'],
                'k_neighbors': mesh_info['k_neighbors'],
                'validation_status': 'consistent',
                'batch_total_points': mesh_info['total_points']
            }
        }

        # Extract metrics from outputs
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                batch_results['metrics'][key] = value.detach().cpu().numpy()
            else:
                batch_results['metrics'][key] = value

        # Convert eigendata to PyTorch tensors if needed
        if eigendata:
            if 'predicted_eigenvalues' in eigendata and eigendata['predicted_eigenvalues'] is not None:
                if isinstance(eigendata['predicted_eigenvalues'], np.ndarray):
                    batch_results['eigendata']['predicted_eigenvalues'] = torch.from_numpy(eigendata['predicted_eigenvalues']).float()
            if 'predicted_eigenvectors' in eigendata and eigendata['predicted_eigenvectors'] is not None:
                if isinstance(eigendata['predicted_eigenvectors'], np.ndarray):
                    batch_results['eigendata']['predicted_eigenvectors'] = torch.from_numpy(eigendata['predicted_eigenvectors']).float()

        # CRITICAL ADDITION: Store predicted Laplacian matrix
        if predicted_laplacian is not None:
            # Convert scipy sparse matrix to a serializable format
            laplacian_data = {
                'matrix_format': 'csr',  # Compressed Sparse Row format
                'shape': predicted_laplacian.shape,
                'data': predicted_laplacian.data.copy(),  # Non-zero values
                'indices': predicted_laplacian.indices.copy(),  # Column indices
                'indptr': predicted_laplacian.indptr.copy(),  # Row pointers
                'nnz': predicted_laplacian.nnz,  # Number of non-zero entries
                'dtype': str(predicted_laplacian.dtype)
            }
            batch_results['predicted_laplacian'] = laplacian_data
            print(f"  💾 Stored predicted Laplacian matrix ({predicted_laplacian.shape[0]}x{predicted_laplacian.shape[1]}, {predicted_laplacian.nnz} non-zeros)")
        else:
            batch_results['predicted_laplacian'] = None
            print(f"  📋 No predicted Laplacian matrix to store")

        self._validation_results.append(batch_results)
        print(f"✅ Successfully stored validation result for batch {batch_idx}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Upload validation data as W&B artifacts at the end of validation epoch."""
        if not self._validation_results:
            print("No validation results to upload")
            return

        # Create directory for artifacts
        artifacts_dir = Path("validation_artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        epoch = pl_module.current_epoch
        epoch_dir = artifacts_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)

        rank = trainer.global_rank

        # Convert validation results to use PyTorch tensors
        converted_results = []
        for batch_result in self._validation_results:
            converted_result = {
                'batch_idx': batch_result['batch_idx'],
                'epoch': batch_result['epoch'],
                'rank': batch_result['rank'],
                'metrics': {},
                'eigendata': {},
                'mesh_data': batch_result['mesh_data'],  # Already converted to tensors in on_validation_batch_end
                'predicted_laplacian': batch_result.get('predicted_laplacian', None)  # Add Laplacian data
            }

            # Convert metrics to PyTorch tensors
            for key, value in batch_result['metrics'].items():
                if isinstance(value, np.ndarray):
                    converted_result['metrics'][key] = torch.from_numpy(value).float()
                elif isinstance(value, (int, float)):
                    converted_result['metrics'][key] = torch.tensor(value).float()
                else:
                    converted_result['metrics'][key] = value

            # Convert eigendata to PyTorch tensors (with proper type checking)
            if 'eigendata' in batch_result and batch_result['eigendata']:
                eigendata = batch_result['eigendata']

                # Handle predicted_eigenvalues
                if 'predicted_eigenvalues' in eigendata and eigendata['predicted_eigenvalues'] is not None:
                    pred_eigenvals = eigendata['predicted_eigenvalues']
                    if isinstance(pred_eigenvals, np.ndarray):
                        converted_result['eigendata']['predicted_eigenvalues'] = torch.from_numpy(pred_eigenvals).float()
                    elif torch.is_tensor(pred_eigenvals):
                        converted_result['eigendata']['predicted_eigenvalues'] = pred_eigenvals.float()
                    else:
                        print(f"Warning: Unexpected type for predicted_eigenvalues: {type(pred_eigenvals)}")
                        converted_result['eigendata']['predicted_eigenvalues'] = pred_eigenvals

                # Handle predicted_eigenvectors
                if 'predicted_eigenvectors' in eigendata and eigendata['predicted_eigenvectors'] is not None:
                    pred_eigenvecs = eigendata['predicted_eigenvectors']
                    if isinstance(pred_eigenvecs, np.ndarray):
                        converted_result['eigendata']['predicted_eigenvectors'] = torch.from_numpy(pred_eigenvecs).float()
                    elif torch.is_tensor(pred_eigenvecs):
                        converted_result['eigendata']['predicted_eigenvectors'] = pred_eigenvecs.float()
                    else:
                        print(f"Warning: Unexpected type for predicted_eigenvectors: {type(pred_eigenvecs)}")
                        converted_result['eigendata']['predicted_eigenvectors'] = pred_eigenvecs

                # Copy other eigendata fields
                converted_result['eigendata']['num_eigenvalues'] = eigendata.get('num_eigenvalues', 0)
                converted_result['eigendata']['matrix_size'] = eigendata.get('matrix_size', 0)

            converted_results.append(converted_result)

        # Create the final data structure
        combined_data = {
            'epoch': epoch,
            'rank': rank,
            'num_batches': len(converted_results),
            'validation_results': converted_results,
            # Note: mesh data is now stored per validation result, not globally
        }

        # Save everything in a single pickle file
        combined_pickle_name = f"rank_{rank}_validation_data.pkl"
        combined_pickle_path = epoch_dir / combined_pickle_name

        with open(combined_pickle_path, 'wb') as f:
            pickle.dump(combined_data, f)

        print(f"Saved validation data to {combined_pickle_path}")
        print(f"Data includes {len(converted_results)} validation results, each with its own mesh data")

        # Synchronize all processes if using distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Only rank 0 uploads to W&B
        if trainer.global_rank == 0 and wandb.run is not None:
            self._upload_to_wandb(epoch, epoch_dir, artifacts_dir, len(converted_results))

        # Clear results for next epoch
        self._validation_results = []

    def _upload_to_wandb(self, epoch: int, epoch_dir: Path, artifacts_dir: Path, num_results: int):
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
                'num_validation_results': num_results,
                'eigenvalue_computation_k': self._k,
                'mesh_data_per_result': True,  # Flag indicating new format
                'includes_predicted_laplacian': True,  # New flag for Laplacian matrices
                'includes_gt_mean_curvature': True,  # New flag for GT mean curvature
                'description': 'Each validation result contains its own mesh data, eigendecomposition, predicted Laplacian matrix, and GT mean curvature'
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