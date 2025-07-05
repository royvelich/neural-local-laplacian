#!/usr/bin/env python3
"""
Real-time Eigenanalysis Visualization with MeshDataset and Model Inference

This script:
1. Uses Hydra to instantiate a MeshDataset from config
2. Loads a trained SurfaceTransformerModule from checkpoint
3. For each mesh, performs real-time inference to get predicted Laplacian
4. Computes ground-truth quantities using the same methods as visualize_validation.py
5. Applies normal orientation correction and creates corrected predictions
6. Visualizes comprehensive comparison of GT vs Original vs Corrected predictions

Usage:
    python visualize_validation2.py --config-path=config --config-name=mesh_config ckpt_path=path/to/model.ckpt
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import sys

import numpy as np
import torch
import polyscope as ps
import open3d as o3d
import trimesh

# For GT eigendecomposition computation
from pyFM.mesh import TriMesh

# For GT mean curvature computation
try:
    import igl

    HAS_IGL = True
except ImportError:
    HAS_IGL = False
    print("Warning: libigl not available, GT mean curvature will not be computed")

# For eigendecomposition
import scipy.sparse
import scipy.sparse.linalg

# Hydra
import hydra
from omegaconf import DictConfig

# PyTorch Lightning
import pytorch_lightning as pl

# Local imports
from neural_local_laplacian.modules.laplacian_modules import SurfaceTransformerModule
from neural_local_laplacian.utils import utils
from torch_geometric.data import Data, Batch


@dataclass
class VisualizationConfig:
    """Configuration for eigenanalysis visualization."""
    point_radius: float = 0.005
    show_wireframe: bool = False
    colormap: str = 'coolwarm'
    num_eigenvectors_to_show: int = 8
    enable_eigenvalue_info: bool = True
    enable_correlation_analysis: bool = True


class ColorPalette:
    """Color palette for different visualization elements."""

    # Normal comparison colors
    GT_NORMALS = (0.0, 1.0, 1.0)  # Cyan for ground truth
    PREDICTED_NORMALS = (1.0, 0.5, 0.0)  # Orange for predictions

    # Default colors
    DEFAULT_VECTOR = (0.5, 0.5, 0.5)  # Gray

    @classmethod
    def get_vector_color(cls, vector_name: str) -> Tuple[float, float, float]:
        """Get color for vector visualization."""
        color_map = {
            'gt_normals': cls.GT_NORMALS,
            'predicted_normals': cls.PREDICTED_NORMALS
        }
        return color_map.get(vector_name, cls.DEFAULT_VECTOR)


class VectorScales:
    """Scale factors for different vector visualizations."""

    # Mean curvature vector scales
    GT_MEAN_CURVATURE_VECTOR = 0.005
    PREDICTED_MEAN_CURVATURE_VECTOR = 0.005

    # Normal vector scales
    GT_NORMALS = 0.05
    PREDICTED_NORMALS = 0.05

    # Default scale
    DEFAULT_VECTOR = 0.05

    @classmethod
    def get_vector_scale(cls, vector_name: str) -> float:
        """Get scale factor for vector visualization."""
        scale_map = {
            'gt_mean_curvature_vector': cls.GT_MEAN_CURVATURE_VECTOR,
            'predicted_mean_curvature_vector': cls.PREDICTED_MEAN_CURVATURE_VECTOR,
            'gt_normals': cls.GT_NORMALS,
            'predicted_normals': cls.PREDICTED_NORMALS
        }
        return scale_map.get(vector_name, cls.DEFAULT_VECTOR)


class RealTimeEigenanalysisVisualizer:
    """Real-time eigenanalysis visualizer using MeshDataset and model inference."""

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.color_palette = ColorPalette()
        self.vector_scales = VectorScales()
        self.training_k = None  # Will be set from command line argument

    def setup_polyscope(self):
        """Initialize and configure polyscope."""
        ps.init()
        ps.set_up_dir("z_up")
        ps.look_at(camera_location=[2.0, 2.0, 2.0], target=[0, 0, 0])
        ps.set_ground_plane_mode("none")
        ps.set_background_color((0.05, 0.05, 0.05))  # Dark background

    def load_trained_model(self, ckpt_path: Path, device: torch.device, cfg: DictConfig) -> SurfaceTransformerModule:
        """
        Load trained SurfaceTransformerModule from checkpoint.

        Args:
            ckpt_path: Path to the checkpoint file
            device: Device to load the model on
            cfg: Hydra config containing model configuration

        Returns:
            Loaded model in evaluation mode
        """
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        try:
            print(f"Loading model checkpoint from: {ckpt_path}")

            # Extract model arguments from config
            model_cfg = cfg.model.module

            # Load model from checkpoint with model arguments from config
            model = SurfaceTransformerModule.load_from_checkpoint(
                str(ckpt_path),
                map_location=device,
                input_dim=model_cfg.input_dim,
                d_model=model_cfg.d_model,
                nhead=model_cfg.nhead,
                num_encoder_layers=model_cfg.num_encoder_layers,
                dim_feedforward=model_cfg.dim_feedforward,
                num_eigenvalues=model_cfg.num_eigenvalues,
                dropout=model_cfg.dropout,
                loss_configs=hydra.utils.instantiate(model_cfg.loss_configs) if hasattr(model_cfg, 'loss_configs') else None,
                optimizer_cfg=None  # Not needed for inference
            )

            model.eval()
            model.to(device)

            print(f"✅ Model loaded successfully on {device}")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Input dim: {model._input_dim}")
            print(f"   Model dim: {model._d_model}")
            print(f"   Num eigenvalues: {model._num_eigenvalues}")

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {ckpt_path}: {e}")

    def assemble_sparse_laplacian(self, weights: torch.Tensor, vertex_indices: torch.Tensor,
                                  center_indices: torch.Tensor, batch_indices: torch.Tensor) -> scipy.sparse.csr_matrix:
        """
        Assemble sparse Laplacian matrix from patch weights using vectorized operations.
        Copied from SurfaceTransformerModule._assemble_sparse_laplacian().

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

    def compute_eigendecomposition(self, laplacian_matrix: scipy.sparse.csr_matrix, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of the Laplacian matrix using shift-invert mode.
        Copied from SurfaceTransformerModule._compute_eigendecomposition().

        Args:
            laplacian_matrix: Sparse positive semi-definite Laplacian matrix
            k: Number of eigenvalues to compute

        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted in ascending order by eigenvalue.
            - eigenvalues: Array of shape (k,) with smallest k eigenvalues
            - eigenvectors: Array of shape (n, k) with corresponding eigenvectors
        """
        try:
            # Use shift-invert mode with sigma=-0.01 to find eigenvalues closest to 0
            # This is robust for positive semi-definite Laplacian matrices
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(laplacian_matrix, k=k, sigma=-0.01)

            # Sort eigenvalues and eigenvectors in ascending order
            sort_indices = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sort_indices]
            eigenvectors = eigenvectors[:, sort_indices]

            # Ensure predicted eigenvectors are normalized
            if eigenvectors is not None:
                # Normalize each eigenvector to unit length
                eigenvectors_norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
                # Avoid division by zero
                eigenvectors_norms = np.where(eigenvectors_norms > 1e-10, eigenvectors_norms, 1.0)
                eigenvectors = eigenvectors / eigenvectors_norms

                print(f"Predicted eigenvectors normalized to unit length")

            return eigenvalues, eigenvectors

        except Exception as e:
            print(f"Error computing eigendecomposition: {e}")
            return None, None

    def load_original_mesh_for_gt(self, mesh_file_path: str) -> Dict[str, Any]:
        """
        Load original mesh and compute ground-truth data using the same methods as ValidationMeshUploader.

        Args:
            mesh_file_path: Path to the mesh file

        Returns:
            Dictionary containing all GT quantities
        """
        print(f"Loading mesh for GT computation: {Path(mesh_file_path).name}")

        try:
            # Load mesh using trimesh
            mesh = trimesh.load(str(mesh_file_path))
            raw_vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)

            # Apply the same normalization as MeshDataset
            vertices = utils.normalize_mesh_vertices(raw_vertices)

            # Update mesh with normalized vertices for normal computation
            mesh.vertices = vertices
            gt_vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)

            print(f"Mesh has {len(vertices)} vertices and {len(faces)} faces")
            print(f"Normalized vertices: center at origin, max distance = {np.linalg.norm(vertices, axis=1).max():.6f}")

        except Exception as e:
            raise RuntimeError(f"Failed to load mesh {mesh_file_path}: {e}")

        # Compute GT mean curvature using libigl if available
        gt_mean_curvature = None
        gt_mean_curvature_vector = None
        if HAS_IGL:
            try:
                print("Computing GT mean curvature using libigl...")
                # Convert to float64 for libigl (more stable)
                vertices_igl = vertices.astype(np.float64)
                faces_igl = faces.astype(np.int32)

                # Compute principal curvatures using libigl
                _, _, principal_curvature1, principal_curvature2, _ = igl.principal_curvature(
                    vertices_igl, faces_igl
                )

                # Mean curvature is the average of principal curvatures: H = (k1 + k2) / 2
                gt_mean_curvature = (principal_curvature1 + principal_curvature2) / 2.0
                gt_mean_curvature = gt_mean_curvature.astype(np.float32)

                # GT mean curvature vector = GT normal * GT mean curvature
                gt_mean_curvature_vector = gt_vertex_normals * gt_mean_curvature[:, np.newaxis]

                print(f"GT mean curvature range: [{gt_mean_curvature.min():.6f}, {gt_mean_curvature.max():.6f}]")

            except Exception as e:
                print(f"Warning: Failed to compute GT mean curvature with libigl: {e}")
                gt_mean_curvature = None
                gt_mean_curvature_vector = None

        # Compute GT Laplacian eigendecomposition using PyFM
        print("Computing GT Laplacian eigendecomposition using PyFM...")
        try:
            # Create PyFM TriMesh object
            pyfm_mesh = TriMesh(vertices, faces)

            # Process the mesh and compute the Laplacian spectrum
            pyfm_mesh.process(k=self.config.num_eigenvectors_to_show, intrinsic=False, verbose=False)

            # Retrieve eigenvalues, eigenfunctions, and vertex areas
            gt_eigenvalues = pyfm_mesh.eigenvalues
            gt_eigenvectors = pyfm_mesh.eigenvectors
            vertex_areas = pyfm_mesh.vertex_areas

            # Ensure GT eigenvectors are normalized
            if gt_eigenvectors is not None:
                # Normalize each eigenvector to unit length
                gt_eigenvectors_norms = np.linalg.norm(gt_eigenvectors, axis=0, keepdims=True)
                # Avoid division by zero
                gt_eigenvectors_norms = np.where(gt_eigenvectors_norms > 1e-10, gt_eigenvectors_norms, 1.0)
                gt_eigenvectors = gt_eigenvectors / gt_eigenvectors_norms

                print(f"GT eigenvectors normalized to unit length")

            print(f"Computed {len(gt_eigenvalues)} GT eigenvalues")
            print(f"GT eigenvalue range: [{gt_eigenvalues[0]:.2e}, {gt_eigenvalues[-1]:.6f}]")

        except Exception as e:
            print(f"Warning: Failed to compute GT eigendecomposition: {e}")
            gt_eigenvalues = None
            gt_eigenvectors = None
            vertex_areas = None

        return {
            'vertices': vertices,
            'faces': faces,
            'gt_vertex_normals': gt_vertex_normals,
            'gt_mean_curvature': gt_mean_curvature,
            'gt_mean_curvature_vector': gt_mean_curvature_vector,  # New: GT mean curvature vector
            'gt_eigenvalues': gt_eigenvalues,
            'gt_eigenvectors': gt_eigenvectors,
            'vertex_areas': vertex_areas
        }

    def perform_model_inference(self, model: SurfaceTransformerModule, batch_data: Data, device: torch.device) -> Dict[str, Any]:
        """
        Perform model inference and compute predicted quantities.

        Args:
            model: Trained SurfaceTransformerModule
            batch_data: Preprocessed batch data from MeshDataset
            device: Device for computation

        Returns:
            Dictionary containing predicted quantities and Laplacian matrix
        """
        print("Performing model inference...")

        # Move batch data to device
        batch_data = batch_data.to(device)

        with torch.no_grad():
            # Forward pass to get token weights
            forward_result = model._forward_pass(batch_data)
            token_weights = forward_result['token_weights']

            print(f"Got token weights shape: {token_weights.shape}")

            # Apply k-ratio correction if training k differs from inference k
            if self.training_k is not None:
                inference_k = token_weights.shape[1]  # Number of points per patch
                if inference_k != self.training_k:
                    k_ratio = inference_k / self.training_k
                    token_weights = token_weights / k_ratio
                    print(f"Applied k-ratio correction: inference_k={inference_k}, training_k={self.training_k}, ratio={k_ratio:.3f}")

            # Assemble sparse Laplacian matrix
            laplacian_matrix = self.assemble_sparse_laplacian(
                weights=token_weights,
                vertex_indices=batch_data.vertex_indices,
                center_indices=batch_data.center_indices,
                batch_indices=batch_data.batch
            )

            print(f"Assembled Laplacian matrix: {laplacian_matrix.shape} ({laplacian_matrix.nnz} non-zeros)")

            # Compute eigendecomposition
            predicted_eigenvalues, predicted_eigenvectors = self.compute_eigendecomposition(
                laplacian_matrix, k=self.config.num_eigenvectors_to_show
            )

            if predicted_eigenvalues is not None:
                print(f"Computed {len(predicted_eigenvalues)} predicted eigenvalues")
                print(f"Predicted eigenvalue range: [{predicted_eigenvalues[0]:.2e}, {predicted_eigenvalues[-1]:.6f}]")

        return {
            'laplacian_matrix': laplacian_matrix,
            'predicted_eigenvalues': predicted_eigenvalues,
            'predicted_eigenvectors': predicted_eigenvectors,
            'token_weights': token_weights.cpu().numpy()
        }

    def compute_predicted_quantities_from_laplacian(self, laplacian_matrix: scipy.sparse.csr_matrix,
                                                    vertices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute predicted mean curvature vector and derived quantities from predicted Laplacian matrix.

        Args:
            laplacian_matrix: Predicted Laplacian matrix (scipy sparse)
            vertices: Mesh vertices array of shape (N, 3)

        Returns:
            Dictionary containing predicted quantities
        """
        print("Computing predicted quantities from Laplacian...")

        try:
            # Apply Laplacian to vertex positions: Δr = L @ r
            predicted_mean_curvature_vector = laplacian_matrix @ vertices  # Shape: (N, 3)

            # Compute magnitudes (predicted mean curvature values)
            predicted_mean_curvature = np.linalg.norm(predicted_mean_curvature_vector, axis=1)  # Shape: (N,)

            # Compute predicted normals (normalized mean curvature vectors)
            predicted_normals = np.zeros_like(predicted_mean_curvature_vector)
            non_zero_mask = predicted_mean_curvature > 1e-10

            predicted_normals[non_zero_mask] = (
                    predicted_mean_curvature_vector[non_zero_mask] /
                    predicted_mean_curvature[non_zero_mask, np.newaxis]
            )

            # For zero curvature points, use a default normal (e.g., z-up)
            predicted_normals[~non_zero_mask] = np.array([0, 0, 1])

            print(f"Predicted mean curvature range: [{predicted_mean_curvature.min():.6f}, {predicted_mean_curvature.max():.6f}]")
            print(f"Zero curvature points: {(~non_zero_mask).sum()}/{len(predicted_mean_curvature)}")

            return {
                'predicted_mean_curvature_vector': predicted_mean_curvature_vector,  # Raw, unnormalized vectors
                'predicted_normals': predicted_normals,
                'predicted_mean_curvature': predicted_mean_curvature
            }

        except Exception as e:
            print(f"Error computing predicted quantities from Laplacian: {e}")
            return {}

    def compute_mesh_reconstruction(self, original_vertices: np.ndarray, eigenvectors: np.ndarray,
                                    eigenvalues: np.ndarray, max_eigenvectors: int,
                                    vertex_areas: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Reconstruct mesh geometry using progressive number of eigenvectors.

        Args:
            original_vertices: Original mesh vertices of shape (N, 3)
            eigenvectors: Eigenvectors of shape (N, k)
            eigenvalues: Eigenvalues of shape (k,)
            max_eigenvectors: Maximum number of eigenvectors to use
            vertex_areas: Optional vertex areas for area-weighted computation (GT case)

        Returns:
            List of reconstructed vertex arrays, one for each number of eigenvectors (1, 2, ..., max_eigenvectors)
        """
        if eigenvectors is None or eigenvalues is None:
            return []

        num_available = min(eigenvectors.shape[1], max_eigenvectors)
        reconstructed_meshes = []

        if vertex_areas is not None:
            print(f"Computing area-weighted mesh reconstruction with up to {num_available} eigenvectors...")
            reconstructed_meshes = self._compute_area_weighted_reconstruction(
                original_vertices, eigenvectors, num_available, vertex_areas
            )
        else:
            print(f"Computing standard mesh reconstruction with up to {num_available} eigenvectors...")
            reconstructed_meshes = self._compute_standard_reconstruction(
                original_vertices, eigenvectors, num_available
            )

        print(f"Generated {len(reconstructed_meshes)} reconstructed meshes")
        return reconstructed_meshes

    def _compute_area_weighted_reconstruction(self, original_vertices: np.ndarray, eigenvectors: np.ndarray,
                                              num_available: int, vertex_areas: np.ndarray) -> List[np.ndarray]:
        """
        Compute mesh reconstruction using area-weighted least squares (for GT case).

        Solves the weighted least squares problem:
        min_c ||f - A_ℓ c||_M^2
        where M = diag(vertex_areas) is the mass matrix.

        Args:
            original_vertices: Original mesh vertices f ∈ R^{n×3}
            eigenvectors: Eigenvectors Φ ∈ R^{n×k}
            num_available: Number of available eigenvectors to use
            vertex_areas: Vertex areas a ∈ R^n

        Returns:
            List of reconstructed vertex arrays
        """
        reconstructed_meshes = []

        # Create mass matrix M = diag(vertex_areas)
        M = np.diag(vertex_areas)  # Shape: (n, n)
        f = original_vertices  # Shape: (n, 3)

        for num_eigenvecs in range(1, num_available + 1):
            # Step 1: Extract basis A_ℓ = Φ[:, 1:ℓ] (first ℓ eigenvectors)
            A_l = eigenvectors[:, :num_eigenvecs]  # Shape: (n, ℓ)

            # Step 2: Compute Gram matrix G = A_ℓ^T M A_ℓ
            G = A_l.T @ M @ A_l  # Shape: (ℓ, ℓ)

            # Step 3: Compute projection target b = A_ℓ^T M f
            b = A_l.T @ M @ f  # Shape: (ℓ, 3)

            # Step 4: Solve for coefficients G c = b
            try:
                c = np.linalg.solve(G, b)  # Shape: (ℓ, 3)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if G is singular
                print(f"Warning: Gram matrix singular for {num_eigenvecs} eigenvectors, using pseudoinverse")
                c = np.linalg.pinv(G) @ b  # Shape: (ℓ, 3)

            # Step 5: Reconstruct f̂_ℓ = A_ℓ c
            reconstructed_vertices = A_l @ c  # Shape: (n, 3)

            reconstructed_meshes.append(reconstructed_vertices)

        return reconstructed_meshes

    def _compute_standard_reconstruction(self, original_vertices: np.ndarray, eigenvectors: np.ndarray,
                                         num_available: int) -> List[np.ndarray]:
        """
        Compute mesh reconstruction using standard Euclidean inner products (for PRED case).

        Args:
            original_vertices: Original mesh vertices of shape (N, 3)
            eigenvectors: Eigenvectors of shape (N, k)
            num_available: Number of available eigenvectors to use

        Returns:
            List of reconstructed vertex arrays
        """
        reconstructed_meshes = []

        for num_eigenvecs in range(1, num_available + 1):
            # Use first num_eigenvecs eigenvectors
            current_eigenvectors = eigenvectors[:, :num_eigenvecs]  # Shape: (N, num_eigenvecs)

            # Compute standard projection coefficients
            # c_i = ⟨coords, φ_i⟩ = Σ coords_j * φ_i_j
            coefficients = np.dot(original_vertices.T, current_eigenvectors)  # Shape: (3, num_eigenvecs)

            # Reconstruct coordinates: coords = Σ c_i * φ_i
            reconstructed_vertices = np.dot(current_eigenvectors, coefficients.T)  # Shape: (N, 3)

            reconstructed_meshes.append(reconstructed_vertices)

        return reconstructed_meshes

    def compute_eigenvector_correlations(self, gt_eigenvectors: np.ndarray, pred_eigenvectors: np.ndarray) -> np.ndarray:
        """Compute correlation matrix between ground-truth and predicted eigenvectors."""
        min_cols = min(gt_eigenvectors.shape[1], pred_eigenvectors.shape[1])
        correlation_matrix = np.zeros((min_cols, min_cols))

        for i in range(min_cols):
            for j in range(min_cols):
                # Compute absolute correlation (eigenvectors can have sign ambiguity)
                corr = np.abs(np.corrcoef(gt_eigenvectors[:, i], pred_eigenvectors[:, j])[0, 1])
                correlation_matrix[i, j] = corr

        return correlation_matrix

    def visualize_mesh_reconstructions(self, original_faces: np.ndarray, gt_reconstructions: List[np.ndarray],
                                       pred_reconstructions: List[np.ndarray], gt_eigenvalues: Optional[np.ndarray],
                                       pred_eigenvalues: Optional[np.ndarray]):
        """
        Visualize progressive mesh reconstructions using eigenvectors.
        All GT reconstructions are overlaid on the right, all PRED reconstructions on the left.

        Args:
            original_faces: Mesh faces for topology
            gt_reconstructions: List of GT reconstructed vertices
            pred_reconstructions: List of predicted reconstructed vertices
            gt_eigenvalues: GT eigenvalues for labeling
            pred_eigenvalues: Predicted eigenvalues for labeling
        """
        print("Adding mesh reconstruction visualizations...")

        # Fixed positions for overlaid reconstructions
        gt_offset = np.array([3.0, 0.0, 0.0])  # GT reconstructions on the right
        pred_offset = np.array([-3.0, 0.0, 0.0])  # PRED reconstructions on the left

        # Visualize GT reconstructions (all overlaid on the right)
        for i, gt_vertices in enumerate(gt_reconstructions):
            num_eigenvecs = i + 1
            gt_eigenval = gt_eigenvalues[i] if gt_eigenvalues is not None else 0.0

            # Position all GT reconstructions at the same location (right side)
            offset_vertices = gt_vertices + gt_offset

            mesh_name = f"GT Recon {num_eigenvecs:02d} eigenvec (λ={gt_eigenval:.3f})"

            try:
                gt_mesh = ps.register_surface_mesh(
                    name=mesh_name,
                    vertices=offset_vertices,
                    faces=original_faces,
                    enabled=(i == 0)  # Only enable the first one by default
                )

                # Color GT reconstructions in blue tones with slight variation
                blue_intensity = 0.5 + 0.5 * (i / max(1, len(gt_reconstructions) - 1))
                mesh_color = np.array([0.2, 0.4, blue_intensity])
                vertex_colors = np.tile(mesh_color, (len(offset_vertices), 1))
                gt_mesh.add_color_quantity(
                    name="gt_color",
                    values=vertex_colors,
                    enabled=True
                )

            except Exception as e:
                print(f"Warning: Failed to visualize GT reconstruction {num_eigenvecs}: {e}")

        # Visualize predicted reconstructions (all overlaid on the left)
        for i, pred_vertices in enumerate(pred_reconstructions):
            num_eigenvecs = i + 1
            pred_eigenval = pred_eigenvalues[i] if pred_eigenvalues is not None else 0.0

            # Position all PRED reconstructions at the same location (left side)
            offset_vertices = pred_vertices + pred_offset

            mesh_name = f"PRED Recon {num_eigenvecs:02d} eigenvec (λ={pred_eigenval:.3f})"

            try:
                pred_mesh = ps.register_surface_mesh(
                    name=mesh_name,
                    vertices=offset_vertices,
                    faces=original_faces,
                    enabled=(i == 0)  # Only enable the first one by default
                )

                # Color predicted reconstructions in orange tones with slight variation
                orange_intensity = 0.5 + 0.5 * (i / max(1, len(pred_reconstructions) - 1))
                mesh_color = np.array([orange_intensity, 0.4, 0.2])
                vertex_colors = np.tile(mesh_color, (len(offset_vertices), 1))
                pred_mesh.add_color_quantity(
                    name="pred_color",
                    values=vertex_colors,
                    enabled=True
                )

            except Exception as e:
                print(f"Warning: Failed to visualize predicted reconstruction {num_eigenvecs}: {e}")

        print(f"Added {len(gt_reconstructions)} GT and {len(pred_reconstructions)} predicted mesh reconstructions")
        print("Toggle visibility to compare different numbers of eigenvectors")

    def print_eigenvalue_analysis(self, gt_eigenvalues: Optional[np.ndarray],
                                  predicted_eigenvalues: Optional[np.ndarray],
                                  mesh_name: str):
        """Print detailed eigenvalue comparison analysis."""
        print(f"\n" + "-" * 70)
        print(f"EIGENVALUE COMPARISON ANALYSIS - {mesh_name}")
        print("-" * 70)

        # Ground-truth analysis
        if gt_eigenvalues is not None:
            print("GROUND-TRUTH EIGENVALUES:")
            print(f"  Number of eigenvalues: {len(gt_eigenvalues)}")
            print(f"  First eigenvalue (should be ~0): {gt_eigenvalues[0]:.2e}")
            if len(gt_eigenvalues) > 1:
                print(f"  Second eigenvalue (Fiedler): {gt_eigenvalues[1]:.6f}")
                print(f"  Spectral gap: {gt_eigenvalues[1] - gt_eigenvalues[0]:.6f}")
            print(f"  Largest eigenvalue: {gt_eigenvalues[-1]:.6f}")

        # Predicted analysis
        if predicted_eigenvalues is not None:
            print("\nPREDICTED EIGENVALUES:")
            print(f"  Number of eigenvalues: {len(predicted_eigenvalues)}")
            print(f"  First eigenvalue (should be ~0): {predicted_eigenvalues[0]:.2e}")
            if len(predicted_eigenvalues) > 1:
                print(f"  Second eigenvalue (Fiedler): {predicted_eigenvalues[1]:.6f}")
                print(f"  Spectral gap: {predicted_eigenvalues[1] - predicted_eigenvalues[0]:.6f}")
            print(f"  Largest eigenvalue: {predicted_eigenvalues[-1]:.6f}")

        # Comparison metrics
        if gt_eigenvalues is not None and predicted_eigenvalues is not None:
            min_len = min(len(gt_eigenvalues), len(predicted_eigenvalues))
            if min_len > 0:
                gt_subset = gt_eigenvalues[:min_len]
                pred_subset = predicted_eigenvalues[:min_len]

                abs_errors = np.abs(pred_subset - gt_subset)
                rel_errors = abs_errors / (np.abs(gt_subset) + 1e-10)

                print(f"\nPREDICTED VS GT COMPARISON:")
                print(f"  Mean absolute error: {abs_errors.mean():.6f}")
                print(f"  Max absolute error: {abs_errors.max():.6f}")
                print(f"  Correlation coefficient: {np.corrcoef(gt_subset, pred_subset)[0, 1]:.6f}")

        print("-" * 70)

    def print_eigenvector_correlation_analysis(self, correlation_matrix: np.ndarray, comparison_name: str):
        """Print eigenvector correlation analysis."""
        print(f"\nEIGENVECTOR CORRELATION ANALYSIS ({comparison_name})")
        print("-" * 50)

        # Find best matches for each GT eigenvector
        print("Best matches for each GT eigenvector:")
        for i in range(min(8, correlation_matrix.shape[0])):
            best_match_idx = np.argmax(correlation_matrix[i, :])
            best_correlation = correlation_matrix[i, best_match_idx]
            print(f"  GT Eigenvector {i} ↔ Pred Eigenvector {best_match_idx}: {best_correlation:.4f}")

        # Overall statistics
        diagonal_corrs = np.diag(correlation_matrix)
        print(f"\nDiagonal correlations (GT_i vs PRED_i):")
        print(f"  Mean: {diagonal_corrs.mean():.4f}")
        print(f"  Min: {diagonal_corrs.min():.4f}")
        print(f"  Max: {diagonal_corrs.max():.4f}")

        well_aligned = diagonal_corrs > 0.8
        print(f"  Well-aligned (>0.8): {well_aligned.sum()}/{len(diagonal_corrs)}")

    def visualize_mesh(self, vertices: np.ndarray, vertex_normals: np.ndarray, faces: np.ndarray):
        """Visualize the base mesh."""
        # Register mesh as surface mesh
        if len(faces) > 0:
            mesh_surface = ps.register_surface_mesh(
                name="Mesh Surface",
                vertices=vertices,
                faces=faces,
                enabled=True
            )

            print(f"Registered mesh surface with {len(vertices)} vertices and {len(faces)} faces")
            return mesh_surface
        else:
            # Fallback to point cloud if no faces
            mesh_cloud = ps.register_point_cloud(
                name="Mesh",
                points=vertices,
                radius=self.config.point_radius,
                enabled=True
            )

            print(f"Registered mesh point cloud with {len(vertices)} vertices")
            return mesh_cloud

    def add_comprehensive_curvature_visualizations(self, mesh_structure,
                                                   gt_data: Dict[str, np.ndarray],
                                                   predicted_data: Dict[str, np.ndarray]):
        """Add comprehensive curvature and normal visualizations to the mesh."""
        print("Adding comprehensive curvature visualizations...")

        # === MEAN CURVATURE ===
        if gt_data.get('gt_mean_curvature') is not None:
            mesh_structure.add_scalar_quantity(
                name="A Mean Curvature - GT",
                values=gt_data['gt_mean_curvature'],
                enabled=False,
                cmap='plasma'
            )

        if predicted_data.get('predicted_mean_curvature') is not None:
            mesh_structure.add_scalar_quantity(
                name="B Mean Curvature - PRED",
                values=predicted_data['predicted_mean_curvature'],
                enabled=False,
                cmap='plasma'
            )

        # === MEAN CURVATURE VECTORS (RAW, UNNORMALIZED) ===
        if gt_data.get('gt_mean_curvature_vector') is not None:
            mesh_structure.add_vector_quantity(
                name="C Mean Curvature Vector - GT",
                values=gt_data['gt_mean_curvature_vector'] * self.vector_scales.get_vector_scale('gt_mean_curvature_vector'),
                enabled=False,
                color=(0.0, 1.0, 1.0),  # Cyan
                vectortype="ambient"
            )

        if predicted_data.get('predicted_mean_curvature_vector') is not None:
            mesh_structure.add_vector_quantity(
                name="D Mean Curvature Vector - PRED",
                values=predicted_data['predicted_mean_curvature_vector'] * self.vector_scales.get_vector_scale('predicted_mean_curvature_vector'),
                enabled=False,
                color=(1.0, 0.5, 0.0),  # Orange
                vectortype="ambient"
            )

        # === COMPARISON METRICS FOR MEAN CURVATURE VECTORS ===
        gt_mean_curv_vector = gt_data.get('gt_mean_curvature_vector')
        pred_mean_curv_vector = predicted_data.get('predicted_mean_curvature_vector')

        # Mean curvature vector alignment comparison
        if gt_mean_curv_vector is not None and pred_mean_curv_vector is not None:
            # Normalize both vectors for alignment comparison
            gt_norm = np.linalg.norm(gt_mean_curv_vector, axis=1, keepdims=True)
            pred_norm = np.linalg.norm(pred_mean_curv_vector, axis=1, keepdims=True)

            # Avoid division by zero
            gt_normalized = np.where(gt_norm > 1e-10, gt_mean_curv_vector / gt_norm, 0)
            pred_normalized = np.where(pred_norm > 1e-10, pred_mean_curv_vector / pred_norm, 0)

            vector_alignment = np.sum(pred_normalized * gt_normalized, axis=1)
            mesh_structure.add_scalar_quantity(
                name="E Mean Curvature Vector Alignment",
                values=vector_alignment,
                enabled=False,
                cmap='coolwarm'
            )

    def visualize_comprehensive_eigenvectors(self, mesh_structure,
                                             gt_eigenvalues: Optional[np.ndarray], gt_eigenvectors: Optional[np.ndarray],
                                             pred_eigenvalues: Optional[np.ndarray], pred_eigenvectors: Optional[np.ndarray]):
        """Add GT and predicted eigenvector scalar fields to the mesh."""
        num_to_show = self.config.num_eigenvectors_to_show

        # Determine how many eigenvectors we can show for each type
        gt_available = gt_eigenvectors.shape[1] if gt_eigenvectors is not None else 0
        pred_available = pred_eigenvectors.shape[1] if pred_eigenvectors is not None else 0

        max_available = max(gt_available, pred_available)
        num_to_show = min(num_to_show, max_available)

        print(f"Adding eigenvector visualization:")
        print(f"  GT eigenvectors available: {gt_available}")
        print(f"  Predicted eigenvectors available: {pred_available}")
        print(f"  Showing: {num_to_show} eigenvectors per type")

        # Add eigenvectors in groups
        for i in range(num_to_show):
            # Add GT eigenvector
            if gt_eigenvectors is not None and i < gt_available:
                gt_eigenvector = gt_eigenvectors[:, i]
                gt_eigenvalue = gt_eigenvalues[i] if gt_eigenvalues is not None else 0.0

                if i == 0:
                    gt_name = f"Eigenvector {i:02d}a GT (λ={gt_eigenvalue:.2e}, constant)"
                elif i == 1:
                    gt_name = f"Eigenvector {i:02d}a GT (λ={gt_eigenvalue:.6f}, Fiedler)"
                else:
                    gt_name = f"Eigenvector {i:02d}a GT (λ={gt_eigenvalue:.6f})"

                mesh_structure.add_scalar_quantity(
                    name=gt_name,
                    values=gt_eigenvector,
                    enabled=(i == 1),  # Enable GT Fiedler vector by default
                    cmap=self.config.colormap
                )

            # Add predicted eigenvector
            if pred_eigenvectors is not None and i < pred_available:
                pred_eigenvector = pred_eigenvectors[:, i]
                pred_eigenvalue = pred_eigenvalues[i] if pred_eigenvalues is not None else 0.0

                if i == 0:
                    pred_name = f"Eigenvector {i:02d}b PRED (λ={pred_eigenvalue:.2e}, constant)"
                elif i == 1:
                    pred_name = f"Eigenvector {i:02d}b PRED (λ={pred_eigenvalue:.6f}, Fiedler)"
                else:
                    pred_name = f"Eigenvector {i:02d}b PRED (λ={pred_eigenvalue:.6f})"

                mesh_structure.add_scalar_quantity(
                    name=pred_name,
                    values=pred_eigenvector,
                    enabled=False,
                    cmap=self.config.colormap
                )

    def process_batch(self, model: SurfaceTransformerModule, batch_data, batch_idx: int, device: torch.device):
        """Process a single batch through the complete pipeline."""
        print(f"\n{'=' * 80}")
        print(f"PROCESSING BATCH {batch_idx + 1}")
        print('=' * 80)

        # Clear previous visualization
        ps.remove_all_structures()

        # STEP 1: Extract mesh data from batch
        print("STEP 1: Extracting mesh data from batch")

        # Handle case where batch_data is a list (from MeshDataset)
        if isinstance(batch_data, list):
            data = batch_data[0]  # Take the first Data object
        else:
            data = batch_data

        # Extract mesh metadata - handle both single values and lists
        def extract_first_if_list(attr_value):
            """Extract single value from potentially batched attribute."""
            if isinstance(attr_value, list):
                return attr_value[0] if len(attr_value) > 0 else None
            return attr_value

        mesh_file_path = extract_first_if_list(data.mesh_file_path)
        original_num_vertices = extract_first_if_list(data.original_num_vertices)

        print(f"Mesh file: {Path(mesh_file_path).name}")
        print(f"Original vertices: {original_num_vertices}")
        print(f"Total patch points: {len(data.pos)}")
        print(f"Number of patches: {len(data.center_indices)}")

        # STEP 2: Load original mesh for GT computation
        print(f"\nSTEP 2: Loading original mesh for GT computation")
        gt_data = self.load_original_mesh_for_gt(mesh_file_path)

        # STEP 3: Model inference
        print(f"\nSTEP 3: Model inference")
        inference_result = self.perform_model_inference(model, data, device)

        if inference_result['predicted_eigenvalues'] is None:
            print("❌ Failed to compute eigendecomposition, skipping this batch")
            return

        # STEP 4: Compute predicted quantities
        print(f"\nSTEP 4: Computing predicted quantities")
        predicted_data = self.compute_predicted_quantities_from_laplacian(
            inference_result['laplacian_matrix'], gt_data['vertices']
        )

        # STEP 5: Visualization
        print(f"\nSTEP 5: Creating comprehensive visualization")
        mesh_structure = self.visualize_mesh(gt_data['vertices'], gt_data['gt_vertex_normals'], gt_data['faces'])

        # Print analysis
        if self.config.enable_eigenvalue_info:
            self.print_eigenvalue_analysis(
                gt_data.get('gt_eigenvalues'),
                inference_result['predicted_eigenvalues'],
                Path(mesh_file_path).name
            )

        # Print correlation analysis
        if self.config.enable_correlation_analysis:
            gt_eigenvecs = gt_data.get('gt_eigenvectors')
            if gt_eigenvecs is not None and inference_result['predicted_eigenvectors'] is not None:
                correlation_matrix = self.compute_eigenvector_correlations(
                    gt_eigenvecs, inference_result['predicted_eigenvectors']
                )
                self.print_eigenvector_correlation_analysis(correlation_matrix, "Predicted vs GT")

        # Add visualizations
        self.visualize_comprehensive_eigenvectors(
            mesh_structure,
            gt_data.get('gt_eigenvalues'), gt_data.get('gt_eigenvectors'),
            inference_result['predicted_eigenvalues'], inference_result['predicted_eigenvectors']
        )

        self.add_comprehensive_curvature_visualizations(mesh_structure, gt_data, predicted_data)

        # Add mesh reconstructions using eigenvectors
        print(f"\nSTEP 6: Computing and visualizing mesh reconstructions")
        gt_reconstructions = []
        pred_reconstructions = []

        # Compute GT reconstructions (with area weighting)
        if gt_data.get('gt_eigenvectors') is not None:
            gt_reconstructions = self.compute_mesh_reconstruction(
                gt_data['vertices'],
                gt_data['gt_eigenvectors'],
                gt_data.get('gt_eigenvalues'),
                self.config.num_eigenvectors_to_show,
                vertex_areas=gt_data.get('vertex_areas')  # Pass vertex areas for GT case
            )

        # Compute predicted reconstructions (using predicted mean curvature as vertex areas)
        if inference_result['predicted_eigenvectors'] is not None:
            # Use predicted mean curvature as pseudo vertex areas for PRED case
            predicted_vertex_areas = None
            if predicted_data.get('predicted_mean_curvature') is not None:
                predicted_mean_curvature = predicted_data['predicted_mean_curvature']

                # Convert mean curvature to pseudo areas
                # Use absolute values and add small epsilon to avoid zeros
                predicted_vertex_areas = np.abs(predicted_mean_curvature) + 1e-7

                # Optionally normalize to have similar scale as GT areas
                if gt_data.get('vertex_areas') is not None:
                    gt_area_scale = np.mean(gt_data['vertex_areas'])
                    pred_area_scale = np.mean(predicted_vertex_areas)
                    if pred_area_scale > 1e-10:
                        predicted_vertex_areas = predicted_vertex_areas * (gt_area_scale / pred_area_scale)

                print(f"Using predicted mean curvature as vertex areas for PRED reconstruction")
                print(f"Predicted area range: [{predicted_vertex_areas.min():.6f}, {predicted_vertex_areas.max():.6f}]")

            pred_reconstructions = self.compute_mesh_reconstruction(
                gt_data['vertices'],  # Use original vertices as reference
                inference_result['predicted_eigenvectors'],
                inference_result['predicted_eigenvalues'],
                self.config.num_eigenvectors_to_show,
                vertex_areas=predicted_vertex_areas  # Use predicted mean curvature as areas
            )

        # Visualize reconstructions
        if gt_reconstructions or pred_reconstructions:
            self.visualize_mesh_reconstructions(
                gt_data['faces'],
                gt_reconstructions,
                pred_reconstructions,
                gt_data.get('gt_eigenvalues'),
                inference_result['predicted_eigenvalues']
            )

        print(f"\n✅ Comprehensive visualization completed for {Path(mesh_file_path).name}")

    def run_dataset_iteration(self, cfg: DictConfig):
        """Run visualization on all meshes in the dataset."""
        print(f"\n{'=' * 80}")
        print("REAL-TIME EIGENANALYSIS VISUALIZATION")
        print("=" * 80)

        # Get and validate checkpoint path from config
        if not hasattr(cfg, 'ckpt_path') or cfg.ckpt_path is None:
            raise ValueError("ckpt_path parameter is required. Please specify the path to the model checkpoint.")

        ckpt_path = Path(cfg.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        # Get and validate training k from config
        if not hasattr(cfg, 'training_k') or cfg.training_k is None:
            raise ValueError("training_k parameter is required. Please specify the k value used during training.")

        training_k = int(cfg.training_k)
        if training_k <= 0:
            raise ValueError(f"training_k must be positive, got {training_k}")

        print(f"Checkpoint: {ckpt_path}")
        print(f"Training k: {training_k}")
        print("=" * 80)

        # Store training k for k-ratio correction
        self.training_k = training_k

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load trained model
        model = self.load_trained_model(ckpt_path, device, cfg)

        # Set random seed for reproducibility
        pl.seed_everything(cfg.globals.seed)

        # Initialize data module and loader
        data_module = hydra.utils.instantiate(cfg.data_module)
        data_loader = data_module.val_dataloader()

        # Handle case where val_dataloader returns a list of dataloaders
        if isinstance(data_loader, list):
            data_loader = data_loader[0]  # Take the first validation dataloader

        print(f"DataLoader ready with batch size: {data_loader.batch_size}")

        # Setup polyscope
        self.setup_polyscope()

        # Process all batches
        for batch_idx, batch_data in enumerate(data_loader):
            print(f"\n🔍 Processing batch {batch_idx + 1}")

            try:
                self.process_batch(model, batch_data, batch_idx, device)

                print(f"\nVisualization ready for batch {batch_idx + 1}. Close window to continue to next batch.")
                ps.show()

            except Exception as e:
                print(f"❌ Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()

                user_input = input("Continue to next batch? (y/n): ").strip().lower()
                if user_input != 'y':
                    break

        print(f"\n✅ Completed processing all batches!")


@hydra.main(version_base="1.2", config_path='./config')
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration."""

    # Create visualization config
    vis_config = VisualizationConfig(
        point_radius=0.005,
        num_eigenvectors_to_show=60,
        colormap='coolwarm',
        enable_eigenvalue_info=True,
        enable_correlation_analysis=True
    )

    # Check dependencies
    print("Checking dependencies...")
    try:
        from pyFM.mesh import TriMesh
        print("✓ PyFM available")
    except ImportError:
        raise ImportError("PyFM is required for GT eigendecomposition. Install with: pip install pyFM")

    if HAS_IGL:
        print("✓ libigl available")
    else:
        print("⚠ libigl not available - GT mean curvature will be skipped")

    # Create visualizer and run
    visualizer = RealTimeEigenanalysisVisualizer(config=vis_config)
    visualizer.run_dataset_iteration(cfg)


if __name__ == "__main__":
    main()