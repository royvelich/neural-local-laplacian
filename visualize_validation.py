#!/usr/bin/env python3
"""
Real-time Eigenanalysis Visualization with MeshDataset and Model Inference

This script:
1. Uses Hydra to instantiate a MeshDataset from config
2. Loads a trained LaplacianTransformerModule from checkpoint
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
from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from neural_local_laplacian.utils.utils import (
    normalize_mesh_vertices,
    assemble_stiffness_and_mass_matrices,
    compute_laplacian_eigendecomposition
)
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


@dataclass
class ReconstructionSettings:
    """Settings for mesh reconstruction visualization."""
    use_pred_areas: bool = False  # Use predicted areas from model's area head
    current_pred_k: int = 20  # Will be updated with actual k from dataset


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

        # NEW: Reconstruction settings with UI control
        self.reconstruction_settings = ReconstructionSettings()

        # Store current batch data for re-computation
        self.current_gt_data = None
        self.current_inference_result = None
        self.current_predicted_data = None
        self.current_mesh_structure = None

        # Store raw data for k-NN slider updates
        self.current_stiffness_weights = None
        self.current_areas = None
        self.current_original_vertices = None
        self.original_k = None
        self.current_vertex_indices = None
        self.current_center_indices = None
        self.current_batch_indices = None

        # NEW: Track reconstruction structure names for removal
        self.reconstruction_structure_names = []

    def setup_polyscope(self):
        """Initialize and configure polyscope with UI callback."""
        ps.init()
        ps.set_up_dir("z_up")
        ps.look_at(camera_location=[2.0, 2.0, 2.0], target=[0, 0, 0])
        ps.set_ground_plane_mode("none")
        ps.set_background_color((0.05, 0.05, 0.05))  # Dark background

        # Set up UI callback for reconstruction settings
        ps.set_user_callback(self._reconstruction_settings_ui_callback)

    def _reconstruction_settings_ui_callback(self):
        """ImGui callback for reconstruction settings window."""
        import polyscope.imgui as psim

        # Checkbox for using predicted areas for area-weighted reconstruction
        psim.Text("PRED Mesh Reconstruction Options:")
        psim.Separator()

        changed, new_setting = psim.Checkbox(
            "Use Predicted Areas (Area-Weighted)",
            self.reconstruction_settings.use_pred_areas
        )

        if changed:
            self.reconstruction_settings.use_pred_areas = new_setting
            print(f"[*] Reconstruction setting changed: use_pred_areas = {new_setting}")

            # Re-compute and update reconstructions if we have current data
            if self._has_current_batch_data():
                self._recompute_and_update_reconstructions()

        psim.Text("")

        # k-NN input field for PRED reconstructions
        if self.original_k is not None:
            # Use EnterReturnsTrue flag to only trigger on Enter press
            k_changed, new_k = psim.InputInt(
                "PRED k-NN neighbors",
                self.reconstruction_settings.current_pred_k,
                flags=psim.ImGuiInputTextFlags_EnterReturnsTrue
            )

            # Clamp the value to valid range
            new_k = max(5, min(100, new_k))

            if k_changed and new_k != self.reconstruction_settings.current_pred_k:
                self.reconstruction_settings.current_pred_k = new_k
                print(f"[*] PRED k changed: {new_k}")

                # Re-compute PRED with new k if we have current data
                if self._has_current_batch_data():
                    self._update_pred_with_new_k(new_k)
        else:
            psim.Text("k-NN input: (no data loaded)")

        psim.Text("(Press Enter to apply changes)")

        psim.Text("")
        psim.Text("Current Settings:")
        if self.reconstruction_settings.use_pred_areas:
            psim.TextColored((0.0, 1.0, 0.0, 1.0), "[OK] Using predicted areas from model")
            psim.Text("  (Area-weighted reconstruction for PRED)")
        else:
            psim.TextColored((1.0, 0.5, 0.0, 1.0), "[o] Using standard Euclidean reconstruction")
            psim.Text("  (Standard L2 projection for PRED)")

        if self.original_k is not None:
            psim.Text(f"Original k: {self.original_k}, Current PRED k: {self.reconstruction_settings.current_pred_k}")

        psim.Text("")
        psim.Separator()
        psim.Text("Note: GT always uses PyFM vertex areas and original mesh connectivity")

    def _has_current_batch_data(self) -> bool:
        """Check if we have current batch data available for re-computation."""
        return (self.current_gt_data is not None and
                self.current_inference_result is not None and
                self.current_predicted_data is not None and
                self.current_stiffness_weights is not None and
                self.current_areas is not None and
                self.current_original_vertices is not None)

    def _compute_predicted_vertex_areas(self, inference_result: Dict) -> Optional[np.ndarray]:
        """Compute predicted vertex areas based on current settings.

        Uses the areas predicted by the model's area head.
        """
        if not self.reconstruction_settings.use_pred_areas:
            return None

        if inference_result.get('areas') is not None:
            predicted_vertex_areas = inference_result['areas']
            print(f"Using predicted areas from model (range: [{predicted_vertex_areas.min():.6f}, {predicted_vertex_areas.max():.6f}])")
            return predicted_vertex_areas
        else:
            print(f"[!] No predicted areas available, falling back to standard reconstruction")
            return None

    def _compute_all_reconstructions(self, gt_data: Dict, inference_result: Dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute both GT and predicted reconstructions."""
        gt_reconstructions = []
        pred_reconstructions = []

        # Compute GT reconstructions (always use area weighting)
        if gt_data.get('gt_eigenvectors') is not None:
            gt_reconstructions = self.compute_mesh_reconstruction(
                gt_data['vertices'],
                gt_data['gt_eigenvectors'],
                gt_data.get('gt_eigenvalues'),
                self.config.num_eigenvectors_to_show,
                vertex_areas=gt_data.get('vertex_areas')  # Always use GT areas
            )

        # Compute predicted reconstructions (respecting current setting)
        if inference_result['predicted_eigenvectors'] is not None:
            predicted_vertex_areas = self._compute_predicted_vertex_areas(inference_result)
            pred_reconstructions = self.compute_mesh_reconstruction(
                gt_data['vertices'],  # Use original vertices as reference
                inference_result['predicted_eigenvectors'],
                inference_result['predicted_eigenvalues'],
                self.config.num_eigenvectors_to_show,
                vertex_areas=predicted_vertex_areas  # None for standard, or predicted areas from model
            )

        return gt_reconstructions, pred_reconstructions

    def _update_mesh_reconstructions(self, gt_data: Dict, inference_result: Dict):
        """Complete pipeline for computing and visualizing mesh reconstructions."""
        print(f"Computing and visualizing mesh reconstructions...")

        # Compute reconstructions
        gt_reconstructions, pred_reconstructions = self._compute_all_reconstructions(
            gt_data, inference_result
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
            print("[OK] Mesh reconstructions completed")
        else:
            print("[!] No reconstructions to visualize")

    def _recompute_knn_connectivity_for_k(self, new_k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute k-NN connectivity for new k value.

        Args:
            new_k: New number of neighbors per patch

        Returns:
            Tuple of (vertex_indices, center_indices, batch_indices) for new k
        """
        print(f"  Recomputing k-NN connectivity for k={new_k}...")

        vertices = self.current_original_vertices
        num_vertices = len(vertices)

        # Build k-NN index for the entire mesh
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=new_k + 1, algorithm='auto').fit(vertices)

        # Get k+1 nearest neighbors for ALL vertices at once
        distances, neighbor_indices = nbrs.kneighbors(vertices)  # Shape: (N, k+1)

        # Vectorized removal of center point from neighbors
        center_positions = np.arange(num_vertices)[:, np.newaxis]  # Shape: (N, 1)
        is_center_mask = neighbor_indices == center_positions  # Shape: (N, k+1)

        # Create mask to keep only non-center neighbors
        keep_mask = ~is_center_mask  # Shape: (N, k+1)

        # For each row, we want to keep the first k True values in keep_mask
        keep_positions = np.cumsum(keep_mask, axis=1)  # Shape: (N, k+1)
        final_mask = (keep_positions <= new_k) & keep_mask  # Shape: (N, k+1)

        # Extract neighbor indices using the mask
        neighbor_indices_flat = neighbor_indices[final_mask]  # Shape: (N*k,)
        neighbor_indices_filtered = neighbor_indices_flat.reshape(num_vertices, new_k)  # Shape: (N, k)

        # Create new connectivity tensors
        vertex_indices = torch.from_numpy(neighbor_indices_filtered.flatten()).long()  # Shape: (N*k,)
        center_indices = torch.from_numpy(np.arange(num_vertices)).long()  # Shape: (N,)
        batch_indices = torch.from_numpy(np.repeat(range(num_vertices), new_k)).long()  # Shape: (N*k,)

        print(f"  Created connectivity: {len(vertex_indices)} total points for {num_vertices} patches")
        return vertex_indices, center_indices, batch_indices

    def _recompute_pred_laplacian_with_k(self, new_k: int):
        """
        Recompute PRED stiffness/mass matrices and eigendata with new k.

        Args:
            new_k: New number of neighbors per patch
        """
        print(f"  Recomputing PRED matrices with k={new_k}...")

        # Get new k-NN connectivity
        new_vertex_indices, new_center_indices, new_batch_indices = self._recompute_knn_connectivity_for_k(new_k)

        # Clone stiffness weights for modification
        corrected_weights = self.current_stiffness_weights.clone()

        # Resize stiffness weights to match new k if necessary
        current_k = corrected_weights.shape[1]
        if new_k != current_k:
            if new_k < current_k:
                # Truncate to new_k
                corrected_weights = corrected_weights[:, :new_k]
                print(f"  Truncated stiffness weights from {current_k} to {new_k}")
            else:
                # Pad with zeros or repeat existing weights
                num_patches = corrected_weights.shape[0]
                padding_size = new_k - current_k
                # Use mean of existing weights for padding
                mean_weights = corrected_weights.mean(dim=1, keepdim=True)
                padding = mean_weights.expand(-1, padding_size)
                corrected_weights = torch.cat([corrected_weights, padding], dim=1)
                print(f"  Padded stiffness weights from {current_k} to {new_k} using mean weights")

        # Create attention mask (all True for uniform k)
        num_patches = corrected_weights.shape[0]
        attention_mask = torch.ones(num_patches, new_k, dtype=torch.bool, device=corrected_weights.device)

        # Assemble new stiffness and mass matrices
        new_stiffness_matrix, new_mass_matrix = assemble_stiffness_and_mass_matrices(
            stiffness_weights=corrected_weights,
            areas=self.current_areas,
            attention_mask=attention_mask,
            vertex_indices=new_vertex_indices,
            center_indices=new_center_indices,
            batch_indices=new_batch_indices
        )

        print(f"  Assembled new stiffness matrix: {new_stiffness_matrix.shape} ({new_stiffness_matrix.nnz} non-zeros)")

        # Compute new eigendecomposition using generalized problem
        new_eigenvalues, new_eigenvectors = self.compute_eigendecomposition(
            new_stiffness_matrix, k=self.config.num_eigenvectors_to_show, mass_matrix=new_mass_matrix
        )

        if new_eigenvalues is not None:
            print(f"  Computed {len(new_eigenvalues)} new eigenvalues")
            print(f"  New eigenvalue range: [{new_eigenvalues[0]:.2e}, {new_eigenvalues[-1]:.6f}]")

            # Update current inference result
            self.current_inference_result['stiffness_matrix'] = new_stiffness_matrix
            self.current_inference_result['mass_matrix'] = new_mass_matrix
            self.current_inference_result['predicted_eigenvalues'] = new_eigenvalues
            self.current_inference_result['predicted_eigenvectors'] = new_eigenvectors
        else:
            print(f"  Failed to compute eigendecomposition for k={new_k}")

    def _update_pred_with_new_k(self, new_k: int):
        """
        Complete pipeline for updating PRED with new k.

        Args:
            new_k: New number of neighbors per patch
        """
        print(f"Updating PRED reconstructions with k={new_k}...")

        # Recompute PRED matrices and eigendata
        self._recompute_pred_laplacian_with_k(new_k)

        # Recompute predicted quantities from new matrices
        if self.current_inference_result.get('stiffness_matrix') is not None:
            new_predicted_data = self.compute_predicted_quantities_from_laplacian(
                self.current_inference_result['stiffness_matrix'],
                self.current_gt_data['vertices'],
                mass_matrix=self.current_inference_result.get('mass_matrix')
            )
            self.current_predicted_data = new_predicted_data
            print(f"  Updated predicted quantities")

        # Recompute and update reconstructions (reuses existing pipeline!)
        self._recompute_and_update_reconstructions()

        print(f"PRED updated with k={new_k}")

    def _recompute_and_update_reconstructions(self):
        """Re-compute and update mesh reconstructions with new settings."""
        if not self._has_current_batch_data():
            print("[!] No current batch data available for re-computation")
            return

        print("[*] Re-computing mesh reconstructions with new settings...")

        # Remove existing reconstruction structures
        self._remove_existing_reconstructions()

        # Re-compute and visualize with new settings
        self._update_mesh_reconstructions(
            self.current_gt_data, self.current_inference_result
        )

        print("[OK] Mesh reconstructions updated with new settings")

    def _remove_existing_reconstructions(self):
        """Remove existing reconstruction structures from polyscope."""
        try:
            # Remove tracked reconstruction structures
            for struct_name in self.reconstruction_structure_names:
                try:
                    ps.remove_surface_mesh(struct_name)
                    print(f"  Removed reconstruction structure: {struct_name}")
                except Exception as e:
                    print(f"  Warning: Could not remove structure {struct_name}: {e}")

            # Clear the tracking list
            self.reconstruction_structure_names.clear()
            print(f"  Cleared reconstruction structure tracking list")

        except Exception as e:
            print(f"  Warning: Failed to remove some reconstruction structures: {e}")

    def load_trained_model(self, ckpt_path: Path, device: torch.device, cfg: DictConfig) -> LaplacianTransformerModule:
        """
        Load trained LaplacianTransformerModule from checkpoint.

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

            # Load model from checkpoint with inference-time settings:
            # - normalize_patch_features=True: normalize inputs to unit sphere (same as training)
            # - scale_areas_by_patch_size=False: don't scale areas back (Laplacian assembly uses normalized space)
            model = LaplacianTransformerModule.load_from_checkpoint(
                str(ckpt_path),
                map_location=device,
                normalize_patch_features=True,
                scale_areas_by_patch_size=True,
            )

            model.eval()
            model.to(device)

            print(f"[OK] Model loaded successfully on {device}")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Input dim: {model._input_dim}")
            print(f"   Model dim: {model._d_model}")
            print(f"   Num eigenvalues: {model._num_eigenvalues}")

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {ckpt_path}: {e}")

    def compute_eigendecomposition(self, stiffness_matrix: scipy.sparse.csr_matrix,
                                   k: int = 50,
                                   mass_matrix: scipy.sparse.csr_matrix = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of the Laplacian using generalized eigenvalue problem.

        Solves S @ v = lambda * M @ v where S is stiffness and M is mass matrix.

        Args:
            stiffness_matrix: Sparse symmetric stiffness matrix
            k: Number of eigenvalues to compute
            mass_matrix: Sparse diagonal mass matrix (optional, uses identity if None)

        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted in ascending order by eigenvalue.
            - eigenvalues: Array of shape (k,) with smallest k eigenvalues
            - eigenvectors: Array of shape (n, k) with corresponding eigenvectors
        """
        try:
            # Use the centralized eigendecomposition function from utils
            eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
                stiffness_matrix, k, mass_matrix=mass_matrix
            )

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
            vertices = normalize_mesh_vertices(raw_vertices)

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

    def perform_model_inference(self, model: LaplacianTransformerModule, batch_data: Data, device: torch.device) -> Dict[str, Any]:
        """
        Perform model inference and compute predicted quantities.

        Args:
            model: Trained LaplacianTransformerModule
            batch_data: Preprocessed batch data from MeshDataset
            device: Device for computation

        Returns:
            Dictionary containing predicted quantities, stiffness and mass matrices
        """
        print("Performing model inference...")

        # Move batch data to device
        batch_data = batch_data.to(device)

        with torch.no_grad():
            # Forward pass to get new result structure with stiffness weights and areas
            forward_result = model._forward_pass(batch_data)

            # Extract components from new forward result structure
            stiffness_weights = forward_result['stiffness_weights']  # Shape: (batch_size, max_k)
            areas = forward_result['areas']  # Shape: (batch_size,)
            attention_mask = forward_result['attention_mask']  # Shape: (batch_size, max_k)
            batch_sizes = forward_result['batch_sizes']  # Shape: (batch_size,)

            print(f"Got stiffness weights shape: {stiffness_weights.shape}")
            print(f"Got areas shape: {areas.shape}")
            print(f"Got attention mask shape: {attention_mask.shape}")
            print(f"Got batch sizes: {batch_sizes}")
            print(f"Area statistics: mean={areas.mean():.6f}, std={areas.std():.6f}, min={areas.min():.6f}, max={areas.max():.6f}")

            # Use patch_idx if available (MeshDataset), otherwise use batch (synthetic)
            batch_indices = getattr(batch_data, 'patch_idx', batch_data.batch)

            # Assemble separate stiffness and mass matrices
            stiffness_matrix, mass_matrix = assemble_stiffness_and_mass_matrices(
                stiffness_weights=stiffness_weights,
                areas=areas,
                attention_mask=attention_mask,
                vertex_indices=batch_data.vertex_indices,
                center_indices=batch_data.center_indices,
                batch_indices=batch_indices
            )

            print(f"Assembled stiffness matrix: {stiffness_matrix.shape} ({stiffness_matrix.nnz} non-zeros)")
            print(f"Assembled mass matrix: {mass_matrix.shape} (diagonal)")

            # Compute eigendecomposition using generalized eigenvalue problem
            predicted_eigenvalues, predicted_eigenvectors = self.compute_eigendecomposition(
                stiffness_matrix, k=self.config.num_eigenvectors_to_show, mass_matrix=mass_matrix
            )

            if predicted_eigenvalues is not None:
                print(f"Computed {len(predicted_eigenvalues)} predicted eigenvalues")
                print(f"Predicted eigenvalue range: [{predicted_eigenvalues[0]:.2e}, {predicted_eigenvalues[-1]:.6f}]")

        return {
            'stiffness_matrix': stiffness_matrix,
            'mass_matrix': mass_matrix,
            'predicted_eigenvalues': predicted_eigenvalues,
            'predicted_eigenvectors': predicted_eigenvectors,
            'stiffness_weights': stiffness_weights.cpu().numpy(),
            'areas': areas.cpu().numpy(),
            'attention_mask': attention_mask.cpu().numpy(),
            'batch_sizes': batch_sizes.cpu().numpy(),
            'forward_result': forward_result  # Store complete forward result
        }

    def compute_predicted_quantities_from_laplacian(self, stiffness_matrix: scipy.sparse.csr_matrix,
                                                    vertices: np.ndarray,
                                                    mass_matrix: scipy.sparse.csr_matrix = None) -> Dict[str, np.ndarray]:
        """
        Compute predicted mean curvature vector and derived quantities from predicted matrices.

        The mean curvature vector is computed as: Delta_r = M^-1 @ S @ r = 2Hn

        Args:
            stiffness_matrix: Predicted stiffness matrix (scipy sparse)
            vertices: Mesh vertices array of shape (N, 3)
            mass_matrix: Predicted mass matrix (scipy sparse diagonal)

        Returns:
            Dictionary containing predicted quantities
        """
        print("Computing predicted quantities from stiffness/mass matrices...")

        try:
            # Compute S @ r first
            stiffness_times_vertices = stiffness_matrix @ vertices  # Shape: (N, 3)

            # If mass matrix is provided, compute M^-1 @ S @ r
            if mass_matrix is not None:
                # Extract diagonal of mass matrix and invert
                mass_diag = mass_matrix.diagonal()
                # Avoid division by zero
                mass_diag_inv = np.where(mass_diag > 1e-10, 1.0 / mass_diag, 0.0)

                # Apply M^-1 (element-wise division since M is diagonal)
                predicted_mean_curvature_vector = stiffness_times_vertices * mass_diag_inv[:, np.newaxis]
            else:
                # Without mass matrix, just use S @ r
                predicted_mean_curvature_vector = stiffness_times_vertices

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
            print(f"Error computing predicted quantities from matrices: {e}")
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
        min_c ||f - A_l c||_M^2
        where M = diag(vertex_areas) is the mass matrix.

        Args:
            original_vertices: Original mesh vertices f in R^{n x 3}
            eigenvectors: Eigenvectors Phi in R^{n x k}
            num_available: Number of available eigenvectors to use
            vertex_areas: Vertex areas a in R^n

        Returns:
            List of reconstructed vertex arrays
        """
        reconstructed_meshes = []

        # Create mass matrix M = diag(vertex_areas)
        M = np.diag(vertex_areas)  # Shape: (n, n)
        f = original_vertices  # Shape: (n, 3)

        for num_eigenvecs in range(1, num_available + 1):
            # Step 1: Extract basis A_l = Phi[:, 1:l] (first l eigenvectors)
            A_l = eigenvectors[:, :num_eigenvecs]  # Shape: (n, l)

            # Step 2: Compute Gram matrix G = A_l^T M A_l
            G = A_l.T @ M @ A_l  # Shape: (l, l)

            # Step 3: Compute projection target b = A_l^T M f
            b = A_l.T @ M @ f  # Shape: (l, 3)

            # Step 4: Solve for coefficients G c = b
            try:
                c = np.linalg.solve(G, b)  # Shape: (l, 3)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if G is singular
                print(f"Warning: Gram matrix singular for {num_eigenvecs} eigenvectors, using pseudoinverse")
                c = np.linalg.pinv(G) @ b  # Shape: (l, 3)

            # Step 5: Reconstruct f_l = A_l c
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
            # c_i = <coords, phi_i> = Sum coords_j * phi_i_j
            coefficients = np.dot(original_vertices.T, current_eigenvectors)  # Shape: (3, num_eigenvecs)

            # Reconstruct coordinates: coords = Sum c_i * phi_i
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

            mesh_name = f"GT Recon {num_eigenvecs:02d} eigenvec (lambda={gt_eigenval:.3f})"

            try:
                gt_mesh = ps.register_surface_mesh(
                    name=mesh_name,
                    vertices=offset_vertices,
                    faces=original_faces,
                    enabled=(i == 0)  # Only enable the first one by default
                )

                # Track this structure for later removal
                self.reconstruction_structure_names.append(mesh_name)

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

            # Include reconstruction method in name
            method_suffix = " (Pred Areas)" if self.reconstruction_settings.use_pred_areas else " (Standard)"
            mesh_name = f"PRED Recon {num_eigenvecs:02d} eigenvec (lambda={pred_eigenval:.3f}){method_suffix}"

            try:
                pred_mesh = ps.register_surface_mesh(
                    name=mesh_name,
                    vertices=offset_vertices,
                    faces=original_faces,
                    enabled=(i == 0)  # Only enable the first one by default
                )

                # Track this structure for later removal
                self.reconstruction_structure_names.append(mesh_name)

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
            print(f"  GT Eigenvector {i} -> Pred Eigenvector {best_match_idx}: {best_correlation:.4f}")

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
                                                   predicted_data: Dict[str, np.ndarray],
                                                   inference_result: Dict[str, Any] = None):
        """
        Add comprehensive curvature and normal visualizations to the mesh.

        Visualization naming scheme:
        - A: Mean Curvature - GT (scalar)
        - B: Mean Curvature - PRED (scalar)
        - C: Mean Curvature Vector - GT (raw Hn vector)
        - D: Mean Curvature Vector - PRED (raw Hn vector)
        - E0: Mesh Normals - GT (from mesh geometry)
        - E: Normal (from MCV) - GT (normalized Hn)
        - F: Normal (from MCV) - PRED (normalized Hn)
        - G1: Normal Alignment (GT MCV vs PRED MCV) (cosine similarity)
        - G2: Normal Alignment (PRED vs Mesh GT) (cosine similarity)
        - H: Vertex Areas - GT (from PyFM)
        - I: Vertex Areas - PRED (from model's area head)
        """
        print("Adding comprehensive curvature visualizations...")

        # === VERTEX AREAS ===
        if gt_data.get('vertex_areas') is not None:
            mesh_structure.add_scalar_quantity(
                name="H Vertex Areas - GT",
                values=gt_data['vertex_areas'],
                enabled=False,
                cmap='viridis'
            )

        if inference_result is not None and inference_result.get('areas') is not None:
            mesh_structure.add_scalar_quantity(
                name="I Vertex Areas - PRED",
                values=inference_result['areas'],
                enabled=False,
                cmap='viridis'
            )

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

        # === NORMALIZED MEAN CURVATURE VECTORS (SURFACE NORMALS) ===
        # These are the mean curvature vectors normalized to unit length,
        # representing the surface normal direction derived from the Laplacian

        # First, add GT mesh normals (from mesh geometry) for reference
        if gt_data.get('gt_vertex_normals') is not None:
            mesh_structure.add_vector_quantity(
                name="E0 Mesh Normals - GT",
                values=gt_data['gt_vertex_normals'] * self.vector_scales.get_vector_scale('gt_normals'),
                enabled=False,
                color=(0.0, 1.0, 0.0),  # Green
                vectortype="ambient"
            )

        if gt_data.get('gt_mean_curvature_vector') is not None:
            gt_mcv = gt_data['gt_mean_curvature_vector']
            gt_mcv_norm = np.linalg.norm(gt_mcv, axis=1, keepdims=True)
            gt_normals_from_mcv = np.where(gt_mcv_norm > 1e-10, gt_mcv / gt_mcv_norm, 0)

            mesh_structure.add_vector_quantity(
                name="E Normal (from MCV) - GT",
                values=gt_normals_from_mcv * self.vector_scales.get_vector_scale('gt_normals'),
                enabled=False,
                color=(0.0, 0.8, 0.8),  # Dark Cyan
                vectortype="ambient"
            )

        if predicted_data.get('predicted_normals') is not None:
            mesh_structure.add_vector_quantity(
                name="F Normal (from MCV) - PRED",
                values=predicted_data['predicted_normals'] * self.vector_scales.get_vector_scale('predicted_normals'),
                enabled=False,
                color=(0.8, 0.4, 0.0),  # Dark Orange
                vectortype="ambient"
            )

        # === COMPARISON METRICS FOR MEAN CURVATURE VECTORS ===
        gt_mean_curv_vector = gt_data.get('gt_mean_curvature_vector')
        pred_normals = predicted_data.get('predicted_normals')
        gt_mesh_normals = gt_data.get('gt_vertex_normals')

        # Mean curvature vector alignment comparison (normal alignment between GT MCV and PRED MCV)
        if gt_mean_curv_vector is not None and pred_normals is not None:
            # Normalize GT mean curvature vector
            gt_norm = np.linalg.norm(gt_mean_curv_vector, axis=1, keepdims=True)
            gt_normalized = np.where(gt_norm > 1e-10, gt_mean_curv_vector / gt_norm, 0)

            # Cosine similarity between normalized vectors (normal alignment)
            normal_alignment = np.sum(pred_normals * gt_normalized, axis=1)
            mesh_structure.add_scalar_quantity(
                name="G1 Normal Alignment (GT MCV vs PRED MCV)",
                values=normal_alignment,
                enabled=False,
                cmap='coolwarm'
            )

        # Alignment between predicted normals and GT mesh normals
        if pred_normals is not None and gt_mesh_normals is not None:
            # Cosine similarity between predicted normal and GT mesh normal
            pred_vs_mesh_alignment = np.sum(pred_normals * gt_mesh_normals, axis=1)
            mesh_structure.add_scalar_quantity(
                name="G2 Normal Alignment (PRED vs Mesh GT)",
                values=pred_vs_mesh_alignment,
                enabled=False,
                cmap='coolwarm'
            )

    def _compute_eigenvector_cosine_similarities(
            self,
            gt_eigenvectors: Optional[np.ndarray],
            pred_eigenvectors: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Compute cosine similarity between GT and predicted eigenvectors.

        Handles sign ambiguity by taking absolute value of dot product.

        Args:
            gt_eigenvectors: GT eigenvectors of shape (N, k_gt)
            pred_eigenvectors: Predicted eigenvectors of shape (N, k_pred)

        Returns:
            Array of cosine similarities for each eigenvector pair, or None if not computable
        """
        if gt_eigenvectors is None or pred_eigenvectors is None:
            return None

        num_pairs = min(gt_eigenvectors.shape[1], pred_eigenvectors.shape[1])
        cosine_similarities = np.zeros(num_pairs)

        for i in range(num_pairs):
            gt_vec = gt_eigenvectors[:, i]
            pred_vec = pred_eigenvectors[:, i]

            # Normalize vectors
            gt_norm = np.linalg.norm(gt_vec)
            pred_norm = np.linalg.norm(pred_vec)

            if gt_norm > 1e-8 and pred_norm > 1e-8:
                gt_vec_normalized = gt_vec / gt_norm
                pred_vec_normalized = pred_vec / pred_norm

                # Absolute value to handle sign ambiguity
                cosine_similarities[i] = np.abs(np.dot(gt_vec_normalized, pred_vec_normalized))
            else:
                cosine_similarities[i] = 0.0

        return cosine_similarities

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

        # Compute cosine similarities between GT and predicted eigenvectors
        cosine_similarities = self._compute_eigenvector_cosine_similarities(
            gt_eigenvectors, pred_eigenvectors
        )

        # Print cosine similarities to console
        if cosine_similarities is not None:
            print(f"\n" + "-" * 70)
            print("EIGENVECTOR COSINE SIMILARITIES (GT vs PRED)")
            print("-" * 70)
            print(f"{'Index':<8} {'Cosine Sim':<12} {'Description'}")
            print("-" * 70)
            for i in range(min(num_to_show, len(cosine_similarities))):
                if i == 0:
                    desc = "constant"
                elif i == 1:
                    desc = "Fiedler"
                else:
                    desc = ""
                print(f"{i:<8} {cosine_similarities[i]:<12.6f} {desc}")

            # Summary statistics
            print("-" * 70)
            print(f"Mean cosine similarity: {cosine_similarities[:num_to_show].mean():.6f}")
            print(f"Min cosine similarity:  {cosine_similarities[:num_to_show].min():.6f}")
            print(f"Max cosine similarity:  {cosine_similarities[:num_to_show].max():.6f}")
            if num_to_show > 1:
                # Exclude first (constant) eigenvector for non-trivial stats
                print(f"Mean (excluding constant): {cosine_similarities[1:num_to_show].mean():.6f}")
            print("-" * 70 + "\n")

        # Add eigenvectors in groups
        for i in range(num_to_show):
            # Get cosine similarity for this pair (if available)
            cos_sim = cosine_similarities[i] if cosine_similarities is not None and i < len(cosine_similarities) else None
            cos_sim_str = f", cos={cos_sim:.4f}" if cos_sim is not None else ""

            # Add GT eigenvector
            if gt_eigenvectors is not None and i < gt_available:
                gt_eigenvector = gt_eigenvectors[:, i]
                gt_eigenvalue = gt_eigenvalues[i] if gt_eigenvalues is not None else 0.0

                if i == 0:
                    gt_name = f"Eigenvector {i:02d}a GT (lambda={gt_eigenvalue:.2e}, constant{cos_sim_str})"
                elif i == 1:
                    gt_name = f"Eigenvector {i:02d}a GT (lambda={gt_eigenvalue:.6f}, Fiedler{cos_sim_str})"
                else:
                    gt_name = f"Eigenvector {i:02d}a GT (lambda={gt_eigenvalue:.6f}{cos_sim_str})"

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
                    pred_name = f"Eigenvector {i:02d}b PRED (lambda={pred_eigenvalue:.2e}, constant{cos_sim_str})"
                elif i == 1:
                    pred_name = f"Eigenvector {i:02d}b PRED (lambda={pred_eigenvalue:.6f}, Fiedler{cos_sim_str})"
                else:
                    pred_name = f"Eigenvector {i:02d}b PRED (lambda={pred_eigenvalue:.6f}{cos_sim_str})"

                mesh_structure.add_scalar_quantity(
                    name=pred_name,
                    values=pred_eigenvector,
                    enabled=False,
                    cmap=self.config.colormap
                )

    def process_batch(self, model: LaplacianTransformerModule, batch_data, batch_idx: int, device: torch.device):
        """Process a single batch through the complete pipeline."""
        print(f"\n{'=' * 80}")
        print(f"PROCESSING BATCH {batch_idx + 1}")
        print('=' * 80)

        # Clear previous visualization and tracking
        ps.remove_all_structures()
        self.reconstruction_structure_names.clear()  # Clear reconstruction tracking

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
            print("[X] Failed to compute eigendecomposition, skipping this batch")
            return

        # STEP 4: Compute predicted quantities
        print(f"\nSTEP 4: Computing predicted quantities")
        predicted_data = self.compute_predicted_quantities_from_laplacian(
            inference_result['stiffness_matrix'], gt_data['vertices'],
            mass_matrix=inference_result.get('mass_matrix')
        )

        # STEP 5: Store current batch data for UI-driven re-computation
        self.current_gt_data = gt_data
        self.current_inference_result = inference_result
        self.current_predicted_data = predicted_data

        # Store raw data for k-NN slider updates
        self.current_stiffness_weights = torch.from_numpy(inference_result['stiffness_weights'])
        self.current_areas = torch.from_numpy(inference_result['areas'])
        self.current_original_vertices = gt_data['vertices']
        self.original_k = len(data.pos) // len(data.center_indices)  # Calculate original k
        self.current_vertex_indices = data.vertex_indices
        self.current_center_indices = data.center_indices
        self.current_batch_indices = data.batch

        # Initialize PRED k slider with original k
        self.reconstruction_settings.current_pred_k = self.original_k
        print(f"Initialized PRED k slider with original k={self.original_k}")

        # STEP 6: Visualization
        print(f"\nSTEP 6: Creating comprehensive visualization")
        mesh_structure = self.visualize_mesh(gt_data['vertices'], gt_data['gt_vertex_normals'], gt_data['faces'])
        self.current_mesh_structure = mesh_structure

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

        self.add_comprehensive_curvature_visualizations(mesh_structure, gt_data, predicted_data, inference_result)

        # Add mesh reconstructions using eigenvectors
        print(f"\nSTEP 7: Computing and visualizing mesh reconstructions")
        self._update_mesh_reconstructions(gt_data, inference_result)

        print(f"\n[OK] Comprehensive visualization completed for {Path(mesh_file_path).name}")

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

        print(f"Checkpoint: {ckpt_path}")
        print("=" * 80)

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

        # Setup polyscope with UI callback
        self.setup_polyscope()

        # Process all batches
        for batch_idx, batch_data in enumerate(data_loader):
            print(f"\n[>] Processing batch {batch_idx + 1}")

            try:
                self.process_batch(model, batch_data, batch_idx, device)

                print(f"\nVisualization ready for batch {batch_idx + 1}. Use the 'Reconstruction Settings' window to control PRED reconstruction method.")
                print("Close window to continue to next batch.")
                ps.show()

            except Exception as e:
                print(f"[X] Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()

                user_input = input("Continue to next batch? (y/n): ").strip().lower()
                if user_input != 'y':
                    break

        print(f"\n[OK] Completed processing all batches!")


@hydra.main(version_base="1.2", config_path='./visualization_config')
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
        print("[OK] PyFM available")
    except ImportError:
        raise ImportError("PyFM is required for GT eigendecomposition. Install with: pip install pyFM")

    if HAS_IGL:
        print("[OK] libigl available")
    else:
        print("[!] libigl not available - GT mean curvature will be skipped")

    # Create visualizer and run
    visualizer = RealTimeEigenanalysisVisualizer(config=vis_config)
    visualizer.run_dataset_iteration(cfg)


if __name__ == "__main__":
    main()