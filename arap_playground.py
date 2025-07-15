#!/usr/bin/env python3
"""
ARAP Playground with GT vs PRED Laplacian Comparison

This script loads a 3D mesh and a trained model checkpoint to compare
ARAP deformation using Ground Truth (cotangent) vs Predicted Laplacian matrices.
"""

import polyscope as ps
import trimesh
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

# Hydra imports
import hydra
from omegaconf import DictConfig

# Local imports - reuse from visualize_validation2.py
from neural_local_laplacian.modules.laplacian_modules import SurfaceTransformerModule
from neural_local_laplacian.utils import utils
from torch_geometric.data import Data
from neural_local_laplacian.arap import ARAPDeformer

# PyFM for GT Laplacian
from pyFM.mesh import TriMesh


class ARAPController:
    """Controller for ARAP deformation with GT vs PRED Laplacian comparison."""

    def __init__(self, mesh_file_path: str, model_ckpt_path: str, training_k: int, cfg: DictConfig):
        self.cfg = cfg
        self.mesh_file_path = Path(mesh_file_path)
        self.model_ckpt_path = Path(model_ckpt_path)
        self.training_k = training_k

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # UI state
        self.use_pred_laplacian = False  # Start with GT
        self.pred_k = training_k  # Start with training k

        # Animation state
        self.frame = 0
        self.n_frames = cfg.animation.n_frames
        self.mode = cfg.animation.initial_mode
        self.axis = cfg.animation.initial_axis
        self.max_angle = cfg.animation.max_angle
        self.max_displacement = cfg.animation.max_displacement

        # GUI state
        self.mode_index = 0 if self.mode == "rotation" else 1
        self.axis_index = {"x": 0, "y": 1, "z": 2}[self.axis]

        # Initialize data
        self._load_and_setup()

    def _load_and_setup(self):
        """Load mesh, model, and compute Laplacian matrices."""
        print(f"Loading mesh: {self.mesh_file_path}")
        print(f"Loading model: {self.model_ckpt_path}")

        # Load and normalize mesh
        self.gt_data = self._load_original_mesh_for_gt()
        self.vertices = self.gt_data['vertices']
        self.faces = self.gt_data['faces']

        # Load trained model
        self.model = self._load_trained_model()

        # Compute GT Laplacian
        self.gt_laplacian_matrix = self._compute_gt_laplacian()

        # Compute PRED Laplacian
        self._compute_pred_laplacian()

        # Get constraint indices
        self.fixed_indices, self.handle_indices = self._get_constraint_indices()

        # Initialize ARAP deformer (start with GT)
        self._initialize_arap_deformer()

        # Setup visualization
        self._setup_visualization()

    def _load_original_mesh_for_gt(self) -> Dict[str, Any]:
        """Load and normalize mesh (same as visualize_validation2.py)."""
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(str(self.mesh_file_path))
            raw_vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)

            # Apply the same normalization as MeshDataset
            vertices = utils.normalize_mesh_vertices(raw_vertices)

            # Update mesh with normalized vertices for normal computation
            mesh.vertices = vertices
            gt_vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)

            print(f"Mesh has {len(vertices)} vertices and {len(faces)} faces")
            print(f"Normalized vertices: center at origin, max distance = {np.linalg.norm(vertices, axis=1).max():.6f}")

            return {
                'vertices': vertices,
                'faces': faces,
                'gt_vertex_normals': gt_vertex_normals,
            }

        except Exception as e:
            raise RuntimeError(f"Failed to load mesh {self.mesh_file_path}: {e}")

    def _load_trained_model(self) -> SurfaceTransformerModule:
        """Load trained model (same as visualize_validation2.py)."""
        if not self.model_ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.model_ckpt_path}")

        try:
            print(f"Loading model checkpoint from: {self.model_ckpt_path}")

            # Load model from checkpoint with default parameters
            model = SurfaceTransformerModule.load_from_checkpoint(
                str(self.model_ckpt_path),
                map_location=self.device,
                input_dim=3,  # XYZ coordinates
                d_model=256,
                nhead=8,
                num_encoder_layers=4,
                dim_feedforward=256,
                num_eigenvalues=50,
                dropout=0,
                loss_configs=None,
                optimizer_cfg=None
            )

            model.eval()
            model.to(self.device)

            print(f"✅ Model loaded successfully")
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_ckpt_path}: {e}")

    def _compute_gt_laplacian(self) -> sp.csr_matrix:
        """Compute GT Laplacian using PyFM (same as visualize_validation2.py)."""
        print("Computing GT Laplacian using PyFM...")
        try:
            # Create PyFM TriMesh object
            pyfm_mesh = TriMesh(self.vertices, self.faces)

            # Process the mesh to compute Laplacian
            pyfm_mesh.process(k=10, intrinsic=False, verbose=False)

            # Get the Laplacian matrix from PyFM
            # PyFM stores the Laplacian as a sparse matrix
            gt_laplacian = pyfm_mesh.W  # This is the cotangent Laplacian

            # Convert to CSR format if it's not already
            if hasattr(gt_laplacian, 'tocsr'):
                gt_laplacian = gt_laplacian.tocsr()

            print(f"GT Laplacian computed: {gt_laplacian.shape} ({gt_laplacian.nnz} non-zeros)")
            return gt_laplacian

        except Exception as e:
            print(f"Warning: PyFM failed, using simple cotangent weights: {e}")
            # Fallback to simple cotangent computation
            return self._compute_simple_cotangent_laplacian()

    def _compute_simple_cotangent_laplacian(self) -> sp.csr_matrix:
        """Fallback: compute simple cotangent Laplacian."""
        # Use the ARAPDeformer's internal cotangent computation
        temp_arap = ARAPDeformer(
            vertices=torch.from_numpy(self.vertices),
            faces=torch.from_numpy(self.faces),
            device=self.device,
            precomputed_laplacian=None
        )
        # Extract the Laplacian matrix (this is a bit hacky)
        temp_arap._build_laplacian_matrix()
        laplacian_dense = temp_arap._laplacian_matrix.cpu().numpy()
        return sp.csr_matrix(laplacian_dense)

    def _extract_mesh_patches_for_inference(self, k: int) -> Data:
        """Extract k-NN patches for model inference (same as MeshDataset logic)."""
        vertices = self.vertices
        num_vertices = len(vertices)

        # Build k-NN index for the entire mesh
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(vertices)

        # Get k+1 nearest neighbors for ALL vertices at once
        distances, neighbor_indices = nbrs.kneighbors(vertices)  # Shape: (N, k+1)

        # Vectorized removal of center point from neighbors
        center_positions = np.arange(num_vertices)[:, np.newaxis]  # Shape: (N, 1)
        is_center_mask = neighbor_indices == center_positions  # Shape: (N, k+1)

        # Create mask to keep only non-center neighbors
        keep_mask = ~is_center_mask  # Shape: (N, k+1)

        # For each row, we want to keep the first k True values in keep_mask
        keep_positions = np.cumsum(keep_mask, axis=1)  # Shape: (N, k+1)
        final_mask = (keep_positions <= k) & keep_mask  # Shape: (N, k+1)

        # Extract neighbor indices using the mask
        neighbor_indices_flat = neighbor_indices[final_mask]  # Shape: (N*k,)
        neighbor_indices_filtered = neighbor_indices_flat.reshape(num_vertices, k)  # Shape: (N, k)

        # Vectorized extraction of neighbor positions
        all_neighbor_positions = vertices[neighbor_indices_filtered]  # Shape: (N, k, 3)

        # Vectorized translation: subtract center from each patch
        center_positions_expanded = vertices[:, np.newaxis, :]  # Shape: (N, 1, 3)
        patch_positions = all_neighbor_positions - center_positions_expanded  # Shape: (N, k, 3)

        # Prepare data for PyTorch Geometric
        all_positions = patch_positions.reshape(-1, 3)  # Shape: (N*k, 3)
        all_neighbor_indices = neighbor_indices_filtered.flatten()  # Shape: (N*k,)
        all_center_indices = np.arange(num_vertices)  # Shape: (N,)

        # Create batch indices - each patch is a separate graph
        batch_indices = np.repeat(range(num_vertices), k)  # Shape: (N*k,)

        # Convert to tensors
        pos_tensor = torch.from_numpy(all_positions).float()
        features_tensor = pos_tensor.clone()  # Use XYZ as features
        batch_tensor = torch.from_numpy(batch_indices).long()
        vertex_indices_tensor = torch.from_numpy(all_neighbor_indices).long()
        center_indices_tensor = torch.from_numpy(all_center_indices).long()

        # Create Data object
        data = Data(
            pos=pos_tensor,
            x=features_tensor,
            batch=batch_tensor,
            vertex_indices=vertex_indices_tensor,
            center_indices=center_indices_tensor
        )

        return data

    def _perform_model_inference(self, k: int) -> Dict[str, Any]:
        """Perform model inference (same as visualize_validation2.py)."""
        print(f"Performing model inference with k={k}...")

        # Extract patches for inference
        batch_data = self._extract_mesh_patches_for_inference(k)
        batch_data = batch_data.to(self.device)

        with torch.no_grad():
            # Forward pass to get token weights
            forward_result = self.model._forward_pass(batch_data)
            token_weights = forward_result['token_weights']

            print(f"Got token weights shape: {token_weights.shape}")

            # Apply k-ratio correction if training k differs from inference k
            if k != self.training_k:
                k_ratio = k / self.training_k
                token_weights = token_weights / k_ratio
                print(f"Applied k-ratio correction: inference_k={k}, training_k={self.training_k}, ratio={k_ratio:.3f}")

            return {
                'token_weights': token_weights.cpu().numpy(),
                'vertex_indices': batch_data.vertex_indices.cpu().numpy(),
                'center_indices': batch_data.center_indices.cpu().numpy(),
                'batch_indices': batch_data.batch.cpu().numpy()
            }

    def _assemble_sparse_laplacian(self, weights: np.ndarray, vertex_indices: np.ndarray,
                                   center_indices: np.ndarray, batch_indices: np.ndarray) -> sp.csr_matrix:
        """Assemble sparse Laplacian matrix (same as visualize_validation2.py)."""
        # Get dimensions
        num_patches = weights.shape[0]
        num_points_per_patch = weights.shape[1]
        num_vertices = max(vertex_indices.max(), center_indices.max()) + 1

        # Flatten weights to match vertex_indices structure
        weights_flat = weights.flatten()  # Shape: (total_points,)

        # Expand center indices to match the structure of vertex_indices
        center_vertices_expanded = np.repeat(center_indices, num_points_per_patch)

        # Create off-diagonal entries (negative weights)
        row_indices = center_vertices_expanded  # From center
        col_indices = vertex_indices  # To neighbor
        data_values = -weights_flat  # Negative weights for off-diagonal

        # Create symmetric connections
        row_indices_sym = vertex_indices  # From neighbor
        col_indices_sym = center_vertices_expanded  # To center
        data_values_sym = -weights_flat  # Same negative weights

        # Combine all off-diagonal connections
        all_row_indices = np.concatenate([row_indices, row_indices_sym])
        all_col_indices = np.concatenate([col_indices, col_indices_sym])
        all_data_values = np.concatenate([data_values, data_values_sym])

        # Create sparse matrix from coordinates
        laplacian_coo = sp.coo_matrix(
            (all_data_values, (all_row_indices, all_col_indices)),
            shape=(num_vertices, num_vertices)
        )

        # Sum duplicate entries and convert to CSR
        laplacian_csr = laplacian_coo.tocsr()
        laplacian_csr.sum_duplicates()

        # Vectorized diagonal computation
        row_sums = np.array(laplacian_csr.sum(axis=1)).flatten()
        diagonal_values = -row_sums

        # Set diagonal entries
        laplacian_csr.setdiag(diagonal_values)

        # Ensure numerical symmetry
        laplacian_csr = 0.5 * (laplacian_csr + laplacian_csr.T)

        return laplacian_csr

    def _compute_pred_laplacian(self):
        """Compute predicted Laplacian matrix."""
        print(f"Computing PRED Laplacian with k={self.pred_k}...")

        # Perform model inference
        inference_result = self._perform_model_inference(self.pred_k)

        # Assemble sparse Laplacian matrix
        self.pred_laplacian_matrix = self._assemble_sparse_laplacian(
            weights=inference_result['token_weights'],
            vertex_indices=inference_result['vertex_indices'],
            center_indices=inference_result['center_indices'],
            batch_indices=inference_result['batch_indices']
        )

        print(f"PRED Laplacian computed: {self.pred_laplacian_matrix.shape} ({self.pred_laplacian_matrix.nnz} non-zeros)")

    def _get_constraint_indices(self) -> Tuple[list, list]:
        """Get constraint indices from config."""
        n_vertices = len(self.vertices)

        if self.cfg.constraints.auto_select.enabled:
            return self._get_auto_constraint_indices()

        # Manual selection
        fixed_indices = [idx for idx in self.cfg.constraints.fixed_indices if idx < n_vertices]
        handle_indices = [idx for idx in self.cfg.constraints.handle_indices if idx < n_vertices]

        # Fallback if empty
        if not fixed_indices:
            fallback_idx = int(n_vertices * self.cfg.constraints.fixed_fallback_ratio)
            fixed_indices = [fallback_idx]

        if not handle_indices:
            fallback_idx = int(n_vertices * self.cfg.constraints.handle_fallback_ratio)
            handle_indices = [fallback_idx]

        return fixed_indices, handle_indices

    def _get_auto_constraint_indices(self) -> Tuple[list, list]:
        """Auto-select constraint indices based on vertex positions."""
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_idx = axis_map[self.cfg.constraints.auto_select.axis.lower()]

        axis_coords = self.vertices[:, axis_idx]

        k_fixed = self.cfg.constraints.auto_select.k_fixed
        k_handle = self.cfg.constraints.auto_select.k_handle

        fixed_indices = np.argsort(axis_coords)[:k_fixed].tolist()
        handle_indices = np.argsort(axis_coords)[-k_handle:].tolist()

        print(f"Auto-selected constraints along {self.cfg.constraints.auto_select.axis}-axis:")
        print(f"  Fixed indices: {fixed_indices}")
        print(f"  Handle indices: {handle_indices}")

        return fixed_indices, handle_indices

    def _initialize_arap_deformer(self):
        """Initialize ARAP deformer with current Laplacian."""
        laplacian_to_use = self.pred_laplacian_matrix if self.use_pred_laplacian else self.gt_laplacian_matrix

        # Convert scipy sparse matrix to torch tensor if needed
        if laplacian_to_use is not None and hasattr(laplacian_to_use, 'toarray'):
            # It's a scipy sparse matrix, convert to dense torch tensor
            laplacian_dense = torch.from_numpy(laplacian_to_use.toarray()).float().to(self.device)
        else:
            # It's already a torch tensor or None
            laplacian_dense = laplacian_to_use

        self.arap_deformer = ARAPDeformer(
            vertices=torch.from_numpy(self.vertices),
            faces=torch.from_numpy(self.faces),
            device=self.device,
            precomputed_laplacian=laplacian_dense
        )

        self.arap_deformer.set_fixed(self.fixed_indices)
        self.arap_deformer.set_handles(self.handle_indices)

        print(f"ARAP deformer initialized with {'PRED' if self.use_pred_laplacian else 'GT'} Laplacian")

    def _setup_visualization(self):
        """Setup polyscope visualization."""
        # Register mesh
        if self.faces is not None:
            self.ps_mesh = ps.register_surface_mesh("mesh", self.vertices, self.faces)
            self.ps_mesh.set_edge_width(self.cfg.visualization.edge_width)

        # Register constraint vertices
        if len(self.fixed_indices) > 0:
            fixed_vertices = self.vertices[self.fixed_indices]
            ps.register_point_cloud("Fixed Vertices (Red)", fixed_vertices,
                                    color=self.cfg.visualization.fixed_vertex_color)
            ps.get_point_cloud("Fixed Vertices (Red)").set_radius(self.cfg.visualization.constraint_vertex_radius)

        if len(self.handle_indices) > 0:
            handle_vertices = self.vertices[self.handle_indices]
            ps.register_point_cloud("Handle Vertices (Blue)", handle_vertices,
                                    color=self.cfg.visualization.handle_vertex_color)
            ps.get_point_cloud("Handle Vertices (Blue)").set_radius(self.cfg.visualization.constraint_vertex_radius)

        print(f"Visualization setup complete")
        print(f"Fixed vertices (Red): {self.fixed_indices}")
        print(f"Handle vertices (Blue): {self.handle_indices}")

    def _update_laplacian_mode(self):
        """Update ARAP deformer when Laplacian mode changes."""
        print(f"Switching to {'PRED' if self.use_pred_laplacian else 'GT'} Laplacian...")
        self._initialize_arap_deformer()

    def _recompute_pred_laplacian(self):
        """Recompute PRED Laplacian when k changes."""
        if self.use_pred_laplacian:  # Only recompute if currently using PRED
            print(f"Recomputing PRED Laplacian with k={self.pred_k}...")
            self._compute_pred_laplacian()
            self._initialize_arap_deformer()

    def _get_rotation_matrix(self, angle: float, axis: str) -> np.ndarray:
        """Get rotation matrix for animation."""
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        if axis == "x":
            return np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
        elif axis == "y":
            return np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        elif axis == "z":
            return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        else:
            return np.eye(3)

    def _get_translation_vector(self, displacement: float, axis: str) -> np.ndarray:
        """Get translation vector for animation."""
        translation = np.zeros(3)
        if axis == "x":
            translation[0] = displacement
        elif axis == "y":
            translation[1] = displacement
        elif axis == "z":
            translation[2] = displacement
        return translation

    def ui_callback(self):
        """Main UI callback with controls."""
        import polyscope.imgui as psim

        # Laplacian Settings
        if psim.TreeNode("Laplacian Settings"):
            # GT vs PRED checkbox
            changed, new_setting = psim.Checkbox("Use Predicted Laplacian", self.use_pred_laplacian)
            if changed:
                self.use_pred_laplacian = new_setting
                self._update_laplacian_mode()

            # k input for PRED
            k_changed, new_k = psim.InputInt("PRED k-NN neighbors", self.pred_k,
                                             flags=psim.ImGuiInputTextFlags_EnterReturnsTrue)
            new_k = max(4, min(100, new_k))  # Clamp to valid range
            if k_changed and new_k != self.pred_k:
                self.pred_k = new_k
                self._recompute_pred_laplacian()

            # Status
            psim.Separator()
            laplacian_type = "PRED" if self.use_pred_laplacian else "GT"
            psim.Text(f"Current Laplacian: {laplacian_type}")
            if self.use_pred_laplacian:
                psim.Text(f"PRED k: {self.pred_k}")
            psim.Text(f"Training k: {self.training_k}")

            psim.TreePop()

        # Animation Controls
        if psim.TreeNode("Animation Controls"):
            # Mode selection
            changed, self.mode_index = psim.RadioButton("Rotation", self.mode_index == 0), self.mode_index
            if changed:
                self.mode_index = 0
                self.mode = "rotation"

            psim.SameLine()
            changed, self.mode_index = psim.RadioButton("Translation", self.mode_index == 1), self.mode_index
            if changed:
                self.mode_index = 1
                self.mode = "translation"

            # Axis selection
            psim.Text("Axis:")
            changed, self.axis_index = psim.RadioButton("X", self.axis_index == 0), self.axis_index
            if changed:
                self.axis_index = 0
                self.axis = "x"

            psim.SameLine()
            changed, self.axis_index = psim.RadioButton("Y", self.axis_index == 1), self.axis_index
            if changed:
                self.axis_index = 1
                self.axis = "y"

            psim.SameLine()
            changed, self.axis_index = psim.RadioButton("Z", self.axis_index == 2), self.axis_index
            if changed:
                self.axis_index = 2
                self.axis = "z"

            # Animation parameters
            psim.Separator()
            if self.mode == "rotation":
                changed, self.max_angle = psim.SliderFloat("Max Angle (rad)", self.max_angle, 0.0, np.pi)
            else:
                changed, self.max_displacement = psim.SliderFloat("Max Displacement", self.max_displacement, 0.0, 1.0)

            # Reset button
            if psim.Button("Reset Animation"):
                self.frame = 0

            # Status
            psim.Separator()
            psim.Text(f"Mode: {self.mode}")
            psim.Text(f"Axis: {self.axis}")
            psim.Text(f"Frame: {self.frame}/{self.n_frames}")

            psim.TreePop()

        # Perform animation
        if len(self.handle_indices) > 0:
            self._animate_frame()

    def _animate_frame(self):
        """Perform animation for current frame."""
        # Compute animation parameter
        t = self.frame / self.n_frames
        handle_vertices = self.vertices[self.handle_indices]

        if self.mode == "rotation":
            angle = self.max_angle * np.sin(2 * np.pi * t)
            rotation_matrix = self._get_rotation_matrix(angle, self.axis)
            transformed_vertices = (rotation_matrix @ handle_vertices.T).T
        else:  # translation
            displacement = self.max_displacement * np.sin(2 * np.pi * t)
            translation = self._get_translation_vector(displacement, self.axis)
            transformed_vertices = handle_vertices + translation

        # Apply ARAP deformation
        deformed_vertices, _ = self.arap_deformer.deform(
            target_positions=transformed_vertices,
            max_iterations=self.cfg.arap.max_iterations,
            tolerance=self.cfg.arap.tolerance
        )
        deformed_vertices = deformed_vertices.cpu().numpy()

        # Update visualization
        self.ps_mesh.update_vertex_positions(deformed_vertices)

        if len(self.fixed_indices) > 0:
            fixed_deformed = deformed_vertices[self.fixed_indices]
            ps.get_point_cloud("Fixed Vertices (Red)").update_point_positions(fixed_deformed)

        if len(self.handle_indices) > 0:
            handle_deformed = deformed_vertices[self.handle_indices]
            ps.get_point_cloud("Handle Vertices (Blue)").update_point_positions(handle_deformed)

        # Increment frame
        self.frame = (self.frame + 1) % self.n_frames


@hydra.main(version_base="1.2", config_path="./playgrounds_config")
def main(cfg: DictConfig):
    """Main function."""
    # Validate required parameters
    if not hasattr(cfg, 'ckpt_path') or cfg.ckpt_path is None:
        raise ValueError("ckpt_path parameter is required")

    if not hasattr(cfg, 'training_k') or cfg.training_k is None:
        raise ValueError("training_k parameter is required")

    if not hasattr(cfg, 'mesh_path') or cfg.mesh_path is None:
        raise ValueError("mesh_path parameter is required")

    # Initialize Polyscope
    ps.init()
    ps.set_up_dir(cfg.visualization.up_dir)
    ps.set_ground_plane_mode(cfg.visualization.ground_plane_mode)
    ps.set_background_color(cfg.visualization.background_color)

    # Create ARAP controller
    controller = ARAPController(
        mesh_file_path=cfg.mesh_path,
        model_ckpt_path=cfg.ckpt_path,
        training_k=cfg.training_k,
        cfg=cfg
    )

    # Set UI callback
    ps.set_user_callback(controller.ui_callback)

    print("\n" + "=" * 60)
    print("ARAP PLAYGROUND WITH GT VS PRED LAPLACIAN")
    print("=" * 60)
    print("Controls:")
    print("  - Use 'Laplacian Settings' to switch between GT and PRED")
    print("  - Adjust 'PRED k-NN neighbors' to change model inference")
    print("  - Use 'Animation Controls' to change deformation mode")
    print("=" * 60)

    # Show visualization
    ps.show()


if __name__ == "__main__":
    main()