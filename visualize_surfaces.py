#!/usr/bin/env python3
"""
Enhanced Surface Visualization with Model Prediction and Mean Curvature Display

This script visualizes synthetic surface datasets and optionally compares:
1. Ground-truth analytic normals at surface centers
2. Predicted normals from trained SurfaceTransformerModule models
3. Mean curvature values at the origin (now printed to screen)

Usage:
    python visualize_surfaces.py                    # Original functionality
    python visualize_surfaces.py ckpt_path=model.ckpt  # With model prediction comparison
"""

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import polyscope as ps
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, replace
import torch
import torch.nn.functional as F
from pathlib import Path

# Import model class
from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from torch_geometric.data import Data, Batch


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    vector_scale: float
    point_radius: float
    param_radius: float
    surface_spacing_factor: float
    enable_mesh: bool
    enable_point_cloud: bool
    enable_parametrization: bool
    enable_normals: bool
    enable_differential_geometry: bool
    enable_model_prediction: bool  # NEW: Enable model prediction visualization
    smooth_shade: bool
    edge_width: float
    mesh_scalar_colormap: str
    pointcloud_scalar_colormap: str
    pointcloud_color: Tuple[float, float, float] = (0.0, 0.8, 0.0)  # Default: green
    # Display scale constants
    normal_display_scale: float = 8.0  # Multiplier for normal vector display
    origin_indicator_scale: float = 3.0  # Multiplier for origin point radius
    reference_frame_point_radius: float = 0.02
    reference_frame_line_radius: float = 0.01
    normal_origin_point_radius: float = 0.01


class ColorPalette:
    """Color palette for different visualization elements."""

    # Principal directions
    PRINCIPAL_V1 = (1.0, 0.0, 0.0)  # Red
    PRINCIPAL_V2 = (0.0, 0.0, 1.0)  # Blue

    # Curvature gradients
    GRAD_MEAN_CURVATURE = (0.0, 1.0, 0.0)  # Green
    GRAD_GAUSSIAN_CURVATURE = (1.0, 1.0, 0.0)  # Yellow

    # Surface properties
    NORMALS = (0.0, 1.0, 1.0)  # Cyan (also used for ground truth normals)
    PREDICTED_NORMALS = (1.0, 0.5, 0.0)  # Orange for predictions

    # Default colors
    DEFAULT_VECTOR = (0.5, 0.5, 0.5)  # Gray

    @classmethod
    def get_vector_color(cls, vector_name: str) -> Tuple[float, float, float]:
        """Get color for vector visualization."""
        # Normalize: remove _2d/_3d suffix for direction lookups
        base_name = vector_name.replace('_2d', '').replace('_3d', '')

        color_map = {
            'v1': cls.PRINCIPAL_V1,
            'v2': cls.PRINCIPAL_V2,
            'grad_H': cls.GRAD_MEAN_CURVATURE,
            'grad_K': cls.GRAD_GAUSSIAN_CURVATURE,
            'normals': cls.NORMALS,
            'gt_normals': cls.NORMALS,
            'predicted_normals': cls.PREDICTED_NORMALS
        }
        return color_map.get(base_name, color_map.get(vector_name, cls.DEFAULT_VECTOR))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vector array to unit length."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.where(norms > 0, vectors / norms, vectors)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def load_trained_model(ckpt_path: Path, device: torch.device) -> LaplacianTransformerModule:
    """
    Load trained LaplacianTransformerModule from checkpoint.

    Args:
        ckpt_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Loaded model in evaluation mode

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If model loading fails
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    try:
        print(f"Loading model checkpoint from: {ckpt_path}")

        # Load model from checkpoint - hyperparameters are saved automatically via save_hyperparameters()
        model = LaplacianTransformerModule.load_from_checkpoint(
            str(ckpt_path),
            map_location=device,
        )

        model.eval()
        model.to(device)

        print(f"[OK] Model loaded successfully on {device}")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Input dim: {model._input_dim}")
        print(f"   Model dim: {model._d_model}")

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {ckpt_path}: {e}")


def prepare_surface_for_model(surface_data: Data, device: torch.device) -> Batch:
    """
    Prepare a single surface Data object for model inference.

    Args:
        surface_data: Single surface Data object
        device: Device to move data to

    Returns:
        Batch object ready for model inference
    """
    # Move surface data to device
    surface_data = surface_data.to(device)

    # Create a list with single surface and convert to batch
    # This simulates what the dataloader does
    batch = Batch.from_data_list([surface_data])

    return batch


def predict_normal_from_patch(model: LaplacianTransformerModule,
                              surface_data: Data,
                              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Use trained model to predict mean curvature vector and normal at center.

    The mean curvature vector is computed as:
        mcv = (Sum_j s_ij * p_j) / A_i

    where:
    - s_ij are the learned stiffness weights
    - p_j are the neighbor positions (centered at origin)
    - A_i is the learned area for the center vertex

    Args:
        model: Trained LaplacianTransformerModule in eval mode
        surface_data: Single surface patch Data object
        device: Device for computation

    Returns:
        Tuple of (predicted_normal, stiffness_weights, predicted_mean_curvature_vector)
        - predicted_normal: Normalized predicted normal vector (3,)
        - stiffness_weights: Raw learned stiffness weights for visualization (num_points,)
        - predicted_mean_curvature_vector: Raw mean curvature vector before normalization (3,)
    """
    with torch.no_grad():
        # Prepare batch for model
        batch = prepare_surface_for_model(surface_data, device)

        # Forward pass to get stiffness weights and areas
        forward_result = model.forward(batch)
        stiffness_weights = forward_result['stiffness_weights']  # Shape: (1, max_k)
        areas = forward_result['areas']  # Shape: (1,)
        attention_mask = forward_result['attention_mask']  # Shape: (1, max_k)

        # Apply attention mask (zero out padded positions)
        stiffness_weights = stiffness_weights.masked_fill(~attention_mask, 0.0)

        # Extract positions
        positions = batch.pos  # Shape: (num_points, 3)
        num_points = positions.shape[0]

        # Get valid stiffness weights (handle case where max_k might differ from num_points)
        weights = stiffness_weights[0, :num_points]  # Shape: (num_points,)

        # Compute weighted sum of positions: Sum_j s_ij * p_j
        stiffness_sum = (weights.unsqueeze(-1) * positions).sum(dim=0)  # Shape: (3,)

        # Divide by area to get mean curvature vector: mcv = stiffness_sum / A_i
        predicted_mean_curvature_vector = stiffness_sum / areas[0]  # Shape: (3,)

        # Normalize to get predicted normal direction
        predicted_normal = F.normalize(predicted_mean_curvature_vector.unsqueeze(0), p=2, dim=1).squeeze(0)  # Shape: (3,)

        return predicted_normal, weights, predicted_mean_curvature_vector


def visualize_patch(points: np.ndarray, faces: np.ndarray, name: str, vis_config: VisualizationConfig) -> ps.SurfaceMesh:
    """Register surface mesh with polyscope."""
    mesh = ps.register_surface_mesh(
        name=name,
        vertices=points,
        faces=faces,
        smooth_shade=vis_config.smooth_shade,
        edge_width=vis_config.edge_width
    )
    return mesh


def visualize_point_cloud(points: np.ndarray, name: str, vis_config: VisualizationConfig, enabled: bool = False) -> ps.PointCloud:
    """Register point cloud with polyscope."""
    cloud = ps.register_point_cloud(
        name=name,
        points=points,
        radius=vis_config.point_radius,
        enabled=enabled,
        color=vis_config.pointcloud_color
    )
    return cloud


def add_reference_frame(scale: float = 1.0, vis_config: Optional[VisualizationConfig] = None) -> None:
    """Add XYZ reference frame at the origin."""
    # Get radii from config or use defaults
    point_radius = vis_config.reference_frame_point_radius if vis_config else 0.02
    line_radius = vis_config.reference_frame_line_radius if vis_config else 0.01

    # Define axis endpoints
    origin = np.array([0.0, 0.0, 0.0])
    x_axis = np.array([scale, 0.0, 0.0])
    y_axis = np.array([0.0, scale, 0.0])
    z_axis = np.array([0.0, 0.0, scale])

    # Create axis lines as point clouds with connections
    axes_points = np.array([
        origin, x_axis,  # X-axis
        origin, y_axis,  # Y-axis
        origin, z_axis  # Z-axis
    ])

    # Register the reference frame as a point cloud
    frame_cloud = ps.register_point_cloud(
        name="Reference Frame",
        points=axes_points,
        radius=point_radius,
        enabled=True
    )

    # Create colors for the axes (Red=X, Green=Y, Blue=Z)
    axis_colors = np.array([
        [0.5, 0.5, 0.5],  # Origin (gray)
        [1.0, 0.0, 0.0],  # X-axis end (red)
        [0.5, 0.5, 0.5],  # Origin (gray)
        [0.0, 1.0, 0.0],  # Y-axis end (green)
        [0.5, 0.5, 0.5],  # Origin (gray)
        [0.0, 0.0, 1.0]  # Z-axis end (blue)
    ])

    frame_cloud.add_color_quantity(
        name="axis_colors",
        values=axis_colors,
        enabled=True
    )

    # Add curve networks for the axes lines
    try:
        # X-axis line
        ps.register_curve_network(
            name="X-axis",
            nodes=np.array([origin, x_axis]),
            edges=np.array([[0, 1]]),
            color=(1.0, 0.0, 0.0),
            radius=line_radius,
            enabled=True
        )

        # Y-axis line
        ps.register_curve_network(
            name="Y-axis",
            nodes=np.array([origin, y_axis]),
            edges=np.array([[0, 1]]),
            color=(0.0, 1.0, 0.0),
            radius=line_radius,
            enabled=True
        )

        # Z-axis line
        ps.register_curve_network(
            name="Z-axis",
            nodes=np.array([origin, z_axis]),
            edges=np.array([[0, 1]]),
            color=(0.0, 0.0, 1.0),
            radius=line_radius,
            enabled=True
        )
    except Exception as e:
        print(f"Note: Could not create axis lines (polyscope version may not support curve networks): {e}")


class SurfaceVisualizer:
    """Handles surface visualization with differential geometry quantities and optional model predictions."""

    def __init__(self, config: DictConfig, vis_config: Optional[VisualizationConfig] = None,
                 trained_model: Optional[LaplacianTransformerModule] = None, device: Optional[torch.device] = None,
                 data_module=None):
        self.config = config
        self.vis_config = vis_config or VisualizationConfig()
        self.color_palette = ColorPalette()

        # NEW: Model prediction capabilities
        self.trained_model = trained_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # NEW: Storage for surface metrics to display in UI
        self.surface_metrics = []  # List of dicts with surface info and metrics

        # NEW: Store data module and current surfaces for live toggle
        self.data_module = data_module
        self._current_surfaces = None
        self._current_surface_names = None
        self._diff_geom_at_origin_only = self._get_initial_origin_only_setting()
        self._include_origin_in_grid = self._get_initial_include_origin_setting()

        # UI display state
        self._patch_idx = 0          # 0 = regular, 1 = downsampled, ...
        self._show_mesh = True
        self._show_pointcloud = False
        self._show_gt_normal_origin = True
        self._show_pred_normal_origin = True
        self._show_gt_normals_all = False

        # Cached prediction results (computed once per surface set)
        self._prediction_cache = {}  # patch_idx → (pred_normal, pred_weights, pred_mcv)
        self._show_reference_frame = False

        # Normal display settings (mutable for ImGui)
        self._gt_normal_color = list(ColorPalette.NORMALS)          # [0, 1, 1] cyan
        self._pred_normal_color = list(ColorPalette.PREDICTED_NORMALS)  # [1, 0.5, 0] orange
        self._normal_length = self.vis_config.normal_display_scale   # default 2.0
        self._normal_radius = self.vis_config.normal_origin_point_radius  # default 0.01
        self._point_radius = self.vis_config.point_radius            # default 0.008

        # Surface and origin dot colors
        self._surface_color = [0.5, 0.5, 0.5]  # gray default for mesh + point cloud
        self._origin_dot_color = [1.0, 0.0, 0.0]  # red default for GT + Pred dots

    def _get_initial_origin_only_setting(self) -> bool:
        """Get initial diff_geom_at_origin_only setting from config."""
        try:
            dataset_config = self.config.data_module.train_dataset_specification.dataset
            return getattr(dataset_config, 'diff_geom_at_origin_only', False)
        except (AttributeError, KeyError):
            return False

    def _get_initial_include_origin_setting(self) -> bool:
        """Get initial include_origin_in_grid setting from config."""
        try:
            dataset_config = self.config.data_module.train_dataset_specification.dataset
            return getattr(dataset_config, 'include_origin_in_grid', False)
        except (AttributeError, KeyError):
            return False

    @property
    def is_diff_geom_at_origin_only(self) -> bool:
        """Check if currently using origin-only differential geometry computation."""
        return self._diff_geom_at_origin_only

    def _get_train_dataset(self):
        """Get the training dataset from the data module."""
        if self.data_module is None:
            return None
        try:
            return self.data_module._train_dataset_specification.dataset
        except AttributeError:
            return None

    def toggle_diff_geom_at_origin_only(self) -> bool:
        """
        Toggle the diff_geom_at_origin_only setting and regenerate surfaces.
        Returns the new setting value.
        """
        dataset = self._get_train_dataset()
        if dataset is None:
            print("Cannot toggle: no dataset available")
            print(f"  data_module is None: {self.data_module is None}")
            if self.data_module is not None:
                print(f"  data_module type: {type(self.data_module)}")
                print(f"  has _train_dataset_specification: {hasattr(self.data_module, '_train_dataset_specification')}")
            return self._diff_geom_at_origin_only

        # Toggle the setting
        self._diff_geom_at_origin_only = not self._diff_geom_at_origin_only
        dataset._diff_geom_at_origin_only = self._diff_geom_at_origin_only

        print(f"\n{'=' * 60}")
        print(f"Toggled diff_geom_at_origin_only to: {self._diff_geom_at_origin_only}")
        print(f"{'=' * 60}")

        # Regenerate surfaces
        self._regenerate_and_visualize()

        return self._diff_geom_at_origin_only

    def toggle_include_origin_in_grid(self) -> bool:
        """
        Toggle the include_origin_in_grid setting and regenerate surfaces.
        Returns the new setting value.
        """
        dataset = self._get_train_dataset()
        if dataset is None:
            print("Cannot toggle: no dataset available")
            return self._include_origin_in_grid

        # Toggle the setting
        self._include_origin_in_grid = not self._include_origin_in_grid
        dataset._include_origin_in_grid = self._include_origin_in_grid

        print(f"\n{'=' * 60}")
        print(f"Toggled include_origin_in_grid to: {self._include_origin_in_grid}")
        print(f"{'=' * 60}")

        # Regenerate surfaces
        self._regenerate_and_visualize()

        return self._include_origin_in_grid

    def _regenerate_and_visualize(self) -> None:
        """Regenerate surfaces from dataset and re-visualize."""
        dataset = self._get_train_dataset()
        if dataset is None:
            return

        # Reset dataset RNG for reproducibility
        dataset.reset_rng()

        # Generate new surfaces
        print("Regenerating surfaces...")
        new_surfaces = dataset._generate_surfaces()

        # Get surface names and visualize (handles clearing + rendering)
        surface_names = self._get_surface_names(new_surfaces)
        self.visualize_surface_set(new_surfaces, surface_names)
        print("Regeneration complete!")

    def _get_surface_names(self, surfaces: List) -> List[str]:
        """Generate surface names based on the number of surfaces and grid samplers."""
        num_surfaces = len(surfaces)
        names = []

        try:
            # Try to get grid sampler information from config if available
            grid_samplers = self.config.data_module.train_dataset_specification.dataset.grid_samplers

            for i, grid_sampler in enumerate(grid_samplers):
                sampler_type = grid_sampler._target_.split('.')[-1]  # Get class name

                if sampler_type == 'RegularGridSampler':
                    num_points = grid_sampler.num_points
                    names.append(f'Regular Grid ({num_points} points)')
                elif sampler_type == 'RandomGridSampler':
                    num_points_range = grid_sampler.num_points_range
                    if isinstance(num_points_range, (list, tuple)) and len(num_points_range) == 2:
                        names.append(f'Random Grid ({num_points_range[0]}-{num_points_range[1]} points)')
                    else:
                        names.append(f'Random Grid ({num_points_range} points)')
                else:
                    names.append(f'Surface {i + 1}')

        except (AttributeError, KeyError):
            # Fallback: generate generic names if config access fails
            names = [f'Surface {i + 1}' for i in range(num_surfaces)]

        return names


    def _get_origin_position(self, surface: Data, translation: np.ndarray) -> np.ndarray:
        """
        Get the 3D origin position for a surface, applying translation.

        Uses origin_idx if available (actual mesh vertex), falls back to origin_pos,
        then to (0,0,0) as last resort.

        Args:
            surface: Surface data object
            translation: Translation offset to apply

        Returns:
            Origin position as numpy array of shape (1, 3)
        """
        if hasattr(surface, 'origin_idx') and hasattr(surface, 'pos'):
            origin_idx = surface.origin_idx.item()
            return to_numpy(surface.pos[origin_idx:origin_idx + 1]) + translation
        elif hasattr(surface, 'origin_pos'):
            return to_numpy(surface.origin_pos) + translation
        else:
            return np.array([[0.0, 0.0, 0.0]]) + translation

    def _extract_surface_data(self, surface: Data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract position, face, and normal data from surface."""
        pos = to_numpy(surface.pos)
        face = to_numpy(surface.face).T

        # Handle origin-only mode: normal might be shape (1, 3) instead of (N, 3)
        if hasattr(surface, 'normal'):
            normals = to_numpy(surface.normal)
            if self.is_diff_geom_at_origin_only and normals.shape[0] == 1:
                # Broadcast single normal to all vertices for visualization
                normals = np.broadcast_to(normals, (pos.shape[0], 3)).copy()
        else:
            # Fallback: compute simple normals if not available
            normals = np.array([[0.0, 0.0, 1.0]] * pos.shape[0])

        return pos, face, normals

    def _extract_differential_geometry(self, surface: Data) -> dict:
        """Extract differential geometry quantities from surface, handling origin-only mode."""

        def safe_extract_and_broadcast(attr_name: str) -> Optional[np.ndarray]:
            if not hasattr(surface, attr_name):
                return None

            value = to_numpy(getattr(surface, attr_name))

            # If origin-only mode and we have a single value, broadcast to all points
            if self.is_diff_geom_at_origin_only and value.shape[0] == 1:
                num_points = surface.pos.shape[0]
                if value.ndim == 1:
                    # Scalar quantity: shape (1,) -> (N,)
                    value = np.broadcast_to(value, (num_points,)).copy()
                else:
                    # Vector quantity: shape (1, D) -> (N, D)
                    value = np.broadcast_to(value, (num_points,) + value.shape[1:]).copy()

            return value

        return {
            'vectors_3d': {
                'v1': safe_extract_and_broadcast('v1_3d'),
                'v2': safe_extract_and_broadcast('v2_3d'),
                'grad_H': safe_extract_and_broadcast('grad_H_3d'),
                'grad_K': safe_extract_and_broadcast('grad_K_3d')
            },
            'vectors_2d': {
                'v1_2d': safe_extract_and_broadcast('v1_2d'),
                'v2_2d': safe_extract_and_broadcast('v2_2d'),
                'grad_H_2d': safe_extract_and_broadcast('grad_H_2d'),
                'grad_K_2d': safe_extract_and_broadcast('grad_K_2d')
            },
            'scalars': {
                'Mean Curvature (H)': safe_extract_and_broadcast('H'),
                'Gaussian Curvature (K)': safe_extract_and_broadcast('K'),
                'Principal Curvature (k1)': safe_extract_and_broadcast('k1'),
                'Principal Curvature (k2)': safe_extract_and_broadcast('k2')
            }
        }

    def _add_vector_quantities(self, structure, surface, structure_type: str = "default") -> None:
        """Add vector quantities to polyscope structure."""
        if not self.vis_config.enable_differential_geometry:
            return

        if not hasattr(surface, 'H'):  # Check if differential geometry is available
            return

        # Extract differential geometry quantities with proper broadcasting
        diff_geom = self._extract_differential_geometry(surface)

        # Add vector fields with normalized scaling
        for name, vectors in diff_geom['vectors_3d'].items():
            if vectors is not None:
                normalized_vectors = normalize_vectors(vectors) * self.vis_config.vector_scale
                color = self.color_palette.get_vector_color(name)
                structure.add_vector_quantity(
                    name=name,
                    values=normalized_vectors,
                    enabled=False,
                    color=color,
                    vectortype="ambient"
                )

        # Add scalar quantities (all disabled by default — user can enable in polyscope UI)
        colormap = self._get_scalar_colormap(structure_type)
        for name, scalars in diff_geom['scalars'].items():
            if scalars is not None:
                structure.add_scalar_quantity(
                    name=name,
                    values=scalars,
                    enabled=False,
                    cmap=colormap
                )

    def _get_scalar_colormap(self, structure_type: str) -> str:
        """Get the appropriate colormap for scalar quantities based on structure type."""
        if structure_type == "mesh":
            return self.vis_config.mesh_scalar_colormap
        elif structure_type == "pointcloud":
            return self.vis_config.pointcloud_scalar_colormap
        else:
            # Fallback to mesh colormap if structure type is unknown
            return self.vis_config.mesh_scalar_colormap

    def _add_normals_to_structure(self, structure, normals: np.ndarray) -> None:
        """Add normal vectors to structure if enabled."""
        if not self.vis_config.enable_normals:
            return

        structure.add_vector_quantity(
            name="normals",
            values=normals * self.vis_config.vector_scale,
            enabled=True,
            color=self.color_palette.get_vector_color("normals"),
            vectortype="ambient"
        )

    def _visualize_parametrization(self, surface: Data, name: str, pos: np.ndarray) -> None:
        """Visualize 2D parametrization domain."""
        if not self.vis_config.enable_parametrization:
            return

        # Extract differential geometry with proper broadcasting
        diff_geom = self._extract_differential_geometry(surface)

        # Create 2D parametrization points (Z = 0)
        param_points = pos.copy()
        param_points[:, 2] = 0

        # Create a temporary config with smaller radius for parametrization
        param_config = replace(self.vis_config, point_radius=self.vis_config.param_radius)

        param_cloud = visualize_point_cloud(
            points=param_points,
            name=f"{name} - Domain",
            vis_config=param_config
        )

        # Add 2D vector quantities
        for vector_name, vectors in diff_geom['vectors_2d'].items():
            if vectors is not None:
                normalized_vectors = normalize_vectors(vectors) * self.vis_config.vector_scale
                color = self.color_palette.get_vector_color(vector_name)
                param_cloud.add_vector_quantity(
                    name=vector_name,
                    values=normalized_vectors,
                    enabled=False,
                    color=color,
                    vectortype="ambient"
                )

    def _add_origin_indicator(self, surface: Data, name: str, translation: np.ndarray) -> None:
        """Add visual indicator for the origin point when in origin-only mode."""
        if not self.is_diff_geom_at_origin_only:
            return

        # Get the origin position using the helper
        origin_3d = self._get_origin_position(surface, translation)

        try:
            origin_indicator = ps.register_point_cloud(
                name=f"{name} - Origin",
                points=origin_3d,
                radius=self.vis_config.point_radius * self.vis_config.origin_indicator_scale,
                enabled=True
            )

            # Color it differently to highlight it's the origin
            origin_indicator.add_color_quantity(
                name="origin_color",
                values=np.array([[1.0, 0.0, 1.0]]),  # Magenta
                enabled=True
            )

            print(f"Origin point for {name} at: {origin_3d[0]}")

        except Exception as e:
            print(f"Could not add origin indicator: {e}")

    def _setup_ui_callback(self) -> None:
        """Register ImGui callback for display controls and metrics."""

        def ui_callback():
            import polyscope.imgui as psim

            needs_rerender = False

            # ── Patch selection ──────────────────────────────────────
            psim.Text("Patch Selection")
            psim.Separator()

            if self._current_surfaces and self._current_surface_names:
                patch_names = self._current_surface_names
                changed, new_idx = psim.Combo(
                    "Patch", self._patch_idx, patch_names)
                if changed:
                    self._patch_idx = new_idx
                    needs_rerender = True

                n_pts = self._current_surfaces[self._patch_idx].pos.shape[0]
                psim.Text(f"  Points: {n_pts}")

            psim.Text("")

            # ── Display mode ─────────────────────────────────────────
            psim.Text("Display Mode")
            psim.Separator()

            changed_mesh, new_mesh = psim.Checkbox("Show Mesh", self._show_mesh)
            if changed_mesh:
                self._show_mesh = new_mesh
                needs_rerender = True

            changed_pc, new_pc = psim.Checkbox("Show Point Cloud", self._show_pointcloud)
            if changed_pc:
                self._show_pointcloud = new_pc
                needs_rerender = True

            changed_rf, new_rf = psim.Checkbox("Show Reference Frame", self._show_reference_frame)
            if changed_rf:
                self._show_reference_frame = new_rf
                needs_rerender = True

            changed_sc, new_sc = psim.ColorEdit3("Surface Color", self._surface_color)
            if changed_sc:
                self._surface_color = list(new_sc)
                needs_rerender = True

            changed_pr, new_pr = psim.SliderFloat(
                "Point Size", self._point_radius, v_min=0.001, v_max=0.05)
            if changed_pr:
                self._point_radius = new_pr
                needs_rerender = True

            psim.Text("")

            # ── Normal display ───────────────────────────────────────
            psim.Text("Normal Display")
            psim.Separator()

            changed_gt, new_gt = psim.Checkbox(
                "GT Normal at Origin", self._show_gt_normal_origin)
            if changed_gt:
                self._show_gt_normal_origin = new_gt
                needs_rerender = True

            if self.trained_model is not None:
                changed_pred, new_pred = psim.Checkbox(
                    "Predicted Normal at Origin", self._show_pred_normal_origin)
                if changed_pred:
                    self._show_pred_normal_origin = new_pred
                    needs_rerender = True

            changed_all, new_all = psim.Checkbox(
                "GT Normals (all points)", self._show_gt_normals_all)
            if changed_all:
                self._show_gt_normals_all = new_all
                needs_rerender = True

            # Normal appearance controls
            psim.Text("")
            changed_len, new_len = psim.SliderFloat(
                "Normal Length", self._normal_length, v_min=0.1, v_max=10.0)
            if changed_len:
                self._normal_length = new_len
                needs_rerender = True

            changed_nr, new_nr = psim.SliderFloat(
                "Normal Width", self._normal_radius, v_min=0.001, v_max=0.05)
            if changed_nr:
                self._normal_radius = new_nr
                needs_rerender = True

            changed_gtc, new_gtc = psim.ColorEdit3(
                "GT Normal Color", self._gt_normal_color)
            if changed_gtc:
                self._gt_normal_color = list(new_gtc)
                needs_rerender = True

            if self.trained_model is not None:
                changed_prc, new_prc = psim.ColorEdit3(
                    "Pred Normal Color", self._pred_normal_color)
                if changed_prc:
                    self._pred_normal_color = list(new_prc)
                    needs_rerender = True

            changed_dot, new_dot = psim.ColorEdit3(
                "Origin Dot Color", self._origin_dot_color)
            if changed_dot:
                self._origin_dot_color = list(new_dot)
                needs_rerender = True

            psim.Text("")

            # ── Dataset settings ─────────────────────────────────────
            psim.Text("Dataset Settings")
            psim.Separator()

            changed_origin, new_origin = psim.Checkbox(
                "Diff Geom at Origin Only", self._diff_geom_at_origin_only)
            if changed_origin:
                self.toggle_diff_geom_at_origin_only()

            changed_incl, new_incl = psim.Checkbox(
                "Include Origin in Grid", self._include_origin_in_grid)
            if changed_incl:
                self.toggle_include_origin_in_grid()

            psim.Text("")

            # ── Surface metrics ──────────────────────────────────────
            psim.Text("Surface Metrics")
            psim.Separator()

            if self._current_surfaces and self._patch_idx < len(self.surface_metrics):
                metrics = self.surface_metrics[self._patch_idx]
                H = metrics.get('mean_curvature_at_origin')
                if H is not None:
                    psim.Text(f"GT Mean Curvature: {H:.6f}")
                else:
                    psim.Text("GT Mean Curvature: N/A")

                pred = metrics.get('prediction_metrics')
                if pred:
                    pred_H = pred.get('predicted_mean_curvature')
                    if pred_H is not None:
                        psim.Text(f"Pred Mean Curvature: {pred_H:.6f}")
                        if H is not None:
                            psim.Text(f"Curvature Error: {abs(pred_H - H):.6f}")
                    cosim = pred.get('cosine_similarity')
                    if cosim is not None:
                        psim.Text(f"Normal Cosine Sim: {cosim:.4f}")
                    ang = pred.get('angular_error')
                    if ang is not None:
                        psim.Text(f"Angular Error: {ang:.2f} deg")

            # ── Screenshot ────────────────────────────────────────────
            psim.Text("")
            psim.Separator()
            if psim.Button("Save All Screenshots"):
                self._save_all_screenshots()

            # ── Rerender if needed ───────────────────────────────────
            if needs_rerender:
                self._render()

        ps.set_user_callback(ui_callback)

    def _extract_mean_curvature_at_origin(self, surface: Data) -> Optional[float]:
        """
        Extract the mean curvature value at the origin from the surface data.

        Args:
            surface: Surface data object

        Returns:
            Mean curvature value at origin, or None if not available
        """
        if not hasattr(surface, 'H'):
            return None

        H_tensor = surface.H.detach().cpu()

        # If origin_idx is available, use it to get the exact value at origin
        if hasattr(surface, 'origin_idx'):
            origin_idx = surface.origin_idx.item()
            if origin_idx < H_tensor.numel():
                return H_tensor.flatten()[origin_idx].item()

        # Origin-only mode: H should be a single value
        if H_tensor.numel() == 1:
            return H_tensor.item()

        # Fallback: return first value
        return H_tensor.flatten()[0].item()

    def visualize_surface_set(self, surfaces: List[Data], surface_names: List[str]) -> None:
        """Store surfaces, compute model predictions, then render the current view."""
        self._current_surfaces = surfaces
        self._current_surface_names = surface_names
        self._patch_idx = 0
        self.surface_metrics = []
        self._prediction_cache = {}

        # Compute metrics and predictions for ALL surfaces up front
        for i, (name, surface) in enumerate(zip(surface_names, surfaces)):
            metric = {
                'name': name,
                'num_points': surface.pos.shape[0] if hasattr(surface, 'pos') else None,
                'mean_curvature_at_origin': self._extract_mean_curvature_at_origin(surface),
            }

            # Run model prediction and cache result
            if (self.trained_model is not None and
                    self.vis_config.enable_model_prediction and
                    hasattr(surface, 'normal')):
                try:
                    pred_normal, pred_weights, pred_mcv = predict_normal_from_patch(
                        model=self.trained_model,
                        surface_data=surface,
                        device=self.device)
                    self._prediction_cache[i] = (pred_normal, pred_weights, pred_mcv)

                    # Compute prediction metrics for UI display
                    gt_normal = to_numpy(surface.normal)
                    if gt_normal.shape[0] == 1:
                        gt_at_origin = gt_normal[0]
                    elif hasattr(surface, 'origin_idx'):
                        gt_at_origin = gt_normal[surface.origin_idx.item()]
                    else:
                        gt_at_origin = gt_normal[0]
                    pred_np = to_numpy(pred_normal)
                    pred_H = torch.norm(pred_mcv, p=2).item()

                    gt_t = torch.from_numpy(gt_at_origin).float().to(self.device)
                    cosim = torch.dot(gt_t, pred_normal.float().to(self.device)).item()
                    cosim_c = np.clip(cosim, -1.0, 1.0)
                    ang_err = np.arccos(np.abs(cosim_c)) * 180 / np.pi

                    metric['prediction_metrics'] = {
                        'cosine_similarity': cosim,
                        'angular_error': ang_err,
                        'gt_normal': gt_at_origin,
                        'predicted_normal': pred_np,
                        'predicted_mean_curvature': pred_H,
                    }

                    print(f"  [{name}] pred H={pred_H:.4f}, "
                          f"cos={cosim:.4f}, ang_err={ang_err:.2f}deg")

                except Exception as e:
                    print(f"  [{name}] prediction failed: {e}")

            self.surface_metrics.append(metric)

        # Setup UI and render
        self._setup_ui_callback()
        self._render()

    def _render(self) -> None:
        """Clear and redraw the scene based on current UI state."""
        ps.remove_all_structures()
        if self._show_reference_frame:
            add_reference_frame(vis_config=self.vis_config)

        if not self._current_surfaces:
            return

        idx = self._patch_idx
        surface = self._current_surfaces[idx]
        name = self._current_surface_names[idx]

        # Extract data (no translation — everything at origin)
        pos, face, normals = self._extract_surface_data(surface)

        # ── Mesh ─────────────────────────────────────────────────
        if self._show_mesh:
            mesh = ps.register_surface_mesh(
                name=f"{name} - Mesh",
                vertices=pos, faces=face,
                smooth_shade=self.vis_config.smooth_shade,
                edge_width=self.vis_config.edge_width,
                color=tuple(self._surface_color),
                transparency=0.7)
            self._add_vector_quantities(mesh, surface, "mesh")
            if self._show_gt_normals_all:
                self._add_normals_to_structure(mesh, normals)

        # ── Point cloud ──────────────────────────────────────────
        if self._show_pointcloud:
            cloud = ps.register_point_cloud(
                f"{name} - Point Cloud", pos,
                radius=self._point_radius, enabled=True)
            darker = tuple(c * 0.5 for c in self._surface_color)
            cloud.set_color(darker)
            self._add_vector_quantities(cloud, surface, "pointcloud")
            if self._show_gt_normals_all:
                self._add_normals_to_structure(cloud, normals)

        # ── Origin indicator ─────────────────────────────────────
        translation = np.zeros(3)
        self._add_origin_indicator(surface, name, translation)

        # ── GT normal at origin ──────────────────────────────────
        if self._show_gt_normal_origin and hasattr(surface, 'normal'):
            origin_3d = self._get_origin_position(surface, translation)
            gt_normal = to_numpy(surface.normal)
            if gt_normal.shape[0] == 1:
                gt_at_origin = gt_normal[0]
            elif hasattr(surface, 'origin_idx'):
                gt_at_origin = gt_normal[surface.origin_idx.item()]
            else:
                gt_at_origin = gt_normal[0]
            normal_scale = self.vis_config.vector_scale * self._normal_length
            gt_color = tuple(self._gt_normal_color)
            dot_color = tuple(self._origin_dot_color)

            gt_cloud = ps.register_point_cloud(
                f"{name} - GT Normal", origin_3d,
                radius=self._normal_radius,
                color=dot_color, enabled=True)
            gt_cloud.add_vector_quantity(
                "GT Normal",
                gt_at_origin.reshape(1, 3) * normal_scale,
                enabled=True,
                color=gt_color,
                radius=self._normal_radius,
                vectortype="ambient")

        # ── Predicted normal at origin ───────────────────────────
        if (self._show_pred_normal_origin and
                idx in self._prediction_cache):
            pred_normal, pred_weights, pred_mcv = self._prediction_cache[idx]
            pred_np = to_numpy(pred_normal)
            origin_3d = self._get_origin_position(surface, translation)
            normal_scale = self.vis_config.vector_scale * self._normal_length
            pred_color = tuple(self._pred_normal_color)
            dot_color = tuple(self._origin_dot_color)

            pred_cloud = ps.register_point_cloud(
                f"{name} - Pred Normal", origin_3d,
                radius=self._normal_radius,
                color=dot_color, enabled=True)
            pred_cloud.add_vector_quantity(
                "Predicted Normal",
                pred_np.reshape(1, 3) * normal_scale,
                enabled=True,
                color=pred_color,
                radius=self._normal_radius,
                vectortype="ambient")

    def _save_all_screenshots(self) -> None:
        """Save screenshots for all display combos with transparent background."""
        import os
        from datetime import datetime

        if not self._current_surfaces:
            print("No surfaces loaded — cannot save screenshots.")
            return

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"surface_patch_screenshots_{timestamp}")
        out_dir.mkdir(exist_ok=True)

        # Save current state
        saved = (self._patch_idx, self._show_mesh, self._show_pointcloud,
                 self._show_gt_normal_origin, self._show_pred_normal_origin)

        has_pred = bool(self._prediction_cache)
        n_patches = len(self._current_surfaces)

        display_modes = [
            ("surface", True, False),
            ("pointcloud", False, True),
            ("surface_and_pointcloud", True, True),
        ]

        normal_modes = [
            ("no_normals", False, False),
            ("gt_normal", True, False),
        ]
        if has_pred:
            normal_modes.append(("pred_normal", False, True))
            normal_modes.append(("gt_and_pred", True, True))

        total = n_patches * len(display_modes) * len(normal_modes)
        count = 0

        print(f"\nSaving {total} screenshots to {out_dir}/...")

        for patch_idx in range(n_patches):
            patch_name = self._current_surface_names[patch_idx]
            # Sanitize name for filename
            patch_label = patch_name.replace(" ", "_").replace("(", "").replace(")", "")

            for disp_label, show_mesh, show_pc in display_modes:
                for norm_label, show_gt, show_pred in normal_modes:
                    # Set state
                    self._patch_idx = patch_idx
                    self._show_mesh = show_mesh
                    self._show_pointcloud = show_pc
                    self._show_gt_normal_origin = show_gt
                    self._show_pred_normal_origin = show_pred

                    # Render and screenshot
                    self._render()
                    filename = f"{patch_label}_{disp_label}_{norm_label}.png"
                    filepath = str(out_dir / filename)
                    ps.screenshot(filepath, transparent_bg=True)

                    count += 1
                    print(f"  [{count}/{total}] {filename}")

        # Restore original state
        (self._patch_idx, self._show_mesh, self._show_pointcloud,
         self._show_gt_normal_origin, self._show_pred_normal_origin) = saved
        self._render()

        print(f"Done! {total} screenshots saved to {out_dir}/")


def setup_polyscope() -> None:
    """Initialize and configure polyscope."""
    ps.init()
    ps.set_up_dir("z_up")
    ps.look_at(camera_location=[2.4, 2, 3.9], target=[0, 0, 0])
    ps.set_ground_plane_mode("none")
    ps.set_SSAA_factor(4)

    # Set black background
    ps.set_background_color((0.0, 0.0, 0.0))


def create_custom_visualization_config(
        vector_scale: float = 0.1,
        point_radius: float = 0.01,
        param_radius: float = 0.002,
        surface_spacing_factor: float = 2.5,
        enable_mesh: bool = True,
        enable_point_cloud: bool = True,
        enable_parametrization: bool = True,
        enable_normals: bool = True,
        enable_differential_geometry: bool = True,
        enable_model_prediction: bool = True,  # NEW
        smooth_shade: bool = True,
        edge_width: float = 0.0,
        mesh_scalar_colormap: str = 'coolwarm',
        pointcloud_scalar_colormap: str = 'coolwarm',
        pointcloud_color: Tuple[float, float, float] = (0.0, 0.8, 0.0),
) -> VisualizationConfig:
    """Create custom visualization configuration."""
    return VisualizationConfig(
        vector_scale=vector_scale,
        point_radius=point_radius,
        param_radius=param_radius,
        surface_spacing_factor=surface_spacing_factor,
        enable_mesh=enable_mesh,
        enable_point_cloud=enable_point_cloud,
        enable_parametrization=enable_parametrization,
        enable_normals=enable_normals,
        enable_differential_geometry=enable_differential_geometry,
        enable_model_prediction=enable_model_prediction,
        smooth_shade=smooth_shade,
        edge_width=edge_width,
        mesh_scalar_colormap=mesh_scalar_colormap,
        pointcloud_scalar_colormap=pointcloud_scalar_colormap,
        pointcloud_color=pointcloud_color,
    )


@hydra.main(version_base="1.2", config_path="visualization_config")
def main(cfg: DictConfig) -> None:
    """Main visualization function with optional model prediction and mean curvature display."""

    # Get checkpoint path from config if specified
    ckpt_path = getattr(cfg, 'ckpt_path', None)

    # Set random seed for reproducibility
    pl.seed_everything(cfg.globals.seed)

    # Initialize data module and loader
    data_module = hydra.utils.instantiate(cfg.data_module)
    data_loader = data_module.train_dataloader()

    # Setup polyscope
    setup_polyscope()

    # Determine device for model loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load trained model if checkpoint provided
    trained_model = None
    if ckpt_path:
        try:
            ckpt_path_obj = Path(ckpt_path)
            trained_model = load_trained_model(ckpt_path_obj, device)
            print(f"[OK] Successfully loaded model from {ckpt_path}")
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            print("Continuing with visualization-only mode...")
            trained_model = None

    # Create custom visualization config
    vis_config = create_custom_visualization_config(
        vector_scale=0.15,  # Slightly larger vectors
        point_radius=0.008,  # Slightly smaller points
        surface_spacing_factor=2.0,  # Closer surface spacing
        enable_mesh=True,
        enable_point_cloud=True,
        enable_parametrization=True,
        enable_differential_geometry=True,
        enable_model_prediction=(trained_model is not None),  # Enable only if model loaded
        mesh_scalar_colormap='coolwarm',
        pointcloud_scalar_colormap='coolwarm'
    )

    # Create visualizer with optional model and data module
    visualizer = SurfaceVisualizer(
        config=cfg,
        vis_config=vis_config,
        trained_model=trained_model,
        device=device,
        data_module=data_module
    )

    print(f"\n{'=' * 80}")
    print("SURFACE VISUALIZATION WITH OPTIONAL MODEL PREDICTION")
    print('=' * 80)
    if trained_model is not None:
        print("[Model] MODE: Model prediction comparison enabled")
        print("   - Cyan arrows: Ground truth analytic normals")
        print("   - Orange arrows: Predicted normals from trained model")
        print("   - Console output: Angular errors, similarity metrics, and mean curvature values")
        print("   - UI window: Real-time surface metrics display")
    else:
        print(" MODE: Visualization only")
        print("   - Use ckpt_path=path/to/model.ckpt to enable model predictions")
        print("   - Console output: Mean curvature values at origin")
        print("   - UI window: Real-time mean curvature display")
    print('=' * 80)

    # Process batches
    for batch_idx, surfaces in enumerate(data_loader):
        print(f"\nProcessing batch {batch_idx + 1}")

        surface_names = visualizer._get_surface_names(surfaces)

        # Store surfaces, compute predictions, render first view
        visualizer.visualize_surface_set(surfaces, surface_names)

        print(f"\n[OK] Batch {batch_idx + 1} visualization complete!")
        print("   Use the UI panel to switch patches, toggle normals, etc.")
        print("   Close the window to proceed to the next batch, or Ctrl+C to exit")

        # Show visualization (blocks until user closes window)
        ps.show()


if __name__ == "__main__":
    main()