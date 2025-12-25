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

        # Clear existing structures
        ps.remove_all_structures()

        # Add reference frame
        add_reference_frame()

        # Get surface names and visualize
        surface_names = self._get_surface_names(new_surfaces)
        self._current_surfaces = new_surfaces
        self._current_surface_names = surface_names

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

    def _compute_translation(self, surface_index: int, surfaces: List) -> np.ndarray:
        """Compute translation offset for surface visualization, centered around x=0."""
        grid_factor = self._get_grid_factor_from_surfaces(surfaces=surfaces)
        num_surfaces = len(surfaces)

        # Center the surfaces around x=0
        if num_surfaces == 1:
            # Single surface at origin
            x_offset = 0.0
        else:
            # Multiple surfaces: distribute evenly around x=0
            # Calculate the total span and center it
            total_span = (num_surfaces - 1) * self.vis_config.surface_spacing_factor * grid_factor
            start_x = -total_span / 2
            x_offset = start_x + surface_index * self.vis_config.surface_spacing_factor * grid_factor

        return np.array([x_offset, 0, 0])

    def _get_grid_factor_from_surfaces(self, surfaces: List) -> float:
        """Get grid factor from surface data by computing bounding box."""
        if not surfaces:
            return 1.0

        # Use the first surface to estimate grid size
        first_surface = surfaces[0]
        pos = to_numpy(first_surface.pos)

        # Compute bounding box in X and Y dimensions
        x_range = pos[:, 0].max() - pos[:, 0].min()
        y_range = pos[:, 1].max() - pos[:, 1].min()

        # Use the maximum range as grid factor
        return max(x_range, y_range)

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

        # Add scalar quantities with appropriate colormap (only enabled for meshes)
        colormap = self._get_scalar_colormap(structure_type)
        enable_scalars = (structure_type == "mesh")
        for name, scalars in diff_geom['scalars'].items():
            if scalars is not None:
                structure.add_scalar_quantity(
                    name=name,
                    values=scalars,
                    enabled=enable_scalars,
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

    def _add_surface_metrics_ui_callback(self) -> None:
        """Add ImGui callback to display surface metrics in a floating window."""

        def surface_metrics_callback():
            import polyscope.imgui as psim

            # Settings section
            psim.Text("Settings")
            psim.Separator()

            # Checkbox for diff_geom_at_origin_only toggle
            changed, new_value = psim.Checkbox(
                "Diff Geom at Origin Only",
                self._diff_geom_at_origin_only
            )
            if changed:
                self.toggle_diff_geom_at_origin_only()

            # Checkbox for include_origin_in_grid toggle
            changed2, new_value2 = psim.Checkbox(
                "Include Origin in Grid",
                self._include_origin_in_grid
            )
            if changed2:
                self.toggle_include_origin_in_grid()

            psim.Text("")  # Spacing

            # Surface metrics section
            psim.Text("Surface Metrics")
            psim.Separator()
            psim.Text(f"Number of surfaces: {len(self.surface_metrics)}")

            # Simple loop through metrics
            for i, metrics in enumerate(self.surface_metrics):
                surface_name = metrics.get('name', f'Surface {i + 1}')
                mean_curvature = metrics.get('mean_curvature_at_origin')

                psim.Text(f"Surface: {surface_name}")
                if mean_curvature is not None:
                    psim.Text(f"  GT Mean Curvature: {mean_curvature:.6f}")
                else:
                    psim.Text(f"  GT Mean Curvature: Not available")

                # NEW: Display predicted mean curvature if available
                prediction_metrics = metrics.get('prediction_metrics')
                if prediction_metrics:
                    predicted_mean_curvature = prediction_metrics.get('predicted_mean_curvature')
                    if predicted_mean_curvature is not None:
                        psim.Text(f"  Predicted Mean Curvature: {predicted_mean_curvature:.6f}")

                        # Show the difference if both are available
                        if mean_curvature is not None:
                            error = abs(predicted_mean_curvature - mean_curvature)
                            psim.Text(f"  Curvature Error: {error:.6f}")
                    else:
                        psim.Text(f"  Predicted Mean Curvature: Not available")
                else:
                    psim.Text(f"  Predicted Mean Curvature: No model prediction")

                psim.Text("")  # Empty line for spacing

        # Register the callback with polyscope
        ps.set_user_callback(surface_metrics_callback)

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

    def _add_normal_comparison_visualization(self, surface: Data, surface_name: str,
                                             gt_normal: np.ndarray,
                                             predicted_normal: Optional[torch.Tensor],
                                             predicted_mean_curvature_vector: Optional[torch.Tensor],
                                             predicted_weights: Optional[torch.Tensor],
                                             translation: np.ndarray) -> Optional[dict]:
        """
        Add GT vs Predicted normal visualization with comparison metrics and mean curvature display.

        Args:
            surface: Surface data object
            surface_name: Name of the surface for labeling
            gt_normal: Ground truth normal vector (1, 3) or (N, 3) - broadcasted for visualization
            predicted_normal: Predicted normal vector (3,) - single vector at center/origin
            predicted_mean_curvature_vector: Predicted mean curvature vector (3,) = stiffness_sum / area
            predicted_weights: Learned stiffness weights (num_points,) or None
            translation: Translation offset for positioning

        Returns:
            Dict with prediction metrics, or None if visualization skipped
        """
        if not self.vis_config.enable_model_prediction or predicted_normal is None:
            return None

        # Extract the actual GT normal at origin (first row if broadcasted)
        if gt_normal.ndim == 2:
            if self.is_diff_geom_at_origin_only and gt_normal.shape[0] > 1:
                # In origin-only mode, all rows should be identical (broadcasted)
                # Take the first row as the actual GT normal at origin
                gt_normal_at_origin = gt_normal[0]  # Shape: (3,)
            else:
                # Take first row or flatten if shape is (1, 3)
                gt_normal_at_origin = gt_normal[0] if gt_normal.shape[0] > 1 else gt_normal.flatten()
        else:
            # Already shape (3,)
            gt_normal_at_origin = gt_normal

        # Convert predicted normal to numpy
        pred_normal_np = to_numpy(predicted_normal)  # Shape: (3,)

        # Compute predicted mean curvature as the norm of the mean curvature vector
        predicted_mean_curvature = torch.norm(predicted_mean_curvature_vector, p=2).item()

        # Get origin point using consistent helper
        origin_3d = self._get_origin_position(surface, translation)

        # Scale factor for normal visualization
        normal_scale = self.vis_config.vector_scale * self.vis_config.normal_display_scale

        try:
            # GT Normal (Cyan) - single point cloud with vector
            gt_origin_cloud = ps.register_point_cloud(
                name=f"{surface_name} - GT Normal Origin",
                points=origin_3d,
                radius=self.vis_config.normal_origin_point_radius,
                enabled=True
            )

            # Add GT normal as vector quantity
            gt_origin_cloud.add_vector_quantity(
                name="GT Normal",
                values=gt_normal_at_origin.reshape(1, 3) * normal_scale,
                enabled=True,
                color=self.color_palette.NORMALS,
                radius=self.vis_config.normal_origin_point_radius,
                vectortype="ambient"
            )

            # Predicted Normal (Orange) - single point cloud with vector
            pred_origin_cloud = ps.register_point_cloud(
                name=f"{surface_name} - Predicted Normal Origin",
                points=origin_3d,
                radius=self.vis_config.normal_origin_point_radius,
                enabled=True
            )

            # Add predicted normal as vector quantity
            pred_origin_cloud.add_vector_quantity(
                name="Predicted Normal",
                values=pred_normal_np.reshape(1, 3) * normal_scale,
                enabled=True,
                color=self.color_palette.PREDICTED_NORMALS,
                radius=self.vis_config.normal_origin_point_radius,
                vectortype="ambient"
            )

            # Compute and display comparison metrics (compare the single vectors)
            # Ensure both tensors are on the same device for computation
            gt_normal_tensor = torch.from_numpy(gt_normal_at_origin).float().to(self.device)
            pred_normal_tensor = predicted_normal.float().to(self.device)

            cosine_similarity = torch.dot(gt_normal_tensor, pred_normal_tensor).item()

            # Clamp to valid range for arccos
            cosine_similarity_clamped = np.clip(cosine_similarity, -1.0, 1.0)
            angular_error = np.arccos(np.abs(cosine_similarity_clamped)) * 180 / np.pi

            # NEW: Extract and display mean curvature at origin
            mean_curvature_at_origin = self._extract_mean_curvature_at_origin(surface)

            print(f"\n   Normal Comparison for {surface_name}:")
            print(f"    GT Normal (at origin):    [{gt_normal_at_origin[0]:7.4f}, {gt_normal_at_origin[1]:7.4f}, {gt_normal_at_origin[2]:7.4f}]")
            print(f"    Predicted Normal:         [{pred_normal_np[0]:7.4f}, {pred_normal_np[1]:7.4f}, {pred_normal_np[2]:7.4f}]")
            print(f"    Cosine Similarity:        {cosine_similarity:7.4f}")
            print(f"    Angular Error:            {angular_error:7.2f}deg")

            # NEW: Display both GT and predicted mean curvature values
            if mean_curvature_at_origin is not None:
                print(f"     GT Mean Curvature at Origin:     {mean_curvature_at_origin:7.4f}")
            else:
                print(f"     GT Mean Curvature at Origin:     Not available")

            print(f"    [Model] Predicted Mean Curvature:       {predicted_mean_curvature:7.4f}")

            # Curvature comparison if both available
            if mean_curvature_at_origin is not None:
                curvature_error = abs(predicted_mean_curvature - mean_curvature_at_origin)
                curvature_relative_error = curvature_error / (abs(mean_curvature_at_origin) + 1e-10) * 100
                print(f"     Curvature Error:                {curvature_error:7.4f}")
                print(f"     Curvature Relative Error:       {curvature_relative_error:7.2f}%")

            # Add weight statistics if available
            if predicted_weights is not None:
                weights_np = to_numpy(predicted_weights)
                print(f"    Weight Statistics:")
                print(f"      Mean:   {weights_np.mean():7.4f}")
                print(f"      Std:    {weights_np.std():7.4f}")
                print(f"      Min:    {weights_np.min():7.4f}")
                print(f"      Max:    {weights_np.max():7.4f}")
                print(f"      Sum:    {weights_np.sum():7.4f}")

                # Show top 5 weights for insight
                top_indices = np.argsort(weights_np)[-5:][::-1]
                print(f"      Top 5 weights: {weights_np[top_indices]}")

            # OPTIONAL: Add predicted normal as vector field on all points for visualization
            if self.vis_config.enable_point_cloud:
                num_points = surface.pos.shape[0]
                predicted_normals_all_points = np.tile(pred_normal_np, (num_points, 1))

                try:
                    point_cloud_name = f"{surface_name} - Point Cloud"
                    if hasattr(ps, 'get_point_cloud'):
                        pc = ps.get_point_cloud(point_cloud_name)
                        pc.add_vector_quantity(
                            name="Predicted Normal (all points)",
                            values=predicted_normals_all_points * self.vis_config.vector_scale,
                            enabled=False,
                            color=self.color_palette.PREDICTED_NORMALS,
                            vectortype="ambient"
                        )
                        print(f"    Added predicted normal field to all {num_points} points")
                except Exception as e:
                    print(f"    Note: Could not add predicted normal field: {e}")

            # Return metrics instead of mutating surface object
            return {
                'cosine_similarity': cosine_similarity,
                'angular_error': angular_error,
                'gt_normal': gt_normal_at_origin,
                'predicted_normal': pred_normal_np,
                'mean_curvature_at_origin': mean_curvature_at_origin,
                'predicted_mean_curvature': predicted_mean_curvature
            }

        except Exception as e:
            print(f"Warning: Failed to visualize normal comparison for {surface_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def visualize_surface_set(self, surfaces: List[Data], surface_names: List[str]) -> None:
        """Visualize a set of surfaces with their differential geometry and optional model predictions."""
        # Clear previous metrics
        self.surface_metrics = []

        # Print information about the mode
        if self.is_diff_geom_at_origin_only:
            print("Visualizing in origin-only differential geometry mode")
            print("Note: All differential quantities are computed only at the origin (0,0) and broadcasted for visualization")

        # Print model prediction status
        if self.trained_model is not None and self.vis_config.enable_model_prediction:
            print(f"[Model] Model prediction enabled - comparing GT vs predicted normals")
            print(f"   Model device: {self.device}")
        else:
            print(" Visualization only mode - no model predictions")

        for i, (name, surface) in enumerate(zip(surface_names, surfaces)):
            print(f"\nProcessing {name}...")

            # NEW: Extract metrics for UI display
            surface_metric = {
                'name': name,
                'num_points': surface.pos.shape[0] if hasattr(surface, 'pos') else None
            }

            # NEW: Print mean curvature at origin for each surface
            mean_curvature_at_origin = self._extract_mean_curvature_at_origin(surface)
            surface_metric['mean_curvature_at_origin'] = mean_curvature_at_origin

            if mean_curvature_at_origin is not None:
                print(f"    Mean Curvature at Origin: {mean_curvature_at_origin:.6f}")
            else:
                print(f"    Mean Curvature at Origin: Not available")

            # Extract basic data
            pos, face, normals = self._extract_surface_data(surface)

            # Apply translation for side-by-side visualization
            translation = self._compute_translation(surface_index=i, surfaces=surfaces)
            pos_translated = pos + translation

            # Create surface mesh if enabled
            if self.vis_config.enable_mesh:
                mesh = visualize_patch(
                    points=pos_translated,
                    faces=face,
                    name=f"{name} - Mesh",
                    vis_config=self.vis_config
                )
                self._add_normals_to_structure(structure=mesh, normals=normals)
                self._add_vector_quantities(structure=mesh, surface=surface, structure_type="mesh")

            # Create point cloud if enabled
            if self.vis_config.enable_point_cloud:
                cloud = visualize_point_cloud(
                    points=pos_translated,
                    name=f"{name} - Point Cloud",
                    vis_config=self.vis_config,
                    enabled=True
                )
                self._add_normals_to_structure(structure=cloud, normals=normals)
                self._add_vector_quantities(structure=cloud, surface=surface, structure_type="pointcloud")

            # Visualize parametrization domain if available and enabled
            if hasattr(surface, 'v1_2d') and self.vis_config.enable_parametrization:
                self._visualize_parametrization(surface, name, pos_translated)

            # Add origin indicator in origin-only mode
            self._add_origin_indicator(surface, name, translation)

            # NEW: Model prediction and comparison
            if (self.trained_model is not None and
                    self.vis_config.enable_model_prediction and
                    hasattr(surface, 'normal')):

                try:
                    print(f"   Predicting normal using trained model...")

                    # Get ground truth normal (from differential geometry computation)
                    gt_normal = to_numpy(surface.normal)  # Shape: (1, 3) or (N, 3)

                    # Predict normal using trained model
                    predicted_normal, predicted_weights, predicted_mcv = predict_normal_from_patch(
                        model=self.trained_model,
                        surface_data=surface,
                        device=self.device
                    )

                    # Add normal comparison visualization (returns metrics instead of mutating)
                    prediction_metrics = self._add_normal_comparison_visualization(
                        surface=surface,
                        surface_name=name,
                        gt_normal=gt_normal,
                        predicted_normal=predicted_normal,
                        predicted_mean_curvature_vector=predicted_mcv,
                        predicted_weights=predicted_weights,
                        translation=translation
                    )

                    # Store prediction metrics for UI
                    if prediction_metrics is not None:
                        surface_metric['prediction_metrics'] = prediction_metrics

                    print(f"   [OK] Normal prediction completed for {name}")

                except Exception as e:
                    print(f"   [Warning]  Failed to predict normal for {name}: {e}")
                    import traceback
                    traceback.print_exc()

            # Add the surface metrics to our list
            self.surface_metrics.append(surface_metric)

        # NEW: Setup UI callback to display metrics
        self._add_surface_metrics_ui_callback()


def setup_polyscope() -> None:
    """Initialize and configure polyscope."""
    ps.init()
    ps.set_up_dir("z_up")
    ps.look_at(camera_location=[2.4, 2, 3.9], target=[0, 0, 0])
    ps.set_ground_plane_mode("none")

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

        # Add reference frame for each batch
        add_reference_frame()

        # Visualize surfaces (with optional model predictions)
        visualizer.visualize_surface_set(surfaces, surface_names)

        print(f"\n[OK] Batch {batch_idx + 1} visualization complete!")
        if trained_model is not None:
            print("   Check the console output above for normal comparison metrics and mean curvature values")
        else:
            print("   Check the console output above for mean curvature values at origin")
        print("   Check the 'Surface Metrics' window in the UI for real-time display")
        print("   Close the window to proceed to the next batch, or Ctrl+C to exit")

        # Show visualization
        ps.show()

        # Clear for next batch
        ps.remove_all_structures()


if __name__ == "__main__":
    main()