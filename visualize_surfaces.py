import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import polyscope as ps
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import torch


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
    smooth_shade: bool
    edge_width: float
    mesh_scalar_colormap: str
    pointcloud_scalar_colormap: str


class ColorPalette:
    """Color palette for different visualization elements."""

    # Principal directions
    PRINCIPAL_V1 = (1.0, 0.0, 0.0)  # Red
    PRINCIPAL_V2 = (0.0, 0.0, 1.0)  # Blue

    # Curvature gradients
    GRAD_MEAN_CURVATURE = (0.0, 1.0, 0.0)  # Green
    GRAD_GAUSSIAN_CURVATURE = (1.0, 1.0, 0.0)  # Yellow

    # Surface properties
    NORMALS = (0.0, 1.0, 1.0)  # Cyan

    # Default colors
    DEFAULT_VECTOR = (0.5, 0.5, 0.5)  # Gray

    @classmethod
    def get_vector_color(cls, vector_name: str) -> Tuple[float, float, float]:
        """Get color for vector visualization."""
        color_map = {
            'v1': cls.PRINCIPAL_V1,
            'v2': cls.PRINCIPAL_V2,
            'grad_H': cls.GRAD_MEAN_CURVATURE,
            'grad_K': cls.GRAD_GAUSSIAN_CURVATURE,
            'v1_2d': cls.PRINCIPAL_V1,
            'v2_2d': cls.PRINCIPAL_V2,
            'grad_H_2d': cls.GRAD_MEAN_CURVATURE,
            'grad_K_2d': cls.GRAD_GAUSSIAN_CURVATURE,
            'normals': cls.NORMALS
        }
        return color_map.get(vector_name, cls.DEFAULT_VECTOR)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vector array to unit length."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.where(norms > 0, vectors / norms, vectors)


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
        enabled=enabled
    )
    return cloud


def compute_diameter(points: np.ndarray) -> float:
    """Compute diameter (maximum pairwise distance) of point cloud."""
    # Using broadcasting for efficient pairwise distance computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.max(distances)


def add_reference_frame(scale: float = 1.0) -> None:
    """Add XYZ reference frame at the origin."""
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
        radius=0.02,
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
            radius=0.01,
            enabled=True
        )

        # Y-axis line
        ps.register_curve_network(
            name="Y-axis",
            nodes=np.array([origin, y_axis]),
            edges=np.array([[0, 1]]),
            color=(0.0, 1.0, 0.0),
            radius=0.01,
            enabled=True
        )

        # Z-axis line
        ps.register_curve_network(
            name="Z-axis",
            nodes=np.array([origin, z_axis]),
            edges=np.array([[0, 1]]),
            color=(0.0, 0.0, 1.0),
            radius=0.01,
            enabled=True
        )
    except Exception as e:
        print(f"Note: Could not create axis lines (polyscope version may not support curve networks): {e}")
        # Fallback: just show the colored points
        pass


class SurfaceVisualizer:
    """Handles surface visualization with differential geometry quantities."""

    def __init__(self, config: DictConfig, vis_config: Optional[VisualizationConfig] = None):
        self.config = config
        self.vis_config = vis_config or VisualizationConfig()
        self.color_palette = ColorPalette()

    def _get_surface_names(self, surfaces: List) -> List[str]:
        """Generate surface names based on configuration."""
        downsampled_count = len(surfaces)
        names = []

        if self.config.data_module.train_dataset_specification.dataset.add_regularized_surface:
            names.append('Full Surface')
            downsampled_count -= 1

        names.extend([f'Downsampled Surface {i + 1}' for i in range(downsampled_count)])
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
        pos = first_surface.pos.detach().cpu().numpy()

        # Compute bounding box in X and Y dimensions
        x_range = pos[:, 0].max() - pos[:, 0].min()
        y_range = pos[:, 1].max() - pos[:, 1].min()

        # Use the maximum range as grid factor
        return max(x_range, y_range)

    def _get_grid_factor(self) -> float:
        """Get grid factor from configuration (deprecated - use _get_grid_factor_from_surfaces)."""
        # Fallback implementation
        return 1.0

    def _extract_surface_data(self, surface) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract position, face, and normal data from surface."""
        pos = surface.pos.detach().cpu().numpy()
        face = surface.face.detach().cpu().numpy().T
        normals = surface.normal.detach().cpu().numpy()
        return pos, face, normals

    def _add_vector_quantities(self, structure, surface, structure_type: str = "default") -> None:
        """Add vector quantities to polyscope structure."""
        if not self.vis_config.enable_differential_geometry:
            return

        if not hasattr(surface, 'H'):  # Check if differential geometry is available
            return

        # Extract differential geometry quantities
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

        # Add scalar quantities with appropriate colormap
        colormap = self._get_scalar_colormap(structure_type)
        for name, scalars in diff_geom['scalars'].items():
            if scalars is not None:
                structure.add_scalar_quantity(
                    name=name,
                    values=scalars,
                    enabled=True,
                    cmap=colormap
                )

    def _extract_differential_geometry(self, surface) -> dict:
        """Extract differential geometry quantities from surface."""

        def safe_extract(attr_name):
            return getattr(surface, attr_name, None).detach().cpu().numpy() if hasattr(surface, attr_name) else None

        return {
            'vectors_3d': {
                'v1': safe_extract('v1_3d'),
                'v2': safe_extract('v2_3d'),
                'grad_H': safe_extract('grad_H_3d'),
                'grad_K': safe_extract('grad_K_3d')
            },
            'vectors_2d': {
                'v1_2d': safe_extract('v1_2d'),
                'v2_2d': safe_extract('v2_2d'),
                'grad_H_2d': safe_extract('grad_H_2d'),
                'grad_K_2d': safe_extract('grad_K_2d')
            },
            'scalars': {
                'Mean Curvature': safe_extract('H'),
                'Gaussian Curvature': safe_extract('K')
            }
        }

    def _get_vector_color(self, vector_name: str) -> Tuple[float, float, float]:
        """Get color for vector visualization (deprecated - use ColorPalette)."""
        return self.color_palette.get_vector_color(vector_name)

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

    def _visualize_parametrization(self, surface, name: str, pos: np.ndarray) -> None:
        """Visualize 2D parametrization domain."""
        if not self.vis_config.enable_parametrization:
            return

        diff_geom = self._extract_differential_geometry(surface)

        # Create 2D parametrization points (Z = 0)
        param_points = pos.copy()
        param_points[:, 2] = 0

        # Create a temporary config with smaller radius for parametrization
        param_config = VisualizationConfig(
            vector_scale=self.vis_config.vector_scale,
            point_radius=self.vis_config.param_radius,  # Use param_radius instead of point_radius
            param_radius=self.vis_config.param_radius,
            surface_spacing_factor=self.vis_config.surface_spacing_factor,
            enable_mesh=self.vis_config.enable_mesh,
            enable_point_cloud=self.vis_config.enable_point_cloud,
            enable_parametrization=self.vis_config.enable_parametrization,
            enable_normals=self.vis_config.enable_normals,
            enable_differential_geometry=self.vis_config.enable_differential_geometry,
            smooth_shade=self.vis_config.smooth_shade,
            edge_width=self.vis_config.edge_width,
            mesh_scalar_colormap=self.vis_config.mesh_scalar_colormap,
            pointcloud_scalar_colormap=self.vis_config.pointcloud_scalar_colormap
        )

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

    def visualize_surface_set(self, surfaces: List, surface_names: List[str]) -> None:
        """Visualize a set of surfaces with their differential geometry."""
        for i, (name, surface) in enumerate(zip(surface_names, surfaces)):
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
        smooth_shade: bool = True,
        edge_width: float = 0.0,
        mesh_scalar_colormap: str = 'coolwarm',
        pointcloud_scalar_colormap: str = 'coolwarm',
        **kwargs
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
        smooth_shade=smooth_shade,
        edge_width=edge_width,
        mesh_scalar_colormap=mesh_scalar_colormap,
        pointcloud_scalar_colormap=pointcloud_scalar_colormap,
        **kwargs
    )


@hydra.main(version_base="1.2", config_path="config")
def main(cfg: DictConfig) -> None:
    """Main visualization function."""
    # Set random seed for reproducibility
    pl.seed_everything(cfg.globals.seed)

    # Initialize data module and loader
    data_module = hydra.utils.instantiate(cfg.data_module)
    data_loader = data_module.train_dataloader()

    # Setup visualization
    setup_polyscope()

    # Create custom visualization config if needed
    # You can modify these parameters to customize the visualization
    vis_config = create_custom_visualization_config(
        vector_scale=0.15,  # Slightly larger vectors
        point_radius=0.008,  # Slightly smaller points
        surface_spacing_factor=2.0,  # Closer surface spacing
        enable_mesh=True,
        enable_point_cloud=True,
        enable_parametrization=True,
        enable_differential_geometry=True,
        mesh_scalar_colormap='coolwarm',  # Different colormap for meshes
        pointcloud_scalar_colormap='coolwarm'  # Different colormap for point clouds
    )

    visualizer = SurfaceVisualizer(config=cfg, vis_config=vis_config)

    # Process batches
    for surfaces in data_loader:
        surface_names = visualizer._get_surface_names(surfaces)

        # Add reference frame for each batch
        add_reference_frame()

        # Visualize surfaces
        visualizer.visualize_surface_set(surfaces, surface_names)

        # Show visualization
        ps.show()

        # Clear for next batch
        ps.remove_all_structures()


if __name__ == "__main__":
    main()