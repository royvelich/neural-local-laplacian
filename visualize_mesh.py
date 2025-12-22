#!/usr/bin/env python3
"""
Mesh Dataset Visualization with Polyscope

This script visualizes mesh datasets by:
1. Rendering the full mesh as a point cloud
2. Overlaying N random patches translated back to their center positions
3. Showing patch features and normal vectors
"""

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import polyscope as ps
import numpy as np
import trimesh
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    point_radius: float = 0.005
    patch_point_radius: float = 0.008
    normal_scale: float = 0.05
    patch_spacing: float = 0.5
    max_patches_per_row: int = 5
    num_patches_to_show: int = 8
    enable_mesh_normals: bool = True
    enable_patch_normals: bool = True
    # Display constants
    center_point_scale: float = 1.5  # Multiplier for center point radius
    camera_distance: float = 2.0
    background_color: Tuple[float, float, float] = (0.05, 0.05, 0.05)


class ColorPalette:
    """Color palette for different visualization elements."""

    # Mesh colors
    MESH_POINTS = (0.7, 0.7, 0.7)  # Light gray
    MESH_NORMALS = (0.0, 1.0, 1.0)  # Cyan

    # Patch colors (different for each patch)
    PATCH_COLORS = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
        (1.0, 0.5, 0.0),  # Orange
        (0.5, 0.0, 1.0),  # Purple
        (0.0, 0.5, 1.0),  # Light Blue
        (1.0, 0.0, 0.5),  # Pink
    ]

    @classmethod
    def get_patch_color(cls, patch_idx: int) -> tuple:
        """Get color for patch visualization."""
        return cls.PATCH_COLORS[patch_idx % len(cls.PATCH_COLORS)]


class MeshDatasetVisualizer:
    """Handles mesh dataset visualization with polyscope."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.color_palette = ColorPalette()

    def load_mesh_vertices_and_normals(self, mesh_file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load mesh vertices and normals directly using trimesh."""
        try:
            mesh = trimesh.load(str(mesh_file_path))
            vertices = np.array(mesh.vertices, dtype=np.float32)
            vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)
            return vertices, vertex_normals
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh {mesh_file_path}: {e}")

    def visualize_full_mesh(self, vertices: np.ndarray, vertex_normals: np.ndarray) -> None:
        """Visualize the full mesh as a point cloud."""
        # Register mesh point cloud
        mesh_cloud = ps.register_point_cloud(
            name="Full Mesh",
            points=vertices,
            radius=self.config.point_radius,
            enabled=True
        )

        # Set mesh color
        mesh_colors = np.tile(self.color_palette.MESH_POINTS, (len(vertices), 1))
        mesh_cloud.add_color_quantity(
            name="mesh_color",
            values=mesh_colors,
            enabled=True
        )

        # Add normals if enabled
        if self.config.enable_mesh_normals:
            mesh_cloud.add_vector_quantity(
                name="normals",
                values=vertex_normals * self.config.normal_scale,
                enabled=False,  # Disabled by default to avoid clutter
                color=self.color_palette.MESH_NORMALS,
                vectortype="ambient"
            )

    def extract_random_patches_from_data(self, data, num_patches: int) -> Tuple[List, List]:
        """Extract random patches from the processed mesh data."""
        # Extract patch information
        positions = data.pos.numpy()  # Shape: (N*k, 3) - flattened patch positions
        center_indices = data.center_indices.numpy()  # Shape: (N,) - center vertex indices

        # Get the number of patches and k (neighbors per patch)
        num_total_patches = len(center_indices)
        k = len(positions) // num_total_patches

        print(f"Total patches in data: {num_total_patches}, k={k}, total points: {len(positions)}")
        print(f"Data.batch unique values: {np.unique(data.batch.numpy())} (DataLoader resets batch indices to 0 for single Data objects)")

        # Randomly sample patches
        if num_patches > num_total_patches:
            print(f"Warning: Requested {num_patches} patches but mesh only has {num_total_patches} vertices. Using all patches.")
            selected_patch_indices = list(range(num_total_patches))
        else:
            selected_patch_indices = np.random.choice(num_total_patches, size=num_patches, replace=False)

        # Extract selected patches by reshaping the flattened positions
        # Even though DataLoader resets batch indices to 0, the sequential order is preserved:
        # [patch0_point0, ..., patch0_pointk-1, patch1_point0, ..., patch1_pointk-1, ...]
        # This is because MeshDataset creates the flattened structure with:
        # all_positions = patch_positions.reshape(-1, 3)  # Shape: (N*k, 3)
        positions_reshaped = positions.reshape(num_total_patches, k, 3)  # Shape: (N, k, 3)

        selected_patches = []
        selected_centers = []

        for patch_idx in selected_patch_indices:
            # Get points for this patch using the preserved sequential order
            patch_points = positions_reshaped[patch_idx]  # Shape: (k, 3)

            print(f"Patch {patch_idx}: extracted {len(patch_points)} points")

            # Get center vertex index
            center_vertex_idx = center_indices[patch_idx]

            selected_patches.append(patch_points)
            selected_centers.append(center_vertex_idx)

        return selected_patches, selected_centers

    def visualize_patches(self, patches: List, center_indices: List, vertices: np.ndarray,
                          vertex_normals: np.ndarray) -> None:
        """Visualize individual patches at their original positions on the mesh."""

        for i, (patch_points, center_idx) in enumerate(zip(patches, center_indices)):
            patch_color = self.color_palette.get_patch_color(i)

            # Get the original center position
            center_pos = vertices[center_idx]  # Shape: (3,)

            # Translate patch back to original position on mesh
            # patch_points are centered at origin, so add back the center position
            original_patch_positions = patch_points + center_pos

            # Register patch point cloud at original position
            patch_name = f"Patch {i + 1} (Center: {center_idx})"
            patch_cloud = ps.register_point_cloud(
                name=patch_name,
                points=original_patch_positions,
                radius=self.config.patch_point_radius,
                enabled=True
            )

            # Set patch color
            patch_colors = np.tile(patch_color, (len(original_patch_positions), 1))
            patch_cloud.add_color_quantity(
                name="patch_color",
                values=patch_colors,
                enabled=True
            )

            # Add center point indicator (make it slightly larger than patch points)
            center_cloud = ps.register_point_cloud(
                name=f"Center {i + 1}",
                points=center_pos.reshape(1, 3),
                radius=self.config.patch_point_radius * self.config.center_point_scale,
                enabled=True
            )
            center_cloud.add_color_quantity(
                name="center_color",
                values=np.array([patch_color]),
                enabled=True
            )

            # Add normal at center if enabled
            if self.config.enable_patch_normals:
                center_normal = vertex_normals[center_idx]
                patch_cloud.add_vector_quantity(
                    name="patch_normal",
                    values=np.tile(center_normal * self.config.normal_scale, (len(original_patch_positions), 1)),
                    enabled=False,  # Disabled by default
                    color=patch_color,
                    vectortype="ambient"
                )


def setup_polyscope(config: Optional[VisualizationConfig] = None) -> None:
    """Initialize and configure polyscope."""
    camera_dist = config.camera_distance if config else 2.0
    bg_color = config.background_color if config else (0.05, 0.05, 0.05)

    ps.init()
    ps.set_up_dir("z_up")
    ps.look_at(camera_location=[camera_dist, camera_dist, camera_dist], target=[0, 0, 0])
    ps.set_ground_plane_mode("none")
    ps.set_background_color(bg_color)


def create_custom_visualization_config(
        point_radius: float = 0.005,
        patch_point_radius: float = 0.008,
        normal_scale: float = 0.05,
        patch_spacing: float = 0.5,
        num_patches_to_show: int = 8,
        **kwargs
) -> VisualizationConfig:
    """Create custom visualization configuration."""
    return VisualizationConfig(
        point_radius=point_radius,
        patch_point_radius=patch_point_radius,
        normal_scale=normal_scale,
        patch_spacing=patch_spacing,
        num_patches_to_show=num_patches_to_show,
        **kwargs
    )


@hydra.main(version_base="1.2", config_path="visualization_config")
def main(cfg: DictConfig) -> None:
    """Main visualization function."""
    # Set random seed for reproducibility
    pl.seed_everything(cfg.globals.seed)

    # Initialize data module and loader
    data_module = hydra.utils.instantiate(cfg.data_module)
    val_data_loaders = data_module.val_dataloader()

    # Use the first validation dataloader (assuming it's a mesh dataset)
    val_data_loader = val_data_loaders[0] if isinstance(val_data_loaders, list) else val_data_loaders

    # Create custom visualization config if needed
    vis_config = create_custom_visualization_config(
        point_radius=0.005,
        patch_point_radius=0.008,
        normal_scale=0.08,
        patch_spacing=0.6,
        num_patches_to_show=10
    )

    # Setup visualization with config
    setup_polyscope(vis_config)

    visualizer = MeshDatasetVisualizer(config=vis_config)

    # Process validation batches
    for batch_data in val_data_loader:
        # batch_data is a list containing a single Data object
        data = batch_data[0]

        # Get the mesh dataset to access original mesh file
        mesh_dataset = val_data_loader.dataset

        # For this example, we'll visualize the first mesh
        # You can modify this to cycle through different meshes
        mesh_idx = 0
        if mesh_idx >= len(mesh_dataset.mesh_file_paths):
            print(f"Mesh index {mesh_idx} out of range, using index 0")
            mesh_idx = 0

        mesh_file_path = mesh_dataset.mesh_file_paths[mesh_idx]
        print(f"Visualizing mesh: {mesh_file_path}")

        # Load original mesh vertices and normals
        vertices, vertex_normals = visualizer.load_mesh_vertices_and_normals(mesh_file_path)
        print(f"Loaded mesh with {len(vertices)} vertices")

        # Visualize full mesh
        visualizer.visualize_full_mesh(vertices, vertex_normals)

        # Extract random patches from the processed data
        patches, center_indices = visualizer.extract_random_patches_from_data(
            data, vis_config.num_patches_to_show
        )

        print(f"Extracted {len(patches)} patches from mesh data")

        # Visualize patches at their original positions on the mesh
        visualizer.visualize_patches(patches, center_indices, vertices, vertex_normals)

        print("Visualization complete! Press any key in the polyscope window to continue...")

        # Show visualization
        ps.show()

        # Clear for next batch (if processing multiple meshes)
        ps.remove_all_structures()

        # For this example, just process the first batch
        break


if __name__ == "__main__":
    main()