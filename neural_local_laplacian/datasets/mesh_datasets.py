# standard library
from pathlib import Path
from typing import List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass

# numpy
import numpy as np

# torch
import torch

# torch geometric
from torch_geometric.data import Dataset, Data

# igl
import igl

# pymeshlab
import pymeshlab

# scipy
from scipy.spatial import cKDTree

# omegaconf
from omegaconf import DictConfig

# torch geometric
from torch_geometric.nn import knn_graph
from torch_geometric.nn import fps

# neural signatures
from neural_local_laplacian.utils import utils
from neural_local_laplacian.datasets.base_datasets import (
    GridGenerationMethod,
    CoeffGenerationMethod,
    DeformationType,
    PoseType,
    FeaturesType)

# polyscope
import polyscope as ps
# from gdist import local_gdist_matrix

# trimesh
import trimesh


@dataclass
class RemeshConfig:
    method_name: str
    method_kwargs: dict


@dataclass
class GeodesicPatchPair:
    source_patch: np.ndarray
    target_patch: np.ndarray
    mesh_id: int


@dataclass
class GeodesicPatchesDataRecord:
    source_mesh: trimesh.Trimesh
    target_mesh: trimesh.Trimesh
    source_indices: np.ndarray
    target_indices: np.ndarray
    source_distances_matrix: np.ndarray
    target_distances_matrix: np.ndarray
    geodesic_patch_pairs: List[GeodesicPatchPair]


class Shrec19Dataset(Dataset):
    def __init__(
            self,
            pairs_file_path: Union[str, Path],
            meshes_dir_path: Union[str, Path],
            correspondences_dir_path: Union[str, Path],
            conv_k_nearest: int
    ):
        self._pairs_file_path = Path(pairs_file_path)
        self._meshes_dir_path = Path(meshes_dir_path)
        self._correspondences_dir_path = Path(correspondences_dir_path)
        self._conv_k_nearest = conv_k_nearest
        with open(str(pairs_file_path), 'r') as f:
            self._pairs = [line.strip().split(',') for line in f if line.strip()]

        super().__init__()

    def _load_mesh_vertices(self, mesh_id: int) -> torch.Tensor:
        mesh_path = self._meshes_dir_path / f"{mesh_id}.obj"
        vertices, _ = igl.read_triangle_mesh(str(mesh_path))
        return torch.tensor(vertices).to(torch.float32)

    def _load_correspondences(self, source_id: int, target_id: int) -> torch.Tensor:
        corr_path = self._correspondences_dir_path / f"{source_id}_{target_id}.txt"
        with open(str(corr_path), 'r') as f:
            correspondences = np.loadtxt(f, dtype=int) - 1
        return torch.tensor(correspondences)

    def len(self) -> int:
        return len(self._pairs)

    def get(self, idx: int):
        source_id, target_id = self._pairs[idx]
        # print(f'NOW: ===== {self._pairs[idx]}')
        gt_correspondences = self._load_correspondences(source_id=source_id, target_id=target_id)
        mesh_ids = [source_id, target_id]
        meshes = []
        for mesh_id in mesh_ids:
            mesh = Data()
            points = self._load_mesh_vertices(mesh_id=mesh_id)
            points = utils.normalize_to_unit_sphere(points=points)
            center = torch.mean(points, dim=0)
            centered_points = points - center
            normals = utils.estimate_normals(points=centered_points)
            risp = utils.compute_risp_features(points=centered_points, normals=normals)
            edge_index = knn_graph(x=centered_points, k=self._conv_k_nearest, loop=False)

            mesh['pos'] = centered_points
            # mesh['normals'] = normals
            # mesh['x'] = centered_points.unsqueeze(dim=1)
            mesh['x'] = risp
            mesh['edge_index'] = edge_index
            mesh['anchor_indices'] = torch.tensor(list(range(points.shape[0]))).to(torch.int32)
            meshes.append(mesh)

        corr_data = Data(gt_correspondences=gt_correspondences)
        return meshes, [corr_data, Data()]


class RemeshingDataset(Dataset):
    def __init__(
            self,
            root_dirs: List[Union[str, Path]],
            remesh_config: DictConfig,
            file_size_threshold: float = 10
    ):
        self.root_dirs: List[Path] = [Path(root_dir) for root_dir in root_dirs]
        self.file_size_threshold = file_size_threshold  # in MB
        self.remesh_config: DictConfig = remesh_config
        self.file_paths: List[Path] = self._get_file_paths()
        super().__init__()

    def _get_file_paths(self) -> List[Path]:
        all_paths: List[Path] = []
        for root_dir in self.root_dirs:
            for ext in ['*.obj', '*.ply', '*.off']:
                all_paths.extend(root_dir.rglob(ext))

        # Filter out large files
        return [p for p in all_paths if p.stat().st_size / (1024 * 1024) <= self.file_size_threshold]

    def len(self) -> int:
        return len(self.file_paths)

    def get(self, idx: int) -> List[Data]:
        file_path: Path = self.file_paths[idx]
        return self._process_mesh(file_path)

    def _process_mesh(self, file_path: Path) -> List[Data]:
        ms_original: pymeshlab.MeshSet = self._load_mesh(file_path)
        original_vertices: np.ndarray = ms_original.current_mesh().vertex_matrix()
        original_faces: np.ndarray = ms_original.current_mesh().face_matrix()

        data_list: List[Data] = [self._create_data_object(original_vertices, original_faces, 'original')]

        for method_name, method_config in self.remesh_config.items():
            ms_remesh: pymeshlab.MeshSet = pymeshlab.MeshSet()
            ms_remesh.add_mesh(ms_original.current_mesh())
            ms_remesh = self._remesh(ms_remesh, method_name, method_config)
            remeshed_vertices: np.ndarray = ms_remesh.current_mesh().vertex_matrix()
            remeshed_faces: np.ndarray = ms_remesh.current_mesh().face_matrix()

            correspondences, distances = self._get_point_correspondences(original_vertices, remeshed_vertices)

            data: Data = self._create_data_object(remeshed_vertices, remeshed_faces, f'remeshed_{method_name}')
            data.correspondences = torch.tensor(correspondences, dtype=torch.long)
            data.distances = torch.tensor(distances, dtype=torch.float)

            data_list.append(data)

        return data_list

    def _load_mesh(self, file_path: Path) -> pymeshlab.MeshSet:
        ms: pymeshlab.MeshSet = pymeshlab.MeshSet()
        ms.load_new_mesh(str(file_path))  # pymeshlab expects a string path
        return ms

    def _remesh(self, ms: pymeshlab.MeshSet, method_name: str, method_config: dict) -> pymeshlab.MeshSet:
        if hasattr(ms, method_name):
            method = getattr(ms, method_name)
            method(**method_config)
        else:
            raise ValueError(f"Unsupported remeshing method: {method_name}")
        return ms

    def _get_point_correspondences(self, original_vertices: np.ndarray, remeshed_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tree: cKDTree = cKDTree(remeshed_vertices)
        distances, indices = tree.query(original_vertices)
        return indices, distances

    def _create_data_object(self, vertices: np.ndarray, faces: np.ndarray, mesh_type: str) -> Data:
        data: Data = Data(
            pos=torch.tensor(vertices, dtype=torch.float),
            face=torch.tensor(faces.T, dtype=torch.long),  # PyG expects faces to be transposed
            mesh_type=mesh_type
        )
        return data


class GeodesicPatchesDataset(Dataset):
    def __init__(
            self,
            root_dirs: List[Union[str, Path]],
            remesh_configs: List[RemeshConfig],
            file_size_threshold: float,
            file_count_limit: int,
            corresponding_points_count: int,
            features_type: FeaturesType,
            pose_type: PoseType,
            radius: float
    ):
        self._root_dirs = [Path(root_dir) for root_dir in root_dirs]
        self._file_size_threshold = file_size_threshold
        self._file_count_limit = file_count_limit
        self._radius = radius
        self._corresponding_points_count = corresponding_points_count
        self._remesh_configs = remesh_configs
        self._features_type = features_type
        self._pose_type = pose_type
        self._file_paths = self._get_file_paths()[:self._file_count_limit]
        self._geodesic_patches_data_records = self._load_meshes()
        self._geodesic_patch_pairs = []
        for geodesic_patches_data_record in self._geodesic_patches_data_records:
            self._geodesic_patch_pairs = self._geodesic_patch_pairs + geodesic_patches_data_record.geodesic_patch_pairs
        super().__init__()

    def _load_meshes(self) -> List[GeodesicPatchesDataRecord]:
        mesh_records = []
        for file_index, file_path in enumerate(self._file_paths):
            source_mesh = trimesh.load_mesh(file_obj=file_path)
            target_mesh = self._remesh(mesh=source_mesh)
            source_indices, target_indices = self._get_vertex_correspondences(original_mesh=source_mesh, remeshed_mesh=target_mesh)
            source_distances_matrix = local_gdist_matrix(vertices=source_mesh.vertices, triangles=source_mesh.faces.astype(np.int32), max_distance=self._radius)
            target_distances_matrix = local_gdist_matrix(vertices=target_mesh.vertices, triangles=target_mesh.faces.astype(np.int32), max_distance=self._radius)
            source_patches = [self._extract_patch_points(vertices=source_mesh.vertices, distances_matrix=source_distances_matrix, vertex_index=vertex_index) for vertex_index in source_indices]
            target_patches = [self._extract_patch_points(vertices=target_mesh.vertices, distances_matrix=target_distances_matrix, vertex_index=vertex_index) for vertex_index in target_indices]
            geodesic_patch_pairs = [GeodesicPatchPair(source_patch=source_patch, target_patch=target_patch, mesh_id=file_index) for source_patch, target_patch in zip(source_patches, target_patches)]
            geodesic_patches_data_record = GeodesicPatchesDataRecord(
                source_mesh=source_mesh,
                target_mesh=target_mesh,
                source_indices=source_indices,
                target_indices=target_indices,
                source_distances_matrix=source_distances_matrix,
                target_distances_matrix=target_distances_matrix,
                geodesic_patch_pairs=geodesic_patch_pairs
            )
            # self._visualize_with_polyscope(geodesic_patches_data_record=geodesic_patches_data_record)
            # self._visualize_corresponding_patches(geodesic_patches_data_record=geodesic_patches_data_record)
            mesh_records.append(geodesic_patches_data_record)
        return mesh_records

    def _remesh(self, mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
        for remesh_config in self._remesh_configs:
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
            if hasattr(ms, remesh_config.method_name):
                method = getattr(ms, remesh_config.method_name)
                method(**remesh_config.method_kwargs)
            else:
                raise ValueError(f"Unsupported remeshing method: {remesh_config.method_name}")

            mesh = trimesh.Trimesh(ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix())
        return mesh

    def _get_vertex_correspondences(self, original_mesh: trimesh.Trimesh, remeshed_mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        # Randomly sample points from the original mesh vertices
        source_indices = np.random.choice(len(original_mesh.vertices), self._corresponding_points_count, replace=False)
        source_points = original_mesh.vertices[source_indices]

        # Create KD-tree from remeshed mesh vertices
        tree = cKDTree(remeshed_mesh.vertices)

        # Find nearest vertices in remeshed mesh
        _, target_indices = tree.query(source_points)

        return source_indices, target_indices

    def _extract_patch_points(self, vertices: np.ndarray, vertex_index: int, distances_matrix: np.ndarray) -> np.ndarray:
        # Get distances from center vertex to all vertices
        distances = distances_matrix[vertex_index].toarray().flatten()

        # Get vertices within radius (including center vertex)
        valid_vertices = np.where(distances > 0)[0]
        valid_vertices = np.append(valid_vertices, vertex_index)

        # Create tensor of patch points
        patch_points = vertices[valid_vertices]

        return np.array(patch_points).astype(np.float32)

    def _get_file_paths(self) -> List[Path]:
        all_paths: List[Path] = []
        for root_dir in self._root_dirs:
            for ext in ['*.obj', '*.ply', '*.off']:
                all_paths.extend(root_dir.rglob(ext))

        # Filter out large files
        return [p for p in all_paths if p.stat().st_size / (1024 * 1024) <= self._file_size_threshold]

    def len(self) -> int:
        return len(self._geodesic_patch_pairs)

    def get(self, idx: int) -> List[Data]:
        meshes = []
        geodesic_patch_pair = self._geodesic_patch_pairs[idx]
        patches = [geodesic_patch_pair.source_patch, geodesic_patch_pair.target_patch]

        for i, patch in enumerate(patches):
            data = Data(pos=torch.from_numpy(patch))
            center = torch.mean(data.pos, dim=0)
            points = data.pos - center

            if self._pose_type == PoseType.RANDOM_ROTATION:
                rotation_matrix = utils.random_rotation_matrix()
                points = torch.matmul(points, rotation_matrix)
            elif self._pose_type == PoseType.PCA:
                points = utils.compute_canonical_pose_pca(points=points)

            data.pos = points

            data['normals'] = utils.estimate_normals(points=data.pos, k_neighbors=8)
            if self._features_type == FeaturesType.RISP:
                data['x'] = utils.compute_risp_features(points=data.pos, normals=data.normals, k=8)
            elif self._features_type == FeaturesType.XYZ:
                data['x'] = data.pos.unsqueeze(dim=1)

            data['mesh_id'] = geodesic_patch_pair.mesh_id

            meshes.append(data)

        return [meshes]

    def _generate_random_colors(self, n_points: int) -> np.ndarray:
        """Generate random colors for points, ensuring good visibility."""
        colors = np.random.rand(n_points, 3)
        brightness = np.sum(colors, axis=1) / 3
        too_dark = brightness < 0.4
        colors[too_dark] = colors[too_dark] * (0.4 / brightness[too_dark, np.newaxis])
        return colors

    def _visualize_corresponding_points(self, geodesic_patches_data_record: GeodesicPatchesDataRecord):
        # Initialize polyscope
        ps.init()
        ps.set_ground_plane_mode("none")

        # Register original mesh
        ps_original = ps.register_surface_mesh(
            "original",
            geodesic_patches_data_record.source_mesh.vertices,
            geodesic_patches_data_record.source_mesh.faces,
            smooth_shade=True
        )
        ps_original.set_color((0.8, 0.8, 0.8))

        # Register remeshed version
        ps_remeshed = ps.register_surface_mesh(
            "remeshed_isotropic",
            geodesic_patches_data_record.target_mesh.vertices,
            geodesic_patches_data_record.target_mesh.faces,
            smooth_shade=True
        )
        ps_remeshed.set_color((0, 0, 1))  # Blue color for remeshed

        # Generate random colors for point pairs
        n_points = len(geodesic_patches_data_record.source_indices)
        point_colors = self._generate_random_colors(n_points=n_points)

        # Register source points (on original mesh)
        ps_points_source = ps.register_point_cloud(
            "source_points",
            geodesic_patches_data_record.source_mesh.vertices[geodesic_patches_data_record.source_indices]
        )
        ps_points_source.add_color_quantity("random_colors", point_colors)
        ps_points_source.set_radius(0.002)

        # Register target points (vertices of remeshed surface)
        ps_points_target = ps.register_point_cloud(
            "target_points",
            geodesic_patches_data_record.target_mesh.vertices[geodesic_patches_data_record.target_indices]
        )
        ps_points_target.add_color_quantity("random_colors", point_colors)
        ps_points_target.set_radius(0.002)

        ps.show()


    def _visualize_corresponding_patches(self, geodesic_patches_data_record: GeodesicPatchesDataRecord):
        # Initialize polyscope
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("none")

        vertices1, faces1 = geodesic_patches_data_record.source_mesh.vertices, geodesic_patches_data_record.source_mesh.faces
        vertices2, faces2 = geodesic_patches_data_record.target_mesh.vertices, geodesic_patches_data_record.target_mesh.faces

        # Offset second mesh to the right for side-by-side visualization
        offset = np.array([vertices1.max(axis=0)[0] - vertices2.min(axis=0)[0] + 0.2 * (vertices1.max(axis=0)[0] - vertices1.min(axis=0)[0]), 0, 0])
        vertices2_offset = vertices2 + offset

        for geodesic_patch_pair in geodesic_patches_data_record.geodesic_patch_pairs:
            # Register both base meshes
            ps.register_surface_mesh(
                "source mesh",
                vertices1,
                faces1,
                material='flat',
                transparency=0.3,
                color=(0.8, 0.8, 0.8)
            )

            ps.register_surface_mesh(
                "target mesh",
                vertices2_offset,
                faces2,
                material='flat',
                transparency=0.3,
                color=(0.8, 0.8, 0.8)
            )

            ps.register_point_cloud(
                "source patch",
                geodesic_patch_pair.source_patch
            )

            ps.register_point_cloud(
                "target patch",
                geodesic_patch_pair.target_patch + offset
            )

            ps.show()

            ps.remove_all_structures()