# Standard library
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Third-party libraries
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from scipy.spatial import cKDTree
import robust_laplacian
import scipy.sparse

# Local imports
from neural_local_laplacian.datasets.base_datasets import PoseType
from neural_local_laplacian.utils.pose_transformers import PoseTransformer
from neural_local_laplacian.utils import utils
from neural_local_laplacian.utils.utils import load_mesh_list_from_lookup, load_mesh
from neural_local_laplacian.utils.geodesic_utils import (
    compute_exact_geodesics,
    select_multiple_geodesic_sources,
    load_cached_geodesics,
)


class MeshPatchData(Data):
    """
    Data object that preserves internal patch structure during PyG batching.

    Uses patch_idx instead of batch to store patch assignments, preventing
    PyG's DataLoader from resetting the indices.
    """

    def __cat_dim__(self, key, value, *args, **kwargs):
        # patch_idx concatenates along dim 0 (standard behavior)
        if key == 'patch_idx':
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # Don't increment patch_idx - preserve internal patch structure
        if key == 'patch_idx':
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class MeshDataset(Dataset):
    """
    Dataset for loading real 3D meshes and extracting local surface patches.

    Requires a pre-built lookup table (.mesh_properties_cache.json) in each
    mesh folder. Run preprocess_mesh_folder.py to create it.
    """

    def __init__(
            self,
            mesh_folder_paths: Union[str, Path, List[Union[str, Path]]] = None,
            k: int = 20,
            num_eigenvalues: int = 20,
            pose_transformers: Optional[List[PoseTransformer]] = None,
            num_geodesic_sources: int = 5,
            geodesic_source_method: str = "farthest_point_sampling",
            max_meshes: Optional[int] = None,
            file_size_range_mb: Optional[Tuple[float, float]] = None,
            vertices_count_range: Optional[Tuple[int, int]] = None,
            faces_count_range: Optional[Tuple[int, int]] = None,
            require_manifold: bool = False,
            shuffle: bool = False,
            require_complete_geodesics: bool = False,
            verbose_diagnostics: bool = False,
            skip_eigendecomposition: bool = False,
            # Backward compatibility: singular form still accepted
            mesh_folder_path: Union[str, Path] = None
    ):
        """
        Initialize the MeshDataset.

        Args:
            mesh_folder_paths: Path or list of paths to folders containing mesh files.
                Also accepts 'mesh_folder_path' (singular) for backward compatibility.
            k: Number of nearest neighbors to extract (excluding center point)
            num_eigenvalues: Number of eigenvalues/eigenvectors to compute for ground truth
            pose_transformers: Optional list of pose transformations to apply sequentially
            num_geodesic_sources: Number of source vertices for geodesic validation (default: 5)
            geodesic_source_method: Method for selecting sources:
                - "farthest_point_sampling": Well-distributed sources (default)
                - "random": Random vertices
                - "mixed": Centroid + random
            max_meshes: Optional maximum number of mesh files to load. Applied after
                optional shuffling.
            file_size_range_mb: Optional (min_mb, max_mb) tuple. Mesh files outside this
                size range on disk are skipped. Use None for no filtering.
            vertices_count_range: Optional (min_verts, max_verts) tuple. Meshes with vertex
                count outside this range are skipped. Requires loading each mesh. Use None
                for no filtering.
            faces_count_range: Optional (min_faces, max_faces) tuple. Meshes with face count
                outside this range are skipped. Requires loading each mesh. Use None for no
                filtering.
            require_manifold: If True, skip non-manifold meshes (meshes with
                non-manifold edges/vertices or multiple connected components).
                Manifolds with boundary are allowed. Requires the lookup table
                to contain 'is_manifold' (run preprocess_mesh_folder.py first).
            shuffle: If True, shuffle the mesh file list before applying max_meshes cap.
                Useful for random subsampling from a large collection.
            require_complete_geodesics: If True, skip meshes where precomputed exact
                geodesics are missing or incomplete (geodesics_num_ok < geodesics_num_sources
                in the lookup table). Requires running preprocess_mesh_folder.py first.
            verbose_diagnostics: If True, print detailed diagnostic messages before each
                native call during mesh loading (useful for debugging segfaults).
            skip_eigendecomposition: If True, skip GT eigendecomposition during loading.
                Useful when the consumer (e.g. quantitative_eval.py) computes its own
                eigendecomposition and only needs mesh paths + geodesic data.
        """
        super().__init__()

        # Backward compatibility: accept mesh_folder_path (singular) as alias
        if mesh_folder_paths is None and mesh_folder_path is not None:
            mesh_folder_paths = mesh_folder_path
        elif mesh_folder_paths is None and mesh_folder_path is None:
            raise ValueError("Must provide mesh_folder_paths (or mesh_folder_path)")

        # Normalize to list of Paths (support single path for backward compat)
        if isinstance(mesh_folder_paths, (str, Path)):
            mesh_folder_paths = [mesh_folder_paths]
        self._mesh_folder_paths = [Path(p) for p in mesh_folder_paths]

        self._k = k
        self._num_eigenvalues = num_eigenvalues
        self._pose_transformers = pose_transformers if pose_transformers is not None else []
        self._num_geodesic_sources = num_geodesic_sources
        self._geodesic_source_method = geodesic_source_method
        self._max_meshes = max_meshes
        self._file_size_range_mb = file_size_range_mb
        self._vertices_count_range = vertices_count_range
        self._faces_count_range = faces_count_range
        self._require_manifold = require_manifold
        self._shuffle = shuffle
        self._require_complete_geodesics = require_complete_geodesics
        self._verbose = verbose_diagnostics
        self._skip_eigendecomposition = skip_eigendecomposition

        # Validate inputs
        self._validate_inputs()

        # Load mesh list from existing lookup table (no scanning)
        self._mesh_file_paths = load_mesh_list_from_lookup(
            folder_paths=self._mesh_folder_paths,
            file_size_range_mb=self._file_size_range_mb,
            vertices_count_range=self._vertices_count_range,
            faces_count_range=self._faces_count_range,
            require_manifold=self._require_manifold,
            max_meshes=self._max_meshes,
            shuffle=self._shuffle,
            require_complete_geodesics=self._require_complete_geodesics,
        )

        if len(self._mesh_file_paths) == 0:
            folders_str = ", ".join(str(p) for p in self._mesh_folder_paths)
            raise ValueError(f"No mesh files found in: {folders_str}")

        print(f"Found {len(self._mesh_file_paths)} mesh files across "
              f"{len(self._mesh_folder_paths)} folder(s):")
        for i, fp in enumerate(self._mesh_file_paths):
            size_mb = fp.stat().st_size / (1024 * 1024)
            print(f"  [{i:>3d}] {fp.name} ({size_mb:.2f} MB)")

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        for folder_path in self._mesh_folder_paths:
            if not folder_path.exists():
                raise ValueError(f"Mesh folder does not exist: {folder_path}")
            if not folder_path.is_dir():
                raise ValueError(f"Mesh folder path is not a directory: {folder_path}")

        if self._k < 1:
            raise ValueError(f"k must be >= 1, got {self._k}")

        if self._max_meshes is not None and self._max_meshes < 1:
            raise ValueError(f"max_meshes must be >= 1, got {self._max_meshes}")

        if self._file_size_range_mb is not None:
            min_mb, max_mb = self._file_size_range_mb
            if min_mb < 0:
                raise ValueError(f"file_size_range_mb min must be >= 0, got {min_mb}")
            if max_mb <= min_mb:
                raise ValueError(f"file_size_range_mb max ({max_mb}) must be > min ({min_mb})")

        if self._vertices_count_range is not None:
            min_v, max_v = self._vertices_count_range
            if min_v < 0:
                raise ValueError(f"vertices_count_range min must be >= 0, got {min_v}")
            if max_v <= min_v:
                raise ValueError(f"vertices_count_range max ({max_v}) must be > min ({min_v})")

        if self._faces_count_range is not None:
            min_f, max_f = self._faces_count_range
            if min_f < 0:
                raise ValueError(f"faces_count_range min must be >= 0, got {min_f}")
            if max_f <= min_f:
                raise ValueError(f"faces_count_range max ({max_f}) must be > min ({min_f})")

    def len(self) -> int:
        """Return the number of mesh files in the dataset."""
        return len(self._mesh_file_paths)

    def get(self, idx: int) -> Data:
        """
        Load and process a mesh file, returning raw mesh data.

        Does NOT compute k-NN or extract patches — that's done by the consumer
        (model validation_step or visualize_validation) using build_patches_from_vertices(),
        which can run on GPU for much faster neighbor search.

        Args:
            idx: Index of the mesh file to load

        Returns:
            Data object with raw mesh data:
            - raw_vertices: (N, 3) normalized vertex positions
            - raw_faces: (F, 3) face indices
            - vertex_normals: (N, 3) per-vertex normals
            - gt_eigen: tuple of (eigenvalues, eigenvectors) or (None, None)
            - geodesic_data: dict with source_indices + exact geodesics
            - mesh_file_path, original_num_vertices, etc.
        """
        if idx >= len(self._mesh_file_paths):
            raise IndexError(f"Index {idx} out of range for {len(self._mesh_file_paths)} mesh files")

        mesh_file_path = self._mesh_file_paths[idx]
        _v = self._verbose
        if _v: print(f"  [MeshDataset] Loading mesh [{idx}]: {mesh_file_path.name}", flush=True)

        try:
            # Load mesh using shared loader
            if _v: print(f"    [diag] load_mesh ...", flush=True)
            vertices, faces, vertex_normals = load_mesh(str(mesh_file_path))
            original_num_vertices = len(vertices)
            original_num_faces = len(faces)

            if _v: print(f"    [diag] Loaded: {original_num_vertices} verts, {original_num_faces} faces", flush=True)

            # Compute ground-truth Laplacian eigendecomposition using robust-laplacian
            if self._skip_eigendecomposition:
                gt_eigenvalues, gt_eigenvectors = None, None
            else:
                if _v: print(f"    [diag] GT eigendecomposition ...", flush=True)
                gt_eigenvalues, gt_eigenvectors = self._compute_ground_truth_eigendecomposition(
                    vertices, faces
                )

        except Exception as e:
            raise RuntimeError(f"Failed to load mesh {mesh_file_path}: {e}")

        # Check if we have enough vertices for the intended k
        if len(vertices) < self._k + 1:
            raise ValueError(f"Mesh has only {len(vertices)} vertices, need at least {self._k + 1}")

        # Compute geodesic validation data (source selection + exact geodesics, no pcdiff operators)
        if _v: print(f"    [diag] geodesic_validation_data ...", flush=True)
        geodesic_data = self._compute_geodesic_validation_data(
            vertices, faces, mesh_file_path=str(mesh_file_path),
        )

        # Build Data object with raw mesh information
        data = Data()
        data.raw_vertices = torch.from_numpy(vertices).float()        # (N, 3)
        data.raw_faces = torch.from_numpy(faces).long()               # (F, 3)
        data.vertex_normals = torch.from_numpy(vertex_normals).float() # (N, 3)
        data.num_mesh_vertices = torch.tensor([len(vertices)])         # scalar as tensor for batching

        # Metadata
        data.mesh_file_path = str(mesh_file_path)
        data.original_num_vertices = original_num_vertices
        data.original_num_faces = original_num_faces
        data.mesh_idx = idx
        data.k_neighbors = self._k

        # Ground-truth eigendecomposition (stored as tuple, PyG won't concatenate)
        data.gt_eigen = (gt_eigenvalues, gt_eigenvectors)

        # Geodesic data (source indices + exact geodesics, no pcdiff operators)
        data.geodesic_data = geodesic_data

        if _v: print(f"    [diag] Done: {mesh_file_path.name}", flush=True)
        return data

    def _compute_ground_truth_eigendecomposition(
            self,
            vertices: np.ndarray,
            faces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ground-truth Laplacian eigendecomposition using robust-laplacian.

        Args:
            vertices: Mesh vertices of shape (N, 3)
            faces: Mesh faces of shape (F, 3)

        Returns:
            Tuple of (eigenvalues, eigenvectors):
            - eigenvalues: Array of shape (num_eigenvalues,)
            - eigenvectors: Array of shape (N, num_eigenvalues)
        """
        # Compute robust Laplacian (returns Laplacian L and mass matrix M)
        if self._verbose: print(f"    [diag] robust_laplacian.mesh_laplacian ({len(vertices)} verts, {len(faces)} faces) ...", flush=True)
        L, M = robust_laplacian.mesh_laplacian(vertices, faces)

        # Use shared eigendecomposition function
        if self._verbose: print(f"    [diag] eigsh (k={self._num_eigenvalues}) ...", flush=True)
        return utils.compute_laplacian_eigendecomposition(
            L, self._num_eigenvalues, mass_matrix=M
        )

    def _compute_geodesic_validation_data(
            self,
            vertices: np.ndarray,
            faces: np.ndarray,
            mesh_file_path: str = None,
    ) -> dict:
        """
        Load geodesic validation data from cache, or compute on the fly.

        Tries to load from .geodesics_cache/ first (built by
        preprocess_mesh_folder.py). Falls back to on-the-fly computation
        if cache is missing.

        Args:
            vertices: Mesh vertices of shape (N, 3)
            faces: Mesh faces of shape (F, 3)
            mesh_file_path: Path to mesh file (for cache lookup)

        Returns:
            Dictionary with:
            - source_indices: Array of source vertex indices (num_sources,)
            - exact_geodesics: Dict mapping source_idx -> exact distances (N,)
            - has_geodesic_data: Whether geodesic data was successfully computed
        """
        result = {
            'source_indices': None,
            'exact_geodesics': None,
            'has_geodesic_data': False
        }

        # Select sources (deterministic — same seed/method as preprocess script)
        source_indices = select_multiple_geodesic_sources(
            vertices,
            num_sources=self._num_geodesic_sources,
            method=self._geodesic_source_method,
            seed=42
        )
        result['source_indices'] = source_indices

        # Try cache first
        if mesh_file_path is not None:
            cached = load_cached_geodesics(mesh_file_path, len(vertices), source_indices)
            if cached is not None:
                result['exact_geodesics'] = cached
                result['has_geodesic_data'] = True
                if self._verbose:
                    print(f"    [diag] Loaded geodesics from cache ({len(cached)} sources)")
                return result

        # Cache miss — compute on the fly
        exact_geodesics = {}
        for source_idx in source_indices:
            if self._verbose:
                print(f"    [diag] compute_exact_geodesics (source={source_idx}) ...", flush=True)
            geodesics = compute_exact_geodesics(vertices, faces, int(source_idx))
            if geodesics is not None:
                exact_geodesics[int(source_idx)] = geodesics

        if len(exact_geodesics) == 0:
            print(f"Warning: Could not compute exact geodesics for any source, skipping geodesic validation")
            return result

        result['exact_geodesics'] = exact_geodesics
        result['has_geodesic_data'] = True

        return result

    def _extract_all_patches(self, vertices: np.ndarray, vertex_normals: np.ndarray) -> Data:
        """
        Extract local patches for all vertices in the mesh using fully vectorized operations.

        Args:
            vertices: Mesh vertices array of shape (N, 3)
            vertex_normals: Mesh vertex normals array of shape (N, 3)

        Returns:
            Data object containing all local patches
        """
        num_vertices = len(vertices)

        # Build k-NN index using cKDTree
        tree = cKDTree(vertices)
        _, neighbor_indices = tree.query(vertices, k=self._k + 1, workers=-1)  # (N, k+1)
        # First column is self (distance=0), drop it
        neighbor_indices_filtered = neighbor_indices[:, 1:]  # (N, k)

        # Vectorized extraction of neighbor positions
        all_neighbor_positions = vertices[neighbor_indices_filtered]  # Shape: (N, k, 3)

        # Vectorized translation: subtract center from each patch
        center_positions_expanded = vertices[:, np.newaxis, :]  # Shape: (N, 1, 3)
        patch_positions = all_neighbor_positions - center_positions_expanded  # Shape: (N, k, 3)

        # Apply pose transformations if specified (vectorized over patches)
        if self._pose_transformers:
            patch_positions, vertex_normals = self._apply_pose_transformation(
                patch_positions, vertex_normals
            )

        # Extract features for all patches (vectorized)
        all_patch_features = self._extract_patch_features(
            patch_positions, vertex_normals
        )

        # Prepare data for PyTorch Geometric
        # Flatten patch positions and features
        all_positions = patch_positions.reshape(-1, 3)  # Shape: (N*k, 3)
        all_features = all_patch_features.reshape(-1, all_patch_features.shape[-1])  # Shape: (N*k, feature_dim)
        all_neighbor_indices = neighbor_indices_filtered.flatten()  # Shape: (N*k,)
        all_center_indices = np.arange(num_vertices)  # Shape: (N,)

        # Create batch indices - each patch is a separate graph
        batch_indices = np.repeat(range(num_vertices), self._k)  # Shape: (N*k,)

        # Convert to tensors
        pos_tensor = torch.from_numpy(all_positions).float()
        features_tensor = torch.from_numpy(all_features).float()
        patch_idx_tensor = torch.from_numpy(batch_indices).long()
        vertex_indices_tensor = torch.from_numpy(all_neighbor_indices).long()
        center_indices_tensor = torch.from_numpy(all_center_indices).long()

        # Create MeshPatchData object (preserves patch_idx during PyG batching)
        data = MeshPatchData(
            pos=pos_tensor,
            x=features_tensor,
            patch_idx=patch_idx_tensor,  # Use patch_idx instead of batch
            vertex_indices=vertex_indices_tensor,
            center_indices=center_indices_tensor
        )

        return data

    def _apply_pose_transformation(self, patch_positions: np.ndarray, vertex_normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply pose transformations sequentially to batch of patch positions and vertex normals.

        Args:
            patch_positions: Patch positions of shape (N, k, 3) - N patches with k neighbors each
            vertex_normals: Center vertex normals of shape (N, 3) - one normal per patch

        Returns:
            Tuple of (transformed_patch_positions, transformed_vertex_normals)
        """
        if not self._pose_transformers:
            return patch_positions, vertex_normals

        # Convert to tensors for pose transformation (preserve batch structure)
        pos_tensor = torch.from_numpy(patch_positions).float()  # Shape: (N, k, 3)
        normal_tensor = torch.from_numpy(vertex_normals).float()  # Shape: (N, 3)

        # Apply each pose transformer sequentially
        for pose_transformer in self._pose_transformers:
            translation, rotation_matrix = pose_transformer.transform(pos_tensor, normal_tensor)

            # Handle single vs batch case
            if translation.dim() == 1:
                translation = translation.unsqueeze(0)
            if rotation_matrix.dim() == 2:
                rotation_matrix = rotation_matrix.unsqueeze(0)

            # Apply translation and rotation to positions
            pos_tensor = pos_tensor + translation.unsqueeze(1)
            pos_tensor = torch.bmm(pos_tensor, rotation_matrix.transpose(-1, -2))

            # Transform normals for next transformer
            normal_tensor = torch.bmm(normal_tensor.unsqueeze(1), rotation_matrix.transpose(-1, -2)).squeeze(1)

        return pos_tensor.numpy(), normal_tensor.numpy()

    def _extract_patch_features(self, patch_positions: np.ndarray, vertex_normals: np.ndarray) -> np.ndarray:
        """Return raw patch positions as features — feature extraction is handled by the model."""
        return patch_positions

    @property
    def mesh_file_paths(self) -> List[Path]:
        """Get list of mesh file paths."""
        return self._mesh_file_paths.copy()

    @property
    def k(self) -> int:
        """Get the number of neighbors per patch."""
        return self._k