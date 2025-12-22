# Standard library
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Third-party libraries
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from sklearn.neighbors import NearestNeighbors
import robust_laplacian

# Trimesh for mesh loading
import trimesh

# Local imports
from neural_local_laplacian.datasets.base_datasets import PoseType
from neural_local_laplacian.utils.features import FeatureExtractor
from neural_local_laplacian.utils.pose_transformers import PoseTransformer
from neural_local_laplacian.utils import utils


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

    For each vertex in a mesh, extracts k nearest neighbors (excluding the center vertex),
    translates the patch so the center would be at origin, and applies feature extraction.
    """

    # Supported mesh file formats
    SUPPORTED_FORMATS = {'.obj', '.ply', '.off', '.stl'}

    def __init__(
            self,
            mesh_folder_path: Union[str, Path],
            k: int,
            num_eigenvalues: int = 20,
            feature_extractor: Optional[FeatureExtractor] = None,
            pose_transformers: Optional[List[PoseTransformer]] = None
    ):
        """
        Initialize the MeshDataset.

        Args:
            mesh_folder_path: Path to folder containing mesh files
            k: Number of nearest neighbors to extract (excluding center point)
            num_eigenvalues: Number of eigenvalues/eigenvectors to compute for ground truth
            feature_extractor: Feature extractor to apply to patch points
            pose_transformers: Optional list of pose transformations to apply sequentially
        """
        super().__init__()

        self._mesh_folder_path = Path(mesh_folder_path)
        self._k = k
        self._num_eigenvalues = num_eigenvalues
        self._feature_extractor = feature_extractor
        self._pose_transformers = pose_transformers if pose_transformers is not None else []

        # Validate inputs
        self._validate_inputs()

        # Scan folder for mesh files
        self._mesh_file_paths = self._scan_mesh_folder()

        if len(self._mesh_file_paths) == 0:
            raise ValueError(f"No mesh files found in {mesh_folder_path}")

        print(f"Found {len(self._mesh_file_paths)} mesh files in {mesh_folder_path}")

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self._mesh_folder_path.exists():
            raise ValueError(f"Mesh folder does not exist: {self._mesh_folder_path}")

        if not self._mesh_folder_path.is_dir():
            raise ValueError(f"Mesh folder path is not a directory: {self._mesh_folder_path}")

        if self._k < 1:
            raise ValueError(f"k must be >= 1, got {self._k}")

    def _scan_mesh_folder(self) -> List[Path]:
        """
        Scan the mesh folder for supported mesh files.

        Returns:
            List of Path objects for found mesh files
        """
        mesh_files = []

        for file_path in self._mesh_folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                mesh_files.append(file_path)

        # Sort for consistent ordering
        mesh_files.sort()
        return mesh_files

    def len(self) -> int:
        """Return the number of mesh files in the dataset."""
        return len(self._mesh_file_paths)

    def get(self, idx: int) -> Data:
        """
        Load and process a mesh file, returning local patches for each vertex.

        Args:
            idx: Index of the mesh file to load

        Returns:
            Data object with all local patches
        """
        if idx >= len(self._mesh_file_paths):
            raise IndexError(f"Index {idx} out of range for {len(self._mesh_file_paths)} mesh files")

        mesh_file_path = self._mesh_file_paths[idx]

        try:
            # Load mesh using trimesh
            mesh = trimesh.load(str(mesh_file_path))

            # Extract vertices as numpy array
            if hasattr(mesh, 'vertices'):
                raw_vertices = np.array(mesh.vertices, dtype=np.float32)
            else:
                raise ValueError(f"Loaded mesh has no vertices: {mesh_file_path}")

            # Store original mesh dimensions before normalization
            original_num_vertices = len(raw_vertices)

            # Extract faces before normalization
            if hasattr(mesh, 'faces'):
                faces = np.array(mesh.faces, dtype=np.int64)
                original_num_faces = len(faces)
            else:
                faces = np.array([])
                original_num_faces = 0

            # Normalize mesh vertices: center at origin and fit in unit sphere
            vertices = utils.normalize_mesh_vertices(raw_vertices)

            # Calculate vertex normals using trimesh with normalized vertices
            # Update the mesh with normalized vertices for normal computation
            mesh.vertices = vertices
            if hasattr(mesh, 'vertex_normals'):
                vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)
            else:
                raise ValueError(f"Could not compute vertex normals for mesh: {mesh_file_path}")

            # Compute ground-truth Laplacian eigendecomposition using robust-laplacian
            gt_eigenvalues, gt_eigenvectors = self._compute_ground_truth_eigendecomposition(
                vertices, faces
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load mesh {mesh_file_path}: {e}")

        # Check if we have enough vertices
        if len(vertices) < self._k + 1:  # Need k neighbors + 1 center
            raise ValueError(f"Mesh has only {len(vertices)} vertices, need at least {self._k + 1}")

        # Extract local patches for all vertices
        patches_data = self._extract_all_patches(vertices, vertex_normals)

        # Add mesh metadata as individual attributes (more robust for PyTorch Geometric batching)
        patches_data.mesh_file_path = str(mesh_file_path)
        patches_data.original_num_vertices = original_num_vertices
        patches_data.original_num_faces = original_num_faces
        patches_data.mesh_idx = idx
        patches_data.normalized_num_vertices = len(vertices)
        patches_data.k_neighbors = self._k

        # Store ground-truth eigendecomposition as a tuple (PyG doesn't concatenate tuples)
        patches_data.gt_eigen = (gt_eigenvalues, gt_eigenvectors)

        return patches_data

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
        L, M = robust_laplacian.mesh_laplacian(vertices, faces)

        # Use shared eigendecomposition function
        return utils.compute_laplacian_eigendecomposition(
            L, self._num_eigenvalues, mass_matrix=M
        )

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

        # Build k-NN index for the entire mesh
        nbrs = NearestNeighbors(n_neighbors=self._k + 1, algorithm='auto').fit(vertices)

        # Get k+1 nearest neighbors for ALL vertices at once
        distances, neighbor_indices = nbrs.kneighbors(vertices)  # Shape: (N, k+1)

        # Vectorized removal of center point from neighbors
        # Create a mask to identify where each vertex appears in its own neighbor list
        center_positions = np.arange(num_vertices)[:, np.newaxis]  # Shape: (N, 1)
        is_center_mask = neighbor_indices == center_positions  # Shape: (N, k+1)

        # Create mask to keep only non-center neighbors
        keep_mask = ~is_center_mask  # Shape: (N, k+1)

        # For each row, we want to keep the first k True values in keep_mask
        # Use cumsum to get the position of each True value
        keep_positions = np.cumsum(keep_mask, axis=1)  # Shape: (N, k+1)
        final_mask = (keep_positions <= self._k) & keep_mask  # Shape: (N, k+1)

        # Extract neighbor indices using the mask
        neighbor_indices_flat = neighbor_indices[final_mask]  # Shape: (N*k,)
        neighbor_indices_filtered = neighbor_indices_flat.reshape(num_vertices, self._k)  # Shape: (N, k)

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
        """
        Extract features for ALL patches using vectorized operations.

        Args:
            patch_positions: Patch positions of shape (N, k, 3)
            vertex_normals: Normal vectors at center vertices, shape (N, 3)

        Returns:
            Feature array of shape (N, k, feature_dim)
        """
        if self._feature_extractor is not None:
            # Use the center vertex normals for feature extraction
            # Expand normals to match patch structure: (N, 3) -> (N, 1, 3)
            center_normals_expanded = vertex_normals[:, np.newaxis, :]  # Shape: (N, 1, 3)

            try:
                # The feature extractor should handle batch processing
                features = self._feature_extractor.extract_features(
                    points=patch_positions,
                    normals=center_normals_expanded
                )
                return features
            except Exception as e:
                import traceback
                print(f"Warning: Feature extraction failed, using positions as features: {e}")
                traceback.print_exc()
                return patch_positions
        else:
            # Fallback: use positions as features
            return patch_positions

    @property
    def mesh_file_paths(self) -> List[Path]:
        """Get list of mesh file paths."""
        return self._mesh_file_paths.copy()

    @property
    def k(self) -> int:
        """Get the number of neighbors per patch."""
        return self._k