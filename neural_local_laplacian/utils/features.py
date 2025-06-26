# standard library
from abc import ABC, abstractmethod

# neural local laplacian
from typing import Optional

# numpy
import numpy as np

# torch
import torch

# scipy
from scipy.spatial import cKDTree


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    def __init__(self,
                 use_fourier: bool = False,
                 fourier_scale: float = 10.0,
                 num_fourier_features: int = 256,
                 distribution: str = 'gaussian',
                 seed: Optional[int] = None):
        """
        Initialize the feature extractor with optional Fourier features.

        Args:
            use_fourier: Whether to use Fourier feature transformation
            fourier_scale: Standard deviation of the distribution used to sample frequencies
            num_fourier_features: Number of Fourier features to generate
            distribution: Distribution to sample frequencies from ('gaussian', 'uniform', 'laplacian')
            seed: Random seed for reproducible frequency matrix initialization (None for random)
        """
        self._use_fourier = use_fourier
        self._fourier_scale = fourier_scale
        self._num_fourier_features = num_fourier_features
        self._distribution = distribution
        self._seed = seed
        self._B = None  # Frequency matrix, initialized when needed

    def extract_features(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        Extract features and apply Fourier transformation if enabled.
        Handles both single patches and batches automatically.

        Args:
            points: Point coordinates of shape (N, 3) or (B, N, 3) for batch processing
            normals: Normal vectors of shape (1, 3) or (B, 1, 3) for batch processing

        Returns:
            Extracted features of shape (N, feature_dim) or (B, N, feature_dim) for batches
        """
        # Determine if input is batched and add batch dimension if needed
        is_single_patch = points.ndim == 2  # Shape: (N, 3)

        if is_single_patch:
            # Add batch dimension
            points_batch = points[np.newaxis, ...]  # Shape: (1, N, 3)

            # Handle different normal shapes for single patch
            if normals.ndim == 1:  # Shape: (3,)
                normals_batch = normals[np.newaxis, np.newaxis, :]  # Shape: (1, 1, 3)
            else:  # Shape: (1, 3)
                normals_batch = normals[np.newaxis, ...]  # Shape: (1, 1, 3)
        else:
            # Already batched
            points_batch = points  # Shape: (B, N, 3)

            # Handle different normal shapes for batch
            if normals.ndim == 2:  # Shape: (B, 3) - single normal per patch
                normals_batch = normals[:, np.newaxis, :]  # Shape: (B, 1, 3)
            else:  # Already shape (B, 1, 3)
                normals_batch = normals

        # Call the implementation-specific batch feature extraction
        features_batch = self._extract_features_batch(points_batch, normals_batch)

        # Apply Fourier transformation if enabled
        if self._use_fourier:
            features_batch = self._apply_fourier_transform_batch(features_batch)

        # Remove batch dimension if input was single patch
        if is_single_patch:
            return features_batch[0]  # Shape: (N, feature_dim)
        else:
            return features_batch  # Shape: (B, N, feature_dim)

    @abstractmethod
    def _extract_features_batch(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        Implementation-specific batch feature extraction (always receives batched inputs).

        Args:
            points: Point coordinates of shape (B, N, 3)
            normals: Normal vectors of shape (B, 1, 3)

        Returns:
            Feature array of shape (B, N, feature_dim)
        """
        pass

    def _apply_fourier_transform_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Apply Fourier feature transform to batch of features.

        Args:
            features: Input features array of shape (B, N, D)

        Returns:
            Transformed features with Fourier features of shape (B, N, 2*num_fourier_features)
        """
        batch_size = features.shape[0]

        # Convert numpy array to torch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # If this is our first call, initialize B
        if self._B is None:
            self._initialize_fourier_matrix(features.shape[-1])  # Use last dimension (feature_dim)

        # Reshape for batch processing: (B, N, D) -> (B*N, D)
        original_shape = features_tensor.shape
        features_flat = features_tensor.view(-1, original_shape[-1])

        # Apply Fourier feature mapping: [cos(2πBx), sin(2πBx)]
        x_proj = 2 * np.pi * torch.matmul(features_flat, self._B)
        fourier_features = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

        # Reshape back to batch format: (B*N, 2*num_fourier) -> (B, N, 2*num_fourier)
        fourier_features = fourier_features.view(original_shape[0], original_shape[1], -1)

        # Convert back to numpy
        return fourier_features.cpu().numpy()

    def _initialize_fourier_matrix(self, input_dim: int):
        """Initialize the matrix of frequencies B with the specified seed for reproducibility."""
        # Set random seed for reproducibility if provided
        if self._seed is not None:
            # Save current random states
            torch_state = torch.get_rng_state()
            np_state = np.random.get_state()

            # Set seeds
            torch.manual_seed(self._seed)
            np.random.seed(self._seed)

        # Generate frequency matrix based on distribution
        if self._distribution == 'gaussian':
            self._B = torch.randn((input_dim, self._num_fourier_features)) * self._fourier_scale
        elif self._distribution == 'uniform':
            self._B = (torch.rand((input_dim, self._num_fourier_features)) * 2 - 1) * self._fourier_scale
        elif self._distribution == 'laplacian':
            # Laplacian distribution via exponential + sign flip
            self._B = torch.sign(torch.rand((input_dim, self._num_fourier_features)) - 0.5) * \
                      torch.tensor(np.random.exponential(
                          scale=self._fourier_scale,
                          size=(input_dim, self._num_fourier_features)
                      ))
        else:
            raise ValueError(f"Unsupported distribution: {self._distribution}")

        # Restore random states if seed was set
        if self._seed is not None:
            torch.set_rng_state(torch_state)
            np.random.set_state(np_state)


class RISPFeatureExtractor(FeatureExtractor):
    """RISP feature extractor with optional Fourier feature transformation."""

    def __init__(self, k: int, **kwargs):
        """
        Initialize RISP feature extractor.

        Args:
            k: Number of neighbors for RISP feature computation (kept for compatibility but not used)
            **kwargs: Fourier feature parameters passed to the base class
        """
        super().__init__(**kwargs)
        self._k = k

    def _compute_angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Compute the angle between two sets of vectors."""
        v1_n = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-10)
        v2_n = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-10)
        return np.arccos(np.clip(np.sum(v1_n * v2_n, axis=-1), -1.0, 1.0))

    def _compute_risp_features_vectorized(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        Compute RISP features for each patch in a vectorized manner.

        :param points: numpy array of shape (B, N, 3) containing point coordinates for B patches
        :param normals: numpy array of shape (B, 1, 3) containing normal vectors at center for each patch
        :return: numpy array of shape (B, N, 14) containing RISP features for each patch
        """
        B, N, _ = points.shape

        # For each patch, we assume points are already the k neighbors of the center
        # The center is implicitly at origin (0, 0, 0) for each patch

        # Extract normals for each patch - shape (B, 3)
        center_normals = normals.squeeze(1)  # Shape: (B, 3)

        # Broadcast normals to all neighbors in each patch
        neighbor_normals = np.broadcast_to(center_normals[:, np.newaxis, :], (B, N, 3))

        # Relative positions are already the points themselves (since center is at origin)
        rel_pos = points  # Shape: (B, N, 3)

        # Project neighbors onto tangent plane at center for each patch
        # Compute dot product of rel_pos with center normals
        dots = np.sum(rel_pos * center_normals[:, np.newaxis, :], axis=2)  # Shape: (B, N)

        # Subtract the normal component to get tangent plane projection
        proj_neighbors = rel_pos - dots[..., np.newaxis] * center_normals[:, np.newaxis, :]  # Shape: (B, N, 3)

        # Compute angles in tangent plane for each patch
        # Create reference direction in tangent plane
        ref_dir_x = np.array([1.0, 0.0, 0.0])
        ref_dir_x = ref_dir_x - np.sum(ref_dir_x * center_normals, axis=1, keepdims=True) * center_normals
        ref_dir_x = ref_dir_x / (np.linalg.norm(ref_dir_x, axis=1, keepdims=True) + 1e-16)

        ref_dir_y = np.cross(ref_dir_x, center_normals)
        ref_dir_y = ref_dir_y / (np.linalg.norm(ref_dir_y, axis=1, keepdims=True) + 1e-16)

        # Compute angles relative to reference direction
        x_coord = np.sum(proj_neighbors * ref_dir_x[:, np.newaxis, :], axis=2)
        y_coord = np.sum(proj_neighbors * ref_dir_y[:, np.newaxis, :], axis=2)
        angles = np.arctan2(y_coord, x_coord)

        # Sort neighbors by angle for each patch
        sort_idx = np.argsort(angles, axis=1)

        # Use advanced indexing to sort all patches at once
        batch_idx = np.arange(B)[:, np.newaxis]
        sorted_neighbors = points[batch_idx, sort_idx]  # Shape: (B, N, 3)
        sorted_neighbor_normals = neighbor_normals[batch_idx, sort_idx]  # Shape: (B, N, 3)
        sorted_rel_pos = rel_pos[batch_idx, sort_idx]  # Shape: (B, N, 3)

        # Compute edge vectors with circular shifts
        e_i = sorted_rel_pos
        e_i_minus_1 = np.roll(sorted_rel_pos, 1, axis=1)
        e_i_plus_1 = np.roll(sorted_rel_pos, -1, axis=1)
        n_i = sorted_neighbors
        n_i_minus_1 = np.roll(sorted_neighbors, 1, axis=1)
        n_i_plus_1 = np.roll(sorted_neighbors, -1, axis=1)

        # Compute RISP features (all vectorized)
        L_0 = np.linalg.norm(e_i, axis=2)  # Shape: (B, N)
        phi_1 = self._compute_angle_between_vectors(e_i_minus_1, e_i)
        phi_2 = self._compute_angle_between_vectors(e_i_plus_1, e_i)
        phi_3 = self._compute_angle_between_vectors(e_i_minus_1, n_i - n_i_minus_1)
        phi_4 = self._compute_angle_between_vectors(e_i_plus_1, n_i_plus_1 - n_i)
        phi_5 = self._compute_angle_between_vectors(np.cross(e_i_plus_1, e_i), np.cross(e_i_minus_1, e_i))

        alpha_1 = self._compute_angle_between_vectors(
            np.broadcast_to(center_normals[:, np.newaxis, :], e_i.shape), e_i
        )
        alpha_2 = self._compute_angle_between_vectors(
            np.broadcast_to(center_normals[:, np.newaxis, :], e_i.shape), e_i_minus_1
        )

        nn_i = sorted_neighbor_normals
        nn_i_minus_1 = np.roll(sorted_neighbor_normals, 1, axis=1)
        nn_i_plus_1 = np.roll(sorted_neighbor_normals, -1, axis=1)

        beta_1 = self._compute_angle_between_vectors(nn_i, e_i)
        beta_2 = self._compute_angle_between_vectors(nn_i, n_i - n_i_minus_1)

        theta_1 = self._compute_angle_between_vectors(nn_i_minus_1, e_i_minus_1)
        theta_2 = self._compute_angle_between_vectors(nn_i_minus_1, n_i - n_i_minus_1)

        gamma_1 = self._compute_angle_between_vectors(nn_i_plus_1, n_i_plus_1 - n_i)
        gamma_2 = self._compute_angle_between_vectors(nn_i_plus_1, e_i_plus_1)

        # Stack features for all patches and neighbors
        risp_features = np.stack([
            L_0, phi_1, phi_2, phi_3, phi_4, phi_5,
            alpha_1, alpha_2, beta_1, beta_2,
            theta_1, theta_2, gamma_1, gamma_2], axis=-1)  # Shape: (B, N, 14)

        return risp_features

    def _extract_features_batch(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Extract RISP features for batch of patches using vectorized computation."""
        # Use the new vectorized RISP implementation
        return self._compute_risp_features_vectorized(points=points, normals=normals)


class XYZFeatureExtractor(FeatureExtractor):
    """Simple XYZ coordinate feature extractor with optional Fourier feature transformation."""

    def _extract_features_batch(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Extract XYZ features (simply return the points) - fully vectorized."""
        # XYZ features are just the points themselves, so this is trivially vectorized
        return points  # Shape: (B, N, 3)