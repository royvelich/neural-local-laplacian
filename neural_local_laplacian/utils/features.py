# standard library
from abc import ABC, abstractmethod

# neural local laplacian
from neural_local_laplacian.utils.utils import compute_risp_features
from typing import Optional

# numpy
import numpy as np

# torch
import torch


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

    @abstractmethod
    def _extract_raw_features(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Extract raw features without Fourier transformation."""
        pass

    def extract_features(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        Extract features and apply Fourier transformation if enabled.

        Args:
            points: Point coordinates
            normals: Normal vectors

        Returns:
            Extracted features, possibly with Fourier transformation applied
        """
        # Extract raw features using the derived class implementation
        features = self._extract_raw_features(points, normals)

        # Apply Fourier transformation if enabled
        if self._use_fourier:
            features = self._apply_fourier_transform(features)

        return features

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

    def _apply_fourier_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Apply Fourier feature transform to features.

        Args:
            features: Input features array of shape [N, D]

        Returns:
            Transformed features with Fourier features
        """
        # Convert numpy array to torch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # If this is our first call, initialize B
        if self._B is None:
            self._initialize_fourier_matrix(features.shape[1])

        # Apply Fourier feature mapping: [cos(2πBx), sin(2πBx)]
        x_proj = 2 * np.pi * torch.matmul(features_tensor, self._B)
        fourier_features = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

        # Convert back to numpy
        return fourier_features.cpu().numpy()


class RISPFeatureExtractor(FeatureExtractor):
    """RISP feature extractor with optional Fourier feature transformation."""

    def __init__(self, k: int, **kwargs):
        """
        Initialize RISP feature extractor.

        Args:
            k: Number of neighbors for RISP feature computation
            **kwargs: Fourier feature parameters passed to the base class
        """
        super().__init__(**kwargs)
        self._k = k

    def _extract_raw_features(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Extract RISP features."""
        return compute_risp_features(points=points, normals=normals)


class XYZFeatureExtractor(FeatureExtractor):
    """Simple XYZ coordinate feature extractor with optional Fourier feature transformation."""

    def _extract_raw_features(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Extract XYZ features (simply return the points)."""
        return points