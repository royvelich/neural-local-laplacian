# Standard library
import math
from abc import ABC, abstractmethod
from typing import Optional

# PyTorch
import torch
import torch.nn as nn


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    All operations are in PyTorch — no numpy. Subclasses implement
    _extract_features_batch() which always receives a batched (B, N, 3) tensor
    and returns (B, N, feature_dim). The public extract_features() handles the
    single-patch / batched dispatch automatically.
    """

    def extract_features(self, points: torch.Tensor, normals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract features from point coordinates.

        Args:
            points:  (N, 3) or (B, N, 3) — neighbor positions relative to patch center
            normals: (N, 3) or (B, N, 3) — optional surface normals (ignored by some extractors)

        Returns:
            (N, feature_dim) or (B, N, feature_dim) matching the input rank
        """
        is_single = points.dim() == 2
        if is_single:
            points = points.unsqueeze(0)                          # (1, N, 3)
            if normals is not None:
                normals = normals.unsqueeze(0)                    # (1, N, 3)

        features = self._extract_features_batch(points, normals)  # (B, N, feature_dim)

        if is_single:
            features = features.squeeze(0)                        # (N, feature_dim)

        return features

    @abstractmethod
    def _extract_features_batch(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch implementation — always receives (B, N, 3), returns (B, N, feature_dim).
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the output features."""
        pass


class XYZFeatureExtractor(FeatureExtractor):
    """Pass-through extractor — returns raw XYZ coordinates unchanged."""

    def _extract_features_batch(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return points  # (B, N, 3)

    @property
    def output_dim(self) -> int:
        return 3


class FourierFeatureExtractor(FeatureExtractor, nn.Module):
    """
    Random Fourier Feature (RFF) mapping from Tancik et al. 2020
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains".

    Maps 3D neighbor positions p to:
        γ(p) = [cos(2π B p), sin(2π B p)]^T

    where B ∈ R^{m×3} is sampled once from N(0, σ²) at construction time and
    kept fixed (registered as a buffer — not a trainable parameter).

    The mapping transforms the NTK from a dot-product kernel (not shift-invariant)
    into a stationary one, enabling the downstream transformer to learn high-frequency
    geometric features (sharp edges, high-curvature regions) that would otherwise be
    suppressed by spectral bias.

    Output dimension: 2 * num_frequencies.

    Args:
        num_frequencies: Number of random frequency vectors m. Output dim = 2m.
        sigma:           Standard deviation of the Gaussian from which B is sampled.
                         Controls the bandwidth of the kernel:
                           - too small  → underfitting (only low frequencies learned)
                           - too large  → overfitting / aliasing
                         Treat as a hyperparameter; sweep on a validation set.
        input_dim:       Dimensionality of the input coordinates (default 3 for 3D).
    """

    def __init__(
        self,
        num_frequencies: int = 64,
        sigma: float = 1.0,
        input_dim: int = 3,
    ):
        nn.Module.__init__(self)
        self.num_frequencies = num_frequencies
        self.sigma = sigma
        self.input_dim = input_dim

        # Sample B once and register as a non-trainable buffer
        B = torch.randn(num_frequencies, input_dim) * sigma
        self.register_buffer('B', B)  # shape: (m, input_dim)

    def _extract_features_batch(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply Fourier feature mapping to a batch of patches.

        Args:
            points:  (B, N, 3) — neighbor positions (relative to patch center)
            normals: ignored

        Returns:
            (B, N, 2m) — concatenated cosine and sine features
        """
        # points: (B, N, 3), B_mat: (m, 3)  →  projection: (B, N, m)
        projection = (2.0 * math.pi * points) @ self.B.T
        return torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)  # (B, N, 2m)

    @property
    def output_dim(self) -> int:
        return 2 * self.num_frequencies