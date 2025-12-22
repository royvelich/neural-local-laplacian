# standard library
from abc import ABC, abstractmethod

# numpy
import numpy as np


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    def extract_features(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        Extract features from point coordinates and normals.
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


class XYZFeatureExtractor(FeatureExtractor):
    """Simple XYZ coordinate feature extractor."""

    def _extract_features_batch(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Extract XYZ features (simply return the points)."""
        return points  # Shape: (B, N, 3)