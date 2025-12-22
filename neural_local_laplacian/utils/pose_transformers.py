# Standard library
from abc import ABC, abstractmethod
from typing import Tuple

# Third-party libraries
import torch
import torch.nn.functional as F


class PoseTransformer(ABC):
    """Abstract base class for pose transformation strategies."""

    def transform(self, points: torch.Tensor, normal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform points and normal to canonical pose.
        Handles both single patches and batches automatically.

        Args:
            points: Point coordinates of shape (N, 3) or (B, N, 3)
            normal: Normal vector of shape (3,), (1, 3), or (B, 3)

        Returns:
            Tuple of (translation_vector, rotation_matrix)
            - translation_vector: shape (3,) or (B, 3)
            - rotation_matrix: shape (3, 3) or (B, 3, 3)
        """
        # Determine if input is batched and add batch dimension if needed
        is_single_patch = points.dim() == 2  # Shape: (N, 3)

        if is_single_patch:
            points_batch = points.unsqueeze(0)  # Shape: (1, N, 3)
            normal_batch = normal.view(1, 3)  # Shape: (1, 3)
        else:
            points_batch = points  # Shape: (B, N, 3)
            # Handle different normal shapes for batch
            if normal.dim() == 1:  # Single normal for all patches
                normal_batch = normal.unsqueeze(0).expand(points.shape[0], -1)
            elif normal.shape[0] == 1:  # Single normal in shape (1, 3)
                normal_batch = normal.expand(points.shape[0], -1)
            else:  # Already shape (B, 3)
                normal_batch = normal

        # Call the implementation-specific batch transform
        translation_batch, rotation_batch = self._transform_batch(points_batch, normal_batch)

        # Remove batch dimension if input was single patch
        if is_single_patch:
            return translation_batch[0], rotation_batch[0]
        return translation_batch, rotation_batch

    @abstractmethod
    def _transform_batch(self, points: torch.Tensor, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implementation-specific batch transformation (always receives batched inputs).

        Args:
            points: Point coordinates of shape (B, N, 3)
            normals: Normal vectors of shape (B, 3)

        Returns:
            Tuple of (translation_vectors, rotation_matrices)
            - translation_vectors: shape (B, 3)
            - rotation_matrices: shape (B, 3, 3)
        """
        pass


class RandomRotationTransformer(PoseTransformer):
    """Applies random 3D rotation around the origin."""

    def __init__(self, seed: int = None):
        """
        Initialize random rotation transformer.

        Args:
            seed: Optional random seed for reproducible rotations
        """
        self._seed = seed

    def _transform_batch(self, points: torch.Tensor, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random SO(3) rotation to batch of patches."""
        batch_size = points.shape[0]
        device = points.device
        dtype = points.dtype

        # Save and set seed if specified
        if self._seed is not None:
            torch_state = torch.get_rng_state()
            torch.manual_seed(self._seed)

        # Generate random rotation matrices via QR decomposition
        random_matrices = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
        q_matrices, r_matrices = torch.linalg.qr(random_matrices)

        # Ensure proper rotation matrices (det = +1)
        # Multiply Q by sign of R's diagonal to get uniform distribution over SO(3)
        r_diag_signs = torch.sign(torch.diagonal(r_matrices, dim1=-2, dim2=-1))
        rotation_matrices = q_matrices * r_diag_signs.unsqueeze(1)

        # Fix any remaining negative determinants
        neg_det_mask = torch.det(rotation_matrices) < 0
        rotation_matrices[neg_det_mask, :, 0] *= -1

        if self._seed is not None:
            torch.set_rng_state(torch_state)

        translation = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        return translation, rotation_matrices


class AlignNormalZTransformer(PoseTransformer):
    """Aligns the surface normal with the positive Z-axis."""

    def _transform_batch(self, points: torch.Tensor, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align normals with +Z axis by constructing orthonormal basis.

        Builds rotation matrix R where R @ n = [0, 0, 1].
        """
        batch_size = points.shape[0]
        device = points.device
        dtype = points.dtype

        # Normalize input normals (these become the new Z-axis)
        z_axis = F.normalize(normals, p=2, dim=1)  # Shape: (B, 3)

        # Choose reference vector for cross product
        # Use [1,0,0] unless normal is nearly parallel to it, then use [0,1,0]
        ref = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        parallel_mask = torch.abs(z_axis[:, 0]) > 0.9
        ref[~parallel_mask] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        ref[parallel_mask] = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

        # Build orthonormal basis: x = ref × z, y = z × x
        x_axis = F.normalize(torch.linalg.cross(ref, z_axis, dim=1), p=2, dim=1)
        y_axis = torch.linalg.cross(z_axis, x_axis, dim=1)  # Already unit length

        # Rotation matrix with rows [x, y, z] satisfies R @ n = [0, 0, 1]
        rotation_matrices = torch.stack([x_axis, y_axis, z_axis], dim=1)

        translation = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        return translation, rotation_matrices