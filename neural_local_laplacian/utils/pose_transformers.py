# Standard library
from abc import ABC, abstractmethod
from typing import Tuple

# Third-party libraries
import torch
import torch.nn.functional as F
import kornia

# Local imports
from neural_local_laplacian.utils import utils


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
        device = points.device
        dtype = points.dtype

        # Determine if input is batched and add batch dimension if needed
        is_single_patch = points.dim() == 2  # Shape: (N, 3)

        if is_single_patch:
            # Add batch dimension
            points_batch = points.unsqueeze(0)  # Shape: (1, N, 3)

            # Handle different normal shapes for single patch
            if normal.dim() == 1:  # Shape: (3,)
                normal_batch = normal.unsqueeze(0)  # Shape: (1, 3)
            else:  # Shape: (1, 3)
                normal_batch = normal  # Already correct shape
        else:
            # Already batched
            points_batch = points  # Shape: (B, N, 3)

            # Handle different normal shapes for batch
            if normal.dim() == 1:  # Single normal for all patches
                batch_size = points.shape[0]
                normal_batch = normal.unsqueeze(0).expand(batch_size, -1)  # Shape: (B, 3)
            elif normal.dim() == 2 and normal.shape[0] == 1:  # Single normal in shape (1, 3)
                batch_size = points.shape[0]
                normal_batch = normal.expand(batch_size, -1)  # Shape: (B, 3)
            else:  # Already shape (B, 3)
                normal_batch = normal

        # Call the implementation-specific batch transform
        translation_batch, rotation_batch = self._transform_batch(points_batch, normal_batch)

        # Remove batch dimension if input was single patch
        if is_single_patch:
            translation = translation_batch[0]  # Shape: (3,)
            rotation = rotation_batch[0]  # Shape: (3, 3)
            return translation, rotation
        else:
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


class IdentityTransformer(PoseTransformer):
    """No-op transformer that applies no transformation."""

    def _transform_batch(self, points: torch.Tensor, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply identity transformation (no change)."""
        batch_size = points.shape[0]
        device = points.device
        dtype = points.dtype

        translation = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        rotation_matrix = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

        return translation, rotation_matrix


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
        """Apply random rotation to batch of patches (fully vectorized)."""
        batch_size = points.shape[0]
        device = points.device
        dtype = points.dtype

        # Set seed if specified
        if self._seed is not None:
            torch_state = torch.get_rng_state()
            torch.manual_seed(self._seed)

        # Generate batch of random rotation matrices (vectorized)
        # Generate random 3x3 matrices
        random_matrices = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)

        # Perform QR decomposition for each matrix in the batch
        q_matrices, r_matrices = torch.linalg.qr(random_matrices)  # Shape: (B, 3, 3), (B, 3, 3)

        # Ensure proper rotation matrices (determinant = 1)
        # Extract diagonal signs from R matrices
        r_diag = torch.diagonal(r_matrices, dim1=-2, dim2=-1)  # Shape: (B, 3)
        r_diag_signs = torch.sign(r_diag)  # Shape: (B, 3)

        # Create diagonal sign matrices for batch multiplication
        sign_matrices = torch.diag_embed(r_diag_signs)  # Shape: (B, 3, 3)

        # Apply sign correction: Q @ sign_matrix
        rotation_matrices = torch.bmm(q_matrices, sign_matrices)  # Shape: (B, 3, 3)

        # Ensure right-handed coordinate system (determinant = 1)
        determinants = torch.det(rotation_matrices)  # Shape: (B,)

        # Flip first column for matrices with negative determinant
        negative_det_mask = determinants < 0  # Shape: (B,)
        rotation_matrices[negative_det_mask, :, 0] *= -1

        if self._seed is not None:
            torch.set_rng_state(torch_state)

        translation_batch = torch.zeros(batch_size, 3, device=device, dtype=dtype)  # Shape: (B, 3)

        return translation_batch, rotation_matrices


class PCATransformer(PoseTransformer):
    """Aligns points using PCA to canonical pose."""

    def _transform_batch(self, points: torch.Tensor, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply PCA-based canonical pose transformation to batch (fully vectorized)."""
        batch_size = points.shape[0]
        device = points.device
        dtype = points.dtype

        # Vectorized PCA computation
        # Center the points for each patch
        centers = torch.mean(points, dim=1, keepdim=True)  # Shape: (B, 1, 3)
        centered_points = points - centers  # Shape: (B, N, 3)

        # Compute covariance matrices for all patches
        # centered_points: (B, N, 3) -> need (B, 3, N) for matrix multiplication
        centered_points_transposed = centered_points.transpose(-1, -2)  # Shape: (B, 3, N)

        # Covariance matrices: (1/(N-1)) * X^T @ X where X is centered data
        # For batch: (B, 3, N) @ (B, N, 3) = (B, 3, 3)
        n_points = points.shape[1]
        covariance_matrices = torch.bmm(
            centered_points_transposed,
            centered_points
        ) / (n_points - 1)  # Shape: (B, 3, 3)

        # Compute eigendecomposition for all covariance matrices
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrices)  # Shapes: (B, 3), (B, 3, 3)

        # Sort eigenvectors by eigenvalues in descending order
        # Get sorted indices for each batch
        sorted_indices = torch.argsort(eigenvalues, dim=1, descending=True)  # Shape: (B, 3)

        # Use advanced indexing to reorder eigenvectors
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # Shape: (B, 1)
        rotation_matrices = eigenvectors[batch_indices, :, sorted_indices]  # Shape: (B, 3, 3)

        # Ensure right-handed coordinate system for all matrices
        determinants = torch.det(rotation_matrices)  # Shape: (B,)

        # Flip the last column for matrices with negative determinant
        negative_det_mask = determinants < 0  # Shape: (B,)
        rotation_matrices[negative_det_mask, :, 2] *= -1

        # Extract translations (centers flattened to remove keepdim)
        translations = centers.squeeze(1)  # Shape: (B, 3)

        return translations, rotation_matrices


class AlignNormalZTransformer(PoseTransformer):
    """Aligns the surface normal with the positive Z-axis."""

    def _transform_batch(self, points: torch.Tensor, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply normal alignment transformation to batch (fully vectorized)."""
        batch_size = points.shape[0]
        device = points.device
        dtype = points.dtype

        # Target normal (positive Z-axis) for all patches
        target_normal = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        target_normals = target_normal.unsqueeze(0).expand(batch_size, -1)  # Shape: (B, 3)

        # Normalize all normals at once
        origin_normals = F.normalize(normals, p=2, dim=1)  # Shape: (B, 3)
        target_normals = F.normalize(target_normals, p=2, dim=1)  # Shape: (B, 3)

        # Compute dot products for all patches
        dot_products = torch.sum(origin_normals * target_normals, dim=1)  # Shape: (B,)
        dot_products = torch.clamp(dot_products, -1.0, 1.0)

        # Vectorized quaternion computation using masks
        # Case 1: Already aligned (dot_product ≈ 1)
        aligned_mask = torch.abs(dot_products - 1.0) < 1e-6  # Shape: (B,)

        # Case 2: Opposite direction (dot_product ≈ -1)
        opposite_mask = torch.abs(dot_products + 1.0) < 1e-6  # Shape: (B,)

        # Case 3: General case
        general_mask = ~(aligned_mask | opposite_mask)  # Shape: (B,)

        # Initialize quaternions tensor
        quaternions = torch.zeros(batch_size, 4, dtype=dtype, device=device)  # Shape: (B, 4)

        # Handle aligned case: identity quaternion [1, 0, 0, 0]
        quaternions[aligned_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)

        # Handle opposite case: 180° rotation (vectorized)
        if opposite_mask.any():
            opposite_normals = origin_normals[opposite_mask]  # Shape: (N_opp, 3)

            # Find perpendicular axes for 180° rotation
            ref_x = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
            ref_y = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

            # Choose reference vector based on normal's x component
            use_x_ref = torch.abs(opposite_normals[:, 0]) < 0.9  # Shape: (N_opp,)

            # Compute rotation axes (vectorized)
            rotation_axes = torch.zeros_like(opposite_normals)  # Shape: (N_opp, 3)

            # For normals where |x| < 0.9, use cross with [1,0,0]
            if use_x_ref.any():
                ref_x_expanded = ref_x.unsqueeze(0).expand(use_x_ref.sum(), -1)
                rotation_axes[use_x_ref] = torch.linalg.cross(
                    opposite_normals[use_x_ref],
                    ref_x_expanded,
                    dim=1
                )

            # For normals where |x| >= 0.9, use cross with [0,1,0]
            if (~use_x_ref).any():
                ref_y_expanded = ref_y.unsqueeze(0).expand((~use_x_ref).sum(), -1)
                rotation_axes[~use_x_ref] = torch.linalg.cross(
                    opposite_normals[~use_x_ref],
                    ref_y_expanded,
                    dim=1
                )

            # Normalize rotation axes
            rotation_axes = F.normalize(rotation_axes, p=2, dim=1)

            # 180° quaternions: [0, axis_x, axis_y, axis_z]
            opposite_quaternions = torch.cat([
                torch.zeros(rotation_axes.shape[0], 1, dtype=dtype, device=device),  # w = 0
                rotation_axes  # [x, y, z]
            ], dim=1)  # Shape: (N_opp, 4)

            quaternions[opposite_mask] = opposite_quaternions

        # Handle general case: compute rotation quaternions (vectorized)
        if general_mask.any():
            general_normals = origin_normals[general_mask]  # Shape: (N_gen, 3)
            general_targets = target_normals[general_mask]  # Shape: (N_gen, 3)
            general_dots = dot_products[general_mask]  # Shape: (N_gen,)

            # Compute rotation axes (cross product) - vectorized
            rotation_axes = torch.linalg.cross(general_normals, general_targets, dim=1)  # Shape: (N_gen, 3)
            axis_lengths = torch.norm(rotation_axes, dim=1, keepdim=True)  # Shape: (N_gen, 1)

            # Check for zero-length axes (shouldn't happen in general case, but be safe)
            valid_axes_mask = axis_lengths.squeeze(1) > 1e-6  # Shape: (N_gen,)

            if valid_axes_mask.any():
                # Normalize rotation axes
                valid_rotation_axes = rotation_axes[valid_axes_mask]
                valid_axis_lengths = axis_lengths[valid_axes_mask]
                valid_rotation_axes = valid_rotation_axes / valid_axis_lengths

                # Compute angles and quaternions - vectorized
                valid_dots = general_dots[valid_axes_mask]
                angles = torch.acos(valid_dots)  # Shape: (N_valid,)
                half_angles = angles / 2.0
                sin_half_angles = torch.sin(half_angles)  # Shape: (N_valid,)
                cos_half_angles = torch.cos(half_angles)  # Shape: (N_valid,)

                # Build quaternions: [cos(θ/2), sin(θ/2)*axis] - vectorized
                general_quaternions = torch.cat([
                    cos_half_angles.unsqueeze(1),  # w
                    sin_half_angles.unsqueeze(1) * valid_rotation_axes  # [x, y, z]
                ], dim=1)  # Shape: (N_valid, 4)

                # Map back to general indices
                general_indices = torch.where(general_mask)[0]
                valid_general_indices = general_indices[valid_axes_mask]
                quaternions[valid_general_indices] = general_quaternions

            # Handle invalid axes (fallback to identity)
            if (~valid_axes_mask).any():
                invalid_general_indices = torch.where(general_mask)[0][~valid_axes_mask]
                quaternions[invalid_general_indices] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)

        # Convert quaternions to rotation matrices (vectorized)
        rotation_matrices = kornia.geometry.conversions.quaternion_to_rotation_matrix(quaternions)  # Shape: (B, 3, 3)
        translation_batch = torch.zeros(batch_size, 3, device=device, dtype=dtype)  # Shape: (B, 3)

        return translation_batch, rotation_matrices