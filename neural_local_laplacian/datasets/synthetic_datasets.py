# Standard library
from typing import Tuple, List, Optional, Union, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod

# torch
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data

# numpy
import numpy as np

# scipy
from scipy.spatial import Delaunay

# neural signatures
from neural_local_laplacian.datasets.base_datasets import CoeffGenerationMethod
from neural_local_laplacian.utils.features import FeatureExtractor
from neural_local_laplacian.utils.pose_transformers import PoseTransformer


# =============================================
# Grid Sampler Classes
# =============================================

class GridSampler(ABC):
    """Abstract base class for grid sampling strategies."""

    @abstractmethod
    def sample(self, grid_range: Tuple[float, float], rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x, y coordinates within the given grid range.

        Args:
            grid_range: Tuple of (min_val, max_val) defining the sampling range
            rng: Random number generator for reproducible sampling

        Returns:
            Tuple of (x, y) tensors containing sampled coordinates
        """
        pass


class RegularGridSampler(GridSampler):
    """Samples points on a regular rectangular grid."""

    def __init__(self, num_points: int):
        """
        Initialize the regular grid sampler.

        Args:
            num_points: Total number of points to sample (will be adjusted to nearest perfect square)
        """
        self._num_points = num_points
        # Adjust to nearest perfect square for regular grid
        self._grid_size = int(np.sqrt(num_points))
        self._actual_points = self._grid_size ** 2

        if self._actual_points != num_points:
            print(f"Warning: Adjusted grid points from {num_points} to {self._actual_points} (nearest perfect square)")

    def sample(self, grid_range: Tuple[float, float], rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points on a regular grid.

        Args:
            grid_range: Tuple of (min_val, max_val) defining the sampling range
            rng: Random number generator (not used for regular grid, but kept for interface consistency)

        Returns:
            Tuple of (x, y) tensors with regularly spaced coordinates
        """
        x_linspace = torch.linspace(start=grid_range[0], end=grid_range[1], steps=self._grid_size)
        y_linspace = torch.linspace(start=grid_range[0], end=grid_range[1], steps=self._grid_size)
        x, y = torch.meshgrid(x_linspace, y_linspace, indexing='ij')
        return x.flatten(), y.flatten()

    @property
    def num_points(self) -> int:
        """Get the actual number of points that will be sampled."""
        return self._actual_points


class RandomGridSampler(GridSampler):
    """Samples points uniformly at random within the grid range."""

    def __init__(self, num_points_range: Union[int, Tuple[int, int]]):
        """
        Initialize the random grid sampler.

        Args:
            num_points_range: Either a single integer for fixed number of points,
                            or a tuple (min_points, max_points) for variable sampling
        """
        # Handle OmegaConf objects (common with Hydra)
        try:
            from omegaconf import ListConfig, DictConfig
            if isinstance(num_points_range, (ListConfig, DictConfig)):
                # Convert OmegaConf to regular Python types
                num_points_range = list(num_points_range) if isinstance(num_points_range, ListConfig) else num_points_range
        except ImportError:
            # OmegaConf not available, continue with regular handling
            pass

        # Convert different input types and provide better error messages
        if isinstance(num_points_range, int):
            self._num_points_range = (num_points_range, num_points_range)
        elif isinstance(num_points_range, (tuple, list)):
            if len(num_points_range) == 2:
                # Convert to integers if they're not already
                try:
                    self._num_points_range = (int(num_points_range[0]), int(num_points_range[1]))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"num_points_range elements must be convertible to integers, got {num_points_range}: {e}")

                if self._num_points_range[0] > self._num_points_range[1]:
                    raise ValueError(f"Min points {self._num_points_range[0]} > max points {self._num_points_range[1]}")
            else:
                raise ValueError(f"num_points_range tuple/list must have exactly 2 elements, got {len(num_points_range)} elements: {num_points_range}")
        else:
            raise ValueError(f"num_points_range must be an integer or a tuple/list of two integers, got {type(num_points_range)}: {num_points_range}")

        # Validate minimum points
        if self._num_points_range[0] < 3:
            raise ValueError(f"Minimum number of points must be >= 3, got {self._num_points_range[0]}")

        # Validate that points are positive integers
        if not all(isinstance(x, int) and x > 0 for x in self._num_points_range):
            raise ValueError(f"All point counts must be positive integers, got {self._num_points_range}")

    def sample(self, grid_range: Tuple[float, float], rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points uniformly at random within the grid range.

        Args:
            grid_range: Tuple of (min_val, max_val) defining the sampling range
            rng: Random number generator for reproducible sampling

        Returns:
            Tuple of (x, y) tensors with randomly sampled coordinates
        """
        # Sample number of points if range is provided
        if self._num_points_range[0] == self._num_points_range[1]:
            num_points = self._num_points_range[0]
        else:
            num_points = int(rng.integers(
                low=self._num_points_range[0],
                high=self._num_points_range[1] + 1
            ))

        # Generate random points
        points = rng.uniform(
            low=grid_range[0],
            high=grid_range[1],
            size=(num_points, 2)
        )

        return torch.tensor(points[:, 0]), torch.tensor(points[:, 1])

    @property
    def num_points_range(self) -> Tuple[int, int]:
        """Get the range of number of points that can be sampled."""
        return self._num_points_range


# =============================================
# Differential Geometry and Dataset Classes
# =============================================

class DifferentialGeometryComponent(Enum):
    MEAN_CURVATURE = 'mean_curvature'
    GAUSSIAN_CURVATURE = 'gaussian_curvature'
    PRINCIPAL_CURVATURES = 'principal_curvatures'
    PRINCIPAL_DIRECTIONS_2D = 'principal_directions_2d'
    PRINCIPAL_DIRECTIONS_3D = 'principal_directions_3d'
    PRINCIPAL_DIRECTIONS = 'principal_directions'  # Legacy alias for both 2D and 3D
    CURVATURE_GRADIENTS_2D = 'curvature_gradients_2d'
    CURVATURE_GRADIENTS_3D = 'curvature_gradients_3d'
    SIGNATURE = 'signature'


# Mapping from enum components to output dictionary keys
DIFF_GEOM_COMPONENT_KEYS = {
    DifferentialGeometryComponent.MEAN_CURVATURE: ['H'],
    DifferentialGeometryComponent.GAUSSIAN_CURVATURE: ['K'],
    DifferentialGeometryComponent.PRINCIPAL_CURVATURES: ['k1', 'k2'],
    DifferentialGeometryComponent.PRINCIPAL_DIRECTIONS_2D: ['v1_2d', 'v2_2d'],
    DifferentialGeometryComponent.PRINCIPAL_DIRECTIONS_3D: ['v1_3d', 'v2_3d'],
    DifferentialGeometryComponent.PRINCIPAL_DIRECTIONS: ['v1_2d', 'v2_2d', 'v1_3d', 'v2_3d'],  # Both
    DifferentialGeometryComponent.CURVATURE_GRADIENTS_2D: ['grad_H_2d', 'grad_K_2d'],
    DifferentialGeometryComponent.CURVATURE_GRADIENTS_3D: ['grad_H_3d', 'grad_K_3d'],
    DifferentialGeometryComponent.SIGNATURE: ['signature'],
}


class SyntheticSurfaceDataset(ABC, Dataset):
    """Base class for synthetic surface datasets."""

    def __init__(
            self,
            epoch_size: int,
            pose_transformers: Optional[List[PoseTransformer]] = None,
            seed: int = 0,
            feature_extractor: Optional[FeatureExtractor] = None,
            conv_k_nearest: Optional[int] = None,
    ):
        super().__init__()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._epoch_size = epoch_size
        self._pose_transformers = pose_transformers if pose_transformers is not None else []
        self._feature_extractor = feature_extractor
        self._conv_k_nearest = conv_k_nearest

    def reset_rng(self) -> None:
        """Reset random number generator to initial seed."""
        self._rng = np.random.default_rng(seed=self._seed)

    def len(self) -> int:
        return self._epoch_size

    def _generate_surfaces(self) -> List[Data]:
        """
        Generate multiple surfaces and add features to them.

        This is the main orchestration method that:
        1. Calls the derived class implementation to generate raw surface data
        2. Adds features to each surface using the feature extractor

        Returns:
            List of Data objects with positions, normals, differential geometry, and features
        """
        # Generate raw surfaces using derived class implementation
        surfaces = self._generate_raw_surfaces()

        # Add features to each surface
        for surface in surfaces:
            self._add_surface_features(surface)

        return surfaces

    @abstractmethod
    def _generate_raw_surfaces(self) -> List[Data]:
        """
        Generate multiple raw surfaces without features.

        This method should be implemented by derived classes to generate
        surface data with positions, normals, and differential geometry,
        but WITHOUT features (the 'x' attribute).

        Returns:
            List of Data objects with surface geometry but no features
        """
        pass

    def _repose_surface_and_quantities(self, data: Data, normals: Optional[torch.Tensor] = None) -> Data:
        """Apply pose transformations sequentially to surface and transform differential quantities accordingly."""
        if not self._pose_transformers:
            return data

        # Use passed normals if provided, otherwise get from data object
        if normals is not None:
            # Use the explicitly passed normal (e.g., origin normal)
            if normals.dim() == 2 and normals.shape[0] == 1:
                # Shape: (1, 3) -> extract to (3,)
                normal = normals[0]
            elif normals.dim() == 1:
                # Shape: (3,) -> use as is
                normal = normals
            else:
                # Handle unexpected shapes - take first normal
                normal = normals.flatten()[:3]
        else:
            # Fallback: get normal from data object (original behavior)
            normal = data['normal'][0] if 'normal' in data else torch.tensor([0., 0., 1.])

        # Apply each transformer sequentially
        for pose_transformer in self._pose_transformers:
            # Get translation and rotation from this transformer
            translation, rotation_matrix = pose_transformer.transform(data.pos, normal)

            # Apply translation and rotation to positions
            data.pos = data.pos + translation
            data.pos = torch.matmul(data.pos, rotation_matrix.T)

            # Transform origin position if present
            if 'origin_pos' in data:
                data['origin_pos'] = data['origin_pos'] + translation
                data['origin_pos'] = torch.matmul(data['origin_pos'], rotation_matrix.T)

            # Transform normals using rotation matrix
            if 'normal' in data:
                data['normal'] = torch.matmul(data['normal'], rotation_matrix.T)

            # Transform differential geometry quantities
            vector_3d_keys = ['v1_3d', 'v2_3d', 'grad_H_3d', 'grad_K_3d']
            for key in vector_3d_keys:
                if key in data:
                    data[key] = torch.matmul(data[key], rotation_matrix.T)

            # Transform the origin normal for the next transformer
            normal = torch.matmul(normal.unsqueeze(0), rotation_matrix.T).squeeze(0)

        return data

    def _add_surface_features(self, data: Data) -> None:
        """
        Add appropriate features to the surface data object using the feature extractor.

        Args:
            data: Surface data object to add features to (modified in place)
        """
        if self._feature_extractor is not None:
            # Extract points and normals as numpy arrays
            points = data.pos.detach().cpu().numpy()
            normals = data.normal.detach().cpu().numpy()

            # Use the feature extractor to compute features
            features = self._feature_extractor.extract_features(points=points, normals=normals)

            # Convert back to tensor and store
            data['x'] = torch.from_numpy(features).float()
        else:
            # Fallback to using positions as features if no extractor provided
            data['x'] = data.pos

    def get(self, idx: int) -> List[Data]:
        """Generate multiple samplings of the same surface."""
        surfaces = self._generate_surfaces()
        return surfaces


class ParametricSurfaceDataset(SyntheticSurfaceDataset):
    """Dataset for parametric surfaces with differential geometry computation."""

    def __init__(
            self,
            grid_samplers: List[GridSampler],
            grid_radius_range: Tuple[float, float],
            points_scale_range: Tuple[float, float],
            diff_geom_components: Optional[List[DifferentialGeometryComponent]] = None,
            diff_geom_at_origin_only: bool = False,
            flip_normal_if_negative_curvature: bool = False,
            include_origin_in_grid: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._grid_samplers = grid_samplers
        self._grid_radius_range = self._validate_range(param_range=grid_radius_range, name="grid_radius_range")
        self._points_scale_range = self._validate_range(param_range=points_scale_range, name="points_scale_range")
        self._diff_geom_at_origin_only = diff_geom_at_origin_only
        self._flip_normal_if_negative_curvature = flip_normal_if_negative_curvature
        self._include_origin_in_grid = include_origin_in_grid

        # Available differential geometry components
        available_components = list(DifferentialGeometryComponent)
        if diff_geom_components is None:
            self._diff_geom_components = available_components
        else:
            invalid = set(diff_geom_components) - set(available_components)
            if invalid:
                raise ValueError(f"Invalid diff_geom_components: {invalid}. Available: {available_components}")
            self._diff_geom_components = diff_geom_components

    def _validate_range(self, param_range: Tuple[float, float], name: str) -> Tuple[float, float]:
        """Validate parameter range."""
        if len(param_range) not in [1, 2]:
            raise ValueError(f"{name} must have 1 or 2 elements, got {len(param_range)}")
        if len(param_range) == 2 and param_range[0] > param_range[1]:
            raise ValueError(f"{name} min value {param_range[0]} > max value {param_range[1]}")
        return param_range

    def _sample_parameter(self, param_range: Tuple[float, ...]) -> float:
        """Sample parameter from range."""
        if len(param_range) == 2:
            return float(self._rng.uniform(low=param_range[0], high=param_range[1]))
        return param_range[0]

    def _compute_origin_data(self, surface_params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute all data at origin (0,0) in a single pass.

        Returns dict with: x, y, z, dz_dx, dz_dy, normal, H (mean curvature)
        """
        x_origin = torch.tensor([0.0], requires_grad=True)
        y_origin = torch.tensor([0.0], requires_grad=True)
        z_origin = self._evaluate_surface_with_parameters(x=x_origin, y=y_origin, surface_params=surface_params)

        dz_dx = torch.autograd.grad(outputs=z_origin, inputs=x_origin, create_graph=True, retain_graph=True)[0]
        dz_dy = torch.autograd.grad(outputs=z_origin, inputs=y_origin, create_graph=True, retain_graph=True)[0]

        # Compute normal (without flipping - flipping uses H which we compute next)
        normal = self._compute_normal_from_derivatives(dz_dx, dz_dy)

        # Compute H for potential normal flipping
        H, _, _, _, _, _ = self._compute_curvature_quantities(dz_dx=dz_dx, dz_dy=dz_dy, x=x_origin, y=y_origin)

        return {
            'x': x_origin,
            'y': y_origin,
            'z': z_origin,
            'dz_dx': dz_dx,
            'dz_dy': dz_dy,
            'normal': normal,
            'H': H
        }

    def _compute_normal_from_derivatives(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> torch.Tensor:
        """Compute normalized surface normal from partial derivatives."""
        normal = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=1)
        return F.normalize(normal, p=2, dim=1)

    def _compute_surface_normals(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor,
                                 H_at_origin: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute surface normals from derivatives.

        Args:
            dz_dx, dz_dy: Precomputed derivatives
            H_at_origin: Mean curvature at origin (for flipping, if enabled)
        """
        normal = self._compute_normal_from_derivatives(dz_dx, dz_dy)

        # Apply normal flipping logic if flag is set
        if self._flip_normal_if_negative_curvature and H_at_origin is not None:
            normal = self._apply_normal_flipping(normal, H_at_origin)

        return normal

    def _apply_normal_flipping(self, normal: torch.Tensor, H_at_origin: torch.Tensor) -> torch.Tensor:
        """Flip normal if mean curvature at origin is negative."""
        if H_at_origin.item() < 0:
            return -normal
        return normal

    def _create_surface_mesh(self, pos: torch.Tensor) -> torch.Tensor:
        """Create triangular mesh from 2D positions."""
        try:
            pos_2d = pos[:, :2].detach().numpy()
            if len(pos_2d) < 3:
                raise ValueError("Need at least 3 points for triangulation")
            return torch.from_numpy(Delaunay(points=pos_2d).simplices).T
        except Exception as e:
            raise RuntimeError(f"Failed to create surface mesh: {e}")

    def _compute_first_derivatives(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute first partial derivatives Ã¢Ë†â€šz/Ã¢Ë†â€šx and Ã¢Ë†â€šz/Ã¢Ë†â€šy."""
        dz_dx = torch.autograd.grad(
            outputs=z.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        dz_dy = torch.autograd.grad(
            outputs=z.sum(),
            inputs=y,
            create_graph=True,
            retain_graph=True
        )[0]
        return dz_dx, dz_dy

    def _compute_second_derivatives(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute second partial derivatives from first derivatives."""
        d2z_dx2, d2z_dxdy = torch.autograd.grad(outputs=dz_dx.sum(), inputs=[x, y], create_graph=True)
        _, d2z_dy2 = torch.autograd.grad(outputs=dz_dy.sum(), inputs=[x, y], create_graph=True)
        return d2z_dx2, d2z_dxdy, d2z_dy2

    def _compute_shape_operator(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor, d2z_dx2: torch.Tensor, d2z_dxdy: torch.Tensor, d2z_dy2: torch.Tensor) -> torch.Tensor:
        """Compute shape operator matrix."""
        E = 1 + dz_dx ** 2
        F = dz_dx * dz_dy
        G = 1 + dz_dy ** 2
        L = d2z_dx2 / torch.sqrt(1 + dz_dx ** 2 + dz_dy ** 2)
        M = d2z_dxdy / torch.sqrt(1 + dz_dx ** 2 + dz_dy ** 2)
        N = d2z_dy2 / torch.sqrt(1 + dz_dx ** 2 + dz_dy ** 2)
        det = E * G - F ** 2
        shape_operator = torch.stack([
            torch.stack([G * L - F * M, G * M - F * N], dim=-1),
            torch.stack([E * M - F * L, E * N - F * M], dim=-1)
        ], dim=-2) / det.unsqueeze(dim=-1).unsqueeze(dim=-2)
        return shape_operator

    def _compute_principal_curvatures(self, shape_operator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract principal curvatures and directions from shape operator."""
        eigenvalues, eigenvectors = torch.linalg.eig(input=shape_operator)
        k1, k2 = eigenvalues.real[..., 0], eigenvalues.real[..., 1]
        v1, v2 = eigenvectors.real[..., 0], eigenvectors.real[..., 1]
        return k1, k2, v1, v2

    def _compute_curvatures(self, k1: torch.Tensor, k2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and Gaussian curvatures."""
        H = (k1 + k2) / 2
        K = k1 * k2
        return H, K

    def _compute_curvature_gradients(self, H: torch.Tensor, K: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Compute gradients of curvatures in parameter space."""
        grad_H = torch.autograd.grad(outputs=H.sum(), inputs=[x, y], create_graph=True)
        grad_K = torch.autograd.grad(outputs=K.sum(), inputs=[x, y], create_graph=True)
        return grad_H, grad_K

    def _compute_jacobian(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of surface parameterization."""
        return torch.stack([
            torch.stack([torch.ones_like(dz_dx), torch.zeros_like(dz_dx)], dim=-1),
            torch.stack([torch.zeros_like(dz_dy), torch.ones_like(dz_dy)], dim=-1),
            torch.stack([dz_dx, dz_dy], dim=-1)
        ], dim=-2)

    def _map_to_3d(self, jacobian: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, grad_H: torch.Tensor, grad_K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map 2D vector fields to 3D using Jacobian."""
        v1_3d = torch.einsum('ijk,ik->ij', jacobian, v1)
        v2_3d = torch.einsum('ijk,ik->ij', jacobian, v2)
        grad_H_3d = torch.einsum('ijk,ik->ij', jacobian, grad_H)
        grad_K_3d = torch.einsum('ijk,ik->ij', jacobian, grad_K)
        return v1_3d, v2_3d, grad_H_3d, grad_K_3d

    def _compute_3d_euclidean_signatures(self, H: torch.Tensor, K: torch.Tensor, grad_H_3d: torch.Tensor, grad_K_3d: torch.Tensor, v1_3d: torch.Tensor, v2_3d: torch.Tensor) -> torch.Tensor:
        """Compute invariant 3D Euclidean signatures."""
        # Normalize principal directions
        v1_3d_norm = v1_3d / torch.norm(v1_3d, dim=1, keepdim=True)
        v2_3d_norm = v2_3d / torch.norm(v2_3d, dim=1, keepdim=True)

        # Compute directional derivatives
        H_1 = torch.sum(grad_H_3d * v1_3d_norm, dim=1)
        H_2 = torch.sum(grad_H_3d * v2_3d_norm, dim=1)
        K_1 = torch.sum(grad_K_3d * v1_3d_norm, dim=1)
        K_2 = torch.sum(grad_K_3d * v2_3d_norm, dim=1)

        # Stack signature components
        signature = torch.stack([H, K, H_1, H_2, K_1, K_2], dim=1)
        return signature

    def _compute_curvature_quantities(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all curvature-related quantities."""
        # Second derivatives
        d2z_dx2, d2z_dxdy, d2z_dy2 = self._compute_second_derivatives(dz_dx=dz_dx, dz_dy=dz_dy, x=x, y=y)

        # Shape operator and principal curvatures
        shape_operator = self._compute_shape_operator(
            dz_dx=dz_dx, dz_dy=dz_dy,
            d2z_dx2=d2z_dx2, d2z_dxdy=d2z_dxdy, d2z_dy2=d2z_dy2
        )
        k1, k2, v1_2d, v2_2d = self._compute_principal_curvatures(shape_operator=shape_operator)
        H, K = self._compute_curvatures(k1=k1, k2=k2)

        return H, K, k1, k2, v1_2d, v2_2d

    def _compute_gradients_2d(self, H: torch.Tensor, K: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute curvature gradients in 2D parameter space."""
        grad_H_2d, grad_K_2d = self._compute_curvature_gradients(H=H, K=K, x=x, y=y)
        grad_H_2d = torch.stack(grad_H_2d, dim=-1)
        grad_K_2d = torch.stack(grad_K_2d, dim=-1)
        return grad_H_2d, grad_K_2d

    def _compute_3d_quantities(self, jacobian: torch.Tensor, v1_2d: torch.Tensor, v2_2d: torch.Tensor,
                               grad_H_2d: torch.Tensor, grad_K_2d: torch.Tensor,
                               H: torch.Tensor, K: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all 3D quantities (principal directions, gradients, signature)."""
        v1_3d, v2_3d, grad_H_3d, grad_K_3d = self._map_to_3d(
            jacobian=jacobian, v1=v1_2d, v2=v2_2d,
            grad_H=grad_H_2d, grad_K=grad_K_2d
        )
        signature = self._compute_3d_euclidean_signatures(
            H=H, K=K, grad_H_3d=grad_H_3d, grad_K_3d=grad_K_3d,
            v1_3d=v1_3d, v2_3d=v2_3d
        )
        return {
            'v1_3d': v1_3d,
            'v2_3d': v2_3d,
            'grad_H_3d': grad_H_3d,
            'grad_K_3d': grad_K_3d,
            'signature': signature
        }

    def _compute_all_differential_geometry(self, x: torch.Tensor, y: torch.Tensor,
                                           dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute ALL differential geometry quantities unconditionally."""
        # Curvature quantities
        H, K, k1, k2, v1_2d, v2_2d = self._compute_curvature_quantities(
            dz_dx=dz_dx, dz_dy=dz_dy, x=x, y=y
        )

        # Gradients in 2D
        grad_H_2d, grad_K_2d = self._compute_gradients_2d(H=H, K=K, x=x, y=y)

        # Jacobian and 3D quantities
        jacobian = self._compute_jacobian(dz_dx=dz_dx, dz_dy=dz_dy)
        quantities_3d = self._compute_3d_quantities(
            jacobian=jacobian, v1_2d=v1_2d, v2_2d=v2_2d,
            grad_H_2d=grad_H_2d, grad_K_2d=grad_K_2d, H=H, K=K
        )

        # Combine all quantities
        return {
            'H': H,
            'K': K,
            'k1': k1,
            'k2': k2,
            'v1_2d': v1_2d,
            'v2_2d': v2_2d,
            'grad_H_2d': grad_H_2d,
            'grad_K_2d': grad_K_2d,
            **quantities_3d
        }

    def _filter_differential_geometry(self, all_diff_geom: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Filter to only keep quantities specified in _diff_geom_components."""
        result = {}
        for component in self._diff_geom_components:
            for key in DIFF_GEOM_COMPONENT_KEYS.get(component, []):
                if key in all_diff_geom:
                    result[key] = all_diff_geom[key]
        return result

    def _compute_differential_geometry(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                                       dz_dx: torch.Tensor, dz_dy: torch.Tensor,
                                       origin_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute requested differential geometry quantities using precomputed first derivatives."""
        if not self._diff_geom_components:
            return {}

        if self._diff_geom_at_origin_only:
            if origin_data is None:
                raise ValueError("origin_data required when diff_geom_at_origin_only=True")
            x_eval = origin_data['x']
            y_eval = origin_data['y']
            dz_dx_eval = origin_data['dz_dx']
            dz_dy_eval = origin_data['dz_dy']
        else:
            x_eval, y_eval, dz_dx_eval, dz_dy_eval = x, y, dz_dx, dz_dy

        all_diff_geom = self._compute_all_differential_geometry(x=x_eval, y=y_eval, dz_dx=dz_dx_eval, dz_dy=dz_dy_eval)
        return self._filter_differential_geometry(all_diff_geom)

    def _create_raw_surface_data(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                                 dz_dx: torch.Tensor, dz_dy: torch.Tensor, points_scale: float,
                                 surface_params: Optional[Dict[str, Any]] = None) -> Data:
        """
        Create raw surface data object with positions, mesh, normals, and differential geometry.
        Does NOT add features - those are added later by the base class.
        Modified to handle origin-only computation and center the surface at origin.
        """
        # Compute all origin data once (derivatives, normal, H for flipping)
        origin_data = self._compute_origin_data(surface_params)

        # Create positions and translate so that (0,0,z_center) becomes (0,0,0)
        center_point = torch.stack([origin_data['x'], origin_data['y'], origin_data['z']], dim=1) * points_scale  # (1, 3)
        positions = torch.stack([x, y, z], dim=1) * points_scale  # (N, 3)
        positions = positions - center_point  # Translate so center is at origin

        # Create base surface data with translated positions
        data = Data()
        data['pos'] = positions.detach()
        data['face'] = self._create_surface_mesh(pos=data.pos).detach()

        # Store the origin position (0,0,0) after centering - will be transformed with pose
        data['origin_pos'] = torch.zeros(1, 3)

        # Compute origin normal (with potential flipping based on H)
        origin_normal = self._compute_surface_normals(
            dz_dx=origin_data['dz_dx'],
            dz_dy=origin_data['dz_dy'],
            H_at_origin=origin_data['H']
        )

        # Compute normals: origin-only or all points
        if self._diff_geom_at_origin_only:
            data['normal'] = origin_normal.detach()
        else:
            data['normal'] = self._compute_surface_normals(
                dz_dx=dz_dx,
                dz_dy=dz_dy,
                H_at_origin=origin_data['H']
            ).detach()

        # Compute differential geometry (unified method handles origin-only vs all points)
        diff_geom = self._compute_differential_geometry(
            x=x, y=y, z=z, dz_dx=dz_dx, dz_dy=dz_dy, origin_data=origin_data
        )
        for key, value in diff_geom.items():
            data[key] = value.detach()

        # Apply pose transformation to positions and differential quantities
        # ALWAYS pass the origin normal, regardless of the diff_geom_at_origin_only flag
        data = self._repose_surface_and_quantities(data=data, normals=origin_normal)

        # NOTE: Features are NOT added here - they're added by the base class
        # after calling _generate_raw_surfaces()

        return data

    def _ensure_origin_in_grid(self, x: torch.Tensor, y: torch.Tensor,
                               grid_range: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """
        Ensure the origin (0,0) is included in the grid points.

        Args:
            x, y: Grid point coordinates
            grid_range: The (min, max) range of the grid

        Returns:
            x, y: Updated coordinates (with origin added if needed)
            origin_idx: Index of the origin point in the grid, or None if origin is outside grid range
        """
        # Check if origin is within the grid range
        if not (grid_range[0] <= 0.0 <= grid_range[1]):
            return x, y, None

        # Check if origin already exists in the grid (within tolerance)
        tolerance = 1e-6
        distances_sq = x ** 2 + y ** 2
        min_dist_sq = distances_sq.min().item()

        if min_dist_sq < tolerance ** 2:
            # Origin already exists, find its index
            origin_idx = distances_sq.argmin().item()
            return x, y, origin_idx

        # Add origin to the grid
        x = torch.cat([x, torch.tensor([0.0])])
        y = torch.cat([y, torch.tensor([0.0])])
        origin_idx = len(x) - 1

        return x, y, origin_idx

    def _generate_raw_surfaces(self) -> List[Data]:
        """Generate multiple samplings of the same parametric surface using grid samplers."""
        # Generate surface parameters once (shared across all samplings)
        surface_params = self._generate_surface_parameters()

        # Sample other parameters once
        grid_radius = self._sample_parameter(param_range=self._grid_radius_range)
        points_scale = self._sample_parameter(param_range=self._points_scale_range)
        grid_range = (-grid_radius, grid_radius)  # Always centered at (0, 0)

        surfaces = []
        for grid_sampler in self._grid_samplers:
            # Generate grid for this sampling using the grid sampler
            x, y = grid_sampler.sample(grid_range=grid_range, rng=self._rng)

            # Optionally ensure origin is in the grid
            origin_idx = None
            if self._include_origin_in_grid:
                x, y, origin_idx = self._ensure_origin_in_grid(x, y, grid_range)

            with torch.enable_grad():
                # Convert to float32 and enable gradients
                x = x.to(dtype=torch.float32).requires_grad_(True)
                y = y.to(dtype=torch.float32).requires_grad_(True)

                # Evaluate surface using shared parameters
                z = self._evaluate_surface_with_parameters(x=x, y=y, surface_params=surface_params).to(dtype=torch.float32)

                # Compute first derivatives BEFORE creating surface data
                dz_dx, dz_dy = self._compute_first_derivatives(x=x, y=y, z=z)

                # Create surface data object WITHOUT features (features added later by base class)
                data = self._create_raw_surface_data(
                    x=x, y=y, z=z, dz_dx=dz_dx, dz_dy=dz_dy,
                    points_scale=points_scale, surface_params=surface_params
                )

            # Store origin index if we added/found one
            if origin_idx is not None:
                data['origin_idx'] = torch.tensor([origin_idx])

            surfaces.append(data)

        return surfaces

    @abstractmethod
    def _generate_surface_parameters(self) -> Dict[str, Any]:
        """Generate random parameters for the surface (coefficients, etc.)."""
        pass

    @abstractmethod
    def _evaluate_surface_with_parameters(self, x: torch.Tensor, y: torch.Tensor, surface_params: Dict[str, Any]) -> torch.Tensor:
        """Evaluate surface height at given parameter coordinates using pre-generated parameters."""
        pass


class PolynomialSurfaceDataset(ParametricSurfaceDataset):
    """Dataset for polynomial surfaces with optional coordinate offset."""

    def __init__(
            self,
            order_range: Tuple[int, int],
            coefficient_scale_range: Tuple[float, float],
            coeff_generation_method: CoeffGenerationMethod,
            polynomial_offset_range: Tuple[float, float] = (0.0, 0.0),
            **kwargs
    ):
        """
        Initialize PolynomialSurfaceDataset.

        Args:
            order_range: Range of polynomial orders (min, max)
            coefficient_scale_range: Range for scaling coefficients
            coeff_generation_method: Method for generating coefficients (UNIFORM or NORMAL)
            polynomial_offset_range: Range for random offset applied to polynomial evaluation.
                                     The polynomial is evaluated at (x + offset_x, y + offset_y),
                                     effectively "sliding" the surface under the grid.
            **kwargs: Additional arguments passed to ParametricSurfaceDataset
        """
        super().__init__(**kwargs)
        self._order_range = self._validate_order_range(order_range=order_range)
        self._coefficient_scale_range = self._validate_range(param_range=coefficient_scale_range, name="coefficient_scale_range")
        self._polynomial_offset_range = self._validate_range(param_range=polynomial_offset_range, name="polynomial_offset_range")
        self._coeff_generation_method = coeff_generation_method

    def _validate_order_range(self, order_range: Tuple[int, int]) -> Tuple[int, int]:
        """Validate polynomial order range."""
        if len(order_range) != 2:
            raise ValueError(f"Order range must have 2 elements, got {len(order_range)}")
        if order_range[0] < 1 or order_range[1] < 1:
            raise ValueError("Polynomial order must be >= 1")
        if order_range[0] > order_range[1]:
            raise ValueError(f"Order range min {order_range[0]} > max {order_range[1]}")
        return order_range

    @staticmethod
    def _get_polynomial_pairs(order: int) -> List[Tuple[int, int]]:
        """Get list of (i, j) exponent pairs for polynomial of given order."""
        return [(i, j) for i in range(order + 1) for j in range(order + 1) if 0 < i + j <= order]

    def _generate_surface_parameters(self) -> Dict[str, Any]:
        """Generate random polynomial coefficients, order, and coordinate offsets."""
        order = int(self._rng.integers(low=self._order_range[0], high=self._order_range[1] + 1))
        pairs = self._get_polynomial_pairs(order)
        num_coeffs = len(pairs)
        coefficient_scale = self._sample_parameter(param_range=self._coefficient_scale_range)

        if self._coeff_generation_method == CoeffGenerationMethod.UNIFORM:
            coefficients = torch.tensor(2 * (self._rng.uniform(size=num_coeffs) - 0.5) * coefficient_scale)
        elif self._coeff_generation_method == CoeffGenerationMethod.NORMAL:
            coefficients = torch.tensor(self._rng.normal(size=num_coeffs) * coefficient_scale)
        else:
            raise ValueError(f"Invalid coefficient generation method: {self._coeff_generation_method}")

        # Sample coordinate offsets
        offset_x = self._sample_parameter(param_range=self._polynomial_offset_range)
        offset_y = self._sample_parameter(param_range=self._polynomial_offset_range)

        return {
            'coefficients': coefficients,
            'order': order,
            'pairs': pairs,
            'offset': (offset_x, offset_y)
        }

    def _evaluate_surface_with_parameters(self, x: torch.Tensor, y: torch.Tensor, surface_params: Dict[str, Any]) -> torch.Tensor:
        """Evaluate polynomial surface using pre-generated parameters with coordinate offset."""
        coefficients = surface_params['coefficients']
        pairs = surface_params['pairs']
        offset_x, offset_y = surface_params['offset']

        # Apply offset: evaluate polynomial at shifted coordinates
        # This effectively "slides" the surface under the grid
        x_shifted = x + offset_x
        y_shifted = y + offset_y

        z = torch.zeros_like(x)
        for c, (i, j) in zip(coefficients, pairs):
            z += c * (x_shifted ** i) * (y_shifted ** j)
        return z