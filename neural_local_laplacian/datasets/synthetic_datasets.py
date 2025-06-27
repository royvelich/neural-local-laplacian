# Standard library
from typing import Tuple, List, Optional, Union, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod

# torch
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data

# torch geometric
from torch_geometric.nn import knn_graph
from torch_geometric.nn import fps

# numpy
import numpy as np

# scipy
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
from scipy.interpolate import Rbf

# triangle
import triangle

# igl
import igl

# neural signatures
from neural_local_laplacian.utils import utils
from neural_local_laplacian.datasets.base_datasets import (
    CoeffGenerationMethod,
    DeformationType,
    FeaturesType)
from neural_local_laplacian.utils.features import FeatureExtractor
from neural_local_laplacian.utils.pose_transformers import PoseTransformer

# trimesh
import trimesh

# noise
from noise import snoise3


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

        return torch.tensor(data=points[:, 0]), torch.tensor(data=points[:, 1])

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
    PRINCIPAL_DIRECTIONS = 'principal_directions'
    SIGNATURE = 'signature'


class SyntheticSurfaceDataset(ABC, Dataset):
    """Base class for synthetic surface datasets."""

    def __init__(
            self,
            epoch_size: int,
            pose_transformer: Optional[PoseTransformer],
            seed: int,
            feature_extractor: Optional[FeatureExtractor] = None,
            conv_k_nearest: Optional[int] = None,
    ):
        super().__init__()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._epoch_size = epoch_size
        self._pose_transformer = pose_transformer
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
        """Apply pose transformation to surface and transform differential quantities accordingly."""
        if self._pose_transformer is None:
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

        # Use transformer to get translation and rotation
        translation, rotation_matrix = self._pose_transformer.transform(data.pos, normal)

        # Apply translation and rotation to positions
        data.pos = data.pos + translation
        data.pos = torch.matmul(data.pos, rotation_matrix.T)

        # Transform normals using rotation matrix
        if 'normal' in data:
            data['normal'] = torch.matmul(data['normal'], rotation_matrix.T)

        # Transform differential geometry quantities
        vector_3d_keys = ['v1_3d', 'v2_3d', 'grad_H_3d', 'grad_K_3d']
        for key in vector_3d_keys:
            if key in data:
                data[key] = torch.matmul(data[key], rotation_matrix.T)

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
            grid_samplers: List[GridSampler],  # Added grid_samplers parameter here
            grid_radius_range: Tuple[float, float],
            grid_offset_range: Tuple[float, float],
            points_scale_range: Tuple[float, float],
            diff_geom_components: Optional[List[DifferentialGeometryComponent]] = None,
            diff_geom_at_origin_only: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._grid_samplers = grid_samplers  # Store grid samplers here
        self._grid_radius_range = self._validate_range(param_range=grid_radius_range, name="grid_radius_range")
        self._grid_offset_range = self._validate_range(param_range=grid_offset_range, name="grid_offset_range")
        self._points_scale_range = self._validate_range(param_range=points_scale_range, name="points_scale_range")
        self._diff_geom_at_origin_only = diff_geom_at_origin_only

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

    def _compute_surface_normals(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor,
                                 surface_params: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Compute surface normals using precomputed derivatives for a graph z = f(x,y)."""

        # Determine if we need to compute normal at origin
        need_origin_normal = (dz_dx is None or dz_dy is None or self._diff_geom_at_origin_only)

        if need_origin_normal:
            if surface_params is None:
                raise ValueError("surface_params required when computing normal at origin")

            # Compute normal at origin using helper method
            return self._compute_normal_at_origin(surface_params)
        else:
            # Compute normals for all points using precomputed derivatives
            return self._compute_normal_from_derivatives(dz_dx, dz_dy)

    def _compute_normal_at_origin(self, surface_params: Dict[str, Any]) -> torch.Tensor:
        """Helper method to compute surface normal at origin (0,0)."""
        # Evaluate derivatives at origin (0,0)
        x_origin = torch.tensor([0.0], requires_grad=True)
        y_origin = torch.tensor([0.0], requires_grad=True)
        z_origin = self._evaluate_surface_with_parameters(x=x_origin, y=y_origin, surface_params=surface_params)

        dz_dx_origin = torch.autograd.grad(
            outputs=z_origin,
            inputs=x_origin,
            create_graph=True,
            retain_graph=True
        )[0]
        dz_dy_origin = torch.autograd.grad(
            outputs=z_origin,
            inputs=y_origin,
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute and return normalized normal at origin
        return self._compute_normal_from_derivatives(dz_dx_origin, dz_dy_origin)

    def _compute_normal_from_derivatives(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> torch.Tensor:
        """Helper method to compute normal from partial derivatives."""
        normal = torch.stack(tensors=[-dz_dx, -dz_dy, torch.ones_like(input=dz_dx)], dim=1)
        return F.normalize(input=normal, p=2, dim=1)

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
        """Compute first partial derivatives ∂z/∂x and ∂z/∂y."""
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
            torch.stack([torch.ones_like(input=dz_dx), torch.zeros_like(input=dz_dx)], dim=-1),
            torch.stack([torch.zeros_like(input=dz_dy), torch.ones_like(input=dz_dy)], dim=-1),
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
        v1_3d_norm = v1_3d / torch.norm(input=v1_3d, dim=1, keepdim=True)
        v2_3d_norm = v2_3d / torch.norm(input=v2_3d, dim=1, keepdim=True)

        # Compute directional derivatives
        H_1 = torch.sum(input=grad_H_3d * v1_3d_norm, dim=1)
        H_2 = torch.sum(input=grad_H_3d * v2_3d_norm, dim=1)
        K_1 = torch.sum(input=grad_K_3d * v1_3d_norm, dim=1)
        K_2 = torch.sum(input=grad_K_3d * v2_3d_norm, dim=1)

        # Stack signature components
        signature = torch.stack(tensors=[H, K, H_1, H_2, K_1, K_2], dim=1)
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

    def _compute_and_add_gradients(self, H: torch.Tensor, K: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradient components for internal use (signature computation)."""
        grad_H_2d, grad_K_2d = self._compute_curvature_gradients(H=H, K=K, x=x, y=y)
        grad_H_2d = torch.stack(tensors=grad_H_2d, dim=-1)
        grad_K_2d = torch.stack(tensors=grad_K_2d, dim=-1)
        return grad_H_2d, grad_K_2d

    def _compute_and_add_signature(self, diff_geom: Dict[str, torch.Tensor], jacobian: torch.Tensor, v1_2d: torch.Tensor, v2_2d: torch.Tensor, grad_H_2d: torch.Tensor, grad_K_2d: torch.Tensor, H: torch.Tensor, K: torch.Tensor) -> None:
        """Compute and add signature components."""
        v1_3d, v2_3d, grad_H_3d, grad_K_3d = self._map_to_3d(
            jacobian=jacobian, v1=v1_2d, v2=v2_2d,
            grad_H=grad_H_2d, grad_K=grad_K_2d
        )
        signature = self._compute_3d_euclidean_signatures(
            H=H, K=K, grad_H_3d=grad_H_3d, grad_K_3d=grad_K_3d,
            v1_3d=v1_3d, v2_3d=v2_3d
        )

        diff_geom.update({
            'v1_3d': v1_3d, 'v2_3d': v2_3d,
            'grad_H_3d': grad_H_3d, 'grad_K_3d': grad_K_3d,
            'signature': signature
        })

    def _compute_differential_geometry_at_origin(self, surface_params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute differential geometry quantities only at the (0,0) point."""
        if not self._diff_geom_components:
            return {}

        # Evaluate directly at (0,0)
        x_origin = torch.tensor([0.0], requires_grad=True)
        y_origin = torch.tensor([0.0], requires_grad=True)
        z_origin = self._evaluate_surface_with_parameters(x=x_origin, y=y_origin, surface_params=surface_params)

        # Compute derivatives at origin
        dz_dx_origin = torch.autograd.grad(
            outputs=z_origin,
            inputs=x_origin,
            create_graph=True,
            retain_graph=True
        )[0]
        dz_dy_origin = torch.autograd.grad(
            outputs=z_origin,
            inputs=y_origin,
            create_graph=True,
            retain_graph=True
        )[0]

        diff_geom = {}

        # Check what computations are needed
        needs_curvatures = any(comp in self._diff_geom_components for comp in [
            DifferentialGeometryComponent.MEAN_CURVATURE,
            DifferentialGeometryComponent.GAUSSIAN_CURVATURE,
            DifferentialGeometryComponent.PRINCIPAL_CURVATURES
        ])
        needs_principal_dirs = DifferentialGeometryComponent.PRINCIPAL_DIRECTIONS in self._diff_geom_components
        needs_signature = DifferentialGeometryComponent.SIGNATURE in self._diff_geom_components

        if any([needs_curvatures, needs_principal_dirs, needs_signature]):
            # Compute curvatures at origin
            H_origin, K_origin, k1_origin, k2_origin, v1_2d_origin, v2_2d_origin = self._compute_curvature_quantities(
                dz_dx=dz_dx_origin, dz_dy=dz_dy_origin, x=x_origin, y=y_origin
            )

            # Store only the origin values (single values, not tensors for all points)
            if DifferentialGeometryComponent.MEAN_CURVATURE in self._diff_geom_components:
                diff_geom['H'] = H_origin  # Shape: (1,)
            if DifferentialGeometryComponent.GAUSSIAN_CURVATURE in self._diff_geom_components:
                diff_geom['K'] = K_origin  # Shape: (1,)
            if DifferentialGeometryComponent.PRINCIPAL_CURVATURES in self._diff_geom_components:
                diff_geom.update({
                    'k1': k1_origin,  # Shape: (1,)
                    'k2': k2_origin  # Shape: (1,)
                })
            if needs_principal_dirs:
                diff_geom.update({
                    'v1_2d': v1_2d_origin,  # Shape: (1, 2)
                    'v2_2d': v2_2d_origin  # Shape: (1, 2)
                })

            # Signature computation at origin
            if needs_signature:
                grad_H_2d_origin, grad_K_2d_origin = self._compute_and_add_gradients(
                    H=H_origin, K=K_origin, x=x_origin, y=y_origin
                )
                jacobian_origin = self._compute_jacobian(dz_dx=dz_dx_origin, dz_dy=dz_dy_origin)

                # Create a temporary dict for signature computation
                temp_diff_geom = {}
                self._compute_and_add_signature(
                    diff_geom=temp_diff_geom, jacobian=jacobian_origin,
                    v1_2d=v1_2d_origin, v2_2d=v2_2d_origin,
                    grad_H_2d=grad_H_2d_origin, grad_K_2d=grad_K_2d_origin,
                    H=H_origin, K=K_origin
                )

                # Store only origin signature components
                for key in ['v1_3d', 'v2_3d', 'grad_H_3d', 'grad_K_3d', 'signature']:
                    if key in temp_diff_geom:
                        diff_geom[key] = temp_diff_geom[key]  # Keep original shapes (1, ...)

        return diff_geom

    def _compute_differential_geometry(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                                       dz_dx: torch.Tensor, dz_dy: torch.Tensor,
                                       surface_params: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Compute requested differential geometry quantities using precomputed first derivatives."""
        if self._diff_geom_at_origin_only:
            if surface_params is None:
                raise ValueError("surface_params required when diff_geom_at_origin_only=True")
            return self._compute_differential_geometry_at_origin(surface_params=surface_params)
        else:
            # Original behavior: compute for all points
            return self._compute_differential_geometry_original(x=x, y=y, z=z, dz_dx=dz_dx, dz_dy=dz_dy)

    def _compute_differential_geometry_original(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                                                dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Original differential geometry computation for all points."""
        if not self._diff_geom_components:
            return {}

        diff_geom = {}

        # Check what computations are needed
        needs_curvatures = any(comp in self._diff_geom_components for comp in [
            DifferentialGeometryComponent.MEAN_CURVATURE,
            DifferentialGeometryComponent.GAUSSIAN_CURVATURE,
            DifferentialGeometryComponent.PRINCIPAL_CURVATURES
        ])
        needs_principal_dirs = DifferentialGeometryComponent.PRINCIPAL_DIRECTIONS in self._diff_geom_components
        needs_signature = DifferentialGeometryComponent.SIGNATURE in self._diff_geom_components

        # Curvature computations
        if any([needs_curvatures, needs_principal_dirs, needs_signature]):
            H, K, k1, k2, v1_2d, v2_2d = self._compute_curvature_quantities(dz_dx=dz_dx, dz_dy=dz_dy, x=x, y=y)

            # Store individual curvature components
            if DifferentialGeometryComponent.MEAN_CURVATURE in self._diff_geom_components:
                diff_geom['H'] = H
            if DifferentialGeometryComponent.GAUSSIAN_CURVATURE in self._diff_geom_components:
                diff_geom['K'] = K
            if DifferentialGeometryComponent.PRINCIPAL_CURVATURES in self._diff_geom_components:
                diff_geom.update({'k1': k1, 'k2': k2})
            if needs_principal_dirs:
                diff_geom.update({'v1_2d': v1_2d, 'v2_2d': v2_2d})

        # Signature computation (computes gradients and jacobian internally)
        if needs_signature:
            # Compute gradients internally for signature (not stored)
            grad_H_2d, grad_K_2d = self._compute_and_add_gradients(H=H, K=K, x=x, y=y)

            # Compute jacobian internally for signature (not stored)
            jacobian = self._compute_jacobian(dz_dx=dz_dx, dz_dy=dz_dy)
            self._compute_and_add_signature(
                diff_geom=diff_geom, jacobian=jacobian, v1_2d=v1_2d, v2_2d=v2_2d,
                grad_H_2d=grad_H_2d, grad_K_2d=grad_K_2d, H=H, K=K
            )

        return diff_geom

    def _create_base_surface_data(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                                  dz_dx: torch.Tensor, dz_dy: torch.Tensor, points_scale: float,
                                  surface_params: Optional[Dict[str, Any]] = None) -> Data:
        """
        Create base surface data object with positions, mesh, normals, and differential geometry.
        When diff_geom_at_origin_only=True, computes ALL quantities only at origin.
        """
        # Create base data object with scaled positions and mesh
        data = Data()
        data['pos'] = torch.stack(tensors=[x, y, z], dim=1) * points_scale
        data['face'] = self._create_surface_mesh(pos=data.pos)

        if self._diff_geom_at_origin_only:
            # Compute ALL quantities (normal + differential geometry) ONLY at origin
            if surface_params is None:
                raise ValueError("surface_params required when diff_geom_at_origin_only=True")

            # Compute normal at origin only
            data['normal'] = self._compute_surface_normals(dz_dx=None, dz_dy=None, surface_params=surface_params)

            # Compute differential geometry at origin only
            diff_geom = self._compute_differential_geometry_at_origin(surface_params=surface_params)

            # Store all quantities with same keys as before (seamless operation)
            for key, value in diff_geom.items():
                data[key] = value

        else:
            # Original behavior: compute for all points
            data['normal'] = self._compute_surface_normals(dz_dx=dz_dx, dz_dy=dz_dy, surface_params=surface_params)

            diff_geom = self._compute_differential_geometry_original(x=x, y=y, z=z, dz_dx=dz_dx, dz_dy=dz_dy)
            for key, value in diff_geom.items():
                data[key] = value

        return data

    def _create_raw_surface_data(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                                 dz_dx: torch.Tensor, dz_dy: torch.Tensor, points_scale: float,
                                 surface_params: Optional[Dict[str, Any]] = None) -> Data:
        """
        Create raw surface data object with positions, mesh, normals, and differential geometry.
        Does NOT add features - those are added later by the base class.
        Modified to handle origin-only computation and center the surface at origin.
        """
        # Evaluate the surface at (0,0) to get the center point
        x_center = torch.tensor([0.0], requires_grad=True)
        y_center = torch.tensor([0.0], requires_grad=True)
        z_center = self._evaluate_surface_with_parameters(x=x_center, y=y_center, surface_params=surface_params)

        # Create positions and translate so that (0,0,z_center) becomes (0,0,0)
        center_point = torch.stack([x_center, y_center, z_center], dim=1) * points_scale  # (1, 3)
        positions = torch.stack([x, y, z], dim=1) * points_scale  # (N, 3)
        positions = positions - center_point  # Translate so center is at origin

        # Create base surface data with translated positions
        data = Data()
        data['pos'] = positions.detach()
        data['face'] = self._create_surface_mesh(pos=data.pos).detach()

        # ALWAYS compute normal at origin for pose transformation
        # This ensures _repose_surface_and_quantities always gets the origin normal
        origin_normal = self._compute_surface_normals(dz_dx=None, dz_dy=None, surface_params=surface_params)

        if self._diff_geom_at_origin_only:
            # When origin-only mode: use the origin normal we just computed
            data['normal'] = origin_normal.detach()

            # Compute differential geometry at origin only
            diff_geom = self._compute_differential_geometry_at_origin(surface_params=surface_params)

            # Store all quantities with same keys as before (seamless operation)
            for key, value in diff_geom.items():
                data[key] = value.detach()

        else:
            # When computing for all points: store normals for all points in data,
            # but we'll still use origin_normal for pose transformation
            data['normal'] = self._compute_surface_normals(dz_dx=dz_dx, dz_dy=dz_dy, surface_params=surface_params).detach()

            diff_geom = self._compute_differential_geometry_original(x=x, y=y, z=z, dz_dx=dz_dx, dz_dy=dz_dy)
            for key, value in diff_geom.items():
                data[key] = value.detach()

        # Apply pose transformation to positions and differential quantities
        # ALWAYS pass the origin normal, regardless of the diff_geom_at_origin_only flag
        data = self._repose_surface_and_quantities(data=data, normals=origin_normal)

        # NOTE: Features are NOT added here - they're added by the base class
        # after calling _generate_raw_surfaces()

        return data

    def _generate_raw_surfaces(self) -> List[Data]:
        """Generate multiple samplings of the same parametric surface using grid samplers."""
        # Generate surface parameters once (shared across all samplings)
        surface_params = self._generate_surface_parameters()

        # Sample other parameters once
        grid_radius = self._sample_parameter(param_range=self._grid_radius_range)
        points_scale = self._sample_parameter(param_range=self._points_scale_range)
        grid_offset = self._sample_parameter(param_range=self._grid_offset_range)
        grid_range = (-grid_radius + grid_offset, grid_radius + grid_offset)

        surfaces = []
        for grid_sampler in self._grid_samplers:
            # Generate grid for this sampling using the grid sampler
            x, y = grid_sampler.sample(grid_range=grid_range, rng=self._rng)

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
    """Dataset for polynomial surfaces."""

    def __init__(
            self,
            order_range: Tuple[int, int],
            coefficient_scale_range: Tuple[float, float],
            coeff_generation_method: CoeffGenerationMethod,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._order_range = self._validate_order_range(order_range=order_range)
        self._coefficient_scale_range = self._validate_range(param_range=coefficient_scale_range, name="coefficient_scale_range")
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
    def _get_num_coeffs(order: int) -> int:
        """Calculate number of coefficients for polynomial order."""
        return sum(1 for x in range(order + 1) for y in range(order + 1) if 0 < x + y <= order)

    def _generate_surface_parameters(self) -> Dict[str, Any]:
        """Generate random polynomial coefficients and order."""
        order = int(self._rng.integers(low=self._order_range[0], high=self._order_range[1] + 1))
        num_coeffs = self._get_num_coeffs(order=order)
        coefficient_scale = self._sample_parameter(param_range=self._coefficient_scale_range)

        if self._coeff_generation_method == CoeffGenerationMethod.UNIFORM:
            coefficients = torch.tensor(data=2 * (self._rng.uniform(size=num_coeffs) - 0.5) * coefficient_scale)
        elif self._coeff_generation_method == CoeffGenerationMethod.NORMAL:
            coefficients = torch.tensor(data=self._rng.normal(size=num_coeffs) * coefficient_scale)
        else:
            raise ValueError(f"Invalid coefficient generation method: {self._coeff_generation_method}")

        return {
            'coefficients': coefficients,
            'order': order
        }

    def _evaluate_surface_with_parameters(self, x: torch.Tensor, y: torch.Tensor, surface_params: Dict[str, Any]) -> torch.Tensor:
        """Evaluate polynomial surface using pre-generated parameters."""
        coefficients = surface_params['coefficients']
        order = surface_params['order']

        pairs = [(i, j) for i in range(order + 1) for j in range(order + 1) if 0 < i + j <= order]
        z = torch.zeros_like(input=x)
        for c, pair in zip(coefficients, pairs):
            z += c * (x ** pair[0]) * (y ** pair[1])
        return z