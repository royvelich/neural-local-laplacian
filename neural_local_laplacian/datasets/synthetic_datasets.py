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
    GridGenerationMethod,
    CoeffGenerationMethod,
    DeformationType,
    PoseType,
    FeaturesType)


class DifferentialGeometryComponent(Enum):
    MEAN_CURVATURE = 'mean_curvature'
    GAUSSIAN_CURVATURE = 'gaussian_curvature'
    PRINCIPAL_CURVATURES = 'principal_curvatures'
    PRINCIPAL_DIRECTIONS = 'principal_directions'
    SIGNATURE = 'signature'


# trimesh
import trimesh

# noise
from noise import snoise3


class SyntheticSurfaceDataset(ABC, Dataset):
    """Base class for synthetic surface datasets."""

    def __init__(
            self,
            epoch_size: int,
            grid_points_count: int,
            sampling_ratio_range: Tuple[float, float],
            sampled_surfaces: int,
            pose_type: Optional[PoseType],
            seed: int,
            add_regularized_surface: bool,
            features_type: FeaturesType,
            conv_k_nearest: Optional[int],
    ):
        super().__init__()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._epoch_size = epoch_size
        self._grid_points_count = grid_points_count
        self._sampling_ratio_range = sampling_ratio_range
        self._sampled_surfaces = sampled_surfaces
        self._pose_type = pose_type
        self._add_regularized_surface = add_regularized_surface
        self._features_type = features_type
        self._conv_k_nearest = conv_k_nearest

    def reset_rng(self) -> None:
        """Reset random number generator to initial seed."""
        self._rng = np.random.default_rng(seed=self._seed)

    def len(self) -> int:
        return self._epoch_size

    @abstractmethod
    def _generate_surface(self, downsample_ratio: Optional[float]) -> Data:
        """Generate a surface with optional downsampling."""
        pass

    def _repose_surface_and_quantities(self, data: Data, normals: torch.Tensor, diff_geom: Dict[str, torch.Tensor]) -> Data:
        """Apply pose transformation to surface and transform differential quantities accordingly."""
        center = torch.mean(data.pos, dim=0)
        points = data.pos - center

        rotation_matrix = None
        if self._pose_type == PoseType.RANDOM_ROTATION:
            rotation_matrix = utils.random_rotation_matrix()
            points = torch.matmul(points, rotation_matrix)
        elif self._pose_type == PoseType.PCA:
            points_canonical, rotation_matrix, translation = utils.compute_canonical_pose_pca(points=points)
            points = points_canonical

        data.pos = points

        # Transform normals using rotation matrix (now available for both pose types)
        if rotation_matrix is not None:
            normals_transformed = torch.matmul(normals, rotation_matrix)
            data['normal'] = normals_transformed

        # Transform differential geometry quantities
        for key, value in diff_geom.items():
            if key in ['v1_3d', 'v2_3d', 'grad_H_3d', 'grad_K_3d'] and rotation_matrix is not None:
                # Transform 3D vector quantities
                data[key] = torch.matmul(value, rotation_matrix)
            else:
                # Keep scalar quantities and 2D quantities unchanged
                data[key] = value

        return data

    def get(self, idx: int) -> List[List[Data]]:
        surfaces = []
        downsample_ratios = list(self._rng.uniform(
            low=self._sampling_ratio_range[0],
            high=self._sampling_ratio_range[1],
            size=self._sampled_surfaces
        ))
        if self._add_regularized_surface:
            downsample_ratios = [None] + downsample_ratios

        for i, downsample_ratio in enumerate(downsample_ratios):
            data = self._generate_surface(downsample_ratio=downsample_ratio)
            # Note: reposing is now handled inside _generate_surface with proper transformation of differential quantities

            if self._features_type == FeaturesType.RISP:
                data['x'] = utils.compute_risp_features(points=data.pos, normals=data.normal)
            elif self._features_type == FeaturesType.XYZ:
                data['x'] = data.pos.unsqueeze(dim=1)

            surfaces.append(data)

        return [surfaces]


class ParametricSurfaceDataset(SyntheticSurfaceDataset):
    """Dataset for parametric surfaces with differential geometry computation."""

    def __init__(
            self,
            grid_generation_method: GridGenerationMethod,
            grid_radius_range: Tuple[float, float],
            grid_offset_range: Tuple[float, float],
            points_scale_range: Tuple[float, float],
            diff_geom_components: Optional[List[DifferentialGeometryComponent]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._grid_generation_method = grid_generation_method
        self._grid_radius_range = self._validate_range(param_range=grid_radius_range, name="grid_radius_range")
        self._grid_offset_range = self._validate_range(param_range=grid_offset_range, name="grid_offset_range")
        self._points_scale_range = self._validate_range(param_range=points_scale_range, name="points_scale_range")

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

    def _compute_surface_normals(self, dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> torch.Tensor:
        """Compute surface normals using precomputed derivatives for a graph z = f(x,y)."""
        # For a surface z = f(x,y), the normal vector is (-∂z/∂x, -∂z/∂y, 1)
        normal = torch.stack(tensors=[-dz_dx, -dz_dy, torch.ones_like(input=dz_dx)], dim=1)
        return F.normalize(input=normal, p=2, dim=1)

    def _generate_grid_points(self, grid_range: Tuple[float, float], downsample_ratio: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate grid points based on method and downsampling."""
        try:
            if self._grid_generation_method == GridGenerationMethod.MESH:
                return self._generate_mesh_grid(grid_range=grid_range)
            elif self._grid_generation_method == GridGenerationMethod.UNIFORM:
                if downsample_ratio is None:
                    raise ValueError("Downsample ratio required for uniform grid generation")
                return self._generate_uniform_grid(grid_range=grid_range, downsample_ratio=downsample_ratio)
            else:
                raise ValueError(f"Invalid grid generation method: {self._grid_generation_method}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate grid points: {e}")

    def _generate_mesh_grid(self, grid_range: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate regular mesh grid."""
        grid_points_count_sqrt = int(np.sqrt(self._grid_points_count))
        if grid_points_count_sqrt ** 2 != self._grid_points_count:
            # Adjust to nearest perfect square
            grid_points_count_sqrt = int(np.sqrt(self._grid_points_count))
            actual_points = grid_points_count_sqrt ** 2
            if actual_points != self._grid_points_count:
                print(f"Warning: Adjusted grid points from {self._grid_points_count} to {actual_points}")

        x_linspace = torch.linspace(start=grid_range[0], end=grid_range[1], steps=grid_points_count_sqrt)
        y_linspace = torch.linspace(start=grid_range[0], end=grid_range[1], steps=grid_points_count_sqrt)
        x, y = torch.meshgrid(x_linspace, y_linspace, indexing='ij')
        return x.flatten(), y.flatten()

    def _generate_uniform_grid(self, grid_range: Tuple[float, float], downsample_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate uniform random grid."""
        if not 0 < downsample_ratio <= 1:
            raise ValueError(f"Downsample ratio must be in (0, 1], got {downsample_ratio}")

        points_count = int(downsample_ratio * self._grid_points_count)
        if points_count < 3:
            raise ValueError(f"Too few points after downsampling: {points_count}")

        points = self._rng.uniform(low=grid_range[0], high=grid_range[1], size=(points_count, 2))
        return torch.tensor(data=points[:, 0]), torch.tensor(data=points[:, 1])

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

    def _compute_differential_geometry(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute requested differential geometry quantities using precomputed first derivatives."""
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

    @abstractmethod
    def _evaluate_random_surface(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate surface height at given parameter coordinates."""
        pass

    def _generate_surface(self, downsample_ratio: Optional[float]) -> Data:
        """Generate surface with differential geometry computation."""
        # Sample parameters
        grid_radius = self._sample_parameter(param_range=self._grid_radius_range)
        points_scale = self._sample_parameter(param_range=self._points_scale_range)
        grid_offset = self._sample_parameter(param_range=self._grid_offset_range)

        # Generate grid
        grid_range = (-grid_radius + grid_offset, grid_radius + grid_offset)
        if downsample_ratio is None:
            x, y = self._generate_mesh_grid(grid_range=grid_range)
        else:
            x, y = self._generate_uniform_grid(grid_range=grid_range, downsample_ratio=downsample_ratio)

        # Convert to float32 and enable gradients
        x = x.to(dtype=torch.float32).requires_grad_(requires_grad=True)
        y = y.to(dtype=torch.float32).requires_grad_(requires_grad=True)

        # Evaluate surface
        z = self._evaluate_random_surface(x=x, y=y).to(dtype=torch.float32)

        # Create data object with scaled positions
        data = Data()
        data['pos'] = torch.stack(tensors=[x, y, z], dim=1) * points_scale
        data['face'] = self._create_surface_mesh(pos=data.pos)

        # Compute first derivatives BEFORE reposing (while we still have parametric form)
        dz_dx, dz_dy = self._compute_first_derivatives(x=x, y=y, z=z)

        # Compute normals BEFORE reposing
        normals_original = self._compute_surface_normals(dz_dx=dz_dx, dz_dy=dz_dy)

        # Compute differential geometry BEFORE reposing
        diff_geom = self._compute_differential_geometry(x=x, y=y, z=z, dz_dx=dz_dx, dz_dy=dz_dy)

        # Now apply pose transformation to positions and differential quantities
        data = self._repose_surface_and_quantities(data=data, normals=normals_original, diff_geom=diff_geom)

        return data


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

    def _generate_coefficients(self) -> Tuple[torch.Tensor, int]:
        """Generate random polynomial coefficients."""
        order = int(self._rng.integers(low=self._order_range[0], high=self._order_range[1] + 1))
        num_coeffs = self._get_num_coeffs(order=order)
        coefficient_scale = self._sample_parameter(param_range=self._coefficient_scale_range)

        if self._coeff_generation_method == CoeffGenerationMethod.UNIFORM:
            coefficients = torch.tensor(data=2 * (self._rng.uniform(size=num_coeffs) - 0.5) * coefficient_scale)
        elif self._coeff_generation_method == CoeffGenerationMethod.NORMAL:
            coefficients = torch.tensor(data=self._rng.normal(size=num_coeffs) * coefficient_scale)
        else:
            raise ValueError(f"Invalid coefficient generation method: {self._coeff_generation_method}")

        return coefficients, order

    def _evaluate_random_surface(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate polynomial surface."""
        coefficients, order = self._generate_coefficients()
        pairs = [(i, j) for i in range(order + 1) for j in range(order + 1) if 0 < i + j <= order]
        z = torch.zeros_like(input=x)
        for c, pair in zip(coefficients, pairs):
            z += c * (x ** pair[0]) * (y ** pair[1])
        return z