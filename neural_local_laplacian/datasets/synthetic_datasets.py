# Standard library
from typing import Tuple, List, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod

# torch
import torch
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

# trimesh
import trimesh

# noise
from noise import snoise3


class SyntheticSurfaceDataset(ABC, Dataset):
    def __init__(
            self,
            epoch_size: int,
            grid_points_count: int,
            sampling_ratio_range: Tuple[float, float],
            sampled_patches: int,
            pose_type: Optional[PoseType],
            seed: int,
            add_regularized_mesh: bool,
            features_type: FeaturesType,
            max_ring: Optional[int],
            conv_k_nearest: Optional[int],
            deform: Optional[bool],
            deform_k_nearest: Optional[int],
            deform_iterations: Optional[int],
            deform_actions_per_iteration: Optional[int],
            deform_delta: Optional[float]
    ):
        super().__init__()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._epoch_size = epoch_size
        self._grid_points_count = grid_points_count
        self._sampling_ratio_range = sampling_ratio_range
        self._sampled_patches = sampled_patches
        self._max_ring = max_ring
        self._pose_type = pose_type
        self._add_regularized_mesh = add_regularized_mesh
        self._features_type = features_type
        self._conv_k_nearest = conv_k_nearest
        self._deform = deform
        self._deform_k_nearest = deform_k_nearest
        self._deform_iterations = deform_iterations
        self._deform_actions_per_iteration = deform_actions_per_iteration
        self._deform_delta = deform_delta

    def reset_rng(self):
        self._rng = np.random.default_rng(seed=self._seed)

    @abstractmethod
    def _generate_random_data(self) -> Data:
        pass

    def len(self):
        return self._epoch_size

    # def _calculate_control_groups(self, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     boundary_loop = igl.boundary_loop(faces)
    #
    #     # Define the minimum and maximum percentage of boundary vertices to use as separation
    #     min_separation_percent: float = 0.1
    #     max_separation_percent: float = 0.3
    #
    #     # Calculate the number of vertices to use for separation
    #     total_boundary_vertices: int = len(boundary_loop)
    #     min_separation: int = int(min_separation_percent * total_boundary_vertices)
    #     max_separation: int = int(max_separation_percent * total_boundary_vertices)
    #     separation: int = np.random.randint(min_separation, max_separation + 1)
    #
    #     # Calculate the number of control points for each group
    #     num_control_points: int = max(int(0.3 * (total_boundary_vertices - separation)), 2)  # Ensure at least 2 control points per group
    #
    #     # Select control points for left group
    #     left_start: int = 0
    #     left_end: int = num_control_points
    #     left_group: np.ndarray = boundary_loop[left_start:left_end]
    #
    #     # Select control points for right group
    #     right_start: int = left_end + separation
    #     right_end: int = right_start + num_control_points
    #     if right_end > total_boundary_vertices:
    #         right_start = (right_start % total_boundary_vertices)
    #         right_end = (right_end % total_boundary_vertices)
    #         right_group: np.ndarray = np.concatenate([boundary_loop[right_start:], boundary_loop[:right_end]])
    #     else:
    #         right_group: np.ndarray = boundary_loop[right_start:right_end]
    #
    #     return left_group, right_group
    #
    # def _deform(self, points: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    #     points = points.detach().cpu().numpy()
    #     faces = faces.detach().cpu().numpy()
    #     left_group, right_group = self._calculate_control_groups(faces=faces)
    #     control_indices = np.concatenate([left_group, right_group])
    #     arap = igl.ARAP(points, faces, 3, control_indices)
    #     bc = points[control_indices].copy()

    @staticmethod
    def _get_nearest_neighbors(edge_index: np.ndarray, point_index: int) -> np.ndarray:
        return edge_index[1][edge_index[0] == point_index]

    @staticmethod
    def _calculate_control_groups(vertices: np.ndarray, edge_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Use Farthest Point Sampling to select two control points
        vertices_torch = torch.from_numpy(vertices).float()
        control_centers = fps(vertices_torch, ratio=2.0 / len(vertices), random_start=True).numpy()
        fixed_center, deformed_center = control_centers

        # Get K nearest neighbors for each control point
        fixed_indices = SyntheticSurfaceDataset._get_nearest_neighbors(edge_index=edge_index, point_index=fixed_center)
        deformed_indices = SyntheticSurfaceDataset._get_nearest_neighbors(edge_index=edge_index, point_index=deformed_center)

        return fixed_indices, deformed_indices

    @staticmethod
    def _rotate_points(points: np.ndarray, axis: List[float], angle: float) -> np.ndarray:
        axis_np: np.ndarray = np.array(axis)
        axis_np = axis_np / np.linalg.norm(axis_np)
        rot: Rotation = Rotation.from_rotvec(axis_np * angle)
        return rot.apply(points)

    def _deform(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        edge_index = knn_graph(x=vertices, k=self._deform_k_nearest, loop=True).numpy()
        vertices = vertices.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()

        for _ in range(self._deform_iterations):
            fixed_indices, deformed_indices = SyntheticSurfaceDataset._calculate_control_groups(vertices=vertices, edge_index=edge_index)
            control_indices = np.concatenate([fixed_indices, deformed_indices])
            arap = igl.ARAP(vertices, faces, 3, control_indices)
            bc = vertices[control_indices].copy()
            # v_deformed = vertices.copy()

            for _ in range(self._deform_actions_per_iteration):
                deformation_type = self._rng.choice(list(DeformationType))
                if deformation_type == DeformationType.TRANSLATION_X:
                    bc[len(fixed_indices):, 0] += self._deform_delta
                elif deformation_type == DeformationType.TRANSLATION_Y:
                    bc[len(fixed_indices):, 1] += self._deform_delta
                elif deformation_type == DeformationType.TRANSLATION_Z:
                    bc[len(fixed_indices):, 2] += self._deform_delta
                elif deformation_type == DeformationType.ROTATION_X:
                    bc[len(fixed_indices):] = SyntheticSurfaceDataset._rotate_points(points=bc[len(fixed_indices):], axis=[1, 0, 0], angle=self._deform_delta)
                elif deformation_type == DeformationType.ROTATION_Y:
                    bc[len(fixed_indices):] = SyntheticSurfaceDataset._rotate_points(points=bc[len(fixed_indices):], axis=[0, 1, 0], angle=self._deform_delta)
                elif deformation_type == DeformationType.ROTATION_Z:
                    bc[len(fixed_indices):] = SyntheticSurfaceDataset._rotate_points(points=bc[len(fixed_indices):], axis=[0, 0, 1], angle=self._deform_delta)

            vertices = arap.solve(bc, vertices)

        return torch.tensor(vertices)

    def _downsample_with_fps(self, points: torch.Tensor, fps_indices_in: torch.Tensor, downsample_ratio: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        num_points = len(points)

        # Step 2: Generate indices for downsampling
        if downsample_ratio is None:
            downsample_ratio = self._rng.uniform(low=self._sampling_ratio_range[0], high=self._sampling_ratio_range[1])
        num_downsample = int(num_points * downsample_ratio)

        if num_points == num_downsample:
            return torch.tensor(list(range(num_points))), fps_indices_in

        downsampled_indices = torch.randperm(num_points)[:num_downsample]

        # Step 3: Remove FPS indices from downsampled indices
        fps_indices_set = set(fps_indices_in.tolist())
        downsampled_indices_set = set(downsampled_indices.tolist())

        # Remove FPS indices from downsampled indices
        filtered_indices = downsampled_indices_set - fps_indices_set

        # Convert back to tensor and sort
        filtered_indices_tensor = torch.tensor(list(filtered_indices), dtype=torch.long)
        filtered_indices_sorted = filtered_indices_tensor.sort()[0]

        # Step 4: Combine filtered downsampled indices with FPS indices
        final_indices = torch.cat([filtered_indices_sorted, fps_indices_in])

        # Compute the indices of FPS points in the final downsampled cloud
        fps_indices_out = torch.arange(len(filtered_indices_sorted), len(final_indices))

        return final_indices, fps_indices_out

    @abstractmethod
    def _recalculate_faces(self, points: torch.Tensor) -> Optional[torch.Tensor]:
        # torch.from_numpy(Delaunay(patch.points[:, :2].detach().numpy()).simplices)
        pass

    @abstractmethod
    def _deform_mesh(self, data: Data) -> Data:
        pass

    @abstractmethod
    def _resample_mesh(self, data: Data, downsample_ratio: Optional[float]) -> Data:
        pass

    def _pose_mesh(self, data: Data) -> Data:
        center = torch.mean(data.pos, dim=0)
        points = data.pos - center

        if self._pose_type == PoseType.RANDOM_ROTATION:
            rotation_matrix = utils.random_rotation_matrix()
            points = torch.matmul(points, rotation_matrix)
        elif self._pose_type == PoseType.PCA:
            points = utils.compute_canonical_pose_pca(points=points)

        data.pos = points
        return data

    def get(self, idx):
        meshes = []
        reference_data = self._generate_random_data()
        downsample_ratios = list(self._rng.uniform(low=self._sampling_ratio_range[0], high=self._sampling_ratio_range[1], size=self._sampled_patches))
        if self._add_regularized_mesh:
            downsample_ratios = [None] + downsample_ratios

        for i, downsample_ratio in enumerate(downsample_ratios):
            data = self._deform_mesh(data=reference_data.clone())
            data = self._resample_mesh(data=data, downsample_ratio=downsample_ratio)
            data = self._pose_mesh(data=data)

            data['normals'] = utils.estimate_normals(points=data.pos)
            if self._features_type == FeaturesType.RISP:
                data['x'] = utils.compute_risp_features(points=data.pos, normals=data.normals)
            elif self._features_type == FeaturesType.XYZ:
                data['x'] = data.pos.unsqueeze(dim=1)

            data['mesh_id'] = 0

            meshes.append(data)

        return [meshes]

    # def get(self, idx):
    #     with torch.enable_grad():
    #         reference_patch = self._generate_points_and_faces()
    #
    #     # reference_patch.points = utils.normalize_to_unit_sphere(points=reference_patch.points)
    #     anchors_ratio = float(self._rng.uniform(low=self._anchors_ratio_range[0], high=self._anchors_ratio_range[1], size=1))
    #     fps_indices = fps(reference_patch.points, ratio=anchors_ratio, random_start=False)
    #
    #     # permutation_indices = torch.tensor(list(range(len(fps_indices))), dtype=torch.long)
    #     # permutation_indices = permutation_indices[torch.randperm(len(permutation_indices))]
    #     # fps_indices_permuted = fps_indices[permutation_indices]
    #     # fps_indices_list = [fps_indices, fps_indices_permuted]
    #
    #     downsample_ratios = self._rng.uniform(low=self._sampling_ratio_range[0], high=self._sampling_ratio_range[1], size=self._sampled_patches)
    #     # downsample_ratios = [1.0, 1.0]
    #     if self._add_unsampled_patch:
    #         downsample_ratios = np.insert(downsample_ratios, 0, 1.0)
    #
    #     meshes = []
    #     for i, downsample_ratio in enumerate(downsample_ratios):
    #         mesh = Data()
    #         # deformed_points = self._deform(vertices=reference_patch.points, faces=reference_patch.faces).to(torch.float32)
    #         deformed_points = reference_patch.points.to(torch.float32)
    #         # indices, fps_indices_out = self._downsample_with_fps(points=deformed_points, fps_indices_in=fps_indices_list[i], downsample_ratio=downsample_ratio)
    #         indices, fps_indices_out = self._downsample_with_fps(points=deformed_points, fps_indices_in=fps_indices, downsample_ratio=downsample_ratio)
    #
    #         # for key, item in reference_patch:
    #         #     if key != 'points':
    #         #         if isinstance(item, torch.Tensor) and item.requires_grad is True:
    #         #             item = item.detach()
    #         #         mesh[key] = item[indices]
    #
    #         points = deformed_points[indices]
    #         center = torch.mean(points, dim=0)
    #         pos = points - center
    #         rotation_matrix = SyntheticSurfaceDataset._random_rotation_matrix()
    #         pos = torch.matmul(pos, rotation_matrix)
    #         normals = utils.estimate_normals(points=pos)
    #
    #         if self._features_type == FeaturesType.RISP:
    #             x = utils.compute_risp_features(points=pos, normals=normals)
    #         elif self._features_type == FeaturesType.XYZ:
    #             x = pos.unsqueeze(dim=1)
    #
    #         # edge_index = knn_graph(x=pos, k=self._conv_k_nearest, loop=False)
    #
    #         if downsample_ratio < 1.0:
    #             mesh['faces'] = self._recalculate_faces(points=pos)
    #         else:
    #             mesh['faces'] = reference_patch.faces
    #
    #         mesh['pos'] = pos
    #         # mesh['normals'] = normals
    #         mesh['anchor_indices'] = fps_indices_out.to(torch.int32)
    #         # mesh['x'] = pos.unsqueeze(dim=1)
    #         mesh['x'] = x
    #         # mesh['edge_index'] = edge_index
    #
    #         meshes.append(mesh)
    #
    #     gt_correspondences = torch.tensor(list(range(fps_indices.shape[0])))
    #     corr_data = Data(gt_correspondences=gt_correspondences)
    #     return meshes, [corr_data, Data()]


class ParametricSurfaceDataset(SyntheticSurfaceDataset):
    def __init__(
            self,
            grid_generation_method: GridGenerationMethod,
            grid_radius_range: Tuple[float, float],
            grid_offset_range: Tuple[float, float],
            points_scale_range: Tuple[float, float],
            use_autograd: bool,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._grid_generation_method = grid_generation_method
        self._grid_radius_range = grid_radius_range
        self._grid_offset_range = grid_offset_range
        self._points_scale_range = points_scale_range
        self._use_autograd = use_autograd

    def _compute_derivatives(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Compute first and second derivatives of z with respect to x and y."""
        dz_dx, dz_dy = torch.autograd.grad(outputs=z.sum(), inputs=[x, y], create_graph=True)
        d2z_dx2, d2z_dxdy = torch.autograd.grad(outputs=dz_dx.sum(), inputs=[x, y], create_graph=True)
        _, d2z_dy2 = torch.autograd.grad(outputs=dz_dy.sum(), inputs=[x, y], create_graph=True)
        return dz_dx, dz_dy, d2z_dx2, d2z_dxdy, d2z_dy2

    def _compute_shape_operator(
            self,
            dz_dx: torch.Tensor,
            dz_dy: torch.Tensor,
            d2z_dx2: torch.Tensor,
            d2z_dxdy: torch.Tensor,
            d2z_dy2: torch.Tensor
    ) -> torch.Tensor:
        """Compute the shape operator of the surface."""
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
        ], dim=-2) / det.unsqueeze(-1).unsqueeze(-2)
        return shape_operator

    def _compute_principal_curvatures(
            self,
            shape_operator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute principal curvatures and directions from the shape operator."""
        eigenvalues, eigenvectors = torch.linalg.eig(shape_operator)
        k1, k2 = eigenvalues.real[..., 0], eigenvalues.real[..., 1]
        v1, v2 = eigenvectors.real[..., 0], eigenvectors.real[..., 1]
        return k1, k2, v1, v2

    def _compute_curvatures(
            self,
            k1: torch.Tensor,
            k2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean (H) and Gaussian (K) curvatures from principal curvatures."""
        H = (k1 + k2) / 2
        K = k1 * k2
        return H, K

    def _compute_curvature_gradients(
            self,
            H: torch.Tensor,
            K: torch.Tensor,
            x: torch.Tensor,
            y: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Compute gradients of mean and Gaussian curvatures."""
        grad_H = torch.autograd.grad(outputs=H.sum(), inputs=[x, y], create_graph=True)
        grad_K = torch.autograd.grad(outputs=K.sum(), inputs=[x, y], create_graph=True)
        return grad_H, grad_K

    def _compute_jacobian(
            self,
            dz_dx: torch.Tensor,
            dz_dy: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Jacobian of the surface parameterization."""
        return torch.stack([
            torch.stack([torch.ones_like(dz_dx), torch.zeros_like(dz_dx)], dim=-1),
            torch.stack([torch.zeros_like(dz_dy), torch.ones_like(dz_dy)], dim=-1),
            torch.stack([dz_dx, dz_dy], dim=-1)
        ], dim=-2)

    def _map_to_3d(
            self,
            jacobian: torch.Tensor,
            v1: torch.Tensor,
            v2: torch.Tensor,
            grad_H: torch.Tensor,
            grad_K: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map 2D vector fields to 3D using the Jacobian."""
        v1_3d = torch.einsum('ijk,ik->ij', jacobian, v1)
        v2_3d = torch.einsum('ijk,ik->ij', jacobian, v2)
        grad_H_3d = torch.einsum('ijk,ik->ij', jacobian, grad_H)
        grad_K_3d = torch.einsum('ijk,ik->ij', jacobian, grad_K)
        return v1_3d, v2_3d, grad_H_3d, grad_K_3d

    def _compute_3d_euclidean_signatures(
            self,
            H: torch.Tensor,
            K: torch.Tensor,
            grad_H_3d: torch.Tensor,
            grad_K_3d: torch.Tensor,
            v1_3d: torch.Tensor,
            v2_3d: torch.Tensor
    ) -> torch.Tensor:
        """Compute the invariant 3D Euclidean signatures (first flavor only)."""

        # Normalize principal directions
        v1_3d_norm = v1_3d / torch.norm(v1_3d, dim=1, keepdim=True)
        v2_3d_norm = v2_3d / torch.norm(v2_3d, dim=1, keepdim=True)

        # Compute directional derivatives
        H_1 = torch.sum(grad_H_3d * v1_3d_norm, dim=1)
        H_2 = torch.sum(grad_H_3d * v2_3d_norm, dim=1)
        K_1 = torch.sum(grad_K_3d * v1_3d_norm, dim=1)
        K_2 = torch.sum(grad_K_3d * v2_3d_norm, dim=1)

        # Compute signature (first flavor only)
        signature = torch.stack([H, K, H_1, H_2, K_1, K_2], dim=1)

        return signature

    def _generate_grid_points(self, grid_range: Tuple[float, float], downsample_ratio: Optional[float], grid_generation_method: GridGenerationMethod) -> Tuple[torch.Tensor, torch.Tensor]:
        if grid_generation_method == GridGenerationMethod.MESH:
            grid_points_count_sqrt = int(np.sqrt(self._grid_points_count))
            x_linspace = torch.linspace(grid_range[0], grid_range[1], grid_points_count_sqrt)
            y_linspace = torch.linspace(grid_range[0], grid_range[1], grid_points_count_sqrt)
            x, y = torch.meshgrid(x_linspace, y_linspace, indexing='ij')
            x = x.flatten()
            y = y.flatten()
            return x, y
        elif grid_generation_method == GridGenerationMethod.UNIFORM:
            points_count = int(downsample_ratio * self._grid_points_count)
            points = self._rng.uniform(low=grid_range[0], high=grid_range[1], size=(points_count, 2))
            x = points[:, 0]
            y = points[:, 1]
            return torch.tensor(x), torch.tensor(y)
        else:
            raise ValueError(f"Invalid grid generation method: {grid_generation_method}")

    def _recalculate_faces(self, points: torch.Tensor) -> Optional[torch.Tensor]:
        return torch.from_numpy(Delaunay(points[:, :2].detach().numpy()).simplices)

    @abstractmethod
    def _evaluate_patch_points(self, x: torch.Tensor, y: torch.Tensor, data: Data) -> torch.Tensor:
        pass

    def _deform_mesh(self, data: Data) -> Data:
        return data

    def _resample_mesh(self, data: Data, downsample_ratio: Optional[float]) -> Data:
        # grid_radius = float(self._rng.uniform(low=self._grid_radius_range[0], high=self._grid_radius_range[1]))
        grid_range = (-data.grid_radius + data.grid_offset, data.grid_radius + data.grid_offset)
        if downsample_ratio is None:
            x, y = self._generate_grid_points(grid_range=grid_range, downsample_ratio=downsample_ratio, grid_generation_method=GridGenerationMethod.MESH)
        else:
            x, y = self._generate_grid_points(grid_range=grid_range, downsample_ratio=downsample_ratio, grid_generation_method=self._grid_generation_method)

        x = x.to(torch.float32)
        y = y.to(torch.float32)

        if self._use_autograd:
            x = x.requires_grad_()
            y = y.requires_grad_()

        z = self._evaluate_patch_points(x=x, y=y, data=data).to(torch.float32)

        if self._use_autograd:
            dz_dx, dz_dy, d2z_dx2, d2z_dxdy, d2z_dy2 = self._compute_derivatives(x=x, y=y, z=z)
            shape_operator = self._compute_shape_operator(dz_dx=dz_dx, dz_dy=dz_dy, d2z_dx2=d2z_dx2, d2z_dxdy=d2z_dxdy, d2z_dy2=d2z_dy2)
            k1, k2, v1_2d, v2_2d = self._compute_principal_curvatures(shape_operator=shape_operator)
            H, K = self._compute_curvatures(k1=k1, k2=k2)
            grad_H_2d, grad_K_2d = self._compute_curvature_gradients(H=H, K=K, x=x, y=y)
            grad_H_2d = torch.stack(grad_H_2d, dim=-1)
            grad_K_2d = torch.stack(grad_K_2d, dim=-1)
            jacobian = self._compute_jacobian(dz_dx=dz_dx, dz_dy=dz_dy)
            v1_3d, v2_3d, grad_H_3d, grad_K_3d = self._map_to_3d(jacobian=jacobian, v1=v1_2d, v2=v2_2d, grad_H=grad_H_2d, grad_K=grad_K_2d)
            signature = self._compute_3d_euclidean_signatures(H, K, grad_H_3d, grad_K_3d, v1_3d, v2_3d)
            data['H'] = H
            data['K'] = K
            data['grad_H_2d'] = grad_H_2d
            data['grad_K_2d'] = grad_K_2d
            data['grad_H_3d'] = grad_H_3d
            data['grad_K_3d'] = grad_K_3d
            data['k1'] = k1
            data['k2'] = k2
            data['v1_2d'] = v1_2d
            data['v2_2d'] = v2_2d
            data['v1_3d'] = v1_3d
            data['v2_3d'] = v2_3d
            data['jacobian'] = jacobian
            data['signature'] = signature

        data['pos'] = torch.stack([x, y, z], dim=1) * data.points_scale
        data['face'] = torch.from_numpy(Delaunay(data.pos[:, :2].detach().numpy()).simplices).T
        # tri_data = {'vertices': data.pos[:, :2].cpu().detach().numpy()}
        # tri = triangle.triangulate(tri_data, 'q20a')
        # data['face'] = torch.tensor(tri['triangles'])

        return data


class PolynomialSurfaceDataset(ParametricSurfaceDataset):
    def __init__(
            self,
            order_range: Tuple[int, int],
            coefficient_scale_range: Tuple[float, float],
            coeff_generation_method: CoeffGenerationMethod,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._order_range = order_range
        self._coefficient_scale_range = coefficient_scale_range
        self._coeff_generation_method = coeff_generation_method

    @staticmethod
    def _get_num_coeffs(order: int) -> int:
        """Calculate the number of coefficients for a given polynomial order."""
        return sum(1 for x in range(order + 1) for y in range(order + 1) if 0 < x + y <= order)

    def _generate_coeffs_and_order(self) -> Tuple[torch.Tensor, int, float]:
        """Generate random coefficients and order for a polynomial."""
        order = self._rng.integers(low=self._order_range[0], high=self._order_range[1] + 1)
        order = 4
        # order = torch.randint(self._order_range[0], self._order_range[1] + 1, (1,)).item()
        num_coeffs = self._get_num_coeffs(order=order)
        # coefficient_scale = torch.empty(1).uniform_(*self._coefficient_scale_range).item()
        coefficient_scale = self._rng.uniform(low=self._coefficient_scale_range[0], high=self._coefficient_scale_range[1])

        if self._coeff_generation_method == CoeffGenerationMethod.UNIFORM:
            # coeffs = 2 * (torch.rand(num_coeffs) - 0.5) * coefficient_scale
            coeffs = torch.tensor(2 * (self._rng.uniform(size=num_coeffs) - 0.5) * coefficient_scale)
            # coeffs = coeffs / order
        elif self._coeff_generation_method == CoeffGenerationMethod.NORMAL:
            # coeffs = torch.randn(num_coeffs) * coefficient_scale
            coeffs = torch.tensor(self._rng.normal(size=num_coeffs) * coefficient_scale)
        else:
            raise ValueError(f"Invalid coefficient generation method: {self._coeff_generation_method}")

        return coeffs, order, coefficient_scale

    def _evaluate_patch_points(self, x: torch.Tensor, y: torch.Tensor, data: Data) -> torch.Tensor:
        pairs = [(i, j) for i in range(data.order + 1) for j in range(data.order + 1) if 0 < i + j <= data.order]
        z = torch.zeros_like(x)
        for c, pair in zip(data.coeffs, pairs):
            z += c * (x ** pair[0]) * (y ** pair[1])
        return z

    def _generate_random_data(self) -> Data:
        coeffs, order, coefficient_scale = self._generate_coeffs_and_order()
        if len(self._grid_radius_range) == 2:
            grid_radius = float(self._rng.uniform(low=self._grid_radius_range[0], high=self._grid_radius_range[1]))
        else:
            grid_radius = self._grid_radius_range[0]

        if len(self._points_scale_range) == 2:
            points_scale = float(self._rng.uniform(low=self._points_scale_range[0], high=self._points_scale_range[1]))
        else:
            points_scale = self._points_scale_range[0]

        grid_offset = float(self._rng.uniform(low=self._grid_offset_range[0], high=self._grid_offset_range[1]))
        data = Data(coeffs=coeffs, order=order, coefficient_scale=coefficient_scale, grid_radius=grid_radius, grid_offset=grid_offset, points_scale=points_scale)
        return data
