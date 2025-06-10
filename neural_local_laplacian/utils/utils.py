# standard library
import importlib
from typing import Type, Callable, List, Tuple, Optional, Union, Optional, Literal, Dict
import inspect

import scipy.sparse
# torch
import torch
import torch.nn.functional as F

# numpy
import numpy as np
from scipy import sparse

# scipy
from scipy.spatial import cKDTree

# open3d
import open3d as o3d

# python-shot
# import handcrafted_descriptor as hd

# sklearn
from sklearn.neighbors import NearestNeighbors

# igl
import igl

# delta-conv
from deltaconv.geometry import build_tangent_basis, build_grad_div, estimate_basis

# pyfm
from pyFM.mesh import TriMesh

# torch geometric
from torch_geometric.data import Batch


def estimate_normals(points: np.ndarray,
                     k_neighbors: int = 10,
                     k_orient: int = 10,
                     lambda_param: float = 0.0,
                     cos_alpha_tol: float = 1.0) -> np.ndarray:
    """
    Estimate normals for a point cloud and orient them consistently using tangent planes.

    Args:
        points: torch.Tensor of shape (K, 3) containing 3D points
        k_neighbors: Number of nearest neighbors to use for normal estimation
        k_orient: Number of neighbors for normal orientation consistency
        lambda_param: Weight parameter for orientation propagation
        cos_alpha_tol: Cosine tolerance for orientation propagation

    Returns:
        torch.Tensor of shape (K, 3) containing the estimated normals
    """
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
    )

    try:
        # Try orienting normals consistently
        pcd.orient_normals_consistent_tangent_plane(
            k_orient,
            lambda_param,
            cos_alpha_tol
        )
    except RuntimeError as e:
        print(f"Warning: Failed to orient normals consistently.")

    # Optional: ensure normals are normalized
    pcd.normalize_normals()

    return np.array(pcd.normals)


def compute_risp_features(points: np.ndarray, normals: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Compute RISP features for each point in the point cloud without loops.

    :param points: numpy array of shape (N, 3) containing point coordinates
    :param normals: numpy array of shape (N, 3) containing normal vectors for each point
    :param k: number of nearest neighbors to consider
    :return: numpy array of shape (N, 14, k) containing RISP features for each point
    """
    N = points.shape[0]
    tree: cKDTree = cKDTree(points)
    _, indices = tree.query(points, k=k + 1)  # +1 because the first neighbor is the point itself

    # Remove the first column (self-distances and self-indices)
    indices = indices[:, 1:]

    # Get neighbors
    neighbors = points[indices]  # Shape: (N, k, 3)
    neighbor_normals = normals[indices]

    # Compute relative positions
    rel_pos: np.ndarray = neighbors - points[:, np.newaxis, :]  # Shape: (N, k, 3)

    # Project neighbors onto tangent plane
    # First compute dot product of rel_pos with normals
    dots = np.sum(rel_pos * normals[:, np.newaxis, :], axis=2)  # Shape: (N, k)

    # Then subtract the normal component
    proj_neighbors: np.ndarray = rel_pos - dots[..., np.newaxis] * normals[:, np.newaxis, :]  # Shape: (N, k, 3)

    # for i in range(100):
    #     for j in range(k):
    #         x = np.cross(proj_neighbors[i, 0], proj_neighbors[i, j+1])
    #         bla1 = x / np.linalg.norm(x)
    #         bla2 = normals[i]
    #         pass

    # Compute angles in tangent plane
    # Step 1: Get vector perpendicular to normal in tangent plane as reference direction
    ref_dir_x = np.array([1.0, 0.0, 0.0])  # Can be any vector not parallel to normal
    ref_dir_x = ref_dir_x - np.sum(ref_dir_x * normals, axis=1, keepdims=True) * normals
    ref_dir_x = ref_dir_x / (np.linalg.norm(ref_dir_x, axis=1, keepdims=True) + 1e-16)

    ref_dir_y = np.cross(ref_dir_x, normals)
    ref_dir_y = ref_dir_y / (np.linalg.norm(ref_dir_y, axis=1, keepdims=True) + 1e-16)

    # Step 2: Compute angles relative to this reference direction
    x_coord = np.sum(proj_neighbors * ref_dir_x[:, np.newaxis, :], axis=2)
    y_coord = np.sum(proj_neighbors * ref_dir_y[:, np.newaxis, :], axis=2)
    angles = np.arctan2(y_coord, x_coord)

    # Sort neighbors by angle
    sort_idx = np.argsort(angles, axis=1)

    row_idx = np.arange(N)[:, np.newaxis]
    sorted_neighbors = neighbors[row_idx, sort_idx]
    sorted_neighbor_normals = neighbor_normals[row_idx, sort_idx]
    sorted_rel_pos = rel_pos[row_idx, sort_idx]

    # Compute edge vectors
    e_i = sorted_rel_pos
    e_i_minus_1 = np.roll(sorted_rel_pos, 1, axis=1)
    e_i_plus_1 = np.roll(sorted_rel_pos, -1, axis=1)
    n_i = sorted_neighbors
    n_i_minus_1 = np.roll(sorted_neighbors, 1, axis=1)
    n_i_plus_1 = np.roll(sorted_neighbors, -1, axis=1)

    # Compute RISP features
    L_0 = np.linalg.norm(e_i, axis=2)
    phi_1 = compute_angle_between_vectors(e_i_minus_1, e_i)
    phi_2 = compute_angle_between_vectors(e_i_plus_1, e_i)
    phi_3 = compute_angle_between_vectors(e_i_minus_1, n_i - n_i_minus_1)
    phi_4 = compute_angle_between_vectors(e_i_plus_1, n_i_plus_1 - n_i)
    phi_5 = compute_angle_between_vectors(np.cross(e_i_plus_1, e_i), np.cross(e_i_minus_1, e_i))

    alpha_1 = compute_angle_between_vectors(np.broadcast_to(normals[:, np.newaxis, :], e_i.shape), e_i)
    alpha_2 = compute_angle_between_vectors(np.broadcast_to(normals[:, np.newaxis, :], e_i.shape), e_i_minus_1)

    nn_i = sorted_neighbor_normals
    nn_i_minus_1 = np.roll(sorted_neighbor_normals, 1, axis=1)
    nn_i_plus_1 = np.roll(sorted_neighbor_normals, -1, axis=1)

    # w_i = np.cross(e_i, e_i_minus_1)
    # w_i = w_i / (np.linalg.norm(w_i, axis=2, keepdims=True) + 1e-10)

    beta_1 = compute_angle_between_vectors(nn_i, e_i)
    beta_2 = compute_angle_between_vectors(nn_i, n_i - n_i_minus_1)

    # n_i_minus_1 = np.cross(e_i_minus_1, e_i_minus_1 - np.roll(sorted_neighbors, 1, axis=1) + points[:, np.newaxis, :])
    # n_i_minus_1 = n_i_minus_1 / (np.linalg.norm(n_i_minus_1, axis=2, keepdims=True) + 1e-10)

    theta_1 = compute_angle_between_vectors(nn_i_minus_1, e_i_minus_1)
    theta_2 = compute_angle_between_vectors(nn_i_minus_1, n_i - n_i_minus_1)

    gamma_1 = compute_angle_between_vectors(nn_i_plus_1, n_i_plus_1 - n_i)
    gamma_2 = compute_angle_between_vectors(nn_i_plus_1, e_i_plus_1)

    risp_features = np.stack([
        L_0, phi_1, phi_2, phi_3, phi_4, phi_5,
        alpha_1, alpha_2, beta_1, beta_2,
        theta_1, theta_2, gamma_1, gamma_2], axis=-1)

    risp_features_max = np.max(risp_features, axis=1)
    # risp_features_min = np.min(risp_features, axis=1)
    # risp_features_mean = np.mean(risp_features, axis=1)
    # risp_features_concat = np.concat([risp_features_max, risp_features_min, risp_features_mean], axis=1)
    return risp_features_max


def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute the angle between two sets of vectors."""
    v1_n = v1 / (np.linalg.norm(v1, axis=2, keepdims=True) + 1e-10)
    v2_n = v2 / (np.linalg.norm(v2, axis=2, keepdims=True) + 1e-10)
    return np.arccos(np.clip(np.sum(v1_n * v2_n, axis=2), -1.0, 1.0))


def import_object(full_type_name: str) -> Type:
    module_name, class_name = full_type_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_input_args(forward_method: Callable) -> List[str]:
    signature = inspect.signature(forward_method)
    return [
        param.name for param in signature.parameters.values()
        if param.default == param.empty and param.name != 'self'
    ]


def create_layers(channels: List[int], conv_class: Type, use_batch_norm: bool, backward: bool, concat_residual: bool) -> Tuple[torch.nn.Module, torch.nn.Module]:
    layers = torch.nn.ModuleList()
    batch_norms = torch.nn.ModuleList()

    if backward:
        channels = list(reversed(channels))

    for in_channels, out_channels in zip(channels, channels[1:]):
        in_channels = 2 * in_channels if concat_residual else in_channels
        layer = conv_class(in_channels=in_channels, out_channels=out_channels)
        layers.append(layer)
        if use_batch_norm:
            # batch_norms.append(torch.nn.LayerNorm(out_channels))
            batch_norms.append(torch.nn.BatchNorm1d(num_features=out_channels))

    return layers, batch_norms


def centroid_to_origin(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0, keepdims=True)
    centered_points = points - centroid
    return centered_points


def normalize_to_unit_cube(points: np.ndarray) -> np.ndarray:
    points = centroid_to_origin(points=points)
    p_max = points.max(axis=0)
    p_min = points.min(axis=0)
    center = (p_max + p_min) / 2
    scale = (p_max - p_min).max()
    return (points - center) / scale


def normalize_to_unit_sphere(points: np.ndarray) -> np.ndarray:
    """
    Rescales a point cloud to fit within a unit sphere centered at the origin.

    Args:
        points (np.ndarray): Point cloud array of shape (K, 3) where K is the number of points

    Returns:
        np.ndarray: Normalized point cloud of shape (K, 3) fitting within a unit sphere

    Raises:
        ValueError: If input array doesn't have shape (K, 3)
    """
    points = centroid_to_origin(points=points)

    # Find the maximum distance from the origin to any point
    distances = np.linalg.norm(points, axis=1)
    max_distance = np.max(distances)

    # Scale the points to fit within a unit sphere
    normalized_points = points / max_distance

    return normalized_points


def normalize_mesh_to_unit_area(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = centroid_to_origin(points=vertices)

    # Calculate the current surface area
    current_area = igl.doublearea(vertices, faces).sum() / 2.0

    # Calculate scaling factor
    scale_factor = 1.0 / np.sqrt(current_area)

    # Scale the vertices
    normalized_vertices = vertices * scale_factor

    # Verify the new area
    # new_area = igl.doublearea(normalized_vertices, faces).sum() / 2.0
    # print(f"Original area: {current_area}")
    # print(f"Normalized area: {new_area}")

    return normalized_vertices


def random_rotation_matrix() -> torch.Tensor:
    """
    Generate a random 3D rotation matrix.

    Returns:
    torch.Tensor: A 3x3 orthonormal rotation matrix.
    """
    # Generate a random 3x3 matrix
    random_matrix: torch.Tensor = torch.randn(3, 3)

    # Perform QR decomposition
    q, r = torch.linalg.qr(random_matrix)

    # Ensure proper rotation matrix (determinant = 1)
    d: torch.Tensor = torch.diag(torch.sign(torch.diag(r)))
    rotation_matrix: torch.Tensor = torch.mm(q, d)

    # Ensure right-handed coordinate system
    if torch.det(rotation_matrix) < 0:
        rotation_matrix[:, 0] *= -1

    return rotation_matrix


def compute_canonical_pose_pca(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the canonical pose of a 3D point cloud using PCA.

    This function centers the point cloud at the origin and aligns its principal axes
    with the coordinate axes.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) containing N 3D points.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - points_canonical: Points in canonical pose, shape (N, 3)
            - R: Rotation matrix, shape (3, 3)
            - t: Translation vector, shape (3,)
    """
    # Center the points
    center: torch.Tensor = torch.mean(points, dim=0)
    centered_points: torch.Tensor = points - center

    # Compute covariance matrix
    cov: torch.Tensor = torch.mm(centered_points.t(), centered_points) / (points.shape[0] - 1)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    R: torch.Tensor = eigenvectors[:, sorted_indices]

    # Ensure right-handed coordinate system
    if torch.det(R) < 0:
        R[:, 2] *= -1

    # Transform points to canonical pose
    points_canonical: torch.Tensor = torch.mm(centered_points, R)

    return points_canonical, R, center


def faces_to_edges(faces: torch.Tensor) -> torch.Tensor:
    """
    Convert triangle faces to edge indices

    Args:
        faces: torch.LongTensor of shape [N, 3] containing triangular faces

    Returns:
        edge_index: torch.LongTensor of shape [2, E] containing unique edges
    """
    # Get all edges from faces (including duplicates)
    # For each triangle, get its 3 edges
    edges: torch.Tensor = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)

    # Sort edges to ensure (v1, v2) and (v2, v1) are treated as the same edge
    edges = torch.sort(edges, dim=1)[0]

    # Remove duplicate edges
    edges = torch.unique(edges, dim=0)

    # Convert to PyG edge_index format (2, E)
    edge_index: torch.Tensor = edges.t().contiguous()

    return edge_index


def farthest_point_sampling(vertices: np.ndarray, num_samples: int, random_start: bool = True) -> np.ndarray:
    """Perform farthest point sampling using Open3D.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions
        num_samples (int): Number of points to sample
        random_start (bool): Whether to use random initialization for FPS

    Returns:
        np.ndarray: Indices of sampled vertices
    """
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # Set start_index based on random_start parameter
    start_index = np.random.randint(len(vertices)) if random_start else 0

    # Perform FPS using Open3D with specified start index
    sampled_pcd = pcd.farthest_point_down_sample(num_samples, start_index=start_index)
    sampled_points = np.asarray(sampled_pcd.points)

    # Find the indices of these points in the original vertices array
    tree = cKDTree(vertices)
    _, indices = tree.query(sampled_points)

    return indices.astype(np.int32)


def split_results_by_nodes(results: torch.Tensor, batch: Batch) -> List[torch.Tensor]:
    return [results[batch.batch == i] for i in range(batch.num_graphs)]


def split_results_by_graphs(results: torch.Tensor, batch: Batch) -> List[torch.Tensor]:
    return [results[i] for i in range(batch.num_graphs)]


def rebuild_batch_from_list(batch: Batch, property_name: str, property_tensor_list: List[torch.Tensor]) -> Batch:
    data_list = batch.to_data_list()
    for data, tensor in zip(data_list, property_tensor_list):
        data[property_name] = tensor
    return Batch.from_data_list(data_list)


def rebuild_batch_from_tensor(batch: Batch, property_name: str, property_tensor: torch.Tensor) -> Batch:
    tensor_list = property_tensor.split(split_size=batch.batch.bincount().tolist(), dim=0)
    return rebuild_batch_from_list(batch=batch, property_name=property_name, property_tensor_list=tensor_list)


def rebuild_batch_from_dictionary_of_lists(batch: Batch, property_dict: Dict[str, List[torch.Tensor]]) -> Batch:
    for property_name, property_tensor_list in property_dict.items():
        batch = rebuild_batch_from_list(batch=batch, property_name=property_name, property_tensor_list=property_tensor_list)
    return batch
