import importlib
from typing import Type, Callable, List, Tuple
import inspect
import torch
from torch_scatter import scatter_max, scatter_add
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
import torch
from typing import List, Tuple, Union, Sequence
from torch import Tensor


def estimate_normals(points: torch.Tensor,
                     k_neighbors: int = 10,
                     k_orient: int = 10,
                     lambda_param: float = 0.0,
                     cos_alpha_tol: float = 1.0) -> torch.Tensor:
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
    # Convert torch tensor to numpy array if needed
    points_np = points.detach().cpu().numpy() if torch.is_tensor(points) else points

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
    )

    # Orient normals consistently using tangent planes
    pcd.orient_normals_consistent_tangent_plane(
        k_orient,
        lambda_param,
        cos_alpha_tol
    )

    # Optional: ensure normals are normalized
    pcd.normalize_normals()

    # Convert back to torch tensor
    normals = torch.from_numpy(np.asarray(pcd.normals))

    # Move tensor to same device and type as input
    if torch.is_tensor(points):
        normals = normals.to(points.device).to(points.dtype)

    return normals


def compute_lra(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index

    # Compute relative positions
    rel_pos = pos[row] - pos[col]

    # Compute distances
    distances = torch.norm(rel_pos, dim=-1, keepdim=True)

    # Compute weights as described in RIConv++
    max_distances, _ = scatter_max(distances, col, dim=0)
    diff_distances = max_distances[col] - distances
    diff_sum_distances = scatter_add(diff_distances, col, dim=0)
    diff_sum_distances = diff_sum_distances[col]
    weights = diff_distances / diff_sum_distances

    # Compute weighted covariance matrices for each point
    outer_products = weights.unsqueeze(-1) * rel_pos.unsqueeze(-1) * rel_pos.unsqueeze(-2)
    cov = scatter_add(outer_products, col, dim=0)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # The LRA is the eigenvector corresponding to the smallest eigenvalue
    lra = eigvecs[..., 0]

    return lra


def compute_risp_features(points: torch.Tensor, normals: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    Compute RISP features for each point in the point cloud without loops.

    :param points: numpy array of shape (N, 3) containing point coordinates
    :param normals: numpy array of shape (N, 3) containing normal vectors for each point
    :param k: number of nearest neighbors to consider
    :return: numpy array of shape (N, 14, k) containing RISP features for each point
    """
    points = points.detach().cpu().numpy()
    normals = normals.detach().cpu().numpy()
    N: int = points.shape[0]
    tree: cKDTree = cKDTree(points)
    distances: np.ndarray
    indices: np.ndarray
    distances, indices = tree.query(points, k=k + 1)  # +1 because the first neighbor is the point itself

    # Remove the first column (self-distances and self-indices)
    distances = distances[:, 1:]
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

    return torch.tensor(risp_features)


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
        in_channels = 2*in_channels if concat_residual else in_channels
        layer = conv_class(in_channels=in_channels, out_channels=out_channels)
        layers.append(layer)
        if use_batch_norm:
            batch_norms.append(torch.nn.LayerNorm(out_channels))
            # batch_norms.append(torch.nn.BatchNorm1d(num_features=out_channels))

    return layers, batch_norms


def sample_gumbel(shape, eps=1e-20):
    """
    Samples arbitrary-shaped standard gumbel variables.
    Args:
        shape (list): list of integers.
        eps (float, optional): epsilon for numerical stability. Default 1e-20.
    Returns:
        (torch.Tensor): a sample of standard Gumbel random variables
    """
    # Sample Gumble from uniform distribution
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def sinkhorn(log_alpha, n_iter=20, slack=False):
    """
    Perform incomplete Sinkhorn normalization to log_alpha
    By a theorem by Sinkhorn and Knopp, a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the successive row and column
    normalization.
    -To ensure positivity, the effective input to sinkhorn has to be
    exp(log_alpha) (element wise).
    -However, for stability, sinkhorn works in the log-space. It is only at
    return time that entries are exponentiated.
    Args:
        log_alpha (torch.Tensor): a batch of 2D tensor of shape [B, V, V]
        n_iter (int, optional): number of iterations. (Default 20)
        slack (bool, optional): augment matrix with slack row and column. Default False.
    Returns:
        (torch.Tensor): a tensor of close-to-doubly-stochastic matrices.
    """
    if not slack:
        for _ in range(n_iter):
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=-2, keepdim=True))
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=-1, keepdim=True))
    else:
        # augment log_alpha
        log_alpha_padded = F.pad(log_alpha.unsqueeze(dim=1), pad=(0, 1, 0, 1), mode='constant', value=0.0).squeeze(dim=1)
        for _ in range(n_iter):
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - torch.logsumexp(log_alpha_padded[:, :, :-1], dim=-2, keepdim=True),
                log_alpha_padded[:, :, [-1]]
            ), dim=-1)

            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - torch.logsumexp(log_alpha_padded[:, :-1, :], dim=-1, keepdim=True),
                log_alpha_padded[:, [-1], :]
            ), dim=-2)
        log_alpha = log_alpha_padded[:, :-1, :-1]

    return torch.exp(log_alpha)


def gumbel_sinkhorn(log_alpha, temp=1.0, noise_factor=0, n_iter=10, slack=False):
    """
    Random doubly-stochastic matrices via gumbel noise.
    In the zero-temperature limit sinkhorn (log_alpha/temp) approaches
    a permutation matrix. Therefore, for low temperatures this method
    can be seen as an approximate sampling of permutation matrices.
    The deterministic case (noise_factor=0) is also interesting: it can be
    shown that lim t->0 sinkhorn(log_alpha/t) = M, where M is a
    permutation matrix, the solution of the
    matching problem M=arg max_M sum_i,j log_alpha_i,j M_i,j.
    Therefore, the deterministic limit case of gumbel_sinkhorn can be seen
    as approximate solving of a matching problem.
    Args:
        log_alpha (torch.Tensor): a single/batch of 2D tensor of shape [V, V] or [B, V, V]
        temp (float, optional): temperature parameter. Default 1.0.
        noise_factor (float, optional) scaling factor for the gumbel samples
        (and the absence of randomness, with noise_factor=0). Default 0.
        n_iter (int, optional): number of sinkhorn iterations. Default 20.
        slack (bool, optional): whether augment matrix with slack row and column. Default False
    Return:
        sink (torch.Tensor): a 3D tensor of close-doubly-stochastic-matrix [B, n_samples, V, V]
    """

    if noise_factor == 0:
        noise = 0.0
    else:
        noise = noise_factor * sample_gumbel(log_alpha.shape)
        noise = noise.to(device=log_alpha.device, dtype=log_alpha.dtype)

    log_alpha_w_noise = log_alpha + noise
    log_alpha_w_noise = log_alpha_w_noise / temp

    sink = sinkhorn(log_alpha_w_noise, n_iter=n_iter, slack=slack)

    return sink


def normalize_to_unit_sphere(points: torch.Tensor) -> torch.Tensor:
    """
    Rescales a point cloud to fit within a unit sphere centered at the origin.

    Args:
        points (torch.Tensor): Point cloud tensor of shape (K, 3) where K is the number of points

    Returns:
        torch.Tensor: Normalized point cloud of shape (K, 3) fitting within a unit sphere

    Raises:
        ValueError: If input tensor doesn't have shape (K, 3)
    """
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected tensor of shape (K, 3), got {tuple(points.shape)}")

    # Compute the centroid
    centroid: torch.Tensor = torch.mean(points, dim=0, keepdim=True)

    # Center the points by subtracting the centroid
    centered_points: torch.Tensor = points - centroid

    # Find the maximum distance from the origin to any point
    distances: torch.Tensor = torch.norm(centered_points, dim=1)
    max_distance: torch.Tensor = torch.max(distances)

    # Scale the points to fit within a unit sphere
    normalized_points: torch.Tensor = centered_points / max_distance

    return normalized_points


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


def compute_canonical_pose_pca(points: torch.Tensor) -> torch.Tensor:
    """
    Compute the canonical pose of a 3D point cloud.

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

    return points_canonical
