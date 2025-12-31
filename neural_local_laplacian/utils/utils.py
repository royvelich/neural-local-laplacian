# standard library
from typing import List, Dict, Tuple

# scipy
import scipy.sparse
import scipy.sparse.linalg

# torch
import torch

# numpy
import numpy as np

# torch geometric
from torch_geometric.data import Batch


def centroid_to_origin(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0, keepdims=True)
    centered_points = points - centroid
    return centered_points


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


def normalize_mesh_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    Normalize mesh vertices to be centered at origin and fit within unit sphere.

    This function ensures consistent mesh scaling and positioning across different
    mesh files, making them suitable for comparative analysis and feature extraction.

    Args:
        vertices: Raw mesh vertices of shape (N, 3)

    Returns:
        Normalized vertices of shape (N, 3) where:
        - Center of mass is at origin (0, 0, 0)
        - All vertices fit within unit sphere (max distance = 1.0)

    Raises:
        ValueError: If vertices array is empty or has wrong shape
    """
    if vertices.size == 0:
        raise ValueError("Cannot normalize empty vertices array")

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shape (N, 3), got {vertices.shape}")

    # Center vertices at origin (center of mass at origin)
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid

    # Scale to fit within unit sphere
    # Find the maximum distance from origin to any vertex
    distances = np.linalg.norm(centered_vertices, axis=1)
    max_distance = np.max(distances)

    if max_distance > 0:
        # Scale so that the farthest vertex is on the unit sphere
        normalized_vertices = centered_vertices / max_distance
    else:
        # Handle degenerate case (all vertices at same point)
        normalized_vertices = centered_vertices

    return normalized_vertices


def assemble_sparse_laplacian_variable(weights: torch.Tensor, attention_mask: torch.Tensor,
                                       vertex_indices: torch.Tensor, center_indices: torch.Tensor,
                                       batch_indices: torch.Tensor) -> scipy.sparse.csr_matrix:
    """
    Assemble sparse Laplacian matrix from variable-sized patch weights using fully vectorized operations.

    Args:
        weights: Token weights of shape (batch_size, max_k)
        attention_mask: Mask of shape (batch_size, max_k) - True for real tokens
        vertex_indices: Neighbor vertex indices of shape (total_points,)
        center_indices: Center vertex index for each patch, shape (num_patches,)
        batch_indices: Batch indices of shape (total_points,)

    Returns:
        Sparse Laplacian matrix
    """
    # Get dimensions
    num_patches = weights.shape[0]
    max_k = weights.shape[1]
    device = weights.device

    # Get number of vertices
    num_vertices = max(vertex_indices.max().item(), center_indices.max().item()) + 1

    # Flatten weights and attention mask
    weights_flat = weights.flatten()  # (batch_size * max_k,)
    attention_mask_flat = attention_mask.flatten()  # (batch_size * max_k,)

    # Create batch indices for the flattened weights (which patch each weight belongs to)
    patch_indices_flat = torch.arange(num_patches, device=device).repeat_interleave(max_k)  # (batch_size * max_k,)

    # Filter out padded positions
    valid_mask = attention_mask_flat  # Only keep valid (non-padded) entries
    valid_weights = weights_flat[valid_mask]  # (num_valid,)
    valid_patch_indices = patch_indices_flat[valid_mask]  # (num_valid,)

    # Get center vertex for each valid weight
    valid_center_vertices = center_indices[valid_patch_indices]  # (num_valid,)

    # For variable-sized patches, we need to map from flattened valid indices back to vertex_indices
    # Create a mapping from valid positions to their corresponding vertex indices

    # Get cumulative sizes to find the start position of each patch in vertex_indices
    batch_sizes = batch_indices.bincount(minlength=num_patches)  # (num_patches,)
    cumsum_sizes = torch.cumsum(batch_sizes, dim=0)
    starts = torch.cat([torch.tensor([0], device=device), cumsum_sizes[:-1]])

    # For each valid weight, find its position within its patch
    # Count how many valid weights we've seen for each patch so far
    patch_counts = torch.zeros(num_patches, device=device, dtype=torch.long)
    positions_in_patch = torch.zeros_like(valid_patch_indices, dtype=torch.long)

    # Vectorized position calculation
    unique_patches, counts = torch.unique_consecutive(valid_patch_indices, return_counts=True)
    positions_in_patch = torch.cat([torch.arange(count, device=device) for count in counts])

    # Get the actual vertex indices for valid weights
    valid_neighbor_vertices = vertex_indices[starts[valid_patch_indices] + positions_in_patch]

    # Create all off-diagonal entries vectorized
    # Each valid weight creates two entries: center->neighbor and neighbor->center
    num_valid = len(valid_weights)

    # Center -> neighbor entries
    center_to_neighbor_rows = valid_center_vertices  # (num_valid,)
    center_to_neighbor_cols = valid_neighbor_vertices  # (num_valid,)
    center_to_neighbor_weights = -valid_weights  # (num_valid,)

    # Neighbor -> center entries (symmetric)
    neighbor_to_center_rows = valid_neighbor_vertices  # (num_valid,)
    neighbor_to_center_cols = valid_center_vertices  # (num_valid,)
    neighbor_to_center_weights = -valid_weights  # (num_valid,)

    # Combine all entries
    all_row_indices = torch.cat([center_to_neighbor_rows, neighbor_to_center_rows])  # (2 * num_valid,)
    all_col_indices = torch.cat([center_to_neighbor_cols, neighbor_to_center_cols])  # (2 * num_valid,)
    all_weights = torch.cat([center_to_neighbor_weights, neighbor_to_center_weights])  # (2 * num_valid,)

    # Convert to numpy for scipy operations
    all_row_indices_np = all_row_indices.detach().cpu().numpy()
    all_col_indices_np = all_col_indices.detach().cpu().numpy()
    all_weights_np = all_weights.detach().cpu().numpy()

    # Create sparse matrix from coordinates (off-diagonal entries only)
    laplacian_coo = scipy.sparse.coo_matrix(
        (all_weights_np, (all_row_indices_np, all_col_indices_np)),
        shape=(num_vertices, num_vertices)
    )

    # Sum duplicate entries and convert to CSR
    laplacian_csr = laplacian_coo.tocsr()
    laplacian_csr.sum_duplicates()

    # Vectorized diagonal computation: each diagonal entry = -sum of off-diagonal entries in that row
    row_sums = np.array(laplacian_csr.sum(axis=1)).flatten()
    diagonal_values = -row_sums

    # Set diagonal entries
    laplacian_csr.setdiag(diagonal_values)

    # Ensure numerical symmetry
    laplacian_csr = 0.5 * (laplacian_csr + laplacian_csr.T)

    return laplacian_csr


def assemble_stiffness_and_mass_matrices(
        stiffness_weights: torch.Tensor,
        areas: torch.Tensor,
        attention_mask: torch.Tensor,
        vertex_indices: torch.Tensor,
        center_indices: torch.Tensor,
        batch_indices: torch.Tensor
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Assemble separate stiffness and mass matrices from predicted weights and areas.

    The stiffness matrix S is symmetric and contains the edge weights.
    The mass matrix M is diagonal and contains the vertex areas.
    Together they define the generalized eigenvalue problem: S @ v = lambda * M @ v

    Args:
        stiffness_weights: Stiffness weights of shape (num_patches, max_k)
        areas: Predicted areas of shape (num_patches,)
        attention_mask: Mask of shape (num_patches, max_k) - True for real tokens
        vertex_indices: Neighbor vertex indices of shape (total_points,)
        center_indices: Center vertex index for each patch, shape (num_patches,)
        batch_indices: Batch indices of shape (total_points,)

    Returns:
        Tuple of (S, M):
        - S: Symmetric stiffness matrix (N, N)
        - M: Diagonal mass matrix (N, N)
    """
    # Get dimensions
    num_patches = stiffness_weights.shape[0]
    max_k = stiffness_weights.shape[1]
    device = stiffness_weights.device

    # Get number of vertices
    num_vertices = max(vertex_indices.max().item(), center_indices.max().item()) + 1

    # Flatten weights and attention mask
    weights_flat = stiffness_weights.flatten()  # (num_patches * max_k,)
    attention_mask_flat = attention_mask.flatten()  # (num_patches * max_k,)

    # Create patch indices for the flattened weights
    patch_indices_flat = torch.arange(num_patches, device=device).repeat_interleave(max_k)

    # Filter out padded positions
    valid_mask = attention_mask_flat
    valid_weights = weights_flat[valid_mask]
    valid_patch_indices = patch_indices_flat[valid_mask]

    # Get center vertex for each valid weight
    valid_center_vertices = center_indices[valid_patch_indices]

    # Get cumulative sizes to find the start position of each patch in vertex_indices
    batch_sizes = batch_indices.bincount(minlength=num_patches)
    cumsum_sizes = torch.cumsum(batch_sizes, dim=0)
    starts = torch.cat([torch.tensor([0], device=device), cumsum_sizes[:-1]])

    # Vectorized position calculation
    unique_patches, counts = torch.unique_consecutive(valid_patch_indices, return_counts=True)
    positions_in_patch = torch.cat([torch.arange(count, device=device) for count in counts])

    # Get the actual vertex indices for valid weights
    valid_neighbor_vertices = vertex_indices[starts[valid_patch_indices] + positions_in_patch]

    # === Build Stiffness Matrix S (symmetric) ===
    # Each edge (i, j) gets contributions from both patches centered at i and j
    # We average to ensure symmetry: S[i,j] = (s_ij + s_ji) / 2

    # Center -> neighbor entries (negative off-diagonal)
    center_to_neighbor_rows = valid_center_vertices
    center_to_neighbor_cols = valid_neighbor_vertices
    center_to_neighbor_weights = -valid_weights

    # Neighbor -> center entries (symmetric contribution)
    neighbor_to_center_rows = valid_neighbor_vertices
    neighbor_to_center_cols = valid_center_vertices
    neighbor_to_center_weights = -valid_weights

    # Combine all entries
    all_row_indices = torch.cat([center_to_neighbor_rows, neighbor_to_center_rows])
    all_col_indices = torch.cat([center_to_neighbor_cols, neighbor_to_center_cols])
    all_weights = torch.cat([center_to_neighbor_weights, neighbor_to_center_weights])

    # Convert to numpy
    all_row_indices_np = all_row_indices.detach().cpu().numpy()
    all_col_indices_np = all_col_indices.detach().cpu().numpy()
    all_weights_np = all_weights.detach().cpu().numpy()

    # Create sparse stiffness matrix (off-diagonal entries)
    stiffness_coo = scipy.sparse.coo_matrix(
        (all_weights_np, (all_row_indices_np, all_col_indices_np)),
        shape=(num_vertices, num_vertices)
    )

    # Sum duplicates and convert to CSR
    stiffness_csr = stiffness_coo.tocsr()
    stiffness_csr.sum_duplicates()

    # Average symmetric entries: S = (S + S^T) / 2
    stiffness_csr = 0.5 * (stiffness_csr + stiffness_csr.T)

    # Set diagonal: S[i,i] = -sum of off-diagonal entries in row i
    row_sums = np.array(stiffness_csr.sum(axis=1)).flatten()
    diagonal_values = -row_sums
    stiffness_csr.setdiag(diagonal_values)

    # === Build Mass Matrix M (diagonal) ===
    # M[i,i] = area of vertex i (from when vertex i was center)
    center_indices_np = center_indices.detach().cpu().numpy()
    areas_np = areas.detach().cpu().numpy()

    # Each vertex should appear exactly once as a center
    # If a vertex appears multiple times (shouldn't happen), we average
    mass_diagonal = np.zeros(num_vertices)
    mass_counts = np.zeros(num_vertices)

    for center_idx, area in zip(center_indices_np, areas_np):
        mass_diagonal[center_idx] += area
        mass_counts[center_idx] += 1

    # Average if vertex appeared multiple times (edge case)
    nonzero_mask = mass_counts > 0
    mass_diagonal[nonzero_mask] /= mass_counts[nonzero_mask]

    # Handle vertices that were never centers (shouldn't happen in complete coverage)
    # Set to small positive value to avoid singularity
    zero_mask = mass_counts == 0
    if np.any(zero_mask):
        mass_diagonal[zero_mask] = 1e-6

    mass_csr = scipy.sparse.diags(mass_diagonal, format='csr')

    return stiffness_csr, mass_csr


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


def compute_laplacian_eigendecomposition(
        laplacian_matrix: scipy.sparse.spmatrix,
        num_eigenvalues: int,
        mass_matrix: scipy.sparse.spmatrix = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigendecomposition of a Laplacian matrix.

    Uses shift-invert mode with sigma close to 0 to find the smallest eigenvalues,
    which encode the most important spectral properties.

    Args:
        laplacian_matrix: Sparse Laplacian matrix (N, N)
        num_eigenvalues: Number of eigenvalues/eigenvectors to compute
        mass_matrix: Optional mass matrix for generalized eigenvalue problem.
                     If provided, solves L @ v = lambda * M @ v

    Returns:
        Tuple of (eigenvalues, eigenvectors):
        - eigenvalues: Array of shape (num_eigenvalues,) sorted ascending
        - eigenvectors: Array of shape (N, num_eigenvalues)
    """
    # Ensure consistent dtype (float64 is more stable for eigendecomposition)
    # This prevents ARPACK convergence issues from mixed precision
    laplacian_matrix = laplacian_matrix.astype(np.float64)
    if mass_matrix is not None:
        mass_matrix = mass_matrix.astype(np.float64)

    # Use shift-invert mode with sigma close to 0 to find smallest eigenvalues
    if mass_matrix is not None:
        # Generalized eigenvalue problem: L @ v = lambda * M @ v
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            laplacian_matrix, k=num_eigenvalues, M=mass_matrix, sigma=1e-8, which='LM'
        )
    else:
        # Standard eigenvalue problem
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            laplacian_matrix, k=num_eigenvalues, sigma=-0.01, which='LM'
        )

    # Sort by eigenvalue (ascending)
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    return eigenvalues, eigenvectors