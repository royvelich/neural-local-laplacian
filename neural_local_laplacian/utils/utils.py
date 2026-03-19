# standard library
from typing import List, Dict, Tuple, Optional

# scipy
import scipy.sparse
import scipy.sparse.linalg

# torch
import torch

# numpy
import numpy as np

# torch geometric
from torch_geometric.data import Batch, Data


# kNN backend selection
# GPU: 'pyg' (torch_geometric knn_graph), 'brute' (torch.cdist + topk)
# CPU: 'cKDTree' (scipy, exact), 'pynndescent' (approximate)
_KNN_BACKEND = 'cKDTree'

def set_knn_backend(backend: str):
    """Set the kNN backend for build_patches_from_vertices.

    Args:
        backend: 'pyg' for torch_geometric knn_graph (GPU, default),
                 'brute' for torch.cdist + topk (GPU, k-independent cost),
                 'cKDTree' for scipy.spatial.cKDTree (CPU, exact, fast),
                 'pynndescent' for PyNNDescent (CPU, approximate).
    """
    global _KNN_BACKEND
    valid = ('pyg', 'brute', 'cKDTree', 'pynndescent')
    if backend not in valid:
        raise ValueError(f"Unknown kNN backend: {backend}. Use one of {valid}.")
    _KNN_BACKEND = backend
    print(f"[kNN] Backend set to: {_KNN_BACKEND}")


def _knn_cpu(vertices_np: np.ndarray, k: int) -> np.ndarray:
    """CPU kNN dispatch — returns (N, k) neighbor index matrix (no self)."""

    if _KNN_BACKEND == 'cKDTree':
        from scipy.spatial import cKDTree
        tree = cKDTree(vertices_np)
        _, indices = tree.query(vertices_np, k=k + 1, workers=-1)  # (N, k+1)
        # First column is self (distance=0), drop it
        nbr_idx = indices[:, 1:]  # (N, k)

        if not getattr(_knn_cpu, '_logged', False):
            print("  [kNN] Using scipy.spatial.cKDTree on CPU")
            _knn_cpu._logged = True
        return nbr_idx

    elif _KNN_BACKEND == 'pynndescent':
        try:
            from pynndescent import NNDescent
            # n_neighbors includes self for pynndescent
            index = NNDescent(vertices_np, n_neighbors=k + 1, metric='euclidean',
                              n_jobs=-1, random_state=42)
            indices, _ = index.neighbor_graph  # (N, k+1)
            # First column is self, drop it
            nbr_idx = indices[:, 1:]  # (N, k)

            if not getattr(_knn_cpu, '_logged_pynn', False):
                print("  [kNN] Using PyNNDescent on CPU (approximate)")
                _knn_cpu._logged_pynn = True
            return nbr_idx
        except ImportError:
            print("  [kNN] pynndescent not installed, falling back to cKDTree")
            from scipy.spatial import cKDTree
            tree = cKDTree(vertices_np)
            _, indices = tree.query(vertices_np, k=k + 1, workers=-1)
            return indices[:, 1:]

    else:
        raise ValueError(f"Unknown kNN backend: {_KNN_BACKEND}. Use 'cKDTree' or 'pynndescent'.")


def build_patches_from_vertices(
    vertices: torch.Tensor,
    k: int,
    device: Optional[torch.device] = None
) -> Data:
    """
    Build k-NN patches from mesh vertices, ready for model inference.

    GPU backends:
    - 'pyg': torch_geometric knn_graph — spatial hashing, scales to large meshes
    - 'brute': torch.cdist + topk — fast for N < ~20k, cost independent of k

    CPU backends:
    - 'cKDTree': scipy.spatial.cKDTree — exact, fast C implementation
    - 'pynndescent': PyNNDescent — approximate, very fast for large N

    Args:
        vertices: Vertex positions as torch tensor (N, 3), any device
        k: Number of nearest neighbors per vertex
        device: Target device. If CUDA and GPU backend selected, uses GPU kNN.

    Returns:
        Data object (on target device) with:
        - pos: (N*k, 3) center-subtracted neighbor positions
        - x: (N*k, 3) same as pos (shared, not cloned)
        - patch_idx: (N*k,) patch assignment indices
        - vertex_indices: (N*k,) global vertex indices of neighbors
        - center_indices: (N,) center vertex indices (0..N-1)
        - neighbor_index_matrix: (N, k) neighbor indices per vertex
    """
    from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData

    if not torch.is_tensor(vertices):
        vertices = torch.from_numpy(np.ascontiguousarray(vertices)).float()

    num_vertices = len(vertices)
    k = int(k)
    use_gpu = device is not None and device.type == 'cuda'
    is_gpu_backend = _KNN_BACKEND in ('pyg', 'brute')

    # =========================================================================
    # GPU path: brute-force (cdist + topk) or torch_geometric knn_graph
    # =========================================================================
    if use_gpu and is_gpu_backend:
        try:
            pos_gpu = vertices.float().to(device)

            if _KNN_BACKEND == 'brute':
                dists = torch.cdist(pos_gpu, pos_gpu)  # (N, N)
                _, topk_indices = dists.topk(k + 1, dim=1, largest=False)  # (N, k+1)
                neighbor_matrix = topk_indices[:, 1:]  # (N, k) — skip self

                if not getattr(build_patches_from_vertices, '_logged_gpu_knn', False):
                    print("  [kNN] Using brute-force (cdist + topk) on GPU")
                    build_patches_from_vertices._logged_gpu_knn = True
            else:
                from torch_geometric.nn.pool import knn_graph as pyg_knn_graph
                edge_index = pyg_knn_graph(pos_gpu, k=k, loop=False, flow='source_to_target')
                neighbor_matrix = edge_index[0].reshape(num_vertices, k)  # (N, k)

                if not getattr(build_patches_from_vertices, '_logged_gpu_knn', False):
                    print("  [kNN] Using torch_geometric knn_graph on GPU")
                    build_patches_from_vertices._logged_gpu_knn = True

            # Build patch positions on GPU (no numpy)
            neighbor_positions = pos_gpu[neighbor_matrix]           # (N, k, 3)
            center_expanded = pos_gpu.unsqueeze(1)                  # (N, 1, 3)
            patch_positions = neighbor_positions - center_expanded  # (N, k, 3)

            # Flatten for PyG — all on GPU
            pos_flat = patch_positions.reshape(-1, 3)                          # (N*k, 3)
            vertex_indices = neighbor_matrix.reshape(-1)                       # (N*k,)
            center_indices = torch.arange(num_vertices, device=device)         # (N,)
            patch_idx = torch.arange(num_vertices, device=device).repeat_interleave(k)  # (N*k,)

            data = MeshPatchData(
                pos=pos_flat,
                x=pos_flat,
                patch_idx=patch_idx,
                vertex_indices=vertex_indices,
                center_indices=center_indices
            )
            data.neighbor_index_matrix = neighbor_matrix
            return data

        except (ImportError, RuntimeError) as e:
            if not getattr(build_patches_from_vertices, '_logged_gpu_knn_fail', False):
                print(f"  [kNN] GPU kNN failed ({e}), falling back to CPU")
                build_patches_from_vertices._logged_gpu_knn_fail = True

    # =========================================================================
    # CPU path: cKDTree, pynndescent, or sklearn fallback
    # =========================================================================
    vertices_np = vertices.cpu().numpy() if vertices.is_cuda else vertices.numpy()
    nbr_idx_np = _knn_cpu(vertices_np, k)  # (N, k)

    # Build patch positions
    all_neighbor_positions = vertices_np[nbr_idx_np]
    center_expanded = vertices_np[:, np.newaxis, :]
    patch_positions = all_neighbor_positions - center_expanded

    # Flatten and convert to tensors
    target_device = device if device is not None else torch.device('cpu')
    pos_tensor = torch.from_numpy(patch_positions.reshape(-1, 3)).float().to(target_device)
    patch_idx_tensor = torch.arange(num_vertices).repeat_interleave(k).to(target_device)
    vertex_indices_tensor = torch.from_numpy(nbr_idx_np.flatten()).long().to(target_device)
    center_indices_tensor = torch.arange(num_vertices).long().to(target_device)

    data = MeshPatchData(
        pos=pos_tensor,
        x=pos_tensor,
        patch_idx=patch_idx_tensor,
        vertex_indices=vertex_indices_tensor,
        center_indices=center_indices_tensor
    )
    data.neighbor_index_matrix = torch.from_numpy(nbr_idx_np.copy()).long().to(target_device)
    return data


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

    GPU-OPTIMIZED VERSION: All computation stays on GPU until final scipy conversion.

    Args:
        weights: Token weights of shape (batch_size, max_k)
        attention_mask: Mask of shape (batch_size, max_k) - True for real tokens
        vertex_indices: Neighbor vertex indices of shape (total_points,)
        center_indices: Center vertex index for each patch, shape (num_patches,)
        batch_indices: Batch indices of shape (total_points,)

    Returns:
        Sparse Laplacian matrix
    """
    device = weights.device
    num_patches = weights.shape[0]
    max_k = weights.shape[1]

    # Get number of vertices
    num_vertices = max(vertex_indices.max().item(), center_indices.max().item()) + 1

    # === STEP 1: Filter valid weights (GPU) ===
    weights_flat = weights.flatten()  # (batch_size * max_k,)
    attention_mask_flat = attention_mask.flatten()

    # Create batch indices for the flattened weights
    patch_indices_flat = torch.arange(num_patches, device=device).repeat_interleave(max_k)

    # Filter to valid (non-padded) entries
    valid_mask = attention_mask_flat
    valid_weights = weights_flat[valid_mask]  # (num_valid,)
    valid_patch_indices = patch_indices_flat[valid_mask]  # (num_valid,)

    # Get center vertex for each valid weight
    valid_center_vertices = center_indices[valid_patch_indices]

    # === STEP 2: Compute positions within each patch (GPU, fully vectorized) ===
    num_valid = len(valid_patch_indices)

    if num_valid > 0:
        # Find where patch index changes (boundaries between patches)
        patch_changes = torch.ones(num_valid, dtype=torch.bool, device=device)
        if num_valid > 1:
            patch_changes[1:] = valid_patch_indices[1:] != valid_patch_indices[:-1]

        # Get start index of each group using the change positions
        change_indices = torch.where(patch_changes)[0]

        # Group IDs (0, 0, 0, 1, 1, 2, 2, 2, ...)
        group_ids = torch.cumsum(patch_changes.long(), dim=0) - 1

        # For each element, subtract its group's start index to get position within group
        group_starts = change_indices[group_ids]
        positions_in_patch = torch.arange(num_valid, device=device, dtype=torch.long) - group_starts
    else:
        positions_in_patch = torch.tensor([], device=device, dtype=torch.long)

    # === STEP 3: Get neighbor vertex indices (GPU) ===
    batch_sizes = batch_indices.bincount(minlength=num_patches)
    cumsum_sizes = torch.cumsum(batch_sizes, dim=0)
    starts = torch.cat([torch.tensor([0], device=device, dtype=torch.long), cumsum_sizes[:-1]])

    valid_neighbor_vertices = vertex_indices[starts[valid_patch_indices] + positions_in_patch]

    # === STEP 4: Build off-diagonal entries (GPU) ===
    all_row_indices = torch.cat([valid_center_vertices, valid_neighbor_vertices])
    all_col_indices = torch.cat([valid_neighbor_vertices, valid_center_vertices])
    all_weights = torch.cat([-valid_weights, -valid_weights])

    # === STEP 5: Compute diagonal using GPU scatter_add ===
    row_sums = torch.zeros(num_vertices, device=device, dtype=all_weights.dtype)
    row_sums.scatter_add_(0, all_row_indices, all_weights)
    diagonal_values = -row_sums

    # Add diagonal entries
    diag_indices = torch.arange(num_vertices, device=device, dtype=torch.long)
    all_row_indices = torch.cat([all_row_indices, diag_indices])
    all_col_indices = torch.cat([all_col_indices, diag_indices])
    all_weights = torch.cat([all_weights, diagonal_values])

    # === STEP 6: Single GPU->CPU transfer and scipy sparse matrix creation ===
    row_np = all_row_indices.cpu().numpy()
    col_np = all_col_indices.cpu().numpy()
    data_np = all_weights.cpu().numpy()

    laplacian_coo = scipy.sparse.coo_matrix(
        (data_np, (row_np, col_np)),
        shape=(num_vertices, num_vertices)
    )
    laplacian_csr = laplacian_coo.tocsr()
    laplacian_csr.sum_duplicates()

    # Symmetrize
    laplacian_csr = 0.5 * (laplacian_csr + laplacian_csr.T)

    return laplacian_csr


def assemble_stiffness_and_mass_matrices(
        stiffness_weights: torch.Tensor,
        areas: torch.Tensor,
        attention_mask: torch.Tensor,
        vertex_indices: torch.Tensor,
        center_indices: torch.Tensor,
        batch_indices: torch.Tensor,
        top_k: Optional[int] = None,
        symmetry_policy: str = "union"
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Assemble separate stiffness and mass matrices from predicted weights and areas.

    GPU-OPTIMIZED VERSION: All computation stays on GPU until final scipy conversion.

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
        top_k: If set, keep only the top_k highest weights per patch (must be < max_k).
            This decouples the receptive field (max_k neighbors seen by the model)
            from the operator support (top_k edges in the assembled Laplacian).
        symmetry_policy: How to symmetrize after top-k pruning. Only matters when top_k is set.
            - "union": Keep edge if either direction was selected (default, most edges).
            - "intersection": Keep edge only if both directions were selected (fewest edges).

    Returns:
        Tuple of (S, M):
        - S: Symmetric stiffness matrix (N, N) as scipy.sparse.csr_matrix
        - M: Diagonal mass matrix (N, N) as scipy.sparse.csr_matrix
    """
    device = stiffness_weights.device
    num_patches = stiffness_weights.shape[0]
    max_k = stiffness_weights.shape[1]

    # Get number of vertices
    num_vertices = max(vertex_indices.max().item(), center_indices.max().item()) + 1

    # === STEP 0 (optional): Top-k pruning per patch ===
    # Keep only the top_k highest weights per patch, masking out the rest.
    if top_k is not None and top_k < max_k:
        # For each patch, find the top_k highest weights among valid entries
        # Set masked (invalid) weights to -inf so they're never selected
        weights_for_topk = stiffness_weights.clone()
        weights_for_topk[~attention_mask] = float('-inf')

        # Get indices of top_k highest weights per patch
        _, topk_indices = torch.topk(weights_for_topk, k=min(top_k, max_k), dim=1)

        # Build new attention mask: only top_k entries are valid
        topk_mask = torch.zeros_like(attention_mask)
        topk_mask.scatter_(1, topk_indices, True)

        # AND with original mask (in case top_k > actual valid count for some patches)
        attention_mask = attention_mask & topk_mask

    # === STEP 1: Filter valid weights (GPU) ===
    weights_flat = stiffness_weights.flatten()  # (num_patches * max_k,)
    attention_mask_flat = attention_mask.flatten()

    # Create patch indices for flattened weights
    patch_indices_flat = torch.arange(num_patches, device=device).repeat_interleave(max_k)

    # Filter to valid (non-padded) entries
    valid_mask = attention_mask_flat
    valid_weights = weights_flat[valid_mask]  # (num_valid,)
    valid_patch_indices = patch_indices_flat[valid_mask]  # (num_valid,)

    # === STEP 2: Compute positions within each patch (GPU, fully vectorized) ===
    # This replaces the slow: torch.cat([torch.arange(count) for count in counts])
    num_valid = len(valid_patch_indices)

    if num_valid > 0:
        # Find where patch index changes (boundaries between patches)
        patch_changes = torch.ones(num_valid, dtype=torch.bool, device=device)
        if num_valid > 1:
            patch_changes[1:] = valid_patch_indices[1:] != valid_patch_indices[:-1]

        # Cumsum of changes gives group IDs (0, 0, 0, 1, 1, 2, 2, 2, ...)
        group_ids = torch.cumsum(patch_changes.long(), dim=0) - 1

        # Get start index of each group using the change positions
        change_indices = torch.where(patch_changes)[0]  # Indices where new groups start

        # For each element, subtract its group's start index to get position within group
        group_starts = change_indices[group_ids]  # Start index for each element's group
        positions_in_patch = torch.arange(num_valid, device=device, dtype=torch.long) - group_starts
    else:
        positions_in_patch = torch.tensor([], device=device, dtype=torch.long)

    # === STEP 3: Get neighbor vertex indices (GPU) ===
    batch_sizes = batch_indices.bincount(minlength=num_patches)
    cumsum_sizes = torch.cumsum(batch_sizes, dim=0)
    starts = torch.cat([torch.tensor([0], device=device, dtype=torch.long), cumsum_sizes[:-1]])

    # Get center and neighbor vertices
    valid_center_vertices = center_indices[valid_patch_indices]
    valid_neighbor_vertices = vertex_indices[starts[valid_patch_indices] + positions_in_patch]

    # === STEP 4: Build stiffness matrix entries (GPU) ===
    # Symmetric entries: both (center, neighbor) and (neighbor, center)
    all_row_indices = torch.cat([valid_center_vertices, valid_neighbor_vertices])
    all_col_indices = torch.cat([valid_neighbor_vertices, valid_center_vertices])
    all_weights = torch.cat([-valid_weights, -valid_weights])  # Negative for off-diagonal

    # === STEP 5: Compute diagonal using GPU scatter_add ===
    # Sum weights going OUT of each vertex (before symmetrization)
    row_sums = torch.zeros(num_vertices, device=device, dtype=all_weights.dtype)
    row_sums.scatter_add_(0, all_row_indices, all_weights)

    # Diagonal values = -row_sums (so each row sums to 0)
    diagonal_values = -row_sums

    # Add diagonal entries to the triplets
    diag_indices = torch.arange(num_vertices, device=device, dtype=torch.long)
    all_row_indices = torch.cat([all_row_indices, diag_indices])
    all_col_indices = torch.cat([all_col_indices, diag_indices])
    all_weights = torch.cat([all_weights, diagonal_values])

    # === STEP 6: Single GPU->CPU transfer and scipy sparse matrix creation ===
    row_np = all_row_indices.cpu().numpy()
    col_np = all_col_indices.cpu().numpy()
    data_np = all_weights.cpu().numpy()

    # Create COO matrix and convert to CSR
    stiffness_coo = scipy.sparse.coo_matrix(
        (data_np, (row_np, col_np)),
        shape=(num_vertices, num_vertices)
    )
    stiffness_csr = stiffness_coo.tocsr()
    stiffness_csr.sum_duplicates()

    # Symmetrize based on policy
    if top_k is not None and symmetry_policy == "intersection":
        # Intersection: only keep edges that appear in BOTH directions
        # An edge (i,j) is kept only if both patch-i selected j AND patch-j selected i
        S = stiffness_csr
        S_T = S.T.tocsr()

        # Non-zero pattern of S and S^T (ignoring diagonal)
        S_pattern = S.copy()
        S_pattern.data = np.ones_like(S_pattern.data)
        S_T_pattern = S_T.copy()
        S_T_pattern.data = np.ones_like(S_T_pattern.data)

        # Intersection mask: element-wise multiply of patterns gives 1 only where both exist
        intersection_mask = S_pattern.multiply(S_T_pattern)

        # Apply mask: zero out entries not in intersection, then symmetrize
        S_intersected = S.multiply(intersection_mask)
        stiffness_csr = 0.5 * (S_intersected + S_intersected.T)

        # Recompute diagonal to ensure zero row sums
        stiffness_csr = stiffness_csr.tolil()
        for i in range(num_vertices):
            row = stiffness_csr.getrow(i).toarray().flatten()
            off_diag_sum = row.sum() - row[i]
            stiffness_csr[i, i] = -off_diag_sum
        stiffness_csr = stiffness_csr.tocsr()
    else:
        # Union (default): keep edge if either direction selected it
        stiffness_csr = 0.5 * (stiffness_csr + stiffness_csr.T)

    # === STEP 7: Build mass matrix (vectorized, no Python loop) ===
    center_indices_np = center_indices.cpu().numpy()
    areas_np = areas.cpu().numpy()

    # Vectorized accumulation using np.add.at
    mass_diagonal = np.zeros(num_vertices, dtype=np.float64)
    mass_counts = np.zeros(num_vertices, dtype=np.float64)
    np.add.at(mass_diagonal, center_indices_np, areas_np)
    np.add.at(mass_counts, center_indices_np, 1.0)

    # Average if vertex appeared multiple times
    nonzero_mask = mass_counts > 0
    mass_diagonal[nonzero_mask] /= mass_counts[nonzero_mask]

    # Handle vertices never seen as centers
    zero_mask = mass_counts == 0
    if np.any(zero_mask):
        mass_diagonal[zero_mask] = 1e-6

    mass_csr = scipy.sparse.diags(mass_diagonal, format='csr')

    return stiffness_csr, mass_csr


def assemble_gradient_operator(
        grad_coeffs: torch.Tensor,
        attention_mask: torch.Tensor,
        vertex_indices: torch.Tensor,
        center_indices: torch.Tensor,
        batch_indices: torch.Tensor
) -> scipy.sparse.csr_matrix:
    """
    Assemble sparse gradient operator G from learned gradient coefficients.

    G is a (3N, N) sparse matrix. For each patch centered at vertex i with neighbor j:
        G[3i:3i+3, j] = +g_ij        (neighbor contribution)
        G[3i:3i+3, i] = -Σ_j g_ij   (center, ensures G @ const = 0)

    The gradient of a scalar function f is computed as:
        (∇f)_flat = G @ f   →   reshape to (N, 3)

    This gives the vertex-based gradient:
        (∇f)_i = Σ_j g_ij (f_j - f_i)

    The divergence (adjoint w.r.t. vertex-area inner product) is:
        div(X) = -(1/A) * G^T @ (A_3d * X_flat)
    where A_3d repeats vertex areas 3x (once per spatial component).

    Uses the same GPU-optimized patch indexing pattern as
    assemble_stiffness_and_mass_matrices for consistency.

    Args:
        grad_coeffs: Gradient coefficients (num_patches, max_k, 3)
        attention_mask: Valid token mask (num_patches, max_k) - True for real tokens
        vertex_indices: Flat neighbor vertex indices (total_points,)
        center_indices: Center vertex per patch (num_patches,)
        batch_indices: Batch/patch index per point (total_points,)

    Returns:
        G: Sparse gradient operator (3*num_vertices, num_vertices) as CSR matrix
    """
    device = grad_coeffs.device
    num_patches = grad_coeffs.shape[0]
    max_k = grad_coeffs.shape[1]

    num_vertices = max(vertex_indices.max().item(), center_indices.max().item()) + 1

    # === STEP 1: Flatten and filter valid entries ===
    coeffs_flat = grad_coeffs.reshape(-1, 3)  # (num_patches * max_k, 3)
    mask_flat = attention_mask.flatten()  # (num_patches * max_k,)

    patch_indices_flat = torch.arange(num_patches, device=device).repeat_interleave(max_k)

    valid_coeffs = coeffs_flat[mask_flat]         # (num_valid, 3)
    valid_patch_indices = patch_indices_flat[mask_flat]  # (num_valid,)

    # === STEP 2: Compute positions within each patch (same pattern as stiffness assembly) ===
    num_valid = len(valid_patch_indices)

    if num_valid == 0:
        return scipy.sparse.csr_matrix((3 * num_vertices, num_vertices))

    patch_changes = torch.ones(num_valid, dtype=torch.bool, device=device)
    if num_valid > 1:
        patch_changes[1:] = valid_patch_indices[1:] != valid_patch_indices[:-1]

    group_ids = torch.cumsum(patch_changes.long(), dim=0) - 1
    change_indices = torch.where(patch_changes)[0]
    group_starts = change_indices[group_ids]
    positions_in_patch = torch.arange(num_valid, device=device, dtype=torch.long) - group_starts

    # === STEP 3: Get vertex indices ===
    batch_sizes = batch_indices.bincount(minlength=num_patches)
    cumsum_sizes = torch.cumsum(batch_sizes, dim=0)
    starts = torch.cat([torch.tensor([0], device=device, dtype=torch.long), cumsum_sizes[:-1]])

    valid_center_vertices = center_indices[valid_patch_indices]
    valid_neighbor_vertices = vertex_indices[starts[valid_patch_indices] + positions_in_patch]

    # === STEP 4: Transfer to CPU/numpy for COO construction ===
    centers_np = valid_center_vertices.cpu().numpy()
    neighbors_np = valid_neighbor_vertices.cpu().numpy()
    coeffs_np = valid_coeffs.detach().float().cpu().numpy().astype(np.float64)  # (num_valid, 3)

    num_valid_np = len(centers_np)

    # === STEP 5: Build off-diagonal entries: G[3*center+d, neighbor] = g_ij[d] ===
    # Each valid edge contributes 3 entries (one per spatial dimension)
    rows_neighbor = np.repeat(3 * centers_np, 3) + np.tile(np.arange(3), num_valid_np)
    cols_neighbor = np.repeat(neighbors_np, 3)
    data_neighbor = coeffs_np.flatten()  # (num_valid * 3,)

    # === STEP 6: Build diagonal entries: G[3*center+d, center] = -Σ_j g_ij[d] ===
    # Accumulate gradient coefficient sums per center vertex
    center_sums = np.zeros((num_vertices, 3), dtype=np.float64)
    np.add.at(center_sums, centers_np, coeffs_np)

    # Only create entries for vertices that are patch centers
    active_centers = np.where(np.any(center_sums != 0, axis=1))[0]
    num_active = len(active_centers)

    rows_diag = np.repeat(3 * active_centers, 3) + np.tile(np.arange(3), num_active)
    cols_diag = np.repeat(active_centers, 3)
    data_diag = -center_sums[active_centers].flatten()

    # === STEP 7: Combine and create sparse matrix ===
    all_rows = np.concatenate([rows_neighbor, rows_diag])
    all_cols = np.concatenate([cols_neighbor, cols_diag])
    all_data = np.concatenate([data_neighbor, data_diag])

    G = scipy.sparse.coo_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(3 * num_vertices, num_vertices)
    ).tocsr()
    G.sum_duplicates()

    return G


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
    # Pre-factorize the shift-invert operator for speed (uses best available backend:
    # pypardiso/MKL if installed, otherwise scipy splu)
    sigma = 1e-8 if mass_matrix is not None else -0.01
    OPinv = None

    try:
        from neural_local_laplacian.utils.geodesic_utils import sparse_factorize
        if mass_matrix is not None:
            OP = laplacian_matrix - sigma * mass_matrix
        else:
            OP = laplacian_matrix - sigma * scipy.sparse.eye(laplacian_matrix.shape[0])
        factor = sparse_factorize(OP, spd=True)
        OPinv = scipy.sparse.linalg.LinearOperator(
            shape=OP.shape,
            matvec=lambda x: factor.solve(x),
            dtype=np.float64
        )
    except Exception:
        pass  # Fall back to eigsh's internal factorization

    if mass_matrix is not None:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            laplacian_matrix, k=num_eigenvalues, M=mass_matrix,
            sigma=sigma, which='LM', OPinv=OPinv
        )
    else:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            laplacian_matrix, k=num_eigenvalues,
            sigma=sigma, which='LM', OPinv=OPinv
        )

    # Sort by eigenvalue (ascending)
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    return eigenvalues, eigenvectors