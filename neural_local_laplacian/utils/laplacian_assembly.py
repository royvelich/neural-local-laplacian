"""
Laplacian assembly functions for gradient-mode models.

Takes gradient coefficients (N, k, 3) and kNN indices (N, k) and returns
either a dense (N, N) torch tensor or a scipy.sparse.csr_matrix, controlled
by the `sparse` flag in LaplacianConfig.

Areas are NOT included in L — the learned g_ij already carry the integration
measure from training against cotangent weights.

The eigenvalue problem is: L v = λ M v, where M = diag(areas) is
constructed separately by the caller from fwd['areas'].

Assembly variants:
    isotropic:   scalar w_ij = ||g_ij||^2, kNN-sparse (no 2-hop)
    anisotropic: full Gram L = G^T G, has 2-hop fill-in

Pruning options (dense mode only):
    none:  keep L as-is
    knn:   zero out entries not in a kNN graph
    topk:  keep k largest-magnitude off-diagonal entries per row
"""

import numpy as np
import torch
import scipy.sparse
from dataclasses import dataclass
from typing import Literal, Optional, Union


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LaplacianConfig:
    """Configuration for Laplacian assembly and pruning.

    Args:
        assembly: 'isotropic' (scalar ||g||^2, kNN-sparse) or
                  'anisotropic' (full Gram G^T G, 2-hop fill-in).
        pruning: 'none' (keep as-is), 'knn' (prune to kNN graph),
                 or 'topk' (keep k largest per row).
        k_prune: For 'topk': number of off-diagonal entries to keep per row.
                 For 'knn': neighbor count for the pruning graph (caller
                 must compute knn_prune and pass it to assemble_laplacian).
                 Ignored when pruning='none'.
        sparse: If True, return scipy.sparse.csr_matrix instead of dense
                torch.Tensor. Much faster for large meshes in eval.
                Pruning is not supported in sparse mode (must be 'none').
    """
    assembly: Literal['isotropic', 'anisotropic'] = 'isotropic'
    pruning: Literal['none', 'knn', 'topk'] = 'none'
    k_prune: Optional[int] = None
    sparse: bool = False

    def __post_init__(self):
        if self.assembly not in ('isotropic', 'anisotropic'):
            raise ValueError(f"assembly must be 'isotropic' or 'anisotropic', got: {self.assembly}")
        if self.pruning not in ('none', 'knn', 'topk'):
            raise ValueError(f"pruning must be 'none', 'knn', or 'topk', got: {self.pruning}")
        if self.pruning in ('knn', 'topk') and self.k_prune is None:
            raise ValueError(f"k_prune is required when pruning='{self.pruning}'")
        if self.sparse and self.pruning != 'none':
            raise ValueError(f"Pruning is not supported in sparse mode (got pruning='{self.pruning}')")

    @property
    def tag(self) -> str:
        """Short string tag for logging/metric prefixes."""
        t = self.assembly
        if self.pruning == 'knn':
            t += f'_knn{self.k_prune}'
        elif self.pruning == 'topk':
            t += f'_top{self.k_prune}'
        if self.sparse:
            t += '_sp'
        return t


# =============================================================================
# Dispatcher
# =============================================================================

def assemble_laplacian(
    grad_coeffs: torch.Tensor,
    knn: torch.Tensor,
    config: LaplacianConfig,
    knn_prune: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, scipy.sparse.csr_matrix]:
    """Assemble a Laplacian matrix according to config.

    Args:
        grad_coeffs: (N, k, 3) gradient coefficients per neighbor.
        knn: (N, k) kNN indices used for assembly.
        config: LaplacianConfig specifying assembly + pruning + sparse.
        knn_prune: (N, k') optional kNN indices for pruning='knn'.
                   If None and pruning='knn', uses the assembly knn.

    Returns:
        L: (N, N) dense torch.Tensor or scipy.sparse.csr_matrix.
    """
    if config.sparse:
        # Sparse path — returns scipy.sparse.csr_matrix
        if config.assembly == 'anisotropic':
            return _assemble_anisotropic_sparse(grad_coeffs, knn)
        else:
            return _assemble_isotropic_sparse(grad_coeffs, knn)

    # Dense path — returns torch.Tensor
    if config.assembly == 'anisotropic':
        L = assemble_anisotropic_laplacian(grad_coeffs, knn)
    else:
        L = assemble_isotropic_laplacian(grad_coeffs, knn)

    # Pruning (dense only)
    if config.pruning == 'knn':
        L = prune_to_knn(L, knn_prune if knn_prune is not None else knn)
    elif config.pruning == 'topk':
        L = prune_to_topk(L, config.k_prune)

    return L


# =============================================================================
# Dense assembly (for training with backprop)
# =============================================================================

def assemble_anisotropic_laplacian(
    grad_coeffs: torch.Tensor,
    knn: torch.Tensor,
) -> torch.Tensor:
    """Assemble dense anisotropic Laplacian L = G^T G.

    Uses the full Gram structure: L_pq = sum_i <g_ip, g_iq> where
    g_ii = -sum_j g_ij (consistency constraint). Has 2-hop fill-in.

    Args:
        grad_coeffs: (N, k, 3) gradient coefficients per neighbor.
        knn: (N, k) kNN indices (long tensor).

    Returns:
        L: (N, N) dense symmetric PSD Laplacian.
    """
    N, k, _ = grad_coeffs.shape
    device = grad_coeffs.device

    # Extended coefficients: prepend center g_ii = -sum_j g_ij
    center_coeffs = -grad_coeffs.sum(dim=1, keepdim=True)  # (N, 1, 3)
    ext_coeffs = torch.cat([center_coeffs, grad_coeffs], dim=1)  # (N, k+1, 3)

    # Local Gram matrices: ext @ ext^T per vertex
    gram = torch.bmm(ext_coeffs, ext_coeffs.transpose(1, 2))  # (N, k+1, k+1)

    # Map local indices to global: [i, knn[i,0], ..., knn[i,k-1]]
    center_idx = torch.arange(N, device=device).unsqueeze(1)  # (N, 1)
    ext_indices = torch.cat([center_idx, knn], dim=1)  # (N, k+1)

    # Scatter Gram entries into N×N matrix
    kp1 = k + 1
    row_idx = ext_indices[:, :, None].expand(-1, -1, kp1)  # (N, k+1, k+1)
    col_idx = ext_indices[:, None, :].expand(-1, kp1, -1)  # (N, k+1, k+1)
    flat_idx = (row_idx * N + col_idx).reshape(-1)

    L_flat = torch.zeros(N * N, device=device, dtype=grad_coeffs.dtype)
    L_flat = L_flat.scatter_add(0, flat_idx, gram.reshape(-1))
    L = L_flat.view(N, N)
    L = 0.5 * (L + L.T)
    return L


def assemble_isotropic_laplacian(
    grad_coeffs: torch.Tensor,
    knn: torch.Tensor,
) -> torch.Tensor:
    """Assemble dense isotropic graph Laplacian from gradient coefficients.

    Drops cross-terms, keeps only w_ij = ||g_ij||^2. Produces a standard
    graph Laplacian L = D - W. Guaranteed PSD and kNN-sparse (no 2-hop).

    Args:
        grad_coeffs: (N, k, 3) gradient coefficients per neighbor.
        knn: (N, k) kNN indices (long tensor).

    Returns:
        L: (N, N) dense symmetric PSD Laplacian (structurally kNN-sparse).
    """
    N, k, _ = grad_coeffs.shape
    device = grad_coeffs.device

    # w_ij = ||g_ij||^2
    edge_weights = (grad_coeffs ** 2).sum(dim=2)  # (N, k)

    # Build graph Laplacian: L_ij = -w_ij, symmetrized
    L = torch.zeros(N, N, device=device, dtype=grad_coeffs.dtype)
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn)
    L[row_idx, knn] -= edge_weights
    L[knn, row_idx] -= edge_weights
    L = 0.5 * L  # average both directions

    # Fix diagonal: L_ii = -sum of off-diagonal in row i
    L.fill_diagonal_(0.0)
    L.diagonal().copy_(-L.sum(dim=1))

    return L


# =============================================================================
# Sparse assembly (for eval — builds torch.sparse on GPU, converts to scipy)
# =============================================================================

def _torch_sparse_to_scipy(L_sparse: torch.Tensor, N: int) -> scipy.sparse.csr_matrix:
    """Convert a coalesced torch sparse COO tensor to scipy CSR."""
    L_sparse = L_sparse.coalesce()
    indices = L_sparse.indices().cpu().numpy()
    values = L_sparse.values().cpu().to(torch.float64).numpy()
    return scipy.sparse.coo_matrix(
        (values, (indices[0], indices[1])), shape=(N, N)
    ).tocsr()


def _assemble_isotropic_sparse(
    grad_coeffs: torch.Tensor,
    knn: torch.Tensor,
) -> scipy.sparse.csr_matrix:
    """Assemble sparse isotropic graph Laplacian from gradient coefficients.

    Same math as assemble_isotropic_laplacian but builds sparse COO
    on GPU — never allocates an N×N matrix. O(N*k) memory.

    Args:
        grad_coeffs: (N, k, 3) gradient coefficients per neighbor.
        knn: (N, k) kNN indices (long tensor).

    Returns:
        L: (N, N) scipy.sparse.csr_matrix, symmetric PSD.
    """
    N, k, _ = grad_coeffs.shape
    device = grad_coeffs.device

    # w_ij = ||g_ij||^2
    edge_weights = (grad_coeffs ** 2).sum(dim=2)  # (N, k)

    # Build COO indices on GPU: both (i→j) and (j→i)
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn).reshape(-1)
    col_idx = knn.reshape(-1)
    vals = -edge_weights.reshape(-1)

    # Stack both directions, average
    all_rows = torch.cat([row_idx, col_idx])
    all_cols = torch.cat([col_idx, row_idx])
    all_vals = torch.cat([vals, vals]) * 0.5

    # Build torch sparse, coalesce (sums duplicates)
    indices = torch.stack([all_rows, all_cols])  # (2, 2*N*k)
    L_sparse = torch.sparse_coo_tensor(indices, all_vals, size=(N, N)).coalesce()

    # Symmetrize: 0.5 * (L + L^T)
    L_sparse = (L_sparse + L_sparse.t()).coalesce()
    L_sparse._values().mul_(0.5)

    # Convert to scipy for diagonal fix
    L_csr = _torch_sparse_to_scipy(L_sparse, N)

    # Fix diagonal: L_ii = -sum of off-diagonal in row i
    L_csr.setdiag(0.0)
    row_sums = np.asarray(L_csr.sum(axis=1)).ravel()
    L_csr.setdiag(-row_sums)
    L_csr.eliminate_zeros()

    return L_csr


def _assemble_anisotropic_sparse(
    grad_coeffs: torch.Tensor,
    knn: torch.Tensor,
) -> scipy.sparse.csr_matrix:
    """Assemble sparse anisotropic Laplacian L = G^T G.

    Same math as assemble_anisotropic_laplacian but builds sparse COO
    on GPU from local Gram matrices — never allocates an N×N matrix.

    Args:
        grad_coeffs: (N, k, 3) gradient coefficients per neighbor.
        knn: (N, k) kNN indices (long tensor).

    Returns:
        L: (N, N) scipy.sparse.csr_matrix, symmetric PSD.
    """
    N, k, _ = grad_coeffs.shape
    device = grad_coeffs.device

    # Extended coefficients: prepend center g_ii = -sum_j g_ij
    center_coeffs = -grad_coeffs.sum(dim=1, keepdim=True)  # (N, 1, 3)
    ext_coeffs = torch.cat([center_coeffs, grad_coeffs], dim=1)  # (N, k+1, 3)

    # Local Gram matrices on GPU
    gram = torch.bmm(ext_coeffs, ext_coeffs.transpose(1, 2))  # (N, k+1, k+1)

    # Global indices: [i, knn[i,0], ..., knn[i,k-1]]
    center_idx = torch.arange(N, device=device).unsqueeze(1)  # (N, 1)
    ext_indices = torch.cat([center_idx, knn], dim=1)  # (N, k+1)

    # Build COO from all local Gram entries on GPU
    kp1 = k + 1
    row_idx = ext_indices[:, :, None].expand(-1, -1, kp1).reshape(-1)
    col_idx = ext_indices[:, None, :].expand(-1, kp1, -1).reshape(-1)
    vals = gram.reshape(-1)

    # Build torch sparse, coalesce (sums duplicates from overlapping patches)
    indices = torch.stack([row_idx, col_idx])  # (2, N*(k+1)^2)
    L_sparse = torch.sparse_coo_tensor(indices, vals, size=(N, N)).coalesce()

    # Symmetrize: 0.5 * (L + L^T)
    L_sparse = (L_sparse + L_sparse.t()).coalesce()
    L_sparse._values().mul_(0.5)

    # Convert to scipy
    return _torch_sparse_to_scipy(L_sparse, N)


# =============================================================================
# Pruning (dense mode only)
# =============================================================================

def prune_to_knn(L: torch.Tensor, knn: torch.Tensor) -> torch.Tensor:
    """Zero out entries of L not in the kNN graph, recompute diagonal.

    Keeps entry L_ij if j in kNN(i) or i in kNN(j) (symmetric).
    Useful for removing 2-hop fill-in from the anisotropic Laplacian.

    Args:
        L: (N, N) dense Laplacian.
        knn: (N, k) kNN indices defining the sparsity pattern.

    Returns:
        L_pruned: (N, N) dense Laplacian with kNN sparsity.
    """
    N = L.shape[0]
    device = L.device

    # Build symmetric kNN adjacency mask
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn)
    mask[row_idx, knn] = True
    mask = mask | mask.T  # symmetrize

    # Keep kNN edges + diagonal
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    L_pruned = L * (mask | diag_mask).float()

    # Recompute diagonal for zero row sums
    off_diag = L_pruned * (1.0 - diag_mask.float())
    L_pruned = off_diag - torch.diag(off_diag.sum(dim=1))

    return L_pruned


def prune_to_topk(L: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k largest-magnitude off-diagonal entries per row.

    Useful for sparsity-quality tradeoff analysis. Recomputes diagonal
    afterwards to maintain zero row sums.

    Args:
        L: (N, N) dense Laplacian.
        k: Number of off-diagonal entries to keep per row.

    Returns:
        L_pruned: (N, N) dense Laplacian with at most k off-diag per row.
    """
    N = L.shape[0]
    device = L.device

    # Work with off-diagonal magnitudes
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    off_diag = L * (1.0 - diag_mask.float())

    # Find top-k by magnitude per row
    magnitudes = off_diag.abs()
    magnitudes.fill_diagonal_(0.0)  # exclude diagonal from ranking
    _, topk_indices = magnitudes.topk(min(k, N - 1), dim=1)

    # Build mask from top-k indices
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(topk_indices)
    mask[row_idx, topk_indices] = True

    # Symmetrize: keep edge if either direction selected it
    mask = mask | mask.T

    # Apply mask to off-diagonal, recompute diagonal
    pruned_off = off_diag * mask.float()
    L_pruned = pruned_off - torch.diag(pruned_off.sum(dim=1))

    return L_pruned


# =============================================================================
# Conversion helpers
# =============================================================================

def to_scipy_sparse(L):
    """Convert Laplacian to scipy CSR sparse matrix.

    Accepts dense torch.Tensor or scipy.sparse matrix (no-op).

    Args:
        L: (N, N) dense torch tensor or scipy.sparse matrix.

    Returns:
        L_scipy: scipy.sparse.csr_matrix
    """
    if scipy.sparse.issparse(L):
        return L.tocsr()
    L_np = L.detach().cpu().numpy()
    return scipy.sparse.csr_matrix(L_np)


def mass_matrix_to_scipy(areas: torch.Tensor):
    """Convert areas vector to scipy sparse diagonal mass matrix.

    Args:
        areas: (N,) torch tensor of vertex areas.

    Returns:
        M_scipy: scipy.sparse.dia_matrix
    """
    return scipy.sparse.diags(areas.detach().cpu().numpy())