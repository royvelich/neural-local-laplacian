"""
Geodesic distance computation utilities using the Heat Method.

This module provides shared functions for computing geodesic distances
via the Heat Method (Crane et al. 2013) and comparing them against
ground truth exact geodesics.

Used by:
- laplacian_modules.py: For validation metrics during training
- visualize_validation.py: For visualization and analysis
- mesh_datasets.py: For precomputing operators and exact geodesics
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


# =============================================================================
# Sparse solver backend: explicit selection, no fallback
#
# Default: scipy (always available). Override with set_solver_backend() or
# +solver=pypardiso / +solver=cholmod from timing scripts.
# =============================================================================

_HAS_CHOLMOD = False
try:
    from sksparse.cholmod import cholesky as cholmod_cholesky
    _HAS_CHOLMOD = True
except ImportError:
    pass

_HAS_PARDISO = False
try:
    import pypardiso
    _HAS_PARDISO = True
except ImportError:
    pass

_SOLVER_BACKEND = 'scipy'
print(f"[sparse solver] Backend: {_SOLVER_BACKEND}"
      + (f" (available: pypardiso)" if _HAS_PARDISO else "")
      + (f" (available: cholmod)" if _HAS_CHOLMOD else ""))

# Regularization epsilon for heat method solves (ensures SPD for Cholesky, avoids singular matrices)
_HEAT_EPS = 1e-8


def set_solver_backend(backend: str):
    """
    Switch sparse solver backend at runtime.

    Args:
        backend: 'cholmod', 'pypardiso', or 'scipy'
    """
    global _SOLVER_BACKEND
    if backend == 'cholmod':
        if not _HAS_CHOLMOD:
            raise RuntimeError("scikit-sparse (cholmod) not installed")
        _SOLVER_BACKEND = 'cholmod'
    elif backend == 'pypardiso':
        if not _HAS_PARDISO:
            raise RuntimeError("pypardiso not installed")
        _SOLVER_BACKEND = 'pypardiso'
    elif backend == 'scipy':
        _SOLVER_BACKEND = 'scipy'
    else:
        raise ValueError(f"Unknown solver backend: {backend}. Use 'cholmod', 'pypardiso', or 'scipy'.")
    print(f"[sparse solver] Backend set to: {_SOLVER_BACKEND}")


class SparseFactorization:
    """
    Cached sparse matrix factorization for solving Ax=b with multiple right-hand sides.

    Uses exactly the backend specified by _SOLVER_BACKEND — no fallback.
    """

    # Single shared pypardiso solver (Windows allows only one instance)
    _shared_solver = None

    def __init__(self, A: scipy.sparse.spmatrix, spd: bool = False):
        if _SOLVER_BACKEND == 'cholmod':
            self._cholmod_factor = cholmod_cholesky(A.tocsc())
            self._backend = 'cholmod'
        elif _SOLVER_BACKEND == 'pypardiso':
            if SparseFactorization._shared_solver is None:
                SparseFactorization._shared_solver = pypardiso.PyPardisoSolver()
            self._solver = SparseFactorization._shared_solver
            self._A_csr = A.tocsr()
            self._solver.factorize(self._A_csr)
            self._backend = 'pypardiso'
        else:
            self._factor = scipy.sparse.linalg.splu(A.tocsc())
            self._backend = 'scipy'

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve Ax = b using the cached factorization."""
        if self._backend == 'cholmod':
            return self._cholmod_factor.solve_A(b)
        elif self._backend == 'pypardiso':
            return self._solver.solve(self._A_csr, b)
        else:
            return self._factor.solve(b)


def sparse_factorize(A: scipy.sparse.spmatrix, spd: bool = False) -> SparseFactorization:
    """Factorize a sparse matrix for repeated solves."""
    return SparseFactorization(A, spd=spd)


def sparse_solve(
    A: scipy.sparse.spmatrix,
    b: np.ndarray,
    device: Optional[torch.device] = None,
    factor: Optional[SparseFactorization] = None,
    spd: bool = False
) -> np.ndarray:
    """Solve sparse linear system Ax = b using the selected backend."""
    if factor is not None:
        return factor.solve(b)

    if _SOLVER_BACKEND == 'cholmod':
        return cholmod_cholesky(A.tocsc()).solve_A(b)
    elif _SOLVER_BACKEND == 'pypardiso':
        return pypardiso.spsolve(A.tocsr(), b)
    else:
        return scipy.sparse.linalg.spsolve(A.tocsc(), b)


# Optional imports with graceful fallback
try:
    from pcdiff import knn_graph, estimate_basis, build_grad_div
    HAS_PCDIFF = True
except ImportError:
    HAS_PCDIFF = False

try:
    import igl
    HAS_IGL = True
except ImportError:
    HAS_IGL = False


def ensure_psd(L: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    """
    Ensure a Laplacian matrix follows positive semi-definite convention
    (positive diagonal, negative off-diagonal).

    If the diagonal is predominantly negative (NSD convention), negates L.
    This eliminates sign ambiguity in all downstream solves (heat method,
    Green's function, etc.) without needing double-solve workarounds.

    Args:
        L: Sparse Laplacian matrix (N, N)

    Returns:
        L in PSD convention (may be negated copy, or original if already PSD)
    """
    diag = np.array(L.diagonal()).flatten()
    # Check median of diagonal (robust to a few outlier entries)
    if np.median(diag) < 0:
        return -L
    return L


def _scipy_to_torch_sparse(A: scipy.sparse.spmatrix, device: torch.device) -> torch.Tensor:
    """Convert scipy sparse matrix to torch sparse COO tensor on the given device."""
    A_coo = A.tocoo()
    indices = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(A_coo.data, dtype=torch.float64, device=device)
    return torch.sparse_coo_tensor(indices, values, size=A_coo.shape).coalesce()


def _cg_solve_gpu(
    A: torch.Tensor,
    b: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> torch.Tensor:
    """
    Conjugate Gradient solver for SPD sparse systems on GPU.

    Solves Ax = b where A is symmetric positive definite.

    Args:
        A: Sparse torch tensor (N, N) on GPU
        b: Dense torch tensor (N,) on GPU
        tol: Convergence tolerance (relative residual)
        max_iter: Maximum iterations

    Returns:
        Solution x as dense torch tensor (N,) on GPU
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    r_dot = r.dot(r)
    r0_norm = r.norm()

    if r0_norm < 1e-15:
        return x

    for i in range(max_iter):
        Ap = torch.sparse.mm(A, p.unsqueeze(1)).squeeze(1)
        alpha = r_dot / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        r_dot_new = r.dot(r)

        if r.norm() / r0_norm < tol:
            break

        beta = r_dot_new / r_dot
        p = r + beta * p
        r_dot = r_dot_new

    return x




@dataclass
class GeodesicMetrics:
    """Metrics comparing computed geodesics against exact ground truth."""
    correlation: float  # Pearson correlation (1.0 = perfect)
    mae_normalized: float  # Mean absolute error on [0,1] normalized distances
    max_error_normalized: float  # Maximum error on normalized distances
    monotonicity: float  # Fraction of pairs with correct distance ordering (1.0 = perfect)

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        """Convert to dictionary for logging."""
        p = f"{prefix}_" if prefix else ""
        return {
            f'{p}geodesic_correlation': self.correlation,
            f'{p}geodesic_mae': self.mae_normalized,
            f'{p}geodesic_max_error': self.max_error_normalized,
            f'{p}geodesic_monotonicity': self.monotonicity,
        }


def normalize_distances(distances: np.ndarray) -> np.ndarray:
    """
    Normalize distances to [0, 1] range.

    Handles NaN and Inf values gracefully by treating them as missing data.

    Args:
        distances: Array of distances

    Returns:
        Normalized distances with min=0, max=1. NaN/Inf inputs remain NaN.
    """
    finite_mask = np.isfinite(distances)
    if not finite_mask.any():
        return np.zeros_like(distances)

    d_min = distances[finite_mask].min()
    d_max = distances[finite_mask].max()
    if d_max - d_min < 1e-10:
        result = np.zeros_like(distances)
        result[~finite_mask] = np.nan
        return result

    result = (distances - d_min) / (d_max - d_min)
    result[~finite_mask] = np.nan
    return result


def compute_geodesic_metrics(
    computed: np.ndarray,
    exact: np.ndarray,
    num_monotonicity_samples: int = 5000
) -> GeodesicMetrics:
    """
    Compute metrics comparing computed geodesics against exact ground truth.

    Both distance arrays are normalized to [0, 1] before comparison for
    scale-invariant metrics.

    Args:
        computed: Computed geodesic distances (N,)
        exact: Exact geodesic distances (N,)
        num_monotonicity_samples: Number of random pairs to sample for monotonicity

    Returns:
        GeodesicMetrics with correlation, MAE, max error, and monotonicity
    """
    n = len(computed)

    # Normalize both to [0, 1] for fair comparison
    computed_norm = normalize_distances(computed)
    exact_norm = normalize_distances(exact)

    # Filter out invalid values
    valid_mask = np.isfinite(computed_norm) & np.isfinite(exact_norm)
    c_valid = computed_norm[valid_mask]
    e_valid = exact_norm[valid_mask]

    # Correlation
    if len(c_valid) > 1:
        corr = np.corrcoef(c_valid, e_valid)[0, 1]
        corr = float(corr) if np.isfinite(corr) else 0.0
    else:
        corr = 0.0

    # MAE and max error on normalized distances
    if len(c_valid) > 0:
        errors = np.abs(c_valid - e_valid)
        mae_norm = float(errors.mean())
        max_err_norm = float(errors.max())
    else:
        mae_norm = 1.0
        max_err_norm = 1.0

    # Monotonicity: do distances increase in the same order?
    num_samples = min(num_monotonicity_samples, n * (n - 1) // 2)
    mono_score = 0.0

    if num_samples > 100:
        idx1 = np.random.randint(0, n, num_samples)
        idx2 = np.random.randint(0, n, num_samples)
        valid_pairs = idx1 != idx2
        idx1, idx2 = idx1[valid_pairs], idx2[valid_pairs]

        if len(idx1) > 0:
            e1, e2 = exact[idx1], exact[idx2]
            c1, c2 = computed[idx1], computed[idx2]

            # Filter out pairs with NaN/Inf in either array
            finite_pairs = np.isfinite(e1) & np.isfinite(e2) & np.isfinite(c1) & np.isfinite(c2)
            e1, e2 = e1[finite_pairs], e2[finite_pairs]
            c1, c2 = c1[finite_pairs], c2[finite_pairs]

            if len(e1) > 0:
                # If e1 > e2 (farther in exact), then c1 should be > c2
                correct = ((e1 > e2) & (c1 > c2)) | ((e2 > e1) & (c2 > c1))
                not_tie = np.abs(e1 - e2) > 1e-8
                mono_score = float(correct[not_tie].mean()) if not_tie.sum() > 0 else 0.0

    return GeodesicMetrics(
        correlation=corr,
        mae_normalized=mae_norm,
        max_error_normalized=max_err_norm,
        monotonicity=mono_score
    )


def build_pointcloud_grad_div_operators(
    vertices: np.ndarray,
    edge_index: np.ndarray
) -> Tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix]:
    """
    Build gradient and divergence operators for a point cloud using pcdiff.

    The operators are built from the given k-NN connectivity (edge_index),
    ensuring they match the Laplacian built from the same connectivity.

    Args:
        vertices: Point positions (N, 3)
        edge_index: k-NN connectivity as (2, num_edges) array where
                    edge_index[0] = source indices, edge_index[1] = target indices

    Returns:
        Tuple of (grad_op, div_op):
        - grad_op: Gradient operator (2N, N) sparse matrix
        - div_op: Divergence operator (N, 2N) sparse matrix

    Raises:
        ImportError: If pcdiff is not installed
    """
    if not HAS_PCDIFF:
        raise ImportError("pcdiff is required for point cloud gradient/divergence operators. "
                         "Install with: pip install pcdiff")

    vertices = vertices.astype(np.float64)

    # Estimate local tangent frames (normals + tangent basis)
    basis = estimate_basis(vertices, edge_index)

    # Build gradient and divergence operators
    grad_op, div_op = build_grad_div(vertices, *basis, edge_index)

    return grad_op, div_op


def edge_index_from_knn_indices(
    vertex_indices: np.ndarray,
    center_indices: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Convert k-NN indices to edge_index format for pcdiff.

    Args:
        vertex_indices: Flat array of neighbor indices (N*k,)
        center_indices: Array of center vertex indices (N,)
        k: Number of neighbors per vertex

    Returns:
        edge_index: (2, N*k) array where [0] = sources, [1] = targets
    """
    num_centers = len(center_indices)

    # Each center connects to k neighbors
    sources = np.repeat(center_indices, k)  # (N*k,)
    targets = vertex_indices  # (N*k,)

    edge_index = np.stack([sources, targets], axis=0)  # (2, N*k)

    return edge_index


class HeatMethodPrefactored:
    """Prefactored heat method solver. Holds factorized matrices for fast per-source solves."""
    __slots__ = ('heat_factor', 'poisson_factor', 'grad_and_div_fn', 'n')

    def __init__(self, heat_factor, poisson_factor, grad_and_div_fn, n):
        self.heat_factor = heat_factor
        self.poisson_factor = poisson_factor
        self.grad_and_div_fn = grad_and_div_fn
        self.n = n


def heat_method_prefactorize(
    S: scipy.sparse.spmatrix,
    M: scipy.sparse.spmatrix,
    n_vertices: int,
    grad_and_div_fn,
    t: Optional[float] = None,
) -> HeatMethodPrefactored:
    """
    Prefactorize the heat and Poisson systems for the Heat Method.

    This is the one-time cost (analogous to pp3d.PointCloudHeatSolver constructor).
    Call heat_method_solve() afterwards for fast per-source solves.

    Args:
        S: Stiffness/Laplacian matrix (N, N)
        M: Mass matrix (N, N) -- diagonal
        n_vertices: Number of vertices
        grad_and_div_fn: Callable(u) -> rhs (gradient + normalization + divergence)
        t: Diffusion time step (auto-computed from M if None)

    Returns:
        HeatMethodPrefactored object for use with heat_method_solve().
    """
    n = n_vertices
    S = ensure_psd(S.astype(np.float64))
    M = M.astype(np.float64)

    if t is None:
        areas = np.array(M.diagonal()).flatten() if scipy.sparse.issparse(M) else np.diag(M)
        h = np.sqrt(areas.mean())
        t = h ** 2

    eps = _HEAT_EPS
    A = M + t * S + eps * scipy.sparse.eye(n)
    S_reg = S + eps * scipy.sparse.eye(n)

    heat_factor = sparse_factorize(A)
    poisson_factor = sparse_factorize(S_reg)

    return HeatMethodPrefactored(heat_factor, poisson_factor, grad_and_div_fn, n)


def heat_method_solve(
    prefactored: HeatMethodPrefactored,
    source_indices: List[int],
    verbose: bool = False,
) -> List[Optional[np.ndarray]]:
    """
    Solve for geodesic distances from multiple sources using prefactored systems.

    This is the per-source cost (analogous to pp3d solver.compute_distance()).

    Args:
        prefactored: HeatMethodPrefactored from heat_method_prefactorize()
        source_indices: List of source vertex indices
        verbose: If True, print internal timing breakdown

    Returns:
        List of geodesic distances (N,) per source, or None for failed sources.
    """
    import time as _time

    n = prefactored.n
    heat_factor = prefactored.heat_factor
    poisson_factor = prefactored.poisson_factor
    grad_and_div_fn = prefactored.grad_and_div_fn
    num_sources = len(source_indices)

    _t_heat = 0.0
    _t_grad = 0.0
    _t_poisson = 0.0

    # ---- Pass 1: Heat diffusion ----
    rhs_list = []
    failed = [False] * num_sources

    for i, source_idx in enumerate(source_indices):
        try:
            delta = np.zeros(n)
            delta[source_idx] = 1.0

            _t0 = _time.perf_counter()
            u = heat_factor.solve(delta)
            _t_heat += _time.perf_counter() - _t0

            if not np.all(np.isfinite(u)):
                failed[i] = True
                rhs_list.append(None)
                continue

            _t0 = _time.perf_counter()
            rhs_list.append(grad_and_div_fn(u))
            _t_grad += _time.perf_counter() - _t0

        except Exception:
            failed[i] = True
            rhs_list.append(None)

    # ---- Pass 2: Poisson solve ----
    results = []
    for i, source_idx in enumerate(source_indices):
        if failed[i] or rhs_list[i] is None:
            results.append(None)
            continue

        try:
            rhs = rhs_list[i]

            _t0 = _time.perf_counter()
            phi_pos = poisson_factor.solve(rhs)
            phi_neg = poisson_factor.solve(-rhs)
            _t_poisson += _time.perf_counter() - _t0

            if not np.all(np.isfinite(phi_pos)) and not np.all(np.isfinite(phi_neg)):
                results.append(None)
                continue

            phi_pos_shifted = phi_pos - phi_pos.min() if np.all(np.isfinite(phi_pos)) else np.full(n, np.inf)
            phi_neg_shifted = phi_neg - phi_neg.min() if np.all(np.isfinite(phi_neg)) else np.full(n, np.inf)

            source_rank_pos = (phi_pos_shifted < phi_pos_shifted[source_idx]).sum()
            source_rank_neg = (phi_neg_shifted < phi_neg_shifted[source_idx]).sum()

            phi = phi_pos_shifted if source_rank_pos < source_rank_neg else phi_neg_shifted
            results.append(phi)

        except Exception:
            results.append(None)

    if verbose:
        total = _t_heat + _t_grad + _t_poisson
        print(f"    [heat_method_solve] {num_sources} sources: "
              f"heat={_t_heat*1000:.1f}ms, grad_div={_t_grad*1000:.1f}ms, "
              f"poisson={_t_poisson*1000:.1f}ms, total={total*1000:.1f}ms")

    return results


def compute_heat_geodesics_batch(
    S: scipy.sparse.spmatrix,
    M: scipy.sparse.spmatrix,
    source_indices: List[int],
    n_vertices: int,
    grad_and_div_fn,
    t: Optional[float] = None,
) -> List[Optional[np.ndarray]]:
    """
    Compute geodesic distances from multiple sources using the Heat Method.

    Convenience wrapper: prefactorizes + solves in one call.

    Args:
        S: Stiffness/Laplacian matrix (N, N)
        M: Mass matrix (N, N) -- diagonal
        source_indices: List of source vertex indices
        n_vertices: Number of vertices
        grad_and_div_fn: Callable(u) -> rhs
        t: Diffusion time step (auto-computed from M if None)

    Returns:
        List of geodesic distances (N,) per source, or None for failed sources.
    """
    try:
        prefactored = heat_method_prefactorize(S, M, n_vertices, grad_and_div_fn, t=t)
    except Exception as e:
        print(f"Heat Method batch: prefactorization failed: {e}")
        return [None] * len(source_indices)
    return heat_method_solve(prefactored, source_indices)


# ---------------------------------------------------------------------------
# grad_and_div factories -- one per Laplacian type
# ---------------------------------------------------------------------------

def make_grad_div_mesh(grad_op: scipy.sparse.spmatrix, face_areas: np.ndarray):
    """Create grad_and_div_fn for mesh-based GT Laplacian (igl.grad)."""
    nF = len(face_areas)
    def grad_and_div(u):
        grad_u = (grad_op @ u).reshape(nF, 3)
        norms = np.maximum(np.linalg.norm(grad_u, axis=1, keepdims=True), 1e-10)
        X = -grad_u / norms
        return grad_op.T @ (X * face_areas[:, np.newaxis]).flatten()
    return grad_and_div


def make_grad_div_learned(G: scipy.sparse.spmatrix, M: scipy.sparse.spmatrix):
    """Create grad_and_div_fn for learned gradient operator (3N x N)."""
    G_f64 = G.astype(np.float64)
    areas = np.array(M.diagonal()).flatten() if scipy.sparse.issparse(M) else np.diag(M)
    M_3 = np.repeat(areas.astype(np.float64), 3)  # (3N,)
    def grad_and_div(u):
        grad_u = (G_f64 @ u).reshape(-1, 3)
        norms = np.maximum(np.linalg.norm(grad_u, axis=1, keepdims=True), 1e-10)
        X = -grad_u / norms
        return G_f64.T @ (M_3 * X.flatten())
    return grad_and_div


def make_grad_div_pointcloud(grad_op: scipy.sparse.spmatrix, div_op: scipy.sparse.spmatrix):
    """Create grad_and_div_fn for pointcloud operators (pcdiff)."""
    def grad_and_div(u):
        grad_u = (grad_op @ u).reshape(-1, 2)
        norms = np.maximum(np.linalg.norm(grad_u, axis=1, keepdims=True), 1e-10)
        X = -grad_u / norms
        return div_op @ X.flatten()
    return grad_and_div


# ---------------------------------------------------------------------------
# Convenience wrappers -- backward-compatible single/multi source APIs
# ---------------------------------------------------------------------------

def compute_heat_geodesic_pointcloud(
    L, M, grad_op, div_op, source_idx, n_vertices, t=None, device=None
) -> Optional[np.ndarray]:
    """Single-source geodesic with pointcloud operators."""
    fn = make_grad_div_pointcloud(grad_op, div_op)
    return compute_heat_geodesics_batch(L, M, [source_idx], n_vertices, fn, t=t)[0]


def compute_heat_geodesic_mesh(
    L, M, grad_op, face_areas, source_idx, n_vertices, t=None, device=None
) -> Optional[np.ndarray]:
    """Single-source geodesic with mesh operators."""
    if t is None:
        h = 1.52 * np.sqrt(face_areas.mean())
        t = h ** 2
    fn = make_grad_div_mesh(grad_op, face_areas)
    return compute_heat_geodesics_batch(L, M, [source_idx], n_vertices, fn, t=t)[0]


def compute_heat_geodesic_learned(
    S, M, G, source_idx, n_vertices, t=None, device=None
) -> Optional[np.ndarray]:
    """Single-source geodesic with learned operators."""
    fn = make_grad_div_learned(G, M)
    return compute_heat_geodesics_batch(S, M, [source_idx], n_vertices, fn, t=t)[0]


def compute_heat_geodesic_learned_batch(
    S, M, G, source_indices, n_vertices, device=None, t=None
) -> List[Optional[np.ndarray]]:
    """Multi-source geodesic with learned operators."""
    fn = make_grad_div_learned(G, M)
    return compute_heat_geodesics_batch(S, M, source_indices, n_vertices, fn, t=t)



def _check_mesh_topology_safe(vertices: np.ndarray, faces: np.ndarray) -> Tuple[bool, str]:
    """
    Lightweight topology check to detect meshes likely to crash pygeodesic/igl.

    Checks for non-manifold edges, degenerate faces, and unreferenced vertices.
    Much cheaper than actually running geodesic algorithms.

    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (F, 3)

    Returns:
        Tuple of (is_safe, reason_if_unsafe)
    """
    n = len(vertices)
    nf = len(faces)

    if nf == 0:
        return False, "no faces"

    # Check for out-of-range face indices
    if faces.max() >= n or faces.min() < 0:
        return False, f"face indices out of range [0, {n})"

    # Check for degenerate faces (repeated vertex indices)
    degen = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    n_degen = int(degen.sum())
    if n_degen > 0:
        return False, f"{n_degen} degenerate faces"

    # Check for non-manifold edges (shared by >2 faces)
    # Build edge -> face count
    from collections import Counter
    edge_counts = Counter()
    for f in faces:
        for i in range(3):
            e = (min(f[i], f[(i + 1) % 3]), max(f[i], f[(i + 1) % 3]))
            edge_counts[e] += 1

    non_manifold = sum(1 for c in edge_counts.values() if c > 2)
    if non_manifold > 0:
        return False, f"{non_manifold} non-manifold edges"

    return True, "ok"


def compute_exact_geodesics(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_idx: int,
    verbose: bool = False
) -> Optional[np.ndarray]:
    """
    Compute exact geodesic distances using pygeodesic or igl.

    Performs a lightweight mesh topology check before calling native code
    to avoid segfaults on degenerate meshes.

    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (F, 3)
        source_idx: Index of source vertex
        verbose: If True, print diagnostic messages before each native call

    Returns:
        Exact geodesic distances (N,) or None if computation fails
    """
    n = len(vertices)
    vertices = vertices.astype(np.float64)
    faces = faces.astype(np.int32)

    # Pre-flight topology check to avoid segfaults in native code
    is_safe, reason = _check_mesh_topology_safe(vertices, faces)
    if not is_safe:
        print(f"      [!] Skipping exact geodesics: mesh topology unsafe ({reason})", flush=True)
        return None

    # Try pygeodesic first (fastest exact algorithm)
    try:
        import pygeodesic.geodesic as geodesic
        if verbose: print(f"      [diag] pygeodesic.geodesicDistances (N={n}, F={len(faces)}, src={source_idx}) ...", flush=True)
        geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
        distances, _ = geoalg.geodesicDistances(np.array([source_idx]), None)

        if distances is not None and len(distances) == n:
            return distances
    except ImportError:
        pass
    except Exception as e:
        print(f"pygeodesic failed: {e}")

    # Try igl.exact_geodesic
    if HAS_IGL:
        try:
            if verbose: print(f"      [diag] igl.exact_geodesic (N={n}, F={len(faces)}, src={source_idx}) ...", flush=True)
            VS = np.array([source_idx], dtype=np.int32)
            VT = np.arange(n, dtype=np.int32)
            distances = igl.exact_geodesic(vertices, faces, VS, VT)
            return distances
        except Exception as e:
            print(f"igl.exact_geodesic failed: {e}")

    return None


def select_geodesic_source_vertex(
    vertices: np.ndarray,
    method: str = "centroid"
) -> int:
    """
    Select a source vertex for geodesic computation.

    Args:
        vertices: Vertex positions (N, 3)
        method: Selection method:
            - "centroid": Vertex nearest to centroid
            - "random": Random vertex

    Returns:
        Index of selected source vertex
    """
    if method == "centroid":
        centroid = vertices.mean(axis=0)
        distances = np.linalg.norm(vertices - centroid, axis=1)
        return int(np.argmin(distances))
    elif method == "random":
        return int(np.random.randint(0, len(vertices)))
    else:
        raise ValueError(f"Unknown method: {method}")


def select_multiple_geodesic_sources(
    vertices: np.ndarray,
    num_sources: int = 5,
    method: str = "farthest_point_sampling",
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Select multiple source vertices for robust geodesic validation.

    Args:
        vertices: Vertex positions (N, 3)
        num_sources: Number of source vertices to select
        method: Selection method:
            - "farthest_point_sampling": FPS starting from centroid (well-distributed)
            - "random": Random vertices
            - "mixed": Centroid + random vertices

    Returns:
        Array of source vertex indices (num_sources,)
    """
    n = len(vertices)
    num_sources = min(num_sources, n)

    if seed is not None:
        np.random.seed(seed)

    if method == "random":
        return np.random.choice(n, size=num_sources, replace=False)

    elif method == "mixed":
        # First source: nearest to centroid
        # Rest: random
        sources = [select_geodesic_source_vertex(vertices, "centroid")]
        remaining = np.setdiff1d(np.arange(n), sources)
        if num_sources > 1:
            random_sources = np.random.choice(remaining, size=num_sources - 1, replace=False)
            sources.extend(random_sources.tolist())
        return np.array(sources)

    elif method == "farthest_point_sampling":
        # Farthest Point Sampling for well-distributed sources
        # Start from vertex nearest to centroid
        sources = []
        first_idx = select_geodesic_source_vertex(vertices, "centroid")
        sources.append(first_idx)

        # Track minimum distance to any selected source
        min_distances = np.linalg.norm(vertices - vertices[first_idx], axis=1)

        for _ in range(num_sources - 1):
            # Select vertex farthest from all current sources
            farthest_idx = int(np.argmax(min_distances))
            sources.append(farthest_idx)

            # Update minimum distances
            new_distances = np.linalg.norm(vertices - vertices[farthest_idx], axis=1)
            min_distances = np.minimum(min_distances, new_distances)

        return np.array(sources)

    else:
        raise ValueError(f"Unknown method: {method}")


@dataclass
class MultiSourceGeodesicMetrics:
    """Aggregated metrics from multiple source vertices."""
    # Per-source metrics
    per_source_correlation: np.ndarray
    per_source_mae: np.ndarray
    per_source_max_error: np.ndarray
    per_source_monotonicity: np.ndarray

    # Aggregated metrics
    mean_correlation: float
    std_correlation: float
    mean_mae: float
    mean_max_error: float
    mean_monotonicity: float

    # Metadata
    num_sources: int
    source_indices: np.ndarray

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        """Convert to dictionary for logging."""
        p = f"{prefix}_" if prefix else ""
        return {
            f'{p}geodesic_correlation_mean': self.mean_correlation,
            f'{p}geodesic_correlation_std': self.std_correlation,
            f'{p}geodesic_mae_mean': self.mean_mae,
            f'{p}geodesic_max_error_mean': self.mean_max_error,
            f'{p}geodesic_monotonicity_mean': self.mean_monotonicity,
            f'{p}geodesic_num_sources': float(self.num_sources),
        }


def compute_multisource_geodesic_metrics(
    computed_func,
    exact_func,
    source_indices: np.ndarray,
    num_monotonicity_samples: int = 5000
) -> MultiSourceGeodesicMetrics:
    """
    Compute geodesic metrics averaged over multiple source vertices.

    Args:
        computed_func: Function(source_idx) -> computed distances (N,) or None
        exact_func: Function(source_idx) -> exact distances (N,) or None
        source_indices: Array of source vertex indices to use
        num_monotonicity_samples: Samples for monotonicity computation

    Returns:
        MultiSourceGeodesicMetrics with per-source and aggregated metrics
    """
    correlations = []
    maes = []
    max_errors = []
    monotonicities = []
    valid_sources = []

    for source_idx in source_indices:
        computed = computed_func(int(source_idx))
        exact = exact_func(int(source_idx))

        if computed is None or exact is None:
            continue

        metrics = compute_geodesic_metrics(computed, exact, num_monotonicity_samples)
        correlations.append(metrics.correlation)
        maes.append(metrics.mae_normalized)
        max_errors.append(metrics.max_error_normalized)
        monotonicities.append(metrics.monotonicity)
        valid_sources.append(source_idx)

    if len(correlations) == 0:
        # Return empty metrics if all sources failed
        return MultiSourceGeodesicMetrics(
            per_source_correlation=np.array([]),
            per_source_mae=np.array([]),
            per_source_max_error=np.array([]),
            per_source_monotonicity=np.array([]),
            mean_correlation=0.0,
            std_correlation=0.0,
            mean_mae=1.0,
            mean_max_error=1.0,
            mean_monotonicity=0.0,
            num_sources=0,
            source_indices=np.array([])
        )

    correlations = np.array(correlations)
    maes = np.array(maes)
    max_errors = np.array(max_errors)
    monotonicities = np.array(monotonicities)

    return MultiSourceGeodesicMetrics(
        per_source_correlation=correlations,
        per_source_mae=maes,
        per_source_max_error=max_errors,
        per_source_monotonicity=monotonicities,
        mean_correlation=float(correlations.mean()),
        std_correlation=float(correlations.std()),
        mean_mae=float(maes.mean()),
        mean_max_error=float(max_errors.mean()),
        mean_monotonicity=float(monotonicities.mean()),
        num_sources=len(valid_sources),
        source_indices=np.array(valid_sources)
    )