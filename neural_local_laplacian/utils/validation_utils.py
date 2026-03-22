"""
Shared validation step functions for timing and quality evaluation.

Each step is a standalone function: takes inputs, returns (results, timing_dict).
No class state, no side effects. Both timing_sanity_check.py and quantitative_eval.py
import from here so timing is always identical.

Timing convention:
    - All timing values in the returned dicts are in SECONDS.
    - Callers convert to ms for display.
    - GPU synchronization is handled inside each step.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

from neural_local_laplacian.utils.utils import (
    normalize_mesh_vertices,
    assemble_stiffness_and_mass_matrices,
    assemble_gradient_operator,
    build_patches_from_vertices,
    compute_laplacian_eigendecomposition,
)
from neural_local_laplacian.utils.geodesic_utils import (
    heat_method_build,
    heat_factorize,
    heat_solve_all,
    poisson_factorize,
    poisson_solve_all,
    make_grad_div_learned,
    select_multiple_geodesic_sources,
    ensure_psd,
    sparse_factorize,
    compute_geodesic_metrics,
    compute_exact_geodesics,
    normalize_distances,
)


# =============================================================================
# Mesh loading
# =============================================================================

def load_mesh_vertices(mesh_file_path: str) -> np.ndarray:
    """Load and normalize mesh vertices. Returns float32 array."""
    import trimesh
    mesh = trimesh.load(mesh_file_path, process=False, force='mesh')
    vertices = np.array(mesh.vertices, dtype=np.float64)
    vertices = normalize_mesh_vertices(vertices)
    return vertices.astype(np.float32)


# =============================================================================
# Step: PRED k-NN patch extraction
# =============================================================================

def step_pred_knn(
    vertices: torch.Tensor,
    k: int,
    device: torch.device,
) -> Tuple[Any, Dict[str, float]]:
    """
    Extract k-NN patches from vertices.

    Returns:
        (patch_data, timing) where timing has key 'knn'.
    """
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    patch_data = build_patches_from_vertices(vertices, k, device=device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return patch_data, {'knn': elapsed}


# =============================================================================
# Step: PRED model inference (forward + assembly + gradient op)
# =============================================================================

def step_pred_inference(
    model,
    patch_data: Any,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Run model forward pass, assemble L,M matrices and gradient operator.

    Returns:
        (result_dict, timing) where:
        - result_dict has 'L', 'M', 'G' (gradient op, may be None)
        - timing has keys 'forward', 'assembly', 'grad_op'
    """
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    # Forward pass
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                fwd_result = model._forward_pass(patch_data)
        else:
            fwd_result = model._forward_pass(patch_data)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_forward = time.perf_counter() - t0

    # L, M assembly
    stiffness_w = fwd_result['stiffness_weights'].float()
    areas = fwd_result['areas'].float()
    attention_mask = fwd_result['attention_mask']
    vi = patch_data.vertex_indices.to(device)
    ci = patch_data.center_indices.to(device)
    bi = patch_data.patch_idx.to(device)

    t0 = time.perf_counter()
    L, M = assemble_stiffness_and_mass_matrices(
        stiffness_w, areas, attention_mask, vi, ci, bi,
    )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_assembly = time.perf_counter() - t0

    # Gradient operator
    t_grad_op = 0.0
    G = None
    has_grad = getattr(model, '_operator_mode', 'stiffness') == 'gradient'
    if has_grad and fwd_result.get('grad_coeffs') is not None:
        t0 = time.perf_counter()
        G = assemble_gradient_operator(
            grad_coeffs=fwd_result['grad_coeffs'],
            attention_mask=attention_mask,
            vertex_indices=vi,
            center_indices=ci,
            batch_indices=bi,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_grad_op = time.perf_counter() - t0

    result = {'L': L, 'M': M, 'G': G, 'fwd_result': fwd_result}
    timing = {'forward': t_forward, 'assembly': t_assembly, 'grad_op': t_grad_op}
    return result, timing


# =============================================================================
# Step: PRED geodesic (heat method with learned operators)
# =============================================================================

@dataclass
class GeodesicTimingBreakdown:
    """Timing breakdown for heat method geodesics."""
    build: float = 0.0          # Matrix construction (one-time)
    heat_factorize: float = 0.0  # Heat matrix factorization (one-time)
    heat_solve: float = 0.0      # Heat forward-solves + grad/div (per-source)
    poisson_factorize: float = 0.0  # Poisson matrix factorization (one-time)
    poisson_solve: float = 0.0   # Poisson forward-solves (per-source)

    @property
    def onetime(self) -> float:
        """Total one-time cost (build + factorize)."""
        return self.build + self.heat_factorize + self.poisson_factorize

    @property
    def solve(self) -> float:
        """Total per-source cost (all forward-solves)."""
        return self.heat_solve + self.poisson_solve

    @property
    def total(self) -> float:
        """Total wall time."""
        return self.onetime + self.solve


def step_pred_geodesic(
    L: scipy.sparse.spmatrix,
    M: scipy.sparse.spmatrix,
    G: scipy.sparse.spmatrix,
    source_indices: List[int],
    n_vertices: int,
) -> Tuple[List[Optional[np.ndarray]], GeodesicTimingBreakdown]:
    """
    Compute heat method geodesics using learned operators.

    Timing is split into 5 components:
        build, heat_factorize, heat_solve, poisson_factorize, poisson_solve

    Returns:
        (distances_list, timing_breakdown)
    """
    timing = GeodesicTimingBreakdown()
    grad_div_fn = make_grad_div_learned(G, M)

    # 1. Build matrices
    t0 = time.perf_counter()
    matrices = heat_method_build(S=L, M=M, n_vertices=n_vertices, grad_and_div_fn=grad_div_fn)
    timing.build = time.perf_counter() - t0

    # 2. Factorize heat matrix
    t0 = time.perf_counter()
    hf = heat_factorize(matrices)
    timing.heat_factorize = time.perf_counter() - t0

    # 3. Heat forward-solve + grad/div
    t0 = time.perf_counter()
    rhs_list = heat_solve_all(hf, source_indices, matrices)
    timing.heat_solve = time.perf_counter() - t0

    # 4. Factorize Poisson matrix
    t0 = time.perf_counter()
    pf = poisson_factorize(matrices)
    timing.poisson_factorize = time.perf_counter() - t0

    # 5. Poisson forward-solve
    t0 = time.perf_counter()
    distances = poisson_solve_all(pf, source_indices, rhs_list, n_vertices)
    timing.poisson_solve = time.perf_counter() - t0

    return distances, timing


# =============================================================================
# Step: Robust geodesic (pp3d)
# =============================================================================

@dataclass
class RobustGeodesicTiming:
    """Timing breakdown for Robust (pp3d) geodesics."""
    lm_assembly: float = 0.0     # robust_laplacian.point_cloud_laplacian (for non-geodesic tasks)
    constructor: float = 0.0     # pp3d.PointCloudHeatSolver (one-time)
    solve: float = 0.0           # compute_distance × N sources (per-source)

    @property
    def onetime(self) -> float:
        """One-time cost for geodesics (constructor only, NOT lm_assembly)."""
        return self.constructor

    @property
    def total(self) -> float:
        """Total geodesic wall time (constructor + solves)."""
        return self.constructor + self.solve


def step_robust_geodesic(
    vertices: np.ndarray,
    source_indices: List[int],
    robust_k: int = 30,
) -> Tuple[List[Optional[np.ndarray]], RobustGeodesicTiming]:
    """
    Compute geodesics using pp3d PointCloudHeatSolver.

    Also times robust_laplacian assembly (needed for non-geodesic tasks like
    eigendecomposition, Green's function) but NOT included in geodesic E2E.

    Returns:
        (distances_list, timing)
    """
    import robust_laplacian
    import potpourri3d as pp3d

    timing = RobustGeodesicTiming()

    # L,M assembly (for eigen/Green's/HKS — NOT for geodesics)
    t0 = time.perf_counter()
    L_rob, M_rob = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=robust_k)
    timing.lm_assembly = time.perf_counter() - t0

    # pp3d constructor (one-time — builds its own internal L,M + prefactors)
    t0 = time.perf_counter()
    solver = pp3d.PointCloudHeatSolver(vertices)
    timing.constructor = time.perf_counter() - t0

    # Forward-solves (per-source)
    distances = []
    t0 = time.perf_counter()
    for src in source_indices:
        try:
            distances.append(solver.compute_distance(src))
        except Exception:
            distances.append(None)
    timing.solve = time.perf_counter() - t0

    return distances, timing


# =============================================================================
# Step: Load mesh with faces
# =============================================================================

def load_mesh_with_faces(mesh_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load mesh, normalize vertices, return (vertices_f32, faces_i32).

    Raises RuntimeError if mesh has no faces.
    """
    import trimesh
    mesh = trimesh.load(mesh_file_path, process=False, force='mesh')
    vertices = np.array(mesh.vertices, dtype=np.float64)
    vertices = normalize_mesh_vertices(vertices).astype(np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    if len(faces) == 0:
        raise RuntimeError(f"Mesh has no faces: {mesh_file_path}")
    return vertices, faces


# =============================================================================
# Step: GT Laplacian (cotangent) from mesh faces
# =============================================================================

def step_gt_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[Optional[scipy.sparse.spmatrix], Optional[scipy.sparse.spmatrix], Dict[str, float]]:
    """
    Compute GT cotangent Laplacian and mass matrix using igl.

    Args:
        vertices: (N, 3) float64 mesh vertices
        faces: (F, 3) int32 face indices

    Returns:
        (L, M, timing) where timing has key 'assembly'.
        Returns (None, None, timing) if igl is unavailable.
    """
    try:
        import igl
    except ImportError:
        return None, None, {'assembly': 0.0}

    V = vertices.astype(np.float64)
    F = faces.astype(np.int32)

    t0 = time.perf_counter()
    L = igl.cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    elapsed = time.perf_counter() - t0

    return L, M, {'assembly': elapsed}


# =============================================================================
# Step: Robust Laplacian (point cloud)
# =============================================================================

def step_robust_laplacian(
    vertices: np.ndarray,
    k: int = 30,
) -> Tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix, Dict[str, float]]:
    """
    Compute robust Laplacian from point cloud.

    Returns:
        (L, M, timing) where timing has key 'assembly'.
    """
    import robust_laplacian as rl

    t0 = time.perf_counter()
    L, M = rl.point_cloud_laplacian(vertices, n_neighbors=k)
    elapsed = time.perf_counter() - t0

    return L, M, {'assembly': elapsed}


# =============================================================================
# Step: Eigendecomposition
# =============================================================================

def step_eigendecomposition(
    L: scipy.sparse.spmatrix,
    M: scipy.sparse.spmatrix,
    num_eigenvalues: int = 50,
    method: str = "eigsh_sm",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, float]]:
    """
    Compute generalized eigendecomposition L v = λ M v.

    Ensures PSD convention before solving (handles igl NSD cotangent matrices).

    Args:
        L: Sparse Laplacian (N, N)
        M: Sparse mass matrix (N, N)
        num_eigenvalues: Number of smallest eigenpairs to compute
        method:
            "eigsh_sm"  — standard Lanczos with which='SM' (default, fast, no factorization)
            "lobpcg"    — LOBPCG with diagonal preconditioner (fast, no factorization)
            "eigsh"     — shift-invert eigsh with sigma=-0.01 (slow, most robust)

    Returns:
        (eigenvalues, eigenvectors, timing) where timing has key 'eigen'.
        Returns (None, None, timing) on failure.
    """
    t0 = time.perf_counter()
    try:
        L_psd = ensure_psd(L).astype(np.float64)
        M_f64 = M.astype(np.float64)

        if method == "eigsh_sm":
            eigenvalues, eigenvectors = _eigen_eigsh_sm(L_psd, M_f64, num_eigenvalues)
        elif method == "lobpcg":
            eigenvalues, eigenvectors = _eigen_lobpcg(L_psd, M_f64, num_eigenvalues)
        else:
            eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
                L_psd, num_eigenvalues, mass_matrix=M_f64,
            )
    except Exception:
        return None, None, {'eigen': time.perf_counter() - t0}
    elapsed = time.perf_counter() - t0

    return eigenvalues, eigenvectors, {'eigen': elapsed}


def _eigen_eigsh_sm(
    L: scipy.sparse.spmatrix,
    M: scipy.sparse.spmatrix,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k smallest eigenpairs using eigsh with which='SM'.

    Standard Lanczos — no LU factorization, just matrix-vector products.
    Much faster than shift-invert for well-conditioned problems.
    Falls back to shift-invert eigsh if convergence fails.

    Args:
        L: PSD sparse Laplacian (N, N), float64
        M: SPD sparse mass matrix (N, N), float64
        k: Number of eigenpairs

    Returns:
        (eigenvalues, eigenvectors) sorted ascending, shapes (k,) and (N, k).
    """
    try:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            L, k=k, M=M, which='SM',
        )
    except Exception:
        # Fallback to shift-invert
        eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
            L, k, mass_matrix=M,
        )
        return eigenvalues, eigenvectors

    # Sort ascending
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    return eigenvalues, eigenvectors


def _eigen_lobpcg(
    L: scipy.sparse.spmatrix,
    M: scipy.sparse.spmatrix,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k smallest eigenpairs of L v = λ M v using LOBPCG.

    LOBPCG is matrix-free (no LU factorization), so it's much faster than
    shift-invert eigsh for large sparse systems. Uses a diagonal preconditioner
    from L for faster convergence.

    Since L is PSD (null space = constant vector), we add a tiny regularization
    ε·I to make it SPD, then solve. The first eigenvalue will be ≈ ε ≈ 0.

    Falls back to shift-invert eigsh if LOBPCG fails to converge.

    Args:
        L: PSD sparse Laplacian (N, N), float64
        M: SPD sparse mass matrix (N, N), float64
        k: Number of eigenpairs

    Returns:
        (eigenvalues, eigenvectors) sorted ascending, shapes (k,) and (N, k).
    """
    n = L.shape[0]
    eps = 1e-10

    # Regularize: L + εI makes it SPD (removes zero eigenvalue singularity)
    L_reg = L + eps * scipy.sparse.eye(n, dtype=np.float64)

    # Random initial guess — must be (N, k)
    rng = np.random.RandomState(42)
    X0 = rng.randn(n, k).astype(np.float64)

    # Diagonal preconditioner: inv(diag(L_reg)) — cheap but effective
    diag_L = np.array(L_reg.diagonal()).flatten()
    diag_L = np.where(diag_L > 1e-12, diag_L, 1e-12)
    precond = scipy.sparse.diags(1.0 / diag_L)

    try:
        eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
            L_reg, X0, B=M, M=precond,
            tol=1e-8, maxiter=500, largest=False,
        )
    except Exception:
        # Fallback to shift-invert eigsh
        eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
            L, k, mass_matrix=M,
        )
        return eigenvalues, eigenvectors

    # Sort ascending
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Correct for regularization: λ_true ≈ λ_lobpcg - ε (negligible but clean)
    eigenvalues = np.maximum(eigenvalues - eps, 0.0)

    return eigenvalues, eigenvectors


# =============================================================================
# Step: Eigenvalue errors
# =============================================================================

@dataclass
class EigenvalueErrors:
    """Eigenvalue error metrics relative to GT."""
    spectrum_rel_err_mean: float = 0.0   # |pred-gt| / max(gt), mean
    spectrum_rel_err_max: float = 0.0    # |pred-gt| / max(gt), max
    per_eval_rel_err_mean: float = 0.0   # |pred-gt| / |gt| (stable evals), mean
    per_eval_rel_err_max: float = 0.0    # |pred-gt| / |gt| (stable evals), max


def step_eigenvalue_errors(
    method_evals: np.ndarray,
    gt_evals: np.ndarray,
) -> EigenvalueErrors:
    """
    Compute eigenvalue error metrics (method vs GT).

    Skips λ_0 (constant eigenvector). Uses eigenvalues > 1% of max(gt)
    for per-eigenvalue relative error to avoid instability near zero.

    Returns:
        EigenvalueErrors dataclass.
    """
    result = EigenvalueErrors()
    n_compare = min(len(method_evals), len(gt_evals))
    if n_compare <= 1:
        return result

    gt_slice = np.asarray(gt_evals[1:n_compare], dtype=np.float64)
    pred_slice = np.asarray(method_evals[1:n_compare], dtype=np.float64)

    spectral_range = np.abs(gt_slice).max()
    if spectral_range < 1e-10:
        return result

    # Spectrum-relative error
    spectrum_rel = np.abs(pred_slice - gt_slice) / spectral_range
    result.spectrum_rel_err_mean = float(spectrum_rel.mean())
    result.spectrum_rel_err_max = float(spectrum_rel.max())

    # Per-eigenvalue relative error (skip near-zero)
    threshold = spectral_range * 0.01
    stable = np.abs(gt_slice) > threshold
    if stable.sum() > 0:
        rel = np.abs(pred_slice[stable] - gt_slice[stable]) / np.abs(gt_slice[stable])
        result.per_eval_rel_err_mean = float(rel.mean())
        result.per_eval_rel_err_max = float(rel.max())

    return result


# =============================================================================
# Step: Eigenvector cosine similarity
# =============================================================================

def step_eigenvector_cosine_similarity(
    gt_evecs: np.ndarray,
    method_evecs: np.ndarray,
) -> np.ndarray:
    """
    Compute per-eigenvector cosine similarity (handles sign ambiguity).

    For each eigenvector index i, computes |cos(gt_i, method_i)|.

    Args:
        gt_evecs: (N, K_gt) GT eigenvectors
        method_evecs: (N, K_method) method eigenvectors

    Returns:
        cosine_sims: (min(K_gt, K_method),) array of cosine similarities in [0, 1].
    """
    gt = np.asarray(gt_evecs, dtype=np.float64)
    pred = np.asarray(method_evecs, dtype=np.float64)
    n_compare = min(gt.shape[1], pred.shape[1])

    sims = np.zeros(n_compare)
    for i in range(n_compare):
        g = gt[:, i]
        p = pred[:, i]
        gn = np.linalg.norm(g)
        pn = np.linalg.norm(p)
        if gn > 1e-15 and pn > 1e-15:
            sims[i] = abs(float(np.dot(g, p) / (gn * pn)))
    return sims


# =============================================================================
# Step: Green's function (batch, multi-source)
# =============================================================================

@dataclass
class GreensResult:
    """Aggregated Green's function quality metrics across sources."""
    max_principle_pass_rate: float = 0.0   # fraction of sources where max is at source
    mean_corr_with_gt: float = 0.0
    std_corr_with_gt: float = 0.0
    mean_residual_norm: float = 0.0
    num_sources_ok: int = 0
    # Per-source Green's function values (source_idx -> normalized array)
    values: Dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class GreensTiming:
    """Timing breakdown for Green's function computation."""
    factorize: float = 0.0
    solve: float = 0.0

    @property
    def total(self) -> float:
        return self.factorize + self.solve


def step_greens_function(
    L: scipy.sparse.spmatrix,
    M: scipy.sparse.spmatrix,
    source_indices: List[int],
    gt_greens_values: Optional[Dict[int, np.ndarray]] = None,
    regularization: float = 1e-6,
) -> Tuple[GreensResult, GreensTiming]:
    """
    Compute Green's function for multiple sources with one factorization.

    Solves (L + μM)g = δ_s for each source, normalizes, checks maximum principle,
    and optionally computes correlation with GT.

    Args:
        L: Sparse Laplacian (N, N)
        M: Sparse mass matrix (N, N)
        source_indices: Source vertex indices
        gt_greens_values: Optional dict source_idx -> GT Green's values for correlation
        regularization: Base regularization (scaled adaptively by L's diagonal)

    Returns:
        (GreensResult, GreensTiming)
    """
    timing = GreensTiming()
    result = GreensResult()

    n = L.shape[0]
    L_f64 = ensure_psd(L.astype(np.float64))
    M_f64 = M.astype(np.float64)

    # Adaptive regularization
    diag = np.array(L_f64.diagonal()).flatten()
    L_scale = np.abs(diag).mean() if len(diag) > 0 else 1.0
    adaptive_reg = regularization * max(L_scale, 1e-4)

    A = L_f64 + adaptive_reg * M_f64

    # Factorize once
    t0 = time.perf_counter()
    try:
        factor = sparse_factorize(A)
    except Exception:
        return result, timing
    timing.factorize = time.perf_counter() - t0

    M_diag = np.array(M_f64.diagonal()).flatten()
    total_mass = M_diag.sum()

    max_principle_passes = []
    corrs_with_gt = []
    residual_norms = []

    t0 = time.perf_counter()
    for source_idx in source_indices:
        try:
            delta = np.zeros(n, dtype=np.float64)
            delta[source_idx] = 1.0
            g_raw = factor.solve(delta)

            if not np.isfinite(g_raw).all():
                continue

            # Residual before normalization
            Lg = L_f64 @ g_raw
            residual_norm = float(np.linalg.norm(Lg - delta) / max(np.linalg.norm(delta), 1e-15))
            residual_norms.append(residual_norm)

            # Max principle check on raw solution (argmax invariant to normalization)
            max_principle_passes.append(int(np.argmax(g_raw)) == source_idx)

            # Normalize: remove weighted mean, shift min=0, scale max=1
            if total_mass > 0:
                g = g_raw - (g_raw * M_diag).sum() / total_mass
            else:
                g = g_raw - g_raw.mean()
            g = g - g.min()
            if g.max() > 0:
                g = g / g.max()

            result.values[source_idx] = g

            # Correlation with GT
            if gt_greens_values is not None and source_idx in gt_greens_values:
                gt_g = gt_greens_values[source_idx]
                if len(gt_g) == n:
                    corr = np.corrcoef(g, gt_g)[0, 1]
                    if np.isfinite(corr):
                        corrs_with_gt.append(float(corr))

        except Exception:
            continue
    timing.solve = time.perf_counter() - t0

    result.num_sources_ok = len(max_principle_passes)
    if max_principle_passes:
        result.max_principle_pass_rate = float(np.mean(max_principle_passes))
    if corrs_with_gt:
        result.mean_corr_with_gt = float(np.mean(corrs_with_gt))
        result.std_corr_with_gt = float(np.std(corrs_with_gt))
    if residual_norms:
        result.mean_residual_norm = float(np.mean(residual_norms))

    return result, timing


# =============================================================================
# Step: Geodesic quality (compare distances vs exact)
# =============================================================================

@dataclass
class GeodesicQuality:
    """Aggregated geodesic quality metrics across sources."""
    corr_mean: float = 0.0
    corr_std: float = 0.0
    mae_mean: float = 0.0
    mae_std: float = 0.0
    max_err_mean: float = 0.0
    mono_mean: float = 0.0
    mono_std: float = 0.0
    num_sources_ok: int = 0


def step_geodesic_quality(
    computed_distances: List[Optional[np.ndarray]],
    source_indices: List[int],
    vertices: np.ndarray,
    faces: np.ndarray,
) -> GeodesicQuality:
    """
    Compare computed geodesic distances against exact geodesics.

    Computes exact geodesics on-the-fly for each source, then aggregates
    correlation, MAE, max error, and monotonicity across sources.

    Args:
        computed_distances: List of distance arrays (one per source), may contain None
        source_indices: Source vertex indices
        vertices: (N, 3) mesh vertices (float64)
        faces: (F, 3) mesh faces (int32)

    Returns:
        GeodesicQuality dataclass with aggregated metrics.
    """
    result = GeodesicQuality()
    corrs, maes, max_errs, monos = [], [], [], []

    for i, src_idx in enumerate(source_indices):
        computed = computed_distances[i] if i < len(computed_distances) else None
        if computed is None:
            continue

        exact = compute_exact_geodesics(
            vertices.astype(np.float64), faces.astype(np.int32), int(src_idx),
        )
        if exact is None:
            continue

        metrics = compute_geodesic_metrics(computed, exact, num_monotonicity_samples=5000)
        corrs.append(metrics.correlation)
        maes.append(metrics.mae_normalized)
        max_errs.append(metrics.max_error_normalized)
        monos.append(metrics.monotonicity)

    result.num_sources_ok = len(corrs)
    if corrs:
        result.corr_mean = float(np.mean(corrs))
        result.corr_std = float(np.std(corrs))
        result.mae_mean = float(np.mean(maes))
        result.mae_std = float(np.std(maes))
        result.max_err_mean = float(np.mean(max_errs))
        result.mono_mean = float(np.mean(monos))
        result.mono_std = float(np.std(monos))

    return result


# =============================================================================
# Step: Probe function MSE (NeLo Table 1 metric)
# =============================================================================

@dataclass
class ProbeResult:
    """Probe function MSE results for one method."""
    mse: float = 0.0              # Raw MSE, clipped at 1.0 per probe
    cosine_similarity: float = 0.0  # Mean cosine similarity of Lf responses
    failure_rate: float = 0.0     # Fraction of probes with MSE > 1.0
    n_probes: int = 0


def step_probe_function_mse(
    L: scipy.sparse.spmatrix,
    M: scipy.sparse.spmatrix,
    L_gt: scipy.sparse.spmatrix,
    M_gt: scipy.sparse.spmatrix,
    vertices: np.ndarray,
    gt_evals: np.ndarray,
    gt_evecs: np.ndarray,
) -> ProbeResult:
    """
    Compute Lf probe function MSE (NeLo paper Table 1 metric).

    Uses 112 probe functions: 64 spectral + 42 trig + 6 polynomial.
    Reports mean ||M^{-1}Lf - M_gt^{-1}L_gt f||^2 (clipped at 1.0) and
    mean cosine similarity.

    Args:
        L, M: Method's Laplacian and mass matrix
        L_gt, M_gt: GT Laplacian and mass matrix
        vertices: (N, 3) mesh vertices (float64)
        gt_evals: GT eigenvalues
        gt_evecs: GT eigenvectors (N, K)

    Returns:
        ProbeResult dataclass.
    """
    result = ProbeResult()
    vertices = vertices.astype(np.float64)
    n = len(vertices)

    # Build probe functions
    probes = []

    # 1) Spectral probes: first 64 non-constant eigenvectors, scaled by 1/(λ+0.1)
    n_spectral = min(64, len(gt_evals) - 1)
    for i in range(1, n_spectral + 1):
        f = gt_evecs[:, i] / (gt_evals[i] + 0.1)
        probes.append(f)

    # 2) Trigonometric probes
    coords = [vertices[:, 0], vertices[:, 1], vertices[:, 2]]
    for k_val in [1, 2, 4, 8, 16, 32, 64]:
        for coord in coords:
            for phi in [0.0, np.pi / 2.0]:
                probes.append(np.sin(k_val * coord + phi) / (2.0 * k_val))

    # 3) Polynomial probes
    for d in range(3):
        probes.append(vertices[:, d])
    for d in range(3):
        probes.append(vertices[:, d] ** 2)

    probes = np.column_stack(probes)  # (N, n_probes)
    n_probes = probes.shape[1]
    result.n_probes = n_probes

    # GT reference: M_gt^{-1} L_gt f
    M_gt_diag = np.array(M_gt.diagonal()).flatten()
    M_gt_inv = np.where(M_gt_diag > 1e-15, 1.0 / M_gt_diag, 0.0)
    gt_result = M_gt_inv[:, None] * (L_gt @ probes)

    # Method: M^{-1} L f
    try:
        M_diag = np.array(M.diagonal()).flatten()
        M_inv = np.where(M_diag > 1e-15, 1.0 / M_diag, 0.0)
        method_result = M_inv[:, None] * (L @ probes)
    except Exception:
        return result

    diff = method_result - gt_result

    # Raw MSE (clipped at 1.0)
    per_probe_mse = np.mean(diff ** 2, axis=0)
    clipped = np.minimum(per_probe_mse, 1.0)
    result.mse = float(clipped.mean())
    result.failure_rate = float(np.mean(per_probe_mse > 1.0))

    # Cosine similarity
    cosines = []
    for p in range(n_probes):
        gn = np.linalg.norm(gt_result[:, p])
        mn = np.linalg.norm(method_result[:, p])
        if gn > 1e-15 and mn > 1e-15:
            cosines.append(float(np.dot(gt_result[:, p], method_result[:, p]) / (gn * mn)))
    result.cosine_similarity = float(np.mean(cosines)) if cosines else 0.0

    return result


# =============================================================================
# Step: HKS / WKS descriptor comparison
# =============================================================================

@dataclass
class DescriptorResult:
    """HKS and WKS descriptor comparison results."""
    hks_l2_error: float = -1.0     # Normalized L2 error vs GT (-1 = N/A)
    hks_correlation: float = 0.0   # Mean per-scale Pearson correlation with GT
    wks_l2_error: float = -1.0
    wks_correlation: float = 0.0


def step_descriptor_comparison(
    method_evals: np.ndarray,
    method_evecs: np.ndarray,
    gt_evals: np.ndarray,
    gt_evecs: np.ndarray,
    n_scales: int = 100,
    max_eigenvectors: int = 50,
) -> DescriptorResult:
    """
    Compute HKS and WKS descriptor error vs GT.

    Time/energy scales are derived from GT eigenvalues for fair comparison.

    Args:
        method_evals, method_evecs: Method's eigenpairs
        gt_evals, gt_evecs: GT eigenpairs
        n_scales: Number of descriptor scales
        max_eigenvectors: Max eigenvectors to use

    Returns:
        DescriptorResult dataclass.
    """
    result = DescriptorResult()

    def _compute_hks(evals, evecs, t_values, max_k):
        evals_pos = evals[1:]
        evecs_pos = evecs[:, 1:]
        mask = evals_pos > 1e-10
        evals_pos, evecs_pos = evals_pos[mask], evecs_pos[:, mask]
        k = min(len(evals_pos), max_k)
        if k < 2:
            return None
        evals_pos, evecs_pos = evals_pos[:k], evecs_pos[:, :k]
        phi_sq = evecs_pos ** 2
        exp_terms = np.exp(-evals_pos[None, :] * t_values[:, None])
        return phi_sq @ exp_terms.T

    def _compute_wks(evals, evecs, e_values, sigma, max_k):
        evals_pos = evals[1:]
        evecs_pos = evecs[:, 1:]
        mask = evals_pos > 1e-10
        evals_pos, evecs_pos = evals_pos[mask], evecs_pos[:, mask]
        k = min(len(evals_pos), max_k)
        if k < 2:
            return None
        evals_pos, evecs_pos = evals_pos[:k], evecs_pos[:, :k]
        log_evals = np.log(evals_pos)
        phi_sq = evecs_pos ** 2
        weights = np.exp(-((e_values[:, None] - log_evals[None, :]) ** 2) / (2 * sigma ** 2))
        return phi_sq @ weights.T

    def _compare(desc_m, desc_gt):
        if desc_m is None or desc_gt is None:
            return None, None
        l2s, corrs = [], []
        for s in range(desc_gt.shape[1]):
            gn = np.linalg.norm(desc_gt[:, s])
            if gn < 1e-15:
                continue
            l2s.append(np.linalg.norm(desc_m[:, s] - desc_gt[:, s]) / gn)
            if np.std(desc_gt[:, s]) > 1e-15 and np.std(desc_m[:, s]) > 1e-15:
                corrs.append(float(np.corrcoef(desc_m[:, s], desc_gt[:, s])[0, 1]))
        mean_l2 = float(np.mean(l2s)) if l2s else -1.0
        mean_corr = float(np.mean(corrs)) if corrs else 0.0
        return mean_l2, mean_corr

    # Prepare GT scales
    gt_evals_pos = gt_evals[1:]
    gt_evals_pos = gt_evals_pos[gt_evals_pos > 1e-10]
    if len(gt_evals_pos) < 2:
        return result

    # HKS time scales
    t_min = 4.0 * np.log(10.0) / gt_evals_pos[min(len(gt_evals_pos) - 1, max_eigenvectors - 1)]
    t_max = 4.0 * np.log(10.0) / gt_evals_pos[0]
    hks_t = np.exp(np.linspace(np.log(t_min), np.log(t_max), n_scales))

    # WKS energy scales
    log_gt = np.log(gt_evals_pos[:min(len(gt_evals_pos), max_eigenvectors)])
    wks_sigma = 7.0 * (log_gt[-1] - log_gt[0]) / n_scales
    wks_e = np.linspace(log_gt[0], log_gt[-1], n_scales) if wks_sigma > 1e-10 else None

    # GT descriptors
    hks_gt = _compute_hks(gt_evals, gt_evecs, hks_t, max_eigenvectors)
    wks_gt = _compute_wks(gt_evals, gt_evecs, wks_e, wks_sigma, max_eigenvectors) if wks_e is not None else None

    # Method descriptors (using GT scales for fair comparison)
    m_evals = np.asarray(method_evals, dtype=np.float64)
    m_evecs = np.asarray(method_evecs, dtype=np.float64)
    hks_m = _compute_hks(m_evals, m_evecs, hks_t, max_eigenvectors)
    wks_m = _compute_wks(m_evals, m_evecs, wks_e, wks_sigma, max_eigenvectors) if wks_e is not None else None

    hks_l2, hks_corr = _compare(hks_m, hks_gt)
    wks_l2, wks_corr = _compare(wks_m, wks_gt)

    if hks_l2 is not None:
        result.hks_l2_error = hks_l2
        result.hks_correlation = hks_corr
    if wks_l2 is not None:
        result.wks_l2_error = wks_l2
        result.wks_correlation = wks_corr

    return result


# =============================================================================
# Step: Spectral compression error
# =============================================================================

@dataclass
class CompressionResult:
    """Per-k spectral compression errors."""
    # k -> {'mean_l2': float, 'max_l2': float}
    errors: Dict[int, Dict[str, float]] = field(default_factory=dict)


def step_spectral_compression(
    eigenvectors: np.ndarray,
    M: scipy.sparse.spmatrix,
    vertices: np.ndarray,
    k_values: List[int] = None,
) -> CompressionResult:
    """
    Compute spectral mesh compression error at key eigenvector counts.

    Projects original vertices onto k eigenvectors and measures
    per-vertex L2 reconstruction error.

    Args:
        eigenvectors: (N, K) eigenvectors
        M: Mass matrix (N, N) for computing spectral coefficients
        vertices: (N, 3) original mesh vertices
        k_values: List of eigenvector counts to test (default [5, 10, 20, 50])

    Returns:
        CompressionResult with per-k errors.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]

    result = CompressionResult()
    evecs = np.asarray(eigenvectors, dtype=np.float64)
    verts = np.asarray(vertices, dtype=np.float64)

    if M is not None:
        M_diag = np.array(M.diagonal()).flatten() if scipy.sparse.issparse(M) else np.diag(M)
    else:
        M_diag = np.ones(len(verts))

    max_k = min(evecs.shape[1], max(k_values))

    # Coefficients: φᵀ M x for each coordinate
    M_verts = M_diag[:, None] * verts
    coeffs = evecs[:, :max_k].T @ M_verts  # (max_k, 3)

    for k in k_values:
        if k > evecs.shape[1]:
            continue
        recon = evecs[:, :k] @ coeffs[:k]
        per_vertex_err = np.linalg.norm(verts - recon, axis=1)
        result.errors[k] = {
            'mean_l2': float(per_vertex_err.mean()),
            'max_l2': float(per_vertex_err.max()),
        }

    return result