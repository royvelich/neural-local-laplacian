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
import torch

from neural_local_laplacian.utils.utils import (
    normalize_mesh_vertices,
    assemble_stiffness_and_mass_matrices,
    assemble_gradient_operator,
    build_patches_from_vertices,
)
from neural_local_laplacian.utils.geodesic_utils import (
    heat_method_build,
    heat_factorize,
    heat_solve_all,
    poisson_factorize,
    poisson_solve_all,
    make_grad_div_learned,
    select_multiple_geodesic_sources,
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