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
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


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

    Args:
        distances: Array of distances

    Returns:
        Normalized distances with min=0, max=1
    """
    d_min, d_max = distances.min(), distances.max()
    if d_max - d_min < 1e-10:
        return np.zeros_like(distances)
    return (distances - d_min) / (d_max - d_min)


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


def compute_heat_geodesic_pointcloud(
    L: scipy.sparse.spmatrix,
    M: scipy.sparse.spmatrix,
    grad_op: scipy.sparse.spmatrix,
    div_op: scipy.sparse.spmatrix,
    source_idx: int,
    n_vertices: int,
    t: Optional[float] = None
) -> Optional[np.ndarray]:
    """
    Compute geodesic distances using Heat Method with point cloud operators.

    The Heat Method (Crane et al. 2013):
    1. Solve heat equation: (M + t*L) u = delta_source
    2. Normalize gradient: X = -grad(u) / |grad(u)|
    3. Solve Poisson: L @ phi = div(X)

    Args:
        L: Laplacian matrix (N, N) - should be positive semi-definite
        M: Mass matrix (N, N) - diagonal
        grad_op: Gradient operator (2N, N) from pcdiff
        div_op: Divergence operator (N, 2N) from pcdiff
        source_idx: Index of source vertex
        n_vertices: Number of vertices
        t: Time step (default: auto-computed from mass matrix)

    Returns:
        Geodesic distances (N,) or None if computation fails
    """
    n = n_vertices

    # Ensure float64 for numerical stability
    L = L.astype(np.float64)
    M = M.astype(np.float64)

    # Time step: t = h^2 where h is mean point spacing
    if t is None:
        if scipy.sparse.issparse(M):
            areas = np.array(M.diagonal()).flatten()
        else:
            areas = np.diag(M)
        h = np.sqrt(areas.mean())
        t = h ** 2

    try:
        # Step I: Heat diffusion
        # Solve (M + t*L) u = delta_source
        delta = np.zeros(n)
        delta[source_idx] = 1.0

        A = M + t * L
        u = scipy.sparse.linalg.spsolve(A.tocsc(), delta)

        if not np.all(np.isfinite(u)):
            return None

        # Step II: Compute and normalize gradient (2D in tangent plane)
        grad_u = grad_op @ u  # (2N,) - interleaved [x1, y1, x2, y2, ...]
        grad_u_2d = grad_u.reshape(-1, 2)  # (N, 2)

        # Normalize to unit vectors pointing toward source
        norms = np.linalg.norm(grad_u_2d, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X = -grad_u_2d / norms  # Negative: point toward source

        X_flat = X.flatten()

        # Step III: Divergence and Poisson
        # Try both signs and pick the one where source has minimum distance
        div_X = div_op @ X_flat

        eps = 1e-8
        L_reg = L + eps * scipy.sparse.eye(n)

        phi_pos = scipy.sparse.linalg.spsolve(L_reg.tocsc(), div_X)
        phi_neg = scipy.sparse.linalg.spsolve(L_reg.tocsc(), -div_X)

        # The correct sign should have the source at or near the minimum
        phi_pos_shifted = phi_pos - phi_pos.min()
        phi_neg_shifted = phi_neg - phi_neg.min()

        # Check which one has source closer to minimum
        source_rank_pos = (phi_pos_shifted < phi_pos_shifted[source_idx]).sum()
        source_rank_neg = (phi_neg_shifted < phi_neg_shifted[source_idx]).sum()

        if source_rank_pos < source_rank_neg:
            phi = phi_pos_shifted
        else:
            phi = phi_neg_shifted

        return phi

    except Exception as e:
        print(f"Heat Method (pointcloud) failed: {e}")
        return None


def compute_heat_geodesic_mesh(
    L: scipy.sparse.spmatrix,
    M: scipy.sparse.spmatrix,
    grad_op: scipy.sparse.spmatrix,
    face_areas: np.ndarray,
    source_idx: int,
    n_vertices: int,
    t: Optional[float] = None
) -> Optional[np.ndarray]:
    """
    Compute geodesic distances using Heat Method with mesh-based gradient.

    Uses igl.grad() which produces per-face 3D gradients.

    Args:
        L: Laplacian matrix (N, N) - should be positive semi-definite
        M: Mass matrix (N, N) - diagonal
        grad_op: Gradient operator (3*nF, N) from igl.grad()
        face_areas: Face areas (nF,)
        source_idx: Index of source vertex
        n_vertices: Number of vertices
        t: Time step (default: auto-computed from face areas)

    Returns:
        Geodesic distances (N,) or None if computation fails
    """
    n = n_vertices
    nF = len(face_areas)

    # Ensure float64 for numerical stability
    L = L.astype(np.float64)
    M = M.astype(np.float64)

    # Time step: t = h^2 where h is mean edge length
    # For equilateral triangle: area = sqrt(3)/4 * h^2, so h ≈ 1.52 * sqrt(area)
    if t is None:
        h = 1.52 * np.sqrt(face_areas.mean())
        t = h ** 2

    try:
        # Step I: Heat diffusion
        delta = np.zeros(n)
        delta[source_idx] = 1.0

        A = M + t * L
        u = scipy.sparse.linalg.spsolve(A.tocsc(), delta)

        if not np.all(np.isfinite(u)):
            return None

        # Step II: Compute per-face gradient and normalize
        # grad_op @ u gives (3*nF,) vector: [gx_f0, gy_f0, gz_f0, gx_f1, ...]
        grad_u = grad_op @ u  # (3*nF,)
        grad_u_3d = grad_u.reshape(nF, 3)  # (nF, 3) - 3D gradient per face

        # Normalize each face's gradient to unit vector
        norms = np.linalg.norm(grad_u_3d, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X = -grad_u_3d / norms  # Negative: point toward source

        # Step III: Compute integrated divergence
        # For igl.grad(), the discrete divergence is:
        # div(X) = -G^T @ (A ⊗ I_3) @ X
        # But we need to solve: Δφ = ∇·X, and since L = -Δ, we have Lφ = -∇·X
        # So: Lφ = -div(X) = G^T @ (A ⊗ I_3) @ X

        # Create area-weighted X
        X_weighted = X * face_areas[:, np.newaxis]  # (nF, 3)
        X_weighted_flat = X_weighted.flatten()  # (3*nF,)

        # RHS for Poisson (the negatives cancel)
        rhs = grad_op.T @ X_weighted_flat  # (nV,)

        # Step IV: Solve Poisson
        eps = 1e-8
        L_reg = L + eps * scipy.sparse.eye(n)

        phi = scipy.sparse.linalg.spsolve(L_reg.tocsc(), rhs)

        # Shift so source has distance 0
        phi = phi - phi[source_idx]

        # Shift so minimum is 0 (source should be the minimum)
        phi = phi - phi.min()

        return phi

    except Exception as e:
        print(f"Heat Method (mesh) failed: {e}")
        return None


def compute_exact_geodesics(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_idx: int
) -> Optional[np.ndarray]:
    """
    Compute exact geodesic distances using pygeodesic or igl.

    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (F, 3)
        source_idx: Index of source vertex

    Returns:
        Exact geodesic distances (N,) or None if computation fails
    """
    n = len(vertices)
    vertices = vertices.astype(np.float64)
    faces = faces.astype(np.int32)

    # Try pygeodesic first (fastest exact algorithm)
    try:
        import pygeodesic.geodesic as geodesic
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