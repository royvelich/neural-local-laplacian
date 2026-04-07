"""
Shape correspondence evaluators.

Provides a common interface for evaluating shape correspondence quality
from precomputed eigenbases. Two implementations:

    SpectralNNEvaluator       — GT functional map + NN in spectral space (fast, baseline)
    FunctionalMapEvaluator    — Descriptor-driven fmap optimization + optional ZoomOut (GeomFuM)

Both classes take the same input (eigenbases, GT correspondence, vertices)
and return the same metric dict format, so they can be swapped freely in
training validation loops or standalone evaluation scripts.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


# =============================================================================
# Base class
# =============================================================================

class ShapePairEvaluator(ABC):
    """Abstract base for shape pair correspondence evaluation.

    Subclasses implement :meth:`evaluate`, which takes precomputed eigenbases
    for two shapes and returns a dict of scalar metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used as logging prefix (e.g. 'spectral_nn', 'fmap')."""
        ...

    @abstractmethod
    def evaluate(
        self,
        eigvecs_a: np.ndarray,       # (N_A, k)
        eigvecs_b: np.ndarray,       # (N_B, k)
        eigvals_a: np.ndarray,       # (k,)
        eigvals_b: np.ndarray,       # (k,)
        mass_a: np.ndarray,          # (N_A,)
        mass_b: np.ndarray,          # (N_B,)
        vertices_b: np.ndarray,      # (N_B, 3)
        gt_corr: Optional[np.ndarray] = None,  # (N_A,) or None for identity
        geo_cache = None, # (N_B, N_B) normalised geodesic dists
    ) -> Dict[str, float]:
        """Evaluate correspondence quality.

        Args:
            eigvecs_a: Eigenvectors of shape A, shape (N_A, k).
            eigvecs_b: Eigenvectors of shape B, shape (N_B, k).
            eigvals_a: Eigenvalues of shape A, shape (k,).
            eigvals_b: Eigenvalues of shape B, shape (k,).
            mass_a: Mass diagonal of shape A, shape (N_A,).
            mass_b: Mass diagonal of shape B, shape (N_B,).
            vertices_b: Vertex positions of shape B, shape (N_B, 3).
            gt_corr: Ground-truth correspondence.  gt_corr[i] is the vertex
                index in mesh B corresponding to vertex i in mesh A.
                None means identity (vertex i ↔ vertex i).

        Returns:
            Dict of scalar metrics (accuracy, top-k accuracy, error, etc.).
        """
        ...

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _pointwise_metrics(
        pred_corr: np.ndarray,
        nn_indices: np.ndarray,
        gt_corr: np.ndarray,
        vertices_b: np.ndarray,
        geo_cache = None,
    ) -> Dict[str, float]:
        """Compute standard correspondence metrics from predicted map.

        Args:
            pred_corr: (N_A,) predicted correspondence (top-1).
            nn_indices: (N_A, n_query) top-k NN indices.
            gt_corr: (N_A,) ground-truth correspondence.
            vertices_b: (N_B, 3) vertex positions for error computation.
            geo_cache: GeodesicCache object for geodesic error computation.
                If provided, geodesic error metrics are added.

        Returns:
            Dict with accuracy, top-k accuracy, mean/median error,
            and geodesic metrics when geo_cache is available.
        """
        metrics: Dict[str, float] = {}

        # Top-1 accuracy
        metrics['accuracy'] = float((pred_corr == gt_corr).mean())

        # Top-k accuracy
        for k in (3, 5, 10):
            if k <= nn_indices.shape[1]:
                metrics[f'top{k}_acc'] = float(
                    (nn_indices[:, :k] == gt_corr[:, None]).any(axis=1).mean()
                )

        # Euclidean error normalised by bounding-box diagonal
        errors = np.linalg.norm(
            vertices_b[pred_corr] - vertices_b[gt_corr], axis=1
        )
        bb_diag = np.linalg.norm(vertices_b.max(0) - vertices_b.min(0))
        metrics['mean_error'] = float(errors.mean() / bb_diag)
        metrics['median_error'] = float(np.median(errors) / bb_diag)

        # Geodesic error (Princeton benchmark)
        if geo_cache is not None:
            geo_err = geo_cache.compute_errors(pred_corr, gt_corr)
            metrics['geo_mean_error'] = float(geo_err.mean())
            for thresh in (0.01, 0.05, 0.10, 0.25):
                pct_label = f"geo_at_{int(thresh*100):02d}pct"
                metrics[pct_label] = float((geo_err <= thresh).mean())

        return metrics

    @staticmethod
    def _fmap_quality_metrics(C: np.ndarray) -> Dict[str, float]:
        """Quality metrics for a functional map matrix C.

        Args:
            C: (k, k) functional map matrix.

        Returns:
            Dict with orthogonality error, bijectivity error, etc.
        """
        k = C.shape[0]
        I = np.eye(k)
        diag_energy = np.sum(np.diag(C) ** 2)
        total_energy = np.sum(C ** 2)
        return {
            'ortho_error': float(np.linalg.norm(C.T @ C - I, 'fro')),
            'biject_error': float(np.linalg.norm(C @ C.T - I, 'fro')),
            'corr_error': float(np.linalg.norm(C - I, 'fro')),
            'diag_ratio': float(diag_energy / (total_energy + 1e-10)),
        }

    @staticmethod
    def _fmap_to_pointwise(
        C: np.ndarray,
        eigvecs_a: np.ndarray,
        eigvecs_b: np.ndarray,
        n_query: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert functional map to pointwise correspondence via NN search.

        Args:
            C: (k, k) functional map.
            eigvecs_a: (N_A, k) eigenvectors of shape A.
            eigvecs_b: (N_B, k) eigenvectors of shape B.
            n_query: Number of nearest neighbors to return.

        Returns:
            pred_corr: (N_A,) top-1 correspondence.
            nn_indices: (N_A, n_query) top-k NN indices.
        """
        projected_a = eigvecs_a @ C.T  # (N_A, k)
        n_query = min(n_query, eigvecs_b.shape[0])
        tree = cKDTree(eigvecs_b)
        _, indices = tree.query(projected_a, k=n_query)
        if n_query == 1:
            indices = indices[:, np.newaxis]
        return indices[:, 0], indices


# =============================================================================
# SpectralNNEvaluator — refactored from _correspondence_metrics()
# =============================================================================

class SpectralNNEvaluator(ShapePairEvaluator):
    """Evaluate via ground-truth functional map + NN in spectral space.

    This is the current evaluation pipeline:
    1. Build GT functional map C = Φ_B(T)^T M_A Φ_A
    2. Project A's spectral embedding through C
    3. Find nearest neighbours in B's spectral embedding
    4. Measure accuracy and error against GT correspondence
    """

    @property
    def name(self) -> str:
        return "spectral_nn"

    def evaluate(
        self,
        eigvecs_a: np.ndarray,
        eigvecs_b: np.ndarray,
        eigvals_a: np.ndarray,
        eigvals_b: np.ndarray,
        mass_a: np.ndarray,
        mass_b: np.ndarray,
        vertices_b: np.ndarray,
        gt_corr: Optional[np.ndarray] = None,
        geo_cache = None,
    ) -> Dict[str, float]:
        n_a = eigvecs_a.shape[0]
        if gt_corr is None:
            gt_corr = np.arange(n_a)

        # Build GT functional map: C = Φ_B(T)^T M_A Φ_A
        weighted_phi_b = eigvecs_b[gt_corr[:n_a]] * mass_a[:n_a, None]
        C = weighted_phi_b.T @ eigvecs_a[:n_a]

        # Quality metrics on C itself
        metrics = self._fmap_quality_metrics(C)

        # Pointwise correspondence via NN
        pred_corr, nn_indices = self._fmap_to_pointwise(
            C, eigvecs_a[:n_a], eigvecs_b
        )
        metrics.update(self._pointwise_metrics(
            pred_corr, nn_indices, gt_corr[:n_a], vertices_b,
            geo_cache=geo_cache,
        ))

        return metrics


# =============================================================================
# FunctionalMapEvaluator — GeomFuM-based pipeline
# =============================================================================

class FunctionalMapEvaluator(ShapePairEvaluator):
    """Evaluate via descriptor-driven functional map optimization using GeomFuM.

    Pipeline (all steps use GeomFuM's tested implementations):
    1. Create PointCloud shapes, inject precomputed eigenbasis
    2. Compute spectral descriptors (HKS, WKS) via GeomFuM
    3. Solve for C via descriptor preservation + LB commutativity (forward pass)
    4. Optionally refine with ZoomOut
    5. Convert to pointwise map via NN, measure accuracy

    Requires the ``geomfum`` package (``pip install geomfum``).
    """

    def __init__(
        self,
        descriptors: List[str] = None,
        num_descriptor_scales: int = 100,
        use_zoomout: bool = True,
        zoomout_k_init: int = 20,
        zoomout_k_final: int = 50,
        zoomout_n_iters: int = 10,
        fmap_lmbda: float = 1e3,
        fmap_resolvent_gamma: float = 1.0,
    ):
        """
        Args:
            descriptors: List of descriptor types to use.
                Supported: 'hks', 'wks'. Default: ['hks', 'wks'].
            num_descriptor_scales: Number of time/energy scales per descriptor.
            use_zoomout: Whether to apply ZoomOut refinement after solving C.
            zoomout_k_init: Initial number of eigenvectors for the initial fmap.
            zoomout_k_final: Final number of eigenvectors after ZoomOut.
            zoomout_n_iters: Number of ZoomOut iterations.
            fmap_lmbda: LB commutativity weight for ForwardFunctionalMap.
                Standard value: 1e3 (from GeomFuM notebooks).
            fmap_resolvent_gamma: Resolvent regularisation for ForwardFunctionalMap.
                Standard value: 1.0 (from GeomFuM notebooks).
        """
        if descriptors is None:
            descriptors = ['hks', 'wks']
        self.descriptors = descriptors
        self.num_descriptor_scales = num_descriptor_scales
        self.use_zoomout = use_zoomout
        self.zoomout_k_init = zoomout_k_init
        self.zoomout_k_final = zoomout_k_final
        self.zoomout_n_iters = zoomout_n_iters
        self.fmap_lmbda = fmap_lmbda
        self.fmap_resolvent_gamma = fmap_resolvent_gamma

        # Validate geomfum is available at construction time
        try:
            import geomfum  # noqa: F401
        except ImportError:
            raise ImportError(
                "FunctionalMapEvaluator requires the 'geomfum' package. "
                "Install with: pip install geomfum"
            )

    @property
    def name(self) -> str:
        desc_str = "_".join(sorted(self.descriptors))
        suffix = "_zoomout" if self.use_zoomout else ""
        return f"fmap_{desc_str}{suffix}"

    def _build_shape_with_basis(
        self,
        vertices: np.ndarray,
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
        mass: np.ndarray,
    ):
        """Create a GeomFuM PointCloud and inject our precomputed eigenbasis.

        Also sets the mass matrix on the shape's laplacian to prevent
        GeomFuM from trying to recompute it (which would fail on dummy
        vertices or be redundant for real vertices).

        Args:
            vertices: (N, 3) vertex positions.
            eigvals: (k,) eigenvalues.
            eigvecs: (N, k) eigenvectors.
            mass: (N,) mass diagonal.

        Returns:
            shape: GeomFuM PointCloud with basis and mass matrix set.
        """
        import scipy.sparse
        from geomfum.shape import PointCloud
        from geomfum.basis import LaplaceEigenBasis

        shape = PointCloud(vertices)
        basis = LaplaceEigenBasis(shape, eigvals, eigvecs)
        shape.set_basis(basis)

        # Pre-set the mass matrix so GeomFuM doesn't try to recompute
        # a Laplacian from the vertices (which would fail for dummy vertices
        # and is redundant when we already have the eigenbasis).
        n = vertices.shape[0]
        shape.laplacian._mass_matrix = scipy.sparse.diags(mass)
        shape.laplacian._stiffness_matrix = scipy.sparse.diags(np.zeros(n))

        return shape

    def _compute_descriptors(self, shape):
        """Compute and concatenate descriptors using GeomFuM.

        Args:
            shape: GeomFuM shape with basis set.

        Returns:
            descr: (D, N) descriptor matrix (GeomFuM convention: descriptors × vertices).
        """
        from geomfum.descriptor.spectral import (
            HeatKernelSignature,
            WaveKernelSignature,
        )
        from geomfum.descriptor.pipeline import (
            DescriptorPipeline,
            L2InnerNormalizer,
        )

        steps = []
        for desc_type in self.descriptors:
            if desc_type == 'hks':
                steps.append(HeatKernelSignature.from_registry(
                    scale=True, n_domain=self.num_descriptor_scales))
            elif desc_type == 'wks':
                steps.append(WaveKernelSignature.from_registry(
                    n_domain=self.num_descriptor_scales))
            else:
                raise ValueError(f"Unknown descriptor type: {desc_type}")

        steps.append(L2InnerNormalizer())
        pipeline = DescriptorPipeline(steps)
        return pipeline.apply(shape)

    def evaluate(
        self,
        eigvecs_a: np.ndarray,
        eigvecs_b: np.ndarray,
        eigvals_a: np.ndarray,
        eigvals_b: np.ndarray,
        mass_a: np.ndarray,
        mass_b: np.ndarray,
        vertices_b: np.ndarray,
        gt_corr: Optional[np.ndarray] = None,
        geo_cache = None,
    ) -> Dict[str, float]:
        """Evaluate via GeomFuM functional map pipeline.

        Wrapped in try/except because the neural Laplacian can produce
        eigenvalues that cause numerical overflow in HKS (exp(-λ*t) for
        large λ). Rather than crashing the whole training run, we return
        failure metrics for that pair.
        """
        try:
            return self._evaluate_impl(
                eigvecs_a, eigvecs_b, eigvals_a, eigvals_b,
                mass_a, mass_b, vertices_b, gt_corr, geo_cache,
            )
        except Exception as e:
            print(f"    [FunctionalMapEvaluator] FAILED: {type(e).__name__}: {e}",
                  flush=True)
            # Return empty dict — this pair will be excluded from aggregated stats
            return {}

    def _evaluate_impl(
        self,
        eigvecs_a: np.ndarray,
        eigvecs_b: np.ndarray,
        eigvals_a: np.ndarray,
        eigvals_b: np.ndarray,
        mass_a: np.ndarray,
        mass_b: np.ndarray,
        vertices_b: np.ndarray,
        gt_corr: Optional[np.ndarray] = None,
        geo_cache = None,
    ) -> Dict[str, float]:
        from geomfum.refine import ZoomOut
        from geomfum.forward_functional_map import ForwardFunctionalMap
        from geomfum.convert import P2pFromFmConverter

        n_a = eigvecs_a.shape[0]
        n_b = eigvecs_b.shape[0]
        if gt_corr is None:
            gt_corr = np.arange(n_a)

        k_total = eigvecs_a.shape[1]  # total eigenvectors available

        # --- Step 1: Create GeomFuM shapes with our eigenbasis ---
        # We need dummy vertices for shape_a since evaluate() doesn't receive
        # vertices_a. Descriptors only depend on the eigenbasis, not positions.
        vertices_a_dummy = np.zeros((n_a, 3), dtype=np.float32)

        shape_a = self._build_shape_with_basis(
            vertices_a_dummy, eigvals_a, eigvecs_a, mass_a)
        shape_b = self._build_shape_with_basis(
            vertices_b, eigvals_b, eigvecs_b, mass_b)

        # Set use_k for the initial fmap (smaller basis for solving)
        k_init = self.zoomout_k_init if self.use_zoomout else k_total
        k_init = min(k_init, k_total)
        shape_a.basis.use_k = k_init
        shape_b.basis.use_k = k_init

        # --- Step 2: Compute descriptors via GeomFuM ---
        descr_a = self._compute_descriptors(shape_a)
        descr_b = self._compute_descriptors(shape_b)

        # Check for NaN/Inf in descriptors (e.g. from eigenvalue overflow in HKS)
        descr_a_np = np.array(descr_a)
        descr_b_np = np.array(descr_b)
        if not (np.all(np.isfinite(descr_a_np)) and np.all(np.isfinite(descr_b_np))):
            raise ValueError(
                f"Non-finite descriptors detected "
                f"(eigvals range: A=[{eigvals_a.min():.4f}, {eigvals_a.max():.4f}], "
                f"B=[{eigvals_b.min():.4f}, {eigvals_b.max():.4f}]). "
                f"Eigenvalues may be too large for stable HKS/WKS computation."
            )

        # --- Step 3: Solve for C via ForwardFunctionalMap ---
        ffm = ForwardFunctionalMap(
            lmbda=self.fmap_lmbda,
            resolvent_gamma=self.fmap_resolvent_gamma,
        )
        fmap_matrix, _ = ffm(shape_a, shape_b, descr_a, descr_b)

        # --- Step 4: Optional ZoomOut refinement ---
        if self.use_zoomout:
            k_final = min(self.zoomout_k_final, k_total)
            if k_final > k_init:
                # ZoomOut needs the full basis available
                shape_a.basis.use_k = k_total
                shape_b.basis.use_k = k_total

                step_a = max(1, (k_final - k_init) // self.zoomout_n_iters)
                step_b = step_a  # symmetric step for both shapes
                zoomout = ZoomOut(nit=self.zoomout_n_iters, step=(step_a, step_b))
                fmap_matrix = zoomout(fmap_matrix, shape_a.basis, shape_b.basis)

        # Convert to numpy for metrics computation
        fmap_np = np.array(fmap_matrix)

        # --- Step 5: Compute metrics ---
        k_eval_a = min(fmap_np.shape[1], eigvecs_a.shape[1])
        k_eval_b = min(fmap_np.shape[0], eigvecs_b.shape[1])

        metrics = self._fmap_quality_metrics(fmap_np)

        # Our NN search (returns top-k for top3/5/10 metrics)
        pred_corr, nn_indices = self._fmap_to_pointwise(
            fmap_np, eigvecs_a[:, :k_eval_a], eigvecs_b[:, :k_eval_b]
        )
        metrics.update(self._pointwise_metrics(
            pred_corr, nn_indices, gt_corr[:n_a], vertices_b,
            geo_cache=geo_cache,
        ))

        # GeomFuM's P2pFromFmConverter sanity check.
        # Our _fmap_to_pointwise computes A→B: project A into B's spectral
        # space via Φ_A @ C^T, then NN in B.
        # GeomFuM's P2pFromFmConverter(C, basis_a, basis_b) returns B→A.
        # To get A→B, pass C^T with swapped bases.
        try:
            shape_a.basis.use_k = k_eval_a
            shape_b.basis.use_k = k_eval_b
            p2p_converter = P2pFromFmConverter()
            # fmap_np is (k_b, k_a); transpose to (k_a, k_b), swap bases
            import gsops.backend as gs
            fmap_t = gs.array(fmap_np.T.copy())
            p2p_gfm = np.array(p2p_converter(fmap_t, shape_b.basis, shape_a.basis))
            metrics['geomfum_accuracy'] = float((p2p_gfm == gt_corr[:n_a]).mean())
        except Exception as e:
            print(f"    [GeomFuM p2p sanity check failed: {e}]", flush=True)

        return metrics


# =============================================================================
# Formatting helper
# =============================================================================

def fmt_topk(m: Dict[str, float], prefix: str = '') -> str:
    """Format top-1/3/5/10 accuracy + error as an aligned string.

    Works with any metrics dict produced by a ShapePairEvaluator.
    ``prefix`` is prepended to key lookups (e.g. prefix='sp_').
    """
    acc_key = f'{prefix}accuracy'
    err_key = f'{prefix}mean_error'
    parts = [f"top1={m[acc_key]*100:5.1f}%"]
    for k in (3, 5, 10):
        key = f'{prefix}top{k}_acc'
        if key in m:
            parts.append(f"top{k}={m[key]*100:5.1f}%")
    parts.append(f"Err={m[err_key]:.4f}")
    return "  ".join(parts)