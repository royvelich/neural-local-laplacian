"""
Shared evaluation utilities for functional map correspondence.

Extracted from functional_map_module.py so that both the supervised
Laplacian training (LaplacianTransformerModule) and the fmap fine-tuning
(FunctionalMapModule) can run the same validation pipeline without
code duplication.

All functions here are model-agnostic: they accept a generic nn.Module
that returns a dict with 'grad_coeffs' and 'areas' keys.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Batch

from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData
from neural_local_laplacian.utils.utils import compute_laplacian_eigendecomposition
from neural_local_laplacian.utils.laplacian_assembly import (
    LaplacianConfig,
    assemble_laplacian,
)


# =============================================================================
# kNN + patch building
# =============================================================================

def compute_knn(vertices_np: np.ndarray, k: int) -> np.ndarray:
    """Compute k-nearest neighbors excluding self. Returns (N, k) indices."""
    n = len(vertices_np)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(vertices_np)
    _, indices = nbrs.kneighbors(vertices_np)
    center = np.arange(n)[:, np.newaxis]
    keep = ~(indices == center)
    keep_pos = np.cumsum(keep, axis=1)
    final = (keep_pos <= k) & keep
    return indices[final].reshape(n, k)


def build_patch_data(
    vertices_t: torch.Tensor, knn: np.ndarray, device: torch.device,
) -> MeshPatchData:
    """Build MeshPatchData from vertices and precomputed kNN indices."""
    N = vertices_t.shape[0]
    k = knn.shape[1]
    knn_t = torch.from_numpy(knn).long().to(device)
    patch_pos = vertices_t[knn_t] - vertices_t[:, None, :]  # (N, k, 3)
    all_pos = patch_pos.reshape(-1, 3)
    return MeshPatchData(
        pos=all_pos,
        x=all_pos,
        patch_idx=torch.arange(N, device=device).repeat_interleave(k),
        vertex_indices=knn_t.flatten(),
        center_indices=torch.arange(N, device=device),
    )


# =============================================================================
# Ground-truth correspondence helpers
# =============================================================================

def build_gt_corr_from_pair(pair) -> Optional[np.ndarray]:
    """Build a dense (N_A,) GT correspondence array from a PairSample.

    Returns None if the correspondence is identity.
    """
    n_a = len(pair.verts_a)
    if (len(pair.corr_a) == n_a
            and np.array_equal(pair.corr_a, np.arange(n_a))
            and np.array_equal(pair.corr_b, np.arange(n_a))):
        return None
    gt_corr = np.full(n_a, -1, dtype=np.int64)
    gt_corr[pair.corr_a] = pair.corr_b
    unmapped = gt_corr == -1
    if unmapped.any() and (~unmapped).any():
        nbrs = NearestNeighbors(n_neighbors=1).fit(pair.verts_a[~unmapped])
        _, idx = nbrs.kneighbors(pair.verts_a[unmapped])
        mapped_indices = np.where(~unmapped)[0]
        gt_corr[unmapped] = gt_corr[mapped_indices[idx.flatten()]]
    return gt_corr


# =============================================================================
# Geodesic cache
# =============================================================================

class GeodesicCache:
    """Lazy geodesic distance computation on mesh B.

    Factorizes the Laplacian once (fast), then computes geodesic distances
    from individual source vertices on demand. Caches distance vectors so
    the same GT target vertex is never solved twice.

    Usage in _pointwise_metrics:
        geo_errors = geo_cache.compute_errors(pred_corr, gt_corr)
    """

    def __init__(self, pair):
        import potpourri3d as pp3d

        verts_full = pair._verts_b_full if pair._verts_b_full is not None else pair.verts_b
        faces = pair.faces_b
        self._idx_b = pair._idx_b if pair._idx_b is not None else np.arange(len(pair.verts_b))

        # Factorize Laplacian (one-time cost, fast)
        self._solver = pp3d.MeshHeatMethodDistanceSolver(verts_full, faces)

        # Normalisation: sqrt(surface area)
        v0, v1, v2 = verts_full[faces[:, 0]], verts_full[faces[:, 1]], verts_full[faces[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
        self._sqrt_area = np.sqrt(max(area, 1e-12))

        # Cache: subsampled B index -> normalised distance to all subsampled B vertices
        self._dist_cache: Dict[int, np.ndarray] = {}

    @classmethod
    def from_precomputed(
        cls, dist_cache: Dict[int, np.ndarray],
        sqrt_area: float, idx_b: np.ndarray,
    ) -> 'GeodesicCache':
        """Create a cache from precomputed distances (no solver needed)."""
        obj = object.__new__(cls)
        obj._dist_cache = dist_cache
        obj._sqrt_area = sqrt_area
        obj._idx_b = idx_b
        obj._solver = None
        return obj

    def _get_dists_from(self, sub_idx: int) -> np.ndarray:
        """Get normalised geodesic distances from subsampled vertex sub_idx to all others."""
        if sub_idx not in self._dist_cache:
            if self._solver is None:
                raise RuntimeError(
                    f"GeodesicCache: no solver and vertex {sub_idx} not precomputed")
            full_idx = int(self._idx_b[sub_idx])
            full_dists = self._solver.compute_distance(full_idx)  # (N_full,)
            self._dist_cache[sub_idx] = full_dists[self._idx_b] / self._sqrt_area
        return self._dist_cache[sub_idx]

    def precompute_from_gt(self, gt_corr: np.ndarray) -> None:
        """Eagerly compute and cache distances from all unique GT targets."""
        unique_gt = np.unique(gt_corr)
        for g in unique_gt:
            self._get_dists_from(int(g))

    def drop_solver(self) -> None:
        """Release the pp3d solver to free memory (after precomputation)."""
        self._solver = None

    def save_to_disk(self, path: str) -> None:
        """Save cache to disk as .npz file."""
        save_dict = {
            'sqrt_area': np.array(self._sqrt_area),
            'idx_b': self._idx_b,
        }
        for k, v in self._dist_cache.items():
            save_dict[f'dist_{k}'] = v
        np.savez_compressed(path, **save_dict)

    @classmethod
    def load_from_disk(cls, path: str) -> 'GeodesicCache':
        """Load cache from .npz file."""
        data = np.load(path)
        sqrt_area = float(data['sqrt_area'])
        idx_b = data['idx_b']
        dist_cache = {}
        for k in data.files:
            if k.startswith('dist_'):
                target_idx = int(k[5:])
                dist_cache[target_idx] = data[k]
        return cls.from_precomputed(dist_cache, sqrt_area, idx_b)

    def compute_errors(
        self, pred_corr: np.ndarray, gt_corr: np.ndarray,
    ) -> np.ndarray:
        """Compute normalised geodesic error for each (pred, gt) pair.

        Args:
            pred_corr: (N_A,) predicted B vertex indices.
            gt_corr: (N_A,) ground-truth B vertex indices.

        Returns:
            (N_A,) normalised geodesic errors.
        """
        unique_gt = np.unique(gt_corr)
        for g in unique_gt:
            self._get_dists_from(int(g))
        errors = np.empty(len(pred_corr), dtype=np.float32)
        for g in unique_gt:
            mask = gt_corr == g
            errors[mask] = self._dist_cache[int(g)][pred_corr[mask]]
        return errors


def geo_cache_path(cache_dir: str, pair_name: str, max_vertices: int) -> Path:
    """Build the disk path for a geo cache file."""
    mv_label = f"mv{max_vertices}" if max_vertices > 0 else "mv0"
    safe_name = pair_name.replace(":", "_").replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / mv_label / f"{safe_name}.npz"


def precompute_geo_cache_worker(args):
    """Multiprocessing worker: build solver, compute distances, return cache data.

    Runs in a separate process. The solver is discarded after computation —
    only the distance dict (numpy arrays) is returned.

    Returns:
        (name, dist_cache, sqrt_area, idx_b) or (name, None, None, None) on failure.
    """
    import potpourri3d as pp3d

    name, verts_full, faces, idx_b, unique_targets = args
    try:
        solver = pp3d.MeshHeatMethodDistanceSolver(verts_full, faces)

        v0, v1, v2 = verts_full[faces[:, 0]], verts_full[faces[:, 1]], verts_full[faces[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
        sqrt_area = np.sqrt(max(area, 1e-12))

        dist_cache = {}
        for g in unique_targets:
            full_idx = int(idx_b[g])
            full_dists = solver.compute_distance(full_idx)
            dist_cache[int(g)] = full_dists[idx_b] / sqrt_area

        return name, dist_cache, sqrt_area, idx_b
    except Exception as e:
        return name, None, None, None


def build_geo_cache(pair) -> Optional[GeodesicCache]:
    """Build a GeodesicCache for pair, or None if faces/potpourri3d unavailable."""
    if pair.faces_b is None:
        return None
    try:
        return GeodesicCache(pair)
    except ImportError:
        print("    [geodesic] potpourri3d not installed — skipping geodesic metrics",
              flush=True)
        return None
    except Exception as e:
        print(f"    [geodesic] failed to build cache: {e}", flush=True)
        return None


# =============================================================================
# Legacy correspondence metrics (kept for backward compatibility)
# =============================================================================

def correspondence_metrics(
    eigvecs_a: np.ndarray, eigvecs_b: np.ndarray,
    mass_a: np.ndarray, mass_b: np.ndarray,
    vertices_b: np.ndarray, n: int,
    gt_corr: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Functional map metrics from eigenbases."""
    n_a = eigvecs_a.shape[0]
    weighted_phi_b = (eigvecs_b[gt_corr[:n_a]] * mass_a[:n_a, None]
                      if gt_corr is not None
                      else eigvecs_b[:n_a] * mass_a[:n_a, None])
    C = weighted_phi_b.T @ eigvecs_a[:n_a]
    k_fm = C.shape[0]
    I = np.eye(k_fm)
    metrics = {
        'ortho_error':  float(np.linalg.norm(C.T @ C - I, 'fro')),
        'biject_error': float(np.linalg.norm(C @ C.T - I, 'fro')),
        'corr_error':   float(np.linalg.norm(C - I, 'fro')),
        'diag_ratio':   float(np.sum(np.diag(C) ** 2) / (np.sum(C ** 2) + 1e-10)),
    }
    projected_a = eigvecs_a[:n_a] @ C.T
    nbrs = NearestNeighbors(
        n_neighbors=min(10, eigvecs_b.shape[0]), algorithm='auto',
    ).fit(eigvecs_b)
    _, indices = nbrs.kneighbors(projected_a)
    pred_corr = indices[:, 0]
    gt = gt_corr[:n_a] if gt_corr is not None else np.arange(n_a)
    metrics['accuracy'] = float((pred_corr == gt).mean())
    for k in (3, 5, 10):
        if k <= indices.shape[1]:
            metrics[f'top{k}_acc'] = float(
                (indices[:, :k] == gt[:, None]).any(axis=1).mean())
    errors = np.linalg.norm(vertices_b[pred_corr] - vertices_b[gt], axis=1)
    bb_diag = np.linalg.norm(vertices_b.max(0) - vertices_b.min(0))
    metrics['mean_error'] = float(errors.mean() / bb_diag)
    metrics['median_error'] = float(np.median(errors) / bb_diag)
    return metrics


# =============================================================================
# Core evaluation functions
# =============================================================================


def _eigenvalue_summary(evals: np.ndarray, prefix: str) -> Dict[str, float]:
    """Summarise an eigenvalue array into a few loggable scalars.

    Args:
        evals: (k,) sorted eigenvalues from eigendecomposition.
        prefix: e.g. 'eval_A' → keys like 'eval_A/lambda_01', 'eval_A/lambda_max'.
    """
    stats: Dict[str, float] = {}
    n = len(evals)
    for idx in [0, 4, 9, 19, 49, 99]:
        if idx < n:
            stats[f'{prefix}/lambda_{idx + 1:02d}'] = float(evals[idx])
    stats[f'{prefix}/lambda_max'] = float(evals[-1])
    stats[f'{prefix}/lambda_mean'] = float(evals.mean())
    # Ratio of last to first non-trivial (spectral spread)
    if n > 1 and abs(evals[0]) > 1e-12:
        stats[f'{prefix}/lambda_ratio_max_01'] = float(evals[-1] / evals[0])
    return stats


@torch.no_grad()
def evaluate_pair(
    model: nn.Module,
    pair,
    k: int,
    num_eigenvectors: int,
    device: torch.device,
    laplacian_configs: Optional[List[LaplacianConfig]] = None,
    evaluators: Optional[List] = None,
    geo_cache: Optional[GeodesicCache] = None,
    verbose_timing: bool = False,
) -> Dict[str, float]:
    """Evaluate correspondence quality using functional maps (non-differentiable).

    Model-agnostic: only requires model(batch) to return a dict with
    'grad_coeffs' and 'areas' keys.

    Args:
        model: Neural Laplacian model.
        pair: PairSample with verts_a, verts_b, corr_a, corr_b.
        k: kNN for neural Laplacian.
        num_eigenvectors: Number of eigenvectors to compute.
        device: Torch device.
        laplacian_configs: List of LaplacianConfig to evaluate. Each config
            produces a separate eigenbasis and metric set prefixed by config.tag.
            Defaults to [LaplacianConfig(assembly='diagonal_gram')].
        evaluators: List of ShapePairEvaluator instances.
        geo_cache: Precomputed GeodesicCache for mesh B.
        verbose_timing: If True, print per-phase wall-clock timings.
    """
    import time as _time

    if laplacian_configs is None:
        laplacian_configs = [LaplacianConfig(assembly='diagonal_gram')]

    n_a = len(pair.verts_a)
    _t = {}  # timing accumulator

    # bases_by_config[config_tag][label] = (evecs, evals, M_np)
    bases_by_config: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    M_np_shared: Dict[str, np.ndarray] = {}

    for label, verts in [('A', pair.verts_a), ('B', pair.verts_b)]:
        t0 = _time.perf_counter()
        verts_t = torch.from_numpy(verts).float().to(device)
        knn_np = compute_knn(verts, k)
        _t[f'knn_{label}'] = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        batch_data = Batch.from_data_list(
            [build_patch_data(verts_t, knn_np, device)]).to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fwd = model(batch_data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t[f'forward_{label}'] = _time.perf_counter() - t0

        knn_t = torch.from_numpy(knn_np).long().to(device)
        M_diag = fwd['areas'].detach()
        M_np = M_diag.cpu().numpy()
        M_np_shared[label] = M_np

        grad_coeffs = fwd['grad_coeffs']
        # Stiffness head's output, when configured.  The configured
        # assembly inside each lap_cfg decides whether it's actually
        # consumed (only ``'from_stiffness'`` reads it).
        stiffness_weights = fwd.get('stiffness_weights')

        for cfg in laplacian_configs:
            tag = cfg.tag

            # Force sparse assembly for eval (no gradients needed).
            use_sparse = cfg.pruning == 'none'
            eval_cfg = LaplacianConfig(
                assembly=cfg.assembly,
                pruning=cfg.pruning,
                k_prune=cfg.k_prune,
                sparse=use_sparse,
                area_weighted=cfg.area_weighted,
            )

            # Compute knn_prune if needed
            knn_prune = None
            if cfg.pruning == 'knn' and cfg.k_prune is not None and cfg.k_prune != k:
                knn_prune_np = compute_knn(verts, cfg.k_prune)
                knn_prune = torch.from_numpy(knn_prune_np).long().to(device)

            t0 = _time.perf_counter()
            L_assembled = assemble_laplacian(
                grad_coeffs, knn_t, eval_cfg, areas=M_diag, knn_prune=knn_prune,
                stiffness_weights=stiffness_weights)

            # Convert to scipy sparse if dense (pruned configs)
            if use_sparse:
                L_scipy = L_assembled
            else:
                L_np = L_assembled.detach().cpu().numpy()
                L_scipy = scipy.sparse.csr_matrix(L_np)
            _t[f'assembly_{label}'] = _t.get(f'assembly_{label}', 0) + _time.perf_counter() - t0

            t0 = _time.perf_counter()
            M_scipy = scipy.sparse.diags(M_np)
            evals, evecs = compute_laplacian_eigendecomposition(
                L_scipy, num_eigenvectors, mass_matrix=M_scipy)
            _t[f'eigen_{label}'] = _t.get(f'eigen_{label}', 0) + _time.perf_counter() - t0

            if tag not in bases_by_config:
                bases_by_config[tag] = {}
            bases_by_config[tag][label] = (evecs, evals, M_np)

    t0 = _time.perf_counter()
    gt_corr = build_gt_corr_from_pair(pair)
    if geo_cache is None:
        geo_cache = build_geo_cache(pair)

    mA = M_np_shared['A']
    mB = M_np_shared['B']

    # --- Use evaluators if provided ---
    if evaluators is not None:
        metrics: Dict[str, float] = {}
        for tag, bases in bases_by_config.items():
            evA, evalsA, _ = bases['A']
            evB, evalsB, _ = bases['B']
            prefix = f"{tag}_" if len(bases_by_config) > 1 else ""
            for evaluator in evaluators:
                ev_metrics = evaluator.evaluate(
                    evA, evB, evalsA, evalsB,
                    mA, mB, pair.verts_b, gt_corr=gt_corr,
                    geo_cache=geo_cache,
                )
                for mk, mv in ev_metrics.items():
                    metrics[f"{prefix}{evaluator.name}/{mk}"] = mv

        # Eigenvalue summary from first config
        first_tag = list(bases_by_config.keys())[0]
        first_bases = bases_by_config[first_tag]
        _, evalsA, _ = first_bases['A']
        _, evalsB, _ = first_bases['B']
        metrics.update(_eigenvalue_summary(evalsA, 'eval_A'))
        metrics.update(_eigenvalue_summary(evalsB, 'eval_B'))

        _t['evaluators'] = _time.perf_counter() - t0

        if verbose_timing:
            total = sum(_t.values())
            name = getattr(pair, 'name', '?')
            parts = [f"{name} ({n_a}v, total={total:.2f}s):"]
            for phase in ['knn_A', 'forward_A', 'assembly_A', 'eigen_A',
                          'knn_B', 'forward_B', 'assembly_B', 'eigen_B',
                          'evaluators']:
                if phase in _t:
                    parts.append(f"{phase}={_t[phase]*1e3:.0f}ms")
            print("    [timing] " + "  ".join(parts), flush=True)

        return metrics

    # --- Legacy path (backward compatible) ---
    first_tag = list(bases_by_config.keys())[0]
    bases = bases_by_config[first_tag]
    evA, evalsA, _ = bases['A']
    evB, evalsB, _ = bases['B']
    legacy_metrics = correspondence_metrics(evA, evB, mA, mB, pair.verts_b, n_a, gt_corr=gt_corr)
    legacy_metrics.update(_eigenvalue_summary(evalsA, 'eval_A'))
    legacy_metrics.update(_eigenvalue_summary(evalsB, 'eval_B'))
    return legacy_metrics


@torch.no_grad()
def evaluate_pair_robust(
    pair,
    num_eigenvectors: int,
    n_neighbors: int = 30,
    evaluators: Optional[List] = None,
    geo_cache: Optional[GeodesicCache] = None,
    area_weighted: bool = False,
) -> Dict[str, float]:
    """Evaluate using robust Laplacian (baseline, no model).

    Args:
        evaluators: List of ShapePairEvaluator instances.
        geo_cache: Precomputed GeodesicCache for mesh B.
        area_weighted: If True, multiply S by diag(areas).
    """
    import robust_laplacian
    n_a = len(pair.verts_a)
    bases: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, verts in [('A', pair.verts_a), ('B', pair.verts_b)]:
        S, M = robust_laplacian.point_cloud_laplacian(verts, n_neighbors=n_neighbors)
        if area_weighted:
            M_diag = np.array(M.diagonal()).flatten()
            S = scipy.sparse.diags(M_diag) @ S
        evals, evecs = compute_laplacian_eigendecomposition(
            S, num_eigenvectors, mass_matrix=M)
        bases[label] = (evecs, evals, np.array(M.diagonal()).flatten())

    gt_corr = build_gt_corr_from_pair(pair)
    if geo_cache is None:
        geo_cache = build_geo_cache(pair)
    evA, evalsA, mA = bases['A']
    evB, evalsB, mB = bases['B']

    if evaluators is not None:
        metrics: Dict[str, float] = {}
        for evaluator in evaluators:
            ev_metrics = evaluator.evaluate(
                evA, evB, evalsA, evalsB,
                mA, mB, pair.verts_b, gt_corr=gt_corr,
                geo_cache=geo_cache,
            )
            for mk, mv in ev_metrics.items():
                metrics[f"{evaluator.name}/{mk}"] = mv
        return metrics

    # Legacy path
    return correspondence_metrics(
        evA, evB, mA, mB, pair.verts_b, n_a, gt_corr=gt_corr)


def eval_robust_worker(args):
    """Multiprocessing worker for robust Laplacian baseline evaluation."""
    pair, num_eigenvectors, evaluators, geo_cache, area_weighted = args
    try:
        return evaluate_pair_robust(
            pair, num_eigenvectors,
            evaluators=evaluators,
            geo_cache=geo_cache,
            area_weighted=area_weighted,
        )
    except Exception as e:
        print(f"    [robust worker] FAILED on {getattr(pair, 'name', '?')}: {e}",
              flush=True)
        return {}


# =============================================================================
# Summarise and print helper (used by both training pipelines)
# =============================================================================

def summarise_fmap_metrics(
    all_metrics: List[Dict[str, float]],
    evaluators: List,
    label: str,
    silent: bool = False,
) -> Dict[str, float]:
    """Aggregate per-pair metrics and print a summary.

    Args:
        all_metrics: List of per-pair metric dicts from evaluators.
        evaluators: List of ShapePairEvaluator instances (for formatting).
        label: Human-readable label (e.g. "Val epoch 3 [smal]").
        silent: If True, skip printing.

    Returns:
        summary: Dict mapping metric names to averaged values.
    """
    if not all_metrics:
        return {}

    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())
    all_keys = sorted(all_keys)

    summary: Dict[str, float] = {}
    for mk in all_keys:
        vals = [m[mk] for m in all_metrics
                if mk in m and np.isfinite(m[mk])]
        if vals:
            summary[mk] = float(np.mean(vals))

    if not silent:
        rows = []
        for ev in evaluators:
            acc_suffix = f"{ev.name}/accuracy"
            for key in sorted(all_keys):
                if key.endswith(acc_suffix) and summary.get(key) is not None:
                    prefix = key[:-len(acc_suffix)]
                    display = f"{prefix}{ev.name}" if prefix else ev.name
                    rows.append((display, prefix, ev))

        if not rows:
            return summary

        name_w = max(len(r[0]) for r in rows)
        n_pairs = len(all_metrics)

        print(f"  {label}:", flush=True)

        for display_name, prefix, ev in rows:
            full_prefix = f"{prefix}{ev.name}"
            acc = summary.get(f"{full_prefix}/accuracy", 0.0)
            err = summary.get(f"{full_prefix}/mean_error", 0.0)

            acc_key = f"{full_prefix}/accuracy"
            n_valid = sum(1 for m in all_metrics
                          if acc_key in m and np.isfinite(m[acc_key]))
            n_fail = n_pairs - n_valid
            fail_str = f"  ({n_fail}/{n_pairs} failed)" if n_fail else ""

            parts = [f"    {display_name:<{name_w}}  top1={acc*100:5.1f}%"]

            for topk in (3, 5, 10):
                topk_val = summary.get(f"{full_prefix}/top{topk}_acc")
                if topk_val is not None:
                    parts.append(f"  top{topk}={topk_val*100:5.1f}%")

            parts.append(f"  Err={err:.4f}")

            gfm_acc = summary.get(f"{full_prefix}/geomfum_accuracy")
            if gfm_acc is not None:
                parts.append(f"  gfm={gfm_acc*100:5.1f}%")

            geo_err = summary.get(f"{full_prefix}/geo_mean_error")
            if geo_err is not None:
                geo_parts = []
                for thresh in (1, 5, 10, 25):
                    val = summary.get(f"{full_prefix}/geo_at_{thresh:02d}pct", 0.0)
                    geo_parts.append(f"@{thresh}%={val*100:5.1f}%")
                parts.append(f"  │ {'  '.join(geo_parts)}  gErr={geo_err:.4f}")

            parts.append(fail_str)
            print("".join(parts), flush=True)

    return summary