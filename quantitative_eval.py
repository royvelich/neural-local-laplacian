#!/usr/bin/env python3
"""
Quantitative Evaluation for Neural Local Laplacian.

Uses shared step functions from validation_utils.py so timing numbers are
identical across scripts.  Outputs per-mesh CSV with timing breakdown AND
quality metrics.

Supports multi-GPU via mp.spawn.

Methods:
    GT is always computed (cotangent Laplacian from mesh faces, requires igl).
    Additional methods are selected via +methods=pred,robust,nelo (comma-separated).

Usage:
    python quantitative_eval.py +ckpt_path=model.ckpt

    # Choose which methods to evaluate (default: pred,robust)
    python quantitative_eval.py +ckpt_path=model.ckpt +methods=pred,robust,nelo

    # NeLo requires its own checkpoint and k
    python quantitative_eval.py +ckpt_path=model.ckpt +methods=pred,nelo \
        +nelo_ckpt_path=nelo_model.ckpt +nelo_k=8

    # Multi-GPU
    python quantitative_eval.py +ckpt_path=model.ckpt +num_gpus=4
"""

import csv
import gc
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import numpy as np
import torch
import torch.multiprocessing as mp

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from neural_local_laplacian.utils.utils import cuda_warmup
from neural_local_laplacian.utils.geodesic_utils import select_multiple_geodesic_sources
from neural_local_laplacian.utils.validation_utils import (
    load_mesh_with_faces,
    step_pred_laplacian,
    step_pred_geodesic,
    step_robust_geodesic,
    step_gt_laplacian,
    step_robust_laplacian,
    step_greens_function,
    step_eigendecomposition,
    step_eigenvalue_errors,
    step_eigenvector_cosine_similarity,
    GeodesicTimingBreakdown,
    HAS_NELO,
)

if HAS_NELO:
    from neural_local_laplacian.utils.validation_utils import (
        load_nelo_model,
        step_nelo_laplacian,
    )


VALID_METHODS = {'pred', 'robust', 'nelo'}


def _parse_methods(methods_str: str) -> Set[str]:
    """Parse comma-separated methods string, validate."""
    methods = {m.strip().lower() for m in methods_str.split(',')}
    invalid = methods - VALID_METHODS
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}. Valid: {VALID_METHODS}")
    return methods


# ============================================================================
# Helper: add metrics for a method's Green's / eigen / etc.
# ============================================================================

def _add_method_metrics(
    metrics: Dict[str, Any],
    prefix: str,
    L, M,
    source_indices: List[int],
    gt_greens_values,
    gt_evals, gt_evecs,
    num_eigenvalues: int,
    lm_timing_total: float,
):
    """Run Green's + eigen for one method and add to metrics dict."""

    # Green's function
    greens, t_greens = step_greens_function(
        L, M, source_indices, gt_greens_values=gt_greens_values,
    )
    greens_e2e = lm_timing_total + t_greens.total
    metrics[f'{prefix}_greens_fact_ms'] = t_greens.factorize * 1000
    metrics[f'{prefix}_greens_solve_ms'] = t_greens.solve * 1000
    metrics[f'{prefix}_greens_total_ms'] = t_greens.total * 1000
    metrics[f'{prefix}_greens_e2e_ms'] = greens_e2e * 1000
    metrics[f'{prefix}_greens_max_principle'] = greens.max_principle_pass_rate
    metrics[f'{prefix}_greens_gt_corr_mean'] = greens.mean_corr_with_gt
    metrics[f'{prefix}_greens_gt_corr_std'] = greens.std_corr_with_gt
    metrics[f'{prefix}_greens_residual_norm'] = greens.mean_residual_norm

    # Eigendecomposition
    evals, evecs, t_eig = step_eigendecomposition(L, M, num_eigenvalues)
    metrics[f'{prefix}_eigen_ms'] = t_eig['eigen'] * 1000

    # Eigenvalue errors (vs GT)
    if gt_evals is not None and evals is not None:
        ev_err = step_eigenvalue_errors(evals, gt_evals)
        metrics[f'{prefix}_eval_spectrum_rel_err_mean'] = ev_err.spectrum_rel_err_mean
        metrics[f'{prefix}_eval_spectrum_rel_err_max'] = ev_err.spectrum_rel_err_max
        metrics[f'{prefix}_eval_rel_err_mean'] = ev_err.per_eval_rel_err_mean
        metrics[f'{prefix}_eval_rel_err_max'] = ev_err.per_eval_rel_err_max

    # Eigenvector cosine similarity (vs GT)
    if gt_evecs is not None and evecs is not None:
        sims = step_eigenvector_cosine_similarity(gt_evecs, evecs)
        for c in [5, 10, 20, 50]:
            if c <= len(sims):
                metrics[f'{prefix}_eigvec_cos_top{c}'] = float(sims[:c].mean())
        metrics[f'{prefix}_eigvec_cos_all'] = float(sims.mean())

    return greens, evals, evecs


# ============================================================================
# Worker function — runs on a single GPU
# ============================================================================

def _worker(rank: int, world_size: int, cfg_dict: dict, output_dir: str, total_meshes: int):
    """Process a subset of meshes on a single GPU."""

    # ---- Device setup ----
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
    else:
        device = torch.device('cpu')

    tag = f"[GPU {rank}] " if world_size > 1 else ""
    cfg = OmegaConf.create(cfg_dict)

    # ---- Parse methods ----
    methods = _parse_methods(getattr(cfg, 'methods', 'pred,robust'))
    run_pred = 'pred' in methods
    run_robust = 'robust' in methods
    run_nelo = 'nelo' in methods

    # ---- Config ----
    pred_k = getattr(cfg.globals, 'k_pred', None) or getattr(cfg.globals, 'k', 30)
    robust_k = getattr(cfg.globals, 'k_robust', None) or 30
    nelo_k = getattr(cfg, 'nelo_k', 8)
    num_sources = getattr(cfg.globals, 'num_validation_sources', 10)
    num_eigenvalues = getattr(cfg.globals, 'num_eigenvalues', 50)
    use_amp = device.type == 'cuda'

    # ---- Load PRED model ----
    model = None
    if run_pred:
        ckpt_path = Path(cfg.ckpt_path)
        model = LaplacianTransformerModule.load_from_checkpoint(
            str(ckpt_path), map_location=device, strict=False
        )
        model = model.to(device).eval()
        if device.type == 'cuda':
            cuda_warmup(model, device, k=pred_k)

    # ---- Load NeLo model ----
    nelo_pipeline = None
    if run_nelo:
        nelo_ckpt = getattr(cfg, 'nelo_ckpt_path', None)
        if nelo_ckpt is None:
            raise ValueError("nelo_ckpt_path is required when methods includes 'nelo'")
        if not HAS_NELO:
            raise ImportError("NeLo dependencies not available")
        nelo_pipeline = load_nelo_model(str(nelo_ckpt), device)

    # ---- Dataset ----
    pl.seed_everything(cfg.globals.seed)
    data_module = hydra.utils.instantiate(cfg.data_module)
    data_loader = data_module.val_dataloader()
    if isinstance(data_loader, list):
        data_loader = data_loader[0]
    dataset = data_loader.dataset

    # ---- Compute this worker's mesh indices ----
    my_indices = list(range(rank, len(dataset), world_size))
    n_total = len(my_indices)
    methods_str = ','.join(sorted(methods))
    print(f"{tag}Ready — {n_total} meshes on {device}, methods=[{methods_str}], "
          f"PRED k={pred_k}, Robust k={robust_k}")

    # ---- Process meshes ----
    metrics_list: List[Dict[str, Any]] = []
    t_start = time.time()

    for local_idx, global_idx in enumerate(my_indices):
        t_mesh_start = time.time()
        mesh_name = "?"
        metrics: Dict[str, Any] = {}

        try:
            batch_data = dataset[global_idx]
            data = batch_data[0] if isinstance(batch_data, list) else batch_data
            mesh_file_path = data.mesh_file_path
            if isinstance(mesh_file_path, list):
                mesh_file_path = mesh_file_path[0]
            mesh_name = Path(mesh_file_path).name

            vertices, faces = load_mesh_with_faces(mesh_file_path)
            N = len(vertices)
            source_indices = select_multiple_geodesic_sources(
                vertices.astype(np.float64), num_sources=num_sources,
                method="farthest_point_sampling", seed=42
            ).tolist()
            verts_tensor = torch.from_numpy(vertices).float().to(device)

            metrics['mesh_name'] = mesh_name
            metrics['num_vertices'] = N
            metrics['num_faces'] = len(faces)
            metrics['num_sources'] = len(source_indices)

            # ---- Disable GC during timing-sensitive section ----
            gc.collect()
            gc.disable()

            # ==============================================================
            # Phase 1: TIMING-SENSITIVE (GPU hot, GC disabled)
            # Runs all assembly + geodesic steps while GPU caches are warm.
            # ==============================================================

            # Mini warmup: single matmul to wake GPU from idle power state.
            # Prevents cold-start inflation after long CPU phases (eigen ~20s).
            if device.type == 'cuda':
                _dummy = torch.randn(256, 256, device=device) @ torch.randn(256, 256, device=device)
                torch.cuda.synchronize()
                del _dummy

            # ---- PRED assembly + geodesic ----
            pred_L, pred_M, pred_G, t_pred = None, None, None, None
            pred_geo_timing = GeodesicTimingBreakdown()
            if run_pred:
                pred_L, pred_M, pred_G, t_pred = step_pred_laplacian(
                    model, verts_tensor, pred_k, device, use_amp=use_amp,
                )
                metrics['pred_k'] = pred_k
                metrics['pred_knn_ms'] = t_pred.knn * 1000
                metrics['pred_forward_ms'] = t_pred.forward * 1000
                metrics['pred_assembly_ms'] = t_pred.assembly * 1000
                metrics['pred_grad_op_ms'] = t_pred.grad_op * 1000
                metrics['pred_lm_total_ms'] = t_pred.lm_total * 1000

                if pred_G is not None:
                    _, pred_geo_timing = step_pred_geodesic(
                        pred_L, pred_M, pred_G, source_indices, N,
                    )
                    pred_onetime = (t_pred.total
                                    + pred_geo_timing.build
                                    + pred_geo_timing.heat_factorize
                                    + pred_geo_timing.poisson_factorize)
                    pred_e2e = pred_onetime + pred_geo_timing.solve
                    metrics['pred_geo_build_ms'] = pred_geo_timing.build * 1000
                    metrics['pred_geo_heat_fact_ms'] = pred_geo_timing.heat_factorize * 1000
                    metrics['pred_geo_poisson_fact_ms'] = pred_geo_timing.poisson_factorize * 1000
                    metrics['pred_geo_solve_ms'] = pred_geo_timing.solve * 1000
                    metrics['pred_e2e_geodesic_ms'] = pred_e2e * 1000
                    metrics['pred_per_src_ms'] = pred_e2e / len(source_indices) * 1000

            # ---- Robust geodesic (pp3d, self-contained timing) ----
            robust_timing = None
            if run_robust:
                _, robust_timing = step_robust_geodesic(vertices, source_indices, robust_k)
                robust_e2e = robust_timing.total
                metrics['robust_constructor_ms'] = robust_timing.constructor * 1000
                metrics['robust_geo_solve_ms'] = robust_timing.solve * 1000
                metrics['robust_e2e_geodesic_ms'] = robust_e2e * 1000
                metrics['robust_per_src_ms'] = robust_e2e / len(source_indices) * 1000

            # ---- NeLo assembly (timing-sensitive) ----
            nelo_L, nelo_M, t_nelo = None, None, None
            if run_nelo:
                nelo_L, nelo_M, t_nelo = step_nelo_laplacian(
                    nelo_pipeline, vertices, nelo_k, device,
                )
                metrics['nelo_k'] = nelo_k
                metrics['nelo_graph_tree_ms'] = t_nelo.graph_tree * 1000
                metrics['nelo_forward_ms'] = t_nelo.forward * 1000
                metrics['nelo_assembly_ms'] = t_nelo.assembly * 1000
                metrics['nelo_lm_total_ms'] = t_nelo.total * 1000

            # ---- Re-enable GC (timing-sensitive section done) ----
            gc.enable()

            # ==============================================================
            # Phase 2: QUALITY (not timing-sensitive)
            # GT assembly, Green's, eigendecomposition, comparisons.
            # ==============================================================

            # ---- GT Laplacian ----
            gt_L, gt_M, t_gt = step_gt_laplacian(vertices, faces)
            metrics['gt_assembly_ms'] = t_gt['assembly'] * 1000

            # ---- Robust Laplacian (for Green's / eigen, separate from pp3d) ----
            rob_L, rob_M = None, None
            if run_robust:
                rob_L, rob_M, t_rob_lm = step_robust_laplacian(vertices, robust_k)
                metrics['robust_k'] = robust_k
                metrics['robust_lm_assembly_ms'] = t_rob_lm['assembly'] * 1000

            # ---- GT Green's (reference for correlation) ----
            gt_greens = None
            gt_gvals = None
            if gt_L is not None:
                gt_greens, t_greens_gt = step_greens_function(gt_L, gt_M, source_indices)
                metrics['gt_greens_total_ms'] = t_greens_gt.total * 1000
                metrics['gt_greens_max_principle'] = gt_greens.max_principle_pass_rate
                gt_gvals = gt_greens.values

            # ---- GT eigendecomposition ----
            gt_evals, gt_evecs = None, None
            if gt_L is not None:
                gt_evals, gt_evecs, t_eig_gt = step_eigendecomposition(
                    gt_L, gt_M, num_eigenvalues,
                )
                metrics['gt_eigen_ms'] = t_eig_gt['eigen'] * 1000

            # ---- PRED quality (Green's + eigen) ----
            if run_pred and pred_L is not None:
                _add_method_metrics(
                    metrics, 'pred', pred_L, pred_M,
                    source_indices, gt_gvals, gt_evals, gt_evecs,
                    num_eigenvalues, t_pred.lm_total,
                )

            # ---- Robust quality (Green's + eigen) ----
            if run_robust and rob_L is not None:
                _add_method_metrics(
                    metrics, 'robust', rob_L, rob_M,
                    source_indices, gt_gvals, gt_evals, gt_evecs,
                    num_eigenvalues, t_rob_lm['assembly'],
                )

            # ---- NeLo quality (Green's + eigen) ----
            if run_nelo and nelo_L is not None:
                _add_method_metrics(
                    metrics, 'nelo', nelo_L, nelo_M,
                    source_indices, gt_gvals, gt_evals, gt_evecs,
                    num_eigenvalues, t_nelo.total,
                )

            # ==============================================================
            # Timing ratios
            # ==============================================================
            if run_pred and run_robust and robust_timing is not None:
                if robust_timing.lm_assembly > 0:
                    metrics['ratio_lm'] = t_pred.lm_total / robust_timing.lm_assembly
                if metrics.get('pred_e2e_geodesic_ms') and metrics.get('robust_e2e_geodesic_ms'):
                    metrics['ratio_e2e_geodesic'] = (
                        metrics['pred_e2e_geodesic_ms'] / metrics['robust_e2e_geodesic_ms']
                    )
                if metrics.get('pred_greens_e2e_ms') and metrics.get('robust_greens_e2e_ms'):
                    metrics['ratio_greens_e2e'] = (
                        metrics['pred_greens_e2e_ms'] / metrics['robust_greens_e2e_ms']
                    )

            metrics_list.append(metrics)
            status = "OK"
            del verts_tensor

        except Exception as e:
            gc.enable()
            status = f"ERROR — {e}"
            import traceback
            traceback.print_exc()

        t_mesh = time.time() - t_mesh_start
        elapsed = time.time() - t_start
        done = local_idx + 1
        eta = (elapsed / done) * (n_total - done) if done > 0 else 0

        # Per-mesh summary line
        parts = [f"{tag}[{done}/{n_total}] {mesh_name:<16s} {status:<6s} N={N:>6d}"]
        if metrics.get('pred_e2e_geodesic_ms'):
            parts.append(f"PRED={metrics['pred_e2e_geodesic_ms']:.0f}ms")
        if metrics.get('robust_e2e_geodesic_ms'):
            parts.append(f"Rob={metrics['robust_e2e_geodesic_ms']:.0f}ms")
        if metrics.get('nelo_lm_total_ms'):
            parts.append(f"NeLo={metrics['nelo_lm_total_ms']:.0f}ms")
        if metrics.get('pred_greens_gt_corr_mean'):
            parts.append(f"G_corr={metrics['pred_greens_gt_corr_mean']:.3f}")
        if metrics.get('pred_eigvec_cos_all'):
            parts.append(f"evec={metrics['pred_eigvec_cos_all']:.3f}")
        parts.append(f"[{t_mesh:.1f}s, ETA {eta:.0f}s]")
        print(' '.join(parts))

    elapsed = time.time() - t_start
    print(f"\n{tag}Done — {len(metrics_list)} meshes in {elapsed:.1f}s")

    # ---- Write per-rank CSV ----
    rank_csv = Path(output_dir) / f'metrics_rank{rank}.csv'
    _write_csv(metrics_list, rank_csv)
    print(f"{tag}Saved {rank_csv}")


# ============================================================================
# CSV utilities
# ============================================================================

def _write_csv(metrics_list: List[Dict[str, Any]], csv_path: Path):
    """Write a list of metric dicts to CSV."""
    if not metrics_list:
        return

    all_keys = []
    seen = set()
    for m in metrics_list:
        for k in m.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m)


def _read_and_merge_csvs(output_dir: Path, world_size: int) -> List[Dict[str, Any]]:
    """Read per-rank CSVs and merge."""
    all_metrics = []

    for rank in range(world_size):
        rank_csv = output_dir / f'metrics_rank{rank}.csv'
        if not rank_csv.exists():
            print(f"[!] Missing {rank_csv}")
            continue

        with open(rank_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if v == '' or v is None:
                        parsed[k] = None
                    else:
                        try:
                            if '.' not in v and 'e' not in v.lower():
                                parsed[k] = int(v)
                            else:
                                parsed[k] = float(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                all_metrics.append(parsed)

    return all_metrics


def _compute_summary(all_metrics: List[Dict[str, Any]], summary_path: Path):
    """Compute mean/std/min/max for all numeric columns and save."""
    if not all_metrics:
        return

    numeric_keys = []
    for k in all_metrics[0].keys():
        vals = [m[k] for m in all_metrics if m.get(k) is not None]
        if vals and isinstance(vals[0], (int, float)):
            numeric_keys.append(k)

    rows = []
    for k in numeric_keys:
        vals = [float(m[k]) for m in all_metrics if m.get(k) is not None]
        if not vals:
            continue
        rows.append({
            'metric': k,
            'mean': np.mean(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals),
            'count': len(vals),
        })

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'mean', 'std', 'min', 'max', 'count'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ============================================================================
# Main entry point
# ============================================================================

@hydra.main(version_base="1.2", config_path='./visualization_config')
def main(cfg: DictConfig) -> None:
    """Quantitative evaluation."""

    if not hasattr(cfg, 'ckpt_path') or cfg.ckpt_path is None:
        raise ValueError("ckpt_path is required")

    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    methods = _parse_methods(getattr(cfg, 'methods', 'pred,robust'))

    # ---- GPU setup ----
    num_gpus_available = torch.cuda.device_count()
    num_gpus = getattr(cfg, 'num_gpus', None)
    if num_gpus is None:
        num_gpus = max(num_gpus_available, 1)
    else:
        num_gpus = min(int(num_gpus), num_gpus_available)
    if num_gpus_available == 0:
        print("[!] No CUDA devices found — running on CPU")
        num_gpus = 1

    # ---- Get total mesh count ----
    data_module = hydra.utils.instantiate(cfg.data_module)
    data_loader = data_module.val_dataloader()
    if isinstance(data_loader, list):
        data_loader = data_loader[0]
    total_meshes = len(data_loader.dataset)
    del data_module, data_loader

    # ---- Output setup ----
    output_dir = Path(getattr(cfg, 'output_dir', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / '.tmp_quantitative_ranks'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pred_k = getattr(cfg.globals, 'k_pred', None) or getattr(cfg.globals, 'k', 30)
    robust_k = getattr(cfg.globals, 'k_robust', None) or 30
    nelo_k = getattr(cfg, 'nelo_k', 8)
    num_sources = getattr(cfg.globals, 'num_validation_sources', 10)
    num_eigenvalues = getattr(cfg.globals, 'num_eigenvalues', 50)

    print(f"\n{'=' * 80}")
    print(f"QUANTITATIVE EVALUATION")
    print(f"{'=' * 80}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"Methods:     GT + {', '.join(sorted(methods))}")
    print(f"Meshes:      {total_meshes}")
    print(f"GPUs:        {num_gpus} / {num_gpus_available} available")
    if 'pred' in methods:
        print(f"PRED k:      {pred_k}")
    if 'robust' in methods:
        print(f"Robust k:    {robust_k}")
    if 'nelo' in methods:
        print(f"NeLo k:      {nelo_k}")
        print(f"NeLo ckpt:   {getattr(cfg, 'nelo_ckpt_path', 'N/A')}")
    print(f"Sources:     {num_sources}")
    print(f"Eigenvalues: {num_eigenvalues}")
    print(f"Output:      {output_dir}")
    print(f"{'=' * 80}\n")

    # ---- Serialize config for workers ----
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # ---- Run ----
    t_start = time.time()

    if num_gpus == 1:
        _worker(0, 1, cfg_dict, str(tmp_dir), total_meshes)
    else:
        mp.spawn(
            _worker,
            args=(num_gpus, cfg_dict, str(tmp_dir), total_meshes),
            nprocs=num_gpus,
            join=True,
        )

    elapsed = time.time() - t_start

    # ---- Merge per-rank CSVs ----
    all_metrics = _read_and_merge_csvs(tmp_dir, num_gpus)
    print(f"\nMerged {len(all_metrics)} mesh results from {num_gpus} workers")

    if len(all_metrics) == 0:
        print("[!] No metrics collected")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # ---- Save full CSV ----
    csv_path = output_dir / 'quantitative_results.csv'
    _write_csv(all_metrics, csv_path)
    print(f"Saved per-mesh results: {csv_path}")

    # ---- Compute and save summary ----
    summary_path = csv_path.with_name(csv_path.stem + '_summary.csv')
    _compute_summary(all_metrics, summary_path)
    print(f"Saved summary: {summary_path}")

    # ---- Print summary table ----
    _print_summary(all_metrics, methods, num_sources, elapsed, num_gpus)

    # ---- Cleanup ----
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _print_summary(
    all_metrics: List[Dict[str, Any]],
    methods: Set[str],
    num_sources: int,
    elapsed: float,
    num_gpus: int,
):
    """Print human-readable summary table."""

    def _stats(key):
        vals = [float(m[key]) for m in all_metrics if m.get(key) is not None]
        if not vals:
            return None, None, None, None
        return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

    def _print_timing_row(label, key, W=10):
        m, s, mn, mx = _stats(key)
        if m is not None:
            print(f"  {label:<28s} {m:>{W}.1f} {s:>{W}.1f} {mn:>{W}.1f} {mx:>{W}.1f}")

    def _print_quality_row(label, key, W=10):
        m, s, _, _ = _stats(key)
        if m is not None:
            print(f"  {label:<28s} {m:>{W}.4f} {s:>{W}.4f}")

    W = 10
    n_meshes = len(all_metrics)

    print(f"\n{'=' * 80}")
    print(f"SUMMARY ({n_meshes} meshes, {num_sources} sources/mesh)")
    print(f"{'=' * 80}")

    # --- Timing ---
    print(f"\n  TIMING (ms)")
    print(f"  {'':>28s} {'Mean':>{W}s} {'Std':>{W}s} {'Min':>{W}s} {'Max':>{W}s}")
    print(f"  {'-' * 28} {'-' * W} {'-' * W} {'-' * W} {'-' * W}")

    # L,M assembly
    for prefix, label in [('pred', 'PRED'), ('robust', 'Robust'), ('nelo', 'NeLo')]:
        if prefix not in methods:
            continue
        _print_timing_row(f"{label} L,M total", f"{prefix}_lm_total_ms" if prefix != 'robust' else "robust_lm_assembly_ms")

    _print_timing_row("GT assembly", "gt_assembly_ms")
    print()

    # Geodesic E2E
    for prefix, label in [('pred', 'PRED'), ('robust', 'Robust')]:
        if prefix not in methods:
            continue
        _print_timing_row(f"{label} E2E geodesic", f"{prefix}_e2e_geodesic_ms")

    print()

    # Green's E2E
    for prefix, label in [('pred', 'PRED'), ('robust', 'Robust'), ('nelo', 'NeLo')]:
        if prefix not in methods:
            continue
        _print_timing_row(f"{label} Green's E2E", f"{prefix}_greens_e2e_ms")

    print()

    # Eigendecomposition
    for prefix, label in [('pred', 'PRED'), ('robust', 'Robust'), ('nelo', 'NeLo'), ('gt', 'GT')]:
        if prefix != 'gt' and prefix not in methods:
            continue
        _print_timing_row(f"{label} eigendecomp", f"{prefix}_eigen_ms")

    # --- Green's quality ---
    print(f"\n  GREEN'S FUNCTION QUALITY")
    print(f"  {'':>28s} {'Mean':>{W}s} {'Std':>{W}s}")
    print(f"  {'-' * 28} {'-' * W} {'-' * W}")

    _print_quality_row("GT max principle", "gt_greens_max_principle")
    for prefix, label in [('pred', 'PRED'), ('robust', 'Robust'), ('nelo', 'NeLo')]:
        if prefix not in methods:
            continue
        _print_quality_row(f"{label} max principle", f"{prefix}_greens_max_principle")
        _print_quality_row(f"{label} GT correlation", f"{prefix}_greens_gt_corr_mean")
        _print_quality_row(f"{label} residual norm", f"{prefix}_greens_residual_norm")

    # --- Spectral quality ---
    print(f"\n  SPECTRAL QUALITY")
    print(f"  {'':>28s} {'Mean':>{W}s} {'Std':>{W}s}")
    print(f"  {'-' * 28} {'-' * W} {'-' * W}")

    for prefix, label in [('pred', 'PRED'), ('robust', 'Robust'), ('nelo', 'NeLo')]:
        if prefix not in methods:
            continue
        _print_quality_row(f"{label} eval rel err mean", f"{prefix}_eval_spectrum_rel_err_mean")
        _print_quality_row(f"{label} eigvec cos top5", f"{prefix}_eigvec_cos_top5")
        _print_quality_row(f"{label} eigvec cos top20", f"{prefix}_eigvec_cos_top20")
        _print_quality_row(f"{label} eigvec cos all", f"{prefix}_eigvec_cos_all")
        if prefix != list(methods)[-1]:  # separator between methods
            print()

    # --- Ratios ---
    if 'pred' in methods and 'robust' in methods:
        print()
        pred_e2e = [float(m['pred_e2e_geodesic_ms']) for m in all_metrics
                    if m.get('pred_e2e_geodesic_ms')]
        rob_e2e = [float(m['robust_e2e_geodesic_ms']) for m in all_metrics
                   if m.get('robust_e2e_geodesic_ms')]
        if pred_e2e and rob_e2e:
            ratio = np.mean(pred_e2e) / np.mean(rob_e2e)
            wins = sum(1 for p, r in zip(pred_e2e, rob_e2e) if p < r)
            print(f"  PRED/Robust E2E geodesic: {ratio:.2f}x  (PRED wins {wins}/{len(pred_e2e)})")

        pred_g = [float(m['pred_greens_e2e_ms']) for m in all_metrics
                  if m.get('pred_greens_e2e_ms')]
        rob_g = [float(m['robust_greens_e2e_ms']) for m in all_metrics
                 if m.get('robust_greens_e2e_ms')]
        if pred_g and rob_g:
            ratio = np.mean(pred_g) / np.mean(rob_g)
            wins = sum(1 for p, r in zip(pred_g, rob_g) if p < r)
            print(f"  PRED/Robust E2E Green's:  {ratio:.2f}x  (PRED wins {wins}/{len(pred_g)})")

    print(f"\n  Total time: {elapsed:.1f}s ({n_meshes} meshes, {num_gpus} GPUs)")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()