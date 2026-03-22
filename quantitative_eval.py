#!/usr/bin/env python3
"""
Quantitative Evaluation for Neural Local Laplacian.

Uses shared step functions from validation_utils.py so timing numbers are
identical across scripts.  Outputs per-mesh CSV with timing breakdown AND
quality metrics (eigenvalues, eigenvectors, Green's, geodesics, probes,
descriptors, spectral compression).

Supports multi-GPU via mp.spawn.

Usage:
    python quantitative_eval.py +ckpt_path=model.ckpt \
        +data_module=visualize_validation +globals=visualize_validation +model=visualize_validation

    # Multi-GPU
    python quantitative_eval.py +ckpt_path=model.ckpt +num_gpus=4

    # Custom output directory
    python quantitative_eval.py +ckpt_path=model.ckpt +output_dir=results/
"""

import csv
import gc
import shutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

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
    step_pred_knn,
    step_pred_inference,
    step_pred_geodesic,
    step_robust_geodesic,
    step_gt_laplacian,
    step_robust_laplacian,
    step_eigendecomposition,
    step_eigenvalue_errors,
    step_eigenvector_cosine_similarity,
    step_greens_function,
    step_geodesic_quality,
    step_probe_function_mse,
    step_descriptor_comparison,
    step_spectral_compression,
    GeodesicTimingBreakdown,
)


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

    # ---- Load model ----
    ckpt_path = Path(cfg.ckpt_path)
    model = LaplacianTransformerModule.load_from_checkpoint(
        str(ckpt_path), map_location=device, strict=False
    )
    model = model.to(device).eval()

    has_grad = getattr(model, '_operator_mode', 'stiffness') == 'gradient'

    # ---- Config ----
    pred_k = getattr(cfg.globals, 'k_pred', None) or getattr(cfg.globals, 'k', 30)
    robust_k = getattr(cfg.globals, 'k_robust', None) or 30
    num_sources = getattr(cfg.globals, 'num_validation_sources', 10)
    num_eigenvalues = getattr(cfg.globals, 'num_eigenvalues', 50)
    use_amp = device.type == 'cuda'

    # ---- CUDA warmup ----
    if device.type == 'cuda':
        cuda_warmup(model, device, k=pred_k)

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
    print(f"{tag}Ready — {n_total} meshes on {device}, PRED k={pred_k}, Robust k={robust_k}, "
          f"eigenvalues={num_eigenvalues}, sources={num_sources}")

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

            # ==============================================================
            # Load mesh (vertices + faces)
            # ==============================================================
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
            metrics['k'] = pred_k
            metrics['num_sources'] = len(source_indices)

            # ==============================================================
            # Disable GC during timed sections
            # ==============================================================
            gc.collect()
            gc.disable()

            # ==============================================================
            # PRED pipeline: kNN → inference → L,M,G
            # ==============================================================
            patch_data, t_knn = step_pred_knn(verts_tensor, pred_k, device)
            pred_result, t_infer = step_pred_inference(model, patch_data, device, use_amp=use_amp)
            pred_L, pred_M, pred_G = pred_result['L'], pred_result['M'], pred_result['G']

            pred_lm_total = t_knn['knn'] + t_infer['forward'] + t_infer['assembly']

            metrics['pred_knn_ms'] = t_knn['knn'] * 1000
            metrics['pred_forward_ms'] = t_infer['forward'] * 1000
            metrics['pred_assembly_ms'] = t_infer['assembly'] * 1000
            metrics['pred_grad_op_ms'] = t_infer['grad_op'] * 1000
            metrics['pred_lm_total_ms'] = pred_lm_total * 1000

            # ==============================================================
            # GT pipeline: cotangent Laplacian from faces
            # ==============================================================
            gt_L, gt_M, t_gt = step_gt_laplacian(vertices, faces)
            metrics['gt_assembly_ms'] = t_gt['assembly'] * 1000

            # ==============================================================
            # Robust pipeline: point cloud Laplacian
            # ==============================================================
            rob_L, rob_M, t_rob = step_robust_laplacian(vertices, robust_k)
            metrics['robust_assembly_ms'] = t_rob['assembly'] * 1000

            # ==============================================================
            # Eigendecomposition (all methods)
            # ==============================================================
            pred_evals, pred_evecs, t_eig_pred = step_eigendecomposition(
                pred_L, pred_M, num_eigenvalues,
            )
            metrics['pred_eigen_ms'] = t_eig_pred['eigen'] * 1000

            gt_evals, gt_evecs = None, None
            if gt_L is not None:
                gt_evals, gt_evecs, t_eig_gt = step_eigendecomposition(
                    gt_L, gt_M, num_eigenvalues,
                )
                metrics['gt_eigen_ms'] = t_eig_gt['eigen'] * 1000

            rob_evals, rob_evecs, t_eig_rob = step_eigendecomposition(
                rob_L, rob_M, num_eigenvalues,
            )
            metrics['robust_eigen_ms'] = t_eig_rob['eigen'] * 1000

            # ==============================================================
            # Eigenvalue errors (vs GT)
            # ==============================================================
            if gt_evals is not None:
                if pred_evals is not None:
                    ev_err = step_eigenvalue_errors(pred_evals, gt_evals)
                    metrics['pred_eval_spectrum_rel_err_mean'] = ev_err.spectrum_rel_err_mean
                    metrics['pred_eval_spectrum_rel_err_max'] = ev_err.spectrum_rel_err_max
                    metrics['pred_eval_rel_err_mean'] = ev_err.per_eval_rel_err_mean
                    metrics['pred_eval_rel_err_max'] = ev_err.per_eval_rel_err_max

                if rob_evals is not None:
                    ev_err = step_eigenvalue_errors(rob_evals, gt_evals)
                    metrics['robust_eval_spectrum_rel_err_mean'] = ev_err.spectrum_rel_err_mean
                    metrics['robust_eval_spectrum_rel_err_max'] = ev_err.spectrum_rel_err_max
                    metrics['robust_eval_rel_err_mean'] = ev_err.per_eval_rel_err_mean
                    metrics['robust_eval_rel_err_max'] = ev_err.per_eval_rel_err_max

            # ==============================================================
            # Eigenvector cosine similarity (vs GT)
            # ==============================================================
            if gt_evecs is not None:
                if pred_evecs is not None:
                    sims = step_eigenvector_cosine_similarity(gt_evecs, pred_evecs)
                    for c in [5, 10, 20, 50]:
                        if c <= len(sims):
                            metrics[f'pred_eigvec_cos_top{c}'] = float(sims[:c].mean())
                    metrics['pred_eigvec_cos_all'] = float(sims.mean())

                if rob_evecs is not None:
                    sims = step_eigenvector_cosine_similarity(gt_evecs, rob_evecs)
                    for c in [5, 10, 20, 50]:
                        if c <= len(sims):
                            metrics[f'robust_eigvec_cos_top{c}'] = float(sims[:c].mean())
                    metrics['robust_eigvec_cos_all'] = float(sims.mean())

            # ==============================================================
            # Green's function (all methods, multi-source)
            # ==============================================================
            gt_greens = None
            if gt_L is not None:
                gt_greens, t_greens_gt = step_greens_function(
                    gt_L, gt_M, source_indices,
                )
                metrics['gt_greens_ms'] = t_greens_gt.total * 1000
                metrics['gt_greens_max_principle'] = gt_greens.max_principle_pass_rate

            gt_gvals = gt_greens.values if gt_greens is not None else None

            pred_greens, t_greens_pred = step_greens_function(
                pred_L, pred_M, source_indices, gt_greens_values=gt_gvals,
            )
            metrics['pred_greens_ms'] = t_greens_pred.total * 1000
            metrics['pred_greens_max_principle'] = pred_greens.max_principle_pass_rate
            metrics['pred_greens_gt_corr_mean'] = pred_greens.mean_corr_with_gt
            metrics['pred_greens_gt_corr_std'] = pred_greens.std_corr_with_gt
            metrics['pred_greens_residual_norm'] = pred_greens.mean_residual_norm

            rob_greens, t_greens_rob = step_greens_function(
                rob_L, rob_M, source_indices, gt_greens_values=gt_gvals,
            )
            metrics['robust_greens_ms'] = t_greens_rob.total * 1000
            metrics['robust_greens_max_principle'] = rob_greens.max_principle_pass_rate
            metrics['robust_greens_gt_corr_mean'] = rob_greens.mean_corr_with_gt
            metrics['robust_greens_gt_corr_std'] = rob_greens.std_corr_with_gt
            metrics['robust_greens_residual_norm'] = rob_greens.mean_residual_norm

            # ==============================================================
            # PRED geodesic (timing + quality)
            # ==============================================================
            if pred_G is not None:
                pred_distances, pred_geo_timing = step_pred_geodesic(
                    pred_L, pred_M, pred_G, source_indices, N,
                )
                metrics['pred_geo_build_ms'] = pred_geo_timing.build * 1000
                metrics['pred_geo_heat_fact_ms'] = pred_geo_timing.heat_factorize * 1000
                metrics['pred_geo_poisson_fact_ms'] = pred_geo_timing.poisson_factorize * 1000
                metrics['pred_geo_solve_ms'] = pred_geo_timing.solve * 1000

                pred_onetime = (pred_lm_total + t_infer['grad_op']
                                + pred_geo_timing.build
                                + pred_geo_timing.heat_factorize
                                + pred_geo_timing.poisson_factorize)
                pred_e2e = pred_onetime + pred_geo_timing.solve
                metrics['pred_e2e_geodesic_ms'] = pred_e2e * 1000

                geo_qual = step_geodesic_quality(
                    pred_distances, source_indices, vertices, faces,
                )
                metrics['pred_geodesic_corr_mean'] = geo_qual.corr_mean
                metrics['pred_geodesic_corr_std'] = geo_qual.corr_std
                metrics['pred_geodesic_mae_mean'] = geo_qual.mae_mean
                metrics['pred_geodesic_max_err_mean'] = geo_qual.max_err_mean
                metrics['pred_geodesic_mono_mean'] = geo_qual.mono_mean
            else:
                pred_geo_timing = GeodesicTimingBreakdown()

            # ==============================================================
            # Robust geodesic (timing + quality)
            # ==============================================================
            rob_distances, robust_timing = step_robust_geodesic(
                vertices, source_indices, robust_k,
            )
            metrics['robust_lm_ms'] = robust_timing.lm_assembly * 1000
            metrics['robust_constructor_ms'] = robust_timing.constructor * 1000
            metrics['robust_geo_solve_ms'] = robust_timing.solve * 1000

            robust_e2e = robust_timing.total
            metrics['robust_e2e_geodesic_ms'] = robust_e2e * 1000

            rob_geo_qual = step_geodesic_quality(
                rob_distances, source_indices, vertices, faces,
            )
            metrics['robust_geodesic_corr_mean'] = rob_geo_qual.corr_mean
            metrics['robust_geodesic_corr_std'] = rob_geo_qual.corr_std
            metrics['robust_geodesic_mae_mean'] = rob_geo_qual.mae_mean
            metrics['robust_geodesic_max_err_mean'] = rob_geo_qual.max_err_mean
            metrics['robust_geodesic_mono_mean'] = rob_geo_qual.mono_mean

            # ==============================================================
            # Probe function MSE
            # ==============================================================
            if gt_L is not None and gt_evals is not None and gt_evecs is not None:
                probe_pred = step_probe_function_mse(
                    pred_L, pred_M, gt_L, gt_M, vertices, gt_evals, gt_evecs,
                )
                metrics['pred_probe_mse'] = probe_pred.mse
                metrics['pred_probe_cosine_sim'] = probe_pred.cosine_similarity
                metrics['pred_probe_failure_rate'] = probe_pred.failure_rate

                probe_rob = step_probe_function_mse(
                    rob_L, rob_M, gt_L, gt_M, vertices, gt_evals, gt_evecs,
                )
                metrics['robust_probe_mse'] = probe_rob.mse
                metrics['robust_probe_cosine_sim'] = probe_rob.cosine_similarity
                metrics['robust_probe_failure_rate'] = probe_rob.failure_rate

            # ==============================================================
            # HKS / WKS descriptor comparison
            # ==============================================================
            if gt_evals is not None and gt_evecs is not None:
                if pred_evals is not None and pred_evecs is not None:
                    desc = step_descriptor_comparison(
                        pred_evals, pred_evecs, gt_evals, gt_evecs,
                    )
                    metrics['pred_hks_l2_error'] = desc.hks_l2_error
                    metrics['pred_hks_correlation'] = desc.hks_correlation
                    metrics['pred_wks_l2_error'] = desc.wks_l2_error
                    metrics['pred_wks_correlation'] = desc.wks_correlation

                if rob_evals is not None and rob_evecs is not None:
                    desc = step_descriptor_comparison(
                        rob_evals, rob_evecs, gt_evals, gt_evecs,
                    )
                    metrics['robust_hks_l2_error'] = desc.hks_l2_error
                    metrics['robust_hks_correlation'] = desc.hks_correlation
                    metrics['robust_wks_l2_error'] = desc.wks_l2_error
                    metrics['robust_wks_correlation'] = desc.wks_correlation

            # ==============================================================
            # Spectral compression error
            # ==============================================================
            comp_k_values = [5, 10, 20, 50]

            if gt_evecs is not None and gt_M is not None:
                comp = step_spectral_compression(gt_evecs, gt_M, vertices, comp_k_values)
                for k, vals in comp.errors.items():
                    metrics[f'gt_compression_mean_k{k}'] = vals['mean_l2']

            if pred_evecs is not None:
                comp = step_spectral_compression(pred_evecs, pred_M, vertices, comp_k_values)
                for k, vals in comp.errors.items():
                    metrics[f'pred_compression_mean_k{k}'] = vals['mean_l2']

            if rob_evecs is not None:
                comp = step_spectral_compression(rob_evecs, rob_M, vertices, comp_k_values)
                for k, vals in comp.errors.items():
                    metrics[f'robust_compression_mean_k{k}'] = vals['mean_l2']

            # ==============================================================
            # Timing ratios
            # ==============================================================
            if robust_timing.lm_assembly > 0:
                metrics['ratio_lm'] = pred_lm_total / robust_timing.lm_assembly
            if robust_e2e > 0 and metrics.get('pred_e2e_geodesic_ms'):
                metrics['ratio_e2e_geodesic'] = (
                    metrics['pred_e2e_geodesic_ms'] / metrics['robust_e2e_geodesic_ms']
                )

            # ==============================================================
            # Re-enable GC
            # ==============================================================
            gc.enable()

            metrics_list.append(metrics)
            status = "OK"

            del patch_data, pred_result, verts_tensor

        except Exception as e:
            gc.enable()
            status = f"ERROR — {e}"
            import traceback
            traceback.print_exc()

        t_mesh = time.time() - t_mesh_start
        elapsed = time.time() - t_start
        done = local_idx + 1
        eta = (elapsed / done) * (n_total - done) if done > 0 else 0

        pred_e2e_str = f"{metrics.get('pred_e2e_geodesic_ms', 0):.0f}"
        rob_e2e_str = f"{metrics.get('robust_e2e_geodesic_ms', 0):.0f}"
        corr_str = f"{metrics.get('pred_geodesic_corr_mean', 0):.3f}"
        print(f"{tag}[{done}/{n_total}] {mesh_name:<16s} {status:<6s} "
              f"N={metrics.get('num_vertices', 0):>6d} "
              f"PRED={pred_e2e_str:>5s}ms Rob={rob_e2e_str:>5s}ms "
              f"corr={corr_str} "
              f"[{t_mesh:.1f}s, ETA {eta:.0f}s]")

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
    num_sources = getattr(cfg.globals, 'num_validation_sources', 10)
    num_eigenvalues = getattr(cfg.globals, 'num_eigenvalues', 50)

    print(f"\n{'=' * 80}")
    print(f"QUANTITATIVE EVALUATION")
    print(f"{'=' * 80}")
    print(f"Checkpoint:    {ckpt_path}")
    print(f"Meshes:        {total_meshes}")
    print(f"GPUs:          {num_gpus} / {num_gpus_available} available")
    print(f"PRED k:        {pred_k}")
    print(f"Robust k:      {robust_k}")
    print(f"Sources:       {num_sources}")
    print(f"Eigenvalues:   {num_eigenvalues}")
    print(f"Output:        {output_dir}")
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
    print(f"\n{'=' * 80}")
    print(f"SUMMARY ({len(all_metrics)} meshes, {num_sources} sources/mesh)")
    print(f"{'=' * 80}")

    def _stats(key):
        vals = [float(m[key]) for m in all_metrics if m.get(key) is not None]
        if not vals:
            return None, None, None, None
        return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

    W = 10

    # --- Timing ---
    print(f"\n  TIMING (ms)")
    print(f"  {'':>28s} {'Mean':>{W}s} {'Std':>{W}s} {'Min':>{W}s} {'Max':>{W}s}")
    print(f"  {'-' * 28} {'-' * W} {'-' * W} {'-' * W} {'-' * W}")
    for label, key in [
        ("PRED L,M total",         "pred_lm_total_ms"),
        ("GT assembly",            "gt_assembly_ms"),
        ("Robust assembly",        "robust_assembly_ms"),
        ("PRED eigendecomp",       "pred_eigen_ms"),
        ("GT eigendecomp",         "gt_eigen_ms"),
        ("Robust eigendecomp",     "robust_eigen_ms"),
        ("PRED E2E geodesic",      "pred_e2e_geodesic_ms"),
        ("Robust E2E geodesic",    "robust_e2e_geodesic_ms"),
    ]:
        m, s, mn, mx = _stats(key)
        if m is not None:
            print(f"    {label:<26s} {m:>{W}.1f} {s:>{W}.1f} {mn:>{W}.1f} {mx:>{W}.1f}")

    # --- Quality ---
    print(f"\n  QUALITY")
    print(f"  {'':>28s} {'Mean':>{W}s} {'Std':>{W}s}")
    print(f"  {'-' * 28} {'-' * W} {'-' * W}")
    for label, key in [
        ("PRED eval rel err (mean)",   "pred_eval_spectrum_rel_err_mean"),
        ("Robust eval rel err (mean)", "robust_eval_spectrum_rel_err_mean"),
        ("PRED eigvec cos (all)",      "pred_eigvec_cos_all"),
        ("Robust eigvec cos (all)",    "robust_eigvec_cos_all"),
        ("PRED Green's max-princ",     "pred_greens_max_principle"),
        ("Robust Green's max-princ",   "robust_greens_max_principle"),
        ("PRED Green's GT corr",       "pred_greens_gt_corr_mean"),
        ("Robust Green's GT corr",     "robust_greens_gt_corr_mean"),
        ("PRED geodesic corr",         "pred_geodesic_corr_mean"),
        ("Robust geodesic corr",       "robust_geodesic_corr_mean"),
        ("PRED geodesic MAE",          "pred_geodesic_mae_mean"),
        ("Robust geodesic MAE",        "robust_geodesic_mae_mean"),
        ("PRED probe MSE",             "pred_probe_mse"),
        ("Robust probe MSE",           "robust_probe_mse"),
        ("PRED HKS L2 error",          "pred_hks_l2_error"),
        ("Robust HKS L2 error",        "robust_hks_l2_error"),
        ("PRED compress k=20",         "pred_compression_mean_k20"),
        ("Robust compress k=20",       "robust_compression_mean_k20"),
    ]:
        m, s, _, _ = _stats(key)
        if m is not None:
            print(f"    {label:<26s} {m:>{W}.4f} {s:>{W}.4f}")

    # --- Geodesic speed comparison ---
    pred_e2e_vals = [float(m['pred_e2e_geodesic_ms']) for m in all_metrics
                     if m.get('pred_e2e_geodesic_ms')]
    robust_e2e_vals = [float(m['robust_e2e_geodesic_ms']) for m in all_metrics
                       if m.get('robust_e2e_geodesic_ms')]
    if pred_e2e_vals and robust_e2e_vals:
        ratio = np.mean(pred_e2e_vals) / np.mean(robust_e2e_vals)
        wins = sum(1 for p, r in zip(pred_e2e_vals, robust_e2e_vals) if p < r)
        print(f"\n  PRED/Robust E2E ratio: {ratio:.2f}x  "
              f"(PRED wins {wins}/{len(pred_e2e_vals)})")

    print(f"\n  Total time: {elapsed:.1f}s ({len(all_metrics)} meshes, {num_gpus} GPUs)")
    print(f"{'=' * 80}")

    # ---- Cleanup ----
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()