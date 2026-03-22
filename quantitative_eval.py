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
    step_greens_function,
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
    print(f"{tag}Ready — {n_total} meshes on {device}, PRED k={pred_k}, Robust k={robust_k}")

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

            # ---- Disable GC during timing ----
            gc.collect()
            gc.disable()

            # ---- PRED pipeline ----
            patch_data, t_knn = step_pred_knn(verts_tensor, pred_k, device)
            pred_result, t_infer = step_pred_inference(model, patch_data, device, use_amp=use_amp)

            pred_lm_total = t_knn['knn'] + t_infer['forward'] + t_infer['assembly']

            if pred_result['G'] is not None:
                _, pred_geo_timing = step_pred_geodesic(
                    pred_result['L'], pred_result['M'], pred_result['G'],
                    source_indices, N,
                )
            else:
                pred_geo_timing = GeodesicTimingBreakdown()

            # ---- Robust pipeline ----
            _, robust_timing = step_robust_geodesic(vertices, source_indices, robust_k)

            # ---- GT Laplacian (for Green's function reference) ----
            gt_L, gt_M, t_gt = step_gt_laplacian(vertices, faces)

            # ---- Robust Laplacian (for Green's function) ----
            rob_L, rob_M, t_rob_lm = step_robust_laplacian(vertices, robust_k)

            # ---- Green's function (GT → PRED → Robust) ----
            gt_greens = None
            t_greens_gt = None
            if gt_L is not None:
                gt_greens, t_greens_gt = step_greens_function(gt_L, gt_M, source_indices)

            gt_gvals = gt_greens.values if gt_greens is not None else None

            pred_greens, t_greens_pred = step_greens_function(
                pred_result['L'], pred_result['M'], source_indices,
                gt_greens_values=gt_gvals,
            )

            rob_greens, t_greens_rob = step_greens_function(
                rob_L, rob_M, source_indices,
                gt_greens_values=gt_gvals,
            )

            # ---- Re-enable GC ----
            gc.enable()

            # ---- Compute E2E ----
            n_src = len(source_indices)

            pred_onetime = (pred_lm_total + t_infer['grad_op']
                            + pred_geo_timing.build
                            + pred_geo_timing.heat_factorize
                            + pred_geo_timing.poisson_factorize)
            pred_e2e = pred_onetime + pred_geo_timing.solve
            robust_e2e = robust_timing.total

            # Green's E2E: L,M assembly + factorize + solve
            pred_greens_e2e = pred_lm_total + t_greens_pred.total
            robust_greens_e2e = t_rob_lm['assembly'] + t_greens_rob.total

            metrics = {
                'mesh_name': mesh_name,
                'num_vertices': N,
                'num_faces': len(faces),
                'k': pred_k,
                'num_sources': n_src,
                # PRED L,M breakdown (ms)
                'pred_knn_ms': t_knn['knn'] * 1000,
                'pred_forward_ms': t_infer['forward'] * 1000,
                'pred_assembly_ms': t_infer['assembly'] * 1000,
                'pred_grad_op_ms': t_infer['grad_op'] * 1000,
                'pred_lm_total_ms': pred_lm_total * 1000,
                # PRED geodesic breakdown (ms)
                'pred_geo_build_ms': pred_geo_timing.build * 1000,
                'pred_geo_heat_fact_ms': pred_geo_timing.heat_factorize * 1000,
                'pred_geo_poisson_fact_ms': pred_geo_timing.poisson_factorize * 1000,
                'pred_geo_heat_solve_ms': pred_geo_timing.heat_solve * 1000,
                'pred_geo_poisson_solve_ms': pred_geo_timing.poisson_solve * 1000,
                'pred_geo_onetime_ms': pred_onetime * 1000,
                'pred_geo_solve_ms': pred_geo_timing.solve * 1000,
                'pred_e2e_geodesic_ms': pred_e2e * 1000,
                'pred_per_src_ms': pred_e2e / n_src * 1000,
                # Robust breakdown (ms)
                'robust_lm_ms': robust_timing.lm_assembly * 1000,
                'robust_constructor_ms': robust_timing.constructor * 1000,
                'robust_solve_ms': robust_timing.solve * 1000,
                'robust_e2e_geodesic_ms': robust_e2e * 1000,
                'robust_per_src_ms': robust_e2e / n_src * 1000,
                # Ratios (geodesic)
                'ratio_lm': pred_lm_total / robust_timing.lm_assembly if robust_timing.lm_assembly > 0 else None,
                'ratio_e2e': pred_e2e / robust_e2e if robust_e2e > 0 else None,
                # GT assembly (ms)
                'gt_assembly_ms': t_gt['assembly'] * 1000,
                'robust_lm_assembly_ms': t_rob_lm['assembly'] * 1000,
                # Green's function — PRED
                'pred_greens_fact_ms': t_greens_pred.factorize * 1000,
                'pred_greens_solve_ms': t_greens_pred.solve * 1000,
                'pred_greens_total_ms': t_greens_pred.total * 1000,
                'pred_greens_e2e_ms': pred_greens_e2e * 1000,
                'pred_greens_max_principle': pred_greens.max_principle_pass_rate,
                'pred_greens_gt_corr_mean': pred_greens.mean_corr_with_gt,
                'pred_greens_gt_corr_std': pred_greens.std_corr_with_gt,
                'pred_greens_residual_norm': pred_greens.mean_residual_norm,
                # Green's function — Robust
                'robust_greens_fact_ms': t_greens_rob.factorize * 1000,
                'robust_greens_solve_ms': t_greens_rob.solve * 1000,
                'robust_greens_total_ms': t_greens_rob.total * 1000,
                'robust_greens_e2e_ms': robust_greens_e2e * 1000,
                'robust_greens_max_principle': rob_greens.max_principle_pass_rate,
                'robust_greens_gt_corr_mean': rob_greens.mean_corr_with_gt,
                'robust_greens_gt_corr_std': rob_greens.std_corr_with_gt,
                'robust_greens_residual_norm': rob_greens.mean_residual_norm,
            }
            # Green's function — GT (optional, may be None if igl unavailable)
            if t_greens_gt is not None:
                metrics['gt_greens_total_ms'] = t_greens_gt.total * 1000
                metrics['gt_greens_max_principle'] = gt_greens.max_principle_pass_rate
            # Green's E2E ratio
            if robust_greens_e2e > 0:
                metrics['ratio_greens_e2e'] = pred_greens_e2e / robust_greens_e2e

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

        ratio_str = f"{metrics.get('ratio_e2e', 0):.2f}x" if metrics.get('ratio_e2e') else "?"
        greens_corr = f"{metrics.get('pred_greens_gt_corr_mean', 0):.3f}"
        greens_mp = f"{metrics.get('pred_greens_max_principle', 0):.0%}"
        print(f"{tag}[{done}/{n_total}] {mesh_name:<16s} {status:<10s} "
              f"PRED={metrics.get('pred_e2e_geodesic_ms', 0):.0f}ms "
              f"Rob={metrics.get('robust_e2e_geodesic_ms', 0):.0f}ms "
              f"({ratio_str}) "
              f"Green's corr={greens_corr} mp={greens_mp} "
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

    print(f"\n{'=' * 80}")
    print(f"QUANTITATIVE EVALUATION")
    print(f"{'=' * 80}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"Meshes:      {total_meshes}")
    print(f"GPUs:        {num_gpus} / {num_gpus_available} available")
    print(f"PRED k:      {pred_k}")
    print(f"Robust k:    {robust_k}")
    print(f"Sources:     {num_sources}")
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
    print(f"\n{'=' * 80}")
    print(f"SUMMARY ({len(all_metrics)} meshes, {num_sources} sources/mesh)")
    print(f"{'=' * 80}")

    def _stats(key):
        vals = [float(m[key]) for m in all_metrics if m.get(key) is not None]
        if not vals:
            return None, None, None, None
        return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

    W = 10

    print(f"{'':>28s} {'Mean':>{W}s} {'Std':>{W}s} {'Min':>{W}s} {'Max':>{W}s}")
    print(f"{'-' * 28} {'-' * W} {'-' * W} {'-' * W} {'-' * W}")

    for label, key in [
        ("PRED L,M total",         "pred_lm_total_ms"),
        ("Robust L,M",             "robust_lm_ms"),
        ("PRED E2E geodesic",      "pred_e2e_geodesic_ms"),
        ("Robust E2E geodesic",    "robust_e2e_geodesic_ms"),
        ("PRED per-source",        "pred_per_src_ms"),
        ("Robust per-source",      "robust_per_src_ms"),
        ("",                       ""),
        ("PRED Green's E2E",       "pred_greens_e2e_ms"),
        ("Robust Green's E2E",     "robust_greens_e2e_ms"),
        ("PRED Green's solve",     "pred_greens_solve_ms"),
        ("Robust Green's solve",   "robust_greens_solve_ms"),
    ]:
        if not key:
            print()
            continue
        m, s, mn, mx = _stats(key)
        if m is not None:
            print(f"  {label:<26s} {m:>{W}.1f} {s:>{W}.1f} {mn:>{W}.1f} {mx:>{W}.1f}")

    # --- Green's function quality ---
    print(f"\n  GREEN'S FUNCTION QUALITY")
    print(f"  {'':>28s} {'Mean':>{W}s} {'Std':>{W}s}")
    print(f"  {'-' * 28} {'-' * W} {'-' * W}")
    for label, key in [
        ("PRED max principle",     "pred_greens_max_principle"),
        ("Robust max principle",   "robust_greens_max_principle"),
        ("GT max principle",       "gt_greens_max_principle"),
        ("PRED GT correlation",    "pred_greens_gt_corr_mean"),
        ("Robust GT correlation",  "robust_greens_gt_corr_mean"),
        ("PRED residual norm",     "pred_greens_residual_norm"),
        ("Robust residual norm",   "robust_greens_residual_norm"),
    ]:
        m, s, _, _ = _stats(key)
        if m is not None:
            print(f"  {label:<26s} {m:>{W}.4f} {s:>{W}.4f}")

    pred_e2e_vals = [float(m['pred_e2e_geodesic_ms']) for m in all_metrics
                     if m.get('pred_e2e_geodesic_ms')]
    robust_e2e_vals = [float(m['robust_e2e_geodesic_ms']) for m in all_metrics
                       if m.get('robust_e2e_geodesic_ms')]
    if pred_e2e_vals and robust_e2e_vals:
        ratio = np.mean(pred_e2e_vals) / np.mean(robust_e2e_vals)
        wins = sum(1 for p, r in zip(pred_e2e_vals, robust_e2e_vals) if p < r)
        print(f"\n  PRED/Robust E2E geodesic ratio: {ratio:.2f}x  "
              f"(PRED wins {wins}/{len(pred_e2e_vals)})")

    pred_greens_vals = [float(m['pred_greens_e2e_ms']) for m in all_metrics
                        if m.get('pred_greens_e2e_ms')]
    robust_greens_vals = [float(m['robust_greens_e2e_ms']) for m in all_metrics
                          if m.get('robust_greens_e2e_ms')]
    if pred_greens_vals and robust_greens_vals:
        ratio = np.mean(pred_greens_vals) / np.mean(robust_greens_vals)
        wins = sum(1 for p, r in zip(pred_greens_vals, robust_greens_vals) if p < r)
        print(f"  PRED/Robust E2E Green's ratio:  {ratio:.2f}x  "
              f"(PRED wins {wins}/{len(pred_greens_vals)})")

    print(f"\n  Total time: {elapsed:.1f}s ({len(all_metrics)} meshes, {num_gpus} GPUs)")
    print(f"{'=' * 80}")

    # ---- Cleanup ----
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()