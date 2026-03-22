#!/usr/bin/env python3
"""
Timing sanity check: PRED vs Robust geodesic E2E.

Uses shared step functions from validation_utils.py so timing is identical
to quantitative_eval.py.

Flags:
    +use_amp=False    Disable autocast / BF16 (default: True)

Usage:
    python timing_sanity_check.py +ckpt_path=model.ckpt \
        +data_module=visualize_validation +globals=visualize_validation +model=visualize_validation
"""

import gc
import time
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from neural_local_laplacian.utils.utils import cuda_warmup
from neural_local_laplacian.utils.geodesic_utils import select_multiple_geodesic_sources
from neural_local_laplacian.utils.validation_utils import (
    load_mesh_vertices,
    step_pred_knn,
    step_pred_inference,
    step_pred_geodesic,
    step_robust_geodesic,
    GeodesicTimingBreakdown,
)


def parse_bool_flag(cfg, name, default=False):
    val = getattr(cfg, name, default)
    if isinstance(val, str):
        return val.lower() not in ('false', '0', 'no')
    return bool(val)


@hydra.main(version_base="1.2", config_path='./visualization_config')
def main(cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    use_amp = parse_bool_flag(cfg, 'use_amp', True) and device.type == 'cuda'

    # ---- Load model ----
    ckpt_path = cfg.ckpt_path
    print(f"Loading model from: {ckpt_path}")
    model = LaplacianTransformerModule.load_from_checkpoint(ckpt_path, map_location=device, strict=False)
    model = model.to(device).eval()
    d_model = getattr(model, '_d_model', '?')
    n_layers = getattr(model, '_num_layers', '?')
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  d_model={d_model}, layers={n_layers}, params={n_params:.1f}M")

    has_grad = getattr(model, '_operator_mode', 'stiffness') == 'gradient'
    print(f"Operator mode: {getattr(model, '_operator_mode', 'stiffness')}"
          f" (gradient operator: {'YES' if has_grad else 'NO'})")

    # ---- Config ----
    pred_k = getattr(cfg.globals, 'k_pred', None) or getattr(cfg.model, 'k', 20)
    robust_k = getattr(cfg.globals, 'k_robust', 30)
    max_meshes = getattr(cfg.globals, 'max_meshes', 6)
    num_sources = getattr(cfg.globals, 'num_validation_sources', 10)
    print(f"PRED k={pred_k}, Robust k={robust_k}")
    print(f"Geodesic sources per mesh: {num_sources} (FPS)")
    print(f"AMP (autocast): {'ON' if use_amp else 'OFF'}")

    # ---- CUDA warmup ----
    cuda_warmup(model, device, k=pred_k)

    # ---- Dataset ----
    pl.seed_everything(cfg.globals.seed)
    data_module = hydra.utils.instantiate(cfg.data_module, _recursive_=False)
    data_module.setup('validate')
    data_loader = data_module.val_dataloader()
    if isinstance(data_loader, list):
        data_loader = data_loader[0]
    num_meshes = len(data_loader.dataset)
    print(f"Dataset: {num_meshes} meshes (processing up to {max_meshes})\n")

    # ---- Table header ----
    header = (
        f"{'#':>4}  {'Mesh':<14}  {'N':>6}"
        f"  {'PrLM':>7}  {'PrFact':>7}  {'PrSlv':>7}  {'PrOneT':>7}  {'Pr/src':>7}"
        f"  {'RbOneT':>7}  {'RbSlv':>7}  {'Rb/src':>7}"
        f"  {'LMRat':>6}  {'E2ERat':>7}"
    )
    print(header)
    print("-" * len(header))

    # ---- Disable GC during timing ----
    gc.collect()
    gc.disable()

    # ---- Timing loop ----
    results = []

    for batch_idx, batch_data in enumerate(data_loader):
        if batch_idx >= max_meshes:
            break

        data = batch_data[0] if isinstance(batch_data, list) else batch_data
        mesh_file_path = data.mesh_file_path
        if isinstance(mesh_file_path, list):
            mesh_file_path = mesh_file_path[0]
        mesh_name = Path(mesh_file_path).name

        vertices = load_mesh_vertices(mesh_file_path)
        N = len(vertices)
        source_indices = select_multiple_geodesic_sources(
            vertices.astype(np.float64), num_sources=num_sources,
            method="farthest_point_sampling", seed=42
        ).tolist()
        verts_tensor = torch.from_numpy(vertices).float().to(device)

        # ==============================================================
        # PRED: kNN + inference + geodesic
        # ==============================================================

        patch_data, t_knn = step_pred_knn(verts_tensor, pred_k, device)
        pred_result, t_infer = step_pred_inference(model, patch_data, device, use_amp=use_amp)

        pred_lm_total = t_knn['knn'] + t_infer['forward'] + t_infer['assembly']

        if pred_result['G'] is not None:
            pred_geo_distances, pred_geo_timing = step_pred_geodesic(
                pred_result['L'], pred_result['M'], pred_result['G'],
                source_indices, N,
            )
        else:
            pred_geo_distances = []
            pred_geo_timing = GeodesicTimingBreakdown()

        # ==============================================================
        # Robust: pp3d geodesic
        # ==============================================================

        robust_distances, robust_timing = step_robust_geodesic(vertices, source_indices, robust_k)

        # ==============================================================
        # Compute ratios
        # ==============================================================

        n_src = len(source_indices)

        pred_onetime = (pred_lm_total + t_infer['grad_op']
                        + pred_geo_timing.build
                        + pred_geo_timing.heat_factorize
                        + pred_geo_timing.poisson_factorize)

        pred_e2e_total = pred_onetime + pred_geo_timing.solve
        robust_e2e_total = robust_timing.total

        pred_per_src = pred_e2e_total / n_src
        robust_per_src = robust_e2e_total / n_src

        ratio_lm = pred_lm_total / robust_timing.lm_assembly if robust_timing.lm_assembly > 0 else float('inf')
        ratio_e2e = pred_e2e_total / robust_e2e_total if robust_e2e_total > 0 else float('inf')

        results.append({
            'mesh': mesh_name, 'N': N, 'n_src': n_src,
            'knn_ms': t_knn['knn'] * 1000,
            'fwd_ms': t_infer['forward'] * 1000,
            'asm_ms': t_infer['assembly'] * 1000,
            'grad_op_ms': t_infer['grad_op'] * 1000,
            'pred_lm_ms': pred_lm_total * 1000,
            'pred_build_ms': pred_geo_timing.build * 1000,
            'pred_heat_fact_ms': pred_geo_timing.heat_factorize * 1000,
            'pred_poisson_fact_ms': pred_geo_timing.poisson_factorize * 1000,
            'pred_heat_solve_ms': pred_geo_timing.heat_solve * 1000,
            'pred_poisson_solve_ms': pred_geo_timing.poisson_solve * 1000,
            'pred_onetime_ms': pred_onetime * 1000,
            'pred_solve_ms': pred_geo_timing.solve * 1000,
            'pred_per_src_ms': pred_per_src * 1000,
            'pred_e2e_ms': pred_e2e_total * 1000,
            'robust_lm_ms': robust_timing.lm_assembly * 1000,
            'robust_onetime_ms': robust_timing.onetime * 1000,
            'robust_solve_ms': robust_timing.solve * 1000,
            'robust_per_src_ms': robust_per_src * 1000,
            'robust_e2e_ms': robust_e2e_total * 1000,
            'ratio_lm': ratio_lm,
            'ratio_e2e': ratio_e2e,
        })

        pred_fact_ms = (pred_geo_timing.heat_factorize + pred_geo_timing.poisson_factorize) * 1000
        print(
            f"{batch_idx + 1:>4}  {mesh_name:<14}  {N:>6}"
            f"  {pred_lm_total * 1000:>7.1f}  {pred_fact_ms:>7.1f}"
            f"  {pred_geo_timing.solve * 1000:>7.1f}  {pred_onetime * 1000:>7.1f}  {pred_per_src * 1000:>7.1f}"
            f"  {robust_timing.onetime * 1000:>7.1f}  {robust_timing.solve * 1000:>7.1f}  {robust_per_src * 1000:>7.1f}"
            f"  {ratio_lm:>5.2f}x  {ratio_e2e:>6.2f}x"
        )

        del patch_data, pred_result, verts_tensor

        if device.type == 'cuda':
            torch.cuda.synchronize()

    # ---- Re-enable GC ----
    gc.enable()
    gc.collect()

    # ---- Summary ----
    if results:
        n = len(results)

        def stats(key):
            vals = [r[key] for r in results]
            return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

        n_src = results[0]['n_src']
        W = 10
        print(f"\n{'=' * 80}")
        print(f"SUMMARY ({n} meshes, {n_src} sources/mesh, AMP={'ON' if use_amp else 'OFF'})")
        print(f"{'=' * 80}")
        print(f"{'':>22s} {'Mean':>{W}s} {'Std':>{W}s} {'Min':>{W}s} {'Max':>{W}s}")
        print(f"{'-' * 22} {'-' * W} {'-' * W} {'-' * W} {'-' * W}")

        print("  -- L,M Assembly (for eigen/Green's/HKS) --")
        for label, key in [
            ("PRED L,M",         "pred_lm_ms"),
            ("Robust L,M",       "robust_lm_ms"),
        ]:
            m, s, mn, mx = stats(key)
            print(f"  {label:<20s} {m:>{W}.1f} {s:>{W}.1f} {mn:>{W}.1f} {mx:>{W}.1f}")

        pred_lm = [r['pred_lm_ms'] for r in results]
        robust_lm = [r['robust_lm_ms'] for r in results]
        print(f"  PRED/Robust:         {np.mean(pred_lm) / np.mean(robust_lm):.2f}x  (PRED wins {sum(1 for p, r in zip(pred_lm, robust_lm) if p < r)}/{n})")

        print("  -- E2E Geodesic (one-time + solve) --")
        print("  PRED one-time breakdown:")
        for label, key in [
            ("  kNN",              "knn_ms"),
            ("  forward",          "fwd_ms"),
            ("  assembly",         "asm_ms"),
            ("  grad op",          "grad_op_ms"),
            ("  matrix build",     "pred_build_ms"),
            ("  heat factorize",   "pred_heat_fact_ms"),
            ("  poisson factorize","pred_poisson_fact_ms"),
            ("  TOTAL one-time",   "pred_onetime_ms"),
        ]:
            m, s, mn, mx = stats(key)
            print(f"  {label:<20s} {m:>{W}.1f} {s:>{W}.1f} {mn:>{W}.1f} {mx:>{W}.1f}")

        print("  PRED per-source (total for all sources):")
        for label, key in [
            ("  heat solve",       "pred_heat_solve_ms"),
            ("  poisson solve",    "pred_poisson_solve_ms"),
            ("  TOTAL solve",      "pred_solve_ms"),
        ]:
            m, s, mn, mx = stats(key)
            print(f"  {label:<20s} {m:>{W}.1f} {s:>{W}.1f} {mn:>{W}.1f} {mx:>{W}.1f}")

        print("  Robust (pp3d) breakdown:")
        for label, key in [
            ("  constructor",      "robust_onetime_ms"),
            ("  solve (all src)",  "robust_solve_ms"),
        ]:
            m, s, mn, mx = stats(key)
            print(f"  {label:<20s} {m:>{W}.1f} {s:>{W}.1f} {mn:>{W}.1f} {mx:>{W}.1f}")

        print(f"  {'':>22s} {'PRED':>{W}s} {'Robust':>{W}s}")
        print(f"  {'':>22s} {'-' * W} {'-' * W}")

        pred_e2e = [r['pred_e2e_ms'] for r in results]
        robust_e2e = [r['robust_e2e_ms'] for r in results]
        pred_per_src = [r['pred_per_src_ms'] for r in results]
        robust_per_src = [r['robust_per_src_ms'] for r in results]

        print(f"  {'Total (' + str(n_src) + ' sources)':<22s} {np.mean(pred_e2e):>{W}.1f} {np.mean(robust_e2e):>{W}.1f}")
        print(f"  {'Per source (amort.)':<22s} {np.mean(pred_per_src):>{W}.1f} {np.mean(robust_per_src):>{W}.1f}")
        print(f"  PRED/Robust (total):  {np.mean(pred_e2e) / np.mean(robust_e2e):.2f}x  (PRED wins {sum(1 for p, r in zip(pred_e2e, robust_e2e) if p < r)}/{n})")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()