#!/usr/bin/env python3
"""
Minimal timing sanity check for PRED vs Robust: Laplacian assembly + Heat geodesics.

No polyscope, no eigendecomposition, no visualization — just raw timing.
Single forward pass per mesh. CUDA warmup once at startup.

Flags:
    +use_amp=False             Disable autocast / BF16 (default: True)
    +solver=pypardiso|cholmod  Sparse solver backend (default: scipy)

Usage:
    python timing_sanity_check.py +ckpt_path=model.ckpt \
        +data_module=visualize_validation +globals=visualize_validation +model=visualize_validation

    # With explicit solver:
    python timing_sanity_check.py +ckpt_path=model.ckpt \
        +data_module=visualize_validation +globals=visualize_validation +model=visualize_validation \
        +solver=cholmod
"""

import gc
import time
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
import robust_laplacian
import potpourri3d as pp3d
import hydra
from omegaconf import DictConfig

from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from neural_local_laplacian.utils.utils import (
    normalize_mesh_vertices,
    assemble_stiffness_and_mass_matrices,
    assemble_gradient_operator,
    build_patches_from_vertices,
    cuda_warmup,
)
from neural_local_laplacian.utils.geodesic_utils import (
    compute_heat_geodesic_learned,
    compute_heat_geodesic_learned_batch,
    heat_method_prefactorize,
    heat_method_solve,
    make_grad_div_learned,
    select_multiple_geodesic_sources,
)


def load_mesh_vertices(mesh_file_path: str) -> np.ndarray:
    """Load and normalize mesh vertices."""
    import trimesh
    mesh = trimesh.load(mesh_file_path, process=False, force='mesh')
    vertices = np.array(mesh.vertices, dtype=np.float64)
    vertices = normalize_mesh_vertices(vertices)
    return vertices.astype(np.float32)


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
    ckpt_path = Path(cfg.ckpt_path)
    print(f"Loading model from: {ckpt_path}")
    model = LaplacianTransformerModule.load_from_checkpoint(
        str(ckpt_path), map_location=device,
        normalize_patch_features=True, scale_areas_by_patch_size=True,
    )
    model.eval()
    model.to(device)

    num_layers = len(model.transformer_encoder.layers) if hasattr(model.transformer_encoder, 'layers') else '?'
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  d_model={model._d_model}, layers={num_layers}, params={num_params:.1f}M")

    for module in model.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = False

    operator_mode = getattr(model, '_operator_mode', 'stiffness')
    has_grad = operator_mode == "gradient"
    print(f"Operator mode: {operator_mode} (gradient operator: {'YES' if has_grad else 'NO'})")

    # ---- Config ----
    default_k = getattr(cfg.globals, 'k', 30)
    pred_k = getattr(cfg.globals, 'k_pred', None) or default_k
    robust_k = getattr(cfg.globals, 'k_robust', None) or default_k
    print(f"PRED k={pred_k}, Robust k={robust_k}")
    num_sources = getattr(cfg.globals, 'num_validation_sources', 5)
    print(f"Geodesic sources per mesh: {num_sources} (FPS)")
    print(f"AMP (autocast): {'ON' if use_amp else 'OFF'}")

    # ---- Sparse solver backend ----
    import neural_local_laplacian.utils.geodesic_utils as geo_utils
    solver_backend = str(getattr(cfg, 'solver', 'scipy'))
    if solver_backend != 'scipy':
        geo_utils.set_solver_backend(solver_backend)
    print(f"Sparse solver: {geo_utils._SOLVER_BACKEND}")

    # ---- AMP dtype ----
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    # ---- CUDA warmup (forward + assembly + gradient op) ----
    cuda_warmup(model, device, k=pred_k)

    # ---- Load dataset ----
    pl.seed_everything(cfg.globals.seed)
    data_module = hydra.utils.instantiate(cfg.data_module)
    data_loader = data_module.val_dataloader()
    if isinstance(data_loader, list):
        data_loader = data_loader[0]

    num_meshes = len(data_loader.dataset)
    max_meshes = getattr(cfg.globals, 'max_meshes', num_meshes)
    print(f"Dataset: {num_meshes} meshes (processing up to {max_meshes})\n")

    # ---- Table header ----
    header = (
        f"{'#':>4}  {'Mesh':<14}  {'N':>6}"
        f"  {'kNN':>7}  {'Fwd':>7}  {'Asm':>7}  {'GrOp':>7}  {'PrLM':>7}"
        f"  {'PrFact':>7}  {'PrSlv':>7}  {'Pr/src':>7}"
        f"  {'RbFact':>7}  {'RbSlv':>7}  {'Rb/src':>7}"
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
        # PRED: kNN + forward + L,M assembly + gradient op + geodesic
        # ==============================================================

        # kNN
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        patch_data = build_patches_from_vertices(verts_tensor, pred_k, device=device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_knn = time.perf_counter() - t0

        # Forward pass
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    fwd_result = model._forward_pass(patch_data)
            else:
                fwd_result = model._forward_pass(patch_data)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t0

        # L, M assembly
        stiffness_w = fwd_result['stiffness_weights'].float()
        areas = fwd_result['areas'].float()
        attention_mask = fwd_result['attention_mask']
        vi = patch_data.vertex_indices.to(device)
        ci = patch_data.center_indices.to(device)
        bi = patch_data.patch_idx.to(device)

        t0 = time.perf_counter()
        L_pred, M_pred = assemble_stiffness_and_mass_matrices(
            stiffness_w, areas, attention_mask, vi, ci, bi,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_asm = time.perf_counter() - t0

        pred_lm_total = t_knn + t_fwd + t_asm

        # Gradient operator assembly
        t_grad_op = 0.0
        G_pred = None
        if has_grad and fwd_result.get('grad_coeffs') is not None:
            t0 = time.perf_counter()
            G_pred = assemble_gradient_operator(
                grad_coeffs=fwd_result['grad_coeffs'],
                attention_mask=attention_mask,
                vertex_indices=vi,
                center_indices=ci,
                batch_indices=bi,
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_grad_op = time.perf_counter() - t0

        # PRED geodesic: split into prefactorize (one-time) + solve (per-source)
        t_pred_prefactor = 0.0
        t_pred_solve = 0.0
        prefactored = None
        if G_pred is not None:
            grad_div_fn = make_grad_div_learned(G_pred, M_pred)

            t0 = time.perf_counter()
            prefactored = heat_method_prefactorize(
                S=L_pred, M=M_pred, n_vertices=N, grad_and_div_fn=grad_div_fn,
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_pred_prefactor = time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = heat_method_solve(prefactored, source_indices)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_pred_solve = time.perf_counter() - t0

        # ==============================================================
        # Robust: point_cloud_laplacian + pp3d geodesic
        # ==============================================================

        t0 = time.perf_counter()
        L_rob, M_rob = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=robust_k)
        t_robust_lm = time.perf_counter() - t0

        # pp3d: split into constructor (one-time) + solve (per-source)
        t0 = time.perf_counter()
        pp3d_solver = pp3d.PointCloudHeatSolver(vertices)
        t_robust_prefactor = time.perf_counter() - t0

        t0 = time.perf_counter()
        for src in source_indices:
            _ = pp3d_solver.compute_distance(src)
        t_robust_solve = time.perf_counter() - t0

        n_src = len(source_indices)
        ratio_lm = pred_lm_total / t_robust_lm if t_robust_lm > 0 else float('inf')

        # E2E totals for N sources (one-time + all solves)
        pred_e2e_total = pred_lm_total + t_grad_op + t_pred_prefactor + t_pred_solve
        robust_e2e_total = t_robust_prefactor + t_robust_solve

        # Per-source amortized = one-time / N + solve / N
        pred_per_src = pred_e2e_total / n_src
        robust_per_src = robust_e2e_total / n_src

        ratio_e2e = pred_e2e_total / robust_e2e_total if robust_e2e_total > 0 else float('inf')

        results.append({
            'mesh': mesh_name, 'N': N, 'n_src': n_src,
            'knn_ms': t_knn * 1000, 'fwd_ms': t_fwd * 1000,
            'asm_ms': t_asm * 1000, 'grad_op_ms': t_grad_op * 1000,
            'pred_lm_ms': pred_lm_total * 1000,
            'pred_prefactor_ms': t_pred_prefactor * 1000,
            'pred_solve_ms': t_pred_solve * 1000,
            'pred_per_src_ms': pred_per_src * 1000,
            'pred_e2e_ms': pred_e2e_total * 1000,
            'robust_lm_ms': t_robust_lm * 1000,
            'robust_prefactor_ms': t_robust_prefactor * 1000,
            'robust_solve_ms': t_robust_solve * 1000,
            'robust_per_src_ms': robust_per_src * 1000,
            'robust_e2e_ms': robust_e2e_total * 1000,
            'ratio_lm': ratio_lm,
            'ratio_e2e': ratio_e2e,
        })

        print(
            f"{batch_idx + 1:>4}  {mesh_name:<14}  {N:>6}"
            f"  {t_knn * 1000:>7.1f}  {t_fwd * 1000:>7.1f}  {t_asm * 1000:>7.1f}  {t_grad_op * 1000:>7.1f}  {pred_lm_total * 1000:>7.1f}"
            f"  {t_pred_prefactor * 1000:>7.1f}  {t_pred_solve * 1000:>7.1f}  {pred_per_src * 1000:>7.1f}"
            f"  {t_robust_prefactor * 1000:>7.1f}  {t_robust_solve * 1000:>7.1f}  {robust_per_src * 1000:>7.1f}"
            f"  {ratio_lm:>5.2f}x  {ratio_e2e:>6.2f}x"
        )

        del patch_data, fwd_result, stiffness_w, areas, verts_tensor, G_pred, prefactored

        # Fence: ensure all async GPU work is done before next mesh
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

        W = 10
        n_src = results[0]['n_src']
        print(f"\n{'=' * 80}")
        print(f"SUMMARY ({n} meshes, {n_src} sources/mesh, AMP={'ON' if use_amp else 'OFF'}, solver={geo_utils._SOLVER_BACKEND})")
        print(f"{'=' * 80}")
        print(f"{'':>22s} {'Mean':>{W}s} {'Std':>{W}s} {'Min':>{W}s} {'Max':>{W}s}")
        print(f"{'-' * 22} {'-' * W} {'-' * W} {'-' * W} {'-' * W}")

        # --- L,M Assembly (for non-geodesic tasks) ---
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

        # --- E2E Geodesic: one-time + per-source breakdown ---
        print("  -- E2E Geodesic (one-time + solve) --")
        print("  PRED breakdown:")
        for label, key in [
            ("  kNN",             "knn_ms"),
            ("  forward",         "fwd_ms"),
            ("  assembly",        "asm_ms"),
            ("  grad op",         "grad_op_ms"),
            ("  prefactorize",    "pred_prefactor_ms"),
            ("  solve (all src)", "pred_solve_ms"),
        ]:
            m, s, mn, mx = stats(key)
            print(f"  {label:<20s} {m:>{W}.1f} {s:>{W}.1f} {mn:>{W}.1f} {mx:>{W}.1f}")

        print("  Robust (pp3d) breakdown:")
        for label, key in [
            ("  constructor",     "robust_prefactor_ms"),
            ("  solve (all src)", "robust_solve_ms"),
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