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
        f"  {'kNN':>8}  {'Fwd':>8}  {'Asm':>8}  {'GradOp':>8}  {'PRED LM':>9}"
        f"  {'PRED E2E':>9}  {'Rob LM':>9}  {'Rob E2E':>9}"
        f"  {'LM Rat':>7}  {'E2E Rat':>8}"
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

        # PRED geodesic (heat method with learned operators, batch — factorize once)
        t_pred_geo = 0.0
        if G_pred is not None:
            t0 = time.perf_counter()
            _ = compute_heat_geodesic_learned_batch(
                S=L_pred, M=M_pred, G=G_pred,
                source_indices=source_indices, n_vertices=N, device=device,
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_pred_geo = (time.perf_counter() - t0) / len(source_indices)

        # ==============================================================
        # Robust: point_cloud_laplacian + pp3d geodesic
        # ==============================================================

        t0 = time.perf_counter()
        L_rob, M_rob = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=robust_k)
        t_robust_lm = time.perf_counter() - t0

        t0 = time.perf_counter()
        solver = pp3d.PointCloudHeatSolver(vertices)
        for src in source_indices:
            _ = solver.compute_distance(src)
        t_robust_geo = (time.perf_counter() - t0) / len(source_indices)

        ratio_lm = pred_lm_total / t_robust_lm if t_robust_lm > 0 else float('inf')
        pred_e2e = pred_lm_total + t_grad_op + t_pred_geo
        # pp3d.PointCloudHeatSolver builds its own internal Laplacian in C++,
        # so t_robust_geo already includes L,M assembly — don't add t_robust_lm.
        robust_e2e = t_robust_geo
        ratio_e2e = pred_e2e / robust_e2e if robust_e2e > 0 else float('inf')

        results.append({
            'mesh': mesh_name, 'N': N,
            'knn_ms': t_knn * 1000, 'fwd_ms': t_fwd * 1000,
            'asm_ms': t_asm * 1000, 'grad_op_ms': t_grad_op * 1000,
            'pred_lm_ms': pred_lm_total * 1000,
            'pred_geo_ms': t_pred_geo * 1000,
            'pred_e2e_ms': pred_e2e * 1000,
            'robust_lm_ms': t_robust_lm * 1000,
            'robust_geo_ms': t_robust_geo * 1000,
            'robust_e2e_ms': robust_e2e * 1000,
            'ratio_lm': ratio_lm,
            'ratio_e2e': ratio_e2e,
        })

        print(
            f"{batch_idx + 1:>4}  {mesh_name:<14}  {N:>6}"
            f"  {t_knn * 1000:>8.1f}  {t_fwd * 1000:>8.1f}  {t_asm * 1000:>8.1f}  {t_grad_op * 1000:>8.1f}  {pred_lm_total * 1000:>9.1f}"
            f"  {pred_e2e * 1000:>9.1f}  {t_robust_lm * 1000:>9.1f}  {robust_e2e * 1000:>9.1f}"
            f"  {ratio_lm:>6.2f}x  {ratio_e2e:>7.2f}x"
        )

        del patch_data, fwd_result, stiffness_w, areas, verts_tensor, G_pred

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
        print(f"\n{'=' * 75}")
        print(f"SUMMARY ({n} meshes, AMP={'ON' if use_amp else 'OFF'}, solver={geo_utils._SOLVER_BACKEND})")
        print(f"{'=' * 75}")
        print(f"{'':>20s} {'Mean':>{W}s} {'Std':>{W}s} {'Min':>{W}s} {'Max':>{W}s}")
        print(f"{'-' * 20} {'-' * W} {'-' * W} {'-' * W} {'-' * W}")

        # --- L,M Assembly comparison ---
        print("  -- L,M Assembly --")
        for label, key in [
            ("PRED kNN",       "knn_ms"),
            ("PRED forward",   "fwd_ms"),
            ("PRED assembly",  "asm_ms"),
            ("PRED grad op",   "grad_op_ms"),
            ("PRED total",     "pred_lm_ms"),
            ("Robust total",   "robust_lm_ms"),
        ]:
            m, s, mn, mx = stats(key)
            print(f"  {label:<18s} {m:>{W}.1f} {s:>{W}.1f} {mn:>{W}.1f} {mx:>{W}.1f}")

        pred_lm = [r['pred_lm_ms'] for r in results]
        robust_lm = [r['robust_lm_ms'] for r in results]
        print(f"  PRED/Robust:       {np.mean(pred_lm) / np.mean(robust_lm):.2f}x  (PRED wins {sum(1 for p, r in zip(pred_lm, robust_lm) if p < r)}/{n})")

        # --- E2E Geodesic comparison ---
        print("  -- E2E Geodesic (L,M + grad op + heat solve) --")
        for label, key in [
            ("PRED E2E",       "pred_e2e_ms"),
            ("Robust E2E*",    "robust_e2e_ms"),
        ]:
            m, s, mn, mx = stats(key)
            print(f"  {label:<18s} {m:>{W}.1f} {s:>{W}.1f} {mn:>{W}.1f} {mx:>{W}.1f}")

        pred_e2e = [r['pred_e2e_ms'] for r in results]
        robust_e2e = [r['robust_e2e_ms'] for r in results]
        print(f"  PRED/Robust:       {np.mean(pred_e2e) / np.mean(robust_e2e):.2f}x  (PRED wins {sum(1 for p, r in zip(pred_e2e, robust_e2e) if p < r)}/{n})")
        print(f"  * Robust E2E = pp3d (C++ solver with internal L,M)")
        print(f"{'=' * 75}")


if __name__ == "__main__":
    main()