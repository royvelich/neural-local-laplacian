#!/usr/bin/env python3
"""
Minimal timing sanity check for PRED vs Robust: Laplacian assembly + Heat geodesics.

No polyscope, no eigendecomposition, no visualization — just raw timing.
Single forward pass per mesh. CUDA warmup once at startup.

Flags:
    +use_amp=False         Disable autocast / BF16 (default: True)

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
import robust_laplacian
import potpourri3d as pp3d
import hydra
from omegaconf import DictConfig

from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData
from neural_local_laplacian.utils.utils import (
    normalize_mesh_vertices,
    assemble_stiffness_and_mass_matrices,
    assemble_gradient_operator,
    build_patches_from_vertices,
)
from neural_local_laplacian.utils.geodesic_utils import (
    compute_heat_geodesic_learned,
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

    # ---- AMP dtype ----
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    # ---- CUDA warmup (forward + assembly + gradient op) ----
    if device.type == 'cuda':
        warmup_n = 10000
        print(f"Running CUDA warmup (N={warmup_n}, k={pred_k})...")
        warmup_data = MeshPatchData(
            pos=torch.randn(warmup_n * pred_k, 3, device=device),
            x=torch.randn(warmup_n * pred_k, 3, device=device),
            patch_idx=torch.arange(warmup_n, device=device).repeat_interleave(pred_k),
            vertex_indices=torch.randint(0, warmup_n, (warmup_n * pred_k,), device=device),
            center_indices=torch.arange(warmup_n, device=device),
        )
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    warmup_result = model._forward_pass(warmup_data)
            else:
                warmup_result = model._forward_pass(warmup_data)
            torch.cuda.synchronize()
        # Warmup assembly + gradient op (scatter ops have their own cold start)
        _ = assemble_stiffness_and_mass_matrices(
            warmup_result['stiffness_weights'].float(),
            warmup_result['areas'].float(),
            warmup_result['attention_mask'],
            warmup_data.vertex_indices,
            warmup_data.center_indices,
            warmup_data.patch_idx,
        )
        if has_grad and warmup_result.get('grad_coeffs') is not None:
            _ = assemble_gradient_operator(
                grad_coeffs=warmup_result['grad_coeffs'],
                attention_mask=warmup_result['attention_mask'],
                vertex_indices=warmup_data.vertex_indices,
                center_indices=warmup_data.center_indices,
                batch_indices=warmup_data.patch_idx,
            )
        torch.cuda.synchronize()
        del warmup_data, warmup_result
        torch.cuda.empty_cache()
        print("[OK] CUDA warmup complete")

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
        f"  {'PRED Geo':>9}  {'PRED E2E':>9}  {'Rob LM':>9}  {'Rob Geo':>9}  {'Rob E2E':>9}"
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

        # PRED geodesic (heat method with learned operators, averaged over sources)
        t_pred_geo = 0.0
        if G_pred is not None:
            t0 = time.perf_counter()
            for src in source_indices:
                _ = compute_heat_geodesic_learned(
                    S=L_pred, M=M_pred, G=G_pred,
                    source_idx=src, n_vertices=N, device=device,
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
        robust_e2e = t_robust_lm + t_robust_geo
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
            f"  {t_pred_geo * 1000:>9.1f}  {pred_e2e * 1000:>9.1f}  {t_robust_lm * 1000:>9.1f}  {t_robust_geo * 1000:>9.1f}  {robust_e2e * 1000:>9.1f}"
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
        print(f"SUMMARY ({n} meshes, AMP={'ON' if use_amp else 'OFF'})")
        print(f"{'=' * 75}")
        print(f"{'':>20s} {'Mean':>{W}s} {'Std':>{W}s} {'Min':>{W}s} {'Max':>{W}s}")
        print(f"{'-' * 20} {'-' * W} {'-' * W} {'-' * W} {'-' * W}")

        for label, key in [
            ("PRED kNN",       "knn_ms"),
            ("PRED forward",   "fwd_ms"),
            ("PRED assembly",  "asm_ms"),
            ("PRED grad op",   "grad_op_ms"),
            ("PRED L,M total", "pred_lm_ms"),
            ("PRED geodesic",  "pred_geo_ms"),
            ("PRED E2E",       "pred_e2e_ms"),
            ("Robust L,M",     "robust_lm_ms"),
            ("Robust geodesic","robust_geo_ms"),
            ("Robust E2E",     "robust_e2e_ms"),
        ]:
            m, s, mn, mx = stats(key)
            print(f"  {label:<18s} {m:>{W}.1f} {s:>{W}.1f} {mn:>{W}.1f} {mx:>{W}.1f}")

        pred_lm = [r['pred_lm_ms'] for r in results]
        robust_lm = [r['robust_lm_ms'] for r in results]
        pred_e2e = [r['pred_e2e_ms'] for r in results]
        robust_e2e = [r['robust_e2e_ms'] for r in results]

        print(f"\n  L,M ratio (PRED/Robust):     {np.mean(pred_lm) / np.mean(robust_lm):.2f}x")
        print(f"  E2E ratio (PRED/Robust):     {np.mean(pred_e2e) / np.mean(robust_e2e):.2f}x")
        print(f"  PRED L,M wins:               {sum(1 for p, r in zip(pred_lm, robust_lm) if p < r)}/{n}")
        print(f"  PRED E2E wins:               {sum(1 for p, r in zip(pred_e2e, robust_e2e) if p < r)}/{n}")
        print(f"{'=' * 75}")


if __name__ == "__main__":
    main()