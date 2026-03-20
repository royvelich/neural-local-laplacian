#!/usr/bin/env python3
"""
Minimal timing sanity check for PRED vs Robust: Laplacian assembly + Heat geodesics.

No polyscope, no eigendecomposition, no visualization — just raw timing.

Flags:
    +torch_compile=True    Enable torch.compile (default: False)
    +use_amp=False         Disable autocast / BF16 (default: True)
    +fixed_n=10002         Pad all inputs to fixed N (default: 0 = disabled)
    +gpu_keep_warm=True    Keep GPU clocks warm between meshes (default: False)
    +reuse_buffers=True    Pre-allocate GPU buffer, copy_ each mesh (requires fixed_n)
                           Ensures same GPU pointers → cuBLAS cache hits every call.

Usage:
    # Raw timing (no tricks):
    python timing_sanity_check.py +ckpt_path=model.ckpt \
        +data_module=visualize_validation +globals=visualize_validation +model=visualize_validation

    # Fixed N padding (eliminates cuBLAS re-benchmarking):
    python timing_sanity_check.py +ckpt_path=model.ckpt \
        +data_module=visualize_validation +globals=visualize_validation +model=visualize_validation \
        +fixed_n=10002

    # Fixed N + torch.compile (fused kernels, single static shape):
    python timing_sanity_check.py +ckpt_path=model.ckpt \
        +data_module=visualize_validation +globals=visualize_validation +model=visualize_validation \
        +fixed_n=10002 +torch_compile=True
"""

import time
import threading
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
import robust_laplacian
import potpourri3d as pp3d
import hydra
from omegaconf import DictConfig


class GPUKeepWarm:
    """
    Background thread that runs tiny GPU matmuls to prevent clock throttling.

    On Windows, consumer GPUs drop to low-power idle state after a few hundred ms
    of inactivity. The next CUDA call then pays ~100-300ms wake-up penalty.
    This thread keeps the GPU clocks warm by running ~0.01ms matmuls in a loop
    on a separate CUDA stream (so it doesn't interfere with timing).
    """

    def __init__(self, device: torch.device, interval: float = 0.05):
        self._device = device
        self._interval = interval  # seconds between pokes
        self._stop = threading.Event()
        self._thread = None
        # Pre-allocate tiny tensors
        self._a = torch.randn(64, 64, device=device)
        self._b = torch.randn(64, 64, device=device)
        # Separate stream so synchronize() on default stream isn't affected
        self._stream = torch.cuda.Stream(device=device)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.is_set():
            with torch.cuda.stream(self._stream):
                torch.mm(self._a, self._b)
            self._stop.wait(self._interval)

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


def pad_patch_data(patch_data, fixed_n: int, k: int, device: torch.device):
    """
    Pad MeshPatchData from N patches to fixed_n patches.

    Adds (fixed_n - N) dummy patches with zero positions.
    Returns (padded_data, real_n) so outputs can be sliced back.
    """
    real_n = patch_data.center_indices.shape[0]
    if real_n >= fixed_n:
        return patch_data, real_n  # no padding needed

    pad_n = fixed_n - real_n
    pad_points = pad_n * k

    # Pad pos/x: zeros (dummy positions)
    pad_pos = torch.zeros(pad_points, 3, device=device, dtype=patch_data.pos.dtype)
    pos_padded = torch.cat([patch_data.pos, pad_pos], dim=0)

    # Pad patch_idx: continue numbering from real_n
    pad_patch_idx = torch.arange(real_n, fixed_n, device=device).repeat_interleave(k)
    patch_idx_padded = torch.cat([patch_data.patch_idx, pad_patch_idx], dim=0)

    # Pad vertex_indices: point to vertex 0 (harmless — results discarded)
    pad_vi = torch.zeros(pad_points, device=device, dtype=patch_data.vertex_indices.dtype)
    vi_padded = torch.cat([patch_data.vertex_indices, pad_vi], dim=0)

    # Pad center_indices: continue numbering
    pad_ci = torch.arange(real_n, fixed_n, device=device, dtype=patch_data.center_indices.dtype)
    ci_padded = torch.cat([patch_data.center_indices, pad_ci], dim=0)

    return MeshPatchData(
        pos=pos_padded,
        x=pos_padded,
        patch_idx=patch_idx_padded,
        vertex_indices=vi_padded,
        center_indices=ci_padded,
    ), real_n


def unpad_forward_result(fwd_result, real_n: int):
    """Slice forward pass outputs back to real N patches."""
    return {
        'stiffness_weights': fwd_result['stiffness_weights'][:real_n],
        'areas': fwd_result['areas'][:real_n],
        'attention_mask': fwd_result['attention_mask'][:real_n],
        'batch_sizes': fwd_result['batch_sizes'][:real_n],
        'scale_factors': fwd_result.get('scale_factors', None),
        'grad_coeffs': fwd_result['grad_coeffs'][:real_n] if fwd_result.get('grad_coeffs') is not None else None,
    }


def parse_bool_flag(cfg, name, default=False):
    val = getattr(cfg, name, default)
    if isinstance(val, str):
        return val.lower() not in ('false', '0', 'no')
    return bool(val)


@hydra.main(version_base="1.2", config_path='./visualization_config')
def main(cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    use_compile = parse_bool_flag(cfg, 'torch_compile', False)
    use_amp = parse_bool_flag(cfg, 'use_amp', True) and device.type == 'cuda'
    fixed_n = int(getattr(cfg, 'fixed_n', 0))

    # ---- Load model ----
    ckpt_path = Path(cfg.ckpt_path)
    print(f"Loading model from: {ckpt_path}")
    model = LaplacianTransformerModule.load_from_checkpoint(
        str(ckpt_path), map_location=device,
        normalize_patch_features=True, scale_areas_by_patch_size=True,
    )
    model.eval()
    model.to(device)

    # Print model architecture info
    num_layers = len(model.transformer_encoder.layers) if hasattr(model.transformer_encoder, 'layers') else '?'
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  d_model={model._d_model}, layers={num_layers}, params={num_params:.1f}M")

    for module in model.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = False

    if use_compile:
        # dynamic=False when fixed_n is set (single static shape, best optimization)
        # dynamic=True when no fixed_n (variable shapes)
        use_dynamic = (fixed_n == 0)
        mode_str = f"dynamic={use_dynamic}"
        print(f"Applying torch.compile({mode_str})...")
        try:
            model = torch.compile(model, mode="default", dynamic=use_dynamic, fullgraph=False)
            print(f"[OK] torch.compile applied ({mode_str})")
        except Exception as e:
            print(f"[!] torch.compile failed: {e}")
            use_compile = False

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
    print(f"torch.compile: {'ON' if use_compile else 'OFF'}")
    print(f"AMP (autocast): {'ON' if use_amp else 'OFF'}")
    print(f"Fixed N: {fixed_n if fixed_n > 0 else 'OFF'}")

    gpu_keep_warm = parse_bool_flag(cfg, 'gpu_keep_warm', False) and device.type == 'cuda'
    print(f"GPU keep-warm: {'ON' if gpu_keep_warm else 'OFF'}")

    reuse_buffers = parse_bool_flag(cfg, 'reuse_buffers', False)
    if reuse_buffers and fixed_n == 0:
        print("[!] reuse_buffers requires fixed_n — disabling")
        reuse_buffers = False
    print(f"Reuse buffers: {'ON' if reuse_buffers else 'OFF'}")

    # ---- AMP dtype ----
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    # ---- Pre-allocate reusable buffer (same GPU pointers every call → cuBLAS cache hits) ----
    pinned_batch = None
    if reuse_buffers and device.type == 'cuda':
        buf_n = fixed_n
        buf_total = buf_n * pred_k
        pinned_batch = MeshPatchData(
            pos=torch.zeros(buf_total, 3, device=device),
            x=torch.zeros(buf_total, 3, device=device),
            patch_idx=torch.arange(buf_n, device=device).repeat_interleave(pred_k),
            vertex_indices=torch.zeros(buf_total, device=device, dtype=torch.long),
            center_indices=torch.arange(buf_n, device=device, dtype=torch.long),
        )
        print(f"[OK] Pre-allocated buffer: {buf_n} patches × {pred_k} neighbors")

    # ---- Warmup (torch.compile, fixed_n, or reuse_buffers — need at least one pass to cache) ----
    if (use_compile or fixed_n > 0 or reuse_buffers) and device.type == 'cuda':
        # Use pinned buffer for warmup if available (warms cuBLAS on exact same pointers)
        if pinned_batch is not None:
            warmup_batch = pinned_batch
            warmup_batch.pos.normal_()  # fill with random data
            warmup_batch.x = warmup_batch.pos
            print(f"Running warmup on pinned buffer (N={fixed_n}, k={pred_k})...")
        else:
            warmup_n = fixed_n if fixed_n > 0 else 10000
            warmup_batch = MeshPatchData(
                pos=torch.randn(warmup_n * pred_k, 3, device=device),
                x=torch.randn(warmup_n * pred_k, 3, device=device),
                patch_idx=torch.arange(warmup_n, device=device).repeat_interleave(pred_k),
                vertex_indices=torch.randint(0, warmup_n, (warmup_n * pred_k,), device=device),
                center_indices=torch.arange(warmup_n, device=device),
            )
            print(f"Running warmup (N={warmup_n}, k={pred_k})...")
        num_warmup = 3 if use_compile else 1
        with torch.no_grad():
            for _ in range(num_warmup):
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        _ = model._forward_pass(warmup_batch)
                else:
                    _ = model._forward_pass(warmup_batch)
                torch.cuda.synchronize()
        if pinned_batch is None:
            del warmup_batch
        print("[OK] Warmup complete")

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

    # ---- GPU keep-warm (prevents clock throttling during CPU work) ----
    keeper = None
    if gpu_keep_warm:
        keeper = GPUKeepWarm(device)
        keeper.start()
        print("[OK] GPU keep-warm thread started")

    # ---- Disable GC during timing (prevents unpredictable ~100ms pauses) ----
    import gc
    gc.collect()  # clean up before timing
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

        # Pad to fixed_n if requested
        real_n = N
        if fixed_n > 0:
            patch_data, real_n = pad_patch_data(patch_data, fixed_n, pred_k, device)

        # Copy into pinned buffer (same GPU pointers → cuBLAS cache hits)
        if reuse_buffers and pinned_batch is not None:
            pinned_batch.pos.copy_(patch_data.pos)
            pinned_batch.x = pinned_batch.pos  # shared reference
            pinned_batch.vertex_indices.copy_(patch_data.vertex_indices)
            fwd_input = pinned_batch
        else:
            fwd_input = patch_data

        # Forward pass
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    fwd_result = model._forward_pass(fwd_input)
            else:
                fwd_result = model._forward_pass(fwd_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t0

        # Unpad if we padded
        if fixed_n > 0:
            fwd_result = unpad_forward_result(fwd_result, real_n)

        # L, M assembly (always uses real N — unpadded results)
        stiffness_w = fwd_result['stiffness_weights'].float()
        areas = fwd_result['areas'].float()
        attention_mask = fwd_result['attention_mask']
        vi = patch_data.vertex_indices[:real_n * pred_k].to(device) if fixed_n > 0 else patch_data.vertex_indices.to(device)
        ci = patch_data.center_indices[:real_n].to(device) if fixed_n > 0 else patch_data.center_indices.to(device)
        bi = patch_data.patch_idx[:real_n * pred_k].to(device) if fixed_n > 0 else patch_data.patch_idx.to(device)

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

        # L, M
        t0 = time.perf_counter()
        L_rob, M_rob = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=robust_k)
        t_robust_lm = time.perf_counter() - t0

        # Geodesic (potpourri3d — self-contained, averaged over sources)
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

        # Fence: ensure ALL async GPU work from this mesh is done before next
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # ---- Re-enable GC ----
    gc.enable()
    gc.collect()

    # ---- Stop GPU keep-warm ----
    if keeper is not None:
        keeper.stop()

    # ---- Summary ----
    if results:
        n = len(results)

        def stats(key):
            vals = [r[key] for r in results]
            return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

        W = 10
        print(f"\n{'=' * 75}")
        print(f"SUMMARY ({n} meshes, compile={'ON' if use_compile else 'OFF'}, "
              f"AMP={'ON' if use_amp else 'OFF'}, fixed_n={fixed_n if fixed_n > 0 else 'OFF'}, "
              f"reuse_buf={'ON' if reuse_buffers else 'OFF'})")
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