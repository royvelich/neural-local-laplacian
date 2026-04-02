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

    # Top-k weight pruning sweep (quality-sparsity tradeoff)
    python quantitative_eval.py +ckpt_path=model.ckpt '+top_k_values=[3,4,5]'
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from neural_local_laplacian.utils.utils import cuda_warmup, load_mesh
from neural_local_laplacian.utils.validation_utils import (
    step_pred_knn,
    step_pred_inference,
    step_pred_reassemble,
    step_pred_laplacian,
    step_pred_geodesic,
    step_gt_laplacian,
    step_gt_geodesic,
    step_robust_laplacian,
    step_robust_geodesic,
    step_pcdiff_geodesic,
    step_greens_function,
    step_eigendecomposition,
    step_eigenvalue_errors,
    step_eigenvector_cosine_similarity,
    step_exact_geodesics,
    step_geodesic_quality,
    step_descriptor_comparison,
    step_probe_function_mse,
    step_sparsity,
    step_spectral_compression,
    GeodesicTimingBreakdown,
    PredLaplacianTiming,
    HAS_NELO,
    HAS_PCDIFF,
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

def _add_greens_metrics(
    metrics: Dict[str, Any],
    prefix: str,
    L, M,
    source_indices: List[int],
    gt_greens_values,
    lm_timing_total: float,
):
    """Run Green's function for one method and add to metrics dict. Phase 1."""
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
    return greens


def _add_eigen_metrics(
    metrics: Dict[str, Any],
    prefix: str,
    L, M,
    gt_evals, gt_evecs,
    num_eigenpairs: int,
):
    """Run eigen for one method and add to metrics dict. Phase 2."""
    evals, evecs, t_eig = step_eigendecomposition(L, M, num_eigenpairs)
    metrics[f'{prefix}_eigen_ms'] = t_eig['eigen'] * 1000

    if gt_evals is not None and evals is not None:
        ev_err = step_eigenvalue_errors(evals, gt_evals)
        metrics[f'{prefix}_eval_spectrum_rel_err_mean'] = ev_err.spectrum_rel_err_mean
        metrics[f'{prefix}_eval_spectrum_rel_err_max'] = ev_err.spectrum_rel_err_max
        metrics[f'{prefix}_eval_rel_err_mean'] = ev_err.per_eval_rel_err_mean
        metrics[f'{prefix}_eval_rel_err_max'] = ev_err.per_eval_rel_err_max

    if gt_evecs is not None and evecs is not None:
        sims = step_eigenvector_cosine_similarity(gt_evecs, evecs)
        for c in [5, 10, 20, 50]:
            if c <= len(sims):
                metrics[f'{prefix}_eigvec_cos_top{c}'] = float(sims[:c].mean())
        metrics[f'{prefix}_eigvec_cos_all'] = float(sims.mean())
        # Per-eigenvector cosines (for decay curve plots)
        for j, val in enumerate(sims):
            metrics[f'{prefix}_eigvec_cos_{j}'] = float(val)

    return evals, evecs


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
    pred_k = getattr(cfg, 'k_pred', None) or getattr(cfg, 'k', 30)
    robust_k = getattr(cfg, 'k_robust', None) or 30
    nelo_k = getattr(cfg, 'k_nelo', 8)
    num_eigenpairs = getattr(cfg, 'num_eigenpairs', 50)
    use_amp = device.type == 'cuda'

    # ---- Top-k pruning sweep ----
    # Accepts: +top_k_values=[6,10,20] or +'top_k_values=[6,10,20]'
    top_k_raw = getattr(cfg, 'top_k_values', None)
    top_k_values: List[int] = []
    if top_k_raw is not None and run_pred:
        if isinstance(top_k_raw, (list, tuple)):
            top_k_values = sorted([int(x) for x in top_k_raw])
        else:
            # Strip brackets and whitespace, then split
            cleaned = str(top_k_raw).strip().strip('[]()').strip()
            if cleaned:
                top_k_values = sorted([int(x.strip()) for x in cleaned.split(',') if x.strip()])
        top_k_values = [k for k in top_k_values if 2 <= k < pred_k]

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
    mesh_cache: List[Dict[str, Any]] = []
    t_start = time.time()

    # ==================================================================
    # Pre-load all meshes (shared across passes)
    # ==================================================================
    print(f"{tag}Loading {n_total} meshes...")
    mesh_data_list: List[Optional[Dict[str, Any]]] = []
    t_load_start = time.time()
    t_loadmesh_total = 0.0
    t_geodata_total = 0.0

    # Get file paths directly from dataset — no need to call dataset[idx]
    all_mesh_paths = dataset.mesh_file_paths
    num_geo_sources = getattr(dataset, '_num_geodesic_sources', 15)
    geo_source_method = getattr(dataset, '_geodesic_source_method', 'farthest_point_sampling')

    # Import geodesic utilities for direct cache loading
    from neural_local_laplacian.utils.geodesic_utils import (
        select_multiple_geodesic_sources as _select_fps,
        load_cached_geodesics as _load_geo_cache,
    )

    for local_idx, global_idx in enumerate(my_indices):
        try:
            mesh_file_path = str(all_mesh_paths[global_idx])
            mesh_name = Path(mesh_file_path).name

            t0 = time.time()
            vertices, faces, _ = load_mesh(mesh_file_path)
            t_loadmesh = time.time() - t0
            t_loadmesh_total += t_loadmesh
            N = len(vertices)

            # Load geodesic data directly from cache (same logic as MeshDataset)
            t0 = time.time()
            source_indices = None
            exact_geodesics = None
            fps_indices = _select_fps(
                vertices, num_sources=num_geo_sources,
                method=geo_source_method, seed=42,
            )
            source_indices = fps_indices.tolist() if hasattr(fps_indices, 'tolist') else list(fps_indices)
            cached = _load_geo_cache(mesh_file_path, N, fps_indices)
            if cached is not None:
                exact_geodesics = cached
            t_geodata = time.time() - t0
            t_geodata_total += t_geodata

            # Print timing every 50 meshes or on the first one
            if local_idx == 0 or (local_idx + 1) % 50 == 0:
                elapsed = time.time() - t_load_start
                print(f"{tag}  [{local_idx+1}/{n_total}] {mesh_name:<20s} "
                      f"N={N:>6d}  load={t_loadmesh:.3f}s  geo={t_geodata:.3f}s  "
                      f"[cumul: load={t_loadmesh_total:.1f}s  "
                      f"geo={t_geodata_total:.1f}s  wall={elapsed:.1f}s]", flush=True)

            if source_indices is None:
                print(f"{tag}  Warning: no source indices for {mesh_name}, skipping")
                mesh_data_list.append(None)
                metrics_list.append({})
                mesh_cache.append({})
                continue

            mesh_data_list.append({
                'mesh_name': mesh_name,
                'mesh_file_path': mesh_file_path,
                'vertices': vertices,
                'faces': faces,
                'N': N,
                'source_indices': source_indices,
                'exact_geodesics': exact_geodesics,
            })
        except Exception as e:
            print(f"{tag}  Failed to load mesh {global_idx}: {e}")
            mesh_data_list.append(None)

        metrics_list.append({})
        mesh_cache.append({})

    t_load_elapsed = time.time() - t_load_start
    print(f"{tag}Loading done: {t_load_elapsed:.1f}s total  "
          f"(load_mesh={t_loadmesh_total:.1f}s  geodata={t_geodata_total:.1f}s)", flush=True)

    # ==================================================================
    # PASS 1: PRED assembly (GPU, back-to-back, hottest possible)
    #         + top-k reassembly variants (still GPU-warm, assembly only)
    # ==================================================================
    if run_pred:
        tk_str = f" + top_k={top_k_values}" if top_k_values else ""
        print(f"{tag}Pass 1: PRED assembly (k={pred_k}{tk_str})...")
        gc.collect()
        gc.disable()
        for i, md in enumerate(mesh_data_list):
            if md is None:
                continue
            try:
                verts_tensor = torch.from_numpy(md['vertices']).float().to(device)

                # kNN + forward pass + assembly (no pruning)
                patch_data, t_knn = step_pred_knn(verts_tensor, pred_k, device)
                pred_result, t_infer = step_pred_inference(
                    model, patch_data, device, use_amp=use_amp,
                )

                t_pred = PredLaplacianTiming(
                    knn=t_knn['knn'],
                    forward=t_infer['forward'],
                    assembly=t_infer['assembly'],
                    grad_op=t_infer['grad_op'],
                )
                mesh_cache[i]['pred_L'] = pred_result['L']
                mesh_cache[i]['pred_M'] = pred_result['M']
                mesh_cache[i]['pred_G'] = pred_result['G']
                mesh_cache[i]['t_pred'] = t_pred
                metrics_list[i]['pred_k'] = pred_k
                metrics_list[i]['pred_knn_ms'] = t_pred.knn * 1000
                metrics_list[i]['pred_forward_ms'] = t_pred.forward * 1000
                metrics_list[i]['pred_assembly_ms'] = t_pred.assembly * 1000
                metrics_list[i]['pred_grad_op_ms'] = t_pred.grad_op * 1000
                metrics_list[i]['pred_lm_total_ms'] = t_pred.lm_total * 1000

                # Top-k reassembly variants (reuse cached forward pass)
                for tk in top_k_values:
                    tk_L, tk_M, tk_asm_time = step_pred_reassemble(
                        pred_result['fwd_result'], patch_data, device, top_k=tk,
                    )
                    pfx = f'pred_tk{tk}'
                    mesh_cache[i][f'{pfx}_L'] = tk_L
                    mesh_cache[i][f'{pfx}_M'] = tk_M
                    # Timing: same knn + forward, different assembly
                    metrics_list[i][f'{pfx}_assembly_ms'] = tk_asm_time * 1000
                    metrics_list[i][f'{pfx}_lm_total_ms'] = (
                        t_pred.knn + t_pred.forward + tk_asm_time
                    ) * 1000

                del verts_tensor, patch_data
            except Exception as e:
                print(f"{tag}  PRED assembly failed for {md['mesh_name']}: {e}")
        gc.enable()
        print(f"{tag}Pass 1 done: {time.time() - t_start:.1f}s")

    # ==================================================================
    # PASS 2: NeLo assembly (GPU, back-to-back)
    # ==================================================================
    if run_nelo:
        print(f"{tag}Pass 2: NeLo assembly (k={nelo_k})...")
        gc.collect()
        gc.disable()
        for i, md in enumerate(mesh_data_list):
            if md is None:
                continue
            try:
                nelo_L, nelo_M, t_nelo = step_nelo_laplacian(
                    nelo_pipeline, md['vertices'], nelo_k, device,
                )
                mesh_cache[i]['nelo_L'] = nelo_L
                mesh_cache[i]['nelo_M'] = nelo_M
                mesh_cache[i]['t_nelo'] = t_nelo
                metrics_list[i]['nelo_k'] = nelo_k
                metrics_list[i]['nelo_graph_tree_ms'] = t_nelo.graph_tree * 1000
                metrics_list[i]['nelo_forward_ms'] = t_nelo.forward * 1000
                metrics_list[i]['nelo_assembly_ms'] = t_nelo.assembly * 1000
                metrics_list[i]['nelo_lm_total_ms'] = t_nelo.total * 1000
            except Exception as e:
                print(f"{tag}  NeLo assembly failed for {md['mesh_name']}: {e}")
        gc.enable()
        print(f"{tag}Pass 2 done: {time.time() - t_start:.1f}s")

    # ==================================================================
    # PASS 3: Everything else (CPU — GT/Robust assembly, geodesics,
    #         Green's, eigendecomposition, spectral comparisons)
    # ==================================================================
    print(f"{tag}Pass 3: GT/Robust assembly + geodesics + Green's + eigen...")

    for i, md in enumerate(mesh_data_list):
        if md is None:
            continue
        t_mesh_start = time.time()
        metrics = metrics_list[i]
        cache = mesh_cache[i]
        mesh_name = md['mesh_name']
        vertices = md['vertices']
        faces = md['faces']
        N = md['N']
        source_indices = md['source_indices']
        n_src = len(source_indices)

        metrics['mesh_name'] = mesh_name
        metrics['num_vertices'] = N
        metrics['num_faces'] = len(faces)
        metrics['num_sources'] = n_src

        errors = []
        gc.collect()
        gc.disable()

        # ---- GT assembly ----
        gt_L, gt_M, t_gt = None, None, None
        try:
            gt_L, gt_M, t_gt = step_gt_laplacian(vertices, faces)
            metrics['gt_assembly_ms'] = t_gt['assembly'] * 1000
            cache['gt_L'] = gt_L
            cache['gt_M'] = gt_M
        except Exception as e:
            errors.append(f"GT assembly: {e}")

        # ---- Robust assembly ----
        rob_L, rob_M, t_rob_lm = None, None, None
        if run_robust:
            try:
                rob_L, rob_M, t_rob_lm = step_robust_laplacian(vertices, robust_k)
                metrics['robust_k'] = robust_k
                metrics['robust_lm_assembly_ms'] = t_rob_lm['assembly'] * 1000
                cache['rob_L'] = rob_L
                cache['rob_M'] = rob_M
            except Exception as e:
                errors.append(f"Robust assembly: {e}")

        # ---- GT geodesics ----
        gt_distances = None
        if gt_L is not None:
            try:
                gt_distances, gt_geo_timing = step_gt_geodesic(
                    gt_L, gt_M, vertices, faces, source_indices, N,
                )
                metrics['gt_geo_total_ms'] = gt_geo_timing.total * 1000
                metrics['gt_e2e_geodesic_ms'] = (t_gt['assembly'] + gt_geo_timing.total) * 1000
            except Exception as e:
                errors.append(f"GT geodesic: {e}")

        # ---- Robust geodesics ----
        robust_distances = None
        robust_timing = None
        if run_robust and rob_L is not None:
            try:
                robust_distances, robust_timing = step_robust_geodesic(vertices, source_indices, robust_k)
                metrics['robust_constructor_ms'] = robust_timing.constructor * 1000
                metrics['robust_geo_solve_ms'] = robust_timing.solve * 1000
                metrics['robust_e2e_geodesic_ms'] = robust_timing.total * 1000
                metrics['robust_per_src_ms'] = robust_timing.total / n_src * 1000
            except Exception as e:
                errors.append(f"Robust geodesic: {e}")

        # ---- PRED geodesics ----
        pred_distances = None
        pred_L = cache.get('pred_L')
        pred_M = cache.get('pred_M')
        pred_G = cache.get('pred_G')
        t_pred = cache.get('t_pred')
        if run_pred and pred_G is not None:
            try:
                pred_distances, pred_geo_timing = step_pred_geodesic(
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
                metrics['pred_per_src_ms'] = pred_e2e / n_src * 1000
            except Exception as e:
                errors.append(f"PRED geodesic: {e}")

        # ---- NeLo geodesics ----
        nelo_distances = None
        nelo_L = cache.get('nelo_L')
        nelo_M = cache.get('nelo_M')
        t_nelo = cache.get('t_nelo')
        if run_nelo and nelo_L is not None:
            try:
                nelo_distances, nelo_geo_timing = step_pcdiff_geodesic(
                    nelo_L, nelo_M, vertices, nelo_k, source_indices, N,
                )
                nelo_geo_e2e = t_nelo.total + nelo_geo_timing.total
                metrics['nelo_geo_total_ms'] = nelo_geo_timing.total * 1000
                metrics['nelo_e2e_geodesic_ms'] = nelo_geo_e2e * 1000
                metrics['nelo_per_src_ms'] = nelo_geo_e2e / n_src * 1000
            except Exception as e:
                errors.append(f"NeLo geodesic: {e}")

        # ---- Geodesic quality vs exact ----
        exact_geo = md.get('exact_geodesics') or {}
        if not exact_geo:
            try:
                exact_geo = step_exact_geodesics(
                    vertices, faces, source_indices,
                    mesh_file_path=md.get('mesh_file_path'),
                )
            except Exception as e:
                errors.append(f"Exact geodesics: {e}")
                exact_geo = {}
        n_exact = sum(1 for v in exact_geo.values() if v is not None)
        metrics['num_exact_geodesics'] = n_exact

        for prefix, dists in [('gt', gt_distances), ('pred', pred_distances),
                               ('robust', robust_distances), ('nelo', nelo_distances)]:
            if dists is None:
                continue
            try:
                qual = step_geodesic_quality(dists, source_indices, exact_geo)
                metrics[f'{prefix}_geodesic_corr_mean'] = qual.corr_mean
                metrics[f'{prefix}_geodesic_corr_std'] = qual.corr_std
                metrics[f'{prefix}_geodesic_mae_mean'] = qual.mae_mean
                metrics[f'{prefix}_geodesic_max_err_mean'] = qual.max_err_mean
                metrics[f'{prefix}_geodesic_mono_mean'] = qual.mono_mean
            except Exception as e:
                errors.append(f"{prefix} geodesic quality: {e}")

        # ---- Green's function ----
        gt_gvals = None
        if gt_L is not None:
            try:
                gt_greens, t_greens_gt = step_greens_function(gt_L, gt_M, source_indices)
                metrics['gt_greens_total_ms'] = t_greens_gt.total * 1000
                metrics['gt_greens_max_principle'] = gt_greens.max_principle_pass_rate
                gt_gvals = gt_greens.values
            except Exception as e:
                errors.append(f"GT Green's: {e}")

        if run_robust and rob_L is not None:
            _add_greens_metrics(
                metrics, 'robust', rob_L, rob_M,
                source_indices, gt_gvals, t_rob_lm['assembly'] if t_rob_lm else 0,
            )
        if run_pred and pred_L is not None:
            _add_greens_metrics(
                metrics, 'pred', pred_L, pred_M,
                source_indices, gt_gvals, t_pred.lm_total if t_pred else 0,
            )
        if run_nelo and nelo_L is not None:
            _add_greens_metrics(
                metrics, 'nelo', nelo_L, nelo_M,
                source_indices, gt_gvals, t_nelo.total if t_nelo else 0,
            )

        gc.enable()

        # ---- Eigendecomposition (not timing-sensitive) ----
        gt_evals, gt_evecs = None, None
        if gt_L is not None:
            try:
                gt_evals, gt_evecs, t_eig_gt = step_eigendecomposition(
                    gt_L, gt_M, num_eigenpairs,
                )
                metrics['gt_eigen_ms'] = t_eig_gt['eigen'] * 1000
            except Exception as e:
                errors.append(f"GT eigen: {e}")

        pred_evals, pred_evecs = None, None
        if run_pred and pred_L is not None:
            try:
                pred_evals, pred_evecs = _add_eigen_metrics(
                    metrics, 'pred', pred_L, pred_M,
                    gt_evals, gt_evecs, num_eigenpairs,
                )
            except Exception as e:
                errors.append(f"PRED eigen: {e}")
        rob_evals, rob_evecs = None, None
        if run_robust and rob_L is not None:
            try:
                rob_evals, rob_evecs = _add_eigen_metrics(
                    metrics, 'robust', rob_L, rob_M,
                    gt_evals, gt_evecs, num_eigenpairs,
                )
            except Exception as e:
                errors.append(f"Robust eigen: {e}")
        nelo_evals, nelo_evecs = None, None
        if run_nelo and nelo_L is not None:
            try:
                nelo_evals, nelo_evecs = _add_eigen_metrics(
                    metrics, 'nelo', nelo_L, nelo_M,
                    gt_evals, gt_evecs, num_eigenpairs,
                )
            except Exception as e:
                errors.append(f"NeLo eigen: {e}")

        # ---- HKS / WKS descriptor comparison ----
        if gt_evals is not None and gt_evecs is not None:
            for prefix, m_evals, m_evecs in [
                ('pred', pred_evals, pred_evecs),
                ('robust', rob_evals, rob_evecs),
                ('nelo', nelo_evals, nelo_evecs),
            ]:
                if m_evals is None or m_evecs is None:
                    continue
                desc = step_descriptor_comparison(m_evals, m_evecs, gt_evals, gt_evecs)
                metrics[f'{prefix}_hks_corr'] = desc.hks_correlation
                metrics[f'{prefix}_hks_l2_err'] = desc.hks_l2_error
                metrics[f'{prefix}_wks_corr'] = desc.wks_correlation
                metrics[f'{prefix}_wks_l2_err'] = desc.wks_l2_error

        # ---- Probe function MSE ----
        if gt_L is not None and gt_evals is not None and gt_evecs is not None:
            for prefix, m_L, m_M in [
                ('pred', pred_L, pred_M),
                ('robust', rob_L, rob_M),
                ('nelo', nelo_L, nelo_M),
            ]:
                if m_L is None:
                    continue
                probe = step_probe_function_mse(
                    m_L, m_M, gt_L, gt_M, vertices, gt_evals, gt_evecs,
                )
                metrics[f'{prefix}_probe_mse'] = probe.mse
                metrics[f'{prefix}_probe_cosine'] = probe.cosine_similarity
                metrics[f'{prefix}_probe_failure_rate'] = probe.failure_rate

        # ---- Sparsity ----
        if gt_L is not None:
            sp_gt = step_sparsity(gt_L)
            metrics['gt_nnz'] = sp_gt.nnz
            metrics['gt_avg_nnz_per_row'] = sp_gt.avg_nnz_per_row
            metrics['gt_density_pct'] = sp_gt.density_percent
        for prefix, m_L in [('pred', pred_L), ('robust', rob_L), ('nelo', nelo_L)]:
            if m_L is None:
                continue
            sp = step_sparsity(m_L)
            metrics[f'{prefix}_nnz'] = sp.nnz
            metrics[f'{prefix}_avg_nnz_per_row'] = sp.avg_nnz_per_row
            metrics[f'{prefix}_density_pct'] = sp.density_percent

        # ---- Spectral compression ----
        compression_k_values = list(range(5, num_eigenpairs + 1, 5))
        for prefix, m_evecs, m_M in [
            ('gt', gt_evecs, gt_M),
            ('pred', pred_evecs, pred_M),
            ('robust', rob_evecs, rob_M),
            ('nelo', nelo_evecs, nelo_M),
        ]:
            if m_evecs is None or m_M is None:
                continue
            comp = step_spectral_compression(m_evecs, m_M, vertices, compression_k_values)
            for k, errs in comp.errors.items():
                metrics[f'{prefix}_compress_k{k}_mean_l2'] = errs['mean_l2']
                metrics[f'{prefix}_compress_k{k}_max_l2'] = errs['max_l2']

        # ---- Top-k pruning variants (full quality sweep) ----
        for tk in top_k_values:
            pfx = f'pred_tk{tk}'
            tk_L = cache.get(f'{pfx}_L')
            tk_M = cache.get(f'{pfx}_M')
            if tk_L is None:
                continue

            # Green's
            if gt_gvals is not None:
                tk_lm_total = metrics.get(f'{pfx}_lm_total_ms', 0) / 1000
                _add_greens_metrics(
                    metrics, pfx, tk_L, tk_M,
                    source_indices, gt_gvals, tk_lm_total,
                )

            # Eigen
            tk_evals, tk_evecs = None, None
            if gt_evals is not None:
                tk_evals, tk_evecs = _add_eigen_metrics(
                    metrics, pfx, tk_L, tk_M,
                    gt_evals, gt_evecs, num_eigenpairs,
                )

            # Descriptors
            if gt_evals is not None and tk_evals is not None and tk_evecs is not None:
                desc = step_descriptor_comparison(tk_evals, tk_evecs, gt_evals, gt_evecs)
                metrics[f'{pfx}_hks_corr'] = desc.hks_correlation
                metrics[f'{pfx}_hks_l2_err'] = desc.hks_l2_error
                metrics[f'{pfx}_wks_corr'] = desc.wks_correlation
                metrics[f'{pfx}_wks_l2_err'] = desc.wks_l2_error

            # Probe
            if gt_L is not None and gt_evals is not None and gt_evecs is not None:
                probe = step_probe_function_mse(
                    tk_L, tk_M, gt_L, gt_M, vertices, gt_evals, gt_evecs,
                )
                metrics[f'{pfx}_probe_mse'] = probe.mse
                metrics[f'{pfx}_probe_cosine'] = probe.cosine_similarity
                metrics[f'{pfx}_probe_failure_rate'] = probe.failure_rate

            # Sparsity
            sp = step_sparsity(tk_L)
            metrics[f'{pfx}_nnz'] = sp.nnz
            metrics[f'{pfx}_avg_nnz_per_row'] = sp.avg_nnz_per_row
            metrics[f'{pfx}_density_pct'] = sp.density_percent

            # Spectral compression
            if tk_evecs is not None and tk_M is not None:
                comp = step_spectral_compression(tk_evecs, tk_M, vertices, compression_k_values)
                for kc, errs in comp.errors.items():
                    metrics[f'{pfx}_compress_k{kc}_mean_l2'] = errs['mean_l2']
                    metrics[f'{pfx}_compress_k{kc}_max_l2'] = errs['max_l2']

        # ---- Timing ratios ----
        if run_pred and run_robust and robust_timing is not None and t_pred is not None:
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

        # ---- Status ----
        if errors:
            status = f"ERROR — {'; '.join(errors)}"
            metrics['errors'] = '; '.join(errors)
        else:
            status = "OK"

        t_mesh = time.time() - t_mesh_start
        elapsed = time.time() - t_start
        done = i + 1
        eta = (elapsed / max(done, 1)) * (n_total - done)

        parts = [f"{tag}[{done}/{n_total}] {mesh_name:<16s} {status:<6s} N={N:>6d}"]
        if metrics.get('pred_e2e_geodesic_ms'):
            parts.append(f"PRED={metrics['pred_e2e_geodesic_ms']:.0f}ms")
        if metrics.get('robust_e2e_geodesic_ms'):
            parts.append(f"Rob={metrics['robust_e2e_geodesic_ms']:.0f}ms")
        if metrics.get('nelo_e2e_geodesic_ms'):
            parts.append(f"NeLo={metrics['nelo_e2e_geodesic_ms']:.0f}ms")
        if metrics.get('pred_greens_gt_corr_mean'):
            parts.append(f"G_corr={metrics['pred_greens_gt_corr_mean']:.3f}")
        if metrics.get('pred_eigvec_cos_all'):
            parts.append(f"evec={metrics['pred_eigvec_cos_all']:.3f}")
        if metrics.get('pred_geodesic_corr_mean'):
            parts.append(f"geo={metrics['pred_geodesic_corr_mean']:.3f}")
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

@hydra.main(version_base="1.2", config_path='./quantitative_config')
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

    pred_k = getattr(cfg, 'k_pred', None) or getattr(cfg, 'k', 30)
    robust_k = getattr(cfg, 'k_robust', None) or 30
    nelo_k = getattr(cfg, 'k_nelo', 8)
    num_eigenpairs = getattr(cfg, 'num_eigenpairs', 50)

    print(f"\n{'=' * 80}")
    print(f"QUANTITATIVE EVALUATION")
    print(f"{'=' * 80}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"Methods:     GT + {', '.join(sorted(methods))}")
    print(f"Meshes:      {total_meshes}")
    print(f"GPUs:        {num_gpus} / {num_gpus_available} available")
    if 'pred' in methods:
        print(f"PRED k:      {pred_k}")
        top_k_raw = getattr(cfg, 'top_k_values', None)
        if top_k_raw is not None:
            if isinstance(top_k_raw, (list, tuple)):
                print(f"Top-k sweep: {','.join(str(x) for x in top_k_raw)}")
            else:
                print(f"Top-k sweep: {top_k_raw}")
    if 'robust' in methods:
        print(f"Robust k:    {robust_k}")
    if 'nelo' in methods:
        print(f"NeLo k:      {nelo_k}")
        print(f"NeLo ckpt:   {getattr(cfg, 'nelo_ckpt_path', 'N/A')}")
    print(f"Eigenpairs:  {num_eigenpairs}")
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
    _print_summary(all_metrics, methods, num_eigenpairs, elapsed, num_gpus)

    # ---- Generate plots ----
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    _generate_plots(all_metrics, methods, figures_dir, num_eigenpairs)

    # ---- Cleanup ----
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _print_summary(
    all_metrics: List[Dict[str, Any]],
    methods: Set[str],
    num_eigenpairs: int,
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
    # Derive num_sources from metrics (per-mesh, take the most common value)
    src_counts = [int(m['num_sources']) for m in all_metrics if m.get('num_sources') is not None]
    num_sources = max(set(src_counts), key=src_counts.count) if src_counts else 0

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
    _print_timing_row("GT E2E geodesic", "gt_e2e_geodesic_ms")
    for prefix, label in [('pred', 'PRED'), ('robust', 'Robust'), ('nelo', 'NeLo')]:
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

    # --- Geodesic quality vs exact ---
    print(f"\n  GEODESIC QUALITY (vs exact)")
    print(f"  {'':>28s} {'Mean':>{W}s} {'Std':>{W}s}")
    print(f"  {'-' * 28} {'-' * W} {'-' * W}")

    _print_quality_row("GT geodesic corr", "gt_geodesic_corr_mean")
    for prefix, label in [('pred', 'PRED'), ('robust', 'Robust'), ('nelo', 'NeLo')]:
        if prefix not in methods:
            continue
        _print_quality_row(f"{label} geodesic corr", f"{prefix}_geodesic_corr_mean")
        _print_quality_row(f"{label} geodesic MAE", f"{prefix}_geodesic_mae_mean")
        _print_quality_row(f"{label} geodesic mono", f"{prefix}_geodesic_mono_mean")

    # --- Descriptor quality (HKS / WKS) ---
    print(f"\n  DESCRIPTOR QUALITY (HKS / WKS)")
    print(f"  {'':>28s} {'Mean':>{W}s} {'Std':>{W}s}")
    print(f"  {'-' * 28} {'-' * W} {'-' * W}")

    for prefix, label in [('pred', 'PRED'), ('robust', 'Robust'), ('nelo', 'NeLo')]:
        if prefix not in methods:
            continue
        _print_quality_row(f"{label} HKS correlation", f"{prefix}_hks_corr")
        _print_quality_row(f"{label} HKS L2 error", f"{prefix}_hks_l2_err")
        _print_quality_row(f"{label} WKS correlation", f"{prefix}_wks_corr")
        _print_quality_row(f"{label} WKS L2 error", f"{prefix}_wks_l2_err")
        if prefix != list(methods)[-1]:
            print()

    # --- Probe function MSE ---
    print(f"\n  PROBE FUNCTION QUALITY")
    print(f"  {'':>28s} {'Mean':>{W}s} {'Std':>{W}s}")
    print(f"  {'-' * 28} {'-' * W} {'-' * W}")

    for prefix, label in [('pred', 'PRED'), ('robust', 'Robust'), ('nelo', 'NeLo')]:
        if prefix not in methods:
            continue
        _print_quality_row(f"{label} probe MSE", f"{prefix}_probe_mse")
        _print_quality_row(f"{label} probe cosine", f"{prefix}_probe_cosine")
        _print_quality_row(f"{label} probe failure", f"{prefix}_probe_failure_rate")

    # --- Sparsity ---
    print(f"\n  SPARSITY")
    print(f"  {'':>28s} {'Mean':>{W}s} {'Std':>{W}s}")
    print(f"  {'-' * 28} {'-' * W} {'-' * W}")

    _print_quality_row("GT avg NNZ/row", "gt_avg_nnz_per_row")
    for prefix, label in [('pred', 'PRED'), ('robust', 'Robust'), ('nelo', 'NeLo')]:
        if prefix not in methods:
            continue
        _print_quality_row(f"{label} avg NNZ/row", f"{prefix}_avg_nnz_per_row")
        _print_quality_row(f"{label} density %", f"{prefix}_density_pct")

    # --- Spectral compression ---
    print(f"\n  SPECTRAL COMPRESSION (mean L2 recon error)")
    print(f"  {'':>28s} {'Mean':>{W}s} {'Std':>{W}s}")
    print(f"  {'-' * 28} {'-' * W} {'-' * W}")

    for k in range(5, num_eigenpairs + 1, 5):
        _print_quality_row(f"GT k={k}", f"gt_compress_k{k}_mean_l2")
        for prefix, label in [('pred', 'PRED'), ('robust', 'Robust'), ('nelo', 'NeLo')]:
            if prefix not in methods:
                continue
            _print_quality_row(f"{label} k={k}", f"{prefix}_compress_k{k}_mean_l2")
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

    # --- Top-k sweep ---
    tk_values = _detect_top_k_values(all_metrics)
    if tk_values:
        print(f"\n  TOP-K WEIGHT PRUNING SWEEP")
        prefixes = [f'pred_tk{tk}' for tk in tk_values] + ['pred']
        labels = [f'tk={tk}' for tk in tk_values] + ['full']
        header_row = f"  {'Metric':<24s}" + ''.join(f'{l:>{W}s}' for l in labels)
        print(header_row)
        print(f"  {'-' * 24}" + f" {'-' * W}" * len(labels))

        sweep_metrics = [
            ('eigvec cos top5',   '_eigvec_cos_top5'),
            ('eigvec cos all',    '_eigvec_cos_all'),
            ("Green's GT corr",   '_greens_gt_corr_mean'),
            ('probe cosine',      '_probe_cosine'),
            ('HKS correlation',   '_hks_corr'),
            ('WKS correlation',   '_wks_corr'),
            ('avg NNZ/row',       '_avg_nnz_per_row'),
            ('density %',         '_density_pct'),
        ]
        for label, suffix in sweep_metrics:
            parts = [f"  {label:<24s}"]
            for pfx in prefixes:
                m, s, _, _ = _stats(f'{pfx}{suffix}')
                if m is not None:
                    parts.append(f'{m:>{W}.4f}')
                else:
                    parts.append(f'{"N/A":>{W}s}')
            print(''.join(parts))

    print(f"\n  Total time: {elapsed:.1f}s ({n_meshes} meshes, {num_gpus} GPUs)")
    print(f"{'=' * 80}")


# ============================================================================
# Plotting
# ============================================================================

METHOD_COLORS = {
    'pred':   '#2563EB',
    'robust': '#DC2626',
    'nelo':   '#16A34A',
    'gt':     '#6B7280',
}

METHOD_LABELS = {
    'pred':   'PRED (Ours)',
    'robust': 'Robust',
    'nelo':   'NeLo',
    'gt':     'GT',
}


def _setup_plot_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


def _get_vals(all_metrics: List[Dict], key: str) -> Optional[np.ndarray]:
    """Extract numeric values for a key across all meshes."""
    vals = [float(m[key]) for m in all_metrics if m.get(key) is not None]
    return np.array(vals) if vals else None


def _detect_top_k_values(all_metrics: List[Dict]) -> List[int]:
    """Detect top_k values from metric column names like pred_tk3_*."""
    import re
    tk_set = set()
    for m in all_metrics:
        for key in m.keys():
            match = re.match(r'pred_tk(\d+)_', key)
            if match:
                tk_set.add(int(match.group(1)))
    return sorted(tk_set)


def _generate_plots(
    all_metrics: List[Dict[str, Any]],
    methods: Set[str],
    output_dir: Path,
    num_eigenpairs: int = 50,
):
    """Generate all plots from collected metrics."""
    _setup_plot_style()
    active = [m for m in ['pred', 'robust', 'nelo'] if m in methods]
    print(f"\nGenerating plots to {output_dir}...")

    _plot_timing(all_metrics, active, output_dir)
    _plot_quality(all_metrics, active, output_dir)
    _plot_eigvec_decay(all_metrics, active, output_dir)
    _plot_geodesic_quality(all_metrics, active, output_dir)
    _plot_descriptor_quality(all_metrics, active, output_dir)
    _plot_probe_quality(all_metrics, active, output_dir)
    _plot_sparsity(all_metrics, active, output_dir)
    _plot_spectral_compression(all_metrics, active, output_dir, num_eigenpairs)

    # Top-k sweep (if present)
    tk_values = _detect_top_k_values(all_metrics)
    if tk_values:
        _plot_topk_sweep(all_metrics, tk_values, output_dir)

    print(f"All plots saved to: {output_dir}")


def _plot_timing(all_metrics: List[Dict], methods: List[str], output_dir: Path):
    """Grouped bar chart: L,M assembly, geodesic E2E, Green's E2E."""

    groups = [
        ('L,M Assembly', {
            'pred':   'pred_lm_total_ms',
            'robust': 'robust_lm_assembly_ms',
            'nelo':   'nelo_lm_total_ms',
        }),
        ('Geodesic E2E', {
            'pred':   'pred_e2e_geodesic_ms',
            'robust': 'robust_e2e_geodesic_ms',
            'nelo':   'nelo_e2e_geodesic_ms',
        }),
        ("Green's E2E", {
            'pred':   'pred_greens_e2e_ms',
            'robust': 'robust_greens_e2e_ms',
            'nelo':   'nelo_greens_e2e_ms',
        }),
    ]

    n_groups = len(groups)
    n_methods = len(methods)
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(n_groups)

    for j, method in enumerate(methods):
        means, stds = [], []
        for _, col_map in groups:
            vals = _get_vals(all_metrics, col_map.get(method, ''))
            means.append(vals.mean() if vals is not None else 0)
            stds.append(vals.std() if vals is not None else 0)

        offset = (j - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width,
                      yerr=stds, capsize=3,
                      label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method],
                      edgecolor='white', linewidth=0.5,
                      alpha=0.9, zorder=3)
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f'{mean:.0f}', ha='center', va='bottom', fontsize=8,
                        color=METHOD_COLORS[method], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel('Time (ms)')
    ax.set_title('Timing Comparison')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'timing_bars.{ext}')
    plt.close(fig)
    print(f"  Saved: timing_bars.pdf/png")


def _plot_quality(all_metrics: List[Dict], methods: List[str], output_dir: Path):
    """4-panel: Green's GT corr, max principle, eigenvalue error, eigvec cosine."""

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # (a) Green's GT correlation
    ax = axes[0]
    for j, method in enumerate(methods):
        vals = _get_vals(all_metrics, f'{method}_greens_gt_corr_mean')
        if vals is not None:
            ax.bar(j, vals.mean(), yerr=vals.std(), capsize=4,
                   color=METHOD_COLORS[method], edgecolor='white',
                   linewidth=0.5, alpha=0.9, zorder=3)
            ax.text(j, vals.mean() + vals.std() + 0.005,
                    f'{vals.mean():.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=METHOD_COLORS[method])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=9)
    ax.set_ylabel('Correlation')
    ax.set_title("Green's GT Correlation")
    ax.set_ylim(0.9, 1.01)

    # (b) Max principle
    ax = axes[1]
    for j, method in enumerate(methods):
        vals = _get_vals(all_metrics, f'{method}_greens_max_principle')
        if vals is not None:
            ax.bar(j, vals.mean() * 100, capsize=4,
                   color=METHOD_COLORS[method], edgecolor='white',
                   linewidth=0.5, alpha=0.9, zorder=3)
            ax.text(j, vals.mean() * 100 + 0.5,
                    f'{vals.mean() * 100:.0f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=METHOD_COLORS[method])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=9)
    ax.set_ylabel('Pass Rate (%)')
    ax.set_title("Max Principle")
    ax.set_ylim(0, 110)

    # (c) Eigenvalue relative error
    ax = axes[2]
    for j, method in enumerate(methods):
        vals = _get_vals(all_metrics, f'{method}_eval_spectrum_rel_err_mean')
        if vals is not None:
            ax.bar(j, vals.mean(), yerr=vals.std(), capsize=4,
                   color=METHOD_COLORS[method], edgecolor='white',
                   linewidth=0.5, alpha=0.9, zorder=3)
            ax.text(j, vals.mean() + vals.std() + 0.002,
                    f'{vals.mean():.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=METHOD_COLORS[method])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=9)
    ax.set_ylabel('Relative Error')
    ax.set_title('Eigenvalue Error')
    ax.set_ylim(bottom=0)

    # (d) Eigenvector cosine at top-k
    ax = axes[3]
    k_values = [5, 10, 20, 50]
    n_k = len(k_values)
    bar_width = 0.8 / max(len(methods), 1)
    for j, method in enumerate(methods):
        means, stds = [], []
        for k in k_values:
            vals = _get_vals(all_metrics, f'{method}_eigvec_cos_top{k}')
            means.append(vals.mean() if vals is not None else 0)
            stds.append(vals.std() if vals is not None else 0)
        offset = (j - (len(methods) - 1) / 2) * bar_width
        ax.bar(np.arange(n_k) + offset, means, bar_width,
               yerr=stds, capsize=2,
               label=METHOD_LABELS[method],
               color=METHOD_COLORS[method], edgecolor='white',
               linewidth=0.5, alpha=0.9, zorder=3)
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f'top-{k}' for k in k_values])
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Eigenvector Quality')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)

    fig.suptitle('Quality Metrics', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'quality_bars.{ext}')
    plt.close(fig)
    print(f"  Saved: quality_bars.pdf/png")


def _plot_eigvec_decay(all_metrics: List[Dict], methods: List[str], output_dir: Path):
    """Eigenvector cosine similarity vs eigenvector index."""

    # Find max eigenvector index
    max_idx = 0
    for m in all_metrics:
        for key in m.keys():
            for method in methods:
                prefix = f'{method}_eigvec_cos_'
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    if suffix.isdigit():
                        max_idx = max(max_idx, int(suffix))

    if max_idx == 0:
        print("  [!] No per-eigenvector cosine data — skipping decay curve")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for method in methods:
        indices, means, stds = [], [], []
        for idx in range(max_idx + 1):
            vals = _get_vals(all_metrics, f'{method}_eigvec_cos_{idx}')
            if vals is not None:
                indices.append(idx)
                means.append(vals.mean())
                stds.append(vals.std())

        if not indices:
            continue

        indices = np.array(indices)
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(indices, means, '-',
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=2, zorder=3)
        ax.fill_between(indices, means - stds, np.minimum(means + stds, 1.0),
                         color=METHOD_COLORS[method], alpha=0.15, zorder=2)

    ax.set_xlabel('Eigenvector Index')
    ax.set_ylabel('Cosine Similarity with GT')
    ax.set_title('Eigenvector Quality Decay')
    ax.set_xlim(0, max_idx)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axhline(y=1.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.3)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'eigvec_decay.{ext}')
    plt.close(fig)
    print(f"  Saved: eigvec_decay.pdf/png")


def _plot_geodesic_quality(all_metrics: List[Dict], methods: List[str], output_dir: Path):
    """3-panel: geodesic correlation, MAE, monotonicity vs exact."""

    # Include GT as reference
    all_prefixes = ['gt'] + methods

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    panels = [
        ('Correlation with Exact', '_geodesic_corr_mean', True),   # higher is better
        ('MAE (normalized)', '_geodesic_mae_mean', False),          # lower is better
        ('Monotonicity', '_geodesic_mono_mean', True),             # higher is better
    ]

    for ax, (title, suffix, higher_better) in zip(axes, panels):
        present = []
        for prefix in all_prefixes:
            vals = _get_vals(all_metrics, f'{prefix}{suffix}')
            if vals is not None:
                present.append((prefix, vals))

        for j, (prefix, vals) in enumerate(present):
            color = METHOD_COLORS.get(prefix, '#6B7280')
            label = METHOD_LABELS.get(prefix, prefix)
            ax.bar(j, vals.mean(), yerr=vals.std(), capsize=4,
                   color=color, edgecolor='white',
                   linewidth=0.5, alpha=0.9, zorder=3)
            ax.text(j, vals.mean() + vals.std() + 0.005,
                    f'{vals.mean():.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=color)

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([METHOD_LABELS.get(p, p) for p, _ in present], fontsize=9)
        ax.set_title(title)
        if higher_better:
            ax.set_ylim(bottom=max(0, ax.get_ylim()[0]), top=1.05)
        else:
            ax.set_ylim(bottom=0)

    fig.suptitle('Geodesic Quality vs Exact', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'geodesic_quality.{ext}')
    plt.close(fig)
    print(f"  Saved: geodesic_quality.pdf/png")


def _plot_descriptor_quality(all_metrics: List[Dict], methods: List[str], output_dir: Path):
    """4-panel: HKS correlation, HKS L2, WKS correlation, WKS L2."""

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    panels = [
        ('HKS Correlation', '_hks_corr', True),
        ('HKS L2 Error', '_hks_l2_err', False),
        ('WKS Correlation', '_wks_corr', True),
        ('WKS L2 Error', '_wks_l2_err', False),
    ]

    for ax, (title, suffix, higher_better) in zip(axes, panels):
        for j, method in enumerate(methods):
            vals = _get_vals(all_metrics, f'{method}{suffix}')
            if vals is None:
                continue
            # Filter out -1 (N/A) for L2 errors
            if not higher_better:
                vals = vals[vals >= 0]
                if len(vals) == 0:
                    continue
            ax.bar(j, vals.mean(), yerr=vals.std(), capsize=4,
                   color=METHOD_COLORS[method], edgecolor='white',
                   linewidth=0.5, alpha=0.9, zorder=3)
            ax.text(j, vals.mean() + vals.std() + 0.005,
                    f'{vals.mean():.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=METHOD_COLORS[method])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=9)
        ax.set_title(title)
        if higher_better:
            ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
        else:
            ax.set_ylim(bottom=0)

    fig.suptitle('Descriptor Quality (HKS / WKS)', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'descriptor_quality.{ext}')
    plt.close(fig)
    print(f"  Saved: descriptor_quality.pdf/png")


def _plot_probe_quality(all_metrics: List[Dict], methods: List[str], output_dir: Path):
    """3-panel: probe MSE, probe cosine, probe failure rate."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    panels = [
        ('Probe MSE', '_probe_mse', False),
        ('Probe Cosine Similarity', '_probe_cosine', True),
        ('Probe Failure Rate', '_probe_failure_rate', False),
    ]

    for ax, (title, suffix, higher_better) in zip(axes, panels):
        for j, method in enumerate(methods):
            vals = _get_vals(all_metrics, f'{method}{suffix}')
            if vals is None:
                continue
            ax.bar(j, vals.mean(), yerr=vals.std(), capsize=4,
                   color=METHOD_COLORS[method], edgecolor='white',
                   linewidth=0.5, alpha=0.9, zorder=3)
            fmt = f'{vals.mean():.3f}' if vals.mean() < 1 else f'{vals.mean():.1f}'
            ax.text(j, vals.mean() + vals.std() + 0.005,
                    fmt, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=METHOD_COLORS[method])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(bottom=0)

    fig.suptitle('Probe Function Quality', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'probe_quality.{ext}')
    plt.close(fig)
    print(f"  Saved: probe_quality.pdf/png")


def _plot_sparsity(all_metrics: List[Dict], methods: List[str], output_dir: Path):
    """2-panel: avg NNZ per row, density %."""

    all_prefixes = ['gt'] + methods

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    panels = [
        ('Avg NNZ per Row', '_avg_nnz_per_row'),
        ('Density (%)', '_density_pct'),
    ]

    for ax, (title, suffix) in zip(axes, panels):
        present = []
        for prefix in all_prefixes:
            vals = _get_vals(all_metrics, f'{prefix}{suffix}')
            if vals is not None:
                present.append((prefix, vals))

        for j, (prefix, vals) in enumerate(present):
            color = METHOD_COLORS.get(prefix, '#6B7280')
            ax.bar(j, vals.mean(), yerr=vals.std(), capsize=4,
                   color=color, edgecolor='white',
                   linewidth=0.5, alpha=0.9, zorder=3)
            fmt = f'{vals.mean():.1f}' if vals.mean() >= 1 else f'{vals.mean():.4f}'
            ax.text(j, vals.mean() + vals.std() + 0.1,
                    fmt, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=color)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([METHOD_LABELS.get(p, p) for p, _ in present], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(bottom=0)

    fig.suptitle('Laplacian Sparsity', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'sparsity.{ext}')
    plt.close(fig)
    print(f"  Saved: sparsity.pdf/png")


def _plot_spectral_compression(all_metrics: List[Dict], methods: List[str], output_dir: Path,
                               num_eigenpairs: int = 50):
    """Line plot: mean L2 reconstruction error vs number of eigenvectors (k)."""

    all_prefixes = ['gt'] + methods
    k_values = list(range(5, num_eigenpairs + 1, 5))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for prefix in all_prefixes:
        if prefix != 'gt' and prefix not in methods:
            continue

        ks, means, stds = [], [], []
        for k in k_values:
            vals = _get_vals(all_metrics, f'{prefix}_compress_k{k}_mean_l2')
            if vals is not None:
                ks.append(k)
                means.append(vals.mean())
                stds.append(vals.std())

        if not ks:
            continue

        color = METHOD_COLORS.get(prefix, '#6B7280')
        label = METHOD_LABELS.get(prefix, prefix)
        ks = np.array(ks)
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(ks, means, 'o-',
                color=color, label=label,
                linewidth=2, markersize=6, zorder=3)
        ax.fill_between(ks, means - stds, means + stds,
                         color=color, alpha=0.15, zorder=2)

    ax.set_xlabel('Number of Eigenvectors (k)')
    ax.set_ylabel('Mean L2 Reconstruction Error')
    ax.set_title('Spectral Compression Quality')
    ax.set_xticks(list(range(5, num_eigenpairs + 1, 10)))
    ax.set_xticks(k_values, minor=True)
    ax.set_xlim(k_values[0] - 1, k_values[-1] + 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', framealpha=0.9)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'spectral_compression.{ext}')
    plt.close(fig)
    print(f"  Saved: spectral_compression.pdf/png")


def _plot_topk_sweep(all_metrics: List[Dict], tk_values: List[int], output_dir: Path):
    """Quality vs top_k pruning level: shows quality-sparsity tradeoff."""

    # Build x-axis: top_k values + "full" (no pruning)
    # Prefixes: pred_tk3, pred_tk4, ... and pred (full)
    prefixes = [f'pred_tk{tk}' for tk in tk_values] + ['pred']
    x_labels = [f'tk={tk}' for tk in tk_values] + ['full']
    x = np.arange(len(prefixes))

    panels = [
        ('Eigvec Cos (top-5)', '_eigvec_cos_top5', True),
        ('Eigvec Cos (all)', '_eigvec_cos_all', True),
        ("Green's GT Corr", '_greens_gt_corr_mean', True),
        ('Probe Cosine', '_probe_cosine', True),
        ('HKS Correlation', '_hks_corr', True),
        ('Avg NNZ/row', '_avg_nnz_per_row', None),  # None = no direction preference, just show
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (title, suffix, higher_better) in zip(axes, panels):
        means, stds = [], []
        for pfx in prefixes:
            vals = _get_vals(all_metrics, f'{pfx}{suffix}')
            means.append(vals.mean() if vals is not None else 0)
            stds.append(vals.std() if vals is not None else 0)

        colors = ['#93C5FD'] * len(tk_values) + ['#2563EB']  # light blue for pruned, full blue for full
        ax.bar(x, means, yerr=stds, capsize=3,
               color=colors, edgecolor='white', linewidth=0.5,
               alpha=0.9, zorder=3)

        for xi, mean in zip(x, means):
            if mean > 0:
                fmt = f'{mean:.3f}' if mean < 10 else f'{mean:.1f}'
                ax.text(xi, mean + stds[xi] + 0.005, fmt,
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color='#1E40AF')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(title)
        if higher_better is not None:
            if higher_better:
                ax.set_ylim(bottom=max(0, min(means) - 0.1) if means else 0)
            else:
                ax.set_ylim(bottom=0)

    fig.suptitle('Top-k Weight Pruning Sweep', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'topk_sweep.{ext}')
    plt.close(fig)
    print(f"  Saved: topk_sweep.pdf/png")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()