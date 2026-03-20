#!/usr/bin/env python3
"""
Multi-GPU Quantitative Evaluation for Neural Local Laplacian.

Runs the same quantitative evaluation as `visualize_validation.py +quantitative_mode=true`
but distributes meshes across all available GPUs for near-linear speedup.

Each GPU gets its own model + NeLo copy and processes an interleaved subset of meshes.
Per-rank CSVs are merged into a single output at the end.

Usage:
    # Use all available GPUs (auto-detected)
    python quantitative_eval.py +ckpt_path=model.ckpt +data_module=... +globals=...

    # Specify number of GPUs
    python quantitative_eval.py +ckpt_path=model.ckpt +num_gpus=4

    # With NeLo comparison
    python quantitative_eval.py +ckpt_path=model.ckpt +nelo_ckpt_path=nelo.ckpt ...

    # Custom output directory
    python quantitative_eval.py +ckpt_path=model.ckpt +output_dir=results/

Single-GPU mode works identically — it just skips mp.spawn overhead.
Compatible with Windows (uses 'spawn' start method).
"""

import csv
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.multiprocessing as mp

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# Import existing infrastructure (no changes to visualize_validation.py needed)
from visualize_validation import (
    RealTimeEigenanalysisVisualizer,
    VisualizationConfig,
    HAS_IGL,
)


# ============================================================================
# Worker function — runs on a single GPU
# ============================================================================

def _worker(rank: int, world_size: int, cfg_dict: dict, vis_config_dict: dict,
            output_dir: str, total_meshes: int):
    """
    Process a subset of meshes on a single GPU.

    Args:
        rank: GPU index (0-based)
        world_size: Total number of GPUs
        cfg_dict: Resolved Hydra config as plain dict (picklable)
        vis_config_dict: VisualizationConfig fields as dict (picklable)
        output_dir: Directory for per-rank CSV output
        total_meshes: Total number of meshes in dataset
    """
    # ---- Device setup ----
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)

    tag = f"[GPU {rank}/{world_size}]"
    print(f"\n{tag} Starting on {device}")

    # ---- Reconstruct config objects ----
    cfg = OmegaConf.create(cfg_dict)
    vis_config = VisualizationConfig(**vis_config_dict)

    # ---- Create visualizer in quantitative mode ----
    visualizer = RealTimeEigenanalysisVisualizer(config=vis_config)
    visualizer._quantitative_mode = True
    visualizer._skip_robust = getattr(cfg, 'skip_robust', False)
    visualizer._diagnostic_mode = False
    visualizer._robust_geodesic_heat = getattr(cfg, 'robust_geodesic_heat', False)

    # ---- Load model ----
    ckpt_path = Path(cfg.ckpt_path)
    model = visualizer.load_trained_model(
        ckpt_path, device, cfg,
        use_torch_compile=False,  # compile per-worker is wasteful, skip for quantitative
        diagnostic_mode=False
    )

    # ---- Load NeLo (optional) ----
    nelo_ckpt = getattr(cfg, 'nelo_ckpt_path', None)
    if nelo_ckpt is not None:
        try:
            visualizer.load_nelo_pipeline(str(nelo_ckpt), device=str(device))
            print(f"{tag} NeLo loaded")
        except Exception as e:
            print(f"{tag} NeLo load failed: {e} — skipping NeLo")

    # ---- Configure k values ----
    default_k = getattr(cfg.globals, 'k', 30)
    visualizer._initial_k_pred = getattr(cfg.globals, 'k_pred', None) or default_k
    visualizer._initial_k_robust = getattr(cfg.globals, 'k_robust', None) or default_k
    visualizer._initial_k_nelo = getattr(cfg.globals, 'k_nelo', None) or default_k
    visualizer.nelo_k = visualizer._initial_k_nelo

    # ---- Seed for reproducibility ----
    pl.seed_everything(cfg.globals.seed)

    # ---- Create dataset ----
    data_module = hydra.utils.instantiate(cfg.data_module)
    data_loader = data_module.val_dataloader()
    if isinstance(data_loader, list):
        data_loader = data_loader[0]
    dataset = data_loader.dataset

    # ---- Compute this worker's mesh indices (interleaved for load balance) ----
    my_indices = list(range(rank, len(dataset), world_size))
    print(f"{tag} Assigned {len(my_indices)}/{len(dataset)} meshes")

    # ---- Process meshes ----
    metrics_list: List[Dict[str, Any]] = []
    t_start = time.time()

    for local_idx, global_idx in enumerate(my_indices):
        mesh_num = f"{local_idx + 1}/{len(my_indices)}"
        print(f"\n{tag} Mesh {mesh_num} (global #{global_idx})")

        try:
            # Load single mesh from dataset (returns a Data object)
            batch_data = dataset[global_idx]

            # process_batch handles both list and single Data
            success = visualizer.process_batch(model, batch_data, global_idx, device)

            if success:
                mesh_metrics = visualizer._collect_mesh_metrics()
                if mesh_metrics is not None:
                    metrics_list.append(mesh_metrics)
                    print(f"{tag} Mesh {mesh_num}: OK ({mesh_metrics.get('mesh_name', '?')})")
            else:
                print(f"{tag} Mesh {mesh_num}: skipped (process_batch returned False)")

        except Exception as e:
            print(f"{tag} Mesh {mesh_num}: ERROR — {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\n{tag} Done — {len(metrics_list)} meshes in {elapsed:.1f}s "
          f"({elapsed / max(len(my_indices), 1):.1f}s/mesh)")

    # ---- Write per-rank CSV ----
    rank_csv = Path(output_dir) / f'metrics_rank{rank}.csv'
    _write_csv(metrics_list, rank_csv)
    print(f"{tag} Saved {rank_csv}")


# ============================================================================
# CSV utilities
# ============================================================================

def _write_csv(metrics_list: List[Dict[str, Any]], csv_path: Path):
    """Write a list of metric dicts to CSV."""
    if not metrics_list:
        return

    # Gather all columns (preserving insertion order across all meshes)
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
    """Read per-rank CSVs and merge into a single list of metric dicts."""
    all_metrics = []

    for rank in range(world_size):
        rank_csv = output_dir / f'metrics_rank{rank}.csv'
        if not rank_csv.exists():
            print(f"[!] Missing {rank_csv}")
            continue

        with open(rank_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric strings back to proper types
                parsed = {}
                for k, v in row.items():
                    if v == '' or v is None:
                        parsed[k] = None
                    else:
                        try:
                            # Try int first (preserves integer types for counts)
                            if '.' not in v and 'e' not in v.lower():
                                parsed[k] = int(v)
                            else:
                                parsed[k] = float(v)
                        except (ValueError, TypeError):
                            parsed[k] = v  # keep as string (e.g. mesh_name)
                all_metrics.append(parsed)

    return all_metrics


# ============================================================================
# Main entry point
# ============================================================================

@hydra.main(version_base="1.2", config_path='./visualization_config')
def main(cfg: DictConfig) -> None:
    """Multi-GPU quantitative evaluation."""

    # ---- Validate config ----
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
        print("[!] No CUDA devices found — running on CPU (slow)")
        num_gpus = 1  # will use CPU in worker

    # ---- Check dependencies ----
    print("Checking dependencies...")
    try:
        from pyFM.mesh import TriMesh
        print("[OK] PyFM available")
    except ImportError:
        raise ImportError("PyFM required. Install with: pip install pyFM")
    if HAS_IGL:
        print("[OK] libigl available")
    else:
        print("[!] libigl not available — GT curvature will be skipped")

    # ---- Build configs ----
    vis_config = VisualizationConfig(
        point_radius=0.005,
        num_eigenvectors_to_show=60,
        colormap='coolwarm',
        enable_eigenvalue_info=True,
        enable_correlation_analysis=True,
        num_validation_sources=getattr(cfg.globals, 'num_validation_sources', 5),
    )

    # ---- Get total mesh count ----
    data_module = hydra.utils.instantiate(cfg.data_module)
    data_loader = data_module.val_dataloader()
    if isinstance(data_loader, list):
        data_loader = data_loader[0]
    total_meshes = len(data_loader.dataset)
    del data_module, data_loader  # free before spawning

    # ---- Output setup ----
    output_dir = Path(getattr(cfg, 'output_dir', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / '.tmp_quantitative_ranks'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"MULTI-GPU QUANTITATIVE EVALUATION")
    print(f"{'=' * 80}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"NeLo:        {getattr(cfg, 'nelo_ckpt_path', 'none')}")
    print(f"Meshes:      {total_meshes}")
    print(f"GPUs:        {num_gpus} / {num_gpus_available} available")
    print(f"Per GPU:     ~{(total_meshes + num_gpus - 1) // num_gpus} meshes")
    print(f"Output:      {output_dir}")
    print(f"{'=' * 80}\n")

    # ---- Serialize config for workers ----
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    vis_config_dict = asdict(vis_config)

    # ---- Run ----
    t_start = time.time()

    if num_gpus == 1:
        # Single GPU — run directly (no mp.spawn overhead, works on Windows)
        _worker(0, 1, cfg_dict, vis_config_dict, str(tmp_dir), total_meshes)
    else:
        # Multi-GPU — spawn workers
        mp.spawn(
            _worker,
            args=(num_gpus, cfg_dict, vis_config_dict, str(tmp_dir), total_meshes),
            nprocs=num_gpus,
            join=True,
        )

    elapsed = time.time() - t_start
    meshes_per_sec = total_meshes / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"All workers completed in {elapsed:.1f}s ({meshes_per_sec:.2f} meshes/sec)")
    print(f"{'=' * 80}")

    # ---- Merge per-rank CSVs ----
    all_metrics = _read_and_merge_csvs(tmp_dir, num_gpus)
    print(f"Merged {len(all_metrics)} mesh results from {num_gpus} workers")

    if len(all_metrics) == 0:
        print("[!] No metrics collected — check worker output above for errors")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # ---- Aggregate and save using existing infrastructure ----
    csv_path = output_dir / 'quantitative_results.csv'
    visualizer = RealTimeEigenanalysisVisualizer(config=vis_config)
    visualizer._aggregate_and_save_results(all_metrics, csv_path)

    # ---- Cleanup ----
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'=' * 80}")
    print(f"QUANTITATIVE EVALUATION COMPLETE")
    print(f"  Meshes processed: {len(all_metrics)}/{total_meshes}")
    print(f"  Total time:       {elapsed:.1f}s ({num_gpus} GPUs)")
    print(f"  Per-mesh avg:     {elapsed / max(len(all_metrics), 1):.1f}s")
    print(f"  CSV:              {csv_path}")
    print(f"  Summary:          {csv_path.with_name(csv_path.stem + '_summary.csv')}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    # 'spawn' start method required for CUDA in child processes.
    # This is the default on Windows; on Linux the default is 'fork' which
    # doesn't work with CUDA. Setting it explicitly ensures cross-platform compat.
    mp.set_start_method('spawn', force=True)
    main()