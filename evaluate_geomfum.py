#!/usr/bin/env python3
"""
Standalone shape correspondence evaluation using the functional maps pipeline.

Evaluates the neural Laplacian against baselines (robust Laplacian, cotangent
Laplacian) on DT4D Humanoids and/or Animals, using spectral descriptors
(HKS, WKS) and optional ZoomOut refinement.

Usage:
    # Evaluate on DT4D Humanoids
    python evaluate_geomfum.py \\
        --checkpoint model.ckpt \\
        --dataset dt4d_humanoids \\
        --dt4d_root /path/to/DeformingThings4DMatching \\
        --k 20 --num_eigenvectors 50

    # Evaluate on DT4D Animals
    python evaluate_geomfum.py \\
        --checkpoint model.ckpt \\
        --dataset dt4d_animals \\
        --dt4d_root /path/to/DeformingThings4DMatching-Animals \\
        --k 20 --num_eigenvectors 50

    # Evaluate with specific descriptors and ZoomOut
    python evaluate_geomfum.py \\
        --checkpoint model.ckpt \\
        --dataset dt4d_humanoids \\
        --dt4d_root /path/to/DeformingThings4DMatching \\
        --descriptors hks wks \\
        --zoomout --zoomout_k_init 20 --zoomout_k_final 50

    # Baseline only (no neural Laplacian)
    python evaluate_geomfum.py \\
        --no_neural \\
        --dataset dt4d_humanoids \\
        --dt4d_root /path/to/DeformingThings4DMatching
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse
import torch
import robust_laplacian

from fmaps_finetune.datasets.functional_map_dataset import (
    PairSample,
    DT4DHumanoidPairGenerator,
    DT4DAnimalsPairGenerator,
    _stable_hash,
)
from fmaps_finetune.modules.evaluators import (
    ShapePairEvaluator,
    SpectralNNEvaluator,
    FunctionalMapEvaluator,
    fmt_topk,
)


# =============================================================================
# Eigenbasis computation (reused from finetune_functional_maps.py)
# =============================================================================

def compute_knn(vertices_np: np.ndarray, k: int) -> np.ndarray:
    """Compute k-nearest neighbors excluding self."""
    from scipy.spatial import cKDTree
    tree = cKDTree(vertices_np)
    _, indices = tree.query(vertices_np, k=k + 1, workers=-1)
    return indices[:, 1:]


def compute_neural_eigenbasis(
    model,
    vertices_np: np.ndarray,
    k: int,
    num_eigenvectors: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute eigenbasis using the neural Laplacian.

    Returns: (eigenvalues, eigenvectors, mass_diagonal)
    """
    from neural_local_laplacian.utils.utils import (
        assemble_stiffness_and_mass_matrices,
        compute_laplacian_eigendecomposition,
    )
    from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData
    from torch_geometric.data import Batch

    vertices_t = torch.from_numpy(vertices_np).float().to(device)
    knn_np = compute_knn(vertices_np, k)

    # Build patch data
    n, k_nn = vertices_np.shape[0], knn_np.shape[1]
    idx = torch.from_numpy(knn_np).long().to(device)
    patch_pos = vertices_t[idx] - vertices_t[:, None, :]
    all_pos = patch_pos.reshape(-1, 3)
    batch_data = MeshPatchData(
        pos=all_pos, x=all_pos,
        patch_idx=torch.arange(n, device=device).repeat_interleave(k_nn),
        vertex_indices=idx.flatten(),
        center_indices=torch.arange(n, device=device),
    )
    batch_data = Batch.from_data_list([batch_data]).to(device)

    with torch.no_grad():
        fwd = model._forward_pass(batch_data)

    batch_idx = getattr(batch_data, 'patch_idx', batch_data.batch)
    S, M = assemble_stiffness_and_mass_matrices(
        stiffness_weights=fwd['stiffness_weights'],
        areas=fwd['areas'],
        attention_mask=fwd['attention_mask'],
        vertex_indices=batch_data.vertex_indices,
        center_indices=batch_data.center_indices,
        batch_indices=batch_idx,
    )
    eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
        S, num_eigenvectors, mass_matrix=M,
    )
    M_diag = np.array(M.diagonal()).flatten()
    return eigenvalues, eigenvectors, M_diag


def compute_robust_eigenbasis(
    vertices_np: np.ndarray,
    num_eigenvectors: int,
    n_neighbors: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute eigenbasis using robust Laplacian (point cloud mode)."""
    from neural_local_laplacian.utils.utils import compute_laplacian_eigendecomposition
    L, M = robust_laplacian.point_cloud_laplacian(vertices_np, n_neighbors=n_neighbors)
    eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
        L, num_eigenvectors, mass_matrix=M,
    )
    M_diag = np.array(M.diagonal()).flatten()
    return eigenvalues, eigenvectors, M_diag


# =============================================================================
# Evaluation driver
# =============================================================================

def evaluate_all_pairs(
    pairs: List[PairSample],
    evaluators: List[ShapePairEvaluator],
    model=None,
    k: int = 20,
    num_eigenvectors: int = 50,
    device: torch.device = None,
    k_robust: int = 30,
    include_neural: bool = True,
) -> Dict[str, List[Dict[str, float]]]:
    """Run all evaluators on all pairs, for each Laplacian method.

    Returns:
        results[method_label][evaluator.name] = list of per-pair metric dicts
    """
    if device is None:
        device = torch.device('cpu')

    results = {}

    for pair_idx, pair in enumerate(pairs):
        print(f"\n  [{pair_idx+1}/{len(pairs)}] {pair.name}")

        # Build GT correspondence
        n_a = len(pair.verts_a)
        if (len(pair.corr_a) == n_a
                and np.array_equal(pair.corr_a, np.arange(n_a))
                and np.array_equal(pair.corr_b, np.arange(n_a))):
            gt_corr = None  # identity
        else:
            gt_corr = np.full(n_a, -1, dtype=np.int64)
            gt_corr[pair.corr_a] = pair.corr_b
            unmapped = gt_corr == -1
            if unmapped.any():
                from scipy.spatial import cKDTree
                mapped_mask = ~unmapped
                if mapped_mask.any():
                    tree = cKDTree(pair.verts_a[mapped_mask])
                    _, idx = tree.query(pair.verts_a[unmapped])
                    mapped_indices = np.where(mapped_mask)[0]
                    gt_corr[unmapped] = gt_corr[mapped_indices[idx.flatten()]]

        # Compute eigenbases
        bases = {}

        if include_neural and model is not None:
            t0 = time.perf_counter()
            eig_a = compute_neural_eigenbasis(model, pair.verts_a, k, num_eigenvectors, device)
            eig_b = compute_neural_eigenbasis(model, pair.verts_b, k, num_eigenvectors, device)
            t_neural = time.perf_counter() - t0
            bases['neural'] = (eig_a, eig_b)
            print(f"    Neural eigenbasis: {t_neural:.2f}s")

        t0 = time.perf_counter()
        eig_a_r = compute_robust_eigenbasis(pair.verts_a, num_eigenvectors, k_robust)
        eig_b_r = compute_robust_eigenbasis(pair.verts_b, num_eigenvectors, k_robust)
        t_robust = time.perf_counter() - t0
        bases['robust'] = (eig_a_r, eig_b_r)
        print(f"    Robust eigenbasis: {t_robust:.2f}s")

        # Run evaluators
        for method_label, (eig_a, eig_b) in bases.items():
            eigvals_a, eigvecs_a, mass_a = eig_a
            eigvals_b, eigvecs_b, mass_b = eig_b

            for evaluator in evaluators:
                key = f"{method_label}/{evaluator.name}"
                if key not in results:
                    results[key] = []

                t0 = time.perf_counter()
                metrics = evaluator.evaluate(
                    eigvecs_a, eigvecs_b,
                    eigvals_a, eigvals_b,
                    mass_a, mass_b,
                    pair.verts_b,
                    gt_corr=gt_corr,
                )
                t_eval = time.perf_counter() - t0

                results[key].append(metrics)
                print(f"    {key:<40s}  {fmt_topk(metrics)}  ({t_eval:.2f}s)")

    return results


def print_summary(results: Dict[str, List[Dict[str, float]]]):
    """Print aggregated summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Method':<45s}  {'Top-1':>7s}  {'Top-3':>7s}  {'Top-5':>7s}  "
          f"{'Top-10':>7s}  {'Error':>8s}  {'Ortho':>7s}")
    print("-" * 100)

    for key, metrics_list in sorted(results.items()):
        if not metrics_list:
            continue
        avg = {}
        for m_key in metrics_list[0]:
            avg[m_key] = float(np.mean([m[m_key] for m in metrics_list]))

        top1 = avg.get('accuracy', 0) * 100
        top3 = avg.get('top3_acc', 0) * 100
        top5 = avg.get('top5_acc', 0) * 100
        top10 = avg.get('top10_acc', 0) * 100
        err = avg.get('mean_error', 0)
        ortho = avg.get('ortho_error', 0)

        print(f"{key:<45s}  {top1:6.1f}%  {top3:6.1f}%  {top5:6.1f}%  "
              f"{top10:6.1f}%  {err:8.4f}  {ortho:7.3f}")

    print("=" * 100)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Shape correspondence evaluation with functional maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["dt4d_humanoids", "dt4d_animals"],
                        help="Which DT4D dataset to evaluate on")
    parser.add_argument("--dt4d_root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Explicit list of categories to evaluate")
    parser.add_argument("--exclude_categories", nargs="*", default=None,
                        help="Categories to exclude from evaluation")
    parser.add_argument("--poses_per_pair", type=int, default=2,
                        help="Number of pose combinations per category pair")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to neural Laplacian checkpoint")
    parser.add_argument("--no_neural", action="store_true",
                        help="Skip neural Laplacian (baseline-only evaluation)")
    parser.add_argument("--k", type=int, default=20,
                        help="kNN for neural Laplacian")
    parser.add_argument("--k_robust", type=int, default=30,
                        help="kNN for robust Laplacian baseline")
    parser.add_argument("--num_eigenvectors", type=int, default=50,
                        help="Number of eigenvectors to compute")
    parser.add_argument("--device", type=str, default="cuda")

    # Evaluators
    parser.add_argument("--evaluators", nargs="*",
                        default=["spectral_nn", "fmap"],
                        choices=["spectral_nn", "fmap"],
                        help="Which evaluators to run")
    parser.add_argument("--descriptors", nargs="*", default=["hks", "wks"],
                        choices=["hks", "wks"],
                        help="Descriptors for FunctionalMapEvaluator")
    parser.add_argument("--num_descriptor_scales", type=int, default=100,
                        help="Number of time/energy scales per descriptor")
    parser.add_argument("--zoomout", action="store_true",
                        help="Enable ZoomOut refinement")
    parser.add_argument("--zoomout_k_init", type=int, default=20)
    parser.add_argument("--zoomout_k_final", type=int, default=50)
    parser.add_argument("--zoomout_n_iters", type=int, default=10)
    parser.add_argument("--reg_weight", type=float, default=1e-3,
                        help="Regularisation weight for fmap optimisation")

    # Cross-category (humanoids only)
    parser.add_argument("--enable_cross_category", action="store_true",
                        help="[Humanoids] Include cross-category validation pairs")

    # Output
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Directory to save results")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="neural-laplacian-eval")

    args = parser.parse_args()

    # Validate
    if not args.no_neural and args.checkpoint is None:
        parser.error("--checkpoint is required unless --no_neural is set")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    model = None
    if not args.no_neural:
        from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
        print(f"Loading model from: {args.checkpoint}")
        model = LaplacianTransformerModule.load_from_checkpoint(
            args.checkpoint, map_location=device,
            normalize_patch_features=True,
            scale_areas_by_patch_size=True,
        )
        model.eval()
        model.to(device)
        for param in model.parameters():
            param.requires_grad_(False)
        print(f"  Model loaded.")

    # --- Build pair generator ---
    if args.dataset == "dt4d_humanoids":
        pair_gen = DT4DHumanoidPairGenerator(
            root=args.dt4d_root,
            categories=args.categories,
            exclude_categories=args.exclude_categories,
            enable_cross_category=args.enable_cross_category,
        )
    elif args.dataset == "dt4d_animals":
        pair_gen = DT4DAnimalsPairGenerator(
            root=args.dt4d_root,
            categories=args.categories,
            exclude_categories=args.exclude_categories,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # --- Generate val pairs ---
    rng = np.random.RandomState(42)
    pairs = pair_gen.get_val_pairs(rng, poses_per_pair=args.poses_per_pair)
    print(f"\nGenerated {len(pairs)} validation pairs")

    # --- Build evaluators ---
    evaluators = []
    for ev_name in args.evaluators:
        if ev_name == "spectral_nn":
            evaluators.append(SpectralNNEvaluator())
        elif ev_name == "fmap":
            evaluators.append(FunctionalMapEvaluator(
                descriptors=args.descriptors,
                num_descriptor_scales=args.num_descriptor_scales,
                use_zoomout=args.zoomout,
                zoomout_k_init=args.zoomout_k_init,
                zoomout_k_final=args.zoomout_k_final,
                zoomout_n_iters=args.zoomout_n_iters,
                reg_weight=args.reg_weight,
            ))

    print(f"Evaluators: {[e.name for e in evaluators]}")

    # --- Run evaluation ---
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    results = evaluate_all_pairs(
        pairs=pairs,
        evaluators=evaluators,
        model=model,
        k=args.k,
        num_eigenvectors=args.num_eigenvectors,
        device=device,
        k_robust=args.k_robust,
        include_neural=not args.no_neural,
    )

    # --- Summary ---
    print_summary(results)

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-pair results
    serialisable = {}
    for key, metrics_list in results.items():
        serialisable[key] = metrics_list  # list of dicts, all floats
    with open(output_dir / "results.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    # Save args
    with open(output_dir / "eval_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nResults saved to {output_dir}")

    # --- Optional W&B logging ---
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
        for key, metrics_list in results.items():
            avg = {k: float(np.mean([m[k] for m in metrics_list]))
                   for k in metrics_list[0]}
            for m_key, m_val in avg.items():
                wandb.log({f"{key}/{m_key}": m_val})
        wandb.finish()


if __name__ == "__main__":
    main()