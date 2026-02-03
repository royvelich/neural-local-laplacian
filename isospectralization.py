#!/usr/bin/env python3
"""
Isospectralization via Neural Laplacian (Option 1)

Given a target shape A and a source shape B, optimizes the vertex positions of B
so that the eigenvalue spectrum of the neural Laplacian on B matches that of A.

Both spectra are computed using the same frozen neural Laplacian model, ensuring
internal consistency. This demonstrates end-to-end differentiability of the
neural Laplacian pipeline for inverse spectral geometry.

The gradient chain:
    positions → k-NN patches → frozen model → (S_weights, areas)
    → differentiable assembly (S, M) → eigvalsh → loss → ∂loss/∂positions

Usage:
    python isospectralization.py \
        --target path/to/target_shape.obj \
        --source path/to/source_shape.obj \
        --checkpoint path/to/model.ckpt \
        --k 20 --num_eigenvalues 20 --lr 1e-3 --num_steps 500
"""

import argparse
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import polyscope as ps
import polyscope.imgui as psim
import trimesh
from sklearn.neighbors import NearestNeighbors

from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData
from neural_local_laplacian.utils.utils import (
    normalize_mesh_vertices,
    assemble_stiffness_and_mass_matrices,
    compute_laplacian_eigendecomposition,
)
from torch_geometric.data import Batch


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> LaplacianTransformerModule:
    """Load a frozen neural Laplacian model from checkpoint."""
    print(f"Loading model from: {ckpt_path}")
    model = LaplacianTransformerModule.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        normalize_patch_features=True,
        scale_areas_by_patch_size=True,
    )
    model.eval()
    model.to(device)
    # Freeze all parameters - we only optimize vertex positions
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"  Model loaded: input_dim={model._input_dim}, d_model={model._d_model}")
    return model


# ---------------------------------------------------------------------------
# Shape loading
# ---------------------------------------------------------------------------

def load_shape(mesh_path: str) -> np.ndarray:
    """Load a mesh and return normalized vertices."""
    mesh = trimesh.load(str(mesh_path))
    vertices = np.array(mesh.vertices, dtype=np.float32)
    vertices = normalize_mesh_vertices(vertices)
    print(f"  Loaded {mesh_path}: {len(vertices)} vertices")
    return vertices


# ---------------------------------------------------------------------------
# k-NN patch extraction (no gradient through neighbor selection)
# ---------------------------------------------------------------------------

def compute_knn(vertices_np: np.ndarray, k: int) -> np.ndarray:
    """
    Compute k-nearest-neighbor indices (excluding self).

    Args:
        vertices_np: (N, 3) numpy array of vertex positions
        k: number of neighbors

    Returns:
        neighbor_indices: (N, k) numpy array of neighbor indices
    """
    num_vertices = len(vertices_np)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(vertices_np)
    _, neighbor_indices = nbrs.kneighbors(vertices_np)  # (N, k+1)

    # Remove self from neighbor list (vectorized)
    center_positions = np.arange(num_vertices)[:, np.newaxis]
    is_center = neighbor_indices == center_positions
    keep_mask = ~is_center
    keep_positions = np.cumsum(keep_mask, axis=1)
    final_mask = (keep_positions <= k) & keep_mask

    neighbor_indices_flat = neighbor_indices[final_mask]
    return neighbor_indices_flat.reshape(num_vertices, k)


def build_patch_data(
    vertices: torch.Tensor,
    neighbor_indices: np.ndarray,
    device: torch.device,
) -> MeshPatchData:
    """
    Build MeshPatchData from vertex positions and precomputed k-NN.

    Gradient flows through vertex positions → patch positions → features.

    Args:
        vertices: (N, 3) torch tensor WITH gradient tracking
        neighbor_indices: (N, k) numpy array of neighbor indices
        device: torch device

    Returns:
        MeshPatchData ready for model forward pass
    """
    num_vertices = vertices.shape[0]
    k = neighbor_indices.shape[1]

    # Index into differentiable vertex tensor to get neighbor positions
    neighbor_idx_tensor = torch.from_numpy(neighbor_indices).long().to(device)  # (N, k)
    neighbor_positions = vertices[neighbor_idx_tensor]  # (N, k, 3) - differentiable

    # Center each patch at origin: subtract center vertex position
    center_positions = vertices[:, None, :]  # (N, 1, 3) - differentiable
    patch_positions = neighbor_positions - center_positions  # (N, k, 3) - differentiable

    # Flatten for PyG format
    all_positions = patch_positions.reshape(-1, 3)  # (N*k, 3)
    all_features = all_positions  # XYZ as features (same tensor, keeps grad)

    # Index tensors (no gradient needed)
    all_neighbor_indices = neighbor_idx_tensor.flatten()  # (N*k,)
    all_center_indices = torch.arange(num_vertices, device=device)  # (N,)
    batch_indices = torch.arange(num_vertices, device=device).repeat_interleave(k)  # (N*k,)

    data = MeshPatchData(
        pos=all_positions,
        x=all_features,
        patch_idx=batch_indices,
        vertex_indices=all_neighbor_indices,
        center_indices=all_center_indices,
    )
    return data


# ---------------------------------------------------------------------------
# Differentiable Laplacian assembly (stays in PyTorch)
# ---------------------------------------------------------------------------

def assemble_matrices_differentiable(
    stiffness_weights: torch.Tensor,
    areas: torch.Tensor,
    attention_mask: torch.Tensor,
    vertex_indices: torch.Tensor,
    center_indices: torch.Tensor,
    batch_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assemble stiffness (S) and mass (M) matrices as dense PyTorch tensors.

    This is a differentiable version of assemble_stiffness_and_mass_matrices.
    Keeps everything in PyTorch for backpropagation.

    Args:
        stiffness_weights: (num_patches, max_k) predicted edge weights
        areas: (num_patches,) predicted vertex areas
        attention_mask: (num_patches, max_k) True for real tokens
        vertex_indices: (total_points,) neighbor vertex indices
        center_indices: (num_patches,) center vertex index per patch
        batch_indices: (total_points,) patch index per point

    Returns:
        S: (N, N) symmetric stiffness matrix (dense torch tensor)
        M_diag: (N,) diagonal of mass matrix (dense torch tensor)
    """
    device = stiffness_weights.device
    num_patches = stiffness_weights.shape[0]
    max_k = stiffness_weights.shape[1]

    num_vertices = max(vertex_indices.max().item(), center_indices.max().item()) + 1

    # --- Extract valid (non-padded) weights and their indices ---
    weights_flat = stiffness_weights.flatten()  # (num_patches * max_k,)
    mask_flat = attention_mask.flatten()  # (num_patches * max_k,)

    patch_indices_flat = torch.arange(num_patches, device=device).repeat_interleave(max_k)

    valid_weights = weights_flat[mask_flat]  # (num_valid,)
    valid_patch_indices = patch_indices_flat[mask_flat]  # (num_valid,)

    # --- Compute position within each patch (vectorized) ---
    num_valid = len(valid_patch_indices)
    if num_valid > 0:
        patch_changes = torch.ones(num_valid, dtype=torch.bool, device=device)
        if num_valid > 1:
            patch_changes[1:] = valid_patch_indices[1:] != valid_patch_indices[:-1]
        group_ids = torch.cumsum(patch_changes.long(), dim=0) - 1
        change_indices = torch.where(patch_changes)[0]
        group_starts = change_indices[group_ids]
        positions_in_patch = torch.arange(num_valid, device=device, dtype=torch.long) - group_starts
    else:
        positions_in_patch = torch.tensor([], device=device, dtype=torch.long)

    # --- Map to actual vertex indices ---
    batch_sizes = batch_indices.bincount(minlength=num_patches)
    cumsum_sizes = torch.cumsum(batch_sizes, dim=0)
    starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cumsum_sizes[:-1]])

    valid_center_verts = center_indices[valid_patch_indices]  # (num_valid,)
    valid_neighbor_verts = vertex_indices[starts[valid_patch_indices] + positions_in_patch]  # (num_valid,)

    # --- Build dense stiffness matrix S ---
    S = torch.zeros(num_vertices, num_vertices, device=device, dtype=stiffness_weights.dtype)

    # Off-diagonal: S[center, neighbor] -= weight, S[neighbor, center] -= weight
    S.index_put_(
        (valid_center_verts, valid_neighbor_verts),
        -valid_weights,
        accumulate=True,
    )
    S.index_put_(
        (valid_neighbor_verts, valid_center_verts),
        -valid_weights,
        accumulate=True,
    )

    # Diagonal: each row sums to 0
    row_sums = S.sum(dim=1)  # Should be negative
    S = S - torch.diag(row_sums)  # Adds positive values to diagonal

    # Symmetrize (should already be symmetric, but enforce numerically)
    S = 0.5 * (S + S.T)

    # --- Build mass diagonal ---
    M_diag = torch.zeros(num_vertices, device=device, dtype=areas.dtype)
    M_counts = torch.zeros(num_vertices, device=device, dtype=areas.dtype)

    M_diag.scatter_add_(0, center_indices, areas)
    M_counts.scatter_add_(0, center_indices, torch.ones_like(areas))

    # Average areas for vertices that appear as multiple centers
    nonzero = M_counts > 0
    M_diag[nonzero] = M_diag[nonzero] / M_counts[nonzero]

    # Fill vertices never seen as centers with small positive value
    M_diag[~nonzero] = 1e-6

    return S, M_diag


# ---------------------------------------------------------------------------
# Differentiable spectrum computation
# ---------------------------------------------------------------------------

def compute_spectrum(
    model: LaplacianTransformerModule,
    vertices: torch.Tensor,
    neighbor_indices: np.ndarray,
    num_eigenvalues: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the eigenvalue spectrum of the neural Laplacian, differentiably.

    Args:
        model: Frozen neural Laplacian model
        vertices: (N, 3) vertex positions (may require grad)
        neighbor_indices: (N, k) precomputed k-NN indices
        num_eigenvalues: number of eigenvalues to return
        device: torch device

    Returns:
        eigenvalues: (num_eigenvalues,) sorted ascending, differentiable
    """
    # Build patch data (gradient flows through positions)
    batch_data = build_patch_data(vertices, neighbor_indices, device)
    batch_data = Batch.from_data_list([batch_data]).to(device)

    # Model forward pass (frozen weights, but input grad flows)
    forward_result = model._forward_pass(batch_data)

    stiffness_weights = forward_result['stiffness_weights']
    areas = forward_result['areas']
    attention_mask = forward_result['attention_mask']
    batch_sizes = forward_result['batch_sizes']

    batch_idx = getattr(batch_data, 'patch_idx', batch_data.batch)

    # Differentiable assembly
    S, M_diag = assemble_matrices_differentiable(
        stiffness_weights=stiffness_weights,
        areas=areas,
        attention_mask=attention_mask,
        vertex_indices=batch_data.vertex_indices,
        center_indices=batch_data.center_indices,
        batch_indices=batch_idx,
    )

    # Generalized EVP: S v = λ M v  →  M^{-1/2} S M^{-1/2} w = λ w
    M_inv_sqrt = 1.0 / torch.sqrt(M_diag.clamp(min=1e-8))
    M_inv_sqrt_diag = torch.diag(M_inv_sqrt)

    A = M_inv_sqrt_diag @ S @ M_inv_sqrt_diag

    # Symmetrize numerically (important for eigvalsh)
    A = 0.5 * (A + A.T)

    # Differentiable eigenvalue computation (sorted ascending)
    all_eigenvalues = torch.linalg.eigvalsh(A)

    # Return the first num_eigenvalues (smallest)
    return all_eigenvalues[:num_eigenvalues]


# ---------------------------------------------------------------------------
# Non-differentiable spectrum (for target, using scipy - more robust)
# ---------------------------------------------------------------------------

def compute_spectrum_numpy(
    model: LaplacianTransformerModule,
    vertices_np: np.ndarray,
    k: int,
    num_eigenvalues: int,
    device: torch.device,
) -> np.ndarray:
    """
    Compute eigenvalue spectrum using scipy (non-differentiable).
    Used for the target shape where we don't need gradients.
    Falls back to the existing robust scipy pipeline.
    """
    vertices_t = torch.from_numpy(vertices_np).float().to(device)
    neighbor_indices = compute_knn(vertices_np, k)

    # Build patch data and run model
    batch_data = build_patch_data(vertices_t, neighbor_indices, device)
    batch_data = Batch.from_data_list([batch_data]).to(device)

    with torch.no_grad():
        forward_result = model._forward_pass(batch_data)

    stiffness_weights = forward_result['stiffness_weights']
    areas = forward_result['areas']
    attention_mask = forward_result['attention_mask']
    batch_idx = getattr(batch_data, 'patch_idx', batch_data.batch)

    # Use the existing scipy assembly for robustness
    S_scipy, M_scipy = assemble_stiffness_and_mass_matrices(
        stiffness_weights=stiffness_weights,
        areas=areas,
        attention_mask=attention_mask,
        vertex_indices=batch_data.vertex_indices,
        center_indices=batch_data.center_indices,
        batch_indices=batch_idx,
    )

    eigenvalues, _ = compute_laplacian_eigendecomposition(
        S_scipy, num_eigenvalues, mass_matrix=M_scipy
    )
    return eigenvalues


# ---------------------------------------------------------------------------
# Polyscope visualization
# ---------------------------------------------------------------------------

class IsospectralizationVisualizer:
    """Manages polyscope visualization during optimization."""

    def __init__(self, target_verts: np.ndarray, source_verts: np.ndarray,
                 target_spectrum: np.ndarray, num_eigenvalues: int):
        self.target_spectrum = target_spectrum
        self.num_eigenvalues = num_eigenvalues
        self.loss_history = []
        self.current_step = 0
        self.current_loss = float('inf')
        self.current_spectrum = np.zeros(num_eigenvalues)
        self.running = True

        # Initialize polyscope
        ps.init()
        ps.set_up_dir("z_up")
        ps.look_at(camera_location=[2.0, 2.0, 2.0], target=[0, 0, 0])
        ps.set_ground_plane_mode("none")
        ps.set_background_color((0.05, 0.05, 0.05))

        # Register target shape (static)
        self.target_cloud = ps.register_point_cloud(
            "Target Shape", target_verts, radius=0.003, enabled=True
        )
        self.target_cloud.set_color((0.3, 0.3, 0.8))  # Blue-ish
        self.target_cloud.set_transparency(0.4)

        # Register source shape (will be updated)
        self.source_cloud = ps.register_point_cloud(
            "Source Shape (optimizing)", source_verts, radius=0.004, enabled=True
        )
        self.source_cloud.set_color((0.9, 0.5, 0.1))  # Orange

        # Register initial source (static reference)
        self.initial_cloud = ps.register_point_cloud(
            "Source Shape (initial)", source_verts.copy(), radius=0.002, enabled=False
        )
        self.initial_cloud.set_color((0.5, 0.5, 0.5))  # Gray
        self.initial_cloud.set_transparency(0.5)

        self.initial_source_verts = source_verts.copy()

        # Set up UI callback
        ps.set_user_callback(self._ui_callback)

    def _ui_callback(self):
        """ImGui callback for optimization info display."""
        # Scale window height: ~150px base + ~20px per eigenvalue row
        window_height = 200 + 20 * self.num_eigenvalues
        psim.SetNextWindowSize((420, window_height))
        psim.SetNextWindowPos((10, 10))

        opened = psim.Begin("Isospectralization", True)
        if opened:
            psim.Text(f"Step: {self.current_step}")
            psim.Text(f"Loss: {self.current_loss:.6e}")
            psim.Separator()

            # Eigenvalue comparison table
            psim.Text("Eigenvalue Comparison (skip lambda_0):")
            psim.Columns(3, "eig_table")
            psim.Separator()
            psim.Text("Index"); psim.NextColumn()
            psim.Text("Target"); psim.NextColumn()
            psim.Text("Current"); psim.NextColumn()
            psim.Separator()

            n_show = self.num_eigenvalues
            for i in range(1, n_show):  # Skip λ₀
                psim.Text(f"{i}"); psim.NextColumn()
                psim.Text(f"{self.target_spectrum[i]:.4f}"); psim.NextColumn()
                psim.Text(f"{self.current_spectrum[i]:.4f}"); psim.NextColumn()
            psim.Columns(1, "")

            psim.Separator()
            if len(self.loss_history) > 1:
                psim.Text(f"Loss reduction: {self.loss_history[0]:.4e} → {self.current_loss:.4e}")
                ratio = self.current_loss / (self.loss_history[0] + 1e-12)
                psim.Text(f"  ({ratio:.2%} of initial)")

            psim.Separator()
            if psim.Button("Stop Optimization"):
                self.running = False

        psim.End()

    def update(self, step: int, loss: float, source_verts: np.ndarray,
               current_spectrum: np.ndarray):
        """Update visualization with current optimization state."""
        self.current_step = step
        self.current_loss = loss
        self.current_spectrum = current_spectrum
        self.loss_history.append(loss)

        # Update source point cloud positions
        self.source_cloud.update_point_positions(source_verts)

        # Color by displacement from initial
        displacement = np.linalg.norm(source_verts - self.initial_source_verts, axis=1)
        self.source_cloud.add_scalar_quantity(
            "displacement", displacement, enabled=True, cmap='viridis'
        )

        # Render one frame
        ps.frame_tick()

    def show_final(self):
        """Show final result interactively."""
        ps.show()


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def run_isospectralization(
    target_path: str,
    source_path: str,
    ckpt_path: str,
    k: int = 20,
    num_eigenvalues: int = 20,
    lr: float = 1e-3,
    num_steps: int = 500,
    device_str: str = "cuda",
    recompute_knn_every: int = 50,
    vis_every: int = 10,
    scheduler_type: str = None,
):
    """
    Run isospectralization optimization.

    Optimizes source vertex positions so its neural Laplacian spectrum
    matches the target's neural Laplacian spectrum.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(ckpt_path, device)

    # Load shapes
    print("Loading shapes...")
    target_verts = load_shape(target_path)
    source_verts = load_shape(source_path)

    print(f"  Target: {len(target_verts)} vertices")
    print(f"  Source: {len(source_verts)} vertices")

    # --- Compute target spectrum (no grad, scipy for robustness) ---
    print("\nComputing target spectrum...")
    target_spectrum_np = compute_spectrum_numpy(
        model, target_verts, k, num_eigenvalues, device
    )
    target_spectrum = torch.from_numpy(target_spectrum_np).float().to(device)
    print(f"  Target eigenvalues (first 5): {target_spectrum_np[:5]}")

    # --- Initialize source as optimizable parameter ---
    source_verts_param = nn.Parameter(
        torch.from_numpy(source_verts.copy()).float().to(device)
    )

    # Optimizer
    optimizer = torch.optim.Adam([source_verts_param], lr=lr)
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        print(f"Using cosine annealing LR scheduler")

    # Precompute initial k-NN for source
    knn_indices = compute_knn(source_verts, k)

    # --- Initialize visualization ---
    visualizer = IsospectralizationVisualizer(
        target_verts, source_verts, target_spectrum_np, num_eigenvalues
    )

    # --- Optimization loop ---
    print(f"\nStarting optimization: {num_steps} steps, lr={lr}, k={k}")
    print(f"Recomputing k-NN every {recompute_knn_every} steps")
    print(f"Updating visualization every {vis_every} steps")
    print("-" * 60)

    for step in range(num_steps):
        if not visualizer.running:
            print("\nOptimization stopped by user.")
            break

        # Recompute k-NN periodically (topology update)
        if step > 0 and step % recompute_knn_every == 0:
            with torch.no_grad():
                current_verts_np = source_verts_param.detach().cpu().numpy()
            knn_indices = compute_knn(current_verts_np, k)
            print(f"  [Step {step}] Recomputed k-NN")

        optimizer.zero_grad()

        # Forward: compute current spectrum (differentiable)
        current_spectrum = compute_spectrum(
            model, source_verts_param, knn_indices, num_eigenvalues, device
        )

        # Loss: MSE on eigenvalues, skip λ₀ ≈ 0
        loss = torch.nn.functional.mse_loss(
            current_spectrum[1:], target_spectrum[1:]
        )

        # Backward
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([source_verts_param], max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # --- Logging & visualization ---
        loss_val = loss.item()
        current_spec_np = current_spectrum.detach().cpu().numpy()
        current_verts_np = source_verts_param.detach().cpu().numpy()

        if step % 10 == 0 or step == num_steps - 1:
            lr_current = scheduler.get_last_lr()[0] if scheduler else lr
            print(
                f"  Step {step:4d} | Loss: {loss_val:.6e} | "
                f"lr: {lr_current:.2e} | "
                f"λ₁ target: {target_spectrum_np[1]:.4f} current: {current_spec_np[1]:.4f}"
            )

        # Update polyscope periodically (not every step — rendering is expensive)
        if step % vis_every == 0 or step == num_steps - 1:
            visualizer.update(step, loss_val, current_verts_np, current_spec_np)

    # --- Final results ---
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    final_spectrum = current_spectrum.detach().cpu().numpy()
    print(f"\nFinal loss: {loss.item():.6e}")
    print(f"\nEigenvalue comparison (skip λ₀):")
    print(f"  {'Index':>5} | {'Target':>10} | {'Final':>10} | {'Rel Error':>10}")
    print(f"  {'-' * 5} | {'-' * 10} | {'-' * 10} | {'-' * 10}")
    for i in range(1, min(num_eigenvalues, 15)):
        t_val = target_spectrum_np[i]
        f_val = final_spectrum[i]
        rel_err = abs(f_val - t_val) / (abs(t_val) + 1e-8)
        print(f"  {i:>5} | {t_val:>10.4f} | {f_val:>10.4f} | {rel_err:>10.2%}")

    # Show final visualization interactively
    print("\nShowing final result. Close the polyscope window to exit.")
    visualizer.show_final()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Isospectralization via Neural Laplacian",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=str, required=True,
                        help="Path to target shape mesh file")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to source shape mesh file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--k", type=int, default=6,
                        help="Number of k-NN neighbors")
    parser.add_argument("--num_eigenvalues", type=int, default=600,
                        help="Number of eigenvalues to match")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--num_steps", type=int, default=500000,
                        help="Number of optimization steps")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--recompute_knn_every", type=int, default=1,
                        help="Recompute k-NN topology every N steps")
    parser.add_argument("--vis_every", type=int, default=1,
                        help="Update polyscope visualization every N steps")
    parser.add_argument("--scheduler", type=str, default=None, choices=["cosine"],
                        help="LR scheduler type (default: none)")
    args = parser.parse_args()

    run_isospectralization(
        target_path=args.target,
        source_path=args.source,
        ckpt_path=args.checkpoint,
        k=args.k,
        num_eigenvalues=args.num_eigenvalues,
        lr=args.lr,
        num_steps=args.num_steps,
        device_str=args.device,
        recompute_knn_every=args.recompute_knn_every,
        vis_every=args.vis_every,
        scheduler_type=args.scheduler,
    )


if __name__ == "__main__":
    main()