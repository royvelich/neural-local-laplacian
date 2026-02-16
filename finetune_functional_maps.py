#!/usr/bin/env python3
"""
Fine-tune neural Laplacian for functional map correspondence.

Pipeline (fully differentiable, NO eigendecomposition in training):
    vertices → kNN → patches → model(θ) → S_A, S_B (dense)
    → for L landmarks at T scales: d(v) = concat[(S + αM)^{-1} M δ_l]
    → L2-normalize descriptors → cosine similarity matrix
    → InfoNCE contrastive loss (identity correspondence)
    → ∂Loss/∂θ via torch.linalg.solve backward

Key insight: InfoNCE only cares about RANKING, not absolute descriptor values.
Unlike ||d_A - d_B||² which has an irreducible floor from genuine geometric
differences between non-isometric shapes, contrastive loss can reach zero
as long as corresponding vertices are each other's nearest neighbors.

Evaluation uses scipy eigenvectors + functional maps (non-differentiable).

Usage:
    python finetune_functional_maps.py \
        --smal_model smal_CVPR2017.pkl --smal_data smal_CVPR2017_data.pkl \
        --checkpoint model.ckpt \
        --epochs 200 --lr 1e-4 --num_landmarks 128 \
        --output_dir fmap_finetune_runs
"""

import argparse
import copy
import sys
import types
import pickle
import json
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from itertools import combinations

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Batch

from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData
from neural_local_laplacian.utils.utils import (
    normalize_mesh_vertices,
)


# =============================================================================
# SMAL loading
# =============================================================================

def _install_fake_chumpy():
    if 'chumpy' in sys.modules:
        return
    chumpy = types.ModuleType('chumpy')
    chumpy_ch = types.ModuleType('chumpy.ch')
    chumpy_ch_ops = types.ModuleType('chumpy.ch_ops')
    chumpy_reordering = types.ModuleType('chumpy.reordering')
    chumpy_utils = types.ModuleType('chumpy.utils')
    chumpy_logic = types.ModuleType('chumpy.logic')

    class FakeCh:
        def __init__(self, *args, **kwargs):
            self._data = None
            if args and isinstance(args[0], np.ndarray):
                self._data = args[0]
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
                for key in ['x', 'a', '_data']:
                    if key in state and isinstance(state[key], np.ndarray):
                        self._data = state[key]; break
        @property
        def r(self):
            return np.array(self._data) if self._data is not None else np.array([])
        def __array__(self, dtype=None):
            arr = self.r
            return arr.astype(dtype) if dtype else arr

    sys.modules['chumpy'] = chumpy
    sys.modules['chumpy.ch'] = chumpy_ch
    sys.modules['chumpy.ch_ops'] = chumpy_ch_ops
    sys.modules['chumpy.reordering'] = chumpy_reordering
    sys.modules['chumpy.utils'] = chumpy_utils
    sys.modules['chumpy.logic'] = chumpy_logic
    chumpy.ch = chumpy_ch; chumpy.ch_ops = chumpy_ch_ops
    chumpy.reordering = chumpy_reordering; chumpy.utils = chumpy_utils
    chumpy.logic = chumpy_logic; chumpy.Ch = FakeCh
    chumpy_ch.Ch = FakeCh
    for attr in ['add', 'subtract', 'multiply', 'divide']:
        setattr(chumpy_ch_ops, attr, FakeCh)
    chumpy_reordering.transpose = FakeCh
    chumpy_reordering.concatenate = FakeCh


def _to_numpy(x):
    if hasattr(x, 'r'): return np.array(x.r)
    elif isinstance(x, np.ndarray): return x
    elif scipy.sparse.issparse(x): return x
    return np.array(x)


def _rodrigues(axis_angle):
    N = axis_angle.shape[0]
    theta = np.clip(np.linalg.norm(axis_angle, axis=1, keepdims=True), 1e-8, None)
    k = axis_angle / theta
    K = np.zeros((N, 3, 3))
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]; K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    s = np.sin(theta)[:, :, np.newaxis]
    c = np.cos(theta)[:, :, np.newaxis]
    I = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    return I + s * K + (1 - c) * np.einsum('nij,njk->nik', K, K)


class SMALModel:
    """Cached SMAL model — load once, generate shapes quickly."""

    def __init__(self, model_path: str, data_path: str):
        _install_fake_chumpy()

        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')
        with open(data_path, 'rb') as f:
            smal_data = pickle.load(f, encoding='latin1')

        self.v_template = _to_numpy(params['v_template']).astype(np.float64)
        self.shapedirs = _to_numpy(params['shapedirs']).astype(np.float64)
        self.posedirs = _to_numpy(params['posedirs']).astype(np.float64)
        J_regressor = params['J_regressor']
        self.J_regressor = (J_regressor.toarray().astype(np.float64)
                            if scipy.sparse.issparse(J_regressor)
                            else _to_numpy(J_regressor).astype(np.float64))
        self.weights = _to_numpy(params['weights']).astype(np.float64)
        self.kintree_table = _to_numpy(params['kintree_table']).astype(np.int64)
        self.faces = _to_numpy(params['f']).astype(np.int32)
        self.num_joints = self.kintree_table.shape[1]
        self.num_betas = self.shapedirs.shape[2]
        self.cluster_means = smal_data['cluster_means']
        self.num_families = len(self.cluster_means)

    def generate(self, family_idx: int, pose_scale: float = 0.2,
                 rng: np.random.RandomState = None) -> np.ndarray:
        """Generate a normalized SMAL shape. Returns (N, 3) float32 vertices."""
        if rng is None:
            rng = np.random.RandomState()

        betas = np.zeros(self.num_betas)
        family_betas = self.cluster_means[family_idx]
        n = min(len(family_betas), self.num_betas)
        betas[:n] = family_betas[:n]

        pose = rng.randn(self.num_joints * 3) * pose_scale
        pose[:3] *= 0.1

        v_shaped = self.v_template + self.shapedirs.dot(betas)
        J = self.J_regressor.dot(v_shaped)
        pose_vec = pose.reshape(-1, 3)
        R = _rodrigues(pose_vec)
        I_cube = np.broadcast_to(np.eye(3), (self.num_joints - 1, 3, 3))
        lrotmin = (R[1:] - I_cube).ravel()
        v_posed = v_shaped + self.posedirs.dot(lrotmin)

        parent = {}
        for i in range(1, self.num_joints):
            parent[i] = self.kintree_table[0, i]

        def with_zeros(x):
            return np.vstack([x, [0, 0, 0, 1]])

        G = np.empty((self.num_joints, 4, 4))
        G[0] = with_zeros(np.hstack([R[0], J[0].reshape(3, 1)]))
        for i in range(1, self.num_joints):
            G[i] = G[parent[i]].dot(
                with_zeros(np.hstack([R[i], (J[i] - J[parent[i]]).reshape(3, 1)])))

        G_rest = np.matmul(
            G, np.hstack([J, np.zeros((self.num_joints, 1))]).reshape(self.num_joints, 4, 1))
        G_packed = np.zeros((self.num_joints, 4, 4))
        G_packed[:, :, 3] = G_rest.squeeze(-1)
        G = G - G_packed

        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        v_homo = np.hstack([v_posed, np.ones((len(v_posed), 1))])
        v_final = np.einsum('vij,vj->vi', T, v_homo)[:, :3]

        return normalize_mesh_vertices(v_final.astype(np.float32))


# =============================================================================
# Differentiable dense Laplacian assembly
# =============================================================================

def compute_knn(vertices_np: np.ndarray, k: int) -> np.ndarray:
    """Compute k-nearest neighbors. Returns (N, k) indices."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(vertices_np)
    _, indices = nbrs.kneighbors(vertices_np)
    return indices


def build_patch_data(vertices_t, knn, device):
    """Build MeshPatchData for model input."""
    N, k = knn.shape
    knn_t = torch.from_numpy(knn).long().to(device)
    center_indices = torch.arange(N, device=device).repeat_interleave(k)
    neighbor_indices = knn_t.flatten()
    patch_idx = torch.arange(N, device=device).repeat_interleave(k)
    vertex_indices = knn_t.flatten()
    positions = vertices_t[neighbor_indices] - vertices_t[center_indices]

    return MeshPatchData(
        pos=positions,
        x=positions,  # features = relative positions (model's default)
        vertex_indices=vertex_indices,
        center_indices=torch.arange(N, device=device),
        patch_idx=patch_idx,
        batch=patch_idx,
        num_patches=N,
    )


def assemble_dense_stiffness_and_mass(
    stiffness_weights: torch.Tensor,
    areas: torch.Tensor,
    attention_mask: torch.Tensor,
    vertex_indices: torch.Tensor,
    center_indices: torch.Tensor,
    batch_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable dense matrix assembly from model outputs.

    Uses only non-in-place operations so autograd tracks gradients
    from stiffness_weights and areas back through the computation graph.

    Returns:
        S: (N, N) dense stiffness matrix (symmetric, rows sum to 0)
        M_diag: (N,) mass diagonal
    """
    device = stiffness_weights.device
    num_patches = stiffness_weights.shape[0]
    max_k = stiffness_weights.shape[1]
    num_vertices = max(vertex_indices.max().item(), center_indices.max().item()) + 1

    weights_flat = stiffness_weights.flatten()
    mask_flat = attention_mask.flatten()
    patch_indices_flat = torch.arange(num_patches, device=device).repeat_interleave(max_k)

    valid_weights = weights_flat[mask_flat]
    valid_patch_indices = patch_indices_flat[mask_flat]
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

    batch_sizes = batch_indices.bincount(minlength=num_patches)
    cumsum_sizes = torch.cumsum(batch_sizes, dim=0)
    starts = torch.cat([torch.tensor([0], device=device, dtype=torch.long), cumsum_sizes[:-1]])

    valid_centers = center_indices[valid_patch_indices]
    valid_neighbors = vertex_indices[starts[valid_patch_indices] + positions_in_patch]

    # Build S via non-in-place scatter_add (preserves autograd graph)
    all_rows = torch.cat([valid_centers, valid_neighbors])
    all_cols = torch.cat([valid_neighbors, valid_centers])
    all_vals = torch.cat([-valid_weights, -valid_weights])

    flat_indices = all_rows * num_vertices + all_cols
    S_flat = torch.zeros(num_vertices * num_vertices, device=device, dtype=stiffness_weights.dtype)
    S_flat = S_flat.scatter_add(0, flat_indices, all_vals)
    S = S_flat.view(num_vertices, num_vertices)

    S = 0.5 * (S + S.T)
    row_sums = S.sum(dim=1)
    S = S - torch.diag(row_sums)

    # Build M_diag via non-in-place scatter_add
    M_diag = torch.zeros(num_vertices, device=device, dtype=areas.dtype)
    M_diag = M_diag.scatter_add(0, center_indices, areas)
    M_count = torch.zeros(num_vertices, device=device, dtype=areas.dtype)
    M_count = M_count.scatter_add(0, center_indices, torch.ones_like(areas))
    M_count = torch.clamp(M_count, min=1.0)
    M_diag = M_diag / M_count
    M_diag = torch.clamp(M_diag, min=1e-8)

    return S, M_diag


def assemble_anisotropic_laplacian(
    grad_coeffs: torch.Tensor,
    areas: torch.Tensor,
    knn: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable assembly of L = G^T M_3 G from gradient coefficients.

    Instead of collapsing gᵢⱼ ∈ ℝ³ to scalar sᵢⱼ = ‖gᵢⱼ‖², this preserves
    the full directional structure of the learned gradient operator.

    L(p,q) = Σᵢ aᵢ · G_i[:,p] · G_i[:,q]

    where G_i is the (3×N) block of the gradient operator at vertex i:
        G_i[:,j] = gᵢⱼ           for neighbor j
        G_i[:,i] = -Σⱼ gᵢⱼ       (center, ensures G @ const = 0)

    L is guaranteed PSD because L = G^T M_3 G with positive mass.
    Null space = constant functions (same as standard Laplacian).

    Args:
        grad_coeffs: (N, k, 3) gradient coefficients per edge
        areas: (N,) vertex areas (mass)
        knn: (N, k) neighbor indices (long tensor)

    Returns:
        L: (N, N) dense PSD Laplacian-like matrix
        M_diag: (N,) mass diagonal
    """
    N, k, _ = grad_coeffs.shape
    device = grad_coeffs.device

    # Extended coefficients: prepend center coefficient cᵢ = -Σⱼ gᵢⱼ
    center_coeffs = -grad_coeffs.sum(dim=1, keepdim=True)  # (N, 1, 3)
    ext_coeffs = torch.cat([center_coeffs, grad_coeffs], dim=1)  # (N, k+1, 3)

    # Local Gram matrices: aᵢ * (ext_coeffs @ ext_coeffsᵀ) per vertex
    # This is the (k+1, k+1) contribution of vertex i to L
    sqrt_a = areas.sqrt()[:, None, None]  # (N, 1, 1)
    scaled = sqrt_a * ext_coeffs  # (N, k+1, 3)
    gram = torch.bmm(scaled, scaled.transpose(1, 2))  # (N, k+1, k+1)

    # Extended index array: [i, j₁, ..., jₖ] per vertex
    center_idx = torch.arange(N, device=device).unsqueeze(1)  # (N, 1)
    ext_indices = torch.cat([center_idx, knn], dim=1)  # (N, k+1)

    # Scatter Gram entries into (N, N) matrix
    kp1 = k + 1
    row_idx = ext_indices[:, :, None].expand(-1, -1, kp1)  # (N, k+1, k+1)
    col_idx = ext_indices[:, None, :].expand(-1, kp1, -1)  # (N, k+1, k+1)
    flat_idx = (row_idx * N + col_idx).reshape(-1)

    L_flat = torch.zeros(N * N, device=device, dtype=grad_coeffs.dtype)
    L_flat = L_flat.scatter_add(0, flat_idx, gram.reshape(-1))
    L = L_flat.view(N, N)

    # Symmetrize (should already be symmetric, but scatter from different
    # patches hitting the same (p,q) may have slight asymmetry from
    # float accumulation order)
    L = 0.5 * (L + L.T)

    return L, areas.detach()


def compute_laplacian_differentiable(
    model: LaplacianTransformerModule,
    vertices_np: np.ndarray,
    k: int,
    device: torch.device,
    sparsify: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass through model → dense Laplacian (differentiable).

    In gradient mode: builds the full anisotropic L = G^T M_3 G from
    the predicted 3D gradient coefficients, preserving directional structure.
    If sparsify=True, masks L to the 1-hop kNN sparsity pattern.

    In stiffness mode: builds S from scalar edge weights (isotropic).

    Returns:
        L: (N, N) dense Laplacian-like matrix — differentiable w.r.t. model params
        M_diag: (N,) mass diagonal — detached (not optimized)
    """
    vertices_t = torch.from_numpy(vertices_np).float().to(device)
    knn = compute_knn(vertices_np, k)
    knn_t = torch.from_numpy(knn).long().to(device)

    batch_data = build_patch_data(vertices_t, knn, device)
    batch_data = Batch.from_data_list([batch_data]).to(device)

    fwd = model._forward_pass(batch_data)

    if fwd.get('grad_coeffs') is not None:
        # Gradient mode: anisotropic L = G^T M_3 G
        L, M_diag = assemble_anisotropic_laplacian(
            grad_coeffs=fwd['grad_coeffs'],
            areas=fwd['areas'],
            knn=knn_t,
        )
        if sparsify:
            L = _sparsify_L_to_knn(L, knn_t)
        return L, M_diag
    else:
        # Stiffness mode: isotropic S from scalar weights
        batch_idx = getattr(batch_data, 'patch_idx', batch_data.batch)
        S, M_diag = assemble_dense_stiffness_and_mass(
            stiffness_weights=fwd['stiffness_weights'],
            areas=fwd['areas'],
            attention_mask=fwd['attention_mask'],
            vertex_indices=batch_data.vertex_indices,
            center_indices=batch_data.center_indices,
            batch_indices=batch_idx,
        )
        return S, M_diag


# =============================================================================
# Differentiable eigendecomposition (clamped-gap backward)
# =============================================================================

class _StableEigh(torch.autograd.Function):
    """eigh with clamped eigenvalue gaps in backward to prevent NaN.

    The standard eigh backward has terms 1/(λ_j - λ_i) which explode
    when eigenvalues are nearly degenerate. We clamp these gaps.

    Optimized: since only k out of N eigenpairs receive gradients
    (the rest are sliced away), we only compute the relevant terms.
    """

    @staticmethod
    def forward(ctx, A, min_gap):
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        ctx.save_for_backward(eigenvalues, eigenvectors)
        ctx.min_gap = min_gap
        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_evals, grad_evecs):
        evals, evecs = ctx.saved_tensors
        min_gap = ctx.min_gap
        N = evals.shape[0]

        # Find which eigenpairs have nonzero gradients (typically k << N)
        # grad_evecs is (N, N) but only a few columns are nonzero
        col_norms = grad_evecs.norm(dim=0)
        active_mask = col_norms > 0
        if grad_evals is not None:
            active_mask = active_mask | (grad_evals.abs() > 0)
        active_idx = torch.where(active_mask)[0]
        k_active = len(active_idx)

        if k_active == 0:
            return torch.zeros_like(evecs @ evecs.T), None

        # Optimized backward: only compute terms involving active eigenpairs
        # grad_A = V D V^T where D_{ij} = F_{ij} * (V^T dV)_{ij} + δ_{ij} dλ_i
        # Since only k columns of dV are nonzero, D has special structure

        # V^T @ dV_active: (N, N)^T @ (N, k) = (N, k) — much smaller!
        V = evecs
        dV_active = grad_evecs[:, active_idx]  # (N, k_active)
        VtdV_active = V.T @ dV_active  # (N, k_active)

        # F matrix only for active columns: F[i, active_j]
        evals_active = evals[active_idx]  # (k_active,)
        diff = evals.unsqueeze(1) - evals_active.unsqueeze(0)  # (N, k_active)
        sign_diff = torch.sign(diff)
        sign_diff[diff == 0] = 1.0
        diff_clamped = sign_diff * torch.clamp(diff.abs(), min=min_gap)
        F_active = 1.0 / diff_clamped  # (N, k_active)
        # Zero out diagonal entries (where i == active_j)
        for local_j, global_j in enumerate(active_idx):
            F_active[global_j, local_j] = 0.0

        # D_active = F_active ⊙ VtdV_active
        D_active = F_active * VtdV_active  # (N, k_active)

        # Add eigenvalue gradients on the diagonal
        if grad_evals is not None:
            deval_active = grad_evals[active_idx]  # (k_active,)
            for local_j, global_j in enumerate(active_idx):
                D_active[global_j, local_j] += deval_active[local_j]

        # grad_A = V @ D_sparse @ V^T
        # D_sparse is N×N with only k_active nonzero columns
        # So V @ D_sparse = V[:, :] @ D_sparse, but D_sparse has k_active nonzero cols
        # = sum over active j: D_active[:, j] ⊗ V[:, active_j] → gives (N, k_active) intermediate
        # Actually: grad_A = (V @ D_active) @ V[:, active_idx]^T
        # Wait, D_sparse[i, active_j] = D_active[i, local_j], zeros elsewhere
        # V @ D_sparse = sum_j V[:, j] * D_sparse[j, :] - this is full N×N @ N×N
        #
        # Better: grad_A = V D_full V^T where D_full has nonzero cols only at active_idx
        # V @ D_full has row i = sum_j V[i,j] * D_full[j,:] = sum over all j of V[i,j]*D_full[j,:]
        # Since D_full[:,m] = 0 for m not in active_idx:
        #   (V @ D_full)[:, m] = 0 for m not in active
        #   (V @ D_full)[:, active_j] = sum_i V[:, i] * D_active[i, local_j]
        # So (V @ D_full) = V @ D_active when we only keep active columns
        # Then: grad_A = (V @ D_active) @ V[:, active_idx]^T
        #             = (N,k_active) @ (k_active, N) = (N, N) — two small matmuls!

        # But wait - D is not just active columns. D also has diagonal entries for active indices.
        # Actually I already added the diagonal above, so D_active captures everything.
        # But D_full[i, j] is nonzero only when j is active. And D_full[i, i] is nonzero
        # only when i is active (from deval). So D_full has nonzero entries only in
        # active columns. So the factorization above is correct!

        VD = V @ D_active  # (N, N) @ (N, k_active) = (N, k_active)
        V_active = V[:, active_idx]  # (N, k_active)
        grad_A = VD @ V_active.T  # (N, k_active) @ (k_active, N) = (N, N)

        grad_A = 0.5 * (grad_A + grad_A.T)

        return grad_A, None


def stable_eigh(A: torch.Tensor, min_gap: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for _StableEigh with default min_gap."""
    return _StableEigh.apply(A, min_gap)


def differentiable_eigh(
    S: torch.Tensor,
    M_diag: torch.Tensor,
    k: int,
    min_gap: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable generalized eigendecomposition with stable backward.

    Solves S φ = λ M φ by converting to standard form:
        S_std = M^{-1/2} S M^{-1/2}

    Uses custom backward that clamps 1/(λ_j - λ_i) terms to prevent NaN.

    Args:
        S: (N, N) stiffness matrix (differentiable)
        M_diag: (N,) mass diagonal (detached)
        k: number of eigenpairs to return (excluding constant eigenvector)
        min_gap: minimum eigenvalue gap for backward stability

    Returns:
        eigenvalues: (k,) with gradients through S
        eigenvectors: (N, k) M-orthonormal, with gradients through S
    """
    # Convert to standard form
    M_sqrt_inv = 1.0 / M_diag.sqrt().clamp(min=1e-8)
    S_std = (S * M_sqrt_inv[None, :]) * M_sqrt_inv[:, None]
    S_std = 0.5 * (S_std + S_std.T)

    # Stable eigh with clamped-gap backward
    eigenvalues_all, eigenvectors_all = stable_eigh(S_std, min_gap)

    # Skip constant eigenvector (index 0), take next k
    eigenvalues = eigenvalues_all[1:k+1]
    eigenvectors_std = eigenvectors_all[:, 1:k+1]

    # Convert back to generalized form: φ = M^{-1/2} ψ
    eigenvectors = eigenvectors_std * M_sqrt_inv[:, None]

    return eigenvalues, eigenvectors


# =============================================================================
# Full differentiable pipeline: vertices → model → eigenbasis
# =============================================================================

def compute_eigenbasis_differentiable(
    model: LaplacianTransformerModule,
    vertices_np: np.ndarray,
    k: int,
    num_eigenvectors: int,
    device: torch.device,
    min_gap: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full differentiable pipeline: vertices → model → S, M → eigh → Φ, Λ.

    Returns:
        eigenvalues: (num_eigenvectors,) — differentiable
        eigenvectors: (N, num_eigenvectors) — differentiable
        M_diag: (N,) — detached
    """
    S, M_diag = compute_laplacian_differentiable(model, vertices_np, k, device)
    eigenvalues, eigenvectors = differentiable_eigh(S, M_diag, num_eigenvectors, min_gap)
    return eigenvalues, eigenvectors, M_diag


# =============================================================================
# Functional map computation (differentiable)
# =============================================================================

def compute_functional_map_diff(
    eigvecs_a: torch.Tensor,
    eigvecs_b: torch.Tensor,
    M_diag_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute functional map C = Φ_B^T M_B Φ_A (differentiable).
    For identity correspondence (SMAL), this is the GT functional map.
    Returns: C (k, k) tensor
    """
    weighted_phi_b = eigvecs_b * M_diag_b[:, None]
    C = weighted_phi_b.T @ eigvecs_a
    return C


# =============================================================================
# Loss functions
# =============================================================================

class FunctionalMapLoss(torch.nn.Module):
    """Combined loss for functional map fine-tuning.

    - Correspondence: ||C - I||²  (main objective)
    - Orthogonality: ||C^T C - I||²
    - Bijectivity: ||C C^T - I||²
    - Commutativity: ||C Λ_A - Λ_B C||²
    """

    def __init__(
        self,
        w_correspondence: float = 1.0,
        w_orthogonality: float = 0.5,
        w_bijectivity: float = 0.5,
        w_commutativity: float = 0.1,
    ):
        super().__init__()
        self.w_correspondence = w_correspondence
        self.w_orthogonality = w_orthogonality
        self.w_bijectivity = w_bijectivity
        self.w_commutativity = w_commutativity

    def forward(
        self,
        C: torch.Tensor,
        evals_a: torch.Tensor,
        evals_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        k = C.shape[0]
        I = torch.eye(k, device=C.device, dtype=C.dtype)

        # Correspondence: C should be close to identity
        loss_corr = (C - I).pow(2).mean()

        # Orthogonality: C^T C ≈ I
        loss_ortho = (C.T @ C - I).pow(2).mean()

        # Bijectivity: C C^T ≈ I
        loss_biject = (C @ C.T - I).pow(2).mean()

        # Commutativity: C Λ_A ≈ Λ_B C (normalized by eigenvalue scale)
        Lambda_A = torch.diag(evals_a)
        Lambda_B = torch.diag(evals_b)
        commute_raw = (C @ Lambda_A - Lambda_B @ C).pow(2).mean()
        # Normalize by average eigenvalue magnitude² to make scale-invariant
        eval_scale = (0.5 * (evals_a.pow(2).mean() + evals_b.pow(2).mean())).clamp(min=1.0)
        loss_commute = commute_raw / eval_scale

        loss_total = (
            self.w_correspondence * loss_corr
            + self.w_orthogonality * loss_ortho
            + self.w_bijectivity * loss_biject
            + self.w_commutativity * loss_commute
        )

        # Diagnostic: diagonal dominance
        diag_sum = C.diag().abs().sum().item()
        total_sum = C.abs().sum().item()
        diag_ratio = diag_sum / max(total_sum, 1e-8)

        metrics = {
            'loss_total': loss_total.item(),
            'loss_corr': loss_corr.item(),
            'loss_ortho': loss_ortho.item(),
            'loss_biject': loss_biject.item(),
            'loss_commute': loss_commute.item(),
            'C_diag_ratio': diag_ratio,
        }

        return loss_total, metrics


class SoftCorrespondenceLoss(torch.nn.Module):
    """Contrastive correspondence loss via diffusion fingerprints.

    Key idea: instead of comparing descriptors directly (which has an irreducible
    floor from genuine geometric differences), use descriptors for nearest-neighbor
    correspondence and measure RANKING quality via InfoNCE loss.

    Descriptors: diffusion fingerprints from L landmark sources at T scales.
        d(v) = concat over α: row v of (S + αM)^{-1} M E_landmarks

    Loss: InfoNCE = -mean_v log softmax(sim(v, v)) over all w

    Vertex subsampling: compute descriptors for ALL N vertices (so gradients
    flow to all parts of S), but subsample V vertices for the contrastive ranking.
    This makes the classification V-way instead of N-way (much easier to learn),
    reduces memory, and acts as stochastic regularization.
    """

    def __init__(
        self,
        num_landmarks: int = 128,
        alphas: Tuple[float, ...] = (1.0, 10.0, 100.0),
        temperature: float = 0.07,
        num_sample_vertices: int = 512,
        landmark_seed: int = 0,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.alphas = alphas
        self.temperature = temperature
        self.num_sample_vertices = num_sample_vertices
        self.landmark_seed = landmark_seed
        self._landmarks = None

    def _get_landmarks(self, N: int, device: torch.device) -> torch.Tensor:
        """Fixed landmark indices (computed once, reused)."""
        if self._landmarks is None or len(self._landmarks) != self.num_landmarks:
            rng = np.random.RandomState(self.landmark_seed)
            idx = rng.choice(N, size=min(self.num_landmarks, N), replace=False)
            idx.sort()
            self._landmarks = torch.from_numpy(idx).long().to(device)
        return self._landmarks

    def forward(
        self,
        S_A: torch.Tensor,
        S_B: torch.Tensor,
        M_A: torch.Tensor,
        M_B: torch.Tensor,
        rng: np.random.RandomState,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        N = S_A.shape[0]
        landmarks = self._get_landmarks(N, S_A.device)
        L = len(landmarks)
        V = min(self.num_sample_vertices, N)

        # Landmark indicator matrix (N, L)
        E = torch.zeros(N, L, device=S_A.device, dtype=S_A.dtype)
        E[landmarks, torch.arange(L, device=S_A.device)] = 1.0

        # Compute descriptors at multiple diffusion scales for ALL vertices
        desc_A_parts = []
        desc_B_parts = []

        for alpha in self.alphas:
            A_mat = S_A + alpha * torch.diag(M_A)
            B_mat = S_B + alpha * torch.diag(M_B)

            rhs_A = M_A[:, None] * E
            rhs_B = M_B[:, None] * E

            D_A = torch.linalg.solve(A_mat, rhs_A)
            D_B = torch.linalg.solve(B_mat, rhs_B)

            desc_A_parts.append(D_A)
            desc_B_parts.append(D_B)

        # Full descriptors: (N, L * num_scales)
        desc_A = torch.cat(desc_A_parts, dim=1)
        desc_B = torch.cat(desc_B_parts, dim=1)

        # L2-normalize per vertex
        desc_A = F.normalize(desc_A, p=2, dim=1)
        desc_B = F.normalize(desc_B, p=2, dim=1)

        # Subsample vertices for contrastive loss (different each step)
        sample_idx = torch.from_numpy(
            rng.choice(N, size=V, replace=False)
        ).long().to(S_A.device)

        desc_A_sub = desc_A[sample_idx]  # (V, D)
        desc_B_sub = desc_B[sample_idx]  # (V, D)

        # Similarity matrix: (V, V) — much smaller than (N, N)
        sim = (desc_A_sub @ desc_B_sub.T) / self.temperature

        # InfoNCE: correct match for sample_idx[i] on A is sample_idx[i] on B
        labels = torch.arange(V, device=sim.device)
        loss_nce = F.cross_entropy(sim, labels)
        loss_nce_rev = F.cross_entropy(sim.T, labels)
        loss_nce = 0.5 * (loss_nce + loss_nce_rev)

        with torch.no_grad():
            pred_A2B = sim.argmax(dim=1)
            train_acc = (pred_A2B == labels).float().mean().item()

            _, topk = sim.topk(5, dim=1)
            top5_acc = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        metrics = {
            'loss_total': loss_nce.item(),
            'loss_nce': loss_nce.item(),
            'train_acc': train_acc,
            'train_top5': top5_acc,
        }

        return loss_nce, metrics


# =============================================================================
# Evaluation (non-differentiable, uses scipy)
# =============================================================================

def evaluate_pair_robust(
    vertices_a: np.ndarray,
    vertices_b: np.ndarray,
    num_eigenvectors: int,
    n_neighbors: int = 30,
) -> Dict[str, float]:
    """Evaluate correspondence using robust Laplacian (baseline)."""
    import robust_laplacian

    metrics = {}
    n = len(vertices_a)

    for label, verts in [('A', vertices_a), ('B', vertices_b)]:
        S, M = robust_laplacian.point_cloud_laplacian(verts, n_neighbors=n_neighbors)
        from neural_local_laplacian.utils.utils import compute_laplacian_eigendecomposition
        evals, evecs = compute_laplacian_eigendecomposition(S, num_eigenvectors, mass_matrix=M)
        M_diag = np.array(M.diagonal()).flatten()

        if label == 'A':
            eigvecs_a, mass_a = evecs, M_diag
        else:
            eigvecs_b, mass_b = evecs, M_diag

    weighted_phi_b = eigvecs_b * mass_b[:, None]
    C = weighted_phi_b.T @ eigvecs_a

    k_fm = C.shape[0]
    I = np.eye(k_fm)

    metrics['corr_error'] = float(np.linalg.norm(C - I, 'fro'))

    projected_a = eigvecs_a @ C.T
    from sklearn.neighbors import NearestNeighbors as NN
    nbrs = NN(n_neighbors=1, algorithm='auto').fit(eigvecs_b)
    _, indices = nbrs.kneighbors(projected_a)
    pred_corr = indices.flatten()

    gt_corr = np.arange(n)
    metrics['accuracy'] = float((pred_corr == gt_corr).mean())

    errors = np.linalg.norm(vertices_b[pred_corr] - vertices_b[gt_corr], axis=1)
    bb_diag = np.linalg.norm(vertices_b.max(0) - vertices_b.min(0))
    metrics['mean_error'] = float(errors.mean() / bb_diag)

    return metrics


@torch.no_grad()
def _eigh_from_dense_L(L, M_diag_t, num_eigenvectors):
    """Generalized eigenproblem on GPU: L v = λ M v → evals, evecs (numpy)."""
    M_inv_sqrt = 1.0 / M_diag_t.sqrt().clamp(min=1e-8)
    L_std = L * M_inv_sqrt[:, None] * M_inv_sqrt[None, :]
    L_std = 0.5 * (L_std + L_std.T)
    all_evals, all_evecs = torch.linalg.eigh(L_std)
    evals = all_evals[:num_eigenvectors].cpu().numpy()
    evecs = (M_inv_sqrt[:, None] * all_evecs[:, :num_eigenvectors]).cpu().numpy()
    return evals, evecs


def _sparsify_L_to_knn(L, knn_t):
    """Zero out entries of L not in the 1-hop kNN graph, fix diagonal.

    Differentiable: uses element-wise multiplication with a constant mask
    so gradients flow through the retained entries.

    Args:
        L: (N, N) dense Laplacian
        knn_t: (N, k) neighbor indices

    Returns:
        L_sparse: (N, N) dense tensor with kNN sparsity pattern
    """
    N = L.shape[0]
    device = L.device

    # Build kNN mask: mask[i, j] = True if j ∈ kNN(i) or i ∈ kNN(j)
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn_t)
    mask[row_idx, knn_t] = True
    mask = mask | mask.T  # symmetrize

    # Apply mask to off-diagonal entries (differentiable: multiply by float mask)
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    keep = (mask | diag_mask).float()  # 1.0 for kNN + diagonal, 0.0 elsewhere
    L_sp = L * keep

    # Fix diagonal: zero it out then set to -row_sum of off-diagonals
    off_diag = L_sp * (1.0 - diag_mask.float())
    row_sums = off_diag.sum(dim=1)
    L_sp = off_diag - torch.diag(row_sums)

    return L_sp


def _correspondence_metrics(eigvecs_a, eigvecs_b, mass_a, mass_b, vertices_b, n):
    """Compute functional map correspondence metrics from eigenbases."""
    weighted_phi_b = eigvecs_b * mass_b[:, None]
    C = weighted_phi_b.T @ eigvecs_a

    k_fm = C.shape[0]
    I = np.eye(k_fm)

    metrics = {}
    metrics['ortho_error'] = float(np.linalg.norm(C.T @ C - I, 'fro'))
    metrics['biject_error'] = float(np.linalg.norm(C @ C.T - I, 'fro'))
    metrics['corr_error'] = float(np.linalg.norm(C - I, 'fro'))

    diag_energy = np.sum(np.diag(C) ** 2)
    total_energy = np.sum(C ** 2)
    metrics['diag_ratio'] = float(diag_energy / (total_energy + 1e-10))

    # Pointwise correspondence
    projected_a = eigvecs_a @ C.T
    from sklearn.neighbors import NearestNeighbors as NN
    nbrs = NN(n_neighbors=1, algorithm='auto').fit(eigvecs_b)
    _, indices = nbrs.kneighbors(projected_a)
    pred_corr = indices.flatten()

    gt_corr = np.arange(n)
    metrics['accuracy'] = float((pred_corr == gt_corr).mean())

    errors = np.linalg.norm(vertices_b[pred_corr] - vertices_b[gt_corr], axis=1)
    bb_diag = np.linalg.norm(vertices_b.max(0) - vertices_b.min(0))
    metrics['mean_error'] = float(errors.mean() / bb_diag)
    metrics['median_error'] = float(np.median(errors) / bb_diag)

    return metrics


def evaluate_pair(
    model: LaplacianTransformerModule,
    vertices_a: np.ndarray,
    vertices_b: np.ndarray,
    k: int,
    num_eigenvectors: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate correspondence quality using functional maps.

    Uses anisotropic L = G^T M_3 G in gradient mode for consistency with training.
    Also evaluates a kNN-sparsified variant of L (1-hop only, 2-hop entries zeroed).
    """
    from neural_local_laplacian.utils.utils import (
        assemble_stiffness_and_mass_matrices,
        compute_laplacian_eigendecomposition,
    )

    n = len(vertices_a)
    is_gradient_mode = False

    # --- Per-shape: compute eigenbases for dense L, sparse L, and (fallback) S ---
    dense_bases = {}
    sparse_bases = {}

    for label, verts in [('A', vertices_a), ('B', vertices_b)]:
        verts_t = torch.from_numpy(verts).float().to(device)
        knn_np = compute_knn(verts, k)
        batch_data = build_patch_data(verts_t, knn_np, device)
        batch_data = Batch.from_data_list([batch_data]).to(device)

        fwd = model._forward_pass(batch_data)

        if fwd.get('grad_coeffs') is not None:
            is_gradient_mode = True
            knn_t = torch.from_numpy(knn_np).long().to(device)
            with torch.no_grad():
                L, M_diag_t = assemble_anisotropic_laplacian(
                    grad_coeffs=fwd['grad_coeffs'],
                    areas=fwd['areas'],
                    knn=knn_t,
                )

                # Dense L eigenbasis
                evals_d, evecs_d = _eigh_from_dense_L(L, M_diag_t, num_eigenvectors)
                M_diag = M_diag_t.cpu().numpy()
                dense_bases[label] = (evecs_d, M_diag)

                # kNN-sparsified L eigenbasis
                L_sp = _sparsify_L_to_knn(L, knn_t)
                evals_sp, evecs_sp = _eigh_from_dense_L(L_sp, M_diag_t, num_eigenvectors)
                sparse_bases[label] = (evecs_sp, M_diag)
        else:
            batch_idx = getattr(batch_data, 'patch_idx', batch_data.batch)
            S, M = assemble_stiffness_and_mass_matrices(
                stiffness_weights=fwd['stiffness_weights'],
                areas=fwd['areas'],
                attention_mask=fwd['attention_mask'],
                vertex_indices=batch_data.vertex_indices,
                center_indices=batch_data.center_indices,
                batch_indices=batch_idx,
            )
            evals, evecs = compute_laplacian_eigendecomposition(S, num_eigenvectors, mass_matrix=M)
            M_diag = np.array(M.diagonal()).flatten()
            dense_bases[label] = (evecs, M_diag)

    # --- Compute metrics for dense L ---
    evA_d, mA = dense_bases['A']
    evB_d, mB = dense_bases['B']
    metrics = _correspondence_metrics(evA_d, evB_d, mA, mB, vertices_b, n)

    # --- Compute metrics for kNN-sparsified L ---
    if is_gradient_mode and sparse_bases:
        evA_sp, _ = sparse_bases['A']
        evB_sp, _ = sparse_bases['B']
        sp_metrics = _correspondence_metrics(evA_sp, evB_sp, mA, mB, vertices_b, n)
        metrics['sp_accuracy'] = sp_metrics['accuracy']
        metrics['sp_mean_error'] = sp_metrics['mean_error']
        metrics['sp_corr_error'] = sp_metrics['corr_error']

    return metrics


# =============================================================================
# SMAL pair generation
# =============================================================================

class SMALPairGenerator:
    """Generate random SMAL shape pairs for training/evaluation.

    Includes both same-family pairs (different poses, near-isometric) and
    cross-family pairs (different species). Same-family pairs provide
    strong supervision that generalizes, while cross-family pairs are
    the actual target task.

    Curriculum: mix_ratio controls the fraction of cross-family pairs.
    Start with mostly same-family (easy), increase cross-family over time.
    """

    def __init__(
        self,
        smal: SMALModel,
        train_families: List[int],
        val_families: List[int],
        pose_scale: float = 0.2,
    ):
        self.smal = smal
        self.train_families = train_families
        self.val_families = val_families
        self.pose_scale = pose_scale

        self.cross_pairs = list(combinations(train_families, 2))
        self.same_pairs = [(f, f) for f in train_families]
        self.val_pairs = list(combinations(val_families, 2))
        for t in train_families:
            for v in val_families:
                self.val_pairs.append((t, v))

    def sample_train_pair(
        self, rng: np.random.RandomState, cross_ratio: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Sample a training pair.

        Args:
            cross_ratio: probability of sampling a cross-family pair (0-1).
                        0 = all same-family, 1 = all cross-family.
        """
        if rng.rand() < cross_ratio and len(self.cross_pairs) > 0:
            fam_a, fam_b = self.cross_pairs[rng.randint(len(self.cross_pairs))]
            pair_type = "cross"
        else:
            fam_a, fam_b = self.same_pairs[rng.randint(len(self.same_pairs))]
            pair_type = "same"
        verts_a = self.smal.generate(fam_a, self.pose_scale, rng)
        verts_b = self.smal.generate(fam_b, self.pose_scale, rng)
        return verts_a, verts_b, f"{pair_type}_{fam_a}_vs_{fam_b}"

    def get_val_pairs(self, rng: np.random.RandomState) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        pairs = []
        for fam_a, fam_b in self.val_pairs:
            pair_rng = np.random.RandomState(fam_a * 100 + fam_b)
            verts_a = self.smal.generate(fam_a, self.pose_scale, pair_rng)
            verts_b = self.smal.generate(fam_b, self.pose_scale, pair_rng)
            pairs.append((verts_a, verts_b, f"val_{fam_a}_vs_{fam_b}"))
        return pairs


# =============================================================================
# Training loop
# =============================================================================

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Weights & Biases
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.wandb_tags.split(",") if args.wandb_tags else None,
            config=vars(args),
            dir=str(output_dir),
        )
        print(f"  W&B run: {wandb.run.url}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = LaplacianTransformerModule.load_from_checkpoint(
        args.checkpoint, map_location=device,
        normalize_patch_features=True,
        scale_areas_by_patch_size=True,
    )
    model.to(device)
    model.train()

    if args.freeze_input_projection:
        for name, param in model.named_parameters():
            if 'input_projection' in name:
                param.requires_grad_(False)
                print(f"  Frozen: {name}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,}")

    # Load SMAL
    print(f"Loading SMAL model...")
    smal = SMALModel(args.smal_model, args.smal_data)
    print(f"  {smal.num_families} families, {len(smal.v_template)} vertices per shape")

    all_families = list(range(smal.num_families))
    if args.val_families:
        val_families = [int(x) for x in args.val_families.split(',')]
    else:
        val_families = all_families[-2:]
    train_families = [f for f in all_families if f not in val_families]

    print(f"  Train families: {train_families}")
    print(f"  Val families: {val_families}")

    pair_gen = SMALPairGenerator(smal, train_families, val_families, args.pose_scale)
    print(f"  Cross-family pairs: {pair_gen.cross_pairs}")
    print(f"  Same-family pairs: {pair_gen.same_pairs}")
    print(f"  Val pairs: {pair_gen.val_pairs}")

    # Loss and optimizer
    loss_fn = SoftCorrespondenceLoss(
        num_landmarks=args.num_landmarks,
        alphas=tuple(float(a) for a in args.alphas.split(',')),
        temperature=args.temperature,
        num_sample_vertices=args.num_sample_vertices,
    )

    # Frozen copy of pretrained model for parameter-space proximity
    model_ref = copy.deepcopy(model)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)
    # Pre-compute reference param vector (flat)
    ref_params = torch.cat([p.detach().flatten() for p in model_ref.parameters()])
    ref_norm_sq = (ref_params ** 2).sum().clamp(min=1e-8)
    del model_ref  # free GPU memory — we only need the flat param vector
    print(f"  Reference params: {ref_params.numel():,} values, ||θ_ref||²={ref_norm_sq.item():.2e}")
    print(f"  Created frozen reference model for proximity regularizer")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Logging
    log_file = open(output_dir / "train_log.csv", "w")
    log_file.write("epoch,step,loss,loss_nce,loss_prox,train_acc,train_top5,cross_ratio,grad_norm,lr\n")

    val_log = open(output_dir / "val_log.csv", "w")
    val_log.write("epoch,pair,accuracy,mean_error,median_error,ortho_error,"
                  "biject_error,corr_error,diag_ratio\n")

    rng = np.random.RandomState(args.seed)
    best_val_acc = 0.0
    global_step = 0

    # Compute robust Laplacian baseline (once)
    print("\n  Computing robust Laplacian baseline...")
    val_pairs_baseline = pair_gen.get_val_pairs(rng)
    robust_accs = []
    robust_errs = []
    for verts_a, verts_b, pair_name in val_pairs_baseline:
        rb = evaluate_pair_robust(verts_a, verts_b, args.num_eigenvectors)
        robust_accs.append(rb['accuracy'])
        robust_errs.append(rb['mean_error'])
        print(f"    {pair_name}: Acc={rb['accuracy']*100:.1f}% Err={rb['mean_error']:.4f}")
    robust_mean_acc = np.mean(robust_accs)
    robust_mean_err = np.mean(robust_errs)
    print(f"  Robust baseline: Acc={robust_mean_acc*100:.1f}%, Err={robust_mean_err:.4f}")

    if args.wandb:
        wandb.log({"baseline/val_acc": robust_mean_acc, "baseline/val_err": robust_mean_err}, step=0)

    print()
    print("=" * 80)
    lap_mode = "sparse (kNN-masked GᵀMG)" if args.sparsify_laplacian else "dense (full GᵀMG)"
    print(f"TRAINING (InfoNCE contrastive, {args.num_landmarks} landmarks, "
          f"V={args.num_sample_vertices}, scales={args.alphas}, τ={args.temperature}, "
          f"curriculum={args.cross_ratio_start:.0%}→{args.cross_ratio_end:.0%} over {args.curriculum_epochs}ep)")
    print(f"  Laplacian: {lap_mode}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        epoch_start = time.time()

        # Curriculum: start with mostly same-family pairs, increase cross-family
        # Linear ramp from cross_ratio_start to cross_ratio_end over curriculum_epochs
        progress = min(1.0, (epoch - 1) / max(1, args.curriculum_epochs))
        cross_ratio = args.cross_ratio_start + progress * (args.cross_ratio_end - args.cross_ratio_start)

        for step in range(args.steps_per_epoch):
            global_step += 1
            optimizer.zero_grad()

            step_metrics_list = []
            valid_pairs = 0

            for pair_idx in range(args.pairs_per_step):
                verts_a, verts_b, pair_name = pair_gen.sample_train_pair(rng, cross_ratio)

                # Forward: differentiable Laplacians (NO eigendecomposition)
                S_A, M_A = compute_laplacian_differentiable(model, verts_a, args.k_pred, device, sparsify=args.sparsify_laplacian)
                S_B, M_B = compute_laplacian_differentiable(model, verts_b, args.k_pred, device, sparsify=args.sparsify_laplacian)

                # InfoNCE contrastive loss (vertex-subsampled)
                loss_nce, metrics = loss_fn(S_A, S_B, M_A, M_B, rng)

                # Parameter-space proximity: ||θ - θ_ref||² / ||θ_ref||²
                cur_params = torch.cat([p.flatten() for p in model.parameters()])
                loss_prox = ((cur_params - ref_params) ** 2).sum() / ref_norm_sq
                loss = loss_nce + args.w_prox * loss_prox

                metrics['loss_prox'] = loss_prox.item()
                metrics['loss_total'] = loss.item()

                if torch.isnan(loss):
                    print(f"  [Step {global_step}, pair {pair_idx}] NaN loss! Skipping pair.")
                    continue

                # Retain grad for first-step diagnostic
                if global_step == 1 and pair_idx == 0:
                    S_A.retain_grad()
                    S_B.retain_grad()

                (loss / args.pairs_per_step).backward()

                step_metrics_list.append(metrics)
                valid_pairs += 1

            if valid_pairs == 0:
                print(f"  [Step {global_step}] All pairs failed! Skipping.")
                continue

            # Gradient diagnostic (first step)
            if global_step == 1:
                print("\n  === GRADIENT FLOW DIAGNOSTIC ===")
                for name, tensor in [("S_A", S_A), ("S_B", S_B)]:
                    if tensor.grad is not None:
                        print(f"    {name}: grad norm = {tensor.grad.norm().item():.6e}")
                    else:
                        print(f"    {name}: NO GRAD")

                num_with_grad = 0
                num_nan_grad = 0
                total_grad = 0.0
                for _, p in model.named_parameters():
                    if p.grad is not None:
                        g = p.grad.norm().item()
                        if g != g:
                            num_nan_grad += 1
                        elif g > 0:
                            num_with_grad += 1
                            total_grad += g
                print(f"    Model: {num_with_grad} non-zero grad, {num_nan_grad} NaN grad")
                print(f"    Total grad norm: {total_grad:.6e}")
                print("  === END DIAGNOSTIC ===\n")

            # Gradient clipping
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float('inf')
                )

            if torch.isnan(grad_norm):
                print(f"  [Step {global_step}] NaN gradients! Skipping.")
                optimizer.zero_grad()
                continue

            optimizer.step()

            avg_step = {k: np.mean([m[k] for m in step_metrics_list])
                        for k in step_metrics_list[0]}
            avg_step['grad_norm'] = grad_norm.item()
            epoch_losses.append(avg_step)

            lr = optimizer.param_groups[0]['lr']
            log_file.write(f"{epoch},{global_step},{avg_step['loss_total']:.6e},"
                           f"{avg_step['loss_nce']:.6e},{avg_step['loss_prox']:.6e},"
                           f"{avg_step['train_acc']:.4f},{avg_step['train_top5']:.4f},"
                           f"{cross_ratio:.4f},"
                           f"{avg_step['grad_norm']:.6e},{lr:.2e}\n")

        scheduler.step()

        # Epoch summary
        if epoch_losses:
            avg = {k: np.mean([m[k] for m in epoch_losses]) for k in epoch_losses[0]}
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch:4d} | loss={avg['loss_total']:.4f} | "
                  f"nce={avg['loss_nce']:.4f} | prox={avg['loss_prox']:.4f} | "
                  f"acc={avg['train_acc']*100:.1f}% | top5={avg['train_top5']*100:.1f}% | "
                  f"cross={cross_ratio:.0%} | "
                  f"grad={avg.get('grad_norm', 0):.2e} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                  f"time={elapsed:.1f}s")

            if args.wandb:
                wandb.log({
                    "train/loss": avg['loss_total'],
                    "train/loss_nce": avg['loss_nce'],
                    "train/loss_prox": avg['loss_prox'],
                    "train/acc": avg['train_acc'],
                    "train/top5": avg['train_top5'],
                    "train/grad_norm": avg.get('grad_norm', 0),
                    "train/cross_ratio": cross_ratio,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/epoch_time": elapsed,
                }, step=epoch)
        else:
            print(f"Epoch {epoch:4d} | all steps failed")

        # Validation
        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            val_pairs = pair_gen.get_val_pairs(rng)
            val_accs = []
            val_errs = []
            val_sp_accs = []
            val_sp_errs = []

            print(f"  --- Validation (epoch {epoch}) ---")
            for verts_a, verts_b, pair_name in val_pairs:
                val_metrics = evaluate_pair(
                    model, verts_a, verts_b, args.k_pred,
                    args.num_eigenvectors, device
                )
                val_accs.append(val_metrics['accuracy'])
                val_errs.append(val_metrics['mean_error'])

                val_log.write(f"{epoch},{pair_name},"
                              f"{val_metrics['accuracy']:.6f},"
                              f"{val_metrics['mean_error']:.6f},"
                              f"{val_metrics['median_error']:.6f},"
                              f"{val_metrics['ortho_error']:.6f},"
                              f"{val_metrics['biject_error']:.6f},"
                              f"{val_metrics['corr_error']:.6f},"
                              f"{val_metrics['diag_ratio']:.6f}\n")

                sp_str = ""
                if 'sp_accuracy' in val_metrics:
                    val_sp_accs.append(val_metrics['sp_accuracy'])
                    val_sp_errs.append(val_metrics['sp_mean_error'])
                    sp_str = (f" | sp: Acc={val_metrics['sp_accuracy']*100:.1f}% "
                              f"Err={val_metrics['sp_mean_error']:.4f}")

                print(f"    {pair_name}: Acc={val_metrics['accuracy']*100:.1f}% "
                      f"Err={val_metrics['mean_error']:.4f} "
                      f"||C-I||={val_metrics['corr_error']:.3f}{sp_str}")

            mean_acc = np.mean(val_accs)
            mean_err = np.mean(val_errs)
            sp_summary = ""
            if val_sp_accs:
                sp_mean_acc = np.mean(val_sp_accs)
                sp_mean_err = np.mean(val_sp_errs)
                sp_summary = f"  Sparse mean: Acc={sp_mean_acc*100:.1f}%, Err={sp_mean_err:.4f}"
            print(f"  Val mean: Acc={mean_acc*100:.1f}%, Err={mean_err:.4f}  "
                  f"(robust baseline: Acc={robust_mean_acc*100:.1f}%, Err={robust_mean_err:.4f})")
            if sp_summary:
                print(sp_summary)

            if args.wandb:
                val_wandb = {
                    "val/acc_mean": mean_acc,
                    "val/err_mean": mean_err,
                }
                if val_sp_accs:
                    val_wandb["val/sp_acc_mean"] = sp_mean_acc
                    val_wandb["val/sp_err_mean"] = sp_mean_err
                for i, (_, _, pair_name) in enumerate(val_pairs):
                    val_wandb[f"val_pairs/{pair_name}_acc"] = val_accs[i]
                    val_wandb[f"val_pairs/{pair_name}_err"] = val_errs[i]
                    if val_sp_accs:
                        val_wandb[f"val_pairs/{pair_name}_sp_acc"] = val_sp_accs[i]
                        val_wandb[f"val_pairs/{pair_name}_sp_err"] = val_sp_errs[i]
                wandb.log(val_wandb, step=epoch)

            if mean_acc > best_val_acc:
                best_val_acc = mean_acc
                ckpt_path = output_dir / "best_model.ckpt"
                torch.save({
                    'state_dict': model.state_dict(),
                    'hyper_parameters': dict(model.hparams),
                }, str(ckpt_path))
                print(f"  ** New best! Saved to {ckpt_path}")
                if args.wandb:
                    wandb.run.summary["best_val_acc"] = best_val_acc
                    wandb.run.summary["best_val_err"] = mean_err
                    wandb.run.summary["best_epoch"] = epoch

            model.train()

        if epoch % args.save_every == 0:
            ckpt_path = output_dir / f"model_epoch_{epoch:04d}.ckpt"
            torch.save({
                'state_dict': model.state_dict(),
                'hyper_parameters': dict(model.hparams),
            }, str(ckpt_path))

        log_file.flush()
        val_log.flush()

    # Final save
    torch.save({
        'state_dict': model.state_dict(),
        'hyper_parameters': dict(model.hparams),
    }, str(output_dir / "model_final.ckpt"))
    log_file.close()
    val_log.close()
    print(f"\nTraining complete. Best val accuracy: {best_val_acc*100:.1f}%")
    print(f"Results in: {output_dir}")

    if args.wandb:
        # Log best model as artifact
        artifact = wandb.Artifact(
            name=f"best-model-{wandb.run.id}",
            type="model",
            description=f"Best finetuned model (val_acc={best_val_acc*100:.1f}%)",
        )
        best_path = output_dir / "best_model.ckpt"
        if best_path.exists():
            artifact.add_file(str(best_path))
            wandb.log_artifact(artifact)
        wandb.finish()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune neural Laplacian for functional map correspondence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--smal_model", type=str, required=True)
    parser.add_argument("--smal_data", type=str, required=True)
    parser.add_argument("--val_families", type=str, default=None,
                        help="Comma-separated family indices for validation (default: last 2)")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--k_pred", type=int, default=20)
    parser.add_argument("--num_eigenvectors", type=int, default=30,
                        help="Number of eigenvectors for evaluation")
    parser.add_argument("--freeze_input_projection", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps_per_epoch", type=int, default=10)
    parser.add_argument("--pairs_per_step", type=int, default=1,
                        help="Number of pairs to accumulate gradients over per optimizer step")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0,
                        help="Gradient clipping max norm (0 = disabled)")
    parser.add_argument("--pose_scale", type=float, default=0.3,
                        help="Pose variation scale (higher = more diverse training)")
    parser.add_argument("--seed", type=int, default=42)

    # InfoNCE loss parameters
    parser.add_argument("--num_landmarks", type=int, default=128,
                        help="Number of landmark vertices for diffusion fingerprints")
    parser.add_argument("--alphas", type=str, default="1.0,10.0,100.0",
                        help="Comma-separated diffusion scales (α values)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature (lower = sharper ranking)")
    parser.add_argument("--num_sample_vertices", type=int, default=512,
                        help="Vertices to subsample for contrastive loss (V-way ranking)")
    parser.add_argument("--w_prox", type=float, default=10.0,
                        help="Weight for parameter-space proximity ||θ-θ_ref||²/||θ_ref||²")
    parser.add_argument("--sparsify_laplacian", action="store_true",
                        help="Mask L=G^T M G to 1-hop kNN sparsity (drop 2-hop entries)")

    # Curriculum: same-family → cross-family
    parser.add_argument("--cross_ratio_start", type=float, default=0.0,
                        help="Initial fraction of cross-family pairs (0 = all same-family)")
    parser.add_argument("--cross_ratio_end", type=float, default=0.5,
                        help="Final fraction of cross-family pairs")
    parser.add_argument("--curriculum_epochs", type=int, default=50,
                        help="Epochs over which to ramp cross_ratio from start to end")

    # Evaluation & checkpointing
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="fmap_finetune_runs")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="neural-laplacian-finetune",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (team/user). Defaults to your default entity.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not set)")
    parser.add_argument("--wandb_tags", type=str, default=None,
                        help="Comma-separated W&B tags")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()