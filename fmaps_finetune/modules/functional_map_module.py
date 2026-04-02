"""
LightningModule for functional map fine-tuning of the neural Laplacian.

All Laplacian assembly, loss, and evaluation utilities live here alongside
FunctionalMapModule so this file is self-contained on the model side.
"""
from __future__ import annotations
import os
import contextlib
import time

import scipy.sparse
import scipy.sparse.linalg
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import lightning
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Batch

from neural_local_laplacian.modules.laplacian_modules import (
    LaplacianModuleBase,
    LaplacianTransformerModule,
)
from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData
from neural_local_laplacian.utils.utils import (
    normalize_mesh_vertices,
    assemble_stiffness_and_mass_matrices,
    compute_laplacian_eigendecomposition,
)

from fmaps_finetune.datasets.functional_map_dataset import (
    PairSample,
    _compute_bijective_refs,
    subsample_pair,
    _stable_hash,
)

from fmaps_finetune.modules.evaluators import SpectralNNEvaluator, FunctionalMapEvaluator, fmt_topk



# =============================================================================
# Step profiler (optional, activated by profile_steps > 0)
# =============================================================================

class _StepProfiler:
    """Lightweight per-phase CUDA wall-time profiler.

    Wraps major training_step phases with CUDA-synchronized wall timings.
    Activated when hparams.profile_steps > 0; prints a summary every
    profile_steps steps then resets accumulators.

    Usage::
        with prof.phase("knn"):        ...
        with prof.phase("transformer"): ...
        with prof.phase("assembly"):   ...
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._times:  Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    @contextlib.contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self._times[name]  = self._times.get(name, 0.0)  + dt
        self._counts[name] = self._counts.get(name, 0)    + 1

    def summary_str(self, step: int) -> str:
        if not self._times:
            return ""
        total = sum(self._times.values())
        lines = [f"  [Profiler step={step}]  total={total*1e3:.1f}ms"]
        for name, t in self._times.items():
            n   = self._counts[name]
            pct = 100.0 * t / total if total > 0 else 0.0
            lines.append(
                f"    {name:<28s}  {t/n*1e3:7.2f}ms/call  "
                f"{t*1e3:8.2f}ms total  {pct:5.1f}%  (n={n})")
        return "\n".join(lines)

    def reset(self):
        self._times.clear()
        self._counts.clear()


# =============================================================================
# kNN + patch building
# =============================================================================

def compute_knn(vertices_np: np.ndarray, k: int) -> np.ndarray:
    """Compute k-nearest neighbors excluding self. Returns (N, k) indices."""
    n    = len(vertices_np)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(vertices_np)
    _, indices = nbrs.kneighbors(vertices_np)
    center     = np.arange(n)[:, np.newaxis]
    keep       = ~(indices == center)
    keep_pos   = np.cumsum(keep, axis=1)
    final      = (keep_pos <= k) & keep
    return indices[final].reshape(n, k)


def build_patch_data(vertices_t: torch.Tensor, knn: np.ndarray, device: torch.device):
    """Build MeshPatchData from vertices and precomputed kNN indices."""
    N   = vertices_t.shape[0]
    k   = knn.shape[1]
    knn_t     = torch.from_numpy(knn).long().to(device)
    patch_pos = vertices_t[knn_t] - vertices_t[:, None, :]   # (N, k, 3)
    all_pos   = patch_pos.reshape(-1, 3)
    return MeshPatchData(
        pos=all_pos,
        x=all_pos,
        patch_idx=torch.arange(N, device=device).repeat_interleave(k),
        vertex_indices=knn_t.flatten(),
        center_indices=torch.arange(N, device=device),
    )


# =============================================================================
# Laplacian assembly
# =============================================================================

def assemble_dense_stiffness_and_mass(
    stiffness_weights, areas, attention_mask,
    vertex_indices, center_indices, batch_indices,
):
    """Differentiable dense S and M from scalar edge weights."""
    device       = stiffness_weights.device
    num_patches  = stiffness_weights.shape[0]
    max_k        = stiffness_weights.shape[1]
    num_vertices = max(vertex_indices.max().item(), center_indices.max().item()) + 1

    weights_flat      = stiffness_weights.flatten()
    mask_flat         = attention_mask.flatten()
    patch_indices_flat = torch.arange(num_patches, device=device).repeat_interleave(max_k)

    valid_weights       = weights_flat[mask_flat]
    valid_patch_indices = patch_indices_flat[mask_flat]
    num_valid           = len(valid_patch_indices)

    if num_valid > 0:
        patch_changes = torch.ones(num_valid, dtype=torch.bool, device=device)
        if num_valid > 1:
            patch_changes[1:] = valid_patch_indices[1:] != valid_patch_indices[:-1]
        group_ids    = torch.cumsum(patch_changes.long(), dim=0) - 1
        change_indices = torch.where(patch_changes)[0]
        group_starts = change_indices[group_ids]
        positions_in_patch = torch.arange(num_valid, device=device, dtype=torch.long) - group_starts
    else:
        positions_in_patch = torch.tensor([], device=device, dtype=torch.long)

    batch_sizes   = batch_indices.bincount(minlength=num_patches)
    cumsum_sizes  = torch.cumsum(batch_sizes, dim=0)
    starts        = torch.cat([torch.tensor([0], device=device, dtype=torch.long),
                               cumsum_sizes[:-1]])

    valid_centers   = center_indices[valid_patch_indices]
    valid_neighbors = vertex_indices[starts[valid_patch_indices] + positions_in_patch]

    all_rows = torch.cat([valid_centers, valid_neighbors])
    all_cols = torch.cat([valid_neighbors, valid_centers])
    all_vals = torch.cat([-valid_weights, -valid_weights])

    flat_indices = all_rows * num_vertices + all_cols
    S_flat = torch.zeros(num_vertices * num_vertices, device=device,
                         dtype=stiffness_weights.dtype)
    S_flat = S_flat.scatter_add(0, flat_indices, all_vals)
    S      = S_flat.view(num_vertices, num_vertices)
    S      = 0.5 * (S + S.T)
    S      = S - torch.diag(S.sum(dim=1))

    M_diag  = torch.zeros(num_vertices, device=device, dtype=areas.dtype)
    M_diag  = M_diag.scatter_add(0, center_indices, areas)
    M_count = torch.zeros(num_vertices, device=device, dtype=areas.dtype)
    M_count = M_count.scatter_add(0, center_indices, torch.ones_like(areas))
    M_diag  = M_diag / M_count.clamp(min=1.0)
    M_diag  = M_diag.clamp(min=1e-8)
    return S, M_diag


def assemble_anisotropic_laplacian(
    grad_coeffs: torch.Tensor,
    areas: torch.Tensor,
    knn: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable L = G^T M_3 G from gradient coefficients."""
    N, k, _ = grad_coeffs.shape
    device   = grad_coeffs.device

    center_coeffs = -grad_coeffs.sum(dim=1, keepdim=True)
    ext_coeffs    = torch.cat([center_coeffs, grad_coeffs], dim=1)
    sqrt_a        = areas.sqrt()[:, None, None]
    scaled        = sqrt_a * ext_coeffs
    gram          = torch.bmm(scaled, scaled.transpose(1, 2))

    center_idx  = torch.arange(N, device=device).unsqueeze(1)
    ext_indices = torch.cat([center_idx, knn], dim=1)
    kp1         = k + 1
    row_idx     = ext_indices[:, :, None].expand(-1, -1, kp1)
    col_idx     = ext_indices[:, None, :].expand(-1, kp1, -1)
    flat_idx    = (row_idx * N + col_idx).reshape(-1)

    L_flat = torch.zeros(N * N, device=device, dtype=grad_coeffs.dtype)
    L_flat = L_flat.scatter_add(0, flat_idx, gram.reshape(-1))
    L      = L_flat.view(N, N)
    L      = 0.5 * (L + L.T)
    return L, areas.detach()


def _sparsify_L_to_knn(L: torch.Tensor, knn_t: torch.Tensor) -> torch.Tensor:
    """Zero out entries of L not in the 1-hop kNN graph, fix diagonal."""
    N      = L.shape[0]
    device = L.device
    mask   = torch.zeros(N, N, dtype=torch.bool, device=device)
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn_t)
    mask[row_idx, knn_t] = True
    mask   = mask | mask.T
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    L_sp   = L * (mask | diag_mask).float()
    off    = L_sp * (1.0 - diag_mask.float())
    L_sp   = off - torch.diag(off.sum(dim=1))
    return L_sp


def assemble_isotropic_laplacian(
    grad_coeffs: torch.Tensor,
    areas: torch.Tensor,
    knn: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Isotropic graph Laplacian from gradient coefficients.

    Uses scalar edge weights w_ij = area_i * ||g_ij||^2, producing a
    standard graph Laplacian L = D - W with non-negative weights.
    This is guaranteed PSD and sparse (kNN only, no 2-hop fill-in).

    Args:
        grad_coeffs: (N, k, 3) gradient coefficients per neighbor.
        areas: (N,) per-vertex areas.
        knn: (N, k) kNN indices.

    Returns:
        L: (N, N) isotropic Laplacian (PSD, sparse).
        M_diag: (N,) mass diagonal (same as areas).
    """
    N, k, _ = grad_coeffs.shape
    device  = grad_coeffs.device

    # w_ij = area_i * ||g_ij||^2  — scalar weight per edge (i, j)
    edge_weights = areas[:, None] * (grad_coeffs ** 2).sum(dim=2)  # (N, k)

    # Build sparse graph Laplacian: L_ij = -w_ij, L_ii = sum_j w_ij
    # Symmetrise: w_ij_sym = 0.5 * (w_ij + w_ji)
    L = torch.zeros(N, N, device=device, dtype=grad_coeffs.dtype)
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn)
    L[row_idx, knn] -= edge_weights
    L[knn, row_idx] -= edge_weights
    L = 0.5 * L  # average the two directions
    # Fix diagonal: L_ii = -sum of off-diagonal in row i
    L.fill_diagonal_(0.0)
    L.diagonal().copy_(-L.sum(dim=1))

    return L, areas.detach()


def compute_laplacian_differentiable(
    model: LaplacianTransformerModule,
    vertices_np: np.ndarray,
    k: int,
    device: torch.device,
    sparsify: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full forward pass: vertices → dense differentiable (S, M)."""
    vertices_t = torch.from_numpy(vertices_np).float().to(device)
    knn        = compute_knn(vertices_np, k)
    knn_t      = torch.from_numpy(knn).long().to(device)
    batch_data = Batch.from_data_list([build_patch_data(vertices_t, knn, device)]).to(device)
    fwd        = model._forward_pass(batch_data)

    if fwd.get('grad_coeffs') is not None:
        L, M_diag = assemble_anisotropic_laplacian(fwd['grad_coeffs'], fwd['areas'], knn_t)
        if sparsify:
            L = _sparsify_L_to_knn(L, knn_t)
        return L, M_diag
    else:
        batch_idx = getattr(batch_data, 'patch_idx', batch_data.batch)
        return assemble_dense_stiffness_and_mass(
            stiffness_weights=fwd['stiffness_weights'],
            areas=fwd['areas'],
            attention_mask=fwd['attention_mask'],
            vertex_indices=batch_data.vertex_indices,
            center_indices=batch_data.center_indices,
            batch_indices=batch_idx,
        )


# =============================================================================
# Differentiable eigendecomposition (stable backward)
# =============================================================================

class _StableEigh(torch.autograd.Function):
    """eigh with clamped eigenvalue gaps in backward to prevent NaN."""

    @staticmethod
    def forward(ctx, A, min_gap):
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        ctx.save_for_backward(eigenvalues, eigenvectors)
        ctx.min_gap = min_gap
        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_evals, grad_evecs):
        evals, evecs = ctx.saved_tensors
        min_gap      = ctx.min_gap
        N            = evals.shape[0]

        col_norms    = grad_evecs.norm(dim=0)
        active_mask  = col_norms > 0
        if grad_evals is not None:
            active_mask = active_mask | (grad_evals.abs() > 0)
        active_idx  = torch.where(active_mask)[0]
        if len(active_idx) == 0:
            return torch.zeros_like(evecs @ evecs.T), None

        dV_active = grad_evecs[:, active_idx]
        VtdV      = evecs.T @ dV_active
        deval     = grad_evals if grad_evals is not None else torch.zeros(N, device=evals.device)
        gaps      = evals[:, None] - evals[None, active_idx]
        gaps_clamped = gaps.sign() * gaps.abs().clamp(min=min_gap)
        F_active  = VtdV / gaps_clamped
        for j_local, j_global in enumerate(active_idx):
            F_active[j_global, j_local] = deval[j_global]

        V_active = evecs[:, active_idx]
        grad_A   = (evecs @ F_active) @ V_active.T
        return 0.5 * (grad_A + grad_A.T), None


def stable_eigh(A: torch.Tensor, min_gap: float = 1.0):
    return _StableEigh.apply(A, min_gap)


def differentiable_eigh(
    S: torch.Tensor, M_diag: torch.Tensor, k: int, min_gap: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable generalized eigendecomposition with stable backward."""
    M_sqrt_inv = 1.0 / M_diag.sqrt().clamp(min=1e-8)
    S_std      = (S * M_sqrt_inv[None, :]) * M_sqrt_inv[:, None]
    S_std      = 0.5 * (S_std + S_std.T)
    evals_all, evecs_all = stable_eigh(S_std, min_gap)
    return evals_all[1:k+1], evecs_all[:, 1:k+1] * M_sqrt_inv[:, None]


# =============================================================================
# Shared diffusion descriptor computation
# =============================================================================

def _diffusion_solve(
    S: torch.Tensor, M: torch.Tensor, E: torch.Tensor, alpha: float,
) -> torch.Tensor:
    """Solve (S + αM) x = M·E for raw diffusion values.

    Args:
        S: (N, N) stiffness matrix.
        M: (N,) mass diagonal.
        E: (N, L) indicator matrix (one-hot columns for landmarks).
        alpha: diffusion scale.

    Returns:
        (N, L) raw diffusion values (unnormalised).
    """
    return torch.linalg.solve(S + alpha * torch.diag(M), M[:, None] * E)


def _diffusion_descriptors(
    S: torch.Tensor, M: torch.Tensor, E: torch.Tensor,
    alphas: Tuple[float, ...],
) -> torch.Tensor:
    """Compute L2-normalised diffusion fingerprint descriptors.

    For each diffusion scale α, solves (S + αM) x = M·E and concatenates.
    The result is a per-vertex descriptor of dimension L * len(alphas),
    where L is the number of landmark columns in E.

    Args:
        S: (N, N) stiffness matrix.
        M: (N,) mass diagonal.
        E: (N, L) indicator matrix (one-hot columns for landmarks).
        alphas: tuple of diffusion scales.

    Returns:
        (N, L*len(alphas)) L2-normalised descriptors.
    """
    parts = [_diffusion_solve(S, M, E, a) for a in alphas]
    return F.normalize(torch.cat(parts, dim=1), p=2, dim=1)


# =============================================================================
# Differentiable gradient and divergence from learned grad_coeffs
# =============================================================================

def _apply_gradient(
    grad_coeffs: torch.Tensor, knn: torch.Tensor, u: torch.Tensor,
) -> torch.Tensor:
    """Apply the learned gradient operator to scalar field(s).

    Computes (∇f)_i = Σ_j g_ij (f_j - f_i) for each vertex i.
    Fully differentiable through grad_coeffs and u.

    Args:
        grad_coeffs: (N, k, 3) per-neighbor gradient coefficients.
        knn: (N, k) neighbor indices.
        u: (N,) or (N, L) scalar field(s).

    Returns:
        (N, 3) or (N, L, 3) gradient field.
    """
    if u.dim() == 1:
        du = u[knn] - u[:, None]                            # (N, k)
        return (grad_coeffs * du.unsqueeze(2)).sum(dim=1)    # (N, 3)
    else:
        # Batched: u is (N, L)
        du = u[knn] - u[:, None, :]                          # (N, k, L)
        return torch.einsum('nkd,nkl->nld', grad_coeffs, du)  # (N, L, 3)


def _apply_divergence(
    grad_coeffs: torch.Tensor, knn: torch.Tensor, areas: torch.Tensor,
    X: torch.Tensor,
) -> torch.Tensor:
    """Apply the learned divergence operator to vector field(s).

    Computes div(X)_j = Σ_{i: j∈nbr(i)} area_i * g_ij · X_i
                      + area_j * center_coeff_j · X_j

    This is the adjoint of _apply_gradient w.r.t. the area-weighted inner product.
    Fully differentiable through grad_coeffs, areas, and X.

    Args:
        grad_coeffs: (N, k, 3) per-neighbor gradient coefficients.
        knn: (N, k) neighbor indices.
        areas: (N,) per-vertex areas.
        X: (N, 3) or (N, L, 3) vector field(s).

    Returns:
        (N,) or (N, L) divergence.
    """
    N = grad_coeffs.shape[0]
    device = grad_coeffs.device

    if X.dim() == 2:
        # Single field: X is (N, 3)
        dot_nb = (grad_coeffs * X[:, None, :]).sum(dim=2)   # (N, k)
        weighted_nb = areas[:, None] * dot_nb                # (N, k)
        div = torch.zeros(N, device=device, dtype=X.dtype)
        div.scatter_add_(0, knn.reshape(-1), weighted_nb.reshape(-1))

        center_coeffs = -grad_coeffs.sum(dim=1)              # (N, 3)
        div = div + areas * (center_coeffs * X).sum(dim=1)
        return div
    else:
        # Batched: X is (N, L, 3)
        L = X.shape[1]
        dot_nb = torch.einsum('nkd,nld->nkl', grad_coeffs, X)   # (N, k, L)
        weighted_nb = areas[:, None, None] * dot_nb               # (N, k, L)
        div = torch.zeros(N, L, device=device, dtype=X.dtype)
        knn_exp = knn.unsqueeze(2).expand(-1, -1, L)              # (N, k, L)
        div.scatter_add_(0, knn_exp.reshape(-1, L), weighted_nb.reshape(-1, L))

        center_coeffs = -grad_coeffs.sum(dim=1)                   # (N, 3)
        div = div + areas[:, None] * torch.einsum('nd,nld->nl', center_coeffs, X)
        return div


# =============================================================================
# Loss modules — share forward signature:
#   forward(S_A, S_B, M_A, M_B, rng, *, corr_a=None, corr_b=None)
#   → (loss: Tensor, metrics: Dict[str, float])
# Each has a `weight` attribute for the weighted sum in training_step.
# =============================================================================


# =============================================================================
# InfoNCE / DCL contrastive loss (supervised)
# =============================================================================

class SoftCorrespondenceLoss(nn.Module):
    """Correspondence-aware contrastive loss via diffusion fingerprint descriptors."""

    def __init__(
        self,
        num_landmarks: int = 128,
        alphas: Tuple[float, ...] = (1.0, 10.0, 100.0),
        temperature: float = 0.07,
        num_sample_vertices: int = 512,
        landmark_seed: int = 0,
        loss_type: str = "infonce",
        dclw_sigma: float = 0.5,
        weight: float = 1.0,
    ):
        super().__init__()
        self.num_landmarks       = num_landmarks
        self.alphas              = alphas
        self.temperature         = temperature
        self.num_sample_vertices = num_sample_vertices
        self.landmark_seed       = landmark_seed
        self.loss_type           = loss_type
        self.dclw_sigma          = dclw_sigma
        self.weight              = weight

    @staticmethod
    def _compute_descriptors(S, M, E, alphas) -> torch.Tensor:
        return _diffusion_descriptors(S, M, E, alphas)

    def forward(
        self,
        S_A: torch.Tensor, S_B: torch.Tensor,
        M_A: torch.Tensor, M_B: torch.Tensor,
        rng: np.random.RandomState,
        corr_a: Optional[np.ndarray] = None,
        corr_b: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        N_A, N_B = S_A.shape[0], S_B.shape[0]
        device   = S_A.device

        if corr_a is None or corr_b is None:
            assert N_A == N_B
            pool = np.arange(N_A); ca = pool; cb = pool
        else:
            pool = _compute_bijective_refs(corr_a, corr_b)
            if len(pool) < self.num_landmarks + self.num_sample_vertices:
                pool = np.arange(len(corr_a))
            ca, cb = corr_a, corr_b

        lm_rng  = np.random.RandomState(self.landmark_seed)
        L       = min(self.num_landmarks, len(pool))
        lm_refs = pool[lm_rng.choice(len(pool), size=L, replace=False)]

        E_A = torch.zeros(N_A, L, device=device, dtype=S_A.dtype)
        E_A[ca[lm_refs], torch.arange(L, device=device)] = 1.0
        E_B = torch.zeros(N_B, L, device=device, dtype=S_B.dtype)
        E_B[cb[lm_refs], torch.arange(L, device=device)] = 1.0

        desc_A = self._compute_descriptors(S_A, M_A, E_A, self.alphas)
        desc_B = self._compute_descriptors(S_B, M_B, E_B, self.alphas)

        V           = min(self.num_sample_vertices, len(pool))
        sample_refs = pool[rng.choice(len(pool), size=V, replace=False)]
        dA          = desc_A[ca[sample_refs]]
        dB          = desc_B[cb[sample_refs]]
        sim         = (dA @ dB.T) / self.temperature
        labels      = torch.arange(V, device=device)

        if self.loss_type == "infonce":
            loss_nce = 0.5 * (F.cross_entropy(sim, labels) +
                              F.cross_entropy(sim.T, labels))
        elif self.loss_type in ("dcl", "dclw"):
            pos_sim  = torch.diag(sim)
            neg_mask = ~torch.eye(V, dtype=torch.bool, device=device)
            neg_A2B  = sim.masked_select(neg_mask).view(V, V - 1)
            neg_B2A  = sim.T.masked_select(neg_mask).view(V, V - 1)
            lA = -pos_sim + torch.logsumexp(neg_A2B, dim=1)
            lB = -pos_sim + torch.logsumexp(neg_B2A, dim=1)
            if self.loss_type == "dclw":
                w  = (2 - V * F.softmax(pos_sim * (self.temperature / self.dclw_sigma),
                                        dim=0)).detach()
                lA = w * lA; lB = w * lB
            loss_nce = 0.5 * (lA.mean() + lB.mean())
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        with torch.no_grad():
            pred       = sim.argmax(dim=1)
            train_acc  = (pred == labels).float().mean().item()
            topk_accs  = {}
            for k in (3, 5, 10):
                if k <= V:
                    _, tk = sim.topk(k, dim=1)
                    topk_accs[f'train_top{k}'] = (
                        tk == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        metrics = {'loss_total': loss_nce.item(), 'loss_nce': loss_nce.item(),
                   'train_acc': train_acc, **topk_accs}
        return loss_nce, metrics


class IsospectralityLoss(nn.Module):
    """Isospectrality loss: penalise eigenvalue mismatch between two Laplacians.

    Two shapes from the same category should have similar LBO spectra
    (isometry invariance). Differentiable via torch.linalg.eigvalsh —
    eigenvalue gradients are stable even for repeated eigenvalues.

    This is an unsupervised loss — it does not require GT correspondences.
    """

    def __init__(self, num_eigenvalues: int = 30, weight: float = 1.0):
        super().__init__()
        self.num_eigenvalues = num_eigenvalues
        self.weight = weight

    def forward(
        self,
        S_A: torch.Tensor, S_B: torch.Tensor,
        M_A: torch.Tensor, M_B: torch.Tensor,
        rng: np.random.RandomState,
        corr_a: Optional[np.ndarray] = None,
        corr_b: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        k = min(self.num_eigenvalues, S_A.shape[0] - 1, S_B.shape[0] - 1)
        if k < 1:
            zero = torch.tensor(0.0, device=S_A.device)
            return zero, {'loss_iso': 0.0}

        eigs = []
        for S, M_diag in [(S_A, M_A), (S_B, M_B)]:
            M_inv_sqrt = 1.0 / M_diag.sqrt().clamp(min=1e-8)
            L_std = S * M_inv_sqrt[:, None] * M_inv_sqrt[None, :]
            L_std = 0.5 * (L_std + L_std.T)
            all_eigs = torch.linalg.eigvalsh(L_std)
            eigs.append(all_eigs[1:k+1])  # skip λ_0 ≈ 0

        loss = F.mse_loss(eigs[0], eigs[1])
        return loss, {'loss_iso': loss.item()}


class DiffusionDistributionLoss(nn.Module):
    """Distributional matching of diffusion distances (unsupervised).

    Requires that the distribution of pairwise diffusion descriptor distances
    within shape A matches that within shape B. No GT correspondences needed —
    just the assumption that same-category shapes have similar intrinsic geometry.

    Computes diffusion fingerprints using the shared _diffusion_descriptors(),
    samples random vertex pairs on each shape independently, computes pairwise
    cosine similarities, sorts them, and penalises the L2 difference between
    sorted distributions (1D Wasserstein / Earth Mover's Distance).
    """

    def __init__(
        self,
        num_landmarks: int = 128,
        alphas: Tuple[float, ...] = (1.0, 10.0, 100.0),
        num_sample_vertices: int = 256,
        landmark_seed: int = 0,
        weight: float = 1.0,
    ):
        super().__init__()
        self.num_landmarks       = num_landmarks
        self.alphas              = alphas
        self.num_sample_vertices = num_sample_vertices
        self.landmark_seed       = landmark_seed
        self.weight              = weight

    def forward(
        self,
        S_A: torch.Tensor, S_B: torch.Tensor,
        M_A: torch.Tensor, M_B: torch.Tensor,
        rng: np.random.RandomState,
        corr_a: Optional[np.ndarray] = None,
        corr_b: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        N_A, N_B = S_A.shape[0], S_B.shape[0]
        device = S_A.device

        # Place landmarks independently on each shape (no correspondences needed)
        lm_rng = np.random.RandomState(self.landmark_seed)
        L = min(self.num_landmarks, N_A, N_B)

        lm_A = lm_rng.choice(N_A, size=L, replace=False)
        lm_B = lm_rng.choice(N_B, size=L, replace=False)

        E_A = torch.zeros(N_A, L, device=device, dtype=S_A.dtype)
        E_A[lm_A, torch.arange(L, device=device)] = 1.0
        E_B = torch.zeros(N_B, L, device=device, dtype=S_B.dtype)
        E_B[lm_B, torch.arange(L, device=device)] = 1.0

        desc_A = _diffusion_descriptors(S_A, M_A, E_A, self.alphas)
        desc_B = _diffusion_descriptors(S_B, M_B, E_B, self.alphas)

        # Sample random vertices on each shape independently
        V = min(self.num_sample_vertices, N_A, N_B)
        sample_A = rng.choice(N_A, size=V, replace=False)
        sample_B = rng.choice(N_B, size=V, replace=False)

        dA = desc_A[sample_A]  # (V, D)
        dB = desc_B[sample_B]  # (V, D)

        # Pairwise cosine similarities within each shape
        sim_A = dA @ dA.T  # (V, V) — descriptors are L2-normalised
        sim_B = dB @ dB.T  # (V, V)

        # Extract upper triangle (exclude diagonal = self-similarity)
        triu_idx = torch.triu_indices(V, V, offset=1, device=device)
        vals_A = sim_A[triu_idx[0], triu_idx[1]]
        vals_B = sim_B[triu_idx[0], triu_idx[1]]

        # Sort and compare: 1D Wasserstein distance approximation
        vals_A_sorted = vals_A.sort().values
        vals_B_sorted = vals_B.sort().values
        loss = F.mse_loss(vals_A_sorted, vals_B_sorted)

        return loss, {'loss_dist': loss.item()}


class GeodesicDiffusionLoss(nn.Module):
    """Geodesic transfer loss via diffusion distances (supervised).

    Computes diffusion distances from landmark vertices on both shapes using
    the learned Laplacian, transfers via GT correspondence, and penalises the
    area-weighted L2 mismatch. This is a direct regression signal — stronger
    than contrastive ranking (InfoNCE) because it penalises the actual distance
    values, not just the relative ordering.

    For each landmark l on A with correspondent corr(l) on B:
        loss_l = Σ_v area_B(corr(v)) · (d_A(l, v) - d_B(corr(l), corr(v)))²

    where d_X(s, ·) = (S_X + αM_X)⁻¹ M_X δ_s are diffusion distances from the
    learned Laplacian. Averaged over landmarks and diffusion scales α.

    Reuses _diffusion_solve() — same linear system as SoftCorrespondenceLoss.
    """

    def __init__(
        self,
        num_landmarks: int = 128,
        alphas: Tuple[float, ...] = (0.1, 1.0, 10.0),
        num_sample_vertices: int = 512,
        landmark_seed: int = 0,
        normalize: bool = True,
        weight: float = 1.0,
    ):
        super().__init__()
        self.num_landmarks       = num_landmarks
        self.alphas              = alphas
        self.num_sample_vertices = num_sample_vertices
        self.landmark_seed       = landmark_seed
        self.normalize           = normalize
        self.weight              = weight

    def forward(
        self,
        S_A: torch.Tensor, S_B: torch.Tensor,
        M_A: torch.Tensor, M_B: torch.Tensor,
        rng: np.random.RandomState,
        corr_a: Optional[np.ndarray] = None,
        corr_b: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        N_A, N_B = S_A.shape[0], S_B.shape[0]
        device = S_A.device

        if corr_a is None or corr_b is None:
            assert N_A == N_B
            pool = np.arange(N_A); ca = pool; cb = pool
        else:
            pool = _compute_bijective_refs(corr_a, corr_b)
            if len(pool) < self.num_landmarks + self.num_sample_vertices:
                pool = np.arange(len(corr_a))
            ca, cb = corr_a, corr_b

        # Pick fixed landmarks (same seed every step for stability)
        lm_rng = np.random.RandomState(self.landmark_seed)
        L = min(self.num_landmarks, len(pool))
        lm_refs = pool[lm_rng.choice(len(pool), size=L, replace=False)]

        # Pick random sample vertices for evaluation
        V = min(self.num_sample_vertices, len(pool))
        sample_refs = pool[rng.choice(len(pool), size=V, replace=False)]

        # Build indicator matrices for landmarks
        E_A = torch.zeros(N_A, L, device=device, dtype=S_A.dtype)
        E_A[ca[lm_refs], torch.arange(L, device=device)] = 1.0
        E_B = torch.zeros(N_B, L, device=device, dtype=S_B.dtype)
        E_B[cb[lm_refs], torch.arange(L, device=device)] = 1.0

        # Area weights at sample vertices on B (for area-weighted L2)
        area_B = M_B[cb[sample_refs]]                  # (V,)
        area_B = area_B / area_B.sum()                  # normalise to sum to 1

        # Accumulate loss over diffusion scales
        total_loss = torch.tensor(0.0, device=device)
        for alpha in self.alphas:
            # Raw diffusion distances: (N_X, L) — column l is distance from landmark l
            D_A = _diffusion_solve(S_A, M_A, E_A, alpha)   # (N_A, L)
            D_B = _diffusion_solve(S_B, M_B, E_B, alpha)   # (N_B, L)

            # Sample corresponding vertices
            d_A = D_A[ca[sample_refs]]                      # (V, L)
            d_B = D_B[cb[sample_refs]]                      # (V, L)

            # Normalise each landmark's profile to unit norm so the loss
            # compares profile *shape*, not absolute scale (which is tiny
            # and produces negligible gradients without normalisation).
            if self.normalize:
                d_A = F.normalize(d_A, p=2, dim=0)              # (V, L)
                d_B = F.normalize(d_B, p=2, dim=0)              # (V, L)

            # Area-weighted L2 per landmark, averaged over landmarks
            sq_diff = (d_A - d_B) ** 2                      # (V, L)
            total_loss = total_loss + (area_B[:, None] * sq_diff).sum(dim=0).mean()

        loss = total_loss / len(self.alphas)

        return loss, {'loss_geodiff': loss.item()}


class HeatMethodGeodesicLoss(nn.Module):
    """Geodesic transfer loss via the full differentiable heat method (supervised).

    Implements all three steps of the heat method (Crane et al. 2013) in
    differentiable PyTorch:
        1. Heat diffuse:  u = (S + t·diag(M))⁻¹ M δ_s
        2. Grad + norm:   X = -∇u / max(|∇u|, ε)    via _apply_gradient
        3. Poisson solve:  φ = (S + ε·I)⁻¹ div(X)     via _apply_divergence

    The resulting φ approximates geodesic distance from the source.
    Gradients flow through all three steps back to grad_coeffs and areas.

    Requires grad_coeffs and knn to be passed as kwargs from training_step.
    """

    def __init__(
        self,
        num_landmarks: int = 32,
        num_sample_vertices: int = 512,
        landmark_seed: int = 0,
        eps: float = 1e-6,
        normalize: bool = True,
        weight: float = 1.0,
    ):
        super().__init__()
        self.num_landmarks       = num_landmarks
        self.num_sample_vertices = num_sample_vertices
        self.landmark_seed       = landmark_seed
        self.eps                 = eps
        self.normalize           = normalize
        self.weight              = weight

    @staticmethod
    def _heat_method_distances(
        S: torch.Tensor, M: torch.Tensor,
        grad_coeffs: torch.Tensor, knn: torch.Tensor,
        source_indices: torch.Tensor, eps: float = 1e-6,
    ) -> torch.Tensor:
        """Full heat method in differentiable PyTorch.

        Args:
            S: (N, N) stiffness matrix.
            M: (N,) mass diagonal.
            grad_coeffs: (N, k, 3) gradient coefficients.
            knn: (N, k) neighbor indices.
            source_indices: (L,) landmark vertex indices.
            eps: regularisation for Poisson solve.

        Returns:
            phi: (N, L) approximate geodesic distances from each landmark.
        """
        N = S.shape[0]
        L = len(source_indices)
        device = S.device

        # Diffusion time from mean vertex area
        h = M.mean().sqrt()
        t = h * h

        # Step 1: Heat diffuse — reuses _diffusion_solve with alpha = t
        E = torch.zeros(N, L, device=device, dtype=S.dtype)
        E[source_indices, torch.arange(L, device=device)] = 1.0
        u = _diffusion_solve(S, M, E, t)                       # (N, L)

        # Step 2: Gradient + normalise
        grad_u = _apply_gradient(grad_coeffs, knn, u)           # (N, L, 3)
        norms = grad_u.norm(dim=2, keepdim=True).clamp(min=1e-8)
        X = -grad_u / norms                                      # (N, L, 3)

        # Step 3: Divergence + Poisson solve
        div_X = _apply_divergence(grad_coeffs, knn, M, X)       # (N, L)
        S_reg = S + eps * torch.eye(N, device=device, dtype=S.dtype)
        phi = torch.linalg.solve(S_reg, div_X)                   # (N, L)

        # Shift so source has distance 0, take abs
        phi = phi - phi[source_indices, torch.arange(L, device=device)].unsqueeze(0)
        phi = phi.abs()

        return phi

    def forward(
        self,
        S_A: torch.Tensor, S_B: torch.Tensor,
        M_A: torch.Tensor, M_B: torch.Tensor,
        rng: np.random.RandomState,
        corr_a: Optional[np.ndarray] = None,
        corr_b: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        grad_coeffs_a = kwargs.get('grad_coeffs_a')
        grad_coeffs_b = kwargs.get('grad_coeffs_b')
        knn_a = kwargs.get('knn_a')
        knn_b = kwargs.get('knn_b')

        if grad_coeffs_a is None or knn_a is None:
            zero = torch.tensor(0.0, device=S_A.device)
            return zero, {'loss_heatgeo': 0.0}

        N_A, N_B = S_A.shape[0], S_B.shape[0]
        device = S_A.device

        if corr_a is None or corr_b is None:
            assert N_A == N_B
            pool = np.arange(N_A); ca = pool; cb = pool
        else:
            pool = _compute_bijective_refs(corr_a, corr_b)
            if len(pool) < self.num_landmarks + self.num_sample_vertices:
                pool = np.arange(len(corr_a))
            ca, cb = corr_a, corr_b

        # Pick landmarks and sample vertices
        lm_rng = np.random.RandomState(self.landmark_seed)
        L = min(self.num_landmarks, len(pool))
        lm_refs = pool[lm_rng.choice(len(pool), size=L, replace=False)]

        V = min(self.num_sample_vertices, len(pool))
        sample_refs = pool[rng.choice(len(pool), size=V, replace=False)]

        # Source indices on each shape
        src_A = torch.from_numpy(ca[lm_refs].copy()).long().to(device)
        src_B = torch.from_numpy(cb[lm_refs].copy()).long().to(device)

        # Full heat method on both shapes
        phi_A = self._heat_method_distances(
            S_A, M_A, grad_coeffs_a, knn_a, src_A, self.eps)   # (N_A, L)
        phi_B = self._heat_method_distances(
            S_B, M_B, grad_coeffs_b, knn_b, src_B, self.eps)   # (N_B, L)

        # Sample corresponding vertices
        d_A = phi_A[ca[sample_refs]]                             # (V, L)
        d_B = phi_B[cb[sample_refs]]                             # (V, L)

        # Normalise each landmark's profile (unit norm) for scale-invariance
        if self.normalize:
            d_A = F.normalize(d_A, p=2, dim=0)
            d_B = F.normalize(d_B, p=2, dim=0)

        # Area-weighted L2
        area_B = M_B[cb[sample_refs]]
        area_B = area_B / area_B.sum()

        sq_diff = (d_A - d_B) ** 2                               # (V, L)
        loss = (area_B[:, None] * sq_diff).sum(dim=0).mean()

        return loss, {'loss_heatgeo': loss.item()}


# =============================================================================
# Evaluation utilities (non-differentiable)
# =============================================================================

def _fmt_topk(m: Dict[str, float], prefix: str = '') -> str:
    parts = [f"top1={m[f'{prefix}accuracy']*100:5.1f}%"]
    for k in (3, 5, 10):
        key = f'{prefix}top{k}_acc'
        if key in m:
            parts.append(f"top{k}={m[key]*100:5.1f}%")
    parts.append(f"Err={m[f'{prefix}mean_error']:.4f}")
    return "  ".join(parts)


def _correspondence_metrics(
    eigvecs_a: np.ndarray, eigvecs_b: np.ndarray,
    mass_a: np.ndarray,    mass_b: np.ndarray,
    vertices_b: np.ndarray, n: int,
    gt_corr: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Functional map metrics from eigenbases."""
    n_a = eigvecs_a.shape[0]
    weighted_phi_b = eigvecs_b[gt_corr[:n_a]] * mass_b[gt_corr[:n_a], None] \
        if gt_corr is not None else eigvecs_b[:n_a] * mass_b[:n_a, None]
    C    = weighted_phi_b.T @ eigvecs_a[:n_a]
    k_fm = C.shape[0]
    I    = np.eye(k_fm)
    metrics = {
        'ortho_error':   float(np.linalg.norm(C.T @ C - I, 'fro')),
        'biject_error':  float(np.linalg.norm(C @ C.T - I, 'fro')),
        'corr_error':    float(np.linalg.norm(C - I, 'fro')),
        'diag_ratio':    float(np.sum(np.diag(C) ** 2) / (np.sum(C ** 2) + 1e-10)),
    }
    projected_a = eigvecs_a[:n_a] @ C.T
    nbrs = NearestNeighbors(n_neighbors=min(10, eigvecs_b.shape[0]),
                            algorithm='auto').fit(eigvecs_b)
    _, indices = nbrs.kneighbors(projected_a)
    pred_corr  = indices[:, 0]
    gt         = gt_corr[:n_a] if gt_corr is not None else np.arange(n_a)
    metrics['accuracy'] = float((pred_corr == gt).mean())
    for k in (3, 5, 10):
        if k <= indices.shape[1]:
            metrics[f'top{k}_acc'] = float(
                (indices[:, :k] == gt[:, None]).any(axis=1).mean())
    errors = np.linalg.norm(vertices_b[pred_corr] - vertices_b[gt], axis=1)
    bb_diag = np.linalg.norm(vertices_b.max(0) - vertices_b.min(0))
    metrics['mean_error']   = float(errors.mean()   / bb_diag)
    metrics['median_error'] = float(np.median(errors) / bb_diag)
    return metrics


def _build_gt_corr_from_pair(pair: PairSample) -> Optional[np.ndarray]:
    n_a = len(pair.verts_a)
    if (len(pair.corr_a) == n_a and
            np.array_equal(pair.corr_a, np.arange(n_a)) and
            np.array_equal(pair.corr_b, np.arange(n_a))):
        return None
    gt_corr = np.full(n_a, -1, dtype=np.int64)
    gt_corr[pair.corr_a] = pair.corr_b
    unmapped = gt_corr == -1
    if unmapped.any() and (~unmapped).any():
        nbrs = NearestNeighbors(n_neighbors=1).fit(pair.verts_a[~unmapped])
        _, idx = nbrs.kneighbors(pair.verts_a[unmapped])
        mapped_indices = np.where(~unmapped)[0]
        gt_corr[unmapped] = gt_corr[mapped_indices[idx.flatten()]]
    return gt_corr


class GeodesicCache:
    """Lazy geodesic distance computation on mesh B.

    Factorizes the Laplacian once (fast), then computes geodesic distances
    from individual source vertices on demand. Caches distance vectors so
    the same GT target vertex is never solved twice.

    Usage in _pointwise_metrics:
        geo_errors = geo_cache.compute_errors(pred_corr, gt_corr)
    """

    def __init__(self, pair: PairSample):
        import potpourri3d as pp3d

        verts_full = pair._verts_b_full if pair._verts_b_full is not None else pair.verts_b
        faces = pair.faces_b
        self._idx_b = pair._idx_b if pair._idx_b is not None else np.arange(len(pair.verts_b))

        # Factorize Laplacian (one-time cost, fast)
        self._solver = pp3d.MeshHeatMethodDistanceSolver(verts_full, faces)

        # Normalisation: sqrt(surface area)
        v0, v1, v2 = verts_full[faces[:, 0]], verts_full[faces[:, 1]], verts_full[faces[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
        self._sqrt_area = np.sqrt(max(area, 1e-12))

        # Cache: subsampled B index → normalised distance to all subsampled B vertices
        self._dist_cache: Dict[int, np.ndarray] = {}

    @classmethod
    def from_precomputed(cls, dist_cache: Dict[int, np.ndarray],
                         sqrt_area: float, idx_b: np.ndarray) -> 'GeodesicCache':
        """Create a cache from precomputed distances (no solver needed)."""
        obj = object.__new__(cls)
        obj._dist_cache = dist_cache
        obj._sqrt_area = sqrt_area
        obj._idx_b = idx_b
        obj._solver = None
        return obj

    def _get_dists_from(self, sub_idx: int) -> np.ndarray:
        """Get normalised geodesic distances from subsampled vertex sub_idx to all others."""
        if sub_idx not in self._dist_cache:
            if self._solver is None:
                raise RuntimeError(
                    f"GeodesicCache: no solver and vertex {sub_idx} not precomputed")
            full_idx = int(self._idx_b[sub_idx])
            full_dists = self._solver.compute_distance(full_idx)  # (N_full,)
            self._dist_cache[sub_idx] = full_dists[self._idx_b] / self._sqrt_area
        return self._dist_cache[sub_idx]

    def precompute_from_gt(self, gt_corr: np.ndarray) -> None:
        """Eagerly compute and cache distances from all unique GT targets.

        Call this once during on_fit_start so that per-pair evaluation
        incurs zero geodesic computation cost.
        """
        unique_gt = np.unique(gt_corr)
        for g in unique_gt:
            self._get_dists_from(int(g))

    def drop_solver(self) -> None:
        """Release the pp3d solver to free memory (after precomputation)."""
        self._solver = None

    def compute_errors(self, pred_corr: np.ndarray, gt_corr: np.ndarray) -> np.ndarray:
        """Compute normalised geodesic error for each (pred, gt) pair.

        Only solves from unique GT targets — typically much fewer than N_sub.
        Results are cached, so calling with different pred_corr but same gt_corr
        (e.g. different evaluators on the same eigenbasis) is essentially free.

        Args:
            pred_corr: (N_A,) predicted B vertex indices.
            gt_corr: (N_A,) ground-truth B vertex indices.

        Returns:
            (N_A,) normalised geodesic errors.
        """
        unique_gt = np.unique(gt_corr)
        # Ensure all unique GT targets are cached
        for g in unique_gt:
            self._get_dists_from(int(g))
        # Vectorised lookup
        errors = np.empty(len(pred_corr), dtype=np.float32)
        for g in unique_gt:
            mask = gt_corr == g
            errors[mask] = self._dist_cache[int(g)][pred_corr[mask]]
        return errors


def _precompute_geo_cache_worker(args):
    """Multiprocessing worker: build solver, compute distances, return cache data.

    Runs in a separate process. The solver is discarded after computation —
    only the distance dict (numpy arrays) is returned.

    Returns:
        (name, dist_cache, sqrt_area, idx_b) or (name, None, None, None) on failure.
    """
    import potpourri3d as pp3d

    name, verts_full, faces, idx_b, unique_targets = args
    try:
        solver = pp3d.MeshHeatMethodDistanceSolver(verts_full, faces)

        v0, v1, v2 = verts_full[faces[:, 0]], verts_full[faces[:, 1]], verts_full[faces[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
        sqrt_area = np.sqrt(max(area, 1e-12))

        dist_cache = {}
        for g in unique_targets:
            full_idx = int(idx_b[g])
            full_dists = solver.compute_distance(full_idx)
            dist_cache[int(g)] = full_dists[idx_b] / sqrt_area

        return name, dist_cache, sqrt_area, idx_b
    except Exception as e:
        return name, None, None, None


def _build_geo_cache(pair: PairSample) -> Optional['GeodesicCache']:
    """Build a GeodesicCache for pair, or None if faces/potpourri3d unavailable."""
    if pair.faces_b is None:
        return None
    try:
        return GeodesicCache(pair)
    except ImportError:
        print("    [geodesic] potpourri3d not installed — skipping geodesic metrics",
              flush=True)
        return None
    except Exception as e:
        print(f"    [geodesic] failed to build cache: {e}", flush=True)
        return None


@torch.no_grad()
def _eigh_from_dense_L(
    L: torch.Tensor, M_diag_t: torch.Tensor, num_eigenvectors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    M_inv_sqrt = 1.0 / M_diag_t.sqrt().clamp(min=1e-8)
    L_std      = L * M_inv_sqrt[:, None] * M_inv_sqrt[None, :]
    L_std      = 0.5 * (L_std + L_std.T)
    evals_all, evecs_all = torch.linalg.eigh(L_std)
    evals = evals_all[:num_eigenvectors].cpu().numpy()
    evecs = (M_inv_sqrt[:, None] * evecs_all[:, :num_eigenvectors]).cpu().numpy()
    return evals, evecs


@torch.no_grad()
def _eigh_from_sparse_L(
    L_sp: torch.Tensor, M_diag_t: torch.Tensor, num_eigenvectors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    N      = L_sp.shape[0]
    L_np   = L_sp.cpu().numpy()
    M_np   = M_diag_t.cpu().numpy()
    rows, cols = np.nonzero(L_np)
    L_scipy = scipy.sparse.csc_matrix((L_np[rows, cols], (rows, cols)), shape=(N, N))
    M_scipy = scipy.sparse.diags(M_np)
    try:
        evals, evecs = scipy.sparse.linalg.eigsh(
            L_scipy, k=num_eigenvectors, M=M_scipy,
            sigma=-1e-6, which='LM', v0=np.ones(N))
        order = np.argsort(evals)
        return evals[order], evecs[:, order]
    except Exception:
        return _eigh_from_dense_L(L_sp, M_diag_t, num_eigenvectors)


@torch.no_grad()
def evaluate_pair(
    model: LaplacianTransformerModule,
    pair: PairSample,
    k: int,
    num_eigenvectors: int,
    device: torch.device,
    k_sparsify: Optional[int] = None,
    evaluators: Optional[List] = None,
    geo_cache = None,
    eval_variants: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate correspondence quality using functional maps (non-differentiable).

    Args:
        k_sparsify: kNN for sparsification mask. None = reuse k.
        evaluators: List of ShapePairEvaluator instances. When provided,
            runs each evaluator and prefixes metrics with evaluator.name.
            When None, uses the legacy _correspondence_metrics path.
        geo_cache: Precomputed GeodesicCache for mesh B.
            If None and pair has faces_b, builds on the fly.
        eval_variants: Which eigenbasis variants to evaluate.
            Subset of ['dense', 'sp', 'iso']. None = all three.
    """
    if eval_variants is None:
        eval_variants = ['dense', 'sp', 'iso']
    do_dense = 'dense' in eval_variants
    do_sp    = 'sp' in eval_variants
    do_iso   = 'iso' in eval_variants
    n_a             = len(pair.verts_a)
    is_gradient_mode = False
    dense_bases: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}   # (evecs, evals, mass)
    sparse_bases: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    iso_bases: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for label, verts in [('A', pair.verts_a), ('B', pair.verts_b)]:
        verts_t    = torch.from_numpy(verts).float().to(device)
        knn_np     = compute_knn(verts, k)
        batch_data = Batch.from_data_list(
            [build_patch_data(verts_t, knn_np, device)]).to(device)
        fwd        = model._forward_pass(batch_data)

        if fwd.get('grad_coeffs') is not None:
            is_gradient_mode = True
            knn_t     = torch.from_numpy(knn_np).long().to(device)
            L, M_diag = assemble_anisotropic_laplacian(fwd['grad_coeffs'],
                                                       fwd['areas'], knn_t)
            M_np = M_diag.cpu().numpy()

            if do_dense:
                evals_d, evecs_d = _eigh_from_sparse_L(L, M_diag, num_eigenvectors)
                dense_bases[label] = (evecs_d, evals_d, M_np)

            if do_sp:
                if k_sparsify is not None and k_sparsify != k:
                    knn_sp_np = compute_knn(verts, k_sparsify)
                    knn_sp_t = torch.from_numpy(knn_sp_np).long().to(device)
                    L_sp = _sparsify_L_to_knn(L, knn_sp_t)
                else:
                    L_sp = _sparsify_L_to_knn(L, knn_t)
                evals_sp, evecs_sp = _eigh_from_sparse_L(L_sp, M_diag, num_eigenvectors)
                sparse_bases[label] = (evecs_sp, evals_sp, M_np)

            if do_iso:
                L_iso, M_iso = assemble_isotropic_laplacian(
                    fwd['grad_coeffs'], fwd['areas'], knn_t)
                evals_iso, evecs_iso = _eigh_from_sparse_L(L_iso, M_iso, num_eigenvectors)
                iso_bases[label] = (evecs_iso, evals_iso, M_np)
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
            evals, evecs = compute_laplacian_eigendecomposition(
                S, num_eigenvectors, mass_matrix=M)
            M_np = np.array(M.diagonal()).flatten()
            dense_bases[label] = (evecs, evals, M_np)

    gt_corr = _build_gt_corr_from_pair(pair)
    if geo_cache is None:
        geo_cache = _build_geo_cache(pair)

    # Get mass from whichever variant is available
    first_bases = dense_bases or sparse_bases or iso_bases
    mA = first_bases['A'][2]
    mB = first_bases['B'][2]

    # --- Use evaluators if provided ---
    if evaluators is not None:
        metrics = {}

        # Dense variant
        if dense_bases:
            evA_d, evalsA_d, _ = dense_bases['A']
            evB_d, evalsB_d, _ = dense_bases['B']
            for evaluator in evaluators:
                ev_metrics = evaluator.evaluate(
                    evA_d, evB_d, evalsA_d, evalsB_d,
                    mA, mB, pair.verts_b, gt_corr=gt_corr,
                    geo_cache=geo_cache,
                )
                for mk, mv in ev_metrics.items():
                    metrics[f"{evaluator.name}/{mk}"] = mv

        # Sparse variant
        if is_gradient_mode and sparse_bases:
            evA_sp, evalsA_sp, _ = sparse_bases['A']
            evB_sp, evalsB_sp, _ = sparse_bases['B']
            for evaluator in evaluators:
                sp_metrics = evaluator.evaluate(
                    evA_sp, evB_sp, evalsA_sp, evalsB_sp,
                    mA, mB, pair.verts_b, gt_corr=gt_corr,
                    geo_cache=geo_cache,
                )
                for mk, mv in sp_metrics.items():
                    metrics[f"sp_{evaluator.name}/{mk}"] = mv

        # Isotropic variant
        if is_gradient_mode and iso_bases:
            evA_iso, evalsA_iso, _ = iso_bases['A']
            evB_iso, evalsB_iso, _ = iso_bases['B']
            for evaluator in evaluators:
                iso_metrics = evaluator.evaluate(
                    evA_iso, evB_iso, evalsA_iso, evalsB_iso,
                    mA, mB, pair.verts_b, gt_corr=gt_corr,
                    geo_cache=geo_cache,
                )
                for mk, mv in iso_metrics.items():
                    metrics[f"iso_{evaluator.name}/{mk}"] = mv

        return metrics

    # --- Legacy path (backward compatible) ---
    if dense_bases:
        evA_d, evalsA_d, _ = dense_bases['A']
        evB_d, evalsB_d, _ = dense_bases['B']
        metrics = _correspondence_metrics(evA_d, evB_d, mA, mB,
                                          pair.verts_b, n_a, gt_corr=gt_corr)
    else:
        metrics = {}
    if is_gradient_mode and sparse_bases:
        evA_sp, _, _ = sparse_bases['A']
        evB_sp, _, _ = sparse_bases['B']
        sp = _correspondence_metrics(evA_sp, evB_sp, mA, mB,
                                     pair.verts_b, n_a, gt_corr=gt_corr)
        for key, val in sp.items():
            metrics[f'sp_{key}'] = val
    return metrics


@torch.no_grad()
def evaluate_pair_robust(
    pair: PairSample,
    num_eigenvectors: int,
    n_neighbors: int = 30,
    evaluators: Optional[List] = None,
    geo_cache = None,
) -> Dict[str, float]:
    """Evaluate using robust Laplacian (baseline, no model).

    Args:
        evaluators: List of ShapePairEvaluator instances. When provided,
            runs each evaluator and prefixes metrics with evaluator.name.
            When None, uses the legacy _correspondence_metrics path.
        geo_cache: Precomputed GeodesicCache for mesh B.
            If None and pair has faces_b, builds on the fly.
    """
    import robust_laplacian
    n_a = len(pair.verts_a)
    bases: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, verts in [('A', pair.verts_a), ('B', pair.verts_b)]:
        S, M     = robust_laplacian.point_cloud_laplacian(verts, n_neighbors=n_neighbors)
        evals, evecs = compute_laplacian_eigendecomposition(S, num_eigenvectors, mass_matrix=M)
        bases[label] = (evecs, evals, np.array(M.diagonal()).flatten())

    gt_corr = _build_gt_corr_from_pair(pair)
    if geo_cache is None:
        geo_cache = _build_geo_cache(pair)
    evA, evalsA, mA = bases['A']
    evB, evalsB, mB = bases['B']

    if evaluators is not None:
        metrics = {}
        for evaluator in evaluators:
            ev_metrics = evaluator.evaluate(
                evA, evB, evalsA, evalsB,
                mA, mB, pair.verts_b, gt_corr=gt_corr,
                geo_cache=geo_cache,
            )
            for mk, mv in ev_metrics.items():
                metrics[f"{evaluator.name}/{mk}"] = mv
        return metrics

    # Legacy path
    return _correspondence_metrics(evA, evB, mA, mB,
                                   pair.verts_b, n_a, gt_corr=gt_corr)


# =============================================================================
# LightningModule
# =============================================================================

def cosine_flat_scheduler(
    optimizer: torch.optim.Optimizer,
    T_max: int,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay from base_lr to eta_min over T_max epochs, then flat.

    Unlike CosineAnnealingLR this does not cycle back up after T_max.
    The multiplier is clamped so progress never exceeds 1.0.
    """
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def _lambda(epoch: int, base_lr: float) -> float:
        if base_lr < 1e-12:
            return 1.0
        t       = min(epoch, T_max)
        cosine  = 0.5 * (1.0 + math.cos(math.pi * t / T_max))
        lr      = eta_min + (base_lr - eta_min) * cosine
        return lr / base_lr

    lambdas = [lambda epoch, blr=blr: _lambda(epoch, blr) for blr in base_lrs]
    # LambdaLR accepts a list of lambdas (one per param group) or a single one
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambdas if len(lambdas) > 1 else lambdas[0],
    )



# =============================================================================
# Model export callback
# =============================================================================

class BestModelExportCallback(lightning.pytorch.callbacks.ModelCheckpoint):
    """ModelCheckpoint subclass that saves only the inner LaplacianTransformerModule.

    Because it inherits from ModelCheckpoint, WandbLogger (log_model=true) will
    automatically upload its best checkpoint as a W&B artifact — identical in
    structure and size to the input pretrained checkpoint.

    Pass the same monitor/mode/dirpath as the main ModelCheckpoint so the two
    checkpoints sit side by side and the best-model tracking is consistent.
    """

    def _save_checkpoint(self, trainer: "lightning.pytorch.Trainer", filepath: str) -> None:
        if not trainer.is_global_zero:
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        pl_module: "FunctionalMapModule" = trainer.lightning_module  # type: ignore[assignment]
        inner_model: lightning.pytorch.LightningModule = pl_module.model  # type: ignore[assignment]
        ckpt: Dict[str, Any] = {
            "state_dict":                    inner_model.state_dict(),
            "hyper_parameters":              dict(inner_model.hparams),
            "pytorch-lightning_version":     lightning.__version__,
        }
        torch.save(ckpt, filepath)
        self.best_model_path = filepath

        # WandbLogger only scans the *last* ModelCheckpoint in the callbacks list,
        # so log_model='all' will never reliably upload our callback. Upload directly.
        try:
            import wandb
            if wandb.run is not None:
                artifact_name = f"model-{wandb.run.id}"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="model",
                    metadata={
                        "score":   float(self.current_score) if self.current_score is not None else None,
                        "epoch":   trainer.current_epoch,
                        "monitor": self.monitor,
                    },
                )
                artifact.add_file(filepath, name="model.ckpt")
                wandb.run.log_artifact(artifact, aliases=["latest", "best"])
                print(f"  [Export] Uploaded to W&B artifact '{artifact_name}'")
            else:
                print("  [Export] W&B upload skipped: no active wandb run")
        except Exception as e:
            import traceback
            print(f"  [Export] W&B upload failed: {e}")
            traceback.print_exc()

        print(f"  [Export] Saved inner model checkpoint → {filepath}")





class FunctionalMapModule(LaplacianModuleBase):
    """Fine-tune a pretrained LaplacianTransformerModule for shape correspondence."""

    def __init__(
        self,
        checkpoint_path: str,
        optimizer_cfg: Optional[DictConfig] = None,
        losses: Optional[List[nn.Module]] = None,
        loss_fn: Optional[SoftCorrespondenceLoss] = None,  # backward compat
        scheduler_cfg: Optional[DictConfig] = None,
        random_init: bool = False,
        keep_areas_head: bool = False,
        freeze_input_projection: bool = False,
        freeze_areas_head: bool = False,
        k: int = 15,
        k_sparsify: Optional[int] = None,
        num_eigenvectors: int = 100,
        sparsify_laplacian: bool = False,
        max_vertices: int = 0,
        max_vertices_val: int = 0,
        vertex_noise: float = 0.05,
        w_prox: float = 20.0,
        seed: int = 42,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: str = "all-linear",
        lora_dora: bool = False,
        lora_rslora: bool = False,
        profile_steps: int = 0,   # print profiler summary every N steps (0 = off)
        # Evaluation control
        eval_only: bool = False,       # run baselines only, then stop (no training)
        skip_baselines: bool = False,  # skip baseline computation in on_fit_start
        eval_variants: Optional[List[str]] = None,  # ['dense', 'sp', 'iso'] or subset
        # GeomFuM evaluation (optional)
        use_geomfum_eval: bool = False,
        geomfum_descriptors: Optional[List[str]] = None,
        geomfum_zoomout: bool = True,
        geomfum_zoomout_k_init: int = 20,
        geomfum_zoomout_k_final: int = 50,
        geomfum_zoomout_n_iters: int = 10,
        geomfum_fmap_lmbda: float = 1e3,
        geomfum_fmap_resolvent_gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(optimizer_cfg=optimizer_cfg,
                         scheduler_cfg=scheduler_cfg, **kwargs)
        self.save_hyperparameters(ignore=["losses", "loss_fn", "optimizer_cfg", "scheduler_cfg"])

        # Build loss list: prefer explicit `losses`, fall back to `loss_fn`
        if losses is not None:
            self._losses = nn.ModuleList(losses)
        elif loss_fn is not None:
            if not hasattr(loss_fn, 'weight'):
                loss_fn.weight = 1.0
            self._losses = nn.ModuleList([loss_fn])
        elif eval_only:
            self._losses = nn.ModuleList()
        else:
            raise ValueError("Must provide either `losses` or `loss_fn`")

        self.model: Optional[nn.Module] = None
        self._train_rng: Optional[np.random.RandomState] = None
        self._val_outputs: Dict[int, List[Dict[str, float]]] = {}

        # Validate k_sparsify
        if k_sparsify is not None:
            assert k_sparsify <= k, (
                f"k_sparsify ({k_sparsify}) must be <= k ({k})")

        # Build evaluators
        self._evaluators = [SpectralNNEvaluator()]
        if use_geomfum_eval:
            self._evaluators.append(FunctionalMapEvaluator(
                descriptors=geomfum_descriptors or ['hks', 'wks'],
                use_zoomout=geomfum_zoomout,
                zoomout_k_init=geomfum_zoomout_k_init,
                zoomout_k_final=geomfum_zoomout_k_final,
                zoomout_n_iters=geomfum_zoomout_n_iters,
                fmap_lmbda=geomfum_fmap_lmbda,
                fmap_resolvent_gamma=geomfum_fmap_resolvent_gamma,
            ))

        # Cache for geodesic solvers (pair_name → GeodesicCache)
        # Precomputed in on_fit_start, reused in validation_step.
        self._geo_cache: Dict[str, Optional] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.model is not None:
            return
        hp = self.hparams

        self.model = LaplacianTransformerModule.load_from_checkpoint(
            hp.checkpoint_path, map_location="cpu",
            normalize_patch_features=True, scale_areas_by_patch_size=True,
        )

        if hp.random_init:
            areas_state: Optional[dict] = None
            if hp.keep_areas_head:
                areas_state = {
                    name: {k: v.clone() for k, v in mod.state_dict().items()}
                    for name, mod in self.model.named_modules()
                    if "area" in name.lower()
                    and any(True for _ in mod.parameters(recurse=False))
                }
            for mod in self.model.modules():
                if hasattr(mod, 'reset_parameters'):
                    mod.reset_parameters()
            if areas_state:
                for name, mod in self.model.named_modules():
                    if name in areas_state:
                        mod.load_state_dict(areas_state[name])

        if hp.freeze_input_projection:
            for name, param in self.model.named_parameters():
                if 'input_projection' in name:
                    param.requires_grad_(False)

        if hp.freeze_areas_head:
            for name, param in self.model.named_parameters():
                if 'area' in name.lower():
                    param.requires_grad_(False)

        use_prox = hp.w_prox > 0 and not hp.random_init and not hp.use_lora
        if use_prox:
            ref = torch.cat([p.detach().flatten() for p in self.model.parameters()])
            self.register_buffer("ref_params",  ref)
            self.register_buffer("ref_norm_sq", ref.pow(2).sum().clamp(min=1e-8))

        if hp.use_lora:
            try:
                from peft import get_peft_model, LoraConfig
            except ImportError:
                raise ImportError("LoRA requires: pip install peft")
            target = ("all-linear" if hp.lora_target_modules == "all-linear"
                      else [m.strip() for m in hp.lora_target_modules.split(",")])
            self.model = get_peft_model(self.model, LoraConfig(
                r=hp.lora_rank, lora_alpha=hp.lora_alpha,
                lora_dropout=hp.lora_dropout, target_modules=target,
                bias="none", use_dora=hp.lora_dora, use_rslora=hp.lora_rslora,
            ))
            self.model.print_trainable_parameters()

        self._train_rng = np.random.RandomState(hp.seed + self.global_rank * 10_007)
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"  [FMModule rank={self.global_rank}] "
              f"Trainable: {n_train:,} / {n_total:,}")
        if self.global_rank == 0 and self._scheduler_cfg is not None:
            # Surface the effective T_max so it appears in logs / W&B config
            t_max = getattr(self._scheduler_cfg, 'keywords', {}).get('T_max', '?')
            print(f"  [FMModule] Scheduler T_max={t_max} epochs")

        # Log source code to W&B with proper filtering (excludes venvs, etc.)
        if self.trainer.global_rank == 0:
            from neural_local_laplacian.utils.utils import wandb_log_code
            wandb_log_code(self.logger)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        if self._optimizer_cfg is None:
            return None
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = self._optimizer_cfg(params=trainable)
        if self._scheduler_cfg is None:
            return optimizer
        scheduler = self._scheduler_cfg(optimizer=optimizer)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    # ------------------------------------------------------------------
    # Shared metric aggregation + printing
    # ------------------------------------------------------------------

    def _summarise_and_print(
        self,
        all_metrics: List[Dict[str, float]],
        label: str,
        silent: bool = False,
    ) -> Dict[str, float]:
        """Aggregate per-pair metrics and print a summary.

        Used by both on_fit_start (baselines) and on_validation_epoch_end
        so they produce identical output format.

        Args:
            all_metrics: List of per-pair metric dicts from evaluators.
            label: Human-readable label (e.g. "Robust baseline", "Val epoch 3").
            silent: If True, skip printing (used for non-zero DDP ranks).

        Returns:
            summary: Dict mapping metric names to averaged values.
        """
        if not all_metrics:
            return {}

        # Collect all keys across all pairs (some pairs may have {} from failures)
        all_keys = set()
        for m in all_metrics:
            all_keys.update(m.keys())
        all_keys = sorted(all_keys)

        # Aggregate: mean of each key (skip NaN / missing)
        summary: Dict[str, float] = {}
        for mk in all_keys:
            vals = [m[mk] for m in all_metrics
                    if mk in m and np.isfinite(m[mk])]
            if vals:
                summary[mk] = float(np.mean(vals))

        # --- Identify primary metric (spectral_nn dense) ---
        nn_prefix = "spectral_nn/"
        has_nn = any(k.startswith(nn_prefix) for k in all_keys)

        if has_nn:
            mean_acc = summary.get(f"{nn_prefix}accuracy", 0.0)
            mean_err = summary.get(f"{nn_prefix}mean_error", 0.0)
        else:
            mean_acc = summary.get("accuracy", 0.0)
            mean_err = summary.get("mean_error", 0.0)

        # Sparse / isotropic primary accuracy
        sp_key = f"sp_{nn_prefix}accuracy" if has_nn else "sp_accuracy"
        iso_key = f"iso_{nn_prefix}accuracy" if has_nn else "iso_accuracy"
        sp_acc = summary.get(sp_key)
        iso_acc = summary.get(iso_key)

        # --- Print summary ---
        if not silent:
            # Collect all variant rows: (display_name, prefix) for each evaluator
            # e.g. ("spectral_nn", ""), ("sp_spectral_nn", "sp_"), ("iso_spectral_nn", "iso_")
            rows = []
            for ev in self._evaluators:
                for prefix, variant_label in [("", ""), ("sp_", "sp_"), ("iso_", "iso_")]:
                    key = f"{prefix}{ev.name}/accuracy"
                    if summary.get(key) is not None:
                        rows.append((f"{variant_label}{ev.name}", prefix, ev))

            if not rows:
                return summary

            name_w = max(len(r[0]) for r in rows)
            n_pairs = len(all_metrics)

            print(f"  {label}:", flush=True)

            for display_name, prefix, ev in rows:
                full_prefix = f"{prefix}{ev.name}"
                acc = summary.get(f"{full_prefix}/accuracy", 0.0)
                err = summary.get(f"{full_prefix}/mean_error", 0.0)

                # Count failures
                acc_key = f"{full_prefix}/accuracy"
                n_valid = sum(1 for m in all_metrics
                              if acc_key in m and np.isfinite(m[acc_key]))
                n_fail = n_pairs - n_valid
                fail_str = f"  ({n_fail}/{n_pairs} failed)" if n_fail else ""

                # Build metrics string
                parts = [f"    {display_name:<{name_w}}  top1={acc*100:5.1f}%"]

                for topk in (3, 5, 10):
                    topk_val = summary.get(f"{full_prefix}/top{topk}_acc")
                    if topk_val is not None:
                        parts.append(f"  top{topk}={topk_val*100:5.1f}%")

                parts.append(f"  Err={err:.4f}")

                gfm_acc = summary.get(f"{full_prefix}/geomfum_accuracy")
                if gfm_acc is not None:
                    parts.append(f"  gfm={gfm_acc*100:5.1f}%")

                # Geodesic metrics (Princeton benchmark)
                geo_err = summary.get(f"{full_prefix}/geo_mean_error")
                if geo_err is not None:
                    geo_parts = []
                    for thresh in (1, 5, 10, 25):
                        val = summary.get(f"{full_prefix}/geo_at_{thresh:02d}pct", 0.0)
                        geo_parts.append(f"@{thresh}%={val*100:5.1f}%")
                    parts.append(f"  │ {'  '.join(geo_parts)}  gErr={geo_err:.4f}")

                parts.append(fail_str)

                print("".join(parts), flush=True)

        return summary

    def _export_eval_results(
        self,
        eval_results: Dict[str, Dict[str, Tuple[List[Dict], Dict]]],
        all_val_datasets: List[Tuple[str, List]],
    ) -> None:
        """Save eval-only results as CSVs and comparison plots.

        Each run gets a unique directory based on the W&B run name and key
        hyperparameters, so multiple eval runs don't overwrite each other.

        Args:
            eval_results: {ds_name: {method_name: (per_pair_metrics, summary)}}
            all_val_datasets: [(ds_name, pairs)] for pair name lookup.
        """
        import csv
        from pathlib import Path

        # Build a unique run directory name
        hp = self.hparams
        try:
            import wandb
            run_name = wandb.run.name if wandb.run is not None else "local"
        except ImportError:
            run_name = "local"
        run_tag = f"{run_name}_k{hp.k}"
        if hp.k_sparsify is not None:
            run_tag += f"_ksp{hp.k_sparsify}"

        out_dir = Path(self.trainer.default_root_dir) / "eval_results" / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Saving eval results to {out_dir}/", flush=True)

        # Save run config
        config_path = out_dir / "config.yaml"
        try:
            import yaml
            config_dict = {
                "k": hp.k,
                "k_sparsify": hp.k_sparsify,
                "num_eigenvectors": hp.num_eigenvectors,
                "checkpoint_path": hp.checkpoint_path,
                "eval_variants": hp.eval_variants,
                "max_vertices": hp.max_vertices,
                "max_vertices_val": hp.max_vertices_val,
                "run_name": run_name,
            }
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            print(f"    {config_path}", flush=True)
        except ImportError:
            pass

        # Build pair name lookup per dataset
        pair_names: Dict[str, List[str]] = {}
        for ds_name, pairs in all_val_datasets:
            pair_names[ds_name] = [p.name for p in pairs]

        for ds_name, methods in eval_results.items():
            ds_dir = out_dir / ds_name
            ds_dir.mkdir(parents=True, exist_ok=True)

            # ── Per-pair CSV ────────────────────────────────────────────────
            # Rows = pairs, columns = pair_name + method/metric
            names = pair_names.get(ds_name, [])
            all_metric_keys = set()
            for method_name, (per_pair, _) in methods.items():
                for m in per_pair:
                    all_metric_keys.update(m.keys())
            all_metric_keys = sorted(all_metric_keys)

            csv_path = ds_dir / "per_pair.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                # Header: pair_name, then method/metric for each method
                header = ["pair"]
                for method_name in methods:
                    for mk in all_metric_keys:
                        header.append(f"{method_name}/{mk}")
                writer.writerow(header)

                # Rows
                n_pairs = max(len(pp) for pp, _ in methods.values()) if methods else 0
                for i in range(n_pairs):
                    row = [names[i] if i < len(names) else f"pair_{i}"]
                    for method_name in methods:
                        per_pair, _ = methods[method_name]
                        m = per_pair[i] if i < len(per_pair) else {}
                        for mk in all_metric_keys:
                            row.append(f"{m.get(mk, '')}")
                    writer.writerow(row)
            print(f"    {csv_path}", flush=True)

            # ── Summary CSV ─────────────────────────────────────────────────
            # Rows = methods, columns = metrics
            csv_path = ds_dir / "summary.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["method"] + all_metric_keys)
                for method_name, (_, summary) in methods.items():
                    row = [method_name]
                    for mk in all_metric_keys:
                        row.append(f"{summary.get(mk, '')}")
                    writer.writerow(row)
            print(f"    {csv_path}", flush=True)

            # ── Comparison bar chart ────────────────────────────────────────
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # Select key metrics for the chart
                # Select key metrics for the chart — auto-discover from summaries
                chart_metrics = []
                all_summary_keys = set()
                for _, (_, s) in methods.items():
                    all_summary_keys.update(s.keys())
                # Include accuracy and geodesic threshold metrics
                for key in sorted(all_summary_keys):
                    if any(pat in key for pat in (
                        '/accuracy', '/geo_at_', '/geomfum_accuracy',
                    )):
                        chart_metrics.append(key)

                if chart_metrics and methods:
                    method_names = list(methods.keys())
                    n_methods = len(method_names)
                    n_metrics = len(chart_metrics)
                    x = np.arange(n_metrics)
                    width = 0.8 / n_methods

                    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.2), 6))
                    for j, method_name in enumerate(method_names):
                        _, summary = methods[method_name]
                        vals = [summary.get(mk, 0.0) * 100 for mk in chart_metrics]
                        ax.bar(x + j * width, vals, width, label=method_name)

                    # Short display labels
                    short_labels = []
                    for mk in chart_metrics:
                        label = mk.replace("spectral_nn/", "").replace("_spectral_nn/", " ")
                        short_labels.append(label)

                    ax.set_ylabel("% (higher = better)")
                    ax.set_title(f"Eval comparison — {ds_name}")
                    ax.set_xticks(x + width * (n_methods - 1) / 2)
                    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
                    ax.legend(fontsize=8)
                    ax.set_ylim(0, 105)
                    fig.tight_layout()

                    plot_path = ds_dir / "comparison.png"
                    fig.savefig(plot_path, dpi=150)
                    plt.close(fig)
                    print(f"    {plot_path}", flush=True)
            except ImportError:
                print("    (matplotlib not installed — skipping comparison plot)")

    # ------------------------------------------------------------------
    # Baseline evaluation (runs once before first training epoch)
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Log robust-Laplacian and epoch-0 model baselines to W&B at step 0.

        Only runs on rank 0 to avoid duplicated evaluation in DDP.
        Iterates over all val dataset specifications.
        """
        if not self.trainer.is_global_zero:
            return

        hp     = self.hparams
        device = self.device
        dm     = self.trainer.datamodule
        if dm is None or not dm._val_dataset_specifications:
            return

        def _subsample(pair):
            mv = hp.max_vertices_val if hp.max_vertices_val > 0 else hp.max_vertices
            if mv > 0:
                return subsample_pair(pair, mv,
                                      np.random.RandomState(_stable_hash(pair.name)))
            return pair

        # Collect all val pairs across all datasets
        all_val_datasets: List[Tuple[str, List[PairSample]]] = []
        for spec in dm._val_dataset_specifications:
            ds = spec.dataset
            ds_name = getattr(ds, 'name', ds.__class__.__name__)
            pairs = [ds[i] for i in range(len(ds))]
            all_val_datasets.append((ds_name, pairs))

        # Precompute geodesic caches for all val pairs across all datasets.
        # Build solver + eagerly compute distances from all unique GT targets
        # so that per-pair evaluation incurs zero geodesic cost.
        all_pairs = [(name, p) for name, pairs in all_val_datasets for p in pairs]
        print(f"\n  Precomputing geodesic caches ({len(all_pairs)} pairs)...", flush=True)
        t0_geo = time.perf_counter()

        # Prepare worker args: subsample, extract gt, collect serialisable data
        worker_args = []
        pair_meta = []   # (pair_name, sent_to_worker) — to reconstruct caches later
        for _, p in all_pairs:
            if p.name in self._geo_cache:
                pair_meta.append((p.name, False))
                continue
            sub_p = _subsample(p)
            if sub_p.faces_b is None:
                pair_meta.append((p.name, False))
                continue
            gt_corr = _build_gt_corr_from_pair(sub_p)

            verts_full = sub_p._verts_b_full if sub_p._verts_b_full is not None else sub_p.verts_b
            idx_b = sub_p._idx_b if sub_p._idx_b is not None else np.arange(len(sub_p.verts_b))

            if gt_corr is not None:
                unique_targets = np.unique(gt_corr)
            else:
                # Identity correspondence: every B vertex is a GT target
                unique_targets = np.arange(len(sub_p.verts_b))

            worker_args.append((
                p.name, verts_full, sub_p.faces_b, idx_b, unique_targets,
            ))
            pair_meta.append((p.name, True))

        # Run workers in parallel
        _n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        n_workers = min(len(worker_args), max(1, _n_cpus - 1))
        if worker_args:
            if n_workers > 1:
                import multiprocessing as _mp
                print(f"    Parallel precomputation: {len(worker_args)} pairs, "
                      f"{n_workers} workers", flush=True)
                with _mp.Pool(n_workers) as pool:
                    results = {}
                    for i, r in enumerate(pool.imap_unordered(
                            _precompute_geo_cache_worker, worker_args)):
                        results[r[0]] = r[1:]
                        done = i + 1
                        if done % 10 == 0 or done == len(worker_args):
                            dt = time.perf_counter() - t0_geo
                            print(f"    [{done}/{len(worker_args)}] {dt:.1f}s",
                                  flush=True)
            else:
                results = {}
                for i, args in enumerate(worker_args):
                    r = _precompute_geo_cache_worker(args)
                    results[r[0]] = r[1:]
                    done = i + 1
                    if done % 10 == 0 or done == len(worker_args):
                        dt = time.perf_counter() - t0_geo
                        print(f"    [{done}/{len(worker_args)}] {dt:.1f}s",
                              flush=True)
        else:
            results = {}

        # Reconstruct GeodesicCache objects from worker results
        for pair_name, sent_to_worker in pair_meta:
            if pair_name in self._geo_cache:
                continue
            if sent_to_worker and pair_name in results:
                dist_cache, sqrt_area, idx_b = results[pair_name]
                if dist_cache is not None:
                    self._geo_cache[pair_name] = GeodesicCache.from_precomputed(
                        dist_cache, sqrt_area, idx_b)
                else:
                    self._geo_cache[pair_name] = None
            else:
                self._geo_cache[pair_name] = None

        dt_geo = time.perf_counter() - t0_geo
        n_with_geo = sum(1 for v in self._geo_cache.values() if v is not None)
        print(f"    Done: {n_with_geo}/{len(all_pairs)} pairs with faces "
              f"({dt_geo:.1f}s)", flush=True)

        # Run baselines per dataset
        # Collect results for CSV/plot export in eval_only mode
        eval_results: Dict[str, Dict[str, Tuple[List[Dict], Dict]]] = {}
        # eval_results[ds_name][method_name] = (per_pair_metrics, summary_dict)

        if hp.skip_baselines:
            print("\n  [skip_baselines=true] Skipping baseline computation.",
                  flush=True)
        else:
            for ds_name, val_pairs in all_val_datasets:
                if not val_pairs:
                    continue
                eval_results[ds_name] = {}
                pair_names = [p.name for p in val_pairs]
                n_total = len(val_pairs)
                cw = len(str(n_total))
                ds_label = f" [{ds_name}]" if len(all_val_datasets) > 1 else ""

                # ── Robust Laplacian baseline ──────────────────────────────────
                try:
                    import robust_laplacian  # noqa: F401
                    print(f"\n  Computing robust Laplacian baseline{ds_label} "
                          f"({n_total} pairs)...", flush=True)
                    robust_metrics = []
                    for i, p in enumerate(val_pairs):
                        print(f"    [robust {i+1:>{cw}}/{n_total}] {p.name}...",
                              end="", flush=True)
                        t0_ = time.perf_counter()
                        m = evaluate_pair_robust(_subsample(p), hp.num_eigenvectors,
                                                 evaluators=self._evaluators,
                                                 geo_cache=self._geo_cache.get(p.name))
                        dt_ = time.perf_counter() - t0_
                        robust_metrics.append(m)
                        print(f" {dt_:.1f}s", flush=True)
                    rb_summary = self._summarise_and_print(
                        robust_metrics, f"Robust baseline{ds_label}")
                    self.logger.log_metrics(
                        {f"baseline/robust/{ds_name}/{k}": v
                         for k, v in rb_summary.items()}, step=0)
                    eval_results[ds_name]["robust"] = (robust_metrics, rb_summary)
                except ImportError:
                    print("  (robust_laplacian not installed — skipping robust baseline)")

                # ── Epoch-0 model baseline ─────────────────────────────────────
                init_label = "random init" if hp.random_init else "pretrained"
                print(f"\n  Computing {init_label} model baseline{ds_label} "
                      f"({n_total} pairs)...", flush=True)
                self.model.eval()
                ep0_metrics = []
                with torch.no_grad():
                    for i, p in enumerate(val_pairs):
                        sub_p = _subsample(p)
                        print(f"    [model  {i+1:>{cw}}/{n_total}] {p.name}...",
                              end="", flush=True)
                        t0_ = time.perf_counter()
                        m = evaluate_pair(self.model, sub_p,
                                          hp.k, hp.num_eigenvectors, device,
                                          k_sparsify=hp.k_sparsify,
                                          evaluators=self._evaluators,
                                          geo_cache=self._geo_cache.get(p.name),
                                          eval_variants=hp.eval_variants)
                        dt_ = time.perf_counter() - t0_
                        ep0_metrics.append(m)
                        print(f" {dt_:.1f}s", flush=True)
                self.model.train()

                ep0_summary = self._summarise_and_print(
                    ep0_metrics, f"Model baseline ({init_label}){ds_label}")
                self.logger.log_metrics(
                    {f"baseline/model/{ds_name}/{k}": v
                     for k, v in ep0_summary.items()}, step=0)
                eval_results[ds_name][f"model_{init_label}"] = (ep0_metrics, ep0_summary)

        # ── Eval-only mode: save CSVs and plots, then stop ────────────────
        if hp.eval_only:
            if eval_results:
                self._export_eval_results(eval_results, all_val_datasets)
            print("\n  [eval_only=true] Baselines complete — stopping.", flush=True)
            # Finalize W&B logging before exit
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass
            raise SystemExit(0)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: List[PairSample], batch_idx: int):
        hp, device = self.hparams, self.device
        total_loss = torch.tensor(0.0, device=device)
        metrics_list: List[Dict] = []

        prof = _StepProfiler(enabled=(hp.profile_steps > 0 and self.global_rank == 0))

        for pair in batch:
            with prof.phase("subsample + noise"):
                if hp.max_vertices > 0:
                    pair = subsample_pair(pair, hp.max_vertices, self._train_rng)
                if hp.vertex_noise > 0:
                    for attr in ("verts_a", "verts_b"):
                        v     = getattr(pair, attr)
                        scale = hp.vertex_noise * float(np.linalg.norm(v, axis=1).mean())
                        noise = self._train_rng.randn(*v.shape).astype(np.float32) * scale
                        setattr(pair, attr, v + noise)

            with prof.phase("knn (cpu)"):
                knn_a = compute_knn(pair.verts_a, hp.k)
                knn_b = compute_knn(pair.verts_b, hp.k)

            with prof.phase("patch build + transfer"):
                va_t = torch.from_numpy(pair.verts_a).float().to(device)
                vb_t = torch.from_numpy(pair.verts_b).float().to(device)
                knn_a_t = torch.from_numpy(knn_a).long().to(device)
                knn_b_t = torch.from_numpy(knn_b).long().to(device)
                bd_a = Batch.from_data_list([build_patch_data(va_t, knn_a, device)]).to(device)
                bd_b = Batch.from_data_list([build_patch_data(vb_t, knn_b, device)]).to(device)

            with prof.phase("transformer forward"):
                fwd_a = self.model._forward_pass(bd_a)
                fwd_b = self.model._forward_pass(bd_b)

            with prof.phase("laplacian assembly"):
                S_A, M_A = (assemble_anisotropic_laplacian(fwd_a['grad_coeffs'], fwd_a['areas'], knn_a_t)
                            if fwd_a.get('grad_coeffs') is not None
                            else assemble_dense_stiffness_and_mass(
                                fwd_a['stiffness_weights'], fwd_a['areas'],
                                fwd_a['attention_mask'], bd_a.vertex_indices,
                                bd_a.center_indices, getattr(bd_a, 'patch_idx', bd_a.batch)))
                S_B, M_B = (assemble_anisotropic_laplacian(fwd_b['grad_coeffs'], fwd_b['areas'], knn_b_t)
                            if fwd_b.get('grad_coeffs') is not None
                            else assemble_dense_stiffness_and_mass(
                                fwd_b['stiffness_weights'], fwd_b['areas'],
                                fwd_b['attention_mask'], bd_b.vertex_indices,
                                bd_b.center_indices, getattr(bd_b, 'patch_idx', bd_b.batch)))
                if hp.sparsify_laplacian:
                    if hp.k_sparsify is not None and hp.k_sparsify != hp.k:
                        knn_sp_a = compute_knn(pair.verts_a, hp.k_sparsify)
                        knn_sp_b = compute_knn(pair.verts_b, hp.k_sparsify)
                        knn_sp_a_t = torch.from_numpy(knn_sp_a).long().to(device)
                        knn_sp_b_t = torch.from_numpy(knn_sp_b).long().to(device)
                        S_A = _sparsify_L_to_knn(S_A, knn_sp_a_t)
                        S_B = _sparsify_L_to_knn(S_B, knn_sp_b_t)
                    else:
                        S_A = _sparsify_L_to_knn(S_A, knn_a_t)
                        S_B = _sparsify_L_to_knn(S_B, knn_b_t)

            with prof.phase("losses"):
                pair_loss = torch.tensor(0.0, device=device)
                m: Dict[str, float] = {}
                for loss_fn in self._losses:
                    l, lm = loss_fn(S_A, S_B, M_A, M_B, self._train_rng,
                                    corr_a=pair.corr_a, corr_b=pair.corr_b,
                                    grad_coeffs_a=fwd_a.get('grad_coeffs'),
                                    grad_coeffs_b=fwd_b.get('grad_coeffs'),
                                    knn_a=knn_a_t, knn_b=knn_b_t)
                    pair_loss = pair_loss + loss_fn.weight * l
                    m.update(lm)

            with prof.phase("prox regularization"):
                if hp.w_prox > 0 and hasattr(self, "ref_params"):
                    cur = torch.cat([p.flatten() for p in self.model.parameters()])
                    loss_prox = ((cur - self.ref_params) ** 2).sum() / self.ref_norm_sq
                    loss = pair_loss + hp.w_prox * loss_prox
                else:
                    loss_prox = torch.tensor(0.0, device=device)
                    loss = pair_loss

            if torch.isnan(loss):
                continue

            total_loss = total_loss + loss
            m["loss_prox"]  = loss_prox.item()
            m["loss_total"] = loss.item()
            metrics_list.append(m)

        if hp.profile_steps > 0 and self.global_rank == 0:
            step = self.global_step
            if step % hp.profile_steps == 0:
                print(prof.summary_str(step=step))
            prof.reset()

        if not metrics_list:
            return None

        total_loss = total_loss / len(metrics_list)
        avg = {k: float(np.mean([d[k] for d in metrics_list])) for k in metrics_list[0]}

        self.log("train/loss",      avg["loss_total"],           sync_dist=True, prog_bar=True)
        self.log("train/loss_prox", avg["loss_prox"],            sync_dist=True)
        # Log all loss-specific metrics (loss_nce, loss_iso, etc.)
        for mk in avg:
            if mk.startswith("loss_") and mk not in ("loss_total", "loss_prox"):
                self.log(f"train/{mk}", avg[mk], sync_dist=True)
        self.log("train/top1",      avg.get("train_acc", 0.0),   sync_dist=True, prog_bar=True)
        self.log("train/top3",      avg.get("train_top3", 0.0),  sync_dist=True)
        self.log("train/top5",      avg.get("train_top5", 0.0),  sync_dist=True)
        self.log("train/top10",     avg.get("train_top10", 0.0), sync_dist=True)
        self.log("train/lr",
                 self.trainer.optimizers[0].param_groups[0]["lr"],
                 rank_zero_only=True)
        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_outputs: Dict[int, List[Dict[str, float]]] = {}

    @torch.no_grad()
    def validation_step(self, batch: List[PairSample], batch_idx: int,
                        dataloader_idx: int = 0) -> None:
        assert len(batch) == 1
        pair = batch[0]
        mv = self.hparams.max_vertices_val or self.hparams.max_vertices
        if mv > 0:
            pair = subsample_pair(pair, mv,
                                  np.random.RandomState(_stable_hash(pair.name)))
        if dataloader_idx not in self._val_outputs:
            self._val_outputs[dataloader_idx] = []
        self._val_outputs[dataloader_idx].append(evaluate_pair(
            self.model, pair, self.hparams.k,
            self.hparams.num_eigenvectors, self.device,
            k_sparsify=self.hparams.k_sparsify,
            evaluators=self._evaluators,
            geo_cache=self._geo_cache.get(pair.name),
            eval_variants=self.hparams.eval_variants,
        ))

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return

        dm = self.trainer.datamodule
        val_specs = dm._val_dataset_specifications

        # Process each val dataset independently
        primary_acc = None  # from first dataset, used for checkpointing

        for dl_idx, outputs in sorted(self._val_outputs.items()):
            if not outputs:
                continue

            # Dataset name for logging and printing
            ds = val_specs[dl_idx].dataset if dl_idx < len(val_specs) else None
            ds_name = getattr(ds, 'name', f'val_{dl_idx}')

            # --- DDP gather ---
            sample = outputs[0]
            float_keys = [k for k, v in sample.items()
                          if isinstance(v, (int, float)) and not k.startswith("_")]
            local_t = torch.tensor(
                [[d.get(k, float("nan")) for k in float_keys] for d in outputs],
                device=self.device, dtype=torch.float32)

            gathered = self.all_gather(local_t)
            if gathered.dim() == 2:
                gathered = gathered.unsqueeze(0)
            all_flat = gathered.reshape(-1, len(float_keys))

            true_val_size = len(val_specs[dl_idx].dataset) if dl_idx < len(val_specs) else len(outputs)
            all_flat = all_flat[:true_val_size]

            all_metrics = [{k: all_flat[i, j].item() for j, k in enumerate(float_keys)}
                           for i in range(all_flat.shape[0])]

            # --- Summarise + print (shared code path) ---
            summary = self._summarise_and_print(
                all_metrics, f"Val epoch {self.current_epoch} [{ds_name}]",
                silent=not self.trainer.is_global_zero)

            # --- Log to W&B / Lightning ---
            prefix = f"val/{ds_name}" if len(val_specs) > 1 else "val"
            for mk, mv in summary.items():
                self.log(f"{prefix}/{mk}", mv, sync_dist=True,
                         add_dataloader_idx=False)

            # Primary metrics from first dataset
            nn_prefix = "spectral_nn/"
            has_nn = any(k.startswith(nn_prefix) for k in summary)
            acc_key = f"{nn_prefix}accuracy" if has_nn else "accuracy"
            sp_key = f"sp_{nn_prefix}accuracy" if has_nn else "sp_accuracy"
            iso_geo5_key = f"iso_{nn_prefix}geo_at_05pct" if has_nn else "iso_geo_at_05pct"

            if dl_idx == 0:
                self.log("val/top1", summary.get(acc_key, 0.0),
                         prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                if sp_key in summary:
                    self.log("val/sp_top1", summary[sp_key],
                             prog_bar=True, sync_dist=True, add_dataloader_idx=False)
                # Primary for checkpointing: prefer fmap evaluator geo@5%,
                # fall back to spectral_nn geo@5%, then sp top1, then dense top1
                primary_acc = None
                # Try fmap evaluators (iso variant) first
                for ev in self._evaluators:
                    if ev.name.startswith("fmap_"):
                        fmap_geo5_key = f"iso_{ev.name}/geo_at_05pct"
                        if fmap_geo5_key in summary:
                            primary_acc = summary[fmap_geo5_key]
                            break
                if primary_acc is None:
                    primary_acc = summary.get(
                        iso_geo5_key,
                        summary.get(sp_key, summary.get(acc_key, 0.0)))

        if primary_acc is not None:
            self.log("val/best_acc", primary_acc,
                     prog_bar=True, sync_dist=True, add_dataloader_idx=False)