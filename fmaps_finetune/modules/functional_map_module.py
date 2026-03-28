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
# InfoNCE / DCL contrastive loss
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
    ):
        super().__init__()
        self.num_landmarks       = num_landmarks
        self.alphas              = alphas
        self.temperature         = temperature
        self.num_sample_vertices = num_sample_vertices
        self.landmark_seed       = landmark_seed
        self.loss_type           = loss_type
        self.dclw_sigma          = dclw_sigma

    def _compute_descriptors(self, S, M, E) -> torch.Tensor:
        parts = []
        for alpha in self.alphas:
            D = torch.linalg.solve(S + alpha * torch.diag(M), M[:, None] * E)
            parts.append(D)
        return F.normalize(torch.cat(parts, dim=1), p=2, dim=1)

    def forward(
        self,
        S_A: torch.Tensor, S_B: torch.Tensor,
        M_A: torch.Tensor, M_B: torch.Tensor,
        rng: np.random.RandomState,
        corr_a: Optional[np.ndarray] = None,
        corr_b: Optional[np.ndarray] = None,
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

        desc_A = self._compute_descriptors(S_A, M_A, E_A)
        desc_B = self._compute_descriptors(S_B, M_B, E_B)

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


def _compute_geodesic_dists(pair: PairSample) -> Optional[np.ndarray]:
    """Compute normalised geodesic distance matrix on (subsampled) mesh B.

    Uses the heat method (potpourri3d) on the original mesh for accurate
    geodesics that can cut across face interiors. Distances are normalised
    by sqrt(total surface area).

    Returns:
        (N_B_sub, N_B_sub) matrix, or None if faces are not available.
    """
    if pair.faces_b is None:
        return None

    try:
        import potpourri3d as pp3d
    except ImportError:
        print("    [geodesic] potpourri3d not installed — skipping geodesic metrics",
              flush=True)
        return None

    verts_full = pair._verts_b_full if pair._verts_b_full is not None else pair.verts_b
    faces = pair.faces_b
    idx_b = pair._idx_b if pair._idx_b is not None else np.arange(len(pair.verts_b))
    n_sub = len(idx_b)

    # Heat method solver on the full mesh
    solver = pp3d.MeshHeatMethodDistanceSolver(verts_full, faces)

    # Compute geodesic distances from each subsampled vertex
    geo_sub = np.zeros((n_sub, n_sub), dtype=np.float32)
    for i, src in enumerate(idx_b):
        dists_from_src = solver.compute_distance(int(src))  # (N_full,)
        geo_sub[i] = dists_from_src[idx_b]

    # Normalise by sqrt(surface area)
    v0 = verts_full[faces[:, 0]]
    v1 = verts_full[faces[:, 1]]
    v2 = verts_full[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
    sqrt_area = np.sqrt(max(area, 1e-12))
    geo_sub /= sqrt_area

    return geo_sub


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
    geo_dists: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate correspondence quality using functional maps (non-differentiable).

    Args:
        k_sparsify: kNN for sparsification mask. None = reuse k.
        evaluators: List of ShapePairEvaluator instances. When provided,
            runs each evaluator and prefixes metrics with evaluator.name.
            When None, uses the legacy _correspondence_metrics path.
        geo_dists: Precomputed normalised geodesic distance matrix on B.
            If None and pair has faces_b, computes on the fly.
    """
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
            evals_d, evecs_d = _eigh_from_sparse_L(L, M_diag, num_eigenvectors)
            M_np = M_diag.cpu().numpy()
            dense_bases[label] = (evecs_d, evals_d, M_np)

            # Sparsify with a separate kNN graph when k_sparsify is set
            if k_sparsify is not None and k_sparsify != k:
                knn_sp_np = compute_knn(verts, k_sparsify)
                knn_sp_t = torch.from_numpy(knn_sp_np).long().to(device)
                L_sp = _sparsify_L_to_knn(L, knn_sp_t)
            else:
                L_sp = _sparsify_L_to_knn(L, knn_t)
            evals_sp, evecs_sp = _eigh_from_sparse_L(L_sp, M_diag, num_eigenvectors)
            sparse_bases[label] = (evecs_sp, evals_sp, M_np)

            # Isotropic Laplacian: w_ij = area_i * ||g_ij||^2 (PSD, sparse)
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
    if geo_dists is None:
        geo_dists = _compute_geodesic_dists(pair)
    evA_d, evalsA_d, mA = dense_bases['A']
    evB_d, evalsB_d, mB = dense_bases['B']

    # --- Use evaluators if provided ---
    if evaluators is not None:
        metrics = {}
        for evaluator in evaluators:
            ev_metrics = evaluator.evaluate(
                evA_d, evB_d, evalsA_d, evalsB_d,
                mA, mB, pair.verts_b, gt_corr=gt_corr,
                geo_dists=geo_dists,
            )
            for mk, mv in ev_metrics.items():
                metrics[f"{evaluator.name}/{mk}"] = mv

        # Also run evaluators on sparse bases
        if is_gradient_mode and sparse_bases:
            evA_sp, evalsA_sp, _ = sparse_bases['A']
            evB_sp, evalsB_sp, _ = sparse_bases['B']
            for evaluator in evaluators:
                sp_metrics = evaluator.evaluate(
                    evA_sp, evB_sp, evalsA_sp, evalsB_sp,
                    mA, mB, pair.verts_b, gt_corr=gt_corr,
                    geo_dists=geo_dists,
                )
                for mk, mv in sp_metrics.items():
                    metrics[f"sp_{evaluator.name}/{mk}"] = mv

        # Also run evaluators on isotropic bases (PSD, sparse)
        if is_gradient_mode and iso_bases:
            evA_iso, evalsA_iso, _ = iso_bases['A']
            evB_iso, evalsB_iso, _ = iso_bases['B']
            for evaluator in evaluators:
                iso_metrics = evaluator.evaluate(
                    evA_iso, evB_iso, evalsA_iso, evalsB_iso,
                    mA, mB, pair.verts_b, gt_corr=gt_corr,
                    geo_dists=geo_dists,
                )
                for mk, mv in iso_metrics.items():
                    metrics[f"iso_{evaluator.name}/{mk}"] = mv

        return metrics

    # --- Legacy path (backward compatible) ---
    metrics = _correspondence_metrics(evA_d, evB_d, mA, mB,
                                      pair.verts_b, n_a, gt_corr=gt_corr)
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
    geo_dists: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate using robust Laplacian (baseline, no model).

    Args:
        evaluators: List of ShapePairEvaluator instances. When provided,
            runs each evaluator and prefixes metrics with evaluator.name.
            When None, uses the legacy _correspondence_metrics path.
        geo_dists: Precomputed normalised geodesic distance matrix on B.
            If None and pair has faces_b, computes on the fly.
    """
    import robust_laplacian
    n_a = len(pair.verts_a)
    bases: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, verts in [('A', pair.verts_a), ('B', pair.verts_b)]:
        S, M     = robust_laplacian.point_cloud_laplacian(verts, n_neighbors=n_neighbors)
        evals, evecs = compute_laplacian_eigendecomposition(S, num_eigenvectors, mass_matrix=M)
        bases[label] = (evecs, evals, np.array(M.diagonal()).flatten())

    gt_corr = _build_gt_corr_from_pair(pair)
    if geo_dists is None:
        geo_dists = _compute_geodesic_dists(pair)
    evA, evalsA, mA = bases['A']
    evB, evalsB, mB = bases['B']

    if evaluators is not None:
        metrics = {}
        for evaluator in evaluators:
            ev_metrics = evaluator.evaluate(
                evA, evB, evalsA, evalsB,
                mA, mB, pair.verts_b, gt_corr=gt_corr,
                geo_dists=geo_dists,
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
        optimizer_cfg: DictConfig,
        loss_fn: SoftCorrespondenceLoss,
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
        cross_ratio_start: float = 0.0,
        cross_ratio_end: float = 0.5,
        curriculum_epochs: int = 50,
        seed: int = 42,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: str = "all-linear",
        lora_dora: bool = False,
        lora_rslora: bool = False,
        profile_steps: int = 0,   # print profiler summary every N steps (0 = off)
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
        self.save_hyperparameters(ignore=["loss_fn", "optimizer_cfg", "scheduler_cfg"])
        self.loss_fn: SoftCorrespondenceLoss = loss_fn
        self.model: Optional[nn.Module] = None
        self._train_rng: Optional[np.random.RandomState] = None
        self._val_outputs: List[Dict[str, float]] = []

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

        # Cache for geodesic distance matrices (pair_name → (N_sub, N_sub) array)
        # Precomputed in on_fit_start, reused in validation_step.
        self._geo_cache: Dict[str, Optional[np.ndarray]] = {}

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

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
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

    # ------------------------------------------------------------------
    # Baseline evaluation (runs once before first training epoch)
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Log robust-Laplacian and epoch-0 model baselines to W&B at step 0.

        Only runs on rank 0 to avoid duplicated evaluation in DDP.
        Uses the first val dataset specification.
        """
        if not self.trainer.is_global_zero:
            return

        hp     = self.hparams
        device = self.device
        dm     = self.trainer.datamodule
        if dm is None or not dm._val_dataset_specifications:
            return

        val_dataset = dm._val_dataset_specifications[0].dataset
        val_pairs   = [val_dataset[i] for i in range(len(val_dataset))]
        if not val_pairs:
            return

        def _subsample(pair):
            mv = hp.max_vertices_val if hp.max_vertices_val > 0 else hp.max_vertices
            if mv > 0:
                return subsample_pair(pair, mv,
                                      np.random.RandomState(_stable_hash(pair.name)))
            return pair

        # Precompute geodesic distances for all val pairs (once)
        print("\n  Precomputing geodesic distances on mesh B...", flush=True)
        t0_geo = time.perf_counter()
        for p in val_pairs:
            sub_p = _subsample(p)
            self._geo_cache[p.name] = _compute_geodesic_dists(sub_p)
        dt_geo = time.perf_counter() - t0_geo
        n_with_geo = sum(1 for v in self._geo_cache.values() if v is not None)
        print(f"    Done: {n_with_geo}/{len(val_pairs)} pairs with faces "
              f"({dt_geo:.1f}s)", flush=True)

        # ── Robust Laplacian baseline ──────────────────────────────────────────
        try:
            import robust_laplacian  # noqa: F401
            print(f"\n  Computing robust Laplacian baseline ({len(val_pairs)} pairs)...",
                  flush=True)
            robust_metrics = []
            n_total = len(val_pairs)
            cw = len(str(n_total))  # counter width
            for i, p in enumerate(val_pairs):
                print(f"    [robust {i+1:>{cw}}/{n_total}] {p.name}...",
                      end="", flush=True)
                t0_ = time.perf_counter()
                m = evaluate_pair_robust(_subsample(p), hp.num_eigenvectors,
                                         evaluators=self._evaluators,
                                         geo_dists=self._geo_cache.get(p.name))
                dt_ = time.perf_counter() - t0_
                robust_metrics.append(m)
                print(f" {dt_:.1f}s", flush=True)
            rb_summary = self._summarise_and_print(robust_metrics, "Robust baseline")
            self.logger.log_metrics(
                {f"baseline/robust/{k}": v for k, v in rb_summary.items()}, step=0)
        except ImportError:
            print("  (robust_laplacian not installed — skipping robust baseline)")

        # ── Epoch-0 model baseline ────────────────────────────────────────────
        init_label = "random init" if hp.random_init else "pretrained"
        print(f"\n  Computing {init_label} model baseline ({len(val_pairs)} pairs)...",
              flush=True)
        self.model.eval()
        ep0_metrics = []
        n_total = len(val_pairs)
        cw = len(str(n_total))
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
                                  geo_dists=self._geo_cache.get(p.name))
                dt_ = time.perf_counter() - t0_
                ep0_metrics.append(m)
                print(f" {dt_:.1f}s", flush=True)
        self.model.train()

        ep0_summary = self._summarise_and_print(
            ep0_metrics, f"Model baseline ({init_label})")
        self.logger.log_metrics(
            {f"baseline/model/{k}": v for k, v in ep0_summary.items()}, step=0)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _current_cross_ratio(self) -> float:
        hp       = self.hparams
        progress = min(1.0, self.current_epoch / max(1, hp.curriculum_epochs))
        return hp.cross_ratio_start + progress * (hp.cross_ratio_end - hp.cross_ratio_start)

    def on_train_epoch_start(self) -> None:
        cross_ratio = self._current_cross_ratio()
        dm = self.trainer.datamodule
        if dm is not None:
            dm._train_dataset_specification.dataset.cross_ratio = cross_ratio
        self.log("train/cross_ratio", cross_ratio, sync_dist=True)

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

            with prof.phase("loss (linear solves + InfoNCE)"):
                loss_nce, m = self.loss_fn(S_A, S_B, M_A, M_B, self._train_rng,
                                           corr_a=pair.corr_a, corr_b=pair.corr_b)

            with prof.phase("prox regularization"):
                if hp.w_prox > 0 and hasattr(self, "ref_params"):
                    cur = torch.cat([p.flatten() for p in self.model.parameters()])
                    loss_prox = ((cur - self.ref_params) ** 2).sum() / self.ref_norm_sq
                    loss = loss_nce + hp.w_prox * loss_prox
                else:
                    loss_prox = torch.tensor(0.0, device=device)
                    loss = loss_nce

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
        self.log("train/loss_nce",  avg["loss_nce"],             sync_dist=True)
        self.log("train/loss_prox", avg["loss_prox"],            sync_dist=True)
        self.log("train/top1",      avg["train_acc"],            sync_dist=True, prog_bar=True)
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
        self._val_outputs = []

    @torch.no_grad()
    def validation_step(self, batch: List[PairSample], batch_idx: int) -> None:
        assert len(batch) == 1
        pair = batch[0]
        mv = self.hparams.max_vertices_val or self.hparams.max_vertices
        if mv > 0:
            pair = subsample_pair(pair, mv,
                                  np.random.RandomState(_stable_hash(pair.name)))
        self._val_outputs.append(evaluate_pair(
            self.model, pair, self.hparams.k,
            self.hparams.num_eigenvectors, self.device,
            k_sparsify=self.hparams.k_sparsify,
            evaluators=self._evaluators,
            geo_dists=self._geo_cache.get(pair.name),
        ))

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return

        # --- DDP gather ---
        sample     = self._val_outputs[0]
        float_keys = [k for k, v in sample.items()
                      if isinstance(v, (int, float)) and not k.startswith("_")]
        local_t = torch.tensor(
            [[d.get(k, float("nan")) for k in float_keys] for d in self._val_outputs],
            device=self.device, dtype=torch.float32)

        gathered = self.all_gather(local_t)
        if gathered.dim() == 2:
            gathered = gathered.unsqueeze(0)
        all_flat = gathered.reshape(-1, len(float_keys))

        true_val_size = len(self.trainer.datamodule._val_dataset_specifications[0].dataset)
        all_flat = all_flat[:true_val_size]

        all_metrics = [{k: all_flat[i, j].item() for j, k in enumerate(float_keys)}
                       for i in range(all_flat.shape[0])]

        # --- Summarise + print (shared code path) ---
        summary = self._summarise_and_print(
            all_metrics, f"Val epoch {self.current_epoch}",
            silent=not self.trainer.is_global_zero)

        # --- Log to W&B / Lightning ---
        for mk, mv in summary.items():
            self.log(f"val/{mk}", mv, sync_dist=True)

        # Prog-bar shortcuts
        nn_prefix = "spectral_nn/"
        has_nn = any(k.startswith(nn_prefix) for k in summary)
        acc_key = f"{nn_prefix}accuracy" if has_nn else "accuracy"
        self.log("val/top1", summary.get(acc_key, 0.0), prog_bar=True, sync_dist=True)

        sp_key = f"sp_{nn_prefix}accuracy" if has_nn else "sp_accuracy"
        if sp_key in summary:
            self.log("val/sp_top1", summary[sp_key], prog_bar=True, sync_dist=True)

        primary = summary.get(sp_key, summary.get(acc_key, 0.0))
        self.log("val/best_acc", primary, prog_bar=True, sync_dist=True)