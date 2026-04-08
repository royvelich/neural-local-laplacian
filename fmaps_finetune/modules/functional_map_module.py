"""
LightningModule for functional map fine-tuning of the neural Laplacian.

All Laplacian assembly, loss, and evaluation utilities live here alongside
FunctionalMapModule so this file is self-contained on the model side.
"""
from __future__ import annotations
import os
import contextlib
import time
from pathlib import Path

import scipy.sparse
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
    compute_laplacian_eigendecomposition,
)
from neural_local_laplacian.utils.laplacian_assembly import (
    LaplacianConfig,
    assemble_laplacian,
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


def _select_landmarks_and_samples(
    N_A: int, N_B: int,
    corr_a: Optional[np.ndarray],
    corr_b: Optional[np.ndarray],
    rng: np.random.RandomState,
    num_landmarks: int,
    num_sample_vertices: int,
    landmark_seed: int,
    landmark_ratio: Optional[float] = None,
    loss_name: str = "",
    _log_state: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared landmark and sample vertex selection for supervised losses.

    Two modes:
        landmark_ratio (not None): Split bijective pool into disjoint
            landmarks (ratio) and samples (1 - ratio).
        Fixed-count (landmark_ratio=None): Use num_landmarks and
            num_sample_vertices with fallback to all correspondences
            if bijective pool is too small.

    Args:
        N_A, N_B: Number of vertices on shapes A and B.
        corr_a, corr_b: Ground-truth correspondence arrays.
        rng: Per-step RNG (used for sample vertex selection in fixed-count mode).
        num_landmarks: Fixed landmark count (ignored if landmark_ratio is set).
        num_sample_vertices: Fixed sample count (ignored if landmark_ratio is set).
        landmark_seed: Deterministic seed for landmark selection.
        landmark_ratio: If set, fraction of bijective pool to use as landmarks.
        loss_name: Name for diagnostic prints (e.g. "GeodesicDiffusionLoss").
        _log_state: Mutable [count, max] list for silencing prints.
            Pass a list stored on self so state persists across calls.

    Returns:
        (lm_refs, sample_refs, ca, cb) where:
            lm_refs: indices into the reference pool for landmarks
            sample_refs: indices into the reference pool for samples
            ca, cb: correspondence arrays (identity if no correspondences)
    """
    if corr_a is None or corr_b is None:
        assert N_A == N_B
        bijective_pool = np.arange(N_A)
        ca = bijective_pool; cb = bijective_pool
    else:
        bijective_pool = _compute_bijective_refs(corr_a, corr_b)
        ca, cb = corr_a, corr_b

    bijective_n = len(bijective_pool)

    if landmark_ratio is not None:
        # Ratio mode: split bijective pool into disjoint landmarks + samples
        lm_rng = np.random.RandomState(landmark_seed)
        shuffled = bijective_pool[lm_rng.permutation(bijective_n)]
        L = max(1, int(bijective_n * landmark_ratio))
        lm_refs = shuffled[:L]
        sample_refs = shuffled[L:]
        V = len(sample_refs)

        if _log_state is not None and _log_state[0] < _log_state[1]:
            _log_state[0] += 1
            print(f"[{loss_name}] pair {_log_state[0]}: "
                  f"bijective={bijective_n}, ratio={landmark_ratio}, "
                  f"L={L}, V={V} (disjoint)",
                  flush=True)
            if _log_state[0] == _log_state[1]:
                print(f"[{loss_name}] (silencing after {_log_state[1]} pairs)",
                      flush=True)
    else:
        # Fixed-count mode: fallback to all corr if bijective pool too small
        if bijective_n < num_landmarks + num_sample_vertices:
            pool = np.arange(len(corr_a)) if corr_a is not None else bijective_pool
        else:
            pool = bijective_pool

        lm_rng = np.random.RandomState(landmark_seed)
        L = min(num_landmarks, len(pool))
        lm_refs = pool[lm_rng.choice(len(pool), size=L, replace=False)]
        V = min(num_sample_vertices, len(pool))
        sample_refs = pool[rng.choice(len(pool), size=V, replace=False)]

        if _log_state is not None and _log_state[0] < _log_state[1]:
            _log_state[0] += 1
            used_bij = len(pool) == bijective_n
            print(f"[{loss_name}] pair {_log_state[0]}: "
                  f"pool={len(pool)} ({'bijective' if used_bij else f'all corr, bij={bijective_n}'}), "
                  f"L={L}/{num_landmarks}, "
                  f"V={V}/{num_sample_vertices}",
                  flush=True)
            if _log_state[0] == _log_state[1]:
                print(f"[{loss_name}] (silencing after {_log_state[1]} pairs)",
                      flush=True)

    return lm_refs, sample_refs, ca, cb


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

        lm_refs, sample_refs, ca, cb = _select_landmarks_and_samples(
            N_A, N_B, corr_a, corr_b, rng,
            self.num_landmarks, self.num_sample_vertices,
            self.landmark_seed)
        L = len(lm_refs)

        E_A = torch.zeros(N_A, L, device=device, dtype=S_A.dtype)
        E_A[ca[lm_refs], torch.arange(L, device=device)] = 1.0
        E_B = torch.zeros(N_B, L, device=device, dtype=S_B.dtype)
        E_B[cb[lm_refs], torch.arange(L, device=device)] = 1.0

        desc_A = self._compute_descriptors(S_A, M_A, E_A, self.alphas)
        desc_B = self._compute_descriptors(S_B, M_B, E_B, self.alphas)

        V           = len(sample_refs)
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
        landmark_ratio: Optional[float] = None,
        landmark_seed: int = 0,
        normalize: bool = True,
        weight: float = 1.0,
    ):
        """
        Args:
            num_landmarks: Fixed number of landmarks (ignored if landmark_ratio is set).
            num_sample_vertices: Fixed number of sample vertices (ignored if landmark_ratio is set).
            landmark_ratio: If set (0 < ratio < 1), use ONLY bijective correspondences
                and split them into landmarks (ratio) and samples (1-ratio) with no
                overlap. E.g. 0.2 = 20% landmarks, 80% samples.
                When None, uses num_landmarks/num_sample_vertices with fallback.
            alphas: Diffusion time scales.
            landmark_seed: Seed for deterministic landmark selection.
            normalize: L2-normalize diffusion profiles before comparing.
            weight: Loss weight.
        """
        super().__init__()
        self.num_landmarks       = num_landmarks
        self.alphas              = alphas
        self.num_sample_vertices = num_sample_vertices
        self.landmark_ratio      = landmark_ratio
        self.landmark_seed       = landmark_seed
        self.normalize           = normalize
        self.weight              = weight
        self._log_state          = [0, 10]  # [count, max] for diagnostic prints

        if landmark_ratio is not None:
            assert 0.0 < landmark_ratio < 1.0, \
                f"landmark_ratio must be in (0, 1), got {landmark_ratio}"

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

        lm_refs, sample_refs, ca, cb = _select_landmarks_and_samples(
            N_A, N_B, corr_a, corr_b, rng,
            self.num_landmarks, self.num_sample_vertices,
            self.landmark_seed, self.landmark_ratio,
            loss_name="GeodesicDiffusionLoss",
            _log_state=self._log_state)
        L = len(lm_refs)

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
        landmark_ratio: Optional[float] = None,
        landmark_seed: int = 0,
        eps: float = 1e-4,
        normalize: bool = True,
        weight: float = 1.0,
    ):
        """
        Args:
            num_landmarks: Fixed number of landmarks (ignored if landmark_ratio is set).
            num_sample_vertices: Fixed number of sample vertices (ignored if landmark_ratio is set).
            landmark_ratio: If set (0 < ratio < 1), use ONLY bijective correspondences
                and split them into landmarks (ratio) and samples (1-ratio) with no
                overlap. E.g. 0.2 = 20% landmarks, 80% samples.
                When None, uses num_landmarks/num_sample_vertices with fallback.
            landmark_seed: Seed for deterministic landmark selection.
            eps: regularisation for Poisson solve.
            normalize: L2-normalize geodesic profiles before comparing.
            weight: Loss weight.
        """
        super().__init__()
        self.num_landmarks       = num_landmarks
        self.num_sample_vertices = num_sample_vertices
        self.landmark_ratio      = landmark_ratio
        self.landmark_seed       = landmark_seed
        self.eps                 = eps
        self.normalize           = normalize
        self.weight              = weight
        self._log_state          = [0, 10]  # [count, max] for diagnostic prints

        if landmark_ratio is not None:
            assert 0.0 < landmark_ratio < 1.0, \
                f"landmark_ratio must be in (0, 1), got {landmark_ratio}"

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

        lm_refs, sample_refs, ca, cb = _select_landmarks_and_samples(
            N_A, N_B, corr_a, corr_b, rng,
            self.num_landmarks, self.num_sample_vertices,
            self.landmark_seed, self.landmark_ratio,
            loss_name="HeatMethodGeodesicLoss",
            _log_state=self._log_state)
        L = len(lm_refs)

        # Source indices on each shape
        src_A = torch.from_numpy(ca[lm_refs].copy()).long().to(device)
        src_B = torch.from_numpy(cb[lm_refs].copy()).long().to(device)

        # Full heat method on both shapes
        try:
            phi_A = self._heat_method_distances(
                S_A, M_A, grad_coeffs_a, knn_a, src_A, self.eps)   # (N_A, L)
            phi_B = self._heat_method_distances(
                S_B, M_B, grad_coeffs_b, knn_b, src_B, self.eps)   # (N_B, L)
        except torch.linalg.LinAlgError:
            # Singular matrix — skip this pair (early training / degenerate mesh)
            zero = torch.tensor(0.0, device=device)
            return zero, {'loss_heatgeo': 0.0}

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
    weighted_phi_b = eigvecs_b[gt_corr[:n_a]] * mass_a[:n_a, None] \
        if gt_corr is not None else eigvecs_b[:n_a] * mass_a[:n_a, None]
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

    def save_to_disk(self, path: str) -> None:
        """Save cache to disk as .npz file."""
        save_dict = {
            'sqrt_area': np.array(self._sqrt_area),
            'idx_b': self._idx_b,
        }
        for k, v in self._dist_cache.items():
            save_dict[f'dist_{k}'] = v
        np.savez_compressed(path, **save_dict)

    @classmethod
    def load_from_disk(cls, path: str) -> 'GeodesicCache':
        """Load cache from .npz file."""
        data = np.load(path)
        sqrt_area = float(data['sqrt_area'])
        idx_b = data['idx_b']
        dist_cache = {}
        for k in data.files:
            if k.startswith('dist_'):
                target_idx = int(k[5:])
                dist_cache[target_idx] = data[k]
        return cls.from_precomputed(dist_cache, sqrt_area, idx_b)

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


def _geo_cache_path(cache_dir: str, pair_name: str, max_vertices: int) -> Path:
    """Build the disk path for a geo cache file."""
    mv_label = f"mv{max_vertices}" if max_vertices > 0 else "mv0"
    safe_name = pair_name.replace(":", "_").replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / mv_label / f"{safe_name}.npz"


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
def evaluate_pair(
    model: LaplacianTransformerModule,
    pair: PairSample,
    k: int,
    num_eigenvectors: int,
    device: torch.device,
    laplacian_configs: Optional[List[LaplacianConfig]] = None,
    evaluators: Optional[List] = None,
    geo_cache = None,
) -> Dict[str, float]:
    """Evaluate correspondence quality using functional maps (non-differentiable).

    Args:
        laplacian_configs: List of LaplacianConfig to evaluate. Each config
            produces a separate eigenbasis and metric set prefixed by config.tag.
            Defaults to [LaplacianConfig(assembly='diagonal_gram')].
        evaluators: List of ShapePairEvaluator instances.
        geo_cache: Precomputed GeodesicCache for mesh B.
    """
    if laplacian_configs is None:
        laplacian_configs = [LaplacianConfig(assembly='diagonal_gram')]

    n_a = len(pair.verts_a)

    # bases_by_config[config_tag][label] = (evecs, evals, M_np)
    bases_by_config: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    M_np_shared = None  # mass is the same for all configs (areas don't change)

    for label, verts in [('A', pair.verts_a), ('B', pair.verts_b)]:
        verts_t    = torch.from_numpy(verts).float().to(device)
        knn_np     = compute_knn(verts, k)
        batch_data = Batch.from_data_list(
            [build_patch_data(verts_t, knn_np, device)]).to(device)
        fwd        = model(batch_data)

        knn_t     = torch.from_numpy(knn_np).long().to(device)
        M_diag    = fwd['areas'].detach()
        M_np      = M_diag.cpu().numpy()
        if M_np_shared is None:
            M_np_shared = {}
        M_np_shared[label] = M_np

        grad_coeffs = fwd['grad_coeffs']

        for cfg in laplacian_configs:
            tag = cfg.tag

            # Force sparse assembly for eval (no gradients needed).
            # Pruning is only supported in dense mode, so fall back for pruned configs.
            use_sparse = cfg.pruning == 'none'
            eval_cfg = LaplacianConfig(
                assembly=cfg.assembly,
                pruning=cfg.pruning,
                k_prune=cfg.k_prune,
                sparse=use_sparse,
                area_weighted=cfg.area_weighted,
            )

            # Compute knn_prune if needed
            knn_prune = None
            if cfg.pruning == 'knn' and cfg.k_prune is not None and cfg.k_prune != k:
                knn_prune_np = compute_knn(verts, cfg.k_prune)
                knn_prune = torch.from_numpy(knn_prune_np).long().to(device)

            L_assembled = assemble_laplacian(grad_coeffs, knn_t, eval_cfg, areas=M_diag,
                                             knn_prune=knn_prune)

            # Convert to scipy sparse if dense (pruned configs)
            if use_sparse:
                L_scipy = L_assembled
            else:
                L_np = L_assembled.detach().cpu().numpy()
                L_scipy = scipy.sparse.csr_matrix(L_np)

            M_scipy = scipy.sparse.diags(M_np)
            evals, evecs = compute_laplacian_eigendecomposition(
                L_scipy, num_eigenvectors, mass_matrix=M_scipy)

            if tag not in bases_by_config:
                bases_by_config[tag] = {}
            bases_by_config[tag][label] = (evecs, evals, M_np)

    gt_corr = _build_gt_corr_from_pair(pair)
    if geo_cache is None:
        geo_cache = _build_geo_cache(pair)

    mA = M_np_shared['A']
    mB = M_np_shared['B']

    # --- Use evaluators if provided ---
    if evaluators is not None:
        metrics = {}
        for tag, bases in bases_by_config.items():
            evA, evalsA, _ = bases['A']
            evB, evalsB, _ = bases['B']
            # First config gets unprefixed metrics, rest get tag prefix
            prefix = f"{tag}_" if len(bases_by_config) > 1 else ""
            for evaluator in evaluators:
                ev_metrics = evaluator.evaluate(
                    evA, evB, evalsA, evalsB,
                    mA, mB, pair.verts_b, gt_corr=gt_corr,
                    geo_cache=geo_cache,
                )
                for mk, mv in ev_metrics.items():
                    metrics[f"{prefix}{evaluator.name}/{mk}"] = mv
        return metrics

    # --- Legacy path (backward compatible) ---
    first_tag = list(bases_by_config.keys())[0]
    bases = bases_by_config[first_tag]
    evA, evalsA, _ = bases['A']
    evB, evalsB, _ = bases['B']
    metrics = _correspondence_metrics(evA, evB, mA, mB,
                                      pair.verts_b, n_a, gt_corr=gt_corr)
    return metrics


@torch.no_grad()
def evaluate_pair_robust(
    pair: PairSample,
    num_eigenvectors: int,
    n_neighbors: int = 30,
    evaluators: Optional[List] = None,
    geo_cache = None,
    area_weighted: bool = False,
) -> Dict[str, float]:
    """Evaluate using robust Laplacian (baseline, no model).

    Args:
        evaluators: List of ShapePairEvaluator instances. When provided,
            runs each evaluator and prefixes metrics with evaluator.name.
            When None, uses the legacy _correspondence_metrics path.
        geo_cache: Precomputed GeodesicCache for mesh B.
            If None and pair has faces_b, builds on the fly.
        area_weighted: If True, multiply S by diag(areas) to get
            graph-Laplacian-like behaviour (areas in both S and M).
    """
    import robust_laplacian
    n_a = len(pair.verts_a)
    bases: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, verts in [('A', pair.verts_a), ('B', pair.verts_b)]:
        S, M     = robust_laplacian.point_cloud_laplacian(verts, n_neighbors=n_neighbors)
        if area_weighted:
            M_diag = np.array(M.diagonal()).flatten()
            S = scipy.sparse.diags(M_diag) @ S
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
        num_eigenvectors: int = 100,
        train_laplacian: Optional[Dict] = None,
        eval_laplacians: Optional[List[Dict]] = None,
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
        geo_cache_workers: Optional[int] = None,  # max mp.Pool workers per rank (None = auto)
        geo_cache_dir: Optional[str] = None,     # disk cache directory (None = in-memory only)
        eval_results_dir: Optional[str] = None,  # eval output directory (None = {default_root_dir}/eval_results)
        mode: str = 'train',                     # 'train', 'eval', or 'geo_cache'
        baselines: Optional[List[str]] = None,   # ['robust', 'model'] (default), or subset
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

        # Validate mode
        if mode not in ('train', 'eval', 'geo_cache'):
            raise ValueError(f"mode must be 'train', 'eval', or 'geo_cache', got: {mode!r}")
        if mode == 'geo_cache' and not geo_cache_dir:
            raise ValueError("mode='geo_cache' requires geo_cache_dir to be set")

        # Build loss list: prefer explicit `losses`, fall back to `loss_fn`
        if losses is not None:
            self._losses = nn.ModuleList(losses)
        elif loss_fn is not None:
            if not hasattr(loss_fn, 'weight'):
                loss_fn.weight = 1.0
            self._losses = nn.ModuleList([loss_fn])
        elif mode != 'train':
            self._losses = nn.ModuleList()
        else:
            raise ValueError("Must provide either `losses` or `loss_fn`")

        self.model: Optional[nn.Module] = None
        self._train_rng: Optional[np.random.RandomState] = None
        self._val_outputs: Dict[int, List[Dict[str, float]]] = {}

        # Build LaplacianConfig objects from dict parameters
        _train_lap = train_laplacian or {'assembly': 'full_gram', 'pruning': 'none'}
        self._train_lap_config = LaplacianConfig(**_train_lap)

        _eval_laps = eval_laplacians or [{'assembly': 'diagonal_gram', 'pruning': 'none'}]
        self._eval_lap_configs = [LaplacianConfig(**d) for d in _eval_laps]

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

        # DDP fix: reseed training dataset so each rank generates different pairs.
        # ShapePairDataset.__getitem__ ignores idx and uses self._rng, so all ranks
        # would produce identical pairs without per-rank seeding.
        if self.trainer.world_size > 1:
            dm = self.trainer.datamodule
            train_ds = getattr(dm, '_train_dataset_specification', None)
            if train_ds is not None:
                train_ds = getattr(train_ds, 'dataset', None)
            if train_ds is not None and hasattr(train_ds, '_rng'):
                new_seed = hp.seed + self.global_rank * 10_007
                train_ds._rng = np.random.RandomState(new_seed)
                if self.global_rank == 0:
                    print(f"  [FMModule] DDP: reseeded training dataset "
                          f"(world_size={self.trainer.world_size})")

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

        # --- Print summary ---
        if not silent:
            # Auto-discover variant prefixes from metric keys.
            # Metrics are keyed as "{prefix}{evaluator.name}/{metric}".
            # Prefix is "" for single config, or "{tag}_" for multiple configs.
            # We discover all prefixes by looking for "*/accuracy" keys.
            rows = []
            for ev in self._evaluators:
                acc_suffix = f"{ev.name}/accuracy"
                for key in sorted(all_keys):
                    if key.endswith(acc_suffix) and summary.get(key) is not None:
                        prefix = key[:-len(acc_suffix)]
                        display = f"{prefix}{ev.name}" if prefix else ev.name
                        rows.append((display, prefix, ev))

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

        # Build a unique run directory name
        hp = self.hparams
        try:
            import wandb
            run_name = wandb.run.name if wandb.run is not None else "local"
        except ImportError:
            run_name = "local"
        run_tag = f"{run_name}_k{hp.k}"
        if self._train_lap_config.pruning != 'none':
            run_tag += f"_{self._train_lap_config.tag}"

        base_dir = Path(hp.eval_results_dir) if hp.eval_results_dir else \
            Path(self.trainer.default_root_dir) / "eval_results"
        out_dir = base_dir / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Saving eval results to {out_dir}/", flush=True)

        # Save run config
        config_path = out_dir / "config.yaml"
        try:
            import yaml
            config_dict = {
                "k": hp.k,
                "train_laplacian": hp.train_laplacian,
                "eval_laplacians": hp.eval_laplacians,
                "num_eigenvectors": hp.num_eigenvectors,
                "checkpoint_path": hp.checkpoint_path,
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
    # Geodesic cache helpers
    # ------------------------------------------------------------------

    def _effective_max_vertices(self) -> int:
        """Return the effective max_vertices for subsampling."""
        hp = self.hparams
        mv = hp.max_vertices_val if hp.max_vertices_val > 0 else hp.max_vertices
        return mv

    def _get_geo_cache(self, pair_name: str) -> Optional[GeodesicCache]:
        """Get geo cache for a pair: from memory, or lazy-load from disk.

        In-memory mode (geo_cache_dir=None): looks up self._geo_cache dict.
        Disk mode (geo_cache_dir set): loads .npz from disk on demand.
        Returns None if cache is not available.
        """
        # In-memory mode: direct dict lookup
        if not self.hparams.geo_cache_dir:
            return self._geo_cache.get(pair_name)

        # Disk mode: load on demand (not cached in memory)
        cache_path = _geo_cache_path(
            self.hparams.geo_cache_dir, pair_name,
            self._effective_max_vertices())
        if cache_path.exists():
            return GeodesicCache.load_from_disk(str(cache_path))
        return None

    # ------------------------------------------------------------------
    # Baseline evaluation (runs once before first training epoch)
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Compute baselines and precompute geo caches.

        All ranks participate in geo cache precomputation and model baseline
        (distributed across GPUs). Robust baseline runs on rank 0 only
        (CPU-parallelised with ThreadPoolExecutor).
        """
        hp     = self.hparams
        device = self.device
        dm     = self.trainer.datamodule
        is_rank0 = self.trainer.is_global_zero
        world_size = self.trainer.world_size

        if dm is None or not dm._val_dataset_specifications:
            return

        def _subsample(pair):
            mv = hp.max_vertices_val if hp.max_vertices_val > 0 else hp.max_vertices
            if mv > 0:
                return subsample_pair(pair, mv,
                                      np.random.RandomState(_stable_hash(pair.name)))
            return pair

        # Collect all val pairs across all datasets (all ranks need this)
        all_val_datasets: List[Tuple[str, List[PairSample]]] = []
        for spec in dm._val_dataset_specifications:
            ds = spec.dataset
            ds_name = getattr(ds, 'name', ds.__class__.__name__)
            pairs = [ds[i] for i in range(len(ds))]
            all_val_datasets.append((ds_name, pairs))

        # ==================================================================
        # Geodesic caches
        # ==================================================================
        all_pairs = [(name, p) for name, pairs in all_val_datasets for p in pairs]
        mv = self._effective_max_vertices()
        _n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)

        if hp.mode == 'geo_cache' and hp.geo_cache_dir:
            # ── Mode: geo_cache — compute all caches and save to disk ────
            cache_dir = Path(hp.geo_cache_dir)
            mv_dir = cache_dir / (f"mv{mv}" if mv > 0 else "mv0")
            mv_dir.mkdir(parents=True, exist_ok=True)

            # Filter pairs that need computation (skip existing)
            to_compute = []
            for _, p in all_pairs:
                cache_path = _geo_cache_path(hp.geo_cache_dir, p.name, mv)
                if cache_path.exists():
                    continue
                sub_p = _subsample(p)
                if sub_p.faces_b is None:
                    continue
                gt_corr = _build_gt_corr_from_pair(sub_p)
                verts_full = sub_p._verts_b_full if sub_p._verts_b_full is not None else sub_p.verts_b
                idx_b = sub_p._idx_b if sub_p._idx_b is not None else np.arange(len(sub_p.verts_b))
                unique_targets = np.unique(gt_corr) if gt_corr is not None else np.arange(len(sub_p.verts_b))
                to_compute.append((
                    p.name, verts_full, sub_p.faces_b, idx_b, unique_targets,
                ))

            if is_rank0:
                print(f"\n  [geo_cache] {len(to_compute)} pairs to compute "
                      f"({len(all_pairs) - len(to_compute)} already cached), "
                      f"saving to {mv_dir}", flush=True)

            # Distribute across ranks
            my_args = to_compute[self.global_rank::world_size]
            t0_geo = time.perf_counter()
            _max_per_rank = max(1, _n_cpus // max(world_size, 1))
            if hp.geo_cache_workers is not None:
                _max_per_rank = min(_max_per_rank, hp.geo_cache_workers)
            n_workers = min(len(my_args), _max_per_rank)
            n_my = len(my_args)
            cw = len(str(n_my))

            if my_args:
                if n_workers > 1:
                    import multiprocessing as _mp
                    print(f"    [rank {self.global_rank}] Computing {n_my} pairs "
                          f"with {n_workers} workers...", flush=True)
                    with _mp.Pool(n_workers) as pool:
                        for i, r in enumerate(pool.imap_unordered(
                                _precompute_geo_cache_worker, my_args)):
                            name_r, dist_cache, sqrt_area, idx_b = r
                            if dist_cache is not None:
                                gc_obj = GeodesicCache.from_precomputed(
                                    dist_cache, sqrt_area, idx_b)
                                gc_obj.save_to_disk(
                                    str(_geo_cache_path(hp.geo_cache_dir, name_r, mv)))
                            done = i + 1
                            if self.global_rank == 0:
                                dt = time.perf_counter() - t0_geo
                                eta = (dt / done) * (n_my - done)
                                print(f"    [{done:>{cw}}/{n_my}] "
                                      f"{dt:.1f}s elapsed, ~{eta:.0f}s remaining",
                                      flush=True)
                else:
                    print(f"    [rank {self.global_rank}] Computing {n_my} pairs "
                          f"sequentially...", flush=True)
                    for i, args in enumerate(my_args):
                        r = _precompute_geo_cache_worker(args)
                        name_r, dist_cache, sqrt_area, idx_b = r
                        if dist_cache is not None:
                            gc_obj = GeodesicCache.from_precomputed(
                                dist_cache, sqrt_area, idx_b)
                            gc_obj.save_to_disk(
                                str(_geo_cache_path(hp.geo_cache_dir, name_r, mv)))
                        done = i + 1
                        if self.global_rank == 0:
                            dt = time.perf_counter() - t0_geo
                            eta = (dt / done) * (n_my - done)
                            print(f"    [{done:>{cw}}/{n_my}] "
                                  f"{dt:.1f}s elapsed, ~{eta:.0f}s remaining",
                                  flush=True)

            # Synchronize all ranks, then exit
            if world_size > 1:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
                    dist.destroy_process_group()

            dt_geo = time.perf_counter() - t0_geo
            if is_rank0:
                n_files = sum(1 for _ in mv_dir.glob("*.npz"))
                print(f"\n  [geo_cache] Done: {n_files} cache files "
                      f"in {mv_dir} ({dt_geo:.1f}s)", flush=True)

            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass
            raise SystemExit(0)

        elif hp.geo_cache_dir:
            # ── Mode: disk cache — lazy loading, no precomputation ─────
            cache_dir = Path(hp.geo_cache_dir)
            n_found = sum(1 for _, p in all_pairs
                          if _geo_cache_path(hp.geo_cache_dir, p.name, mv).exists())
            if is_rank0:
                print(f"\n  Using disk geo cache: {n_found}/{len(all_pairs)} pairs "
                      f"found in {cache_dir}", flush=True)
                if n_found < len(all_pairs):
                    print(f"    WARNING: {len(all_pairs) - n_found} pairs missing — "
                          f"run with mode=geo_cache first", flush=True)

        else:
            # ── Mode: in-memory precomputation (no disk cache) ─────────
            if is_rank0:
                print(f"\n  Precomputing geodesic caches ({len(all_pairs)} pairs)...",
                      flush=True)
            t0_geo = time.perf_counter()

            worker_args = []
            pair_meta = []
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
                unique_targets = np.unique(gt_corr) if gt_corr is not None else np.arange(len(sub_p.verts_b))
                worker_args.append((
                    p.name, verts_full, sub_p.faces_b, idx_b, unique_targets,
                ))
                pair_meta.append((p.name, True))

            _max_per_rank = max(1, _n_cpus // max(world_size, 1))
            if hp.geo_cache_workers is not None:
                _max_per_rank = min(_max_per_rank, hp.geo_cache_workers)
            n_workers = min(len(worker_args), _max_per_rank)
            _geo_n = len(worker_args)
            _geo_cw = len(str(_geo_n))
            if worker_args:
                if n_workers > 1:
                    import multiprocessing as _mp
                    if is_rank0:
                        print(f"    Parallel precomputation: {_geo_n} pairs, "
                              f"{n_workers} workers", flush=True)
                    with _mp.Pool(n_workers) as pool:
                        results = {}
                        for i, r in enumerate(pool.imap_unordered(
                                _precompute_geo_cache_worker, worker_args)):
                            results[r[0]] = r[1:]
                            done = i + 1
                            if is_rank0:
                                dt = time.perf_counter() - t0_geo
                                eta = (dt / done) * (_geo_n - done)
                                print(f"    [{done:>{_geo_cw}}/{_geo_n}] "
                                      f"{dt:.1f}s elapsed, ~{eta:.0f}s remaining",
                                      flush=True)
                else:
                    results = {}
                    for i, args in enumerate(worker_args):
                        r = _precompute_geo_cache_worker(args)
                        results[r[0]] = r[1:]
                        done = i + 1
                        if is_rank0:
                            dt = time.perf_counter() - t0_geo
                            eta = (dt / done) * (_geo_n - done)
                            print(f"    [{done:>{_geo_cw}}/{_geo_n}] "
                                  f"{dt:.1f}s elapsed, ~{eta:.0f}s remaining",
                                  flush=True)
            else:
                results = {}

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
            if is_rank0:
                print(f"    Done: {n_with_geo}/{len(all_pairs)} pairs with faces "
                      f"({dt_geo:.1f}s)", flush=True)

        # ==================================================================
        # Baselines
        # ==================================================================
        eval_results: Dict[str, Dict[str, Tuple[List[Dict], Dict]]] = {}
        baselines_to_run = hp.baselines if hp.baselines is not None else ['robust', 'model']

        if not baselines_to_run:
            if is_rank0:
                print("\n  [baselines=[]] Skipping baseline computation.",
                      flush=True)
        else:
            if is_rank0:
                print(f"\n  Baselines to run: {baselines_to_run}", flush=True)

            for ds_name, val_pairs in all_val_datasets:
                if not val_pairs:
                    continue
                eval_results[ds_name] = {}
                n_total = len(val_pairs)
                ds_label = f" [{ds_name}]" if len(all_val_datasets) > 1 else ""

                # ── Robust Laplacian baseline (all ranks, distributed) ────
                robust_variants = []
                if 'robust' in baselines_to_run:
                    robust_variants.append(('robust', False))
                if 'robust_aw' in baselines_to_run:
                    robust_variants.append(('robust_aw', True))

                for rb_name, rb_area_weighted in robust_variants:
                    try:
                        import robust_laplacian  # noqa: F401
                        if is_rank0:
                            print(f"\n  Computing {rb_name} Laplacian baseline{ds_label} "
                                  f"({n_total} pairs, {world_size} rank(s))...", flush=True)

                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        rb_n_workers = max(1, _n_cpus // max(world_size, 1))
                        if hp.geo_cache_workers is not None:
                            rb_n_workers = min(rb_n_workers, hp.geo_cache_workers)

                        def _eval_robust(p, _aw=rb_area_weighted):
                            return evaluate_pair_robust(
                                _subsample(p), hp.num_eigenvectors,
                                evaluators=self._evaluators,
                                geo_cache=self._get_geo_cache(p.name),
                                area_weighted=_aw)

                        # Distribute pairs across ranks (same pattern as model baseline)
                        my_indices = list(range(self.global_rank, n_total, world_size))
                        my_pairs = [val_pairs[i] for i in my_indices]
                        t0_rb = time.perf_counter()

                        my_robust_metrics = [None] * len(my_pairs)
                        with ThreadPoolExecutor(max_workers=rb_n_workers) as executor:
                            futures = {executor.submit(_eval_robust, p): j
                                       for j, p in enumerate(my_pairs)}
                            done_count = 0
                            for future in as_completed(futures):
                                j = futures[future]
                                my_robust_metrics[j] = future.result()
                                done_count += 1
                                if is_rank0:
                                    dt = time.perf_counter() - t0_rb
                                    total_done = done_count * world_size
                                    n_my = len(my_pairs)
                                    eta = (dt / done_count) * (n_my - done_count)
                                    cw = len(str(n_total))
                                    print(f"    [{rb_name} ~{min(total_done, n_total):>{cw}}/{n_total}] "
                                          f"{my_pairs[j].name}... "
                                          f"{dt:.1f}s elapsed, ~{eta:.0f}s remaining",
                                          flush=True)

                        # Gather from all ranks
                        if world_size > 1:
                            max_per_rank = (n_total + world_size - 1) // world_size
                            if my_robust_metrics and my_robust_metrics[0] is not None:
                                float_keys = sorted(k for k, v in my_robust_metrics[0].items()
                                                    if isinstance(v, (int, float)))
                            else:
                                float_keys = []

                            n_keys_t = torch.tensor(
                                [len(float_keys)], device=device, dtype=torch.long)
                            import torch.distributed as dist
                            if dist.is_initialized():
                                dist.broadcast(n_keys_t, src=0)
                            n_keys = n_keys_t.item()

                            if n_keys > 0:
                                local_t = torch.full(
                                    (max_per_rank, n_keys),
                                    float("nan"), device=device, dtype=torch.float32)
                                for j, m in enumerate(my_robust_metrics):
                                    if m is not None:
                                        for ki, k in enumerate(float_keys):
                                            local_t[j, ki] = m.get(k, float("nan"))

                                gathered = self.all_gather(local_t)
                                if gathered.dim() == 2:
                                    gathered = gathered.unsqueeze(0)

                                if is_rank0:
                                    robust_metrics = [{}] * n_total
                                    for rank in range(world_size):
                                        rank_indices = list(range(rank, n_total, world_size))
                                        for j, orig_idx in enumerate(rank_indices):
                                            row = gathered[rank, j]
                                            if not torch.isnan(row).all():
                                                robust_metrics[orig_idx] = {
                                                    k: row[ki].item()
                                                    for ki, k in enumerate(float_keys)
                                                }
                            else:
                                robust_metrics = []
                        else:
                            robust_metrics = my_robust_metrics

                        if is_rank0:
                            dt_rb = time.perf_counter() - t0_rb
                            print(f"    Done: {n_total} pairs in {dt_rb:.1f}s "
                                  f"({rb_n_workers} threads/rank × {world_size} ranks)",
                                  flush=True)

                            rb_summary = self._summarise_and_print(
                                robust_metrics, f"{rb_name} baseline{ds_label}")
                            self.logger.log_metrics(
                                {f"baseline/{rb_name}/{ds_name}/{k}": v
                                 for k, v in rb_summary.items()}, step=0)
                            eval_results[ds_name][rb_name] = (robust_metrics, rb_summary)
                    except ImportError:
                        if is_rank0:
                            print("  (robust_laplacian not installed — "
                                  f"skipping {rb_name} baseline)")

                    # Synchronise ranks after robust baseline
                    if world_size > 1:
                        import torch.distributed as dist
                        if dist.is_initialized():
                            dist.barrier()

                # ── Epoch-0 model baseline (all ranks, distributed) ─────────
                if 'model' in baselines_to_run:
                    init_label = "random init" if hp.random_init else "pretrained"
                    if is_rank0:
                        print(f"\n  Computing {init_label} model baseline{ds_label} "
                              f"({n_total} pairs, {world_size} GPU(s))...", flush=True)

                    self.model.eval()
                    # Distribute pairs across ranks
                    my_indices = list(range(self.global_rank, n_total, world_size))
                    my_metrics = []
                    t0_ep0 = time.perf_counter()

                    with torch.no_grad():
                        for count, i in enumerate(my_indices, 1):
                            t0_pair = time.perf_counter()
                            sub_p = _subsample(val_pairs[i])
                            m = evaluate_pair(self.model, sub_p,
                                              hp.k, hp.num_eigenvectors, device,
                                              laplacian_configs=self._eval_lap_configs,
                                              evaluators=self._evaluators,
                                              geo_cache=self._get_geo_cache(val_pairs[i].name))
                            my_metrics.append(m)
                            n_my = len(my_indices)
                            if is_rank0:
                                dt_pair = time.perf_counter() - t0_pair
                                dt = time.perf_counter() - t0_ep0
                                total_done = count * world_size
                                eta = (dt / count) * (n_my - count)
                                cw = len(str(n_total))
                                print(f"    [model {min(total_done, n_total):>{cw}}/{n_total}] "
                                      f"{val_pairs[i].name}... {dt_pair:.1f}s",
                                      flush=True)
                    self.model.train()

                    # Gather metrics from all ranks
                    if world_size > 1:
                        # All ranks must participate in all_gather even with 0 local pairs.
                        # Determine float_keys from any rank that has metrics.
                        max_per_rank = (n_total + world_size - 1) // world_size
                        if my_metrics:
                            sample = my_metrics[0]
                            float_keys = sorted(k for k, v in sample.items()
                                                if isinstance(v, (int, float)))
                        else:
                            float_keys = []

                        # Broadcast float_keys length from rank 0 so all ranks agree
                        n_keys_t = torch.tensor(
                            [len(float_keys)], device=device, dtype=torch.long)
                        import torch.distributed as dist
                        if dist.is_initialized():
                            dist.broadcast(n_keys_t, src=0)
                        n_keys = n_keys_t.item()

                        if n_keys > 0:
                            local_t = torch.full(
                                (max_per_rank, n_keys),
                                float("nan"), device=device, dtype=torch.float32)
                            for j, m in enumerate(my_metrics):
                                for ki, k in enumerate(float_keys):
                                    val = m.get(k, float("nan"))
                                    local_t[j, ki] = val

                            gathered = self.all_gather(local_t)
                            if gathered.dim() == 2:
                                gathered = gathered.unsqueeze(0)

                            if is_rank0:
                                # Reconstruct original pair order from gathered results.
                                # Rank r processed pairs[r::world_size].
                                ep0_metrics = [{}] * n_total
                                for rank in range(world_size):
                                    rank_indices = list(range(rank, n_total, world_size))
                                    for j, orig_idx in enumerate(rank_indices):
                                        row = gathered[rank, j]
                                        if not torch.isnan(row).all():
                                            ep0_metrics[orig_idx] = {
                                                k: row[ki].item()
                                                for ki, k in enumerate(float_keys)
                                            }
                        else:
                            ep0_metrics = []
                    elif not my_metrics:
                        ep0_metrics = []
                    else:
                        ep0_metrics = my_metrics

                    if is_rank0:
                        dt_ep0 = time.perf_counter() - t0_ep0
                        print(f"    {n_total} pairs in {dt_ep0:.1f}s", flush=True)
                        ep0_summary = self._summarise_and_print(
                            ep0_metrics, f"Model baseline ({init_label}){ds_label}")
                        self.logger.log_metrics(
                            {f"baseline/model/{ds_name}/{k}": v
                             for k, v in ep0_summary.items()}, step=0)
                        eval_results[ds_name][f"model_{init_label}"] = (
                            ep0_metrics, ep0_summary)

                    # Synchronise ranks before moving to next dataset
                    if world_size > 1:
                        import torch.distributed as dist
                        if dist.is_initialized():
                            dist.barrier()

        # Store baseline summaries for TrainingOutputCallback access
        self._baseline_summaries = {}
        for ds_name, methods in eval_results.items():
            for method_name, (_, summary) in methods.items():
                for k, v in summary.items():
                    self._baseline_summaries[f"baseline/{method_name}/{ds_name}/{k}"] = v

        # ── Eval mode: save CSVs and plots, then stop ─────────────────
        if hp.mode == 'eval':
            if eval_results and is_rank0:
                self._export_eval_results(eval_results, all_val_datasets)
            if is_rank0:
                print("\n  [mode=eval] Baselines complete — stopping.",
                      flush=True)
            # Synchronize all DDP ranks before exit to prevent zombies
            if world_size > 1:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
                    dist.destroy_process_group()
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
                fwd_a = self.model(bd_a)
                fwd_b = self.model(bd_b)

            with prof.phase("laplacian assembly"):
                lap_cfg = self._train_lap_config
                knn_sp_a_t, knn_sp_b_t = None, None
                if lap_cfg.pruning == 'knn' and lap_cfg.k_prune is not None and lap_cfg.k_prune != hp.k:
                    knn_sp_a = compute_knn(pair.verts_a, lap_cfg.k_prune)
                    knn_sp_b = compute_knn(pair.verts_b, lap_cfg.k_prune)
                    knn_sp_a_t = torch.from_numpy(knn_sp_a).long().to(device)
                    knn_sp_b_t = torch.from_numpy(knn_sp_b).long().to(device)
                S_A = assemble_laplacian(fwd_a['grad_coeffs'], knn_a_t, lap_cfg,
                                        areas=fwd_a['areas'], knn_prune=knn_sp_a_t)
                M_A = fwd_a['areas']
                S_B = assemble_laplacian(fwd_b['grad_coeffs'], knn_b_t, lap_cfg,
                                        areas=fwd_b['areas'], knn_prune=knn_sp_b_t)
                M_B = fwd_b['areas']

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
        self._val_t0 = time.perf_counter()

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
            laplacian_configs=self._eval_lap_configs,
            evaluators=self._evaluators,
            geo_cache=self._get_geo_cache(pair.name),
        ))

        # Progress print (rank 0)
        if self.trainer.is_global_zero:
            done = batch_idx + 1
            try:
                total = self.trainer.num_val_batches[dataloader_idx]
            except (IndexError, TypeError):
                total = None
            dt = time.perf_counter() - self._val_t0
            if total:
                eta = (dt / done) * (total - done)
                print(f"    [val {done:>{len(str(total))}}/{total}] "
                      f"{dt:.1f}s elapsed, ~{eta:.0f}s remaining", flush=True)
            else:
                print(f"    [val {done}] {dt:.1f}s elapsed", flush=True)

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
            prefix = f"val/{ds_name}"
            for mk, mv in summary.items():
                self.log(f"{prefix}/{mk}", mv, sync_dist=True,
                         add_dataloader_idx=False)

            # Primary metrics from first dataset
            if dl_idx == 0:
                # Find unprefixed (first config) accuracy for top1
                # Auto-discover: first evaluator with unprefixed accuracy key
                acc_key = None
                for ev in self._evaluators:
                    k = f"{ev.name}/accuracy"
                    if k in summary:
                        acc_key = k
                        break
                if acc_key:
                    self.log("val/top1", summary[acc_key],
                             prog_bar=True, sync_dist=True, add_dataloader_idx=False)

                # Find secondary variant accuracy (any prefixed evaluator accuracy)
                secondary_acc_key = None
                for ev in self._evaluators:
                    suffix = f"{ev.name}/accuracy"
                    for k in sorted(summary.keys()):
                        if k.endswith(suffix) and k != suffix and summary.get(k) is not None:
                            secondary_acc_key = k
                            break
                    if secondary_acc_key:
                        break
                if secondary_acc_key:
                    self.log("val/secondary_top1", summary[secondary_acc_key],
                             prog_bar=True, sync_dist=True, add_dataloader_idx=False)

                # Primary for checkpointing: prefer fmap evaluator geo@5%
                # from any variant, fall back to spectral_nn geo@5%, then accuracy
                primary_acc = None

                # Try fmap evaluators geo@5% (prefer non-default / prefixed variant)
                for ev in self._evaluators:
                    if ev.name.startswith("fmap_"):
                        suffix = f"{ev.name}/geo_at_05pct"
                        # Prefer prefixed (non-default) variant
                        for k in sorted(summary.keys()):
                            if k.endswith(suffix) and k != suffix:
                                primary_acc = summary[k]
                                break
                        if primary_acc is None:
                            # Fall back to unprefixed
                            if suffix in summary:
                                primary_acc = summary[suffix]
                        if primary_acc is not None:
                            break

                # Fall back to any spectral_nn geo@5%
                if primary_acc is None:
                    for k in sorted(summary.keys()):
                        if k.endswith("spectral_nn/geo_at_05pct"):
                            primary_acc = summary[k]
                            break

                # Fall back to secondary accuracy, then default accuracy
                if primary_acc is None:
                    if secondary_acc_key:
                        primary_acc = summary[secondary_acc_key]
                    elif acc_key:
                        primary_acc = summary.get(acc_key, 0.0)

        if primary_acc is not None:
            self.log("val/best_acc", primary_acc,
                     prog_bar=True, sync_dist=True, add_dataloader_idx=False)