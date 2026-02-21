#!/usr/bin/env python3
"""
Fine-tune (or train from scratch) neural Laplacian for functional map correspondence.

Supports two datasets:
  --dataset smal   : SMAL parametric animal model (on-the-fly generation)
  --dataset dt4d   : DeformingThings4DMatching Humanoids (pre-loaded meshes)

Pipeline (fully differentiable, NO eigendecomposition in training):
    vertices → kNN → patches → model(θ) → S_A, S_B (dense)
    → for L landmarks at T scales: d(v) = concat[(S + αM)^{-1} M δ_l]
    → L2-normalize descriptors → cosine similarity matrix
    → InfoNCE contrastive loss (correspondence-aware)
    → ∂Loss/∂θ via torch.linalg.solve backward

Key insight: InfoNCE only cares about RANKING, not absolute descriptor values.
Unlike ||d_A - d_B||² which has an irreducible floor from genuine geometric
differences between non-isometric shapes, contrastive loss can reach zero
as long as corresponding vertices are each other's nearest neighbors.

Evaluation uses scipy eigenvectors + functional maps (non-differentiable).

Usage:
    # Fine-tune from pretrained checkpoint
    python finetune_functional_maps.py --dataset dt4d \
        --dt4d_root /path/to/DeformingThings4DMatching \
        --checkpoint model.ckpt --epochs 200 --lr 1e-4

    # Train from random initialization (uses checkpoint for architecture only)
    python finetune_functional_maps.py --dataset dt4d \
        --dt4d_root /path/to/DeformingThings4DMatching \
        --checkpoint model.ckpt --random_init --epochs 200 --lr 1e-4
"""

import argparse
import copy
import math
import sys
import types
import pickle
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
# Common pair protocol
# =============================================================================

import hashlib

def _stable_hash(obj) -> int:
    """Deterministic hash that is consistent across Python runs.

    Unlike built-in hash(), this is not affected by PYTHONHASHSEED and
    returns the same value across different processes and machines.
    """
    return int(hashlib.sha256(str(obj).encode()).hexdigest(), 16) % (2**31)


@dataclass
class PairSample:
    """A shape pair with known vertex-to-vertex correspondence.

    corr_a[i] on mesh A corresponds to corr_b[i] on mesh B,
    for i in range(len(corr_a)).

    For identity correspondence (e.g. SMAL), corr_a = corr_b = arange(N).
    For general correspondence (e.g. DT4D), they may differ.
    """
    verts_a: np.ndarray   # (N_A, 3)
    verts_b: np.ndarray   # (N_B, 3)
    corr_a: np.ndarray    # (C,) int — vertex indices into mesh A
    corr_b: np.ndarray    # (C,) int — vertex indices into mesh B
    name: str


class PairGenerator(ABC):
    """Abstract base for training/validation pair generators."""

    @abstractmethod
    def sample_train_pair(
        self, rng: np.random.RandomState, cross_ratio: float,
    ) -> PairSample:
        ...

    @abstractmethod
    def get_val_pairs(
        self, rng: np.random.RandomState,
        poses_per_pair: int = 1,
    ) -> List[PairSample]:
        ...


# =============================================================================
# Correspondence utilities
# =============================================================================

def _compute_bijective_refs(corr_a: np.ndarray, corr_b: np.ndarray) -> np.ndarray:
    """Find reference indices where the mapping is injective on both sides.

    Returns indices r such that corr_a[r] appears exactly once in corr_a
    AND corr_b[r] appears exactly once in corr_b. This ensures no two
    selected reference indices produce identical descriptors on either shape,
    which would create ambiguous contrastive targets.
    """
    # Count how many times each mesh vertex is referenced
    counts_a = np.bincount(corr_a, minlength=corr_a.max() + 1)
    counts_b = np.bincount(corr_b, minlength=corr_b.max() + 1)

    # Keep reference indices where both sides map to a unique vertex
    mask = (counts_a[corr_a] == 1) & (counts_b[corr_b] == 1)
    return np.where(mask)[0]


def identity_correspondence(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Identity correspondence: vertex i ↔ vertex i."""
    c = np.arange(n)
    return c, c


def subsample_pair(
    pair: PairSample, max_vertices: int, rng: np.random.RandomState,
) -> PairSample:
    """Subsample both shapes to at most max_vertices, remapping correspondence.

    Each shape is independently randomly subsampled. Correspondence pairs
    where either vertex was dropped are discarded. Different random subset
    each call → data augmentation effect.
    """
    verts_a, verts_b = pair.verts_a, pair.verts_b
    corr_a, corr_b = pair.corr_a, pair.corr_b
    n_a, n_b = len(verts_a), len(verts_b)

    # Subsample A
    if n_a > max_vertices:
        idx_a = np.sort(rng.choice(n_a, max_vertices, replace=False))
        verts_a = verts_a[idx_a]
        remap_a = np.full(n_a, -1, dtype=np.int64)
        remap_a[idx_a] = np.arange(max_vertices)
    else:
        remap_a = np.arange(n_a, dtype=np.int64)

    # Subsample B
    if n_b > max_vertices:
        idx_b = np.sort(rng.choice(n_b, max_vertices, replace=False))
        verts_b = verts_b[idx_b]
        remap_b = np.full(n_b, -1, dtype=np.int64)
        remap_b[idx_b] = np.arange(max_vertices)
    else:
        remap_b = np.arange(n_b, dtype=np.int64)

    # Remap correspondences — keep only surviving pairs
    new_ca = remap_a[corr_a]
    new_cb = remap_b[corr_b]
    valid = (new_ca >= 0) & (new_cb >= 0)
    new_ca = new_ca[valid]
    new_cb = new_cb[valid]

    return PairSample(verts_a, verts_b, new_ca, new_cb, pair.name)


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
            rng = np.random.RandomState(0)  # deterministic fallback

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
# SMAL pair generator
# =============================================================================

class SMALPairGenerator(PairGenerator):
    """Generate SMAL shape pairs with identity correspondence."""

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
    ) -> PairSample:
        if rng.rand() < cross_ratio and len(self.cross_pairs) > 0:
            fam_a, fam_b = self.cross_pairs[rng.randint(len(self.cross_pairs))]
            pair_type = "cross"
        else:
            fam_a, fam_b = self.same_pairs[rng.randint(len(self.same_pairs))]
            pair_type = "same"
        verts_a = self.smal.generate(fam_a, self.pose_scale, rng)
        verts_b = self.smal.generate(fam_b, self.pose_scale, rng)
        n = len(verts_a)
        corr_a, corr_b = identity_correspondence(n)
        return PairSample(verts_a, verts_b, corr_a, corr_b,
                          f"{pair_type}_{fam_a}_vs_{fam_b}")

    def get_val_pairs(self, rng: np.random.RandomState,
                      poses_per_pair: int = 1) -> List[PairSample]:
        pairs = []
        for fam_a, fam_b in self.val_pairs:
            for pose_idx in range(poses_per_pair):
                pair_rng = np.random.RandomState(fam_a * 100 + fam_b + pose_idx * 10000)
                verts_a = self.smal.generate(fam_a, self.pose_scale, pair_rng)
                verts_b = self.smal.generate(fam_b, self.pose_scale, pair_rng)
                n = len(verts_a)
                corr_a, corr_b = identity_correspondence(n)
                pairs.append(PairSample(verts_a, verts_b, corr_a, corr_b,
                                        f"val_{fam_a}_vs_{fam_b}_p{pose_idx}"))
        return pairs


# =============================================================================
# DT4D Humanoid loading
# =============================================================================

def _load_obj_vertices(path: str) -> np.ndarray:
    """Load vertices from an OBJ file. Returns (N, 3) float32."""
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.array(verts, dtype=np.float32)


def _load_vts(path: str) -> np.ndarray:
    """Load a VTS file. Returns (R,) int32 array, converted to 0-indexed."""
    vals = np.loadtxt(path, dtype=np.int32)
    return vals - 1  # VTS files are 1-indexed


class DT4DCategory:
    """One DT4D humanoid category (e.g. 'prisoner') with all poses pre-loaded."""

    def __init__(self, root: Path, name: str):
        self.name = name
        cat_dir = root / name
        corres_dir = cat_dir / "corres"

        # Discover all OBJ/VTS pairs
        obj_files = sorted(cat_dir.glob("*.obj"))
        self.poses: List[str] = []       # pose names (no extension)
        self.vertices: List[np.ndarray] = []   # (N_i, 3) per pose
        self.vts: List[np.ndarray] = []        # (R,) per pose, 0-indexed

        for obj_path in obj_files:
            pose_name = obj_path.stem
            vts_path = corres_dir / f"{pose_name}.vts"
            if not vts_path.exists():
                continue
            verts = normalize_mesh_vertices(_load_obj_vertices(str(obj_path)))
            vts = _load_vts(str(vts_path))
            self.poses.append(pose_name)
            self.vertices.append(verts)
            self.vts.append(vts)

        self.num_poses = len(self.poses)
        self.ref_size = len(self.vts[0]) if self.vts else 0

    def __repr__(self):
        return f"DT4DCategory({self.name}, {self.num_poses} poses, ref={self.ref_size})"


class DT4DPairGenerator(PairGenerator):
    """Generate DT4D Humanoid shape pairs with VTS-based correspondence.

    Same-category pairs: two poses of the same humanoid, correspondence
    via shared reference VTS (near-isometric).

    Cross-category pairs: two different humanoids, correspondence chained
    through cross_category_corres/ bridge files (non-isometric).
    """

    def __init__(
        self,
        root: str,
        train_categories: List[str],
        val_categories: List[str],
    ):
        self.root = Path(root)
        self.train_categories = train_categories
        self.val_categories = val_categories

        # Load all categories
        all_cats = sorted(set(train_categories + val_categories))
        print(f"  Loading DT4D categories: {all_cats}")
        self.categories: Dict[str, DT4DCategory] = {}
        for name in all_cats:
            cat = DT4DCategory(self.root, name)
            self.categories[name] = cat
            print(f"    {cat}")

        # Load cross-category bridges
        self.bridges: Dict[Tuple[str, str], np.ndarray] = {}
        bridge_dir = self.root / "cross_category_corres"
        if bridge_dir.exists():
            for f in bridge_dir.glob("*.vts"):
                parts = f.stem.split("_", 1)
                if len(parts) == 2:
                    src, dst = parts
                    if src in self.categories and dst in self.categories:
                        self.bridges[(src, dst)] = _load_vts(str(f))
            print(f"  Loaded {len(self.bridges)} cross-category bridges")

        # Build pair lists
        self.same_pairs = [(c, c) for c in train_categories]
        self.cross_pairs = [
            (a, b) for a, b in combinations(train_categories, 2)
            if self._can_bridge(a, b)
        ]
        self.val_pairs = [
            (a, b) for a, b in combinations(val_categories, 2)
            if self._can_bridge(a, b)
        ]
        for t in train_categories:
            for v in val_categories:
                if self._can_bridge(t, v):
                    self.val_pairs.append((t, v))

        print(f"  Same-category pairs: {len(self.same_pairs)}")
        print(f"  Cross-category pairs: {self.cross_pairs}")
        print(f"  Val pairs: {self.val_pairs}")

    def _can_bridge(self, cat_a: str, cat_b: str) -> bool:
        """Check if a cross-category bridge exists (direct or 2-hop via hub)."""
        if cat_a == cat_b:
            return True
        # Direct
        if (cat_a, cat_b) in self.bridges or (cat_b, cat_a) in self.bridges:
            return True
        # 2-hop: cat_a → hub → cat_b
        for hub in self.categories:
            if hub == cat_a or hub == cat_b:
                continue
            a_to_hub = (cat_a, hub) in self.bridges or (hub, cat_a) in self.bridges
            hub_to_b = (hub, cat_b) in self.bridges or (cat_b, hub) in self.bridges
            if a_to_hub and hub_to_b:
                return True
        return False

    def _get_bridge(self, cat_a: str, cat_b: str) -> np.ndarray:
        """Get bridge mapping: reference(cat_a) → reference(cat_b), 0-indexed.

        Supports direct bridges and 2-hop chaining through a hub category
        (typically 'crypto', since all DT4D humanoid bridges go through it).
        """
        # Direct bridge
        if (cat_a, cat_b) in self.bridges:
            return self.bridges[(cat_a, cat_b)]
        if (cat_b, cat_a) in self.bridges:
            return self._invert_bridge(self.bridges[(cat_b, cat_a)],
                                       self.categories[cat_a].ref_size)

        # 2-hop: cat_a → hub → cat_b
        for hub in self.categories:
            if hub == cat_a or hub == cat_b:
                continue
            a_to_hub = (cat_a, hub) in self.bridges or (hub, cat_a) in self.bridges
            hub_to_b = (hub, cat_b) in self.bridges or (cat_b, hub) in self.bridges
            if a_to_hub and hub_to_b:
                bridge_a_hub = self._get_bridge(cat_a, hub)   # cat_a ref → hub ref
                bridge_hub_b = self._get_bridge(hub, cat_b)   # hub ref → cat_b ref
                # Chain: for cat_a ref r, hub ref = bridge_a_hub[r],
                #        cat_b ref = bridge_hub_b[bridge_a_hub[r]]
                result = np.full(len(bridge_a_hub), -1, dtype=np.int32)
                valid = (bridge_a_hub >= 0) & (bridge_a_hub < len(bridge_hub_b))
                result[valid] = bridge_hub_b[bridge_a_hub[valid]]
                return result

        raise ValueError(f"No bridge between {cat_a} and {cat_b}")

    @staticmethod
    def _invert_bridge(fwd: np.ndarray, target_size: int) -> np.ndarray:
        """Invert a many-to-one bridge: fwd[r_src] = r_dst → inv[r_dst] = r_src."""
        inv = np.full(target_size, -1, dtype=np.int32)
        for r_src, r_dst in enumerate(fwd):
            if 0 <= r_dst < target_size:
                inv[r_dst] = r_src
        return inv

    def _build_correspondence(
        self, cat_a: str, pose_a: int, cat_b: str, pose_b: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build mesh-level correspondence arrays.

        Returns (corr_a, corr_b) where corr_a[i] on mesh A corresponds
        to corr_b[i] on mesh B. Only includes valid (in-bounds, bijective) pairs.
        """
        vts_a = self.categories[cat_a].vts[pose_a]  # ref → mesh_a
        vts_b = self.categories[cat_b].vts[pose_b]  # ref → mesh_b
        n_a = len(self.categories[cat_a].vertices[pose_a])
        n_b = len(self.categories[cat_b].vertices[pose_b])

        if cat_a == cat_b:
            # Same category: shared reference space
            # For ref index r: mesh_a[vts_a[r]] ↔ mesh_b[vts_b[r]]
            corr_a = vts_a
            corr_b = vts_b
        else:
            # Cross category: chain through bridge
            bridge = self._get_bridge(cat_a, cat_b)
            # For ref index r (in cat_a's space):
            #   mesh_a vertex = vts_a[r]
            #   cat_b ref vertex = bridge[r]
            #   mesh_b vertex = vts_b[bridge[r]]
            valid = (bridge >= 0) & (bridge < len(vts_b))
            ref_indices = np.where(valid)[0]
            corr_a = vts_a[ref_indices]
            corr_b = vts_b[bridge[ref_indices]]

        # Filter out-of-bounds (safety)
        mask = (corr_a >= 0) & (corr_a < n_a) & (corr_b >= 0) & (corr_b < n_b)
        return corr_a[mask], corr_b[mask]

    def sample_train_pair(
        self, rng: np.random.RandomState, cross_ratio: float = 0.5,
    ) -> PairSample:
        if rng.rand() < cross_ratio and len(self.cross_pairs) > 0:
            cat_a, cat_b = self.cross_pairs[rng.randint(len(self.cross_pairs))]
            pair_type = "cross"
        else:
            cat_a, cat_b = self.same_pairs[rng.randint(len(self.same_pairs))]
            pair_type = "same"

        c_a = self.categories[cat_a]
        c_b = self.categories[cat_b]
        pose_a = rng.randint(c_a.num_poses)
        pose_b = rng.randint(c_b.num_poses)

        # Avoid identical pose for same-category
        if cat_a == cat_b and c_a.num_poses > 1:
            while pose_b == pose_a:
                pose_b = rng.randint(c_b.num_poses)

        corr_a, corr_b = self._build_correspondence(cat_a, pose_a, cat_b, pose_b)

        return PairSample(
            verts_a=c_a.vertices[pose_a],
            verts_b=c_b.vertices[pose_b],
            corr_a=corr_a,
            corr_b=corr_b,
            name=f"{pair_type}_{cat_a}:{c_a.poses[pose_a]}_vs_{cat_b}:{c_b.poses[pose_b]}",
        )

    def get_val_pairs(self, rng: np.random.RandomState,
                      poses_per_pair: int = 1) -> List[PairSample]:
        pairs = []
        for cat_a, cat_b in self.val_pairs:
            # Deterministic poses per pair
            pair_rng = np.random.RandomState(_stable_hash((cat_a, cat_b)))
            c_a = self.categories[cat_a]
            c_b = self.categories[cat_b]

            # Collect distinct pose combinations
            seen = set()
            for _ in range(poses_per_pair):
                # Draw candidates until we get an unseen combo
                # (cap attempts to avoid infinite loops if pose space is small)
                for _attempt in range(200):
                    pa = pair_rng.randint(c_a.num_poses)
                    pb = pair_rng.randint(c_b.num_poses)
                    if cat_a == cat_b and pa == pb and c_a.num_poses > 1:
                        continue
                    if (pa, pb) not in seen:
                        break
                else:
                    break  # exhausted attempts, skip remaining combos
                seen.add((pa, pb))

                corr_a, corr_b = self._build_correspondence(cat_a, pa, cat_b, pb)
                pairs.append(PairSample(
                    verts_a=c_a.vertices[pa],
                    verts_b=c_b.vertices[pb],
                    corr_a=corr_a,
                    corr_b=corr_b,
                    name=f"val_{cat_a}:{c_a.poses[pa]}_vs_{cat_b}:{c_b.poses[pb]}",
                ))
        return pairs


# =============================================================================
# Differentiable dense Laplacian assembly
# =============================================================================

def compute_knn(vertices_np: np.ndarray, k: int) -> np.ndarray:
    """Compute k-nearest neighbors excluding self. Returns (N, k) indices."""
    n = len(vertices_np)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(vertices_np)
    _, indices = nbrs.kneighbors(vertices_np)  # (N, k+1)
    # Remove self from neighbor list (vectorized)
    center = np.arange(n)[:, np.newaxis]
    keep = ~(indices == center)
    keep_pos = np.cumsum(keep, axis=1)
    final = (keep_pos <= k) & keep
    return indices[final].reshape(n, k)


def build_patch_data(vertices_t, knn, device):
    """Build MeshPatchData for model input.

    Features = relative positions (neighbor - center), matching
    the format expected by LaplacianTransformerModule.forward_fixed_k.
    """
    N = vertices_t.shape[0]
    k = knn.shape[1]

    knn_t = torch.from_numpy(knn).long().to(device) if isinstance(knn, np.ndarray) else knn

    # Relative positions: differentiable w.r.t. vertices_t
    patch_pos = vertices_t[knn_t] - vertices_t[:, None, :]  # (N, k, 3)
    all_pos = patch_pos.reshape(-1, 3)  # (N*k, 3)

    batch_data = MeshPatchData(
        pos=all_pos,
        x=all_pos,  # features = relative positions
        patch_idx=torch.arange(N, device=device).repeat_interleave(k),
        vertex_indices=knn_t.flatten(),
        center_indices=torch.arange(N, device=device),
    )
    return batch_data


def assemble_dense_stiffness_and_mass(
    stiffness_weights, areas, attention_mask,
    vertex_indices, center_indices, batch_indices,
):
    """Differentiable dense S and M assembly from scalar edge weights.

    Uses only non-in-place operations so autograd tracks gradients.
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

    M_diag = torch.zeros(num_vertices, device=device, dtype=areas.dtype)
    M_diag = M_diag.scatter_add(0, center_indices, areas)
    M_count = torch.zeros(num_vertices, device=device, dtype=areas.dtype)
    M_count = M_count.scatter_add(0, center_indices, torch.ones_like(areas))
    M_count = torch.clamp(M_count, min=1.0)
    M_diag = M_diag / M_count
    M_diag = torch.clamp(M_diag, min=1e-8)

    return S, M_diag


def assemble_anisotropic_laplacian(grad_coeffs, areas, knn):
    """Differentiable L = G^T M_3 G from gradient coefficients."""
    N, k, _ = grad_coeffs.shape
    device = grad_coeffs.device

    center_coeffs = -grad_coeffs.sum(dim=1, keepdim=True)
    ext_coeffs = torch.cat([center_coeffs, grad_coeffs], dim=1)

    sqrt_a = areas.sqrt()[:, None, None]
    scaled = sqrt_a * ext_coeffs
    gram = torch.bmm(scaled, scaled.transpose(1, 2))

    center_idx = torch.arange(N, device=device).unsqueeze(1)
    ext_indices = torch.cat([center_idx, knn], dim=1)

    kp1 = k + 1
    row_idx = ext_indices[:, :, None].expand(-1, -1, kp1)
    col_idx = ext_indices[:, None, :].expand(-1, kp1, -1)
    flat_idx = (row_idx * N + col_idx).reshape(-1)

    L_flat = torch.zeros(N * N, device=device, dtype=grad_coeffs.dtype)
    L_flat = L_flat.scatter_add(0, flat_idx, gram.reshape(-1))
    L = L_flat.view(N, N)
    L = 0.5 * (L + L.T)

    return L, areas.detach()


def _sparsify_L_to_knn(L, knn_t):
    """Zero out entries of L not in the 1-hop kNN graph, fix diagonal."""
    N = L.shape[0]
    device = L.device

    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(knn_t)
    mask[row_idx, knn_t] = True
    mask = mask | mask.T

    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    keep = (mask | diag_mask).float()
    L_sp = L * keep

    off_diag = L_sp * (1.0 - diag_mask.float())
    row_sums = off_diag.sum(dim=1)
    L_sp = off_diag - torch.diag(row_sums)

    return L_sp


def compute_laplacian_differentiable(
    model: LaplacianTransformerModule,
    vertices_np: np.ndarray,
    k: int,
    device: torch.device,
    sparsify: bool = False,
):
    """Forward pass through model → dense Laplacian (differentiable)."""
    vertices_t = torch.from_numpy(vertices_np).float().to(device)
    knn = compute_knn(vertices_np, k)
    knn_t = torch.from_numpy(knn).long().to(device)

    batch_data = build_patch_data(vertices_t, knn, device)
    batch_data = Batch.from_data_list([batch_data]).to(device)

    fwd = model._forward_pass(batch_data)

    if fwd.get('grad_coeffs') is not None:
        L, M_diag = assemble_anisotropic_laplacian(
            grad_coeffs=fwd['grad_coeffs'],
            areas=fwd['areas'],
            knn=knn_t,
        )
        if sparsify:
            L = _sparsify_L_to_knn(L, knn_t)
        return L, M_diag
    else:
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
        min_gap = ctx.min_gap
        N = evals.shape[0]

        col_norms = grad_evecs.norm(dim=0)
        active_mask = col_norms > 0
        if grad_evals is not None:
            active_mask = active_mask | (grad_evals.abs() > 0)
        active_idx = torch.where(active_mask)[0]
        k_active = len(active_idx)

        if k_active == 0:
            return torch.zeros_like(evecs @ evecs.T), None

        V = evecs
        dV_active = grad_evecs[:, active_idx]
        VtdV = V.T @ dV_active  # (N, k_active)

        deval = grad_evals if grad_evals is not None else torch.zeros(N, device=evals.device)

        # Build D_active: (N, k_active)
        gaps = evals[:, None] - evals[None, active_idx]  # (N, k_active)
        gaps_clamped = gaps.sign() * gaps.abs().clamp(min=min_gap)
        F_active = VtdV / gaps_clamped  # (N, k_active)
        # Zero out diagonal-like terms (where i == active_idx[j])
        for j_local, j_global in enumerate(active_idx):
            F_active[j_global, j_local] = deval[j_global]

        D_active = F_active

        VD = V @ D_active
        V_active = V[:, active_idx]
        grad_A = VD @ V_active.T
        grad_A = 0.5 * (grad_A + grad_A.T)

        return grad_A, None


def stable_eigh(A, min_gap=1.0):
    return _StableEigh.apply(A, min_gap)


def differentiable_eigh(S, M_diag, k, min_gap=1.0):
    """Differentiable generalized eigendecomposition with stable backward."""
    M_sqrt_inv = 1.0 / M_diag.sqrt().clamp(min=1e-8)
    S_std = (S * M_sqrt_inv[None, :]) * M_sqrt_inv[:, None]
    S_std = 0.5 * (S_std + S_std.T)

    eigenvalues_all, eigenvectors_all = stable_eigh(S_std, min_gap)

    eigenvalues = eigenvalues_all[1:k+1]
    eigenvectors_std = eigenvectors_all[:, 1:k+1]
    eigenvectors = eigenvectors_std * M_sqrt_inv[:, None]

    return eigenvalues, eigenvectors


# =============================================================================
# Loss function — correspondence-aware InfoNCE
# =============================================================================

class SoftCorrespondenceLoss(torch.nn.Module):
    """InfoNCE contrastive loss via correspondence-aware diffusion fingerprints.

    Handles both identity correspondence (SMAL: same N, vertex i ↔ vertex i)
    and general correspondence (DT4D: different N, corr_a[r] ↔ corr_b[r]).

    Descriptors: diffusion fingerprints from L landmark sources at T scales.
        d(v) = concat over α: row v of (S + αM)^{-1} M E_landmarks

    Loss: InfoNCE = -mean_v log softmax(sim(v, v)) over all w

    Vertex subsampling: compute descriptors for ALL N vertices (so gradients
    flow to all parts of S), but subsample V vertices for the contrastive ranking.
    """

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
        self.num_landmarks = num_landmarks
        self.alphas = alphas
        self.temperature = temperature
        self.num_sample_vertices = num_sample_vertices
        self.landmark_seed = landmark_seed
        self.loss_type = loss_type  # "infonce", "dcl", or "dclw"
        self.dclw_sigma = dclw_sigma

    def _compute_descriptors(
        self, S: torch.Tensor, M: torch.Tensor, E: torch.Tensor,
    ) -> torch.Tensor:
        """Solve diffusion at multiple scales. Returns (N, L*T) descriptors."""
        parts = []
        for alpha in self.alphas:
            A_mat = S + alpha * torch.diag(M)
            rhs = M[:, None] * E
            D = torch.linalg.solve(A_mat, rhs)
            parts.append(D)
        desc = torch.cat(parts, dim=1)
        return F.normalize(desc, p=2, dim=1)

    def forward(
        self,
        S_A: torch.Tensor,
        S_B: torch.Tensor,
        M_A: torch.Tensor,
        M_B: torch.Tensor,
        rng: np.random.RandomState,
        corr_a: Optional[np.ndarray] = None,
        corr_b: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        N_A = S_A.shape[0]
        N_B = S_B.shape[0]
        device = S_A.device

        # --- Determine correspondence pool ---
        if corr_a is None or corr_b is None:
            # Identity correspondence (SMAL)
            assert N_A == N_B, f"Identity corr requires same N, got {N_A} vs {N_B}"
            pool = np.arange(N_A)
            ca = pool
            cb = pool
        else:
            # General correspondence — restrict to bijective reference indices
            pool = _compute_bijective_refs(corr_a, corr_b)
            if len(pool) < self.num_landmarks + self.num_sample_vertices:
                # Fallback: use all indices (allow some duplicates)
                pool = np.arange(len(corr_a))
            ca = corr_a
            cb = corr_b

        # --- Select landmarks (fixed per-shape, deterministic) ---
        lm_rng = np.random.RandomState(self.landmark_seed)
        L = min(self.num_landmarks, len(pool))
        lm_refs = pool[lm_rng.choice(len(pool), size=L, replace=False)]

        E_A = torch.zeros(N_A, L, device=device, dtype=S_A.dtype)
        E_A[ca[lm_refs], torch.arange(L, device=device)] = 1.0

        E_B = torch.zeros(N_B, L, device=device, dtype=S_B.dtype)
        E_B[cb[lm_refs], torch.arange(L, device=device)] = 1.0

        # --- Compute descriptors for ALL vertices ---
        desc_A = self._compute_descriptors(S_A, M_A, E_A)  # (N_A, L*T)
        desc_B = self._compute_descriptors(S_B, M_B, E_B)  # (N_B, L*T)

        # --- Subsample for contrastive ranking (stochastic per step) ---
        V = min(self.num_sample_vertices, len(pool))
        sample_refs = pool[rng.choice(len(pool), size=V, replace=False)]

        desc_A_sub = desc_A[ca[sample_refs]]  # (V, D)
        desc_B_sub = desc_B[cb[sample_refs]]  # (V, D)

        # --- Contrastive loss ---
        sim = (desc_A_sub @ desc_B_sub.T) / self.temperature  # (V, V)
        labels = torch.arange(V, device=device)

        if self.loss_type == "infonce":
            # Standard InfoNCE: positive is in the softmax denominator
            loss_nce = 0.5 * (F.cross_entropy(sim, labels) +
                              F.cross_entropy(sim.T, labels))

        elif self.loss_type in ("dcl", "dclw"):
            # DCL: remove positive from denominator
            # L_DCL(i) = -s_pos(i)/τ + log(Σ_{j≠i} exp(s_neg(i,j)/τ))
            pos_sim = torch.diag(sim)  # (V,) — positive similarities
            neg_mask = ~torch.eye(V, dtype=torch.bool, device=device)

            # A→B direction
            neg_A2B = sim.masked_select(neg_mask).view(V, V - 1)
            loss_A2B = -pos_sim + torch.logsumexp(neg_A2B, dim=1)

            # B→A direction
            neg_B2A = sim.T.masked_select(neg_mask).view(V, V - 1)
            loss_B2A = -pos_sim + torch.logsumexp(neg_B2A, dim=1)

            if self.loss_type == "dclw":
                # DCLW: reweight per-vertex loss by positive difficulty.
                # w(i) = 2 - V · softmax_i(cos_sim_pos / σ)
                # Easy positives (high sim) → w < 1 (downweight)
                # Hard positives (low sim)  → w > 1 (upweight)
                # pos_sim already has /τ baked in, undo it and apply /σ:
                sigma = self.dclw_sigma
                pos_for_weight = pos_sim * (self.temperature / sigma)  # cos_sim / σ
                weight = (2 - V * F.softmax(pos_for_weight, dim=0)).detach()
                loss_A2B = weight * loss_A2B
                loss_B2A = weight * loss_B2A

            loss_nce = 0.5 * (loss_A2B.mean() + loss_B2A.mean())

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        with torch.no_grad():
            pred_A2B = sim.argmax(dim=1)
            train_acc = (pred_A2B == labels).float().mean().item()
            topk_accs = {}
            for k in (3, 5, 10):
                if k <= V:
                    _, topk = sim.topk(k, dim=1)
                    topk_accs[f'train_top{k}'] = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        metrics = {
            'loss_total': loss_nce.item(),
            'loss_nce': loss_nce.item(),
            'train_acc': train_acc,
            **topk_accs,
        }
        return loss_nce, metrics


# =============================================================================
# Evaluation (non-differentiable, uses scipy)
# =============================================================================

def _fmt_topk(m: Dict[str, float], prefix: str = '') -> str:
    """Format top-1/3/5/10 accuracy + error as an aligned string.

    Works for both dense and sparse metrics dicts. ``prefix`` is prepended
    to key lookups (e.g. prefix='sp_' for sparse metrics).
    """
    acc_key = f'{prefix}accuracy'
    err_key = f'{prefix}mean_error'
    parts = [f"top1={m[acc_key]*100:5.1f}%"]
    for k in (3, 5, 10):
        key = f'{prefix}top{k}_acc'
        if key in m:
            parts.append(f"top{k}={m[key]*100:5.1f}%")
    parts.append(f"Err={m[err_key]:.4f}")
    return "  ".join(parts)

def _correspondence_metrics(
    eigvecs_a: np.ndarray,
    eigvecs_b: np.ndarray,
    mass_a: np.ndarray,
    mass_b: np.ndarray,
    vertices_b: np.ndarray,
    n: int,
    gt_corr: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute functional map correspondence metrics from eigenbases.

    Args:
        gt_corr: (n,) ground-truth correspondence array. gt_corr[i] is the
                 index in mesh B that corresponds to vertex i in mesh A.
                 If None, assumes identity (vertex i ↔ vertex i).
    """
    if gt_corr is None:
        gt_corr = np.arange(n)

    # Functional map: C = Φ_B^T M_B Π Φ_A
    # Where Π permutes rows of Φ_A to align with Φ_B's ordering
    # For identity corr: C = Φ_B^T M_B Φ_A
    # For general corr:  C = Φ_B^T M_B Φ_A[gt_corr_inverse]... no.
    #
    # Actually the standard way: C maps functions on A to functions on B.
    # C = Φ_B^T M_B Π Φ_A, where Π[i,j] = 1 iff vertex j on A corresponds
    # to vertex i on B. For identity: Π = I. For general gt_corr:
    # Π[gt_corr[j], j] = 1, so Π Φ_A has row i = Φ_A[j] where gt_corr[j]=i.
    #
    # Efficient computation: (Φ_B^T M_B)[k, i] * Π[i, j] * Φ_A[j, l]
    # = Σ_j (Φ_B[gt_corr[j], :] * M_B[gt_corr[j]])^T @ Φ_A[j, :]
    # = (weighted_phi_b_permuted)^T @ Φ_A
    # where weighted_phi_b_permuted[j, :] = Φ_B[gt_corr[j], :] * M_B[gt_corr[j]]

    n_a = eigvecs_a.shape[0]
    weighted_phi_b_permuted = eigvecs_b[gt_corr[:n_a]] * mass_b[gt_corr[:n_a], None]
    C = weighted_phi_b_permuted.T @ eigvecs_a[:n_a]

    k_fm = C.shape[0]
    I = np.eye(k_fm)

    metrics = {}
    metrics['ortho_error'] = float(np.linalg.norm(C.T @ C - I, 'fro'))
    metrics['biject_error'] = float(np.linalg.norm(C @ C.T - I, 'fro'))
    metrics['corr_error'] = float(np.linalg.norm(C - I, 'fro'))

    diag_energy = np.sum(np.diag(C) ** 2)
    total_energy = np.sum(C ** 2)
    metrics['diag_ratio'] = float(diag_energy / (total_energy + 1e-10))

    # Pointwise correspondence: project A into B's spectral space, NN search
    projected_a = eigvecs_a[:n_a] @ C.T
    from sklearn.neighbors import NearestNeighbors as NN
    nbrs = NN(n_neighbors=min(10, eigvecs_b.shape[0]), algorithm='auto').fit(eigvecs_b)
    dists, indices = nbrs.kneighbors(projected_a)
    pred_corr = indices[:, 0]

    gt = gt_corr[:n_a]
    metrics['accuracy'] = float((pred_corr == gt).mean())
    for k in (3, 5, 10):
        if k <= indices.shape[1]:
            metrics[f'top{k}_acc'] = float((indices[:, :k] == gt[:, None]).any(axis=1).mean())

    errors = np.linalg.norm(vertices_b[pred_corr] - vertices_b[gt], axis=1)
    bb_diag = np.linalg.norm(vertices_b.max(0) - vertices_b.min(0))
    metrics['mean_error'] = float(errors.mean() / bb_diag)
    metrics['median_error'] = float(np.median(errors) / bb_diag)

    return metrics


def _build_gt_corr_from_pair(pair: PairSample) -> Optional[np.ndarray]:
    """Build a dense gt_corr array: gt_corr[i] = corresponding vertex on B.

    Returns (N_A,) array where gt_corr[i] is the index in mesh B corresponding
    to vertex i in mesh A, or None for identity correspondence.
    """
    n_a = len(pair.verts_a)
    # Check identity
    if (len(pair.corr_a) == n_a and
        np.array_equal(pair.corr_a, np.arange(n_a)) and
        np.array_equal(pair.corr_b, np.arange(n_a))):
        return None  # identity

    # Build dense mapping. Multiple corr entries may map the same vertex on A.
    # We keep the last one (arbitrary but consistent).
    gt_corr = np.full(n_a, -1, dtype=np.int64)
    gt_corr[pair.corr_a] = pair.corr_b

    # For vertices with no correspondence, map to nearest correspondent
    unmapped = gt_corr == -1
    if unmapped.any():
        mapped_mask = ~unmapped
        if mapped_mask.any():
            from sklearn.neighbors import NearestNeighbors as NN
            nbrs = NN(n_neighbors=1).fit(pair.verts_a[mapped_mask])
            _, idx = nbrs.kneighbors(pair.verts_a[unmapped])
            mapped_indices = np.where(mapped_mask)[0]
            gt_corr[unmapped] = gt_corr[mapped_indices[idx.flatten()]]

    return gt_corr


def evaluate_pair_robust(
    pair: PairSample,
    num_eigenvectors: int,
    n_neighbors: int = 30,
) -> Dict[str, float]:
    """Evaluate correspondence using robust Laplacian (baseline)."""
    import robust_laplacian
    from neural_local_laplacian.utils.utils import compute_laplacian_eigendecomposition

    n_a = len(pair.verts_a)
    n_b = len(pair.verts_b)

    for label, verts in [('A', pair.verts_a), ('B', pair.verts_b)]:
        S, M = robust_laplacian.point_cloud_laplacian(verts, n_neighbors=n_neighbors)
        evals, evecs = compute_laplacian_eigendecomposition(S, num_eigenvectors, mass_matrix=M)
        M_diag = np.array(M.diagonal()).flatten()
        if label == 'A':
            eigvecs_a, mass_a = evecs, M_diag
        else:
            eigvecs_b, mass_b = evecs, M_diag

    gt_corr = _build_gt_corr_from_pair(pair)
    return _correspondence_metrics(eigvecs_a, eigvecs_b, mass_a, mass_b,
                                   pair.verts_b, n_a, gt_corr=gt_corr)


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


@torch.no_grad()
def _eigh_from_sparse_L(L_sp, M_diag_t, num_eigenvectors):
    """Sparse generalized eigenproblem: convert structurally-sparse L to scipy sparse, use eigsh.

    Much faster than _eigh_from_dense_L for large meshes since eigsh
    exploits sparsity via Lanczos iteration: O(N * nnz_per_row * num_eigenvectors).
    """
    import scipy.sparse.linalg

    N = L_sp.shape[0]
    L_np = L_sp.cpu().numpy()
    M_diag_np = M_diag_t.cpu().numpy()

    # Build scipy sparse from the dense-but-sparse tensor
    rows, cols = np.nonzero(L_np)
    vals = L_np[rows, cols]
    L_scipy = scipy.sparse.csc_matrix((vals, (rows, cols)), shape=(N, N))
    M_scipy = scipy.sparse.diags(M_diag_np)

    try:
        v0 = np.ones(N)  # deterministic starting vector for Lanczos
        evals, evecs = scipy.sparse.linalg.eigsh(
            L_scipy, k=num_eigenvectors, M=M_scipy,
            sigma=-1e-6, which='LM',  # shift-invert for smallest eigenvalues
            v0=v0,
        )
        # Sort by eigenvalue
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
    except Exception:
        # Fallback to dense if eigsh fails (e.g. singular)
        return _eigh_from_dense_L(L_sp, M_diag_t, num_eigenvectors)

    return evals, evecs


def evaluate_pair(
    model: LaplacianTransformerModule,
    pair: PairSample,
    k: int,
    num_eigenvectors: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate correspondence quality using functional maps."""
    from neural_local_laplacian.utils.utils import (
        assemble_stiffness_and_mass_matrices,
        compute_laplacian_eigendecomposition,
    )

    n_a = len(pair.verts_a)
    is_gradient_mode = False
    dense_bases = {}
    sparse_bases = {}

    for label, verts in [('A', pair.verts_a), ('B', pair.verts_b)]:
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
            evals_d, evecs_d = _eigh_from_sparse_L(L, M_diag_t, num_eigenvectors)
            M_diag = M_diag_t.cpu().numpy()
            dense_bases[label] = (evecs_d, M_diag)

            L_sp = _sparsify_L_to_knn(L, knn_t)
            evals_sp, evecs_sp = _eigh_from_sparse_L(L_sp, M_diag_t, num_eigenvectors)
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

    gt_corr = _build_gt_corr_from_pair(pair)
    evA_d, mA = dense_bases['A']
    evB_d, mB = dense_bases['B']
    metrics = _correspondence_metrics(evA_d, evB_d, mA, mB, pair.verts_b, n_a,
                                      gt_corr=gt_corr)

    if is_gradient_mode and sparse_bases:
        evA_sp, _ = sparse_bases['A']
        evB_sp, _ = sparse_bases['B']
        sp_metrics = _correspondence_metrics(evA_sp, evB_sp, mA, mB, pair.verts_b,
                                             n_a, gt_corr=gt_corr)
        for key, val in sp_metrics.items():
            metrics[f'sp_{key}'] = val

    return metrics


# =============================================================================
# Training loop
# =============================================================================

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Reproducibility: seed ALL random sources ---
    import os
    import random
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # deterministic cuBLAS
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Force deterministic CUDA ops (e.g. scatter_add used in Laplacian assembly).
    # If this errors, an op lacks a deterministic impl in your PyTorch version.
    torch.use_deterministic_algorithms(True)

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

    # Load model (always from checkpoint to get architecture/hparams)
    print(f"Loading model from: {args.checkpoint}")
    model = LaplacianTransformerModule.load_from_checkpoint(
        args.checkpoint, map_location=device,
        normalize_patch_features=True,
        scale_areas_by_patch_size=True,
    )

    # --- Detect areas head modules (by convention: module name contains 'area') ---
    def _find_areas_modules(mdl):
        """Return list of (name, module) for the areas head."""
        found = [(n, m) for n, m in mdl.named_modules() if 'area' in n.lower()]
        if not found:
            print("  WARNING: no modules with 'area' in name found. "
                  "--keep_areas_head / --freeze_areas_head will have no effect.")
        return found

    # Optionally reset all weights to random initialization
    from_scratch = getattr(args, 'random_init', False)
    if from_scratch:
        # Save areas head state before reset if requested
        areas_state = None
        if args.keep_areas_head:
            areas_modules = _find_areas_modules(model)
            if areas_modules:
                areas_state = {n: {k: v.clone() for k, v in m.state_dict().items()}
                               for n, m in areas_modules}
                print(f"  Preserving areas head weights ({len(areas_modules)} module(s)): "
                      f"{', '.join(n for n, _ in areas_modules)}")

        print("  Resetting all weights to random initialization")
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        # Restore areas head if saved
        if areas_state is not None:
            for name, m in _find_areas_modules(model):
                if name in areas_state:
                    m.load_state_dict(areas_state[name])
            print("  Restored areas head to pretrained weights")

        # Proximity loss is meaningless without pretrained reference
        if args.w_prox > 0:
            print(f"  Note: --w_prox={args.w_prox} overridden to 0.0 (no pretrained reference)")
            args.w_prox = 0.0
    model.to(device)
    model.train()

    # --- Freeze specific heads ---
    if args.freeze_input_projection:
        for name, param in model.named_parameters():
            if 'input_projection' in name:
                param.requires_grad_(False)
                print(f"  Frozen: {name}")

    if args.freeze_areas_head:
        areas_modules = _find_areas_modules(model)
        for mod_name, _ in areas_modules:
            for name, param in model.named_parameters():
                if name.startswith(mod_name):
                    param.requires_grad_(False)
                    print(f"  Frozen: {name}")

    # --- Optional: list all Linear modules and exit (for LoRA target discovery) ---
    if getattr(args, 'lora_list_modules', False):
        print("\n  All nn.Linear modules in the model:")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"    {name}: in={module.in_features}, out={module.out_features}")
        print("\n  Use --lora_target_modules with comma-separated names (or 'all-linear').")
        sys.exit(0)

    # --- Optional: LoRA adapter via PEFT ---
    use_lora = getattr(args, 'use_lora', False)
    if use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            raise ImportError(
                "LoRA requires the 'peft' package. Install with: pip install peft"
            )

        target = args.lora_target_modules
        if target == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = [m.strip() for m in target.split(',')]

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_dora=args.lora_dora,
            use_rslora=args.lora_rslora,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # LoRA already constrains updates to low-rank — proximity loss is meaningless
        if args.w_prox > 0:
            print(f"  Note: --w_prox={args.w_prox} overridden to 0.0 (LoRA constrains updates by design)")
            args.w_prox = 0.0

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,}")

    # --- Build pair generator based on dataset choice ---
    pair_gen: PairGenerator

    if args.dataset == "smal":
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

    elif args.dataset == "dt4d":
        print(f"Loading DT4D Humanoids from: {args.dt4d_root}")

        # Discover all categories (auto-detect nested directory)
        dt4d_root = Path(args.dt4d_root)
        # If root doesn't directly contain category dirs, check common subdirectory
        if not any((d / "corres").exists() for d in dt4d_root.iterdir() if d.is_dir()):
            for subdir_name in ["DeformingThings4DMatching", "humanoid", "Humanoid"]:
                candidate = dt4d_root / subdir_name
                if candidate.exists() and any(
                    (d / "corres").exists() for d in candidate.iterdir() if d.is_dir()
                ):
                    print(f"  Auto-detected data subdirectory: {candidate}")
                    dt4d_root = candidate
                    break
        all_categories = sorted([
            d.name for d in dt4d_root.iterdir()
            if d.is_dir() and (d / "corres").exists()
        ])
        if not all_categories:
            raise FileNotFoundError(
                f"No DT4D categories found under {dt4d_root}. "
                f"Expected subdirectories like crypto/, prisoner/ each containing a corres/ folder. "
                f"Check that --dt4d_root points to the directory containing the category folders."
            )
        print(f"  Available categories: {all_categories}")

        if args.val_categories:
            val_categories = [c.strip() for c in args.val_categories.split(',')]
        else:
            # Default: last 2 categories as validation
            val_categories = all_categories[-2:]
        train_categories = [c for c in all_categories if c not in val_categories]

        print(f"  Train categories: {train_categories}")
        print(f"  Val categories: {val_categories}")

        pair_gen = DT4DPairGenerator(str(dt4d_root), train_categories, val_categories)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Loss and optimizer
    loss_fn = SoftCorrespondenceLoss(
        num_landmarks=args.num_landmarks,
        alphas=tuple(float(a) for a in args.alphas.split(',')),
        temperature=args.temperature,
        num_sample_vertices=args.num_sample_vertices,
        loss_type=args.loss_type,
        dclw_sigma=args.dclw_sigma,
    )

    # Frozen copy of pretrained model for parameter-space proximity
    # (Skip when using LoRA — base weights are frozen, proximity is meaningless)
    if args.w_prox > 0 and not use_lora:
        model_ref = copy.deepcopy(model)
        model_ref.eval()
        for p in model_ref.parameters():
            p.requires_grad_(False)
        ref_params = torch.cat([p.detach().flatten() for p in model_ref.parameters()])
        ref_norm_sq = (ref_params ** 2).sum().clamp(min=1e-8)
        del model_ref
        print(f"  Reference params: {ref_params.numel():,} values, ||θ_ref||²={ref_norm_sq.item():.2e}")
    else:
        ref_params = None
        ref_norm_sq = None
        if use_lora:
            print(f"  Proximity loss: disabled (using LoRA)")
        elif from_scratch:
            print(f"  Proximity loss: disabled (no pretrained reference)")
        else:
            print(f"  Proximity loss: disabled (w_prox=0)")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine decay to eta_min over lr_T_max epochs, then stay flat (no restart).
    eta_min_ratio = 0.01  # eta_min = lr * 0.01
    def _cosine_no_restart(epoch):
        if epoch >= args.lr_T_max:
            return eta_min_ratio
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * epoch / args.lr_T_max))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_no_restart)

    # Logging
    log_file = open(output_dir / "train_log.csv", "w")
    log_file.write("epoch,step,loss,loss_nce,loss_prox,train_acc,train_top3,train_top5,train_top10,cross_ratio,grad_norm,lr\n")

    val_log = open(output_dir / "val_log.csv", "w")
    val_log.write("epoch,pair,accuracy,top3_acc,top5_acc,top10_acc,"
                  "mean_error,median_error,ortho_error,"
                  "biject_error,corr_error,diag_ratio\n")

    rng = np.random.RandomState(args.seed)
    global_step = 0

    # Compute robust Laplacian baseline (once)
    print(f"\n  Computing robust Laplacian baseline "
          f"({len(pair_gen.val_pairs)} category pairs × {args.val_poses_per_pair} poses "
          f"= {len(pair_gen.val_pairs) * args.val_poses_per_pair} evaluations)...")
    val_pairs_baseline = pair_gen.get_val_pairs(rng, args.val_poses_per_pair)
    robust_metrics_list = []
    for pair in val_pairs_baseline:
        eval_pair = pair
        if args.max_vertices > 0:
            eval_pair = subsample_pair(pair, args.max_vertices,
                                       np.random.RandomState(_stable_hash(pair.name)))
        rb = evaluate_pair_robust(eval_pair, args.num_eigenvectors)
        robust_metrics_list.append(rb)
        print(f"    {pair.name:<78s}  {_fmt_topk(rb)}")
    robust_mean_acc = np.mean([m['accuracy'] for m in robust_metrics_list])
    robust_mean_err = np.mean([m['mean_error'] for m in robust_metrics_list])
    robust_med_acc = np.median([m['accuracy'] for m in robust_metrics_list])
    _topk_summary = "  ".join(
        f"top{k}={np.mean([m[f'top{k}_acc'] for m in robust_metrics_list if f'top{k}_acc' in m])*100:5.1f}%"
        for k in (3, 5, 10)
        if any(f'top{k}_acc' in m for m in robust_metrics_list)
    )
    print(f"  Robust baseline:  top1={robust_mean_acc*100:5.1f}% (med={robust_med_acc*100:5.1f}%)  "
          f"{_topk_summary}  Err={robust_mean_err:.4f}")

    if args.wandb:
        wandb.log({"baseline/val_acc": robust_mean_acc, "baseline/val_err": robust_mean_err}, step=0)

    # --- Epoch 0 baseline (pretrained or random init, before any training) ---
    init_label = "random init" if from_scratch else "pretrained model"
    print(f"\n  Computing {init_label} baseline (epoch 0)...")
    model.eval()
    val_pairs_ep0 = pair_gen.get_val_pairs(rng, args.val_poses_per_pair)
    ep0_metrics_list = []
    for pair in val_pairs_ep0:
        eval_pair = pair
        if args.max_vertices > 0:
            eval_pair = subsample_pair(pair, args.max_vertices,
                                       np.random.RandomState(_stable_hash(pair.name)))
        val_metrics = evaluate_pair(model, eval_pair, args.k_pred, args.num_eigenvectors, device)
        ep0_metrics_list.append(val_metrics)

        sp_str = ""
        if 'sp_accuracy' in val_metrics:
            sp_str = f"  │ sp: {_fmt_topk(val_metrics, prefix='sp_')}"

        val_log.write(f"0,{pair.name},"
                      f"{val_metrics['accuracy']:.6f},"
                      f"{val_metrics.get('top3_acc', 0):.6f},"
                      f"{val_metrics.get('top5_acc', 0):.6f},"
                      f"{val_metrics.get('top10_acc', 0):.6f},"
                      f"{val_metrics['mean_error']:.6f},"
                      f"{val_metrics['median_error']:.6f},"
                      f"{val_metrics['ortho_error']:.6f},"
                      f"{val_metrics['biject_error']:.6f},"
                      f"{val_metrics['corr_error']:.6f},"
                      f"{val_metrics['diag_ratio']:.6f}\n")

        print(f"    {pair.name:<78s}  {_fmt_topk(val_metrics)}{sp_str}")

    val_log.flush()
    ep0_mean_acc = np.mean([m['accuracy'] for m in ep0_metrics_list])
    ep0_mean_err = np.mean([m['mean_error'] for m in ep0_metrics_list])
    ep0_med_acc = np.median([m['accuracy'] for m in ep0_metrics_list])
    ep0_sp_summary = ""
    if any('sp_accuracy' in m for m in ep0_metrics_list):
        sp_ms = [m for m in ep0_metrics_list if 'sp_accuracy' in m]
        ep0_mean_sp_acc = np.mean([m['sp_accuracy'] for m in sp_ms])
        ep0_mean_sp_err = np.mean([m['sp_mean_error'] for m in sp_ms])
        ep0_med_sp_acc = np.median([m['sp_accuracy'] for m in sp_ms])
        sp_topk = "  ".join(
            f"top{k}={np.mean([m[f'sp_top{k}_acc'] for m in sp_ms if f'sp_top{k}_acc' in m])*100:5.1f}%"
            for k in (3, 5, 10) if any(f'sp_top{k}_acc' in m for m in sp_ms)
        )
        ep0_sp_summary = (f"  │ sp: top1={ep0_mean_sp_acc*100:5.1f}% (med={ep0_med_sp_acc*100:5.1f}%)"
                          f"  {sp_topk}  Err={ep0_mean_sp_err:.4f}")
        best_val_acc = ep0_mean_sp_acc
    else:
        best_val_acc = ep0_mean_acc
    ep0_topk = "  ".join(
        f"top{k}={np.mean([m[f'top{k}_acc'] for m in ep0_metrics_list if f'top{k}_acc' in m])*100:5.1f}%"
        for k in (3, 5, 10) if any(f'top{k}_acc' in m for m in ep0_metrics_list)
    )
    ep0_label = "Random init:    " if from_scratch else "Pretrained base:"
    print(f"  {ep0_label}  top1={ep0_mean_acc*100:5.1f}% (med={ep0_med_acc*100:5.1f}%)"
          f"  {ep0_topk}  Err={ep0_mean_err:.4f}{ep0_sp_summary}")

    if args.wandb:
        log_dict = {"val/top1": ep0_mean_acc, "val/mean_error": ep0_mean_err}
        for k in (3, 5, 10):
            vals = [m[f'top{k}_acc'] for m in ep0_metrics_list if f'top{k}_acc' in m]
            if vals:
                log_dict[f"val/top{k}"] = np.mean(vals)
        if any('sp_accuracy' in m for m in ep0_metrics_list):
            log_dict["val/sp_top1"] = ep0_mean_sp_acc
            log_dict["val/sp_mean_error"] = ep0_mean_sp_err
        wandb.log(log_dict, step=0)

    model.train()

    print()
    print("=" * 80)
    lap_mode = "sparse (kNN-masked G^TMG)" if args.sparsify_laplacian else "dense (full G^TMG)"
    init_mode = "from scratch" if from_scratch else "fine-tuning"
    print(f"TRAINING ({args.dataset.upper()}, {init_mode}, {args.loss_type.upper()} contrastive, "
          f"{args.num_landmarks} landmarks, "
          f"V={args.num_sample_vertices}, scales={args.alphas}, τ={args.temperature}, "
          f"curriculum={args.cross_ratio_start:.0%}→{args.cross_ratio_end:.0%} over {args.curriculum_epochs}ep)")
    print(f"  Laplacian: {lap_mode}")
    if use_lora:
        variant = "DoRA" if args.lora_dora else ("rsLoRA" if args.lora_rslora else "LoRA")
        print(f"  {variant}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
              f"target={args.lora_target_modules}")
    if args.max_vertices > 0:
        print(f"  Vertex subsampling: {args.max_vertices} (per shape, per step)")
    if args.vertex_noise > 0:
        print(f"  Vertex noise: {args.vertex_noise:.4f} (relative to mean radius)")
    if args.grad_accum_steps > 1:
        eff_batch = args.pairs_per_step * args.grad_accum_steps
        print(f"  Gradient accumulation: {args.grad_accum_steps} steps "
              f"(effective batch = {args.pairs_per_step} × {args.grad_accum_steps} = {eff_batch} pairs)")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        epoch_start = time.time()

        progress = min(1.0, (epoch - 1) / max(1, args.curriculum_epochs))
        cross_ratio = args.cross_ratio_start + progress * (args.cross_ratio_end - args.cross_ratio_start)

        accum = args.grad_accum_steps
        loss_scale = args.pairs_per_step * accum

        for step in range(args.steps_per_epoch):
            global_step += 1

            # Zero gradients at the start of each accumulation window
            if step % accum == 0:
                optimizer.zero_grad()

            step_metrics_list = []
            valid_pairs = 0

            for pair_idx in range(args.pairs_per_step):
                pair = pair_gen.sample_train_pair(rng, cross_ratio)

                # Subsample large meshes for O(N³) solve tractability
                if args.max_vertices > 0:
                    pair = subsample_pair(pair, args.max_vertices, rng)

                # Vertex noise augmentation (makes each pose presentation unique)
                if args.vertex_noise > 0:
                    for attr in ('verts_a', 'verts_b'):
                        v = getattr(pair, attr)
                        scale = args.vertex_noise * np.mean(np.linalg.norm(v, axis=1))
                        setattr(pair, attr, v + rng.randn(*v.shape).astype(np.float32) * scale)

                # Forward: differentiable Laplacians
                S_A, M_A = compute_laplacian_differentiable(
                    model, pair.verts_a, args.k_pred, device,
                    sparsify=args.sparsify_laplacian)
                S_B, M_B = compute_laplacian_differentiable(
                    model, pair.verts_b, args.k_pred, device,
                    sparsify=args.sparsify_laplacian)

                # InfoNCE contrastive loss (correspondence-aware)
                loss_nce, metrics = loss_fn(
                    S_A, S_B, M_A, M_B, rng,
                    corr_a=pair.corr_a, corr_b=pair.corr_b)

                # Parameter-space proximity (skipped when using LoRA or w_prox=0)
                if args.w_prox > 0 and ref_params is not None:
                    cur_params = torch.cat([p.flatten() for p in model.parameters()])
                    loss_prox = ((cur_params - ref_params) ** 2).sum() / ref_norm_sq
                    loss = loss_nce + args.w_prox * loss_prox
                else:
                    loss_prox = torch.tensor(0.0, device=device)
                    loss = loss_nce

                metrics['loss_prox'] = loss_prox.item()
                metrics['loss_total'] = loss.item()

                if torch.isnan(loss):
                    print(f"  [Step {global_step}, pair {pair_idx}] NaN loss! Skipping pair.")
                    continue

                if global_step == 1 and pair_idx == 0:
                    S_A.retain_grad()
                    S_B.retain_grad()

                (loss / loss_scale).backward()

                step_metrics_list.append(metrics)
                valid_pairs += 1

            if valid_pairs == 0:
                print(f"  [Step {global_step}] All pairs failed! Skipping.")
                # If at accumulation boundary, clear stale grads
                if (step + 1) % accum == 0 or step == args.steps_per_epoch - 1:
                    optimizer.zero_grad()
                continue

            # Clip and step at end of accumulation window (or end of epoch)
            is_accum_boundary = ((step + 1) % accum == 0) or (step == args.steps_per_epoch - 1)
            if is_accum_boundary:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip if args.grad_clip > 0 else float('inf'))
                optimizer.step()
            else:
                grad_norm = torch.tensor(0.0)  # not stepping yet

            avg_step = {k: np.mean([m[k] for m in step_metrics_list])
                        for k in step_metrics_list[0]}
            avg_step['grad_norm'] = grad_norm.item()
            epoch_losses.append(avg_step)

            lr = optimizer.param_groups[0]['lr']
            log_file.write(f"{epoch},{global_step},{avg_step['loss_total']:.6e},"
                           f"{avg_step['loss_nce']:.6e},{avg_step['loss_prox']:.6e},"
                           f"{avg_step['train_acc']:.4f},"
                           f"{avg_step.get('train_top3', 0):.4f},"
                           f"{avg_step.get('train_top5', 0):.4f},"
                           f"{avg_step.get('train_top10', 0):.4f},"
                           f"{cross_ratio:.4f},"
                           f"{avg_step['grad_norm']:.6e},{lr:.2e}\n")

        scheduler.step()

        # Epoch summary
        if epoch_losses:
            avg = {k: np.mean([m[k] for m in epoch_losses]) for k in epoch_losses[0]}
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch:4d} │ loss={avg['loss_total']:8.4f} │ "
                  f"nce={avg['loss_nce']:8.4f} │ prox={avg['loss_prox']:.4f} │ "
                  f"top1={avg['train_acc']*100:5.1f}% │ "
                  f"top3={avg.get('train_top3', 0)*100:5.1f}% │ "
                  f"top5={avg.get('train_top5', 0)*100:5.1f}% │ "
                  f"top10={avg.get('train_top10', 0)*100:5.1f}% │ "
                  f"cross={cross_ratio:4.0%} │ "
                  f"grad={avg.get('grad_norm', 0):9.2e} │ "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} │ "
                  f"time={elapsed:5.1f}s")

            if args.wandb:
                wandb.log({
                    "train/loss": avg['loss_total'],
                    "train/loss_nce": avg['loss_nce'],
                    "train/loss_prox": avg['loss_prox'],
                    "train/top1": avg['train_acc'],
                    "train/top3": avg.get('train_top3', 0),
                    "train/top5": avg.get('train_top5', 0),
                    "train/top10": avg.get('train_top10', 0),
                    "train/grad_norm": avg.get('grad_norm', 0),
                    "train/cross_ratio": cross_ratio,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/epoch_time": elapsed,
                }, step=epoch)
        else:
            print(f"Epoch {epoch:4d} │ all steps failed")

        # Validation
        if epoch % args.eval_every == 0:
            model.eval()
            val_pairs = pair_gen.get_val_pairs(rng, args.val_poses_per_pair)
            val_metrics_list = []

            print(f"  --- Validation (epoch {epoch}) ---")
            for pair in val_pairs:
                eval_pair = pair
                if args.max_vertices > 0:
                    eval_pair = subsample_pair(pair, args.max_vertices,
                                               np.random.RandomState(_stable_hash(pair.name)))
                val_metrics = evaluate_pair(
                    model, eval_pair, args.k_pred, args.num_eigenvectors, device)
                val_metrics_list.append(val_metrics)

                val_log.write(f"{epoch},{pair.name},"
                              f"{val_metrics['accuracy']:.6f},"
                              f"{val_metrics.get('top3_acc', 0):.6f},"
                              f"{val_metrics.get('top5_acc', 0):.6f},"
                              f"{val_metrics.get('top10_acc', 0):.6f},"
                              f"{val_metrics['mean_error']:.6f},"
                              f"{val_metrics['median_error']:.6f},"
                              f"{val_metrics['ortho_error']:.6f},"
                              f"{val_metrics['biject_error']:.6f},"
                              f"{val_metrics['corr_error']:.6f},"
                              f"{val_metrics['diag_ratio']:.6f}\n")

                sp_str = ""
                if 'sp_accuracy' in val_metrics:
                    sp_str = f"  │ sp: {_fmt_topk(val_metrics, prefix='sp_')}"

                print(f"    {pair.name:<78s}  {_fmt_topk(val_metrics)}{sp_str}")

            val_log.flush()
            mean_acc = np.mean([m['accuracy'] for m in val_metrics_list])
            mean_err = np.mean([m['mean_error'] for m in val_metrics_list])
            med_acc = np.median([m['accuracy'] for m in val_metrics_list])
            val_topk = "  ".join(
                f"top{k}={np.mean([m[f'top{k}_acc'] for m in val_metrics_list if f'top{k}_acc' in m])*100:5.1f}%"
                for k in (3, 5, 10) if any(f'top{k}_acc' in m for m in val_metrics_list)
            )
            sp_summary = ""
            val_sp_accs = [m['sp_accuracy'] for m in val_metrics_list if 'sp_accuracy' in m]
            if val_sp_accs:
                sp_ms = [m for m in val_metrics_list if 'sp_accuracy' in m]
                mean_sp_acc = np.mean(val_sp_accs)
                mean_sp_err = np.mean([m['sp_mean_error'] for m in sp_ms])
                med_sp_acc = np.median(val_sp_accs)
                sp_topk = "  ".join(
                    f"top{k}={np.mean([m[f'sp_top{k}_acc'] for m in sp_ms if f'sp_top{k}_acc' in m])*100:5.1f}%"
                    for k in (3, 5, 10) if any(f'sp_top{k}_acc' in m for m in sp_ms)
                )
                sp_summary = (f"  │ sp: top1={mean_sp_acc*100:5.1f}% (med={med_sp_acc*100:5.1f}%)"
                              f"  {sp_topk}  Err={mean_sp_err:.4f}")
            print(f"  Summary:          top1={mean_acc*100:5.1f}% (med={med_acc*100:5.1f}%)"
                  f"  {val_topk}  Err={mean_err:.4f}{sp_summary}")

            if args.wandb:
                log_dict = {
                    "val/top1": mean_acc,
                    "val/median_top1": med_acc,
                    "val/mean_error": mean_err,
                }
                for k in (3, 5, 10):
                    vals = [m[f'top{k}_acc'] for m in val_metrics_list if f'top{k}_acc' in m]
                    if vals:
                        log_dict[f"val/top{k}"] = np.mean(vals)
                if val_sp_accs:
                    log_dict["val/sp_top1"] = mean_sp_acc
                    log_dict["val/sp_median_top1"] = med_sp_acc
                    log_dict["val/sp_mean_error"] = mean_sp_err
                    for k in (3, 5, 10):
                        sp_vals = [m[f'sp_top{k}_acc'] for m in sp_ms if f'sp_top{k}_acc' in m]
                        if sp_vals:
                            log_dict[f"val/sp_top{k}"] = np.mean(sp_vals)
                wandb.log(log_dict, step=epoch)

            # Use sparse acc for best-model when sparsified training
            track_acc = mean_sp_acc if val_sp_accs else mean_acc
            if track_acc > best_val_acc:
                best_val_acc = track_acc
                if use_lora:
                    model.save_pretrained(output_dir / "best_lora_adapter")
                else:
                    torch.save(model.state_dict(), output_dir / "best_model.ckpt")
                print(f"  ★ New best: {best_val_acc*100:5.1f}%")

            model.train()

        # Periodic checkpoint
        if hasattr(args, 'save_every') and epoch % args.save_every == 0:
            if use_lora:
                model.save_pretrained(output_dir / "last_lora_adapter")
            else:
                torch.save(model.state_dict(), output_dir / "last_model.ckpt")

    # Final save
    if use_lora:
        model.save_pretrained(output_dir / "last_lora_adapter")
        # Also save merged model for convenient standalone loading
        print("  Merging LoRA adapter into base model for standalone checkpoint...")
        merged_model = model.merge_and_unload()
        torch.save(merged_model.state_dict(), output_dir / "last_model_merged.ckpt")
        # Re-merge best adapter too
        best_adapter_dir = output_dir / "best_lora_adapter"
        if best_adapter_dir.exists():
            from peft import PeftModel
            # Reload base model fresh, apply best adapter, merge
            base_model = LaplacianTransformerModule.load_from_checkpoint(
                args.checkpoint, map_location=device,
                normalize_patch_features=True,
                scale_areas_by_patch_size=True,
            )
            best_peft = PeftModel.from_pretrained(base_model, str(best_adapter_dir))
            best_merged = best_peft.merge_and_unload()
            torch.save(best_merged.state_dict(), output_dir / "best_model_merged.ckpt")
            print(f"  Saved merged best model to: {output_dir / 'best_model_merged.ckpt'}")
            del base_model, best_peft, best_merged
    else:
        torch.save(model.state_dict(), output_dir / "last_model.ckpt")
    log_file.close()
    val_log.close()

    print(f"\nDone! Best val accuracy: {best_val_acc*100:5.1f}%")
    print(f"Results in: {output_dir}")

    if args.wandb:
        artifact = wandb.Artifact(
            name=f"best-model-{wandb.run.id}",
            type="model",
            description=f"Best finetuned model (val_acc={best_val_acc*100:.1f}%)"
                        + (" [LoRA]" if use_lora else ""),
        )
        if use_lora:
            best_merged = output_dir / "best_model_merged.ckpt"
            if best_merged.exists():
                artifact.add_file(str(best_merged))
            best_adapter = output_dir / "best_lora_adapter"
            if best_adapter.exists():
                artifact.add_dir(str(best_adapter), name="lora_adapter")
        else:
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

    # Dataset choice
    parser.add_argument("--dataset", type=str, default="smal",
                        choices=["smal", "dt4d"],
                        help="Dataset to use for training")

    # SMAL-specific
    parser.add_argument("--smal_model", type=str, default=None,
                        help="Path to smal_CVPR2017.pkl (required for --dataset smal)")
    parser.add_argument("--smal_data", type=str, default=None,
                        help="Path to smal_CVPR2017_data.pkl (required for --dataset smal)")
    parser.add_argument("--val_families", type=str, default=None,
                        help="[SMAL] Comma-separated family indices for validation")

    # DT4D-specific
    parser.add_argument("--dt4d_root", type=str, default=None,
                        help="Path to DeformingThings4DMatching root (required for --dataset dt4d)")
    parser.add_argument("--val_categories", type=str, default=None,
                        help="[DT4D] Comma-separated category names for validation")
    parser.add_argument("--val_poses_per_pair", type=int, default=2,
                        help="Number of pose combinations to evaluate per val category pair "
                             "(default: 1). E.g. 5 → 13 pairs × 5 = 65 val evaluations.")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained model checkpoint (also used for architecture when --random_init)")
    parser.add_argument("--random_init", action="store_true",
                        help="Reset all weights to random initialization after loading checkpoint architecture")
    parser.add_argument("--keep_areas_head", action="store_true",
                        help="[with --random_init] Preserve pretrained areas head weights during reset")
    parser.add_argument("--k_pred", type=int, default=20)
    parser.add_argument("--num_eigenvectors", type=int, default=100)
    parser.add_argument("--freeze_input_projection", action="store_true")
    parser.add_argument("--freeze_areas_head", action="store_true",
                        help="Freeze the areas head (keep pretrained weights, no gradient updates)")

    # LoRA (Low-Rank Adaptation)
    parser.add_argument("--use_lora", action="store_true",
                        help="Fine-tune with LoRA adapters (requires: pip install peft)")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank (lower = fewer params, more constrained)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (scaling factor, typically 2x rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Dropout applied to LoRA layers")
    parser.add_argument("--lora_dora", action="store_true",
                        help="Use DoRA (Weight-Decomposed LoRA) — decomposes into magnitude + direction")
    parser.add_argument("--lora_rslora", action="store_true",
                        help="Use rsLoRA (Rank-Stabilized) — scales by 1/√r, better for higher ranks")
    parser.add_argument("--lora_target_modules", type=str, default="all-linear",
                        help="Comma-separated module names to apply LoRA, or 'all-linear'")
    parser.add_argument("--lora_list_modules", action="store_true",
                        help="Print all nn.Linear module names and exit (for target discovery)")

    # Training
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--steps_per_epoch", type=int, default=10)
    parser.add_argument("--pairs_per_step", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Accumulate gradients over this many steps before optimizer.step(). "
                             "Effective batch = pairs_per_step * grad_accum_steps.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_T_max", type=int, default=200,
                        help="T_max for CosineAnnealingLR (default: 200)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--max_vertices", type=int, default=0,
                        help="Subsample shapes to this many vertices during training "
                             "(0 = no subsampling). Useful for large meshes (e.g. DT4D ~8000).")
    parser.add_argument("--vertex_noise", type=float, default=0.05,
                        help="Gaussian noise scale relative to mean vertex radius "
                             "(e.g. 0.005 = 0.5%%). Augments training poses for diversity.")
    parser.add_argument("--pose_scale", type=float, default=0.3,
                        help="[SMAL] Pose variation scale")
    parser.add_argument("--seed", type=int, default=42)

    # InfoNCE loss
    parser.add_argument("--num_landmarks", type=int, default=512)
    parser.add_argument("--alphas", type=str, default="1.0,10.0,100.0")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_sample_vertices", type=int, default=1024)
    parser.add_argument("--loss_type", type=str, default="infonce",
                        choices=["infonce", "dcl", "dclw"],
                        help="Contrastive loss: infonce, dcl (decoupled), dclw (weighted decoupled)")
    parser.add_argument("--dclw_sigma", type=float, default=0.5,
                        help="[DCLW] Sigma for positive reweighting")
    parser.add_argument("--w_prox", type=float, default=20.0)
    parser.add_argument("--sparsify_laplacian", action="store_true")

    # Curriculum
    parser.add_argument("--cross_ratio_start", type=float, default=0.0)
    parser.add_argument("--cross_ratio_end", type=float, default=0.5)
    parser.add_argument("--curriculum_epochs", type=int, default=50)

    # Evaluation & checkpointing
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="fmap_finetune_runs")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="neural-laplacian-finetune")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)

    args = parser.parse_args()

    # Validate dataset-specific args
    if args.dataset == "smal":
        if not args.smal_model or not args.smal_data:
            parser.error("--dataset smal requires --smal_model and --smal_data")
    elif args.dataset == "dt4d":
        if not args.dt4d_root:
            parser.error("--dataset dt4d requires --dt4d_root")

    if args.keep_areas_head and not args.random_init:
        parser.error("--keep_areas_head only makes sense with --random_init")

    train(args)


if __name__ == "__main__":
    main()