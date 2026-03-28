"""
Data primitives for functional map fine-tuning.

Pair generators (SMAL, DT4D) and torch Datasets live here.
Nothing in this file depends on PyTorch model code or the Lightning module.
"""
from __future__ import annotations

import hashlib
import pickle
import sys
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse
from torch.utils.data import Dataset

from neural_local_laplacian.utils.utils import normalize_mesh_vertices


# =============================================================================
# Stable hash (deterministic across Python runs / processes)
# =============================================================================

def _stable_hash(obj) -> int:
    return int(hashlib.sha256(str(obj).encode()).hexdigest(), 16) % (2 ** 31)


# =============================================================================
# Pair protocol
# =============================================================================

@dataclass
class PairSample:
    """A shape pair with known vertex-to-vertex correspondence.

    corr_a[i] on mesh A corresponds to corr_b[i] on mesh B.
    For identity correspondence (SMAL) corr_a = corr_b = arange(N).
    """
    verts_a: np.ndarray   # (N_A, 3)
    verts_b: np.ndarray   # (N_B, 3)
    corr_a:  np.ndarray   # (C,) int
    corr_b:  np.ndarray   # (C,) int
    name:    str

    # Optional: mesh faces for geodesic error computation.
    # Set by the dataset loader when OBJ files provide faces.
    faces_b: Optional[np.ndarray] = field(default=None, repr=False)  # (F, 3) int

    # Set by subsample_pair when subsampling B:
    _verts_b_full: Optional[np.ndarray] = field(default=None, repr=False)  # (N_B_orig, 3)
    _idx_b: Optional[np.ndarray] = field(default=None, repr=False)         # (N_B_sub,) int


class PairGenerator(ABC):
    @abstractmethod
    def sample_train_pair(
        self, rng: np.random.RandomState, cross_ratio: float,
    ) -> PairSample: ...

    @abstractmethod
    def get_val_pairs(
        self, rng: np.random.RandomState, poses_per_pair: int = 1,
    ) -> List[PairSample]: ...


# =============================================================================
# Correspondence utilities
# =============================================================================

def _compute_bijective_refs(corr_a: np.ndarray, corr_b: np.ndarray) -> np.ndarray:
    """Indices where the mapping is injective on both sides."""
    counts_a = np.bincount(corr_a, minlength=corr_a.max() + 1)
    counts_b = np.bincount(corr_b, minlength=corr_b.max() + 1)
    mask = (counts_a[corr_a] == 1) & (counts_b[corr_b] == 1)
    return np.where(mask)[0]


def identity_correspondence(n: int) -> Tuple[np.ndarray, np.ndarray]:
    c = np.arange(n)
    return c, c


def subsample_pair(
    pair: PairSample, max_vertices: int, rng: np.random.RandomState,
) -> PairSample:
    """Randomly subsample both shapes to at most max_vertices, remapping correspondence."""
    verts_a, verts_b = pair.verts_a, pair.verts_b
    corr_a, corr_b   = pair.corr_a,  pair.corr_b
    n_a, n_b = len(verts_a), len(verts_b)

    # Track original B data for geodesic computation
    verts_b_full = pair._verts_b_full if pair._verts_b_full is not None else verts_b
    idx_b_prev = pair._idx_b  # may already be set from a previous subsample

    if n_a > max_vertices:
        idx_a   = np.sort(rng.choice(n_a, max_vertices, replace=False))
        verts_a = verts_a[idx_a]
        remap_a = np.full(n_a, -1, dtype=np.int64)
        remap_a[idx_a] = np.arange(max_vertices)
    else:
        remap_a = np.arange(n_a, dtype=np.int64)

    if n_b > max_vertices:
        idx_b   = np.sort(rng.choice(n_b, max_vertices, replace=False))
        verts_b = verts_b[idx_b]
        remap_b = np.full(n_b, -1, dtype=np.int64)
        remap_b[idx_b] = np.arange(max_vertices)
        # Compose with previous index map if it exists
        idx_b_orig = idx_b_prev[idx_b] if idx_b_prev is not None else idx_b
    else:
        remap_b = np.arange(n_b, dtype=np.int64)
        idx_b_orig = idx_b_prev  # unchanged

    new_ca = remap_a[corr_a]
    new_cb = remap_b[corr_b]
    valid  = (new_ca >= 0) & (new_cb >= 0)
    return PairSample(
        verts_a, verts_b, new_ca[valid], new_cb[valid], pair.name,
        faces_b=pair.faces_b,
        _verts_b_full=verts_b_full,
        _idx_b=idx_b_orig,
    )


# =============================================================================
# SMAL loading utilities
# =============================================================================

def _install_fake_chumpy():
    if 'chumpy' in sys.modules:
        return
    chumpy             = types.ModuleType('chumpy')
    chumpy_ch          = types.ModuleType('chumpy.ch')
    chumpy_ch_ops      = types.ModuleType('chumpy.ch_ops')
    chumpy_reordering  = types.ModuleType('chumpy.reordering')
    chumpy_utils       = types.ModuleType('chumpy.utils')
    chumpy_logic       = types.ModuleType('chumpy.logic')

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

    sys.modules['chumpy']            = chumpy
    sys.modules['chumpy.ch']         = chumpy_ch
    sys.modules['chumpy.ch_ops']     = chumpy_ch_ops
    sys.modules['chumpy.reordering'] = chumpy_reordering
    sys.modules['chumpy.utils']      = chumpy_utils
    sys.modules['chumpy.logic']      = chumpy_logic
    chumpy.ch = chumpy_ch; chumpy.ch_ops = chumpy_ch_ops
    chumpy.reordering = chumpy_reordering; chumpy.utils = chumpy_utils
    chumpy.logic = chumpy_logic; chumpy.Ch = FakeCh
    chumpy_ch.Ch = FakeCh
    for attr in ['add', 'subtract', 'multiply', 'divide']:
        setattr(chumpy_ch_ops, attr, FakeCh)
    chumpy_reordering.transpose    = FakeCh
    chumpy_reordering.concatenate = FakeCh


def _to_numpy(x):
    if hasattr(x, 'r'):          return np.array(x.r)
    elif isinstance(x, np.ndarray): return x
    elif scipy.sparse.issparse(x):  return x
    return np.array(x)


def _rodrigues(axis_angle):
    N     = axis_angle.shape[0]
    theta = np.clip(np.linalg.norm(axis_angle, axis=1, keepdims=True), 1e-8, None)
    k     = axis_angle / theta
    K     = np.zeros((N, 3, 3))
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] =  k[:, 1]
    K[:, 1, 0] =  k[:, 2]; K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] =  k[:, 0]
    s = np.sin(theta)[:, :, np.newaxis]
    c = np.cos(theta)[:, :, np.newaxis]
    I = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    return I + s * K + (1 - c) * np.einsum('nij,njk->nik', K, K)


# =============================================================================
# SMAL model + pair generator
# =============================================================================

class SMALModel:
    """Cached SMAL model — load once, generate shapes quickly."""

    def __init__(self, model_path: str, data_path: str):
        _install_fake_chumpy()
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')
        with open(data_path, 'rb') as f:
            smal_data = pickle.load(f, encoding='latin1')

        self.v_template  = _to_numpy(params['v_template']).astype(np.float64)
        self.shapedirs   = _to_numpy(params['shapedirs']).astype(np.float64)
        self.posedirs    = _to_numpy(params['posedirs']).astype(np.float64)
        J_reg            = params['J_regressor']
        self.J_regressor = (J_reg.toarray().astype(np.float64)
                            if scipy.sparse.issparse(J_reg)
                            else _to_numpy(J_reg).astype(np.float64))
        self.weights       = _to_numpy(params['weights']).astype(np.float64)
        self.kintree_table = _to_numpy(params['kintree_table']).astype(np.int64)
        self.faces         = _to_numpy(params['f']).astype(np.int32)
        self.num_joints    = self.kintree_table.shape[1]
        self.num_betas     = self.shapedirs.shape[2]
        self.cluster_means = smal_data['cluster_means']
        self.num_families  = len(self.cluster_means)

    def generate(
        self, family_idx: int, pose_scale: float = 0.2,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Generate a normalized SMAL shape. Returns (N, 3) float32 vertices."""
        if rng is None:
            rng = np.random.RandomState(0)
        betas = np.zeros(self.num_betas)
        family_betas = self.cluster_means[family_idx]
        n = min(len(family_betas), self.num_betas)
        betas[:n] = family_betas[:n]

        pose     = rng.randn(self.num_joints * 3) * pose_scale
        pose[:3] *= 0.1

        v_shaped  = self.v_template + self.shapedirs.dot(betas)
        J         = self.J_regressor.dot(v_shaped)
        pose_vec  = pose.reshape(-1, 3)
        R         = _rodrigues(pose_vec)
        I_cube    = np.broadcast_to(np.eye(3), (self.num_joints - 1, 3, 3))
        lrotmin   = (R[1:] - I_cube).ravel()
        v_posed   = v_shaped + self.posedirs.dot(lrotmin)

        parent = {i: self.kintree_table[0, i] for i in range(1, self.num_joints)}

        def with_zeros(x):
            return np.vstack([x, [0, 0, 0, 1]])

        G = np.empty((self.num_joints, 4, 4))
        G[0] = with_zeros(np.hstack([R[0], J[0].reshape(3, 1)]))
        for i in range(1, self.num_joints):
            G[i] = G[parent[i]].dot(
                with_zeros(np.hstack([R[i], (J[i] - J[parent[i]]).reshape(3, 1)])))

        G_rest   = np.matmul(G,
                             np.hstack([J, np.zeros((self.num_joints, 1))
                                        ]).reshape(self.num_joints, 4, 1))
        G_packed = np.zeros((self.num_joints, 4, 4))
        G_packed[:, :, 3] = G_rest.squeeze(-1)
        G = G - G_packed

        T       = np.tensordot(self.weights, G, axes=[[1], [0]])
        v_homo  = np.hstack([v_posed, np.ones((len(v_posed), 1))])
        v_final = np.einsum('vij,vj->vi', T, v_homo)[:, :3]
        return normalize_mesh_vertices(v_final.astype(np.float32))


class SMALPairGenerator(PairGenerator):
    """Generate SMAL shape pairs with identity correspondence.

    train_families / val_families: null → auto (val = last 2, train = rest).
    """

    def __init__(
        self,
        smal: SMALModel,
        train_families: Optional[List[int]] = None,
        val_families:   Optional[List[int]] = None,
        pose_scale: float = 0.2,
    ):
        self.smal       = smal
        self.pose_scale = pose_scale

        all_families = list(range(smal.num_families))
        if val_families is None:
            val_families = all_families[-2:]
        if train_families is None:
            train_families = [f for f in all_families if f not in val_families]

        self.train_families = train_families
        self.val_families   = val_families
        self.cross_pairs    = list(combinations(train_families, 2))
        self.same_pairs     = [(f, f) for f in train_families]
        self.val_pairs      = list(combinations(val_families, 2))
        for t in train_families:
            for v in val_families:
                self.val_pairs.append((t, v))

    def sample_train_pair(
        self, rng: np.random.RandomState, cross_ratio: float = 0.5,
    ) -> PairSample:
        if rng.rand() < cross_ratio and self.cross_pairs:
            fam_a, fam_b = self.cross_pairs[rng.randint(len(self.cross_pairs))]
            pair_type = "cross"
        else:
            fam_a, fam_b = self.same_pairs[rng.randint(len(self.same_pairs))]
            pair_type = "same"
        verts_a = self.smal.generate(fam_a, self.pose_scale, rng)
        verts_b = self.smal.generate(fam_b, self.pose_scale, rng)
        corr_a, corr_b = identity_correspondence(len(verts_a))
        return PairSample(verts_a, verts_b, corr_a, corr_b,
                          f"{pair_type}_{fam_a}_vs_{fam_b}")

    def get_val_pairs(
        self, rng: np.random.RandomState, poses_per_pair: int = 1,
    ) -> List[PairSample]:
        pairs = []
        for fam_a, fam_b in self.val_pairs:
            for pose_idx in range(poses_per_pair):
                pair_rng = np.random.RandomState(fam_a * 100 + fam_b + pose_idx * 10000)
                verts_a  = self.smal.generate(fam_a, self.pose_scale, pair_rng)
                verts_b  = self.smal.generate(fam_b, self.pose_scale, pair_rng)
                corr_a, corr_b = identity_correspondence(len(verts_a))
                pairs.append(PairSample(verts_a, verts_b, corr_a, corr_b,
                                        f"val_{fam_a}_vs_{fam_b}_p{pose_idx}"))
        return pairs


# =============================================================================
# DT4D loading utilities
# =============================================================================

def _load_obj_vertices(path: str) -> np.ndarray:
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.array(verts, dtype=np.float32)


def _load_obj(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load vertices and faces from an OBJ file.

    Returns:
        vertices: (N, 3) float32
        faces: (F, 3) int32 (0-indexed), or None if no faces found.
    """
    verts = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                # OBJ face indices are 1-based, may contain v/vt/vn
                parts = line.split()[1:]
                if len(parts) >= 3:
                    idx = [int(p.split('/')[0]) - 1 for p in parts[:3]]
                    faces.append(idx)
    vertices = np.array(verts, dtype=np.float32)
    face_arr = np.array(faces, dtype=np.int32) if faces else None
    return vertices, face_arr


def _load_vts(path: str) -> np.ndarray:
    """Load VTS file. Returns 0-indexed (R,) int32 array."""
    return np.loadtxt(path, dtype=np.int32) - 1


class DT4DCategory:
    """One DT4D category (humanoid or animal) with all poses pre-loaded."""

    def __init__(self, root: Path, name: str):
        self.name    = name
        cat_dir      = root / name
        corres_dir   = cat_dir / "corres"
        self.poses:    List[str]        = []
        self.vertices: List[np.ndarray] = []
        self.faces:    List[Optional[np.ndarray]] = []
        self.vts:      List[np.ndarray] = []

        for obj_path in sorted(cat_dir.glob("*.obj")):
            vts_path = corres_dir / f"{obj_path.stem}.vts"
            if not vts_path.exists():
                continue
            verts, faces = _load_obj(str(obj_path))
            self.poses.append(obj_path.stem)
            self.vertices.append(normalize_mesh_vertices(verts))
            self.faces.append(faces)
            self.vts.append(_load_vts(str(vts_path)))

        self.num_poses = len(self.poses)
        self.ref_size  = len(self.vts[0]) if self.vts else 0

    def __repr__(self):
        return f"DT4DCategory({self.name}, {self.num_poses} poses, ref={self.ref_size})"


# ── Category discovery helpers ────────────────────────────────────────────────

def _discover_categories(root: Path) -> List[str]:
    """Discover all valid category directories under *root*."""
    cats = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith('.'):
            continue
        corres_dir = d / "corres"
        if not corres_dir.exists():
            continue
        if any((corres_dir / f"{o.stem}.vts").exists() for o in d.glob("*.obj")):
            cats.append(d.name)
    return cats


def _resolve_categories(
    root: Path,
    categories: Optional[List[str]],
    exclude_categories: Optional[List[str]],
) -> List[str]:
    """Resolve category specification to a concrete list.

    Exactly one of *categories* / *exclude_categories* may be non-None.
    Both None → use all categories discovered from *root*.
    """
    if categories is not None and exclude_categories is not None:
        raise ValueError(
            "Specify 'categories' OR 'exclude_categories', not both.")
    all_cats = _discover_categories(root)
    if not all_cats:
        raise FileNotFoundError(
            f"No valid DT4D categories found in {root}.")
    if categories is not None:
        missing = [c for c in categories if c not in all_cats]
        if missing:
            raise ValueError(f"Categories not found in {root}: {missing}")
        return sorted(categories)
    if exclude_categories is not None:
        result = [c for c in all_cats if c not in exclude_categories]
        if not result:
            raise ValueError("All categories excluded.")
        return result
    return all_cats


# ── Base pair generator ──────────────────────────────────────────────────────

class DT4DPairGeneratorBase(PairGenerator):
    """Shared DT4D logic: category loading, same-category pairs, val pairs."""

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        exclude_categories: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        resolved = _resolve_categories(self.root, categories, exclude_categories)
        print(f"  [{self.__class__.__name__}] root: {self.root}")
        print(f"  [{self.__class__.__name__}] Loading {len(resolved)} categories...",
              flush=True)

        self.category_names: List[str] = resolved
        self.categories: Dict[str, DT4DCategory] = {}
        for i, name in enumerate(resolved):
            cat = DT4DCategory(self.root, name)
            self.categories[name] = cat
            print(f"    [{i+1}/{len(resolved)}] {cat}", flush=True)

        self.same_pairs = [(c, c) for c in self.category_names]
        print(f"  [{self.__class__.__name__}] Loading complete.", flush=True)

    def _build_same_category_correspondence(
        self, cat_name: str, pose_a: int, pose_b: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cat = self.categories[cat_name]
        vts_a, vts_b = cat.vts[pose_a], cat.vts[pose_b]
        n_a = len(cat.vertices[pose_a])
        n_b = len(cat.vertices[pose_b])

        # VTS files may differ in length across poses (independently remeshed).
        # Use only the shared reference range.
        n_ref = min(len(vts_a), len(vts_b))
        vts_a = vts_a[:n_ref]
        vts_b = vts_b[:n_ref]

        mask = (vts_a >= 0) & (vts_a < n_a) & (vts_b >= 0) & (vts_b < n_b)
        return vts_a[mask], vts_b[mask]

    def _build_correspondence(
        self, cat_a: str, pose_a: int, cat_b: str, pose_b: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if cat_a == cat_b:
            return self._build_same_category_correspondence(cat_a, pose_a, pose_b)
        raise NotImplementedError(
            f"Cross-category correspondence not supported in {self.__class__.__name__}")

    def _get_val_pair_list(self) -> List[Tuple[str, str]]:
        return self.same_pairs

    def get_val_pairs(
        self, rng: np.random.RandomState, poses_per_pair: int = 1,
    ) -> List[PairSample]:
        pairs = []
        for cat_a, cat_b in self._get_val_pair_list():
            pair_rng = np.random.RandomState(_stable_hash((cat_a, cat_b)))
            c_a, c_b = self.categories[cat_a], self.categories[cat_b]
            seen: set = set()
            for _ in range(poses_per_pair):
                for _attempt in range(200):
                    pa = pair_rng.randint(c_a.num_poses)
                    pb = pair_rng.randint(c_b.num_poses)
                    if cat_a == cat_b and pa == pb and c_a.num_poses > 1:
                        continue
                    if (pa, pb) not in seen:
                        break
                else:
                    break
                seen.add((pa, pb))
                corr_a, corr_b = self._build_correspondence(cat_a, pa, cat_b, pb)
                pairs.append(PairSample(
                    verts_a=c_a.vertices[pa], verts_b=c_b.vertices[pb],
                    corr_a=corr_a, corr_b=corr_b,
                    name=f"val_{cat_a}:{c_a.poses[pa]}_vs_{cat_b}:{c_b.poses[pb]}",
                    faces_b=c_b.faces[pb],
                ))
        return pairs


# ── Humanoid pair generator (with cross-category bridges) ────────────────────

class DT4DHumanoidPairGenerator(DT4DPairGeneratorBase):
    """DT4D Humanoid pairs with optional cross-category bridge support."""

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        exclude_categories: Optional[List[str]] = None,
        enable_cross_category: bool = True,
        # Legacy compat: accept old-style args, ignore them with a warning
        train_categories: Optional[List[str]] = None,
        val_categories: Optional[List[str]] = None,
    ):
        # Legacy compat shim
        if train_categories is not None or val_categories is not None:
            import warnings
            warnings.warn(
                "train_categories / val_categories are deprecated. "
                "Use categories / exclude_categories instead.",
                DeprecationWarning, stacklevel=2,
            )
            if categories is None and exclude_categories is None:
                merged = sorted(set((train_categories or []) + (val_categories or [])))
                categories = merged if merged else None

        super().__init__(root, categories, exclude_categories)
        self.enable_cross_category = enable_cross_category

        self.bridges: Dict[Tuple[str, str], np.ndarray] = {}
        if enable_cross_category:
            bridge_dir = self.root / "cross_category_corres"
            if bridge_dir.exists():
                for f in bridge_dir.glob("*.vts"):
                    parts = f.stem.split("_", 1)
                    if len(parts) == 2:
                        src, dst = parts
                        if src in self.categories and dst in self.categories:
                            self.bridges[(src, dst)] = _load_vts(str(f))
                print(f"  Loaded {len(self.bridges)} cross-category bridges")

        self.cross_pairs = []
        if enable_cross_category:
            self.cross_pairs = [
                (a, b) for a, b in combinations(self.category_names, 2)
                if self._can_bridge(a, b)
            ]
        print(f"  Same-category pairs:  {len(self.same_pairs)}")
        print(f"  Cross-category pairs: {len(self.cross_pairs)}")

    def _can_bridge(self, a: str, b: str) -> bool:
        if a == b: return True
        if (a, b) in self.bridges or (b, a) in self.bridges: return True
        for hub in self.categories:
            if hub in (a, b): continue
            if ((a, hub) in self.bridges or (hub, a) in self.bridges) and \
               ((hub, b) in self.bridges or (b, hub) in self.bridges):
                return True
        return False

    def _get_bridge(self, a: str, b: str) -> np.ndarray:
        if (a, b) in self.bridges: return self.bridges[(a, b)]
        if (b, a) in self.bridges:
            return self._invert_bridge(self.bridges[(b, a)],
                                       self.categories[a].ref_size)
        for hub in self.categories:
            if hub in (a, b): continue
            if ((a, hub) in self.bridges or (hub, a) in self.bridges) and \
               ((hub, b) in self.bridges or (b, hub) in self.bridges):
                ab_hub = self._get_bridge(a, hub)
                hub_b  = self._get_bridge(hub, b)
                result = np.full(len(ab_hub), -1, dtype=np.int32)
                valid  = (ab_hub >= 0) & (ab_hub < len(hub_b))
                result[valid] = hub_b[ab_hub[valid]]
                return result
        raise ValueError(f"No bridge between {a} and {b}")

    @staticmethod
    def _invert_bridge(fwd: np.ndarray, target_size: int) -> np.ndarray:
        inv = np.full(target_size, -1, dtype=np.int32)
        for r_src, r_dst in enumerate(fwd):
            if 0 <= r_dst < target_size:
                inv[r_dst] = r_src
        return inv

    def _build_correspondence(
        self, cat_a: str, pose_a: int, cat_b: str, pose_b: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if cat_a == cat_b:
            return self._build_same_category_correspondence(cat_a, pose_a, pose_b)

        vts_a = self.categories[cat_a].vts[pose_a]
        vts_b = self.categories[cat_b].vts[pose_b]
        n_a   = len(self.categories[cat_a].vertices[pose_a])
        n_b   = len(self.categories[cat_b].vertices[pose_b])

        bridge = self._get_bridge(cat_a, cat_b)
        valid  = (bridge >= 0) & (bridge < len(vts_b))
        ref    = np.where(valid)[0]
        # Ensure ref indices are within vts_a range
        ref    = ref[ref < len(vts_a)]
        corr_a = vts_a[ref]
        corr_b = vts_b[bridge[ref]]
        mask = (corr_a >= 0) & (corr_a < n_a) & (corr_b >= 0) & (corr_b < n_b)
        return corr_a[mask], corr_b[mask]

    def sample_train_pair(
        self, rng: np.random.RandomState, cross_ratio: float = 0.5,
    ) -> PairSample:
        if self.enable_cross_category and rng.rand() < cross_ratio and self.cross_pairs:
            cat_a, cat_b = self.cross_pairs[rng.randint(len(self.cross_pairs))]
            pair_type = "cross"
        else:
            cat_a, cat_b = self.same_pairs[rng.randint(len(self.same_pairs))]
            pair_type = "same"
        c_a, c_b = self.categories[cat_a], self.categories[cat_b]
        pose_a = rng.randint(c_a.num_poses)
        pose_b = rng.randint(c_b.num_poses)
        if cat_a == cat_b and c_a.num_poses > 1:
            while pose_b == pose_a:
                pose_b = rng.randint(c_b.num_poses)
        corr_a, corr_b = self._build_correspondence(cat_a, pose_a, cat_b, pose_b)
        return PairSample(
            verts_a=c_a.vertices[pose_a], verts_b=c_b.vertices[pose_b],
            corr_a=corr_a, corr_b=corr_b,
            name=f"{pair_type}_{cat_a}:{c_a.poses[pose_a]}_vs_{cat_b}:{c_b.poses[pose_b]}",
            faces_b=c_b.faces[pose_b],
        )

    def _get_val_pair_list(self) -> List[Tuple[str, str]]:
        val_pairs = list(self.same_pairs)
        if self.enable_cross_category:
            for a, b in combinations(self.category_names, 2):
                if self._can_bridge(a, b):
                    val_pairs.append((a, b))
        return val_pairs


# Backward compatibility alias
DT4DPairGenerator = DT4DHumanoidPairGenerator


# ── Animals pair generator (inter-category only) ─────────────────────────────

class DT4DAnimalsPairGenerator(DT4DPairGeneratorBase):
    """DT4D Animals — same-category pairs only (no cross-category bridges)."""

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        exclude_categories: Optional[List[str]] = None,
    ):
        super().__init__(root, categories, exclude_categories)
        print(f"  Same-category pairs: {len(self.same_pairs)} "
              f"(cross-category not available)")

    def sample_train_pair(
        self, rng: np.random.RandomState, cross_ratio: float = 0.0,
    ) -> PairSample:
        cat_name = self.category_names[rng.randint(len(self.category_names))]
        cat = self.categories[cat_name]
        pose_a = rng.randint(cat.num_poses)
        pose_b = rng.randint(cat.num_poses)
        if cat.num_poses > 1:
            while pose_b == pose_a:
                pose_b = rng.randint(cat.num_poses)
        corr_a, corr_b = self._build_same_category_correspondence(
            cat_name, pose_a, pose_b)
        return PairSample(
            verts_a=cat.vertices[pose_a], verts_b=cat.vertices[pose_b],
            corr_a=corr_a, corr_b=corr_b,
            name=f"same_{cat_name}:{cat.poses[pose_a]}_vs_{cat_name}:{cat.poses[pose_b]}",
            faces_b=cat.faces[pose_b],
        )


# ── Composite pair generator ─────────────────────────────────────────────────

class CompositePairGenerator(PairGenerator):
    """Merge multiple PairGenerators with configurable sampling weights."""

    def __init__(
        self,
        generators: List[PairGenerator],
        weights: Optional[List[float]] = None,
    ):
        if not generators:
            raise ValueError("CompositePairGenerator requires at least one generator")
        self.generators = generators
        if weights is not None:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            n = len(generators)
            self.weights = [1.0 / n] * n

    def sample_train_pair(
        self, rng: np.random.RandomState, cross_ratio: float = 0.0,
    ) -> PairSample:
        idx = rng.choice(len(self.generators), p=self.weights)
        return self.generators[idx].sample_train_pair(rng, cross_ratio)

    def get_val_pairs(
        self, rng: np.random.RandomState, poses_per_pair: int = 1,
    ) -> List[PairSample]:
        all_pairs = []
        for gen in self.generators:
            all_pairs.extend(gen.get_val_pairs(rng, poses_per_pair))
        return all_pairs


# =============================================================================
# Torch Datasets
# =============================================================================

def pair_collate_fn(batch: List[PairSample]) -> List[PairSample]:
    """Pass-through collate: variable vertex count, cannot be stacked."""
    return batch


class ShapePairDataset(Dataset):
    """Training dataset — generates PairSamples on-the-fly.

    cross_ratio is a mutable attribute updated each epoch by
    FunctionalMapModule.on_train_epoch_start.
    """

    def __init__(
        self,
        pair_generator: PairGenerator,
        num_samples: int,
        cross_ratio: float = 0.0,
    ):
        self.pair_generator = pair_generator
        self.num_samples    = num_samples
        self.cross_ratio    = cross_ratio

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> PairSample:
        rng = np.random.RandomState()   # fresh OS-entropy seed — no epoch tracking needed
        return self.pair_generator.sample_train_pair(rng, self.cross_ratio)


class ValPairDataset(Dataset):
    """Validation dataset — fixed list generated once at construction."""

    def __init__(
        self,
        pair_generator: PairGenerator,
        val_poses_per_pair: int = 2,
        seed: int = 42,
    ):
        rng        = np.random.RandomState(seed)
        self.pairs = pair_generator.get_val_pairs(rng, val_poses_per_pair)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PairSample:
        return self.pairs[idx]