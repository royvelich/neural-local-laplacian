#!/usr/bin/env python3
"""
Functional Maps Sanity Check

Minimal experiment to test whether the neural Laplacian produces better
eigenbases than robust-laplacian for functional map correspondence on
point clouds (no faces).

Setup:
    - Two shapes with KNOWN ground truth correspondence (e.g., SMAL animals
      with shared template topology, or two mesh files with vertex-to-vertex
      correspondence).
    - Faces are discarded — both shapes are treated as raw point clouds.

Pipeline:
    1. Compute k eigenvectors on each shape using:
       (a) Neural Laplacian (your model)
       (b) Robust-laplacian (point cloud mode)
       (c) Robust-laplacian (mesh mode, as upper bound reference — uses faces)
    2. Build ground-truth functional map: C = Φ_B^T M_B Π Φ_A
    3. Measure:
       - Orthogonality: ||C^T C - I||_F  (lower = better basis alignment)
       - Bijectivity:   ||C C^T - I||_F  (lower = better invertibility)
       - Correspondence error: convert C to point-to-point map, measure
         geodesic error against ground truth

Usage:
    # With two mesh files (vertex correspondence = identity)
    python functional_maps_sanity.py \
        --shape_a lion.obj --shape_b dog.obj \
        --checkpoint model.ckpt --k 20 --num_eigenvectors 30

    # With SMAL model (generates two animals from family presets)
    python functional_maps_sanity.py \
        --smal_model smal_CVPR2017.pkl --smal_data smal_CVPR2017_data.pkl \
        --family_a 0 --family_b 1 \
        --checkpoint model.ckpt --k 20 --num_eigenvectors 30
"""

import argparse
import sys
import types
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import scipy.sparse as sp
import torch
import trimesh
from sklearn.neighbors import NearestNeighbors
import robust_laplacian

from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData
from neural_local_laplacian.utils.utils import (
    normalize_mesh_vertices,
    assemble_stiffness_and_mass_matrices,
    compute_laplacian_eigendecomposition,
)
from torch_geometric.data import Batch


# =============================================================================
# SMAL loading (minimal, from smal_viewer.py)
# =============================================================================

def _install_fake_chumpy():
    """Install minimal fake chumpy modules so pickle can deserialize SMAL."""
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
                        self._data = state[key]
                        break

        @property
        def r(self):
            if self._data is not None:
                return np.array(self._data)
            return np.array([])

        def __array__(self, dtype=None):
            arr = self.r
            return arr.astype(dtype) if dtype else arr

    sys.modules['chumpy'] = chumpy
    sys.modules['chumpy.ch'] = chumpy_ch
    sys.modules['chumpy.ch_ops'] = chumpy_ch_ops
    sys.modules['chumpy.reordering'] = chumpy_reordering
    sys.modules['chumpy.utils'] = chumpy_utils
    sys.modules['chumpy.logic'] = chumpy_logic
    chumpy.ch = chumpy_ch
    chumpy.ch_ops = chumpy_ch_ops
    chumpy.reordering = chumpy_reordering
    chumpy.utils = chumpy_utils
    chumpy.logic = chumpy_logic
    chumpy.Ch = FakeCh
    chumpy_ch.Ch = FakeCh
    chumpy_ch_ops.add = FakeCh
    chumpy_ch_ops.subtract = FakeCh
    chumpy_ch_ops.multiply = FakeCh
    chumpy_ch_ops.divide = FakeCh
    chumpy_reordering.transpose = FakeCh
    chumpy_reordering.concatenate = FakeCh


def _to_numpy(x):
    if hasattr(x, 'r'):
        return np.array(x.r)
    elif isinstance(x, np.ndarray):
        return x
    elif sp.issparse(x):
        return x
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


def load_smal_shape(model_path: str, data_path: str, family_idx: int,
                    pose_scale: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load a SMAL animal shape. Returns (vertices, faces)."""
    _install_fake_chumpy()

    with open(model_path, 'rb') as f:
        params = pickle.load(f, encoding='latin1')

    v_template = _to_numpy(params['v_template']).astype(np.float64)
    shapedirs = _to_numpy(params['shapedirs']).astype(np.float64)
    posedirs = _to_numpy(params['posedirs']).astype(np.float64)
    J_regressor = params['J_regressor']
    if sp.issparse(J_regressor):
        J_regressor = J_regressor.toarray().astype(np.float64)
    else:
        J_regressor = _to_numpy(J_regressor).astype(np.float64)
    weights = _to_numpy(params['weights']).astype(np.float64)
    kintree_table = _to_numpy(params['kintree_table']).astype(np.int64)
    faces = _to_numpy(params['f']).astype(np.int32)

    num_joints = kintree_table.shape[1]
    num_betas = shapedirs.shape[2]

    # Load family betas
    with open(data_path, 'rb') as f:
        smal_data = pickle.load(f, encoding='latin1')
    cluster_means = smal_data.get('cluster_means', None)
    if cluster_means is None:
        raise ValueError("No cluster_means in SMAL data file")

    betas = np.zeros(num_betas)
    family_betas = cluster_means[family_idx]
    n = min(len(family_betas), num_betas)
    betas[:n] = family_betas[:n]

    # Optional random pose
    pose = np.random.randn(num_joints * 3) * pose_scale if pose_scale > 0 else np.zeros(num_joints * 3)
    pose[:3] *= 0.1  # small root rotation

    # Forward kinematics (simplified from smal_viewer.py)
    v_shaped = v_template + shapedirs.dot(betas)
    J = J_regressor.dot(v_shaped)
    pose_vec = pose.reshape(-1, 3)
    R = _rodrigues(pose_vec)
    I_cube = np.broadcast_to(np.eye(3), (num_joints - 1, 3, 3))
    lrotmin = (R[1:] - I_cube).ravel()
    v_posed = v_shaped + posedirs.dot(lrotmin)

    parent = {}
    for i in range(1, num_joints):
        parent[i] = kintree_table[0, i]

    def with_zeros(x):
        return np.vstack([x, [0, 0, 0, 1]])

    G = np.empty((num_joints, 4, 4))
    G[0] = with_zeros(np.hstack([R[0], J[0].reshape(3, 1)]))
    for i in range(1, num_joints):
        G[i] = G[parent[i]].dot(with_zeros(np.hstack([R[i], (J[i] - J[parent[i]]).reshape(3, 1)])))

    G_rest = np.matmul(G, np.hstack([J, np.zeros((num_joints, 1))]).reshape(num_joints, 4, 1))
    G_packed = np.zeros((num_joints, 4, 4))
    G_packed[:, :, 3] = G_rest.squeeze(-1)
    G = G - G_packed

    T = np.tensordot(weights, G, axes=[[1], [0]])
    v_homo = np.hstack([v_posed, np.ones((len(v_posed), 1))])
    v_final = np.einsum('vij,vj->vi', T, v_homo)[:, :3]

    return v_final.astype(np.float32), faces


# =============================================================================
# Shared utilities
# =============================================================================

def load_model(ckpt_path: str, device: torch.device) -> LaplacianTransformerModule:
    """Load a frozen neural Laplacian model."""
    print(f"Loading model from: {ckpt_path}")
    model = LaplacianTransformerModule.load_from_checkpoint(
        ckpt_path, map_location=device,
        normalize_patch_features=True,
        scale_areas_by_patch_size=True,
    )
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def compute_knn(vertices_np: np.ndarray, k: int) -> np.ndarray:
    """Compute k-NN indices excluding self. Returns (N, k)."""
    n = len(vertices_np)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(vertices_np)
    _, idx = nbrs.kneighbors(vertices_np)
    center = np.arange(n)[:, np.newaxis]
    keep = ~(idx == center)
    keep_pos = np.cumsum(keep, axis=1)
    final = (keep_pos <= k) & keep
    return idx[final].reshape(n, k)


def build_patch_data(vertices: torch.Tensor, neighbor_indices: np.ndarray,
                     device: torch.device) -> MeshPatchData:
    """Build differentiable patch data."""
    n, k = vertices.shape[0], neighbor_indices.shape[1]
    idx = torch.from_numpy(neighbor_indices).long().to(device)
    patch_pos = vertices[idx] - vertices[:, None, :]
    all_pos = patch_pos.reshape(-1, 3)
    return MeshPatchData(
        pos=all_pos, x=all_pos,
        patch_idx=torch.arange(n, device=device).repeat_interleave(k),
        vertex_indices=idx.flatten(),
        center_indices=torch.arange(n, device=device),
    )


# =============================================================================
# Eigenbasis computation
# =============================================================================

def compute_neural_eigenbasis(
    model: LaplacianTransformerModule,
    vertices_np: np.ndarray,
    k: int,
    num_eigenvectors: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute eigenbasis using the neural Laplacian.

    Returns: (eigenvalues, eigenvectors, mass_diagonal)
        eigenvalues: (num_eigenvectors,)
        eigenvectors: (N, num_eigenvectors)
        mass_diagonal: (N,)  — for building functional maps
    """
    vertices_t = torch.from_numpy(vertices_np).float().to(device)
    knn = compute_knn(vertices_np, k)

    batch_data = build_patch_data(vertices_t, knn, device)
    batch_data = Batch.from_data_list([batch_data]).to(device)

    with torch.no_grad():
        fwd = model._forward_pass(batch_data)

    batch_idx = getattr(batch_data, 'patch_idx', batch_data.batch)

    # Assemble sparse matrices (scipy)
    S, M = assemble_stiffness_and_mass_matrices(
        stiffness_weights=fwd['stiffness_weights'],
        areas=fwd['areas'],
        attention_mask=fwd['attention_mask'],
        vertex_indices=batch_data.vertex_indices,
        center_indices=batch_data.center_indices,
        batch_indices=batch_idx,
    )

    eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
        S, num_eigenvectors, mass_matrix=M
    )

    # Extract mass diagonal for functional map computation
    M_diag = np.array(M.diagonal()).flatten()

    return eigenvalues, eigenvectors, M_diag


def compute_robust_eigenbasis_pointcloud(
    vertices_np: np.ndarray,
    num_eigenvectors: int,
    n_neighbors: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute eigenbasis using robust-laplacian in point cloud mode (no faces).

    Returns: (eigenvalues, eigenvectors, mass_diagonal)
    """
    L, M = robust_laplacian.point_cloud_laplacian(vertices_np, n_neighbors=n_neighbors)
    eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
        L, num_eigenvectors, mass_matrix=M
    )
    M_diag = np.array(M.diagonal()).flatten()
    return eigenvalues, eigenvectors, M_diag


def compute_robust_eigenbasis_mesh(
    vertices_np: np.ndarray,
    faces: np.ndarray,
    num_eigenvectors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute eigenbasis using robust-laplacian in mesh mode (uses faces).
    This serves as an upper-bound reference.

    Returns: (eigenvalues, eigenvectors, mass_diagonal)
    """
    L, M = robust_laplacian.mesh_laplacian(vertices_np, faces)
    eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
        L, num_eigenvectors, mass_matrix=M
    )
    M_diag = np.array(M.diagonal()).flatten()
    return eigenvalues, eigenvectors, M_diag


# =============================================================================
# Functional map computation and evaluation
# =============================================================================

def compute_functional_map(
    eigvecs_a: np.ndarray,
    eigvecs_b: np.ndarray,
    mass_diag_b: np.ndarray,
    correspondence: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the ground-truth functional map C from shape A to shape B.

    C = Φ_B^T M_B Π Φ_A

    where Π is the correspondence matrix. If correspondence is None,
    assumes identity (vertex i on A maps to vertex i on B).

    Args:
        eigvecs_a: (N_A, k) eigenvectors of shape A
        eigvecs_b: (N_B, k) eigenvectors of shape B
        mass_diag_b: (N_B,) mass matrix diagonal of shape B
        correspondence: (N_A,) integer array — correspondence[i] = j means
            vertex i on A maps to vertex j on B. None = identity.

    Returns:
        C: (k, k) functional map matrix
    """
    k = eigvecs_a.shape[1]

    # Π Φ_A: reorder rows of Φ_A according to correspondence
    if correspondence is not None:
        phi_a_permuted = eigvecs_a[correspondence]  # (N_B, k) if N_A == N_B
    else:
        phi_a_permuted = eigvecs_a  # identity correspondence

    # C = Φ_B^T M_B Π Φ_A = (Φ_B^T diag(m_b)) @ (Π Φ_A)
    # For diagonal M: Φ_B^T M_B = (Φ_B * m_b[:, None])^T
    weighted_phi_b = eigvecs_b * mass_diag_b[:, None]  # (N, k)
    C = weighted_phi_b.T @ phi_a_permuted  # (k, k)

    return C


def functional_map_to_pointwise(
    C: np.ndarray,
    eigvecs_a: np.ndarray,
    eigvecs_b: np.ndarray,
) -> np.ndarray:
    """
    Convert functional map C to point-to-point correspondence.

    For each vertex i in A, projects its spectral embedding through C
    and finds the nearest neighbor in B's spectral embedding.

    Args:
        C: (k, k) functional map
        eigvecs_a: (N_A, k) eigenvectors of shape A
        eigvecs_b: (N_B, k) eigenvectors of shape B

    Returns:
        correspondence: (N_A,) — correspondence[i] = index in B
    """
    # Project A's embedding through C: each row of Φ_A @ C^T
    projected_a = eigvecs_a @ C.T  # (N_A, k)

    # Find nearest neighbor in Φ_B
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(eigvecs_b)
    _, indices = nbrs.kneighbors(projected_a)

    return indices.flatten()


def evaluate_functional_map(
    C: np.ndarray,
    eigvecs_a: np.ndarray,
    eigvecs_b: np.ndarray,
    gt_correspondence: np.ndarray = None,
    vertices_b: np.ndarray = None,
) -> Dict[str, float]:
    """
    Evaluate functional map quality.

    Args:
        C: (k, k) functional map
        eigvecs_a, eigvecs_b: eigenvectors of shapes A and B
        gt_correspondence: (N_A,) ground truth — gt[i] = j in B. None = identity.
        vertices_b: (N_B, 3) positions for geodesic error (Euclidean approx)

    Returns:
        Dictionary of metrics
    """
    k = C.shape[0]
    I = np.eye(k)

    # Orthogonality: ||C^T C - I||_F
    ortho_error = np.linalg.norm(C.T @ C - I, 'fro')

    # Bijectivity: ||C C^T - I||_F
    biject_error = np.linalg.norm(C @ C.T - I, 'fro')

    # Off-diagonal energy: measures how diagonal C is
    # For perfectly aligned bases, C should be close to diagonal
    diag_energy = np.sum(np.diag(C) ** 2)
    total_energy = np.sum(C ** 2)
    diag_ratio = diag_energy / (total_energy + 1e-10)

    metrics = {
        'orthogonality_error': ortho_error,
        'bijectivity_error': biject_error,
        'diagonal_ratio': diag_ratio,
    }

    # Point-to-point correspondence error
    if gt_correspondence is None:
        gt_correspondence = np.arange(eigvecs_a.shape[0])

    pred_correspondence = functional_map_to_pointwise(C, eigvecs_a, eigvecs_b)

    # Accuracy: fraction of correct correspondences
    correct = (pred_correspondence == gt_correspondence)
    metrics['accuracy'] = float(correct.mean())

    # Euclidean correspondence error (proxy for geodesic)
    if vertices_b is not None:
        pred_positions = vertices_b[pred_correspondence]
        gt_positions = vertices_b[gt_correspondence]
        errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
        metrics['mean_euclidean_error'] = float(errors.mean())
        metrics['median_euclidean_error'] = float(np.median(errors))
        metrics['max_euclidean_error'] = float(errors.max())

        # Normalized by bounding box diagonal
        bb_diag = np.linalg.norm(vertices_b.max(axis=0) - vertices_b.min(axis=0))
        metrics['mean_error_normalized'] = float(errors.mean() / bb_diag)
        metrics['median_error_normalized'] = float(np.median(errors) / bb_diag)

    return metrics


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(
    vertices_a: np.ndarray,
    vertices_b: np.ndarray,
    faces_a: np.ndarray,
    faces_b: np.ndarray,
    model: LaplacianTransformerModule,
    k_pred: int,
    k_robust: int,
    num_eigenvectors: int,
    device: torch.device,
    name_a: str = "Shape A",
    name_b: str = "Shape B",
) -> Dict:
    """Run the functional map comparison experiment. Returns results dict."""
    n_a, n_b = len(vertices_a), len(vertices_b)
    assert n_a == n_b, (
        f"Shapes must have same number of vertices for identity correspondence "
        f"(got {n_a} vs {n_b})"
    )

    print("=" * 80)
    print("FUNCTIONAL MAP SANITY CHECK")
    print("=" * 80)
    print(f"  {name_a}: {n_a} vertices")
    print(f"  {name_b}: {n_b} vertices")
    print(f"  Correspondence: identity (vertex i <-> vertex i)")
    print(f"  Eigenvectors: {num_eigenvectors}")
    print(f"  Neural Laplacian k: {k_pred}")
    print(f"  Robust Laplacian k: {k_robust}")
    print()

    # --- Compute eigenbases ---
    methods = {}

    # 1. Neural Laplacian (point cloud)
    print("Computing eigenbasis: Neural Laplacian (point cloud)...")
    eig_a = compute_neural_eigenbasis(model, vertices_a, k_pred, num_eigenvectors, device)
    eig_b = compute_neural_eigenbasis(model, vertices_b, k_pred, num_eigenvectors, device)
    methods['Neural (PC)'] = (eig_a, eig_b)
    print(f"  Done. Eigenvalue range A: [{eig_a[0][0]:.4f}, {eig_a[0][-1]:.4f}]")
    print(f"  Done. Eigenvalue range B: [{eig_b[0][0]:.4f}, {eig_b[0][-1]:.4f}]")

    # 2. Robust-laplacian (point cloud)
    print(f"Computing eigenbasis: Robust-laplacian (point cloud, k={k_robust})...")
    eig_a_rpc = compute_robust_eigenbasis_pointcloud(vertices_a, num_eigenvectors, n_neighbors=k_robust)
    eig_b_rpc = compute_robust_eigenbasis_pointcloud(vertices_b, num_eigenvectors, n_neighbors=k_robust)
    methods['Robust (PC)'] = (eig_a_rpc, eig_b_rpc)
    print(f"  Done. Eigenvalue range A: [{eig_a_rpc[0][0]:.4f}, {eig_a_rpc[0][-1]:.4f}]")
    print(f"  Done. Eigenvalue range B: [{eig_b_rpc[0][0]:.4f}, {eig_b_rpc[0][-1]:.4f}]")

    # 3. Robust-laplacian (mesh — upper bound reference)
    if faces_a is not None and faces_b is not None:
        print("Computing eigenbasis: Robust-laplacian (mesh, upper bound)...")
        eig_a_rm = compute_robust_eigenbasis_mesh(vertices_a, faces_a, num_eigenvectors)
        eig_b_rm = compute_robust_eigenbasis_mesh(vertices_b, faces_b, num_eigenvectors)
        methods['Robust (Mesh)'] = (eig_a_rm, eig_b_rm)
        print(f"  Done. Eigenvalue range A: [{eig_a_rm[0][0]:.4f}, {eig_a_rm[0][-1]:.4f}]")
        print(f"  Done. Eigenvalue range B: [{eig_b_rm[0][0]:.4f}, {eig_b_rm[0][-1]:.4f}]")

    # --- Compute and evaluate functional maps ---
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Header
    metric_names = [
        'orthogonality_error', 'bijectivity_error', 'diagonal_ratio',
        'accuracy', 'mean_error_normalized', 'median_error_normalized',
    ]
    header_labels = {
        'orthogonality_error': '||CtC-I||',
        'bijectivity_error': '||CCt-I||',
        'diagonal_ratio': 'Diag%',
        'accuracy': 'Acc%',
        'mean_error_normalized': 'MeanErr',
        'median_error_normalized': 'MedErr',
    }

    header = f"{'Method':<18}"
    for m in metric_names:
        header += f" | {header_labels[m]:>10}"
    print(header)
    print("-" * len(header))

    all_results = {}
    all_correspondences = {}
    all_C = {}
    for method_name, (eig_a_m, eig_b_m) in methods.items():
        eigenvalues_a, eigvecs_a, mass_a = eig_a_m
        eigenvalues_b, eigvecs_b, mass_b = eig_b_m

        # Build functional map (identity correspondence)
        C = compute_functional_map(eigvecs_a, eigvecs_b, mass_b)
        all_C[method_name] = C

        # Point-to-point correspondence
        pred_corr = functional_map_to_pointwise(C, eigvecs_a, eigvecs_b)
        all_correspondences[method_name] = pred_corr

        # Evaluate
        metrics = evaluate_functional_map(
            C, eigvecs_a, eigvecs_b,
            gt_correspondence=None,  # identity
            vertices_b=vertices_b,
        )
        all_results[method_name] = metrics

        # Print row
        row = f"{method_name:<18}"
        for m in metric_names:
            val = metrics.get(m, float('nan'))
            if m in ('accuracy', 'diagonal_ratio'):
                row += f" | {val * 100:>9.2f}%"
            else:
                row += f" | {val:>10.4f}"
        print(row)

    print()

    # --- Detailed per-method output ---
    for method_name, metrics in all_results.items():
        print(f"\n--- {method_name} ---")
        for k_m, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k_m}: {v:.6f}")

    # --- Eigenvalue comparison ---
    print()
    print("=" * 80)
    print("EIGENVALUE COMPARISON (first 10, skip lambda_0)")
    print("=" * 80)

    eig_header = f"{'Idx':>3}"
    for method_name in methods:
        eig_header += f" | {method_name + ' A':>14} | {method_name + ' B':>14}"
    print(eig_header)
    print("-" * len(eig_header))

    for i in range(1, min(11, num_eigenvectors)):
        row = f"{i:>3}"
        for method_name, (eig_a_m, eig_b_m) in methods.items():
            row += f" | {eig_a_m[0][i]:>14.4f} | {eig_b_m[0][i]:>14.4f}"
        print(row)

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("  - ||CtC - I||: Lower = more orthogonal C = better basis alignment")
    print("  - ||CCt - I||: Lower = more bijective correspondence")
    print("  - Diag%: Higher = bases are more naturally aligned (C closer to identity)")
    print("  - Acc%: Fraction of vertices mapped to correct correspondence")
    print("  - MeanErr/MedErr: Euclidean distance error, normalized by bounding box diagonal")
    print()
    print("  If Neural (PC) outperforms Robust (PC), the learned Laplacian")
    print("  produces better eigenbases for cross-shape correspondence on point clouds.")
    print("  Robust (Mesh) is the upper bound — it uses face connectivity.")

    return {
        'methods': methods,
        'results': all_results,
        'correspondences': all_correspondences,
        'C_matrices': all_C,
    }


# =============================================================================
# Polyscope visualization
# =============================================================================

def compute_position_colors(vertices: np.ndarray) -> np.ndarray:
    """
    Map vertex positions to RGB colors for correspondence visualization.
    Normalizes XYZ to [0, 1] range and uses as RGB.
    """
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    v_range = v_max - v_min
    v_range = np.where(v_range < 1e-8, 1.0, v_range)
    colors = (vertices - v_min) / v_range  # (N, 3) in [0, 1]
    return colors


def visualize_correspondences(
    vertices_a: np.ndarray,
    vertices_b: np.ndarray,
    faces_a: Optional[np.ndarray],
    faces_b: Optional[np.ndarray],
    correspondences: Dict[str, np.ndarray],
    results: Dict[str, Dict],
    name_a: str = "Shape A",
    name_b: str = "Shape B",
):
    """
    Visualize functional map correspondences with polyscope.

    Layout (all visible simultaneously):
        Shape A (position colors) | GT | Neural(PC) | Robust(PC) | Robust(Mesh)

    Shape A is colored by position (XYZ -> RGB).
    Each copy of Shape B is colored by transferring A's colors via the method's
    predicted correspondence. If correspondence is perfect, body parts match.
    """
    import polyscope as ps
    import polyscope.imgui as psim

    method_names = list(correspondences.keys())
    n_a, n_b = len(vertices_a), len(vertices_b)

    # Position-based colors on shape A (the reference)
    colors_a = compute_position_colors(vertices_a)

    # Spacing between shapes
    bb_a = vertices_a.max(axis=0) - vertices_a.min(axis=0)
    spacing = bb_a[0] * 1.5

    # Initialize polyscope
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_background_color((0.1, 0.1, 0.12))

    has_faces_b = faces_b is not None and len(faces_b) > 0

    # --- Shape A (leftmost) ---
    if faces_a is not None and len(faces_a) > 0:
        mesh_a = ps.register_surface_mesh(
            f"{name_a} (source)", vertices_a, faces_a, smooth_shade=True
        )
    else:
        mesh_a = ps.register_point_cloud(
            f"{name_a} (source)", vertices_a, radius=0.003
        )
    mesh_a.add_color_quantity("position_color", colors_a, enabled=True)

    # --- GT (identity correspondence) ---
    col = 1
    verts_gt = vertices_b + np.array([spacing * col, 0, 0])
    if has_faces_b:
        mesh_gt = ps.register_surface_mesh(
            "GT (identity)", verts_gt, faces_b, smooth_shade=True
        )
    else:
        mesh_gt = ps.register_point_cloud("GT (identity)", verts_gt, radius=0.003)
    # Identity correspondence: vertex i on A -> vertex i on B
    mesh_gt.add_color_quantity("correspondence", colors_a, enabled=True)

    # --- One copy of Shape B per method ---
    meshes = {}
    for method_name, pred_corr in correspondences.items():
        col += 1
        verts_offset = vertices_b + np.array([spacing * col, 0, 0])

        # Compute transferred colors: color vertex j on B by looking up
        # which vertex on A maps to it
        tc = np.full((n_b, 3), 0.5)  # gray = unmapped
        for i_a, i_b in enumerate(pred_corr):
            if i_b < n_b:
                tc[i_b] = colors_a[i_a]

        # Compute error: per-vertex on A, then transfer to B
        gt_positions = vertices_b[np.arange(len(pred_corr))]
        pred_positions = vertices_b[pred_corr]
        errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
        err_on_b = np.zeros(n_b)
        for i_a, i_b in enumerate(pred_corr):
            if i_b < n_b:
                err_on_b[i_b] = errors[i_a]

        # Build label with metrics
        m = results[method_name]
        acc = m.get('accuracy', 0) * 100
        err = m.get('mean_error_normalized', 0)
        label = f"{method_name} (Acc:{acc:.1f}%)"

        if has_faces_b:
            mesh = ps.register_surface_mesh(
                label, verts_offset, faces_b, smooth_shade=True
            )
        else:
            mesh = ps.register_point_cloud(label, verts_offset, radius=0.003)

        mesh.add_color_quantity("correspondence", tc, enabled=True)
        mesh.add_scalar_quantity("error", err_on_b, enabled=False, cmap='reds')
        meshes[method_name] = mesh

    # --- UI callback for toggling error view ---
    state = {'show_error': False}

    def _ui_callback():
        psim.SetNextWindowSize((350, 300))
        psim.SetNextWindowPos((10, 10))

        opened = psim.Begin("Correspondence Comparison", True)
        if not opened:
            psim.End()
            return

        c, state['show_error'] = psim.Checkbox(
            "Show error heatmap (toggle)", state['show_error']
        )
        if c:
            for method_name, mesh in meshes.items():
                if state['show_error']:
                    mesh.set_enabled(True)
                    # error quantity is already added, just need to re-enable
                    m_results = results[method_name]

        psim.Separator()

        psim.Text("Layout (left to right):")
        psim.Text(f"  1. {name_a} (position colors)")
        psim.Text(f"  2. GT (identity correspondence)")
        for i, mn in enumerate(method_names):
            m = results[mn]
            psim.Text(f"  {i+3}. {mn}")
            psim.Text(f"      Acc: {m['accuracy']*100:.1f}%  "
                       f"||CtC-I||: {m['orthogonality_error']:.3f}  "
                       f"Err: {m.get('mean_error_normalized',0):.4f}")

        psim.Separator()
        psim.Text("Matching colors = correct correspondence")
        psim.Text("Use polyscope's structure menu to")
        psim.Text("toggle error heatmaps per shape.")

        psim.End()

    ps.set_user_callback(_ui_callback)

    print("\nShowing correspondence visualization. Close the polyscope window to exit.")
    print(f"Layout: {name_a} | GT | {' | '.join(method_names)}")
    ps.show()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Functional Maps Sanity Check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shape input: either mesh files or SMAL
    shape_group = parser.add_argument_group("Shape input (mesh files)")
    shape_group.add_argument("--shape_a", type=str, default=None,
                             help="Path to first mesh file")
    shape_group.add_argument("--shape_b", type=str, default=None,
                             help="Path to second mesh file")

    smal_group = parser.add_argument_group("Shape input (SMAL model)")
    smal_group.add_argument("--smal_model", type=str, default=None,
                            help="Path to smal_CVPR2017.pkl")
    smal_group.add_argument("--smal_data", type=str, default=None,
                            help="Path to smal_CVPR2017_data.pkl")
    smal_group.add_argument("--family_a", type=int, default=0,
                            help="SMAL family index for shape A")
    smal_group.add_argument("--family_b", type=int, default=1,
                            help="SMAL family index for shape B")
    smal_group.add_argument("--pose_scale", type=float, default=0.0,
                            help="Random pose scale (0 = T-pose)")

    # Model and params
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to neural Laplacian model checkpoint")
    parser.add_argument("--k_pred", type=int, default=20,
                        help="k-NN for neural Laplacian (PRED model)")
    parser.add_argument("--k_robust", type=int, default=30,
                        help="k-NN for robust-laplacian point cloud mode")
    parser.add_argument("--num_eigenvectors", type=int, default=30,
                        help="Number of eigenvectors to compute")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--no_vis", action="store_true",
                        help="Skip polyscope visualization")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load shapes
    if args.smal_model is not None:
        assert args.smal_data is not None, "Need --smal_data with --smal_model"
        print(f"\nLoading SMAL shapes: family {args.family_a} vs family {args.family_b}")

        vertices_a, faces_a = load_smal_shape(
            args.smal_model, args.smal_data, args.family_a, args.pose_scale
        )
        vertices_b, faces_b = load_smal_shape(
            args.smal_model, args.smal_data, args.family_b, args.pose_scale
        )

        # Normalize both
        vertices_a = normalize_mesh_vertices(vertices_a)
        vertices_b = normalize_mesh_vertices(vertices_b)

        name_a = f"SMAL family {args.family_a}"
        name_b = f"SMAL family {args.family_b}"

    elif args.shape_a is not None and args.shape_b is not None:
        print(f"\nLoading mesh files...")
        mesh_a = trimesh.load(args.shape_a)
        mesh_b = trimesh.load(args.shape_b)

        vertices_a = normalize_mesh_vertices(np.array(mesh_a.vertices, dtype=np.float32))
        vertices_b = normalize_mesh_vertices(np.array(mesh_b.vertices, dtype=np.float32))
        faces_a = np.array(mesh_a.faces, dtype=np.int64) if hasattr(mesh_a, 'faces') else None
        faces_b = np.array(mesh_b.faces, dtype=np.int64) if hasattr(mesh_b, 'faces') else None

        name_a = Path(args.shape_a).stem
        name_b = Path(args.shape_b).stem

    else:
        parser.error("Provide either --shape_a/--shape_b or --smal_model/--smal_data")

    print(f"  {name_a}: {len(vertices_a)} vertices")
    print(f"  {name_b}: {len(vertices_b)} vertices")

    # Run experiment
    exp_results = run_experiment(
        vertices_a, vertices_b,
        faces_a, faces_b,
        model, args.k_pred, args.k_robust, args.num_eigenvectors, device,
        name_a, name_b,
    )

    # Visualize
    if not args.no_vis:
        visualize_correspondences(
            vertices_a, vertices_b,
            faces_a, faces_b,
            exp_results['correspondences'],
            exp_results['results'],
            name_a, name_b,
        )


if __name__ == "__main__":
    main()