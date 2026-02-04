#!/usr/bin/env python3
"""
SMAL Model Viewer with Polyscope

Loads the SMAL (Skinned Multi-Animal Linear) model directly from the pickle file,
implements Linear Blend Skinning (LBS) from scratch (no chumpy/smpl_webuser needed),
and visualizes random shapes and poses interactively with Polyscope.

Usage:
    python smal_viewer.py --model path/to/smal_CVPR2017.pkl
    python smal_viewer.py --model path/to/smal_CVPR2017.pkl --data path/to/smal_CVPR2017_data.pkl
"""

import argparse
import pickle
import sys
import types
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Fake chumpy module — allows pickle to load SMAL/SMPL files that contain
# chumpy objects without actually installing chumpy.
# ---------------------------------------------------------------------------

def _install_fake_chumpy():
    """Install minimal fake chumpy modules so pickle can deserialize."""
    if 'chumpy' in sys.modules:
        return  # real chumpy is installed, nothing to do

    # Create fake modules
    chumpy = types.ModuleType('chumpy')
    chumpy_ch = types.ModuleType('chumpy.ch')
    chumpy_ch_ops = types.ModuleType('chumpy.ch_ops')
    chumpy_reordering = types.ModuleType('chumpy.reordering')
    chumpy_utils = types.ModuleType('chumpy.utils')
    chumpy_logic = types.ModuleType('chumpy.logic')

    # All chumpy types become a simple container that stores state.
    # The to_numpy() helper later extracts array data from these.
    class FakeCh:
        """Stand-in for any chumpy object during pickle deserialization."""
        def __init__(self, *args, **kwargs):
            self._data = None
            if args:
                self._data = args[0] if isinstance(args[0], np.ndarray) else None

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
                # chumpy stores array data under various keys
                for key in ['x', 'a', '_data']:
                    if key in state and isinstance(state[key], np.ndarray):
                        self._data = state[key]
                        break
            elif isinstance(state, (list, tuple)):
                if len(state) > 0 and isinstance(state[0], np.ndarray):
                    self._data = state[0]

        @property
        def r(self):
            """Return numpy array (mimics chumpy .r property)."""
            if self._data is not None:
                return np.array(self._data)
            # Try to find any array-like attribute
            for attr in dir(self):
                if attr.startswith('_'):
                    continue
                val = getattr(self, attr, None)
                if isinstance(val, np.ndarray):
                    return val
            return np.array([])

        def __array__(self, dtype=None):
            arr = self.r
            if dtype is not None:
                return arr.astype(dtype)
            return arr

    # Register all modules
    sys.modules['chumpy'] = chumpy
    sys.modules['chumpy.ch'] = chumpy_ch
    sys.modules['chumpy.ch_ops'] = chumpy_ch_ops
    sys.modules['chumpy.reordering'] = chumpy_reordering
    sys.modules['chumpy.utils'] = chumpy_utils
    sys.modules['chumpy.logic'] = chumpy_logic

    # Attach submodules as attributes (required for attribute access)
    chumpy.ch = chumpy_ch
    chumpy.ch_ops = chumpy_ch_ops
    chumpy.reordering = chumpy_reordering
    chumpy.utils = chumpy_utils
    chumpy.logic = chumpy_logic

    # Register the fake classes
    chumpy.Ch = FakeCh
    chumpy_ch.Ch = FakeCh
    chumpy_ch_ops.add = FakeCh
    chumpy_ch_ops.subtract = FakeCh
    chumpy_ch_ops.multiply = FakeCh
    chumpy_ch_ops.divide = FakeCh
    chumpy_reordering.transpose = FakeCh
    chumpy_reordering.concatenate = FakeCh

_install_fake_chumpy()


# ---------------------------------------------------------------------------
# Numpy conversion helpers (handle chumpy objects in pickle)
# ---------------------------------------------------------------------------

def to_numpy(x):
    """Convert chumpy/FakeCh arrays or other objects to numpy."""
    if hasattr(x, 'r'):
        # chumpy or FakeCh object — .r gives the numpy array
        return np.array(x.r)
    elif isinstance(x, np.ndarray):
        return x
    elif sp.issparse(x):
        return x  # keep sparse matrices as-is
    elif hasattr(x, '__array__'):
        return np.array(x)
    else:
        return np.array(x)


# ---------------------------------------------------------------------------
# Rodrigues rotation: axis-angle → rotation matrix
# ---------------------------------------------------------------------------

def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle rotations to rotation matrices.

    Args:
        axis_angle: (N, 3) axis-angle vectors

    Returns:
        R: (N, 3, 3) rotation matrices
    """
    N = axis_angle.shape[0]
    theta = np.linalg.norm(axis_angle, axis=1, keepdims=True)  # (N, 1)
    theta = np.clip(theta, 1e-8, None)  # avoid division by zero

    k = axis_angle / theta  # (N, 3) unit axis
    K = np.zeros((N, 3, 3))
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    sin_theta = np.sin(theta)[:, :, np.newaxis]  # (N, 1, 1)
    cos_theta = np.cos(theta)[:, :, np.newaxis]  # (N, 1, 1)

    I = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    R = I + sin_theta * K + (1 - cos_theta) * np.einsum('nij,njk->nik', K, K)

    return R


def with_zeros(x: np.ndarray) -> np.ndarray:
    """Append [0, 0, 0, 1] row to a 3x4 matrix to make it 4x4."""
    return np.vstack([x, np.array([0.0, 0.0, 0.0, 1.0])])


# ---------------------------------------------------------------------------
# SMAL Model
# ---------------------------------------------------------------------------

class SMALModel:
    """
    Standalone SMAL model loader and forward kinematics.
    No dependency on chumpy or smpl_webuser.
    """

    def __init__(self, model_path: str, data_path: Optional[str] = None):
        print(f"Loading SMAL model from: {model_path}")
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')

        # Extract and convert all parameters
        self.v_template = to_numpy(params['v_template']).astype(np.float64)  # (V, 3)
        self.shapedirs = to_numpy(params['shapedirs']).astype(np.float64)    # (V, 3, num_betas)
        self.posedirs = to_numpy(params['posedirs']).astype(np.float64)      # (V, 3, num_pose_params)
        self.J_regressor = params['J_regressor']                              # (J, V) sparse
        if sp.issparse(self.J_regressor):
            self.J_regressor = self.J_regressor.toarray().astype(np.float64)
        else:
            self.J_regressor = to_numpy(self.J_regressor).astype(np.float64)
        self.weights = to_numpy(params['weights']).astype(np.float64)        # (V, J)
        self.kintree_table = to_numpy(params['kintree_table']).astype(np.int64)  # (2, J)
        self.faces = to_numpy(params['f']).astype(np.int32)                  # (F, 3)

        self.num_vertices = self.v_template.shape[0]
        self.num_joints = self.kintree_table.shape[1]
        self.num_betas = self.shapedirs.shape[2]
        self.num_pose_params = self.num_joints * 3

        # Build parent index map
        self.parent = {}
        for i in range(1, self.num_joints):
            self.parent[i] = self.kintree_table[0, i]

        print(f"  Vertices: {self.num_vertices}")
        print(f"  Faces: {self.faces.shape[0]}")
        print(f"  Joints: {self.num_joints}")
        print(f"  Shape betas: {self.num_betas}")

        # Load family/cluster data if available
        self.cluster_means = None
        self.toys_betas = None
        if data_path is not None:
            print(f"Loading SMAL data from: {data_path}")
            with open(data_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            if 'cluster_means' in data:
                self.cluster_means = [to_numpy(b) for b in data['cluster_means']]
                self.family_names = [
                    'Felidae (cats)', 'Canidae (dogs)', 'Equidae (horses)',
                    'Bovidae (cows)', 'Hippopotamidae (hippos)'
                ]
                print(f"  Loaded {len(self.cluster_means)} family clusters")
            if 'toys_betas' in data:
                self.toys_betas = [to_numpy(b) for b in data['toys_betas']]
                print(f"  Loaded {len(self.toys_betas)} toy shapes")

    def forward(
        self,
        betas: Optional[np.ndarray] = None,
        pose: Optional[np.ndarray] = None,
        trans: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run SMAL forward kinematics (Linear Blend Skinning).

        Args:
            betas: (num_betas,) shape coefficients (or None for mean shape)
            pose: (num_joints * 3,) pose as axis-angle per joint (or None for rest pose)
            trans: (3,) global translation (or None for origin)

        Returns:
            vertices: (V, 3) posed vertex positions
        """
        if betas is None:
            betas = np.zeros(self.num_betas)
        if pose is None:
            pose = np.zeros(self.num_pose_params)
        if trans is None:
            trans = np.zeros(3)

        betas = betas[:self.num_betas]  # Truncate if too many

        # 1. Shape blend shapes: v_shaped = template + shapedirs @ betas
        v_shaped = self.v_template + self.shapedirs.dot(betas)

        # 2. Joint locations from shaped vertices
        J = self.J_regressor.dot(v_shaped)  # (num_joints, 3)

        # 3. Pose blend shapes
        pose_vec = pose.reshape(-1, 3)  # (num_joints, 3)
        R = rodrigues(pose_vec)  # (num_joints, 3, 3)

        # Pose feature: (R[1:] - I).ravel()
        I_cube = np.broadcast_to(np.eye(3), (self.num_joints - 1, 3, 3))
        lrotmin = (R[1:] - I_cube).ravel()

        v_posed = v_shaped + self.posedirs.dot(lrotmin)

        # 4. Forward kinematics: chain joint transformations
        G = np.empty((self.num_joints, 4, 4))
        G[0] = with_zeros(np.hstack([R[0], J[0].reshape(3, 1)]))

        for i in range(1, self.num_joints):
            G[i] = G[self.parent[i]].dot(
                with_zeros(np.hstack([
                    R[i],
                    (J[i] - J[self.parent[i]]).reshape(3, 1)
                ]))
            )

        # 5. Remove rest-pose joint translations
        G_rest = np.matmul(
            G,
            np.hstack([J, np.zeros((self.num_joints, 1))]).reshape(self.num_joints, 4, 1)
        )
        G_packed = np.zeros((self.num_joints, 4, 4))
        G_packed[:, :, 3] = G_rest.squeeze(-1)
        G = G - G_packed

        # 6. Linear blend skinning
        T = np.tensordot(self.weights, G, axes=[[1], [0]])  # (V, 4, 4)

        v_homo = np.hstack([v_posed, np.ones((self.num_vertices, 1))])  # (V, 4)
        v_final = np.einsum('vij,vj->vi', T, v_homo)[:, :3]  # (V, 3)

        # 7. Apply global translation
        v_final += trans

        return v_final


# ---------------------------------------------------------------------------
# Polyscope Viewer
# ---------------------------------------------------------------------------

class SMALViewer:
    """Interactive SMAL model viewer with Polyscope."""

    def __init__(self, model: SMALModel):
        self.model = model

        # Current parameters
        self.betas = np.zeros(model.num_betas)
        self.pose = np.zeros(model.num_pose_params)
        self.trans = np.zeros(3)

        # UI state
        self.beta_scale = 1.0
        self.pose_scale = 0.0
        self.selected_family = -1  # -1 = custom
        self.randomize_shape = False
        self.randomize_pose = False
        self.show_wireframe = False

        # Initialize polyscope
        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("shadow_only")
        ps.set_background_color((0.1, 0.1, 0.12))

        # Compute initial vertices
        vertices = model.forward(self.betas, self.pose, self.trans)

        # Register mesh
        self.mesh = ps.register_surface_mesh(
            "SMAL", vertices, model.faces, smooth_shade=True
        )
        self.mesh.set_color((0.75, 0.55, 0.35))

        # Set callback
        ps.set_user_callback(self._ui_callback)

    def _update_mesh(self):
        """Recompute and update the mesh."""
        vertices = self.model.forward(self.betas, self.pose, self.trans)
        self.mesh.update_vertex_positions(vertices)

    def _ui_callback(self):
        """ImGui UI for controlling SMAL parameters."""
        psim.SetNextWindowSize((350, 600))
        psim.SetNextWindowPos((10, 10))

        opened = psim.Begin("SMAL Controller", True)
        if not opened:
            psim.End()
            return

        changed = False

        # --- Family presets ---
        if self.model.cluster_means is not None:
            psim.Text("Animal Family Presets:")
            for i, name in enumerate(self.model.family_names):
                if psim.Button(name):
                    self.betas = np.zeros(self.model.num_betas)
                    betas_cluster = self.model.cluster_means[i]
                    n = min(len(betas_cluster), self.model.num_betas)
                    self.betas[:n] = betas_cluster[:n]
                    self.selected_family = i
                    changed = True

            if psim.Button("Mean Shape (T-pose)"):
                self.betas = np.zeros(self.model.num_betas)
                self.pose = np.zeros(self.model.num_pose_params)
                self.selected_family = -1
                changed = True

            psim.Separator()

        # --- Random shape ---
        psim.Text("Random Generation:")

        _, self.beta_scale = psim.SliderFloat(
            "Shape variance", self.beta_scale, 0.0, 3.0
        )

        if psim.Button("Random Shape"):
            self.betas = np.random.randn(self.model.num_betas) * self.beta_scale
            self.selected_family = -1
            changed = True

        _, self.pose_scale = psim.SliderFloat(
            "Pose variance", self.pose_scale, 0.0, 0.5
        )

        if psim.Button("Random Pose"):
            self.pose = np.random.randn(self.model.num_pose_params) * self.pose_scale
            # Keep root rotation small
            self.pose[:3] *= 0.1
            changed = True

        if psim.Button("Random Shape + Pose"):
            self.betas = np.random.randn(self.model.num_betas) * self.beta_scale
            self.pose = np.random.randn(self.model.num_pose_params) * self.pose_scale
            self.pose[:3] *= 0.1
            self.selected_family = -1
            changed = True

        if psim.Button("Reset to T-pose"):
            self.pose = np.zeros(self.model.num_pose_params)
            changed = True

        psim.Separator()

        # --- Manual beta sliders (first 5) ---
        psim.Text("Shape Parameters (first 5):")
        for i in range(min(5, self.model.num_betas)):
            c, val = psim.SliderFloat(f"beta_{i}", self.betas[i], -3.0, 3.0)
            if c:
                self.betas[i] = val
                changed = True

        psim.Separator()

        # --- Info ---
        psim.Text(f"Vertices: {self.model.num_vertices}")
        psim.Text(f"Faces: {self.model.faces.shape[0]}")
        psim.Text(f"Joints: {self.model.num_joints}")
        if self.selected_family >= 0:
            psim.Text(f"Family: {self.model.family_names[self.selected_family]}")

        psim.End()

        if changed:
            self._update_mesh()

    def show(self):
        """Show the interactive viewer."""
        ps.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SMAL Model Viewer with Polyscope",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to smal_CVPR2017.pkl")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to smal_CVPR2017_data.pkl (optional, for family clusters)")
    args = parser.parse_args()

    model = SMALModel(args.model, args.data)
    viewer = SMALViewer(model)
    viewer.show()


if __name__ == "__main__":
    main()