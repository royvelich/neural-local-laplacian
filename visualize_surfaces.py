#!/usr/bin/env python3
"""
Enhanced Surface Visualization with Model Prediction and Mean Curvature Display

This script visualizes synthetic surface datasets and optionally compares:
1. Ground-truth analytic normals at surface centers
2. Predicted normals from trained SurfaceTransformerModule models
3. Mean curvature values at the origin (now printed to screen)

Usage:
    python visualize_surfaces.py                    # Original functionality
    python visualize_surfaces.py ckpt_path=model.ckpt  # With model prediction comparison
"""

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import polyscope as ps
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, replace
import torch
import torch.nn.functional as F
from pathlib import Path

# Import model class
from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from torch_geometric.data import Data, Batch


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    vector_scale: float
    point_radius: float
    param_radius: float
    surface_spacing_factor: float
    enable_mesh: bool
    enable_point_cloud: bool
    enable_parametrization: bool
    enable_normals: bool
    enable_differential_geometry: bool
    enable_model_prediction: bool
    smooth_shade: bool
    edge_width: float
    mesh_scalar_colormap: str
    pointcloud_scalar_colormap: str
    pointcloud_color: Tuple[float, float, float] = (0.0, 0.8, 0.0)
    normal_display_scale: float = 4.0
    origin_indicator_scale: float = 3.0
    reference_frame_point_radius: float = 0.02
    reference_frame_line_radius: float = 0.02
    normal_origin_point_radius: float = 0.02


class ColorPalette:
    PRINCIPAL_V1 = (1.0, 0.0, 0.0)
    PRINCIPAL_V2 = (0.0, 0.0, 1.0)
    GRAD_MEAN_CURVATURE = (0.0, 1.0, 0.0)
    GRAD_GAUSSIAN_CURVATURE = (1.0, 1.0, 0.0)
    NORMALS = (0.0, 0.0, 1.0)
    PREDICTED_NORMALS = (1.0, 0.5, 0.0)
    DEFAULT_VECTOR = (0.5, 0.5, 0.5)

    @classmethod
    def get_vector_color(cls, vector_name: str) -> Tuple[float, float, float]:
        base_name = vector_name.replace('_2d', '').replace('_3d', '')
        color_map = {
            'v1': cls.PRINCIPAL_V1, 'v2': cls.PRINCIPAL_V2,
            'grad_H': cls.GRAD_MEAN_CURVATURE, 'grad_K': cls.GRAD_GAUSSIAN_CURVATURE,
            'normals': cls.NORMALS, 'gt_normals': cls.NORMALS,
            'predicted_normals': cls.PREDICTED_NORMALS
        }
        return color_map.get(base_name, color_map.get(vector_name, cls.DEFAULT_VECTOR))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.where(norms > 0, vectors / norms, vectors)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def load_trained_model(ckpt_path: Path, device: torch.device) -> LaplacianTransformerModule:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    try:
        print(f"Loading model checkpoint from: {ckpt_path}")
        model = LaplacianTransformerModule.load_from_checkpoint(str(ckpt_path), map_location=device)
        model.eval()
        model.to(device)
        print(f"[OK] Model loaded successfully on {device}")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Input dim: {model._input_dim}")
        print(f"   Model dim: {model._d_model}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {ckpt_path}: {e}")


def prepare_surface_for_model(surface_data: Data, device: torch.device) -> Batch:
    surface_data = surface_data.to(device)
    return Batch.from_data_list([surface_data])


def predict_normal_from_patch(model, surface_data, device):
    with torch.no_grad():
        batch = prepare_surface_for_model(surface_data, device)
        forward_result = model.forward(batch)
        stiffness_weights = forward_result['stiffness_weights']
        areas = forward_result['areas']
        attention_mask = forward_result['attention_mask']
        stiffness_weights = stiffness_weights.masked_fill(~attention_mask, 0.0)
        positions = batch.pos
        num_points = positions.shape[0]
        weights = stiffness_weights[0, :num_points]
        stiffness_sum = (weights.unsqueeze(-1) * positions).sum(dim=0)
        predicted_mean_curvature_vector = stiffness_sum / areas[0]
        predicted_normal = F.normalize(predicted_mean_curvature_vector.unsqueeze(0), p=2, dim=1).squeeze(0)
        return predicted_normal, weights, predicted_mean_curvature_vector


def visualize_patch(points, faces, name, vis_config):
    return ps.register_surface_mesh(name=name, vertices=points, faces=faces,
                                     smooth_shade=vis_config.smooth_shade, edge_width=vis_config.edge_width)

def visualize_point_cloud(points, name, vis_config, enabled=False):
    return ps.register_point_cloud(name=name, points=points, radius=vis_config.point_radius,
                                    enabled=enabled, color=vis_config.pointcloud_color)


def add_reference_frame(scale: float = 1.0, vis_config=None):
    point_radius = vis_config.reference_frame_point_radius if vis_config else 0.02
    line_radius = vis_config.reference_frame_line_radius if vis_config else 0.01
    origin = np.array([0.0, 0.0, 0.0])
    x_axis = np.array([scale, 0.0, 0.0])
    y_axis = np.array([0.0, scale, 0.0])
    z_axis = np.array([0.0, 0.0, scale])
    axes_points = np.array([origin, x_axis, origin, y_axis, origin, z_axis])
    frame_cloud = ps.register_point_cloud("Reference Frame", axes_points, radius=point_radius, enabled=True)
    axis_colors = np.array([[0.5,0.5,0.5],[1,0,0],[0.5,0.5,0.5],[0,1,0],[0.5,0.5,0.5],[0,0,1]])
    frame_cloud.add_color_quantity("axis_colors", axis_colors, enabled=True)
    try:
        ps.register_curve_network("X-axis", np.array([origin, x_axis]), np.array([[0,1]]), color=(1,0,0), radius=line_radius, enabled=True)
        ps.register_curve_network("Y-axis", np.array([origin, y_axis]), np.array([[0,1]]), color=(0,1,0), radius=line_radius, enabled=True)
        ps.register_curve_network("Z-axis", np.array([origin, z_axis]), np.array([[0,1]]), color=(0,0,1), radius=line_radius, enabled=True)
    except Exception as e:
        print(f"Note: Could not create axis lines: {e}")


# Sentinel index for the hybrid view (mesh from patch 0 + point cloud from patch 1)
_HYBRID_IDX = -1


class SurfaceVisualizer:
    def __init__(self, config, vis_config=None, trained_model=None, device=None, data_module=None):
        self.config = config
        self.vis_config = vis_config or VisualizationConfig()
        self.color_palette = ColorPalette()
        self.trained_model = trained_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.surface_metrics = []
        self.data_module = data_module
        self._current_surfaces = None
        self._current_surface_names = None
        self._diff_geom_at_origin_only = self._get_initial_origin_only_setting()
        self._include_origin_in_grid = self._get_initial_include_origin_setting()
        self._patch_idx = 0
        self._show_mesh = True
        self._show_pointcloud = False
        self._show_gt_normal_origin = True
        self._show_pred_normal_origin = True
        self._show_gt_normals_all = False
        self._prediction_cache = {}
        self._show_reference_frame = False
        self._gt_normal_color = list(ColorPalette.NORMALS)
        self._pred_normal_color = list(ColorPalette.PREDICTED_NORMALS)
        self._gt_normal_length = 8.5
        self._pred_normal_length = 8.5
        self._normal_radius = self.vis_config.normal_origin_point_radius
        self._point_radius = self.vis_config.point_radius
        self._surface_color = [0.3, 0.3, 0.3]
        self._origin_dot_color = [1.0, 0.0, 0.0]
        self._edge_width = 0.0  # toggled in screenshots
        self._probe_idx = 0    # selected test/probe function index
        self._show_probe_scalar = False  # toggle probe function coloring
        self._probe_color_mode = 1  # 0 = h, 1 = δh, 2 = Δ_LB(h)
        self._probe_symmetric_cmap = False  # if True, use vminmax=(-vmax, vmax)
        self._show_ground_shadow = False
        # GT surface gradient at origin (from test_func_gradients), tied to probe index
        self._show_gt_gradient_origin = False
        self._gt_gradient_color = [0.0, 1.0, 1.0]  # cyan
        self._gt_gradient_length = 8.5
        self._gt_gradient_width = self.vis_config.normal_origin_point_radius
        self._gt_gradient_normalize = False
        # GT surface gradient at all points (from test_func_gradients_all_points)
        self._show_gt_gradients_all = False
        self._gt_gradient_all_color = [0.0, 1.0, 1.0]  # cyan
        self._gt_gradient_all_length = 8.5
        self._gt_gradient_all_width = self.vis_config.normal_origin_point_radius * 0.5
        self._gt_gradient_all_normalize = False
        # Per-target visibility for the all-points gradient field.
        # The master toggle above is on/off; these choose where the vectors
        # attach when it is on. Both True = current default behavior (appear
        # on both structures); turning both off hides the field entirely.
        self._gt_gradients_all_on_mesh = True
        self._gt_gradients_all_on_cloud = True

    def _has_hybrid_option(self):
        return self._current_surfaces is not None and len(self._current_surfaces) >= 2

    def _get_initial_origin_only_setting(self):
        try:
            return getattr(self.config.data_module.train_dataset_specification.dataset, 'diff_geom_at_origin_only', False)
        except (AttributeError, KeyError):
            return False

    def _get_initial_include_origin_setting(self):
        try:
            return getattr(self.config.data_module.train_dataset_specification.dataset, 'include_origin_in_grid', False)
        except (AttributeError, KeyError):
            return False

    @property
    def is_diff_geom_at_origin_only(self):
        return self._diff_geom_at_origin_only

    def _get_train_dataset(self):
        if self.data_module is None:
            return None
        try:
            return self.data_module._train_dataset_specification.dataset
        except AttributeError:
            return None

    def toggle_diff_geom_at_origin_only(self):
        dataset = self._get_train_dataset()
        if dataset is None:
            print("Cannot toggle: no dataset available")
            return self._diff_geom_at_origin_only
        self._diff_geom_at_origin_only = not self._diff_geom_at_origin_only
        dataset._diff_geom_at_origin_only = self._diff_geom_at_origin_only
        print(f"Toggled diff_geom_at_origin_only to: {self._diff_geom_at_origin_only}")
        self._regenerate_and_visualize()
        return self._diff_geom_at_origin_only

    def toggle_include_origin_in_grid(self):
        dataset = self._get_train_dataset()
        if dataset is None:
            print("Cannot toggle: no dataset available")
            return self._include_origin_in_grid
        self._include_origin_in_grid = not self._include_origin_in_grid
        dataset._include_origin_in_grid = self._include_origin_in_grid
        print(f"Toggled include_origin_in_grid to: {self._include_origin_in_grid}")
        self._regenerate_and_visualize()
        return self._include_origin_in_grid

    def _regenerate_and_visualize(self):
        dataset = self._get_train_dataset()
        if dataset is None:
            return
        dataset.reset_rng()
        print("Regenerating surfaces...")
        new_surfaces = dataset._generate_surfaces()
        surface_names = self._get_surface_names(new_surfaces)
        self.visualize_surface_set(new_surfaces, surface_names)
        print("Regeneration complete!")

    def _get_surface_names(self, surfaces):
        names = []
        try:
            grid_samplers = self.config.data_module.train_dataset_specification.dataset.grid_samplers
            for i, gs in enumerate(grid_samplers):
                sampler_type = gs._target_.split('.')[-1]
                if sampler_type == 'RegularGridSampler':
                    names.append(f'Regular Grid ({gs.num_points} points)')
                elif sampler_type == 'RandomGridSampler':
                    npr = gs.num_points_range
                    if isinstance(npr, (list, tuple)) and len(npr) == 2:
                        names.append(f'Random Grid ({npr[0]}-{npr[1]} points)')
                    else:
                        names.append(f'Random Grid ({npr} points)')
                else:
                    names.append(f'Surface {i + 1}')
        except (AttributeError, KeyError):
            names = [f'Surface {i + 1}' for i in range(len(surfaces))]
        return names

    def _get_origin_position(self, surface, translation):
        if hasattr(surface, 'origin_idx') and hasattr(surface, 'pos'):
            return to_numpy(surface.pos[surface.origin_idx.item():surface.origin_idx.item()+1]) + translation
        elif hasattr(surface, 'origin_pos'):
            return to_numpy(surface.origin_pos) + translation
        else:
            return np.array([[0.0, 0.0, 0.0]]) + translation

    def _extract_surface_data(self, surface):
        pos = to_numpy(surface.pos)
        face = to_numpy(surface.face).T
        if hasattr(surface, 'normal'):
            normals = to_numpy(surface.normal)
            if self.is_diff_geom_at_origin_only and normals.shape[0] == 1:
                normals = np.broadcast_to(normals, (pos.shape[0], 3)).copy()
        else:
            normals = np.array([[0.0, 0.0, 1.0]] * pos.shape[0])
        return pos, face, normals

    def _extract_differential_geometry(self, surface):
        def safe_extract_and_broadcast(attr_name):
            if not hasattr(surface, attr_name):
                return None
            value = to_numpy(getattr(surface, attr_name))
            if self.is_diff_geom_at_origin_only and value.shape[0] == 1:
                num_points = surface.pos.shape[0]
                if value.ndim == 1:
                    value = np.broadcast_to(value, (num_points,)).copy()
                else:
                    value = np.broadcast_to(value, (num_points,) + value.shape[1:]).copy()
            return value
        return {
            'vectors_3d': {k: safe_extract_and_broadcast(k) for k in ['v1_3d', 'v2_3d', 'grad_H_3d', 'grad_K_3d']},
            'vectors_2d': {k: safe_extract_and_broadcast(k) for k in ['v1_2d', 'v2_2d', 'grad_H_2d', 'grad_K_2d']},
            'scalars': {
                'Mean Curvature (H)': safe_extract_and_broadcast('H'),
                'Gaussian Curvature (K)': safe_extract_and_broadcast('K'),
                'Principal Curvature (k1)': safe_extract_and_broadcast('k1'),
                'Principal Curvature (k2)': safe_extract_and_broadcast('k2'),
            }
        }

    def _add_vector_quantities(self, structure, surface, structure_type="default"):
        if not self.vis_config.enable_differential_geometry or not hasattr(surface, 'H'):
            return
        diff_geom = self._extract_differential_geometry(surface)
        for name, vectors in diff_geom['vectors_3d'].items():
            if vectors is not None:
                structure.add_vector_quantity(name=name, values=normalize_vectors(vectors) * self.vis_config.vector_scale,
                                             enabled=False, color=self.color_palette.get_vector_color(name), vectortype="ambient")
        colormap = self.vis_config.mesh_scalar_colormap if structure_type == "mesh" else self.vis_config.pointcloud_scalar_colormap
        for name, scalars in diff_geom['scalars'].items():
            if scalars is not None:
                structure.add_scalar_quantity(name=name, values=scalars, enabled=False, cmap=colormap)

    def _add_normals_to_structure(self, structure, normals):
        if not self.vis_config.enable_normals:
            return
        structure.add_vector_quantity("normals", normals * self.vis_config.vector_scale,
                                     enabled=True, color=self.color_palette.get_vector_color("normals"), vectortype="ambient")

    def _get_num_probes(self, surface):
        """Return number of probe functions on this surface, or 0."""
        if hasattr(surface, 'test_func_deltas') and surface.test_func_deltas is not None:
            return surface.test_func_deltas.shape[-1]
        return 0

    def _add_probe_coloring(self, structure, surface, colormap='coolwarm'):
        """Add probe function scalar coloring to a Polyscope structure."""
        if not self._show_probe_scalar:
            return
        num_probes = self._get_num_probes(surface)
        if num_probes == 0:
            return
        probe_idx = min(self._probe_idx, num_probes - 1)

        # DEBUG: dump full probe state for the selected probe
        struct_name = getattr(structure, 'name', '?') if hasattr(structure, 'name') else '?'
        try:
            struct_name = structure.get_name()
        except Exception:
            pass
        has_h = hasattr(surface, 'test_func_values') and surface.test_func_values is not None
        has_lb_all = hasattr(surface, 'test_func_lb_all_points') and surface.test_func_lb_all_points is not None
        deltas_all = to_numpy(surface.test_func_deltas)
        d_p = deltas_all[:, probe_idx]
        print(f"\n[probe_debug] structure='{struct_name}'  probe={probe_idx}/{num_probes}  mode={self._probe_color_mode}")
        print(f"  K={d_p.shape[0]}  has_h={has_h}  has_lb_all={has_lb_all}")
        print(f"  δh: min={d_p.min():.6f}  max={d_p.max():.6f}  mean={d_p.mean():.6f}  std={d_p.std():.6f}")
        if has_h:
            h_p = to_numpy(surface.test_func_values)[:, probe_idx]
            print(f"   h: min={h_p.min():.6f}  max={h_p.max():.6f}  mean={h_p.mean():.6f}  std={h_p.std():.6f}")
            # Find origin index
            origin_idx_val = None
            if hasattr(surface, 'origin_idx'):
                try:
                    origin_idx_val = int(surface.origin_idx.item())
                except Exception:
                    pass
            if origin_idx_val is not None and origin_idx_val < d_p.shape[0]:
                print(f"   h(origin)={h_p[origin_idx_val]:.6f}  δh(origin)={d_p[origin_idx_val]:.6f}  "
                      f"diff(h-δh)@origin={h_p[origin_idx_val] - d_p[origin_idx_val]:.6f}")
            # Per-vertex difference (should equal h(origin) for all vertices if dataset is correct)
            diff = h_p - d_p
            print(f"   (h - δh): min={diff.min():.6f}  max={diff.max():.6f}  "
                  f"mean={diff.mean():.6f}  std={diff.std():.6f}  (should be constant = h(origin))")
        if hasattr(surface, 'test_func_laplacians') and surface.test_func_laplacians is not None:
            lb = to_numpy(surface.test_func_laplacians).flatten()
            if probe_idx < len(lb):
                print(f"   Δ_LB(h_{probe_idx})@origin = {lb[probe_idx]:.6f}")

        if self._probe_color_mode == 0:
            # h(p_j) — raw function value at every vertex
            if has_h:
                vals = to_numpy(surface.test_func_values)  # (K, P)
                values = vals[:, probe_idx]
                label = f"h_{probe_idx}"
            else:
                print("  [probe_debug] mode=h but test_func_values missing — returning")
                return  # not available
        elif self._probe_color_mode == 1:
            # Delta: h(p_j) - h(p_i)
            values = d_p
            label = f"δh_{probe_idx}"
        else:
            # Δ_LB(h) at every vertex
            if has_lb_all:
                lb_all = to_numpy(surface.test_func_lb_all_points)  # (K, P)
                values = lb_all[:, probe_idx]
                label = f"Δ_LB(h_{probe_idx})"
            else:
                print("  [probe_debug] mode=Δ_LB but test_func_lb_all_points missing — returning")
                return  # not available

        print(f"  → coloring: label='{label}'  values: min={values.min():.6f}  max={values.max():.6f}  "
              f"mean={values.mean():.6f}")
        if self._probe_symmetric_cmap:
            vmax = float(np.abs(values).max())
            if vmax < 1e-12:
                vmax = 1e-12
            structure.add_scalar_quantity(
                name=label, values=values,
                enabled=True, cmap=colormap, vminmax=(-vmax, vmax))
        else:
            structure.add_scalar_quantity(
                name=label, values=values,
                enabled=True, cmap=colormap)

    def _render_gt_gradient_arrow(self, surface, name):
        """Render the analytic GT surface gradient at the origin for the selected probe."""
        if not self._show_gt_gradient_origin:
            return
        if not hasattr(surface, 'test_func_gradients') or surface.test_func_gradients is None:
            return
        num_probes = self._get_num_probes(surface)
        if num_probes == 0:
            return
        probe_idx = min(self._probe_idx, num_probes - 1)

        grads = to_numpy(surface.test_func_gradients)
        # Storage convention from the dataset is (1, P, 3) — a leading batch dim
        # added by .unsqueeze(0) so PyG batching stacks across surfaces. In the
        # viz we index a single surface, so drop the leading dim if present.
        if grads.ndim == 3 and grads.shape[0] == 1:
            grads = grads[0]  # (P, 3)
        if probe_idx >= grads.shape[0]:
            return
        grad_vec = grads[probe_idx]  # (3,)

        if self._gt_gradient_normalize:
            n = np.linalg.norm(grad_vec)
            if n > 1e-12:
                grad_vec = grad_vec / n

        origin_3d = self._get_origin_position(surface, np.zeros(3))
        gt_grad_scale = self.vis_config.vector_scale * self._gt_gradient_length

        cloud = ps.register_point_cloud(
            f"{name} - GT Gradient (probe {probe_idx})",
            origin_3d, radius=self._normal_radius,
            color=tuple(self._origin_dot_color), enabled=True)
        cloud.add_vector_quantity(
            f"GT Gradient h_{probe_idx}",
            grad_vec.reshape(1, 3) * gt_grad_scale,
            enabled=True, color=tuple(self._gt_gradient_color),
            radius=self._gt_gradient_width, vectortype="ambient")

    def _add_gt_gradient_all_to_structure(self, structure, surface, structure_type: str = "mesh"):
        """Attach per-vertex GT gradient vectors for the selected probe to a structure.

        Args:
            structure: Polyscope structure (mesh or point cloud) to attach to.
            surface: The surface Data object holding the GT gradient field.
            structure_type: 'mesh' or 'pointcloud'. Checked against the per-target
                visibility flags (_gt_gradients_all_on_mesh / _gt_gradients_all_on_cloud)
                so the field can be shown on one, both, or neither target.
        """
        if not self._show_gt_gradients_all:
            return
        # Per-target gating: skip if this structure type is disabled.
        if structure_type == "mesh" and not self._gt_gradients_all_on_mesh:
            return
        if structure_type == "pointcloud" and not self._gt_gradients_all_on_cloud:
            return
        if not hasattr(surface, 'test_func_gradients_all_points') or \
                surface.test_func_gradients_all_points is None:
            return
        num_probes = self._get_num_probes(surface)
        if num_probes == 0:
            return
        probe_idx = min(self._probe_idx, num_probes - 1)

        grads_all = to_numpy(surface.test_func_gradients_all_points)  # (K, P, 3)
        if probe_idx >= grads_all.shape[1]:
            return
        vectors = grads_all[:, probe_idx, :]  # (K, 3)

        if self._gt_gradient_all_normalize:
            norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
            vectors = np.where(norms > 1e-12, vectors / norms, vectors)

        scale = self.vis_config.vector_scale * self._gt_gradient_all_length
        structure.add_vector_quantity(
            f"GT Gradient h_{probe_idx} (all pts)",
            vectors * scale, enabled=True,
            color=tuple(self._gt_gradient_all_color),
            radius=self._gt_gradient_all_width,
            vectortype="ambient")

    def _add_origin_indicator(self, surface, name, translation):
        if not self.is_diff_geom_at_origin_only:
            return
        origin_3d = self._get_origin_position(surface, translation)
        try:
            ind = ps.register_point_cloud(f"{name} - Origin", origin_3d,
                                          radius=self.vis_config.point_radius * self.vis_config.origin_indicator_scale, enabled=True)
            ind.add_color_quantity("origin_color", np.array([[1.0, 0.0, 1.0]]), enabled=True)
        except Exception as e:
            print(f"Could not add origin indicator: {e}")

    def _extract_mean_curvature_at_origin(self, surface):
        if not hasattr(surface, 'H'):
            return None
        H_tensor = surface.H.detach().cpu()
        if hasattr(surface, 'origin_idx'):
            idx = surface.origin_idx.item()
            if idx < H_tensor.numel():
                return H_tensor.flatten()[idx].item()
        if H_tensor.numel() == 1:
            return H_tensor.item()
        return H_tensor.flatten()[0].item()

    def _setup_ui_callback(self):
        def ui_callback():
            import polyscope.imgui as psim
            needs_rerender = False

            # ── Patch selection ──────────────────────────────────────
            psim.Text("Patch Selection")
            psim.Separator()
            if self._current_surfaces and self._current_surface_names:
                combo_names = list(self._current_surface_names)
                if self._has_hybrid_option():
                    combo_names.append("Hybrid (mesh 0 + points 1)")
                combo_idx = len(self._current_surface_names) if self._patch_idx == _HYBRID_IDX else self._patch_idx
                changed, new_combo_idx = psim.Combo("Patch", combo_idx, combo_names)
                if changed:
                    if self._has_hybrid_option() and new_combo_idx == len(self._current_surface_names):
                        self._patch_idx = _HYBRID_IDX
                    else:
                        self._patch_idx = new_combo_idx
                    needs_rerender = True
                if self._patch_idx == _HYBRID_IDX:
                    psim.Text(f"  Mesh pts: {self._current_surfaces[0].pos.shape[0]}, Cloud pts: {self._current_surfaces[1].pos.shape[0]}")
                else:
                    psim.Text(f"  Points: {self._current_surfaces[self._patch_idx].pos.shape[0]}")
            psim.Text("")

            # ── Display mode ─────────────────────────────────────────
            psim.Text("Display Mode")
            psim.Separator()
            if self._patch_idx != _HYBRID_IDX:
                c1, v1 = psim.Checkbox("Show Mesh", self._show_mesh)
                if c1: self._show_mesh = v1; needs_rerender = True
                c2, v2 = psim.Checkbox("Show Point Cloud", self._show_pointcloud)
                if c2: self._show_pointcloud = v2; needs_rerender = True
            else:
                psim.TextColored((0.7, 0.7, 0.7, 1.0), "  (Hybrid: mesh + cloud always on)")
            c3, v3 = psim.Checkbox("Show Reference Frame", self._show_reference_frame)
            if c3: self._show_reference_frame = v3; needs_rerender = True
            c3b, v3b = psim.Checkbox("Ground Shadow", self._show_ground_shadow)
            if c3b:
                self._show_ground_shadow = v3b
                ps.set_ground_plane_mode("shadow_only" if v3b else "none")
            c4, v4 = psim.ColorEdit3("Surface Color", self._surface_color)
            if c4: self._surface_color = list(v4); needs_rerender = True
            c5, v5 = psim.SliderFloat("Point Size", self._point_radius, v_min=0.001, v_max=0.05)
            if c5: self._point_radius = v5; needs_rerender = True
            c5b, v5b = psim.SliderFloat("Edge Width", self._edge_width, v_min=0.0, v_max=2.0)
            if c5b: self._edge_width = v5b; needs_rerender = True
            psim.Text("")

            # ── Normal display ───────────────────────────────────────
            psim.Text("Normal Display")
            psim.Separator()
            c6, v6 = psim.Checkbox("GT Normal at Origin", self._show_gt_normal_origin)
            if c6: self._show_gt_normal_origin = v6; needs_rerender = True
            if self.trained_model is not None:
                c7, v7 = psim.Checkbox("Predicted Normal at Origin", self._show_pred_normal_origin)
                if c7: self._show_pred_normal_origin = v7; needs_rerender = True
            c8, v8 = psim.Checkbox("GT Normals (all points)", self._show_gt_normals_all)
            if c8: self._show_gt_normals_all = v8; needs_rerender = True
            psim.Text("")
            c9, v9 = psim.SliderFloat("GT Normal Length", self._gt_normal_length, v_min=0.1, v_max=20.0)
            if c9: self._gt_normal_length = v9; needs_rerender = True
            c9b, v9b = psim.SliderFloat("Pred Normal Length", self._pred_normal_length, v_min=0.1, v_max=20.0)
            if c9b: self._pred_normal_length = v9b; needs_rerender = True
            c10, v10 = psim.SliderFloat("Normal Width", self._normal_radius, v_min=0.001, v_max=0.05)
            if c10: self._normal_radius = v10; needs_rerender = True
            c11, v11 = psim.ColorEdit3("GT Normal Color", self._gt_normal_color)
            if c11: self._gt_normal_color = list(v11); needs_rerender = True
            if self.trained_model is not None:
                c12, v12 = psim.ColorEdit3("Pred Normal Color", self._pred_normal_color)
                if c12: self._pred_normal_color = list(v12); needs_rerender = True
            c13, v13 = psim.ColorEdit3("Origin Dot Color", self._origin_dot_color)
            if c13: self._origin_dot_color = list(v13); needs_rerender = True
            psim.Text("")

            # ── Dataset settings ─────────────────────────────────────
            psim.Text("Dataset Settings")
            psim.Separator()
            c14, _ = psim.Checkbox("Diff Geom at Origin Only", self._diff_geom_at_origin_only)
            if c14: self.toggle_diff_geom_at_origin_only()
            c15, _ = psim.Checkbox("Include Origin in Grid", self._include_origin_in_grid)
            if c15: self.toggle_include_origin_in_grid()
            psim.Text("")

            # ── Probe function display ──────────────────────────────
            cur_surface_idx = 1 if self._patch_idx == _HYBRID_IDX else self._patch_idx
            num_probes = 0
            if self._current_surfaces and cur_surface_idx < len(self._current_surfaces):
                num_probes = self._get_num_probes(self._current_surfaces[cur_surface_idx])
            if num_probes > 0:
                psim.Text("Probe Functions")
                psim.Separator()
                cp1, vp1 = psim.Checkbox("Show Probe Coloring", self._show_probe_scalar)
                if cp1: self._show_probe_scalar = vp1; needs_rerender = True

                surface = self._current_surfaces[cur_surface_idx]
                has_gt_grad = (hasattr(surface, 'test_func_gradients')
                               and surface.test_func_gradients is not None)
                has_gt_grad_all = (hasattr(surface, 'test_func_gradients_all_points')
                                   and surface.test_func_gradients_all_points is not None)
                if has_gt_grad:
                    cg, vg = psim.Checkbox("Show GT Gradient at Origin", self._show_gt_gradient_origin)
                    if cg: self._show_gt_gradient_origin = vg; needs_rerender = True
                if has_gt_grad_all:
                    cga, vga = psim.Checkbox("Show GT Gradient (all points)", self._show_gt_gradients_all)
                    if cga: self._show_gt_gradients_all = vga; needs_rerender = True
                elif has_gt_grad:
                    psim.TextColored((0.7, 0.7, 0.7, 1.0), "  (All-points gradient: enable compute_gradients_all_points)")

                # Shared probe index slider (drives coloring + origin/all GT gradients)
                if self._show_probe_scalar or self._show_gt_gradient_origin or self._show_gt_gradients_all:
                    cp2, vp2 = psim.SliderInt("Probe Index", self._probe_idx, v_min=0, v_max=num_probes - 1)
                    if cp2: self._probe_idx = vp2; needs_rerender = True

                if (self._show_gt_gradient_origin and has_gt_grad):
                    cgn, vgn = psim.Checkbox("Normalize GT Gradient (origin)", self._gt_gradient_normalize)
                    if cgn: self._gt_gradient_normalize = vgn; needs_rerender = True
                    cgl, vgl = psim.SliderFloat("GT Gradient (origin) Length", self._gt_gradient_length,
                                                 v_min=0.1, v_max=20.0)
                    if cgl: self._gt_gradient_length = vgl; needs_rerender = True
                    cgw, vgw = psim.SliderFloat("GT Gradient (origin) Width", self._gt_gradient_width,
                                                 v_min=0.001, v_max=0.05)
                    if cgw: self._gt_gradient_width = vgw; needs_rerender = True
                    cgc, vgc = psim.ColorEdit3("GT Gradient (origin) Color", self._gt_gradient_color)
                    if cgc: self._gt_gradient_color = list(vgc); needs_rerender = True

                if (self._show_gt_gradients_all and has_gt_grad_all):
                    # Per-target visibility sub-controls (mesh / point cloud).
                    # Without these, the vector field was attached to both structures
                    # at once, which displayed the arrows twice when both were shown.
                    psim.Text("  Display target:")
                    psim.SameLine()
                    cgtm, vgtm = psim.Checkbox("On Mesh", self._gt_gradients_all_on_mesh)
                    if cgtm: self._gt_gradients_all_on_mesh = vgtm; needs_rerender = True
                    psim.SameLine()
                    cgtp, vgtp = psim.Checkbox("On Point Cloud", self._gt_gradients_all_on_cloud)
                    if cgtp: self._gt_gradients_all_on_cloud = vgtp; needs_rerender = True

                    cgna, vgna = psim.Checkbox("Normalize GT Gradient (all pts)", self._gt_gradient_all_normalize)
                    if cgna: self._gt_gradient_all_normalize = vgna; needs_rerender = True
                    cgla, vgla = psim.SliderFloat("GT Gradient (all pts) Length", self._gt_gradient_all_length,
                                                    v_min=0.1, v_max=20.0)
                    if cgla: self._gt_gradient_all_length = vgla; needs_rerender = True
                    cgwa, vgwa = psim.SliderFloat("GT Gradient (all pts) Width", self._gt_gradient_all_width,
                                                    v_min=0.001, v_max=0.05)
                    if cgwa: self._gt_gradient_all_width = vgwa; needs_rerender = True
                    cgca, vgca = psim.ColorEdit3("GT Gradient (all pts) Color", self._gt_gradient_all_color)
                    if cgca: self._gt_gradient_all_color = list(vgca); needs_rerender = True

                if self._show_gt_gradient_origin and has_gt_grad:
                    pidx = min(self._probe_idx, num_probes - 1)
                    grads = to_numpy(surface.test_func_gradients)
                    # Stored as (1, P, 3) from the dataset — drop the leading
                    # batch dim for single-surface indexing here.
                    if grads.ndim == 3 and grads.shape[0] == 1:
                        grads = grads[0]  # (P, 3)
                    if pidx < grads.shape[0]:
                        g = grads[pidx]
                        psim.Text(f"  ∇h_{pidx}(origin) = [{float(g[0]):.4f}, {float(g[1]):.4f}, {float(g[2]):.4f}]  |g|={float(np.linalg.norm(g)):.4f}")

                if self._show_probe_scalar:
                    # Color mode selector
                    color_modes = ["h (function value)", "δh (function delta)", "Δ_LB(h) (analytic)"]
                    has_h = hasattr(surface, 'test_func_values') and surface.test_func_values is not None
                    has_lb_all = hasattr(surface, 'test_func_lb_all_points') and surface.test_func_lb_all_points is not None

                    # Clamp current mode to an available option
                    if self._probe_color_mode == 0 and not has_h:
                        self._probe_color_mode = 1
                    if self._probe_color_mode == 2 and not has_lb_all:
                        self._probe_color_mode = 1

                    cm_changed, cm_new = psim.Combo("Color Mode", self._probe_color_mode, color_modes)
                    if cm_changed:
                        # Reject picks for modes whose data isn't available
                        if cm_new == 0 and not has_h:
                            pass
                        elif cm_new == 2 and not has_lb_all:
                            pass
                        else:
                            self._probe_color_mode = cm_new
                            needs_rerender = True

                    if not has_h:
                        psim.TextColored((0.7, 0.7, 0.7, 1.0), "  (h coloring: test_func_values missing)")
                    if not has_lb_all:
                        psim.TextColored((0.7, 0.7, 0.7, 1.0), "  (Δ_LB coloring: enable compute_lb_all_points)")

                    csym, vsym = psim.Checkbox("Symmetric Colormap (centered at 0)", self._probe_symmetric_cmap)
                    if csym: self._probe_symmetric_cmap = vsym; needs_rerender = True

                    # Show Δ_LB value at origin for selected probe
                    pidx = min(self._probe_idx, num_probes - 1)
                    if hasattr(surface, 'test_func_laplacians') and surface.test_func_laplacians is not None:
                        lb_vals = to_numpy(surface.test_func_laplacians).flatten()
                        pidx_lb = min(pidx, len(lb_vals) - 1)
                        psim.Text(f"  Δ_LB(h_{pidx_lb}) at origin = {lb_vals[pidx_lb]:.6f}")

                    # Show value ranges
                    if has_h:
                        vals = to_numpy(surface.test_func_values)
                        psim.Text(f"  h range: [{vals[:, pidx].min():.4f}, {vals[:, pidx].max():.4f}]")
                    deltas = to_numpy(surface.test_func_deltas)
                    psim.Text(f"  δh range: [{deltas[:, pidx].min():.4f}, {deltas[:, pidx].max():.4f}]")
                    if has_lb_all:
                        lb_all = to_numpy(surface.test_func_lb_all_points)
                        psim.Text(f"  Δ_LB range: [{lb_all[:, pidx].min():.4f}, {lb_all[:, pidx].max():.4f}]")
                psim.Text("")

            # ── Surface metrics ──────────────────────────────────────
            psim.Text("Surface Metrics")
            psim.Separator()
            metrics_idx = 1 if self._patch_idx == _HYBRID_IDX else self._patch_idx
            if self._current_surfaces and metrics_idx < len(self.surface_metrics):
                metrics = self.surface_metrics[metrics_idx]
                H = metrics.get('mean_curvature_at_origin')
                psim.Text(f"GT Mean Curvature: {H:.6f}" if H is not None else "GT Mean Curvature: N/A")
                pred = metrics.get('prediction_metrics')
                if pred:
                    pred_H = pred.get('predicted_mean_curvature')
                    if pred_H is not None:
                        psim.Text(f"Pred Mean Curvature: {pred_H:.6f}")
                        if H is not None:
                            psim.Text(f"Curvature Error: {abs(pred_H - H):.6f}")
                    cosim = pred.get('cosine_similarity')
                    if cosim is not None: psim.Text(f"Normal Cosine Sim: {cosim:.4f}")
                    ang = pred.get('angular_error')
                    if ang is not None: psim.Text(f"Angular Error: {ang:.2f} deg")

            psim.Text("")
            psim.Separator()
            if psim.Button("Save All Screenshots"):
                self._save_all_screenshots()

            if needs_rerender:
                self._render()

        ps.set_user_callback(ui_callback)

    def visualize_surface_set(self, surfaces, surface_names):
        self._current_surfaces = surfaces
        self._current_surface_names = surface_names
        self._patch_idx = 0
        self.surface_metrics = []
        self._prediction_cache = {}
        for i, (name, surface) in enumerate(zip(surface_names, surfaces)):
            metric = {
                'name': name,
                'num_points': surface.pos.shape[0] if hasattr(surface, 'pos') else None,
                'mean_curvature_at_origin': self._extract_mean_curvature_at_origin(surface),
            }
            if (self.trained_model is not None and self.vis_config.enable_model_prediction and hasattr(surface, 'normal')):
                try:
                    pred_normal, pred_weights, pred_mcv = predict_normal_from_patch(self.trained_model, surface, self.device)
                    self._prediction_cache[i] = (pred_normal, pred_weights, pred_mcv)
                    gt_normal = to_numpy(surface.normal)
                    gt_at_origin = gt_normal[0] if gt_normal.shape[0] == 1 else (gt_normal[surface.origin_idx.item()] if hasattr(surface, 'origin_idx') else gt_normal[0])
                    pred_H = torch.norm(pred_mcv, p=2).item()
                    gt_t = torch.from_numpy(gt_at_origin).float().to(self.device)
                    cosim = torch.dot(gt_t, pred_normal.float().to(self.device)).item()
                    ang_err = np.arccos(np.abs(np.clip(cosim, -1.0, 1.0))) * 180 / np.pi
                    metric['prediction_metrics'] = {
                        'cosine_similarity': cosim, 'angular_error': ang_err,
                        'gt_normal': gt_at_origin, 'predicted_normal': to_numpy(pred_normal),
                        'predicted_mean_curvature': pred_H,
                    }
                    print(f"  [{name}] pred H={pred_H:.4f}, cos={cosim:.4f}, ang_err={ang_err:.2f}deg")
                except Exception as e:
                    print(f"  [{name}] prediction failed: {e}")
            self.surface_metrics.append(metric)
        self._setup_ui_callback()
        self._render()

    def _render(self):
        ps.remove_all_structures()
        if self._show_reference_frame:
            add_reference_frame(vis_config=self.vis_config)
        if not self._current_surfaces:
            return
        if self._patch_idx == _HYBRID_IDX:
            self._render_hybrid()
        else:
            self._render_single(self._patch_idx)

    def _render_single(self, idx):
        surface = self._current_surfaces[idx]
        name = self._current_surface_names[idx]
        pos, face, normals = self._extract_surface_data(surface)
        if self._show_mesh:
            mesh = ps.register_surface_mesh(f"{name} - Mesh", pos, face,
                smooth_shade=self.vis_config.smooth_shade, edge_width=self._edge_width,
                color=tuple(self._surface_color), transparency=0.5)
            mesh.set_material("clay")
            self._add_vector_quantities(mesh, surface, "mesh")
            if self._show_gt_normals_all:
                self._add_normals_to_structure(mesh, normals)
            self._add_probe_coloring(mesh, surface, self.vis_config.mesh_scalar_colormap)
            self._add_gt_gradient_all_to_structure(mesh, surface, structure_type="mesh")
        if self._show_pointcloud:
            cloud = ps.register_point_cloud(f"{name} - Point Cloud", pos, radius=self._point_radius, enabled=True)
            cloud.set_color(tuple(c * 0.5 for c in self._surface_color))
            self._add_vector_quantities(cloud, surface, "pointcloud")
            if self._show_gt_normals_all:
                self._add_normals_to_structure(cloud, normals)
            self._add_probe_coloring(cloud, surface, self.vis_config.pointcloud_scalar_colormap)
            self._add_gt_gradient_all_to_structure(cloud, surface, structure_type="pointcloud")
        self._add_origin_indicator(surface, name, np.zeros(3))
        self._render_origin_normals(surface, name, pred_cache_idx=idx)
        self._render_gt_gradient_arrow(surface, name)

    def _render_hybrid(self):
        surf_mesh = self._current_surfaces[0]
        surf_pts = self._current_surfaces[1]
        name_mesh = self._current_surface_names[0]
        name_pts = self._current_surface_names[1]
        pos_mesh, face_mesh, normals_mesh = self._extract_surface_data(surf_mesh)
        pos_pts, _, normals_pts = self._extract_surface_data(surf_pts)

        # Smooth mesh from regular grid (surface 0)
        mesh = ps.register_surface_mesh(f"Hybrid Mesh ({name_mesh})", pos_mesh, face_mesh,
            smooth_shade=self.vis_config.smooth_shade, edge_width=self._edge_width,
            color=tuple(self._surface_color), transparency=0.5)
        mesh.set_material("clay")
        self._add_vector_quantities(mesh, surf_mesh, "mesh")
        if self._show_gt_normals_all:
            self._add_normals_to_structure(mesh, normals_mesh)
        self._add_probe_coloring(mesh, surf_mesh, self.vis_config.mesh_scalar_colormap)
        self._add_gt_gradient_all_to_structure(mesh, surf_mesh, structure_type="mesh")

        # Point cloud from downsampled grid (surface 1)
        cloud = ps.register_point_cloud(f"Hybrid Points ({name_pts})", pos_pts,
            radius=self._point_radius, enabled=True)
        cloud.set_color(tuple(c * 0.5 for c in self._surface_color))
        self._add_vector_quantities(cloud, surf_pts, "pointcloud")
        if self._show_gt_normals_all:
            self._add_normals_to_structure(cloud, normals_pts)
        self._add_probe_coloring(cloud, surf_pts, self.vis_config.pointcloud_scalar_colormap)
        self._add_gt_gradient_all_to_structure(cloud, surf_pts, structure_type="pointcloud")

        # Origin + normals from downsampled patch (what model sees)
        self._add_origin_indicator(surf_pts, "Hybrid", np.zeros(3))
        self._render_origin_normals(surf_pts, "Hybrid", pred_cache_idx=1)
        self._render_gt_gradient_arrow(surf_pts, "Hybrid")

    def _render_origin_normals(self, surface, name, pred_cache_idx=None):
        translation = np.zeros(3)
        if self._show_gt_normal_origin and hasattr(surface, 'normal'):
            origin_3d = self._get_origin_position(surface, translation)
            gt_normal = to_numpy(surface.normal)
            gt_at_origin = gt_normal[0] if gt_normal.shape[0] == 1 else (gt_normal[surface.origin_idx.item()] if hasattr(surface, 'origin_idx') else gt_normal[0])
            gt_scale = self.vis_config.vector_scale * self._gt_normal_length
            gt_cloud = ps.register_point_cloud(f"{name} - GT Normal", origin_3d,
                radius=self._normal_radius, color=tuple(self._origin_dot_color), enabled=True)
            gt_cloud.add_vector_quantity("GT Normal", gt_at_origin.reshape(1, 3) * gt_scale,
                enabled=True, color=tuple(self._gt_normal_color), radius=self._normal_radius, vectortype="ambient")

        cache_key = pred_cache_idx if pred_cache_idx is not None else self._patch_idx
        if self._show_pred_normal_origin and cache_key in self._prediction_cache:
            pred_normal, _, _ = self._prediction_cache[cache_key]
            pred_np = to_numpy(pred_normal)
            origin_3d = self._get_origin_position(surface, translation)
            pred_scale = self.vis_config.vector_scale * self._pred_normal_length
            pred_cloud = ps.register_point_cloud(f"{name} - Pred Normal", origin_3d,
                radius=self._normal_radius, color=tuple(self._origin_dot_color), enabled=True)
            pred_cloud.add_vector_quantity("Predicted Normal", pred_np.reshape(1, 3) * pred_scale,
                enabled=True, color=tuple(self._pred_normal_color), radius=self._normal_radius, vectortype="ambient")

    def _save_all_screenshots(self):
        from datetime import datetime
        if not self._current_surfaces:
            print("No surfaces loaded.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"surface_patch_screenshots_{timestamp}")
        out_dir.mkdir(exist_ok=True)
        saved = (self._patch_idx, self._show_mesh, self._show_pointcloud,
                 self._show_gt_normal_origin, self._show_pred_normal_origin,
                 self._edge_width)
        has_pred = bool(self._prediction_cache)

        # Build patch index list: individual patches + hybrid
        patch_indices = list(range(len(self._current_surfaces)))
        if self._has_hybrid_option():
            patch_indices.append(_HYBRID_IDX)

        display_modes = [("surface", True, False), ("pointcloud", False, True), ("surface_and_pointcloud", True, True)]
        hybrid_display_modes = [("hybrid", True, True)]
        normal_modes = [("no_normals", False, False), ("gt_normal", True, False)]
        if has_pred:
            normal_modes += [("pred_normal", False, True), ("gt_and_pred", True, True)]

        # Edge styles for mesh-visible screenshots
        edge_styles = [("no_edges", 0.0), ("edges", 1.0)]

        # Count total screenshots
        total = 0
        for pidx in patch_indices:
            modes = hybrid_display_modes if pidx == _HYBRID_IDX else display_modes
            for _, show_mesh, _ in modes:
                n_edge = len(edge_styles) if show_mesh else 1
                total += n_edge * len(normal_modes)

        count = 0
        print(f"\nSaving {total} screenshots to {out_dir}/...")

        for pidx in patch_indices:
            if pidx == _HYBRID_IDX:
                patch_label = "hybrid"
                modes = hybrid_display_modes
            else:
                patch_label = self._current_surface_names[pidx].replace(" ", "_").replace("(", "").replace(")", "")
                modes = display_modes
            for disp_label, show_mesh, show_pc in modes:
                # Determine edge variations for this display mode
                cur_edge_styles = edge_styles if show_mesh else [("", 0.0)]
                for edge_label, edge_w in cur_edge_styles:
                    for norm_label, show_gt, show_pred in normal_modes:
                        self._patch_idx = pidx
                        self._show_mesh = show_mesh
                        self._show_pointcloud = show_pc
                        self._show_gt_normal_origin = show_gt
                        self._show_pred_normal_origin = show_pred
                        self._edge_width = edge_w
                        self._render()
                        if edge_label:
                            filename = f"{patch_label}_{disp_label}_{edge_label}_{norm_label}.png"
                        else:
                            filename = f"{patch_label}_{disp_label}_{norm_label}.png"
                        ps.screenshot(str(out_dir / filename), transparent_bg=True)
                        count += 1
                        print(f"  [{count}/{total}] {filename}")

        (self._patch_idx, self._show_mesh, self._show_pointcloud,
         self._show_gt_normal_origin, self._show_pred_normal_origin,
         self._edge_width) = saved
        self._render()
        print(f"Done! {total} screenshots saved to {out_dir}/")


def setup_polyscope():
    ps.init()
    ps.set_up_dir("z_up")
    ps.look_at(camera_location=[2.4, 2, 3.9], target=[0, 0, 0])
    ps.set_ground_plane_mode("none")
    ps.set_shadow_blur_iters(20)
    ps.set_shadow_darkness(0.15)
    ps.set_SSAA_factor(4)
    ps.set_background_color((0.0, 0.0, 0.0))
    # 'pretty' enables proper depth-sorted alpha blending so transparent meshes
    # don't make opaque arrows/point-clouds appear translucent.
    try:
        ps.set_transparency_mode("pretty")
    except Exception:
        pass


def create_custom_visualization_config(**kwargs):
    defaults = dict(vector_scale=0.1, point_radius=0.01, param_radius=0.002, surface_spacing_factor=2.5,
                    enable_mesh=True, enable_point_cloud=True, enable_parametrization=True,
                    enable_normals=True, enable_differential_geometry=True, enable_model_prediction=True,
                    smooth_shade=True, edge_width=0.0, mesh_scalar_colormap='coolwarm',
                    pointcloud_scalar_colormap='coolwarm', pointcloud_color=(0.0, 0.8, 0.0))
    defaults.update(kwargs)
    return VisualizationConfig(**defaults)


@hydra.main(version_base="1.2", config_path="visualization_config")
def main(cfg: DictConfig) -> None:
    ckpt_path = getattr(cfg, 'ckpt_path', None)
    pl.seed_everything(cfg.globals.seed)
    data_module = hydra.utils.instantiate(cfg.data_module)
    data_loader = data_module.train_dataloader()
    setup_polyscope()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    trained_model = None
    if ckpt_path:
        try:
            trained_model = load_trained_model(Path(ckpt_path), device)
            print(f"[OK] Successfully loaded model from {ckpt_path}")
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")

    vis_config = create_custom_visualization_config(
        vector_scale=0.15, point_radius=0.008, surface_spacing_factor=2.0,
        enable_model_prediction=(trained_model is not None))

    visualizer = SurfaceVisualizer(config=cfg, vis_config=vis_config,
                                    trained_model=trained_model, device=device, data_module=data_module)

    print(f"\n{'=' * 80}")
    print("SURFACE VISUALIZATION WITH OPTIONAL MODEL PREDICTION")
    print('=' * 80)

    for batch_idx, surfaces in enumerate(data_loader):
        print(f"\nProcessing batch {batch_idx + 1}")
        surface_names = visualizer._get_surface_names(surfaces)
        visualizer.visualize_surface_set(surfaces, surface_names)
        print(f"\n[OK] Batch {batch_idx + 1} complete! Close window to continue.")
        ps.show()


if __name__ == "__main__":
    main()