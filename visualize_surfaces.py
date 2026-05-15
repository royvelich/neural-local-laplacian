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

        # GT geodesic distances (attached by MongeSurfaceVariationalDataset
        # when compute_geodesics: true).  Surface carries:
        #   surface.geodesic_sources:   (S,)    training-vertex idx per source
        #   surface.geodesic_distances: (S, n)  exact GT distance per vertex
        self._show_gt_geodesics = False
        self._geodesic_source_idx = 0
        self._geodesic_cmap = 'viridis'
        self._geodesic_symmetric_cmap = False
        self._show_geodesic_source_marker = True
        self._geodesic_source_color = [1.0, 0.2, 0.2]   # red marker at source
        self._geodesic_source_radius_scale = 2.0         # multiplier on point_radius
        # Where to attach the geodesic scalar field. Same convention as the
        # all-points gradient: master toggle + per-target visibility.
        self._geodesics_on_mesh = True
        self._geodesics_on_cloud = True

        # PRED geodesics: run the loaded model on the surface and recover
        # the distance field via the heat method on its (areas, stiffness,
        # grad) heads.  Same source as GT for direct comparison.
        # Only available when self.trained_model is not None.
        self._show_pred_geodesics = False
        self._pred_geodesics_on_mesh = True
        self._pred_geodesics_on_cloud = True
        self._pred_geodesics_cmap = 'viridis'
        self._pred_geodesics_L_assembly = 'from_stiffness'
        self._pred_geodesics_t = None              # None ⇒ auto (A_total/n)
        self._pred_geodesics_t_scale = 1.0
        self._pred_geodesics_epsilon_reg = 1.0e-6
        # Cache: (surface_idx, source_idx, L_assembly, t, t_scale, eps_reg)
        # → np.ndarray of distances.  Cleared on every visualize_surface_set().
        self._pred_geodesics_cache: dict = {}
        # Forward-result cache (per surface_idx) — avoid re-running the model
        # when only the source / t changes.  Cleared on every batch.
        self._pred_forward_cache: dict = {}

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
        # Prefer the explicit per-vertex positions when present
        # (_VariationalSurfaceData stores them in vertex_pos; surface.pos
        # there is the stacked (n*k, 3) per-patch points, not vertex
        # positions, and would mis-align with vertex-level fields like
        # vertex_normals / geodesic_distances).  Patch-level datasets
        # don't have vertex_pos, so we fall back to surface.pos.
        if hasattr(surface, 'vertex_pos') and surface.vertex_pos is not None:
            pos = to_numpy(surface.vertex_pos)
        else:
            pos = to_numpy(surface.pos)
        # `face` is present on the patch-level synthetic datasets but not on
        # _VariationalSurfaceData (the variational dataset emits a vertex
        # cloud, not a triangulation).  Fall back to a 2D Delaunay over the
        # xy projection — exact triangulation of the parameter domain for
        # Monge patches up to position_noise_std.  None if Delaunay fails
        # (e.g. < 3 vertices or all-collinear); the renderer then skips the
        # mesh structure for this surface.
        face = None
        surf_face = getattr(surface, 'face', None)
        if surf_face is not None:
            face = to_numpy(surf_face).T
        elif pos.shape[0] >= 3:
            try:
                from scipy.spatial import Delaunay
                face = Delaunay(pos[:, :2]).simplices.astype(np.int32)
            except Exception as e:
                print(f"  [!] Falling back to point-cloud only "
                      f"(Delaunay failed: {e})")
                face = None
        if hasattr(surface, 'normal'):
            normals = to_numpy(surface.normal)
            if self.is_diff_geom_at_origin_only and normals.shape[0] == 1:
                normals = np.broadcast_to(normals, (pos.shape[0], 3)).copy()
        elif hasattr(surface, 'vertex_normals') and surface.vertex_normals is not None:
            # Variational dataset stores normals under vertex_normals.
            normals = to_numpy(surface.vertex_normals)
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

    # ── Probe-field name resolution ──────────────────────────────────
    # MongeSurfaceVariationalDataset attaches probe data under different
    # names than the patch-level synthetic datasets:
    #   patch level                       variational
    #   ─────────────────────────────────  ───────────────────────────────
    #   test_func_deltas                   (not present — no center concept)
    #   test_func_values                   test_func_values            (same)
    #   test_func_gradients_all_points     test_func_gradients_at_vertices
    #   test_func_lb_all_points            test_func_laplacians_at_vertices
    # These helpers return the first present field from a fallback chain
    # so the rest of the viz code can stay branch-free.
    @staticmethod
    def _probe_attr(surface, *names):
        """Return the first present, non-None attribute among ``names``."""
        for n in names:
            v = getattr(surface, n, None)
            if v is not None:
                return v
        return None

    def _get_test_func_deltas(self, surface):
        return self._probe_attr(surface, 'test_func_deltas')

    def _get_test_func_values(self, surface):
        return self._probe_attr(surface, 'test_func_values')

    def _get_test_func_gradients_all(self, surface):
        return self._probe_attr(
            surface, 'test_func_gradients_all_points',
            'test_func_gradients_at_vertices')

    def _get_test_func_lb_all(self, surface):
        return self._probe_attr(
            surface, 'test_func_lb_all_points',
            'test_func_laplacians_at_vertices')

    def _get_num_probes(self, surface):
        """Return number of probe functions on this surface, or 0.

        Reads `test_func_deltas` (patch-level) or falls back to
        `test_func_values` (variational, which has no center-relative
        deltas).  Either has shape (..., P) so the trailing axis is P.
        """
        for getter in (self._get_test_func_deltas, self._get_test_func_values):
            v = getter(surface)
            if v is not None:
                return v.shape[-1]
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
        vals_t = self._get_test_func_values(surface)
        lb_all_t = self._get_test_func_lb_all(surface)
        deltas_t = self._get_test_func_deltas(surface)
        has_h = vals_t is not None
        has_lb_all = lb_all_t is not None
        has_deltas = deltas_t is not None

        d_p = None
        if has_deltas:
            deltas_all = to_numpy(deltas_t)
            d_p = deltas_all[:, probe_idx]
        print(f"\n[probe_debug] structure='{struct_name}'  probe={probe_idx}/{num_probes}  mode={self._probe_color_mode}")
        print(f"  has_h={has_h}  has_lb_all={has_lb_all}  has_deltas={has_deltas}")
        if has_deltas:
            print(f"  δh: min={d_p.min():.6f}  max={d_p.max():.6f}  mean={d_p.mean():.6f}  std={d_p.std():.6f}")
        if has_h:
            h_p = to_numpy(vals_t)[:, probe_idx]
            print(f"   h: min={h_p.min():.6f}  max={h_p.max():.6f}  mean={h_p.mean():.6f}  std={h_p.std():.6f}")
            # Find origin index
            origin_idx_val = None
            if hasattr(surface, 'origin_idx'):
                try:
                    origin_idx_val = int(surface.origin_idx.item())
                except Exception:
                    pass
            if has_deltas and origin_idx_val is not None and origin_idx_val < d_p.shape[0]:
                print(f"   h(origin)={h_p[origin_idx_val]:.6f}  δh(origin)={d_p[origin_idx_val]:.6f}  "
                      f"diff(h-δh)@origin={h_p[origin_idx_val] - d_p[origin_idx_val]:.6f}")
            if has_deltas:
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
                vals = to_numpy(vals_t)  # (K, P)
                values = vals[:, probe_idx]
                label = f"h_{probe_idx}"
            else:
                print("  [probe_debug] mode=h but test_func_values missing — returning")
                return  # not available
        elif self._probe_color_mode == 1:
            # Delta: h(p_j) - h(p_i) — patch-level only
            if not has_deltas:
                print("  [probe_debug] mode=δh but test_func_deltas missing "
                      "(variational dataset has no center-relative deltas) — returning")
                return
            values = d_p
            label = f"δh_{probe_idx}"
        else:
            # Δ_LB(h) at every vertex
            if has_lb_all:
                lb_all = to_numpy(lb_all_t)  # (K, P)
                values = lb_all[:, probe_idx]
                label = f"Δ_LB(h_{probe_idx})"
            else:
                print("  [probe_debug] mode=Δ_LB but per-vertex Laplacian field missing — returning")
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
        grads_t = self._get_test_func_gradients_all(surface)
        if grads_t is None:
            return
        num_probes = self._get_num_probes(surface)
        if num_probes == 0:
            return
        probe_idx = min(self._probe_idx, num_probes - 1)

        grads_all = to_numpy(grads_t)  # (K, P, 3)
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

    def _has_gt_geodesics(self, surface) -> bool:
        return (hasattr(surface, 'geodesic_distances')
                and surface.geodesic_distances is not None
                and hasattr(surface, 'geodesic_sources')
                and surface.geodesic_sources is not None)

    def _num_geodesic_sources(self, surface) -> int:
        if not self._has_gt_geodesics(surface):
            return 0
        return int(surface.geodesic_sources.shape[0])

    def _add_gt_geodesic_to_structure(self, structure, surface, structure_type: str = "mesh"):
        """Color a Polyscope structure by per-vertex GT geodesic distance.

        Reads ``surface.geodesic_distances`` (S, n) and
        ``surface.geodesic_sources`` (S,), attached by
        MongeSurfaceVariationalDataset when ``compute_geodesics: true``.
        Adds a scalar quantity named ``GT geodesic d(v_s, ·)`` whose
        values are the GT distance from the currently selected source
        vertex; selectable via ``Source idx`` slider in the UI.

        Args:
            structure: Polyscope mesh / point cloud.
            surface:   surface item carrying the geodesic GT fields.
            structure_type: 'mesh' or 'pointcloud'; checked against per-
                target visibility flags so the field can be shown on
                one, both, or neither target.
        """
        if not self._show_gt_geodesics or not self._has_gt_geodesics(surface):
            return
        if structure_type == "mesh" and not self._geodesics_on_mesh:
            return
        if structure_type == "pointcloud" and not self._geodesics_on_cloud:
            return

        S = self._num_geodesic_sources(surface)
        src_idx = min(self._geodesic_source_idx, S - 1)
        distances = to_numpy(surface.geodesic_distances[src_idx])  # (n,)
        # Defensive: scalar field length must match the structure's vertex count.
        # vertex_pos is the right reference on variational data (surface.pos is
        # the stacked per-patch tensor there).
        vertex_pos_src = (surface.vertex_pos
                          if hasattr(surface, 'vertex_pos') and surface.vertex_pos is not None
                          else surface.pos)
        if distances.shape[0] != to_numpy(vertex_pos_src).shape[0]:
            return
        src_vertex = int(to_numpy(surface.geodesic_sources[src_idx]))
        name = f"GT geodesic d(v_{src_vertex}, ·)"

        vmin = float(distances.min())
        vmax = float(distances.max())
        if self._geodesic_symmetric_cmap:
            bound = max(abs(vmin), abs(vmax))
            vrange = (-bound, bound)
        else:
            vrange = (vmin, vmax)
        structure.add_scalar_quantity(name=name, values=distances,
                                      enabled=True, cmap=self._geodesic_cmap,
                                      vminmax=vrange)

    def _render_gt_geodesic_source_marker(self, surface, name: str):
        """Place a small marker sphere at the currently selected geodesic source."""
        if not (self._show_gt_geodesics and self._show_geodesic_source_marker):
            return
        if not self._has_gt_geodesics(surface):
            return
        S = self._num_geodesic_sources(surface)
        src_idx = min(self._geodesic_source_idx, S - 1)
        src_vertex = int(to_numpy(surface.geodesic_sources[src_idx]))
        # Use vertex_pos on variational data — surface.pos is the stacked
        # per-patch tensor there and would mis-index.
        vertex_pos_src = (surface.vertex_pos
                          if hasattr(surface, 'vertex_pos') and surface.vertex_pos is not None
                          else surface.pos)
        pos = to_numpy(vertex_pos_src)
        if src_vertex >= pos.shape[0]:
            return
        marker = ps.register_point_cloud(
            f"{name} - Geodesic source v_{src_vertex}",
            pos[src_vertex:src_vertex + 1],
            radius=self._point_radius * self._geodesic_source_radius_scale,
            enabled=True)
        marker.set_color(tuple(self._geodesic_source_color))

    # ── Predicted geodesics (via loaded model + heat method) ─────────────

    def _pred_geodesics_cfg_key(self) -> tuple:
        """Cache key fragment for the current heat-method config."""
        return (self._pred_geodesics_L_assembly,
                self._pred_geodesics_t,
                self._pred_geodesics_t_scale,
                self._pred_geodesics_epsilon_reg)

    def _compute_forward_for_surface(self, surface_idx: int, surface):
        """Run model.forward on the variational surface, cached.  Returns
        a dict with detached ``grad_coeffs``, ``areas``, ``stiffness_weights``
        (all on self.device, float dtype) and ``knn`` from the surface."""
        if surface_idx in self._pred_forward_cache:
            return self._pred_forward_cache[surface_idx]
        if self.trained_model is None:
            return None
        try:
            forward_batch = Batch.from_data_list([surface]).to(self.device)
            self.trained_model.eval()
            with torch.no_grad():
                fwd = self.trained_model(forward_batch)
            knn = surface.knn.to(self.device)
            out = {
                'grad_coeffs': fwd['grad_coeffs'].float(),
                'areas': fwd['areas'].float(),
                'stiffness_weights': fwd['stiffness_weights'].float(),
                'knn': knn,
            }
            self._pred_forward_cache[surface_idx] = out
            return out
        except Exception as e:
            print(f"  [!] Model forward failed on surface {surface_idx}: {e}")
            return None

    def _compute_pred_geodesic_distances(self, surface_idx: int, surface,
                                         source_vertex: int):
        """Heat-method distance field from ``source_vertex``.  Caches by
        (surface_idx, source_vertex, config_key) so flipping the source slider
        doesn't re-run the model."""
        key = (surface_idx, int(source_vertex)) + self._pred_geodesics_cfg_key()
        if key in self._pred_geodesics_cache:
            return self._pred_geodesics_cache[key]
        fwd = self._compute_forward_for_surface(surface_idx, surface)
        if fwd is None:
            return None
        try:
            from neural_local_laplacian.utils.laplacian_assembly import (
                assemble_from_stiffness_weights,
                assemble_diagonal_gram_laplacian,
            )
            from neural_local_laplacian.modules.losses import HeatMethodGeodesicLoss

            grad_coeffs = fwd['grad_coeffs']                          # (n, k, 3)
            areas = fwd['areas']                                      # (n,)
            stiffness = fwd['stiffness_weights']                      # (n, k)
            knn = fwd['knn']                                          # (n, k)
            n = areas.shape[0]
            if source_vertex < 0 or source_vertex >= n:
                return None

            if self._pred_geodesics_L_assembly == 'from_stiffness':
                L = assemble_from_stiffness_weights(stiffness, knn, areas=None)
            else:
                L = assemble_diagonal_gram_laplacian(grad_coeffs, knn, areas=None)

            device = L.device
            dtype = L.dtype
            eye = torch.eye(n, device=device, dtype=dtype)
            M_full = areas.unsqueeze(-1) * eye

            if self._pred_geodesics_t is None:
                t = self._pred_geodesics_t_scale * float(areas.sum().item()) / n
            else:
                t = float(self._pred_geodesics_t)

            A_heat = M_full + t * L
            A_poisson = L + self._pred_geodesics_epsilon_reg * M_full

            b_heat = L.new_zeros(n)
            b_heat[source_vertex] = 1.0
            u = torch.linalg.solve(A_heat, b_heat)
            grad_u = HeatMethodGeodesicLoss._per_vertex_gradient(grad_coeffs, knn, u)
            X = -grad_u / (grad_u.norm(dim=-1, keepdim=True) + 1e-8)
            rhs = HeatMethodGeodesicLoss._per_vertex_divergence(grad_coeffs, knn, X, areas)
            phi = torch.linalg.solve(A_poisson, rhs)
            phi = phi - phi[source_vertex]
            dist = phi.detach().cpu().numpy().astype(np.float32)
            self._pred_geodesics_cache[key] = dist
            return dist
        except Exception as e:
            print(f"  [!] Heat-method PRED solve failed: {e}")
            return None

    def _add_pred_geodesic_to_structure(self, structure, surface_idx: int,
                                        surface, structure_type: str = "mesh"):
        """Color a structure by the PREDICTED heat-method distance field."""
        if not (self._show_pred_geodesics and self._has_gt_geodesics(surface)):
            return
        if self.trained_model is None:
            return
        if structure_type == "mesh" and not self._pred_geodesics_on_mesh:
            return
        if structure_type == "pointcloud" and not self._pred_geodesics_on_cloud:
            return
        S = self._num_geodesic_sources(surface)
        if S == 0:
            return
        src_idx = min(self._geodesic_source_idx, S - 1)
        src_vertex = int(to_numpy(surface.geodesic_sources[src_idx]))
        distances = self._compute_pred_geodesic_distances(
            surface_idx, surface, src_vertex)
        if distances is None:
            return
        vertex_pos_src = (surface.vertex_pos
                          if hasattr(surface, 'vertex_pos') and surface.vertex_pos is not None
                          else surface.pos)
        if distances.shape[0] != to_numpy(vertex_pos_src).shape[0]:
            return
        name = f"PRED geodesic d(v_{src_vertex}, ·)"
        structure.add_scalar_quantity(
            name=name, values=distances, enabled=True,
            cmap=self._pred_geodesics_cmap)

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
                # `test_func_gradients` is the origin-only gradient — only
                # patch-level datasets compute it.  `*_all_points` /
                # `*_at_vertices` are the per-vertex variants (the variational
                # dataset only has the latter).
                has_gt_grad = (hasattr(surface, 'test_func_gradients')
                               and surface.test_func_gradients is not None)
                has_gt_grad_all = self._get_test_func_gradients_all(surface) is not None
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
                    vals_t = self._get_test_func_values(surface)
                    lb_all_t = self._get_test_func_lb_all(surface)
                    deltas_t = self._get_test_func_deltas(surface)
                    has_h = vals_t is not None
                    has_lb_all = lb_all_t is not None
                    has_deltas = deltas_t is not None

                    # Clamp current mode to an available option
                    if self._probe_color_mode == 0 and not has_h:
                        self._probe_color_mode = 1 if has_deltas else 2
                    if self._probe_color_mode == 1 and not has_deltas:
                        self._probe_color_mode = 0 if has_h else 2
                    if self._probe_color_mode == 2 and not has_lb_all:
                        self._probe_color_mode = 0 if has_h else 1

                    cm_changed, cm_new = psim.Combo("Color Mode", self._probe_color_mode, color_modes)
                    if cm_changed:
                        # Reject picks for modes whose data isn't available
                        if cm_new == 0 and not has_h:
                            pass
                        elif cm_new == 1 and not has_deltas:
                            pass
                        elif cm_new == 2 and not has_lb_all:
                            pass
                        else:
                            self._probe_color_mode = cm_new
                            needs_rerender = True

                    if not has_h:
                        psim.TextColored((0.7, 0.7, 0.7, 1.0), "  (h coloring: test_func_values missing)")
                    if not has_deltas:
                        psim.TextColored((0.7, 0.7, 0.7, 1.0),
                            "  (δh coloring: variational dataset has no center-relative deltas)")
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
                        vals = to_numpy(vals_t)
                        psim.Text(f"  h range: [{vals[:, pidx].min():.4f}, {vals[:, pidx].max():.4f}]")
                    if has_deltas:
                        deltas = to_numpy(deltas_t)
                        psim.Text(f"  δh range: [{deltas[:, pidx].min():.4f}, {deltas[:, pidx].max():.4f}]")
                    if has_lb_all:
                        lb_all = to_numpy(lb_all_t)
                        psim.Text(f"  Δ_LB range: [{lb_all[:, pidx].min():.4f}, {lb_all[:, pidx].max():.4f}]")
                psim.Text("")

            # ── GT geodesic distances ────────────────────────────────
            # Surface-level field attached by MongeSurfaceVariationalDataset
            # when compute_geodesics: true.  Only rendered when present.
            current_surface = (self._current_surfaces[self._patch_idx]
                               if (self._current_surfaces
                                   and self._patch_idx != _HYBRID_IDX
                                   and self._patch_idx < len(self._current_surfaces))
                               else None)
            if (current_surface is None and self._current_surfaces
                    and self._patch_idx == _HYBRID_IDX
                    and len(self._current_surfaces) >= 2):
                current_surface = self._current_surfaces[1]  # points surface in hybrid

            if current_surface is not None and self._has_gt_geodesics(current_surface):
                psim.Text("GT Geodesics")
                psim.Separator()
                cgg, vgg = psim.Checkbox("Show GT Geodesics", self._show_gt_geodesics)
                if cgg: self._show_gt_geodesics = vgg; needs_rerender = True

                S = self._num_geodesic_sources(current_surface)
                if self._show_gt_geodesics and S > 0:
                    if self._geodesic_source_idx >= S:
                        self._geodesic_source_idx = 0
                    if S > 1:
                        csi, vsi = psim.SliderInt("Source idx",
                                                  self._geodesic_source_idx,
                                                  v_min=0, v_max=S - 1)
                        if csi:
                            self._geodesic_source_idx = vsi
                            needs_rerender = True
                    src_vertex = int(to_numpy(
                        current_surface.geodesic_sources[self._geodesic_source_idx]))
                    psim.Text(f"  Source vertex: v_{src_vertex}  "
                              f"({self._geodesic_source_idx + 1}/{S})")
                    dists = to_numpy(
                        current_surface.geodesic_distances[self._geodesic_source_idx])
                    psim.Text(f"  d range: [{dists.min():.4f}, {dists.max():.4f}]")

                    # Per-target visibility (same idiom as the gradient field).
                    psim.Text("  Display target:")
                    psim.SameLine()
                    cgom, vgom = psim.Checkbox("On Mesh##geo", self._geodesics_on_mesh)
                    if cgom: self._geodesics_on_mesh = vgom; needs_rerender = True
                    psim.SameLine()
                    cgop, vgop = psim.Checkbox("On Point Cloud##geo",
                                               self._geodesics_on_cloud)
                    if cgop: self._geodesics_on_cloud = vgop; needs_rerender = True

                    cgsm, vgsm = psim.Checkbox("Show Source Marker",
                                               self._show_geodesic_source_marker)
                    if cgsm:
                        self._show_geodesic_source_marker = vgsm
                        needs_rerender = True
                    cgmc, vgmc = psim.ColorEdit3("Source Marker Color",
                                                 self._geodesic_source_color)
                    if cgmc: self._geodesic_source_color = list(vgmc); needs_rerender = True
                    cgms, vgms = psim.SliderFloat("Source Marker Size",
                                                  self._geodesic_source_radius_scale,
                                                  v_min=0.5, v_max=8.0)
                    if cgms:
                        self._geodesic_source_radius_scale = vgms
                        needs_rerender = True
                    csym_g, vsym_g = psim.Checkbox("Symmetric Cmap##geo",
                                                   self._geodesic_symmetric_cmap)
                    if csym_g:
                        self._geodesic_symmetric_cmap = vsym_g
                        needs_rerender = True

                # ── PRED geodesics (heat method on the loaded model) ──
                if self.trained_model is not None and S > 0:
                    psim.Text("")
                    psim.Text("PRED Geodesics (via loaded model)")
                    cpg, vpg = psim.Checkbox("Show PRED Geodesics",
                                             self._show_pred_geodesics)
                    if cpg: self._show_pred_geodesics = vpg; needs_rerender = True

                    if self._show_pred_geodesics:
                        # L assembly switch — mirrors visualize_validation UI.
                        assembly_opts = ['from_stiffness', 'diagonal_gram']
                        try:
                            cur_assembly_idx = assembly_opts.index(
                                self._pred_geodesics_L_assembly)
                        except ValueError:
                            cur_assembly_idx = 0
                        ca, va = psim.Combo("L assembly##pred",
                                            cur_assembly_idx, assembly_opts)
                        if ca and assembly_opts[va] != self._pred_geodesics_L_assembly:
                            self._pred_geodesics_L_assembly = assembly_opts[va]
                            self._pred_geodesics_cache = {}
                            needs_rerender = True

                        ct, vt = psim.SliderFloat(
                            "t scale (× A_total / n)",
                            self._pred_geodesics_t_scale,
                            v_min=0.01, v_max=10.0)
                        if ct and abs(vt - self._pred_geodesics_t_scale) > 1e-12:
                            self._pred_geodesics_t_scale = vt
                            self._pred_geodesics_cache = {}
                            needs_rerender = True

                        psim.Text("  Display target:")
                        psim.SameLine()
                        cpm, vpm = psim.Checkbox("On Mesh##pred",
                                                 self._pred_geodesics_on_mesh)
                        if cpm: self._pred_geodesics_on_mesh = vpm; needs_rerender = True
                        psim.SameLine()
                        cpc, vpc = psim.Checkbox("On Point Cloud##pred",
                                                 self._pred_geodesics_on_cloud)
                        if cpc: self._pred_geodesics_on_cloud = vpc; needs_rerender = True

                        # Error metrics (computed on the fly, light).
                        try:
                            src_vertex = int(to_numpy(
                                current_surface.geodesic_sources[
                                    min(self._geodesic_source_idx, S - 1)]))
                            pred_d = self._compute_pred_geodesic_distances(
                                self._patch_idx if self._patch_idx != _HYBRID_IDX else 1,
                                current_surface, src_vertex)
                            gt_d = to_numpy(current_surface.geodesic_distances[
                                min(self._geodesic_source_idx, S - 1)])
                            if pred_d is not None and pred_d.shape == gt_d.shape:
                                diff = pred_d - gt_d
                                gt_norm = float(np.linalg.norm(gt_d)) + 1e-12
                                rel_l2 = float(np.linalg.norm(diff)) / gt_norm
                                max_abs = float(np.abs(diff).max())
                                psim.Text(f"  rel L2 err: {rel_l2:.4f}")
                                psim.Text(f"  max abs err: {max_abs:.4f}  "
                                          f"(GT range [{gt_d.min():.4f}, {gt_d.max():.4f}])")
                                psim.Text(f"  PRED range: [{pred_d.min():.4f}, "
                                          f"{pred_d.max():.4f}]")
                        except Exception as e:
                            psim.TextColored((1.0, 0.5, 0.5, 1.0),
                                             f"  (PRED metric error: {e})")
                elif self.trained_model is None:
                    psim.TextColored((0.7, 0.7, 0.7, 1.0),
                        "  (Load a checkpoint to enable PRED geodesics)")

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
        self._pred_geodesics_cache = {}
        self._pred_forward_cache = {}
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
        # Auto-fallback: if the surface has no triangulation (and Delaunay
        # couldn't produce one), force the point-cloud view so the surface
        # is still visible.
        if face is None and self._show_mesh and not self._show_pointcloud:
            print("  [!] No triangulation available — showing point cloud instead")
            self._show_pointcloud = True
        if self._show_mesh and face is not None:
            mesh = ps.register_surface_mesh(f"{name} - Mesh", pos, face,
                smooth_shade=self.vis_config.smooth_shade, edge_width=self._edge_width,
                color=tuple(self._surface_color), transparency=0.5)
            mesh.set_material("clay")
            self._add_vector_quantities(mesh, surface, "mesh")
            if self._show_gt_normals_all:
                self._add_normals_to_structure(mesh, normals)
            self._add_probe_coloring(mesh, surface, self.vis_config.mesh_scalar_colormap)
            self._add_gt_gradient_all_to_structure(mesh, surface, structure_type="mesh")
            self._add_gt_geodesic_to_structure(mesh, surface, structure_type="mesh")
            self._add_pred_geodesic_to_structure(mesh, idx, surface, structure_type="mesh")
        if self._show_pointcloud:
            cloud = ps.register_point_cloud(f"{name} - Point Cloud", pos, radius=self._point_radius, enabled=True)
            cloud.set_color(tuple(c * 0.5 for c in self._surface_color))
            self._add_vector_quantities(cloud, surface, "pointcloud")
            if self._show_gt_normals_all:
                self._add_normals_to_structure(cloud, normals)
            self._add_probe_coloring(cloud, surface, self.vis_config.pointcloud_scalar_colormap)
            self._add_gt_gradient_all_to_structure(cloud, surface, structure_type="pointcloud")
            self._add_gt_geodesic_to_structure(cloud, surface, structure_type="pointcloud")
            self._add_pred_geodesic_to_structure(cloud, idx, surface, structure_type="pointcloud")
        self._add_origin_indicator(surface, name, np.zeros(3))
        self._render_origin_normals(surface, name, pred_cache_idx=idx)
        self._render_gt_gradient_arrow(surface, name)
        self._render_gt_geodesic_source_marker(surface, name)

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
        self._add_gt_geodesic_to_structure(mesh, surf_mesh, structure_type="mesh")
        # Hybrid: model trained on the point-cloud surface, so the prediction
        # is computed and attached using surf_pts; we paint it on the mesh
        # only if the two share a vertex set.  Skip mesh attachment by default.

        # Point cloud from downsampled grid (surface 1)
        cloud = ps.register_point_cloud(f"Hybrid Points ({name_pts})", pos_pts,
            radius=self._point_radius, enabled=True)
        cloud.set_color(tuple(c * 0.5 for c in self._surface_color))
        self._add_vector_quantities(cloud, surf_pts, "pointcloud")
        if self._show_gt_normals_all:
            self._add_normals_to_structure(cloud, normals_pts)
        self._add_probe_coloring(cloud, surf_pts, self.vis_config.pointcloud_scalar_colormap)
        self._add_gt_gradient_all_to_structure(cloud, surf_pts, structure_type="pointcloud")
        self._add_gt_geodesic_to_structure(cloud, surf_pts, structure_type="pointcloud")
        self._add_pred_geodesic_to_structure(cloud, 1, surf_pts, structure_type="pointcloud")

        # Origin + normals from downsampled patch (what model sees)
        self._add_origin_indicator(surf_pts, "Hybrid", np.zeros(3))
        self._render_origin_normals(surf_pts, "Hybrid", pred_cache_idx=1)
        self._render_gt_gradient_arrow(surf_pts, "Hybrid")
        # Geodesic source marker uses the points surface (what the model trained on)
        self._render_gt_geodesic_source_marker(surf_pts, "Hybrid")

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
        # Normalize whatever the loader yields into a flat list of Data items.
        # The patch-level datasets emit List[Data] (one per grid sampler) and
        # the PyG Collater passes that through.  The variational dataset emits
        # a single _VariationalSurfaceData per item, which the Collater wraps
        # in a Batch (batch_size=1) — unpack it back into a one-element list
        # so the rest of the pipeline doesn't need to special-case it.
        if isinstance(surfaces, Batch):
            surfaces = surfaces.to_data_list()
        elif isinstance(surfaces, Data):
            surfaces = [surfaces]
        surface_names = visualizer._get_surface_names(surfaces)
        visualizer.visualize_surface_set(surfaces, surface_names)
        print(f"\n[OK] Batch {batch_idx + 1} complete! Close window to continue.")
        ps.show()


if __name__ == "__main__":
    main()