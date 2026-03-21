#!/usr/bin/env python3
"""
Real-time Eigenanalysis Visualization with MeshDataset and Model Inference

This script:
1. Uses Hydra to instantiate a MeshDataset from config
2. Loads a trained LaplacianTransformerModule from checkpoint
3. For each mesh, performs real-time inference to get predicted Laplacian
4. Computes ground-truth quantities using the same methods as visualize_validation.py
5. Applies normal orientation correction and creates corrected predictions
6. Visualizes comprehensive comparison of GT vs Original vs Corrected predictions

Usage:
    python visualize_validation2.py --config-path=config --config-name=mesh_config ckpt_path=path/to/model.ckpt
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import contextlib
import os
import sys
import time

import numpy as np
import torch
import torch_geometric
import polyscope as ps
import open3d as o3d
import trimesh

# For GT eigendecomposition computation
from pyFM.mesh import TriMesh

# For GT mean curvature computation
try:
    import igl

    HAS_IGL = True
except ImportError:
    HAS_IGL = False
    print("Warning: libigl not available, GT mean curvature will not be computed")

# For robust-laplacian eigendecomposition
import robust_laplacian

# For eigendecomposition
import scipy.sparse
import scipy.sparse.linalg

# For point cloud gradient/divergence operators (Heat Method geodesics)
try:
    from pcdiff import knn_graph

    HAS_PCDIFF = True
except ImportError:
    HAS_PCDIFF = False
    print("Warning: pcdiff not available, Heat Method geodesics will not be computed")
    print("Install with: pip install pcdiff")

# Hydra
import hydra
from omegaconf import DictConfig

# PyTorch Lightning
import pytorch_lightning as pl

# Local imports
from neural_local_laplacian.modules.laplacian_modules import LaplacianTransformerModule
from neural_local_laplacian.utils.utils import (
    normalize_mesh_vertices,
    assemble_stiffness_and_mass_matrices,
    assemble_gradient_operator,
    compute_laplacian_eigendecomposition,
    build_patches_from_vertices,
    set_knn_backend,
    cuda_warmup,
)
from neural_local_laplacian.utils.geodesic_utils import (
    compute_heat_geodesics_batch,
    compute_heat_geodesic_pointcloud,
    compute_heat_geodesic_learned,
    compute_heat_geodesic_learned_batch,
    compute_heat_geodesic_mesh,
    make_grad_div_mesh,
    make_grad_div_learned,
    make_grad_div_pointcloud,
    compute_geodesic_metrics,
    compute_exact_geodesics,
    ensure_psd,
    sparse_solve,
    normalize_distances,
    build_pointcloud_grad_div_operators,
    edge_index_from_knn_indices,
    select_geodesic_source_vertex,
    select_multiple_geodesic_sources,
    compute_multisource_geodesic_metrics,
)
from torch_geometric.data import Data, Batch


# =============================================================================
# NELO — imports from nelo.src (NeLo SIGGRAPH Asia 2024).
# Assumes the NeLo repo's src folder is importable as nelo.src.*
# =============================================================================

# Add NeLo's root folder to sys.path so its internal `from src.xxx` imports resolve.
# (NeLo's own modules use bare `src.*` / `config.*` imports internally.)
_nelo_root = str(Path(__file__).parent / 'nelo')
if _nelo_root not in sys.path:
    sys.path.insert(0, _nelo_root)

from nelo.src.pipeline import MyPipeline as _NeLo_MyPipeline
from nelo.src.graphtree.graph_tree import construct_graph_tree_from_point_cloud as _nelo_construct_graph_tree_orig
# Import via the bare path that NeLo's own modules use (sys.path includes nelo/),
# so this is the same singleton instance graph_tree.py holds.
from config.global_config import global_config as _nelo_global_config


def _nelo_construct_graph_tree(vertices, k: int = 8):
    """Thin wrapper: configures global_config for point-cloud mode then calls NeLo's function."""
    _nelo_global_config.point_cloud_knn_k = k
    _nelo_global_config.use_ref_mesh = False   # ensure point-cloud branch, not mesh branch
    return _nelo_construct_graph_tree_orig(vertices, ref_mesh=None)


class _NeLo_Pipeline(_NeLo_MyPipeline):
    """
    Thin subclass of NeLo's MyPipeline that adds forward_inference() for clean inference.
    All weights / architecture come from the parent; only the inference helper is ours.
    """

    def forward_inference(self, graph_tree):
        """
        Run inference. Returns (edge_weight [E,1], vert_mass [N,1], edge_index [2,E]).
        Graph tree must already be on the correct device.
        """
        graph = graph_tree.treedict[0]

        feat = torch.ones(graph.x.shape[0], 3, device=graph.x.device)
        degrees = torch_geometric.utils.degree(
            graph.edge_index[0],
            num_nodes=graph.x.shape[0],
            dtype=torch.float32
        )
        feat = torch.cat([feat, degrees.view(-1, 1)], dim=1)  # [N, 4]

        vert_feat = self.eigen_network(feat, graph_tree, graph_tree.depth)
        if graph.x.is_cuda:
            torch.cuda.synchronize()

        va_idx = graph.edge_index[0]
        vb_idx = graph.edge_index[1]
        edge_w = self.edge_decoder((vert_feat[va_idx] - vert_feat[vb_idx]).pow(2))
        edge_w = -torch.nn.functional.relu(-edge_w)   # clamp negatives (test mode)
        mass   = self.mass_decoder(vert_feat)
        if graph.x.is_cuda:
            torch.cuda.synchronize()

        return edge_w, mass, graph.edge_index

# =============================================================================
# END OF NELO IMPORTS
# =============================================================================


@dataclass
class VisualizationConfig:
    """Configuration for eigenanalysis visualization."""
    point_radius: float = 0.005
    show_wireframe: bool = False
    colormap: str = 'coolwarm'
    num_eigenvectors_to_show: int = 8
    enable_eigenvalue_info: bool = True
    enable_correlation_analysis: bool = True
    num_validation_sources: int = 5  # Number of FPS-sampled source vertices for geodesic/Green's validation


@dataclass
class ReconstructionSettings:
    """Settings for mesh reconstruction visualization."""
    use_pred_areas: bool = True  # Required for M-orthonormal eigenvectors from generalized EVP
    current_pred_k: int = 20  # Will be updated with actual k from dataset
    current_robust_k: int = 20  # Independent k for robust-laplacian

    # Top-k weight pruning for PRED
    enable_top_k_pruning: bool = False  # When enabled, prune to top_k highest weights per patch
    pred_top_k: int = 6  # Number of highest weights to keep per patch
    symmetry_policy: str = "union"  # "union" or "intersection"

    # kNN backend: 0='pyg', 1='brute', 2='cKDTree', 3='pynndescent'
    knn_backend_idx: int = 2

    # Weyl's law eigenvalue self-calibration for PRED
    enable_weyl_calibration: bool = False
    weyl_area_source: str = "gt"  # "gt" or "pred"


class ColorPalette:
    """Color palette for different visualization elements."""

    # Normal comparison colors
    GT_NORMALS = (0.0, 1.0, 1.0)  # Cyan for ground truth
    PREDICTED_NORMALS = (1.0, 0.5, 0.0)  # Orange for predictions

    # Default colors
    DEFAULT_VECTOR = (0.5, 0.5, 0.5)  # Gray

    @classmethod
    def get_vector_color(cls, vector_name: str) -> Tuple[float, float, float]:
        """Get color for vector visualization."""
        color_map = {
            'gt_normals': cls.GT_NORMALS,
            'predicted_normals': cls.PREDICTED_NORMALS
        }
        return color_map.get(vector_name, cls.DEFAULT_VECTOR)


class VectorScales:
    """Scale factors for different vector visualizations."""

    # Mean curvature vector scales
    GT_MEAN_CURVATURE_VECTOR = 0.005
    PREDICTED_MEAN_CURVATURE_VECTOR = 0.005

    # Normal vector scales
    GT_NORMALS = 0.05
    PREDICTED_NORMALS = 0.05

    # Default scale
    DEFAULT_VECTOR = 0.05

    @classmethod
    def get_vector_scale(cls, vector_name: str) -> float:
        """Get scale factor for vector visualization."""
        scale_map = {
            'gt_mean_curvature_vector': cls.GT_MEAN_CURVATURE_VECTOR,
            'predicted_mean_curvature_vector': cls.PREDICTED_MEAN_CURVATURE_VECTOR,
            'gt_normals': cls.GT_NORMALS,
            'predicted_normals': cls.PREDICTED_NORMALS
        }
        return scale_map.get(vector_name, cls.DEFAULT_VECTOR)


@dataclass
class LaplacianTimingResults:
    """Timing results for Laplacian matrix assembly comparison.

    All times are in seconds. We time ONLY the matrix assembly (L, M computation),
    NOT the eigendecomposition, for a fair comparison.

    For each method we time the full pipeline from (vertices, k or faces) to (L, M):
    - PRED: k-NN extraction + model forward pass + sparse matrix assembly
    - Robust: point_cloud_laplacian (includes k-NN + weight computation + assembly)
    - GT: cotangent matrix computation from mesh connectivity (igl.cotmatrix + massmatrix)
    """
    # PRED timings (in seconds) - broken down for analysis
    pred_patch_extraction_time: float = 0.0  # k-NN search + data preparation
    pred_model_inference_time: float = 0.0  # Neural network forward pass
    pred_matrix_assembly_time: float = 0.0  # Sparse matrix construction from weights
    pred_total_time: float = 0.0  # Sum of above

    # NeLo timings - mirrors PRED breakdown exactly
    nelo_graph_tree_time: float = 0.0         # construct_graph_tree_from_point_cloud (KNN build)
    nelo_model_inference_time: float = 0.0    # pipeline.make_prediction forward pass
    nelo_matrix_assembly_time: float = 0.0    # sparse L, M assembly from edge_weight_predicted
    nelo_total_time: float = 0.0              # Sum of above
    nelo_k: int = 8                           # NeLo's independent k value

    # GT (mesh-based cotangent) timing
    gt_matrix_assembly_time: float = 0.0  # igl.cotmatrix + massmatrix

    # Robust (k-NN point cloud) timing
    robust_matrix_assembly_time: float = 0.0  # point_cloud_laplacian (k-NN + weights + assembly)

    # Downstream solve timings (seconds) — for E2E cost measurement
    gt_greens_solve_time: float = 0.0
    pred_greens_solve_time: float = 0.0
    robust_greens_solve_time: float = 0.0
    nelo_greens_solve_time: float = 0.0

    # Gradient operator build times (seconds) — needed for geodesics
    gt_grad_op_time: float = 0.0       # igl.grad + igl.doublearea
    pred_grad_op_time: float = 0.0     # assemble_gradient_operator
    nelo_grad_op_time: float = 0.0     # build_pointcloud_grad_div_operators

    # Pure geodesic solve times (seconds) — excludes operator construction
    gt_geodesic_solve_time: float = 0.0
    pred_geodesic_solve_time: float = 0.0
    robust_geodesic_solve_time: float = 0.0  # pp3d: self-contained (includes everything)
    nelo_geodesic_solve_time: float = 0.0

    # Mesh info
    num_vertices: int = 0
    num_faces: int = 0
    current_k: int = 0


@dataclass
class LaplacianSparsityStats:
    """Sparsity statistics for a single Laplacian matrix."""
    method_name: str = ""
    num_vertices: int = 0
    nnz: int = 0  # Number of non-zero entries
    total_entries: int = 0  # n * n
    sparsity_ratio: float = 0.0  # nnz / (n * n)
    density_percent: float = 0.0  # sparsity_ratio * 100
    avg_nnz_per_row: float = 0.0  # nnz / n
    max_nnz_per_row: int = 0
    min_nnz_per_row: int = 0

    def __str__(self) -> str:
        return (f"{self.method_name:<14} {self.num_vertices:>8} {self.nnz:>10} "
                f"{self.density_percent:>8.4f}% {self.avg_nnz_per_row:>8.1f} "
                f"[{self.min_nnz_per_row}, {self.max_nnz_per_row}]")


@dataclass
class GreensFunctionValidationResult:
    """Results from Green's function maximum principle validation.

    The harmonic Green's function g solves Lg = delta_source.
    For a valid Laplacian with nonnegative edge weights:
    - The MAXIMUM should be at the source vertex (key test!)

    After mean-centering, the Green's function has positive and negative values.
    This is expected - the key test is whether the max is at the source.

    If the Laplacian has negative edge weights (e.g., from bad triangulation),
    the maximum may occur away from the source, violating the maximum principle.
    """
    method_name: str  # "GT", "PRED", or "Robust"
    source_vertex_idx: int
    num_vertices: int

    # Green's function statistics (after mean-centering)
    min_value: float = 0.0
    max_value: float = 0.0
    mean_value: float = 0.0  # Should be ~0 after centering
    value_at_source: float = 0.0

    # === PRIMARY TEST: Maximum Principle ===
    max_at_source: bool = True  # Whether maximum is at source vertex (KEY TEST)
    satisfies_maximum_principle: bool = True  # Overall pass/fail
    num_violations: int = 0  # Count of vertices with value > source
    worst_violation_vertex: int = -1  # Index of vertex with highest value (if not source)
    worst_violation_value: float = 0.0  # Value at worst violating vertex

    # === SOURCE-CENTERED CHECK (Sharp & Crane style) ===
    # After shifting so source = 0, all values should be <= 0
    # This directly tests: "Is any vertex hotter than the source?"
    num_positive_vertices: int = 0  # Count of vertices with g > g_source (should be 0)
    max_positive_value: float = 0.0  # Worst violation: max(g - g_source) (should be <= 0)
    positive_vertex_idx: int = -1  # Index of worst positive vertex

    # === SECONDARY TEST: Monotonic Decay ===
    # Green's function should decrease with distance from source
    distance_correlation: float = 0.0  # Correlation(g, -distance), should be positive
    monotonicity_score: float = 0.0  # Fraction of vertex pairs where closerÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢higher value

    # === TERTIARY TEST: Smoothness ===
    laplacian_residual_norm: float = 0.0  # ||Lg - delta|| / ||delta|| (should be small)

    # === COMPARISON: Correlation with GT ===
    correlation_with_gt: float = 0.0  # Pearson correlation with GT (for PRED/Robust)

    # === DIAGNOSTIC: Min location info ===
    min_vertex_idx: int = -1  # Index of vertex with minimum value
    geodesic_dist_to_min: float = 0.0  # Geodesic distance from source to min vertex
    max_geodesic_dist: float = 0.0  # Maximum geodesic distance from source (for reference)

    def __str__(self) -> str:
        status = "[OK] PASS" if self.satisfies_maximum_principle else "[X] FAIL"
        max_status = "[OK]" if self.max_at_source else "[X]"
        return (f"{self.method_name:<14} {self.value_at_source:>10.4f} {self.max_value:>10.4f} "
                f"{max_status:^12} {self.num_violations:>6} {status:>10}")


@dataclass
class HeatMethodGeodesicResult:
    """Results from Heat Method geodesic distance computation.

    The Heat Method (Crane et al. 2013) computes geodesic distances by:
    1. Solving heat equation: (M + t*L) u = delta_source
    2. Normalizing the gradient: X = -grad(u) / |grad(u)|
    3. Solving Poisson: L @ phi = div(X)

    This tests the Laplacian's ability to produce correct geodesic distances
    via the diffusion-based approach.
    """
    method_name: str  # "GT", "PRED", "Robust", "potpourri3d" (reference)
    source_vertex_idx: int
    num_vertices: int
    time_step: float = 0.0  # t = h^2 typically

    # Distance statistics
    min_distance: float = 0.0
    max_distance: float = 0.0
    mean_distance: float = 0.0
    distance_at_source: float = 0.0  # Should be ~0

    # Quality metrics
    correlation_with_reference: float = 0.0  # Correlation with exact/reference geodesic
    mean_absolute_error: float = 0.0  # MAE vs reference
    max_absolute_error: float = 0.0  # Max error vs reference
    relative_error_percent: float = 0.0  # (mean error / mean distance) * 100

    # Monotonicity: does distance increase away from source?
    monotonicity_score: float = 0.0  # Fraction of pairs where farther point has larger distance

    # Timing
    computation_time_ms: float = 0.0

    def __str__(self) -> str:
        return (f"{self.method_name:<14} {self.correlation_with_reference:>8.4f} "
                f"{self.mean_absolute_error:>10.6f} {self.relative_error_percent:>8.2f}%")


@dataclass
class AggregatedValidationMetrics:
    """Aggregated metrics across multiple source vertices."""
    method_name: str
    n_sources: int = 0

    # Geodesic metrics (mean ± std across sources)
    corr_mean: float = 0.0
    corr_std: float = 0.0
    mae_mean: float = 0.0
    mae_std: float = 0.0
    max_err_mean: float = 0.0
    max_err_std: float = 0.0
    mono_mean: float = 0.0
    mono_std: float = 0.0

    # Green's function metrics
    greens_corr_with_gt_mean: float = 0.0
    greens_corr_with_gt_std: float = 0.0
    greens_mono_mean: float = 0.0
    greens_mono_std: float = 0.0
    greens_max_principle_pass_rate: float = 0.0  # fraction passing max principle


class RealTimeEigenanalysisVisualizer:
    """Real-time eigenanalysis visualizer using MeshDataset and model inference."""

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.color_palette = ColorPalette()
        self.vector_scales = VectorScales()

        # NEW: Reconstruction settings with UI control
        self.reconstruction_settings = ReconstructionSettings()

        # Store current batch data for re-computation
        self.current_gt_data = None
        self.current_inference_result = None
        self.current_predicted_data = None
        self.current_mesh_structure = None

        # Store raw data for k-NN slider updates
        self.current_stiffness_weights = None
        self.current_areas = None
        self.current_original_vertices = None
        self._current_vertices_tensor = None  # cached GPU tensor of vertices
        self.original_k = None
        self.current_vertex_indices = None
        self.current_center_indices = None
        self.current_batch_indices = None

        # Store model, device, and mesh info for k updates
        self.current_model = None
        self.current_device = None
        self.current_mesh_file_path = None
        self.current_faces = None

        # Store learned gradient operator (gradient mode only)
        self.current_learned_gradient_op = None

        # NEW: Track reconstruction structure names for removal
        self.reconstruction_structure_names = []

        # Store average eigenvector cosine similarity for UI display
        self.current_avg_cosine_similarity = None

        # Store per-eigenvector cosine similarities for UI comparison table
        self.current_cosine_similarities_pred = None  # Array of |cos| per eigenvector (GT vs PRED)
        self.current_cosine_similarities_robust = None  # Array of |cos| per eigenvector (GT vs Robust)
        self.current_cosine_similarities_nelo = None  # Array of |cos| per eigenvector (GT vs NeLo)

        # NEW: Timing results for Laplacian assembly comparison
        self.timing_results = LaplacianTimingResults()

        # NEW: Green's function validation results
        self.current_greens_results = None

        # NEW: Heat Method geodesic validation results
        self.current_heat_geodesic_results = None

        # NEW: Laplacian sparsity comparison stats
        self.current_sparsity_stats: Dict[str, LaplacianSparsityStats] = {}

        # Cached state for targeted per-method validation recomputation
        # Multi-source support: source vertices shared by Green's and geodesics
        self._source_indices: Optional[List[int]] = None  # Multiple source vertices (FPS-sampled)
        self._exact_geodesics: Optional[Dict[int, np.ndarray]] = None  # source_idx -> exact distances
        self._greens_gt_values: Optional[Dict[int, np.ndarray]] = None  # source_idx -> GT Green's fn

        # Per-source results (all sources × all methods)
        self._all_greens_results: Dict[int, Dict[str, 'GreensFunctionResult']] = {}  # source_idx -> {method -> result}
        self._all_geodesic_results: Dict[int, Dict[str, 'HeatMethodGeodesicResult']] = {}

        # Raw cached values for instant slider refresh (source_idx -> method -> array(N,))
        self._all_greens_values: Dict[int, Dict[str, np.ndarray]] = {}
        self._all_geodesic_distances: Dict[int, Dict[str, np.ndarray]] = {}

        # Aggregated metrics across all sources
        self._aggregated_geodesic_metrics: Dict[str, AggregatedValidationMetrics] = {}
        self._aggregated_greens_metrics: Dict[str, AggregatedValidationMetrics] = {}

        # Currently displayed source (for Polyscope visualization)
        self._current_source_display_idx: int = 0  # Index into _source_indices

        self._pc_grad_op = None   # pcdiff point cloud gradient operator
        self._pc_div_op = None    # pcdiff point cloud divergence operator

        # New metric results
        self._probe_function_results: Dict[str, Dict] = {}   # method -> {mse, failure_rate}
        self._descriptor_results: Dict[str, Dict] = {}       # method -> {hks_l2, hks_corr, wks_l2, wks_corr}
        self._compression_results: Dict[str, Dict] = {}      # method -> {k -> {mean_l2, max_l2}}

        # Weyl's law eigenvalue self-calibration
        self._weyl_calibration_factor: Optional[float] = None  # α: PRED eigs / expected eigs
        self._weyl_calibrated_evals: Optional[np.ndarray] = None  # PRED eigenvalues / α

        # Weight distribution stats (stored for CSV export)
        self._weight_distribution_stats: Dict[str, float] = {}

        # Mesh boundary info (stored for CSV export)
        self._mesh_boundary_info: Dict[str, Any] = {}

        # Debug flags (set from config in run_dataset_iteration)
        self._diagnostic_mode = False
        self._skip_robust = False
        self._quantitative_mode = False
        self._robust_geodesic_heat = False
        self._verbose = True  # If False, suppresses stdout during process_batch

        # NeLo state
        self.nelo_pipeline = None          # loaded once at startup via load_nelo_pipeline(); None = disabled
        self.nelo_k = 30                    # independent k slider (default = 30, matching PRED/Robust)

        # Weight neighborhood visualization (FPS-sampled vertices + their kNN colored by weight)
        self._weight_nbr_sample_indices: Optional[np.ndarray] = None
        self._weight_nbr_num_samples = 8

        self.current_nelo_L = None
        self.current_nelo_M = None
        self.current_nelo_eigenvectors = None
        self.current_nelo_eigenvalues = None

    def setup_polyscope(self):
        """Initialize and configure polyscope with UI callback."""
        ps.init()
        ps.set_up_dir("z_up")
        ps.look_at(camera_location=[2.0, 2.0, 2.0], target=[0, 0, 0])
        ps.set_ground_plane_mode("none")
        ps.set_background_color((0.05, 0.05, 0.05))  # Dark background

        # Set up UI callback for reconstruction settings
        ps.set_user_callback(self._reconstruction_settings_ui_callback)

    def _reconstruction_settings_ui_callback(self):
        """ImGui callback for reconstruction settings window."""
        import polyscope.imgui as psim

        # Checkbox for using predicted areas for area-weighted reconstruction
        psim.Text("PRED Mesh Reconstruction Options:")
        psim.Separator()

        changed, new_setting = psim.Checkbox(
            "Use Predicted Areas (Area-Weighted)",
            self.reconstruction_settings.use_pred_areas
        )

        if changed:
            self.reconstruction_settings.use_pred_areas = new_setting
            print(f"[*] Reconstruction setting changed: use_pred_areas = {new_setting}")

            # Re-compute and update reconstructions if we have current data
            if self._has_current_batch_data():
                self._recompute_and_update_reconstructions()

        psim.Text("")

        # k-NN input fields for PRED and Robust (independent)
        if self.original_k is not None:
            # PRED k input
            pred_k_changed, new_pred_k = psim.InputInt(
                "PRED k-NN neighbors",
                self.reconstruction_settings.current_pred_k,
                flags=psim.ImGuiInputTextFlags_EnterReturnsTrue
            )
            new_pred_k = max(5, min(100, new_pred_k))

            if pred_k_changed and new_pred_k != self.reconstruction_settings.current_pred_k:
                self.reconstruction_settings.current_pred_k = new_pred_k
                print(f"[*] PRED k changed: {new_pred_k}")
                if self._has_current_batch_data():
                    self._update_pred_with_new_k(new_pred_k)

            # Robust k input
            robust_k_changed, new_robust_k = psim.InputInt(
                "Robust k-NN neighbors",
                self.reconstruction_settings.current_robust_k,
                flags=psim.ImGuiInputTextFlags_EnterReturnsTrue
            )
            new_robust_k = max(5, min(100, new_robust_k))

            if robust_k_changed and new_robust_k != self.reconstruction_settings.current_robust_k:
                self.reconstruction_settings.current_robust_k = new_robust_k
                print(f"[*] Robust k changed: {new_robust_k}")
                if self._has_current_batch_data():
                    self._update_robust_with_new_k(new_robust_k)

            # NeLo k input (independent — NeLo was trained at k=8)
            if self.nelo_pipeline is not None:
                nelo_k_changed, new_nelo_k = psim.InputInt(
                    "NeLo k-NN neighbors (trained=8)",
                    self.nelo_k,
                    flags=psim.ImGuiInputTextFlags_EnterReturnsTrue
                )
                new_nelo_k = max(4, min(100, new_nelo_k))

                if nelo_k_changed and new_nelo_k != self.nelo_k:
                    self.nelo_k = new_nelo_k
                    print(f"[*] NeLo k changed: {new_nelo_k}")
                    if self.current_original_vertices is not None:
                        self._update_nelo_with_new_k(new_nelo_k)
        else:
            psim.Text("k-NN input: (no data loaded)")

        psim.Text("(Press Enter to apply changes)")

        # --- kNN backend toggle ---
        knn_backends = ['pyg (torch_geometric)', 'brute (cdist+topk)', 'cKDTree (scipy)', 'pynndescent (approx)']
        knn_backend_names = ['pyg', 'brute', 'cKDTree', 'pynndescent']
        knn_changed, new_knn_idx = psim.Combo(
            "kNN Backend",
            self.reconstruction_settings.knn_backend_idx,
            knn_backends
        )
        if knn_changed:
            self.reconstruction_settings.knn_backend_idx = new_knn_idx
            set_knn_backend(knn_backend_names[new_knn_idx])
            # Reset one-time log flags so next kNN call logs the backend
            build_patches_from_vertices._logged_gpu_knn = False
            from neural_local_laplacian.utils.utils import _knn_cpu
            _knn_cpu._logged = False
            _knn_cpu._logged_pynn = False

        # --- Top-k weight pruning for PRED ---
        psim.Text("")
        psim.Separator()
        psim.Text("PRED Top-k Weight Pruning:")

        topk_changed, new_topk_enabled = psim.Checkbox(
            "Enable top-k pruning",
            self.reconstruction_settings.enable_top_k_pruning
        )
        if topk_changed:
            self.reconstruction_settings.enable_top_k_pruning = new_topk_enabled
            print(f"[*] Top-k pruning: {'enabled' if new_topk_enabled else 'disabled'}")
            if self._has_current_batch_data():
                self._update_pred_with_new_k(self.reconstruction_settings.current_pred_k)

        if self.reconstruction_settings.enable_top_k_pruning:
            topk_k_changed, new_topk_k = psim.InputInt(
                "Top-k (weights to keep)",
                self.reconstruction_settings.pred_top_k,
                flags=psim.ImGuiInputTextFlags_EnterReturnsTrue
            )
            new_topk_k = max(3, min(self.reconstruction_settings.current_pred_k - 1, new_topk_k))

            if topk_k_changed and new_topk_k != self.reconstruction_settings.pred_top_k:
                self.reconstruction_settings.pred_top_k = new_topk_k
                print(f"[*] Top-k value changed: {new_topk_k}")
                if self._has_current_batch_data():
                    self._update_pred_with_new_k(self.reconstruction_settings.current_pred_k)

            # Symmetry policy combo box
            policies = ["union", "intersection"]
            current_idx = policies.index(self.reconstruction_settings.symmetry_policy) if self.reconstruction_settings.symmetry_policy in policies else 0
            policy_changed, new_policy_idx = psim.Combo(
                "Symmetry policy",
                current_idx,
                policies
            )
            if policy_changed and policies[new_policy_idx] != self.reconstruction_settings.symmetry_policy:
                self.reconstruction_settings.symmetry_policy = policies[new_policy_idx]
                print(f"[*] Symmetry policy changed: {policies[new_policy_idx]}")
                if self._has_current_batch_data():
                    self._update_pred_with_new_k(self.reconstruction_settings.current_pred_k)

            psim.TextColored((0.7, 0.7, 0.7, 1.0),
                f"k1={self.reconstruction_settings.current_pred_k} (receptive field) -> "
                f"k2={self.reconstruction_settings.pred_top_k} (operator support), "
                f"policy={self.reconstruction_settings.symmetry_policy}"
            )

        # --- Weyl's Law Eigenvalue Calibration ---
        psim.Text("")
        psim.Separator()
        psim.Text("PRED Eigenvalue Self-Calibration:")

        weyl_changed, new_weyl = psim.Checkbox(
            "Weyl's law calibration",
            self.reconstruction_settings.enable_weyl_calibration
        )
        if weyl_changed:
            self.reconstruction_settings.enable_weyl_calibration = new_weyl
            if new_weyl:
                print("[*] Weyl calibration enabled")
                self._apply_weyl_calibration()
            else:
                print("[*] Weyl calibration disabled")
                self._revert_weyl_calibration()

        # Area source selector (only show when calibration is available)
        if self.reconstruction_settings.enable_weyl_calibration or self._weyl_calibration_factor is not None:
            area_sources = ["gt", "pred"]
            area_labels = ["GT areas (from mesh)", "PRED areas (self-supervised)"]
            current_area_idx = area_sources.index(self.reconstruction_settings.weyl_area_source) if self.reconstruction_settings.weyl_area_source in area_sources else 0
            area_changed, new_area_idx = psim.Combo(
                "Area source",
                current_area_idx,
                area_labels
            )
            if area_changed and area_sources[new_area_idx] != self.reconstruction_settings.weyl_area_source:
                self.reconstruction_settings.weyl_area_source = area_sources[new_area_idx]
                print(f"[*] Weyl area source changed: {area_sources[new_area_idx]}")
                # Recompute with new area source
                self._weyl_calibration_factor = None
                self._weyl_calibrated_evals = None
                self._compute_weyl_calibration()
                if self.reconstruction_settings.enable_weyl_calibration:
                    self._compute_descriptor_comparison()

        # Show calibration info
        if self.reconstruction_settings.enable_weyl_calibration and self._weyl_calibration_factor is not None:
            alpha = self._weyl_calibration_factor
            src_label = "PRED" if self.reconstruction_settings.weyl_area_source == "pred" else "GT"
            psim.TextColored((0.0, 1.0, 1.0, 1.0), f"  Factor: {alpha:.2f}x ({src_label} areas)")
            if self._weyl_calibrated_evals is not None:
                cal = self._weyl_calibrated_evals
                pos_cal = cal[cal > 1e-10]
                if len(pos_cal) > 0:
                    psim.Text(f"  Calibrated range: [{pos_cal[0]:.4f}, {pos_cal[-1]:.4f}]")
                if self.current_gt_data is not None and self.current_gt_data.get('gt_eigenvalues') is not None:
                    gt_evals = self.current_gt_data['gt_eigenvalues']
                    psim.Text(f"  GT range:         [{gt_evals[1]:.4f}, {gt_evals[-1]:.4f}]")

        psim.Text("")
        psim.Text("Current Settings:")
        if self.reconstruction_settings.use_pred_areas:
            psim.TextColored((0.0, 1.0, 0.0, 1.0), "[OK] Using predicted areas from model")
            psim.Text("  (Area-weighted reconstruction for PRED)")
        else:
            psim.TextColored((1.0, 0.5, 0.0, 1.0), "[o] Using standard Euclidean reconstruction")
            psim.Text("  (Standard L2 projection for PRED)")

        if self.original_k is not None:
            psim.Text(f"Dataset k: {self.original_k}, PRED k: {self.reconstruction_settings.current_pred_k}, Robust k: {self.reconstruction_settings.current_robust_k}, NeLo k: {self.nelo_k}")

        psim.Text("")
        psim.Separator()
        psim.Text("Note: GT always uses PyFM vertex areas and original mesh connectivity")

        # Display eigenvector cosine similarity
        psim.Text("")
        psim.Separator()
        psim.Text("Eigenvector Alignment:")
        if self.current_avg_cosine_similarity is not None:
            avg_cos_sim = self.current_avg_cosine_similarity
            # Color code based on similarity quality
            if avg_cos_sim > 0.9:
                psim.TextColored((0.0, 1.0, 0.0, 1.0), f"Avg |cos| similarity: {avg_cos_sim:.4f}")
            elif avg_cos_sim > 0.7:
                psim.TextColored((1.0, 1.0, 0.0, 1.0), f"Avg |cos| similarity: {avg_cos_sim:.4f}")
            else:
                psim.TextColored((1.0, 0.3, 0.0, 1.0), f"Avg |cos| similarity: {avg_cos_sim:.4f}")
        else:
            psim.Text("Avg |cos| similarity: (not computed)")

        # === TIMING COMPARISON TABLE ===
        psim.Text("")
        psim.Separator()
        psim.Text("Laplacian Assembly Timing Comparison:")

        t = self.timing_results
        if t.num_vertices > 0:
            # Mesh info
            pred_k = self.reconstruction_settings.current_pred_k
            robust_k = self.reconstruction_settings.current_robust_k
            psim.Text(f"Mesh: {t.num_vertices} verts, {t.num_faces} faces, PRED k={pred_k}, Robust k={robust_k}")
            psim.Text("")

            # Table header
            psim.TextColored((0.7, 0.7, 0.7, 1.0), f"{'Method':<20} {'Time (ms)':>12}")
            psim.Separator()

            # PRED breakdown
            psim.TextColored((1.0, 0.5, 0.0, 1.0), "PRED (Neural Network)")
            psim.Text(f"  k-NN extraction:   {t.pred_patch_extraction_time * 1000:>10.1f} ms")
            psim.Text(f"  Model inference:   {t.pred_model_inference_time * 1000:>10.1f} ms")
            psim.Text(f"  Matrix assembly:   {t.pred_matrix_assembly_time * 1000:>10.1f} ms")
            pred_total_ms = t.pred_total_time * 1000
            psim.TextColored((1.0, 0.5, 0.0, 1.0), f"  TOTAL:             {pred_total_ms:>10.1f} ms")

            psim.Text("")

            # NeLo breakdown
            if t.nelo_total_time > 0:
                psim.TextColored((0.7, 0.3, 1.0, 1.0), f"NeLo (GNN, k={t.nelo_k})")
                psim.Text(f"  Graph tree:        {t.nelo_graph_tree_time * 1000:>10.1f} ms")
                psim.Text(f"  Model inference:   {t.nelo_model_inference_time * 1000:>10.1f} ms")
                psim.Text(f"  Matrix assembly:   {t.nelo_matrix_assembly_time * 1000:>10.1f} ms")
                nelo_total_ms = t.nelo_total_time * 1000
                psim.TextColored((0.7, 0.3, 1.0, 1.0), f"  TOTAL:             {nelo_total_ms:>10.1f} ms")
            else:
                psim.TextColored((0.5, 0.5, 0.5, 1.0), "NeLo: (not available)")

            psim.Text("")

            # Robust
            robust_ms = t.robust_matrix_assembly_time * 1000
            psim.TextColored((0.0, 0.7, 1.0, 1.0), f"Robust (Point Cloud):{robust_ms:>10.1f} ms")

            # GT
            gt_ms = t.gt_matrix_assembly_time * 1000
            psim.TextColored((0.0, 1.0, 0.0, 1.0), f"GT (Mesh Cotangent): {gt_ms:>10.1f} ms")

            psim.Text("")

            # Speedup comparison
            if t.gt_matrix_assembly_time > 0:
                pred_vs_gt = t.pred_total_time / t.gt_matrix_assembly_time
                robust_vs_gt = t.robust_matrix_assembly_time / t.gt_matrix_assembly_time if t.robust_matrix_assembly_time > 0 else 0
                psim.Text(f"Relative to GT:")
                psim.Text(f"  PRED:   {pred_vs_gt:>6.2f}x")
                psim.Text(f"  Robust: {robust_vs_gt:>6.2f}x")
                if t.nelo_total_time > 0:
                    nelo_vs_gt = t.nelo_total_time / t.gt_matrix_assembly_time
                    psim.Text(f"  NeLo:   {nelo_vs_gt:>6.2f}x")

            if t.robust_matrix_assembly_time > 0 and t.pred_total_time > 0:
                pred_vs_robust = t.pred_total_time / t.robust_matrix_assembly_time
                psim.Text(f"PRED vs Robust: {pred_vs_robust:.2f}x")
            if t.nelo_total_time > 0 and t.robust_matrix_assembly_time > 0:
                nelo_vs_robust = t.nelo_total_time / t.robust_matrix_assembly_time
                psim.Text(f"NeLo vs Robust: {nelo_vs_robust:.2f}x")
            if t.nelo_total_time > 0 and t.pred_total_time > 0:
                pred_vs_nelo = t.pred_total_time / t.nelo_total_time
                psim.Text(f"PRED vs NeLo:   {pred_vs_nelo:.2f}x")

            # E2E timing: assembly + downstream solve
            has_e2e = (t.pred_geodesic_solve_time > 0 or t.pred_greens_solve_time > 0 or
                       t.gt_geodesic_solve_time > 0 or t.gt_greens_solve_time > 0)
            if has_e2e:
                n_src = len(self._source_indices) if self._source_indices else 1
                src_note = f" (avg/{n_src}src)" if n_src > 1 else ""
                psim.Text("")
                psim.Separator()
                psim.Text(f"Geodesic E2E: L,M + Grad + Solve{src_note} (ms):")
                psim.TextColored((0.7, 0.7, 0.7, 1.0), f"{'Method':<12} {'L,M':>7} {'Grad':>7} {'Solve':>7} {'TOTAL':>7}")

                for label, color, lm_t, grad_t, geo_t in [
                    ('GT',     (0.0, 1.0, 0.0, 1.0), t.gt_matrix_assembly_time,     t.gt_grad_op_time,   t.gt_geodesic_solve_time),
                    ('PRED',   (1.0, 0.5, 0.0, 1.0), t.pred_total_time,             t.pred_grad_op_time, t.pred_geodesic_solve_time),
                    ('NeLo',   (0.7, 0.3, 1.0, 1.0), t.nelo_total_time,             t.nelo_grad_op_time, t.nelo_geodesic_solve_time),
                ]:
                    if geo_t > 0:
                        geo_avg = geo_t / n_src
                        total = lm_t + grad_t + geo_avg
                        psim.TextColored(color, f"{label:<12} {lm_t*1000:>7.0f} {grad_t*1000:>7.0f} {geo_avg*1000:>7.0f} {total*1000:>7.0f}")

                # Robust pp3d: self-contained
                if t.robust_geodesic_solve_time > 0:
                    geo_avg = t.robust_geodesic_solve_time / n_src
                    if self._robust_geodesic_heat:
                        total = t.robust_matrix_assembly_time + geo_avg
                        psim.TextColored((0.0, 0.7, 1.0, 1.0), f"{'Robust':<12} {t.robust_matrix_assembly_time*1000:>7.0f} {'incl.':>7} {geo_avg*1000:>7.0f} {total*1000:>7.0f}")
                    else:
                        psim.TextColored((0.0, 0.7, 1.0, 1.0), f"{'Robust(pp3d)':<12} {'—':>7} {'—':>7} {geo_avg*1000:>7.0f} {geo_avg*1000:>7.0f}")

                psim.Text("")
                psim.Text(f"Green's E2E: L,M + Solve{src_note} (ms):")
                psim.TextColored((0.7, 0.7, 0.7, 1.0), f"{'Method':<12} {'L,M':>7} {'Solve':>7} {'TOTAL':>7}")

                for label, color, lm_t, grn_t in [
                    ('GT',     (0.0, 1.0, 0.0, 1.0), t.gt_matrix_assembly_time,     t.gt_greens_solve_time),
                    ('PRED',   (1.0, 0.5, 0.0, 1.0), t.pred_total_time,             t.pred_greens_solve_time),
                    ('NeLo',   (0.7, 0.3, 1.0, 1.0), t.nelo_total_time,             t.nelo_greens_solve_time),
                    ('Robust', (0.0, 0.7, 1.0, 1.0), t.robust_matrix_assembly_time, t.robust_greens_solve_time),
                ]:
                    if grn_t > 0:
                        grn_avg = grn_t / n_src
                        total = lm_t + grn_avg
                        psim.TextColored(color, f"{label:<12} {lm_t*1000:>7.0f} {grn_avg*1000:>7.0f} {total*1000:>7.0f}")
        else:
            psim.Text("(No timing data yet)")

        # === GREEN'S FUNCTION MAXIMUM PRINCIPLE VALIDATION ===
        psim.Text("")
        psim.Separator()
        psim.Text("Maximum Principle Validation:")

        # Aggregated pass rate (multi-source)
        if self._aggregated_greens_metrics:
            n_src = len(self._source_indices) if self._source_indices else 0
            for method, agg in self._aggregated_greens_metrics.items():
                n_pass = int(round(agg.greens_max_principle_pass_rate * agg.n_sources))
                if n_pass == agg.n_sources:
                    psim.TextColored((0.0, 1.0, 0.0, 1.0),
                                     f"  {method}: {n_pass}/{agg.n_sources} passed ({agg.greens_max_principle_pass_rate:.0%})")
                else:
                    psim.TextColored((1.0, 0.0, 0.0, 1.0),
                                     f"  {method}: {n_pass}/{agg.n_sources} passed ({agg.greens_max_principle_pass_rate:.0%})")
        elif self.current_greens_results is not None and len(self.current_greens_results) > 0:
            # Fallback: single source display
            for method_name, result in self.current_greens_results.items():
                if result.satisfies_maximum_principle:
                    psim.TextColored((0.0, 1.0, 0.0, 1.0), f"  {method_name}: [OK] PASS")
                else:
                    psim.TextColored((1.0, 0.0, 0.0, 1.0), f"  {method_name}: [X] FAIL")
                    if result.num_violations > 0:
                        psim.Text(f"    {result.num_violations} vertices above source")
        else:
            psim.Text("(Not computed yet)")

        # === GEODESIC QUALITY METRICS TABLE ===
        psim.Text("")
        psim.Separator()
        psim.Text("Geodesic Quality Metrics (vs Exact):")

        if self.current_heat_geodesic_results is not None and len(self.current_heat_geodesic_results) > 0:
            # Header
            psim.TextColored((0.7, 0.7, 0.7, 1.0), f"  {'Method':<16} {'Corr':>7} {'MAE':>7} {'MaxErr':>7} {'Mono':>7}")
            psim.Separator()

            for method_name, result in self.current_heat_geodesic_results.items():
                corr = result.correlation_with_reference
                mae = result.mean_absolute_error
                max_err = result.max_absolute_error
                mono = result.monotonicity_score

                # Flag unreasonable results (max raw distance >> 1 for normalized meshes)
                is_unreasonable = result.max_distance > 1e3

                if is_unreasonable:
                    color = (1.0, 0.0, 0.0, 1.0)  # Red for unreasonable
                    label = f"  {method_name + ' [!]':<16} {corr:>7.4f} {mae:>7.4f} {max_err:>7.4f} {mono:>7.4f}"
                else:
                    # Color code by correlation quality
                    if corr > 0.99:
                        color = (0.0, 1.0, 0.0, 1.0)  # Green
                    elif corr > 0.95:
                        color = (1.0, 1.0, 0.0, 1.0)  # Yellow
                    else:
                        color = (1.0, 0.3, 0.0, 1.0)  # Orange-red
                    label = f"  {method_name:<16} {corr:>7.4f} {mae:>7.4f} {max_err:>7.4f} {mono:>7.4f}"

                psim.TextColored(color, label)

            # Multi-source aggregated metrics
            if self._aggregated_geodesic_metrics:
                psim.Text("")
                n_src = len(self._source_indices) if self._source_indices else 0
                psim.Text(f"Aggregated ({n_src} sources, mean±std):")
                psim.TextColored((0.7, 0.7, 0.7, 1.0), f"  {'Method':<16} {'Corr':>14} {'MAE':>14} {'Mono':>14}")
                for method, agg in self._aggregated_geodesic_metrics.items():
                    psim.Text(f"  {method:<16} {agg.corr_mean:.3f}±{agg.corr_std:.3f}"
                              f"  {agg.mae_mean:.3f}±{agg.mae_std:.3f}"
                              f"  {agg.mono_mean:.3f}±{agg.mono_std:.3f}")

            # Source vertex selector
            if self._source_indices and len(self._source_indices) > 1:
                psim.Text("")
                src_changed, new_src_idx = psim.SliderInt(
                    "Display source",
                    self._current_source_display_idx,
                    v_min=0, v_max=len(self._source_indices) - 1
                )
                if src_changed:
                    self._current_source_display_idx = new_src_idx
                    self._refresh_source_display()
                src_vtx = self._source_indices[self._current_source_display_idx]
                psim.Text(f"  Source vertex: {src_vtx}")
        else:
            psim.Text("(Not computed yet)")

        # === EIGENVECTOR CUMULATIVE COSINE SIMILARITY COMPARISON ===
        psim.Text("")
        psim.Separator()
        psim.Text("Eigenvector Mean |cos| Similarity (cumulative):")

        pred_sims = self.current_cosine_similarities_pred
        robust_sims = self.current_cosine_similarities_robust
        nelo_sims = self.current_cosine_similarities_nelo

        if pred_sims is not None or robust_sims is not None or nelo_sims is not None:
            max_available = 0
            if pred_sims is not None:
                max_available = max(max_available, len(pred_sims))
            if robust_sims is not None:
                max_available = max(max_available, len(robust_sims))
            if nelo_sims is not None:
                max_available = max(max_available, len(nelo_sims))

            # Header
            psim.TextColored((0.7, 0.7, 0.7, 1.0), f"  {'#Eigvec':>8}  {'PRED':>8}  {'NeLo':>8}  {'Robust':>8}  {'PRED-Rob':>8}")
            psim.Separator()

            # Select which counts to display: show a reasonable subset
            # Always include 1, 2, 5, then every 5 up to 30, then every 10
            display_counts = sorted(set(
                [1, 2, 3, 5] +
                list(range(10, min(max_available, 30) + 1, 5)) +
                list(range(30, min(max_available, 100) + 1, 10)) +
                ([max_available] if max_available > 0 else [])
            ))
            display_counts = [c for c in display_counts if c <= max_available]

            for count in display_counts:
                pred_mean   = float(pred_sims[:count].mean())   if pred_sims   is not None and count <= len(pred_sims)   else None
                robust_mean = float(robust_sims[:count].mean()) if robust_sims is not None and count <= len(robust_sims) else None
                nelo_mean   = float(nelo_sims[:count].mean())   if nelo_sims   is not None and count <= len(nelo_sims)   else None

                pred_str   = f"{pred_mean:>8.4f}"   if pred_mean   is not None else f"{'N/A':>8}"
                robust_str = f"{robust_mean:>8.4f}" if robust_mean is not None else f"{'N/A':>8}"
                nelo_str   = f"{nelo_mean:>8.4f}"   if nelo_mean   is not None else f"{'N/A':>8}"

                # Delta: PRED - Robust (positive = PRED is better)
                if pred_mean is not None and robust_mean is not None:
                    delta = pred_mean - robust_mean
                    delta_str = f"{delta:>+8.4f}"
                    if delta > 0.01:
                        color = (0.0, 1.0, 0.0, 1.0)  # Green - PRED better
                    elif delta < -0.01:
                        color = (1.0, 0.3, 0.0, 1.0)  # Orange - Robust better
                    else:
                        color = (0.8, 0.8, 0.8, 1.0)  # Gray - roughly equal
                else:
                    delta_str = f"{'N/A':>8}"
                    color = (0.8, 0.8, 0.8, 1.0)

                psim.TextColored(color, f"  {count:>8}  {pred_str}  {nelo_str}  {robust_str}  {delta_str}")
        else:
            psim.Text("(Not computed yet)")

        # === PROBE FUNCTION MSE (Lf ERROR) ===
        psim.Text("")
        psim.Separator()
        psim.Text("Probe Function MSE (Lf error, NeLo Table 1):")

        if self._probe_function_results:
            psim.TextColored((0.7, 0.7, 0.7, 1.0), f"  {'Method':<16} {'Raw MSE':>10} {'Cos Sim':>10} {'R_MSE>1':>10}")
            for method, res in self._probe_function_results.items():
                raw_mse = res['mse']
                cos_sim = res.get('cosine_similarity', 0.0)
                fr = res['failure_rate']
                if method == 'GT':
                    color = (0.5, 0.5, 0.5, 1.0)
                elif cos_sim > 0.95:
                    color = (0.0, 1.0, 0.0, 1.0)
                elif cos_sim > 0.8:
                    color = (1.0, 1.0, 0.0, 1.0)
                else:
                    color = (1.0, 0.3, 0.0, 1.0)
                psim.TextColored(color, f"  {method:<16} {raw_mse:>10.6f} {cos_sim:>10.6f} {fr:>9.1%}")
        else:
            psim.Text("(Not computed yet)")

        # === HKS / WKS DESCRIPTOR ERROR ===
        psim.Text("")
        psim.Separator()
        if self.reconstruction_settings.enable_weyl_calibration and self._weyl_calibration_factor is not None:
            src_label = "PRED" if self.reconstruction_settings.weyl_area_source == "pred" else "GT"
            psim.TextColored((0.0, 1.0, 1.0, 1.0), f"Shape Descriptors (HKS/WKS vs GT) [Weyl calibrated, {src_label} areas]:")
        else:
            psim.Text("Shape Descriptors (HKS/WKS vs GT):")

        if self._descriptor_results:
            psim.TextColored((0.7, 0.7, 0.7, 1.0), f"  {'Method':<12} {'HKS L2':>8} {'HKS Corr':>9} {'WKS L2':>8} {'WKS Corr':>9}")
            for method, res in self._descriptor_results.items():
                h_l2 = res['hks_l2_error']
                h_c = res['hks_correlation']
                w_l2 = res['wks_l2_error']
                w_c = res['wks_correlation']
                if h_c > 0.95 and w_c > 0.95:
                    color = (0.0, 1.0, 0.0, 1.0)
                elif h_c > 0.8 and w_c > 0.8:
                    color = (1.0, 1.0, 0.0, 1.0)
                else:
                    color = (1.0, 0.3, 0.0, 1.0)
                psim.TextColored(color, f"  {method:<12} {h_l2:>8.4f} {h_c:>9.4f} {w_l2:>8.4f} {w_c:>9.4f}")
        else:
            psim.Text("(Not computed yet)")

        # === SPECTRAL COMPRESSION ERROR ===
        psim.Text("")
        psim.Separator()
        psim.Text("Spectral Compression (mean vertex L2):")

        if self._compression_results:
            key_counts = [5, 10, 20, 50]
            header = f"  {'Method':<12}" + "".join(f" {'k=' + str(k):>8}" for k in key_counts)
            psim.TextColored((0.7, 0.7, 0.7, 1.0), header)
            for method, k_res in self._compression_results.items():
                line = f"  {method:<12}"
                for k in key_counts:
                    if k in k_res:
                        line += f" {k_res[k]['mean_l2']:>8.5f}"
                    else:
                        line += f" {'N/A':>8}"
                if method == 'GT':
                    color = (0.0, 1.0, 0.0, 1.0)
                else:
                    color = (1.0, 0.5, 0.0, 1.0)
                psim.TextColored(color, line)
        else:
            psim.Text("(Not computed yet)")

        # === LAPLACIAN SPARSITY COMPARISON ===
        psim.Text("")
        psim.Separator()
        psim.Text("Laplacian Sparsity Comparison:")

        if self.current_sparsity_stats:
            # Header
            psim.TextColored((0.7, 0.7, 0.7, 1.0), f"  {'Method':<20} {'NNZ':>10} {'Density':>10} {'Avg/row':>8} {'Range'}")
            psim.Separator()

            for method_key, s in self.current_sparsity_stats.items():
                if method_key == 'GT':
                    color = (0.0, 1.0, 0.0, 1.0)  # Green
                elif method_key == 'PRED':
                    color = (1.0, 0.5, 0.0, 1.0)  # Orange
                else:
                    color = (0.0, 0.7, 1.0, 1.0)  # Blue

                psim.TextColored(color, f"  {s.method_name:<20} {s.nnz:>10} {s.density_percent:>9.4f}% {s.avg_nnz_per_row:>8.1f} [{s.min_nnz_per_row}-{s.max_nnz_per_row}]")
        else:
            psim.Text("(Not computed yet)")

        # === EIGENVALUE COMPARISON ===
        psim.Text("")
        psim.Separator()
        psim.Text("Eigenvalue Comparison:")

        gt_evals = self.current_gt_data.get('gt_eigenvalues') if self.current_gt_data else None
        pred_evals = self.current_inference_result.get('predicted_eigenvalues') if self.current_inference_result else None
        robust_evals = self.current_gt_data.get('robust_eigenvalues') if self.current_gt_data else None
        nelo_evals = self.current_nelo_eigenvalues

        if gt_evals is not None:
            gt_np = gt_evals.cpu().numpy() if torch.is_tensor(gt_evals) else np.asarray(gt_evals)

            psim.TextColored((0.7, 0.7, 0.7, 1.0), f"  {'Method':<12} {'λ₁(Fiedler)':>12} {'λ_max':>12} {'Scale':>8}")
            psim.Separator()

            for label, evals, color in [
                ('GT', gt_evals, (0.0, 1.0, 0.0, 1.0)),
                ('PRED', pred_evals, (1.0, 0.5, 0.0, 1.0)),
                ('Robust', robust_evals, (0.0, 0.7, 1.0, 1.0)),
                ('NeLo', nelo_evals, (0.7, 0.3, 1.0, 1.0)),
            ]:
                if evals is not None:
                    e_np = evals.cpu().numpy() if torch.is_tensor(evals) else np.asarray(evals)
                    if len(e_np) > 1:
                        fiedler = e_np[1]
                        emax = e_np[-1]
                        # Scale ratio vs GT
                        if label != 'GT' and len(gt_np) > 1:
                            gt_pos = gt_np[1:]
                            m_pos = e_np[1:min(len(e_np), len(gt_np))]
                            valid = (gt_pos[:len(m_pos)] > 1e-10) & (m_pos > 1e-10)
                            ratio_str = f"{float(np.median(m_pos[valid] / gt_pos[:len(m_pos)][valid])):.1f}×" if valid.sum() > 0 else "N/A"
                        else:
                            ratio_str = "1.0×"
                        psim.TextColored(color, f"  {label:<12} {fiedler:>12.4f} {emax:>12.4f} {ratio_str:>8}")
        else:
            psim.Text("(Not computed yet)")

        # === VERTEX AREA COMPARISON ===
        psim.Text("")
        psim.Separator()
        psim.Text("Vertex Area Comparison:")

        gt_areas = self.current_gt_data.get('vertex_areas') if self.current_gt_data else None
        pred_areas_raw = self.current_inference_result.get('areas') if self.current_inference_result else None

        if gt_areas is not None or pred_areas_raw is not None:
            psim.TextColored((0.7, 0.7, 0.7, 1.0), f"  {'Source':<12} {'Total':>12} {'Mean':>12} {'Std':>12}")
            psim.Separator()

            if gt_areas is not None:
                psim.TextColored((0.0, 1.0, 0.0, 1.0),
                    f"  {'GT':<12} {float(np.sum(gt_areas)):>12.6f} {float(np.mean(gt_areas)):>12.6f} {float(np.std(gt_areas)):>12.6f}")

            if pred_areas_raw is not None:
                pa = pred_areas_raw.cpu().numpy() if torch.is_tensor(pred_areas_raw) else np.asarray(pred_areas_raw)
                psim.TextColored((1.0, 0.5, 0.0, 1.0),
                    f"  {'PRED':<12} {float(np.sum(pa)):>12.6f} {float(np.mean(pa)):>12.6f} {float(np.std(pa)):>12.6f}")

                if gt_areas is not None:
                    ratio = float(np.sum(pa)) / float(np.sum(gt_areas)) if float(np.sum(gt_areas)) > 1e-15 else 0.0
                    if len(pa) == len(gt_areas):
                        corr = np.corrcoef(pa, gt_areas)[0, 1]
                        corr = float(corr) if np.isfinite(corr) else 0.0
                    else:
                        corr = 0.0
                    color = (0.0, 1.0, 0.0, 1.0) if 0.8 < ratio < 1.2 else (1.0, 0.3, 0.0, 1.0)
                    psim.TextColored(color, f"  Ratio PRED/GT: {ratio:.4f}   Corr: {corr:.4f}")
        else:
            psim.Text("(Not computed yet)")

        # === MEAN CURVATURE COMPARISON ===
        psim.Text("")
        psim.Separator()
        psim.Text("Mean Curvature Comparison:")

        gt_H = self.current_gt_data.get('gt_mean_curvature') if self.current_gt_data else None
        pred_H = self.current_predicted_data.get('predicted_mean_curvature') if self.current_predicted_data else None

        if gt_H is not None or pred_H is not None:
            psim.TextColored((0.7, 0.7, 0.7, 1.0), f"  {'Source':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
            psim.Separator()

            if gt_H is not None:
                psim.TextColored((0.0, 1.0, 0.0, 1.0),
                    f"  {'GT':<12} {float(gt_H.min()):>10.4f} {float(gt_H.max()):>10.4f} {float(np.mean(gt_H)):>10.4f} {float(np.std(gt_H)):>10.4f}")

            if pred_H is not None:
                psim.TextColored((1.0, 0.5, 0.0, 1.0),
                    f"  {'PRED |H|':<12} {float(pred_H.min()):>10.4f} {float(pred_H.max()):>10.4f} {float(np.mean(pred_H)):>10.4f} {float(np.std(pred_H)):>10.4f}")

            if gt_H is not None and pred_H is not None:
                valid = np.isfinite(gt_H) & np.isfinite(pred_H)
                if valid.sum() > 2:
                    corr = np.corrcoef(np.abs(gt_H[valid]), pred_H[valid])[0, 1]
                    corr = float(corr) if np.isfinite(corr) else 0.0
                    color = (0.0, 1.0, 0.0, 1.0) if corr > 0.8 else (1.0, 0.3, 0.0, 1.0) if corr > 0.5 else (1.0, 0.0, 0.0, 1.0)
                    psim.TextColored(color, f"  |H| Correlation (GT vs PRED): {corr:.4f}")
        else:
            psim.Text("(Not computed yet)")

        # === MESH BOUNDARY INFO ===
        if self._mesh_boundary_info:
            psim.Text("")
            psim.Separator()
            info = self._mesh_boundary_info
            is_closed = info.get('is_closed', True)
            if is_closed:
                psim.TextColored((0.0, 1.0, 0.0, 1.0), "Mesh: closed (no boundary)")
            else:
                n_be = info.get('num_boundary_edges', 0)
                n_bv = info.get('num_boundary_vertices', 0)
                n_loops = info.get('num_boundary_loops', 0)
                psim.TextColored((1.0, 1.0, 0.0, 1.0),
                    f"Mesh: open ({n_be} boundary edges, {n_bv} boundary verts, {n_loops} loops)")

        # === PRED WEIGHT DISTRIBUTION ===
        if self._weight_distribution_stats:
            psim.Text("")
            psim.Separator()
            psim.Text("PRED Weight Distribution:")
            ws = self._weight_distribution_stats
            psim.Text(f"  Weights: mean={ws['weight_mean']:.4f}, std={ws['weight_std']:.4f}, "
                       f"range=[{ws['weight_min']:.4f}, {ws['weight_max']:.4f}]")
            psim.Text(f"  Entropy: {ws['entropy_ratio']:.1%} of uniform   "
                       f"Gini: {ws['gini_mean']:.3f}")
            psim.Text(f"  Top-1: {ws['top1_frac_mean']:.1%}   "
                       f"Top-3: {ws['top3_frac_mean']:.1%}   "
                       f"Top-6: {ws['top6_frac_mean']:.1%}")

    def _has_current_batch_data(self) -> bool:
        """Check if we have current batch data available for re-computation."""
        return (self.current_gt_data is not None and
                self.current_inference_result is not None and
                self.current_predicted_data is not None and
                self.current_stiffness_weights is not None and
                self.current_areas is not None and
                self.current_original_vertices is not None)

    def _compute_sparsity_for_matrix(self, matrix: scipy.sparse.spmatrix,
                                      method_name: str) -> LaplacianSparsityStats:
        """Compute sparsity statistics for a sparse matrix."""
        n = matrix.shape[0]
        csr = matrix.tocsr()
        nnz = csr.nnz
        total = n * n
        nnz_per_row = np.diff(csr.indptr)  # number of non-zeros per row

        return LaplacianSparsityStats(
            method_name=method_name,
            num_vertices=n,
            nnz=nnz,
            total_entries=total,
            sparsity_ratio=nnz / total if total > 0 else 0.0,
            density_percent=(nnz / total * 100) if total > 0 else 0.0,
            avg_nnz_per_row=nnz / n if n > 0 else 0.0,
            max_nnz_per_row=int(nnz_per_row.max()) if len(nnz_per_row) > 0 else 0,
            min_nnz_per_row=int(nnz_per_row.min()) if len(nnz_per_row) > 0 else 0
        )

    def _compute_and_store_all_sparsity_stats(self):
        """Compute and store sparsity stats for all available Laplacians, and print to console."""
        stats = {}

        # GT Laplacian (rebuild from igl)
        if HAS_IGL and self.current_gt_data is not None:
            try:
                V = self.current_gt_data['vertices'].astype(np.float64)
                F = self.current_gt_data['faces'].astype(np.int32)
                L_gt = igl.cotmatrix(V, F)
                stats['GT'] = self._compute_sparsity_for_matrix(L_gt, "GT (Cotangent)")
            except Exception as e:
                print(f"  [!] Failed to compute GT sparsity: {e}")

        # PRED Laplacian
        if self.current_inference_result is not None and self.current_inference_result.get('stiffness_matrix') is not None:
            stats['PRED'] = self._compute_sparsity_for_matrix(
                self.current_inference_result['stiffness_matrix'], "PRED (Neural)"
            )

        # Robust Laplacian (rebuild with current k)
        if not self._skip_robust and self.current_original_vertices is not None:
            try:
                k = self.reconstruction_settings.current_robust_k
                L_robust, _ = robust_laplacian.point_cloud_laplacian(
                    self.current_original_vertices, n_neighbors=k
                )
                stats['Robust'] = self._compute_sparsity_for_matrix(L_robust, f"Robust (k={k})")
            except Exception as e:
                print(f"  [!] Failed to compute Robust sparsity: {e}")

        # NeLo Laplacian
        if self.current_nelo_L is not None:
            try:
                stats['NeLo'] = self._compute_sparsity_for_matrix(
                    self.current_nelo_L, f"NeLo (k={self.nelo_k})"
                )
            except Exception as e:
                print(f"  [!] Failed to compute NeLo sparsity: {e}")

        self.current_sparsity_stats = stats
        self._print_sparsity_comparison()

    def _print_sparsity_comparison(self):
        """Print sparsity comparison table to console."""
        stats = self.current_sparsity_stats
        if not stats:
            return

        print(f"\n{'=' * 85}")
        print("LAPLACIAN SPARSITY COMPARISON")
        print(f"{'=' * 85}")
        print(f"{'Method':<20} {'N':>8} {'NNZ':>10} {'Density':>10} {'Avg NNZ/row':>12} {'Range NNZ/row'}")
        print(f"{'-' * 85}")

        for method_name, s in stats.items():
            print(f"{s.method_name:<20} {s.num_vertices:>8} {s.nnz:>10} "
                  f"{s.density_percent:>9.4f}% {s.avg_nnz_per_row:>12.1f} "
                  f"[{s.min_nnz_per_row}, {s.max_nnz_per_row}]")

        print(f"{'=' * 85}\n")

    def _get_method_L_M(self, method_key: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Return the (L, M) sparse matrices for a given method.

        Args:
            method_key: One of 'GT', 'PRED', 'Robust', 'NeLo'

        Returns:
            Tuple of (L, M) sparse matrices, or (None, None) if unavailable.
        """
        vertices = self.current_gt_data['vertices'] if self.current_gt_data else None

        if method_key == 'GT':
            if HAS_IGL and vertices is not None:
                V = vertices.astype(np.float64)
                F = self.current_gt_data['faces'].astype(np.int32)
                return -igl.cotmatrix(V, F), igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)
            return None, None

        if method_key == 'PRED':
            if self.current_inference_result is not None:
                return (self.current_inference_result.get('stiffness_matrix'),
                        self.current_inference_result.get('mass_matrix'))
            return None, None

        if method_key == 'Robust':
            if not self._skip_robust and vertices is not None:
                k = self.reconstruction_settings.current_robust_k
                return robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=k)
            return None, None

        if method_key == 'NeLo':
            return self.current_nelo_L, self.current_nelo_M

        return None, None

    def _update_sparsity_for_method(self, method_key: str):
        """Recompute sparsity stats for a single method, keeping others unchanged."""
        L, _ = self._get_method_L_M(method_key)
        if L is None:
            return

        label_map = {
            'GT': "GT (Cotangent)",
            'PRED': "PRED (Neural)",
            'Robust': f"Robust (k={self.reconstruction_settings.current_robust_k})",
            'NeLo': f"NeLo (k={self.nelo_k})",
        }
        label = label_map.get(method_key, method_key)

        try:
            self.current_sparsity_stats[method_key] = self._compute_sparsity_for_matrix(L, label)
        except Exception as e:
            print(f"  [!] Failed to compute {method_key} sparsity: {e}")

        self._print_sparsity_comparison()

    def _aggregate_greens_metrics(self):
        """Aggregate Green's function metrics across all sources into summary stats."""
        self._aggregated_greens_metrics = {}
        if not self._all_greens_results:
            return

        # Collect per-method stats across all sources
        method_keys = set()
        for src_results in self._all_greens_results.values():
            method_keys.update(src_results.keys())

        for method in method_keys:
            corrs_gt = []
            monos = []
            max_principle_pass = []
            for src_idx, src_results in self._all_greens_results.items():
                if method not in src_results:
                    continue
                r = src_results[method]
                corrs_gt.append(r.correlation_with_gt)
                monos.append(r.monotonicity_score)
                max_principle_pass.append(1.0 if r.satisfies_maximum_principle else 0.0)

            n = len(corrs_gt)
            if n == 0:
                continue
            agg = AggregatedValidationMetrics(method_name=method, n_sources=n)
            agg.greens_corr_with_gt_mean = float(np.mean(corrs_gt))
            agg.greens_corr_with_gt_std = float(np.std(corrs_gt))
            agg.greens_mono_mean = float(np.mean(monos))
            agg.greens_mono_std = float(np.std(monos))
            agg.greens_max_principle_pass_rate = float(np.mean(max_principle_pass))
            self._aggregated_greens_metrics[method] = agg

    def _aggregate_geodesic_metrics(self):
        """Aggregate geodesic metrics across all sources into summary stats."""
        self._aggregated_geodesic_metrics = {}
        if not self._all_geodesic_results:
            return

        method_keys = set()
        for src_results in self._all_geodesic_results.values():
            method_keys.update(src_results.keys())

        for method in method_keys:
            corrs = []
            maes = []
            max_errs = []
            monos = []
            for src_idx, src_results in self._all_geodesic_results.items():
                if method not in src_results:
                    continue
                r = src_results[method]
                corrs.append(r.correlation_with_reference)
                maes.append(r.mean_absolute_error)
                max_errs.append(r.max_absolute_error)
                monos.append(r.monotonicity_score)

            n = len(corrs)
            if n == 0:
                continue
            agg = AggregatedValidationMetrics(method_name=method, n_sources=n)
            agg.corr_mean = float(np.mean(corrs))
            agg.corr_std = float(np.std(corrs))
            agg.mae_mean = float(np.mean(maes))
            agg.mae_std = float(np.std(maes))
            agg.max_err_mean = float(np.mean(max_errs))
            agg.max_err_std = float(np.std(max_errs))
            agg.mono_mean = float(np.mean(monos))
            agg.mono_std = float(np.std(monos))
            self._aggregated_geodesic_metrics[method] = agg

    def _update_greens_for_method(self, method_key: str):
        """Recompute Green's function for a single method across ALL sources.

        Uses the cached source vertices from initial computation.
        """
        if self.current_gt_data is None or not self._source_indices:
            return

        vertices = self.current_gt_data['vertices']
        faces = self.current_gt_data['faces']

        L, M = self._get_method_L_M(method_key)
        if L is None:
            return

        try:
            print(f"\nRecomputing Green's function for {method_key} ({len(self._source_indices)} sources)...")
            _t_greens_start = time.perf_counter()
            device = self.current_device if method_key in ('PRED', 'NeLo') else None

            for source_idx in self._source_indices:
                result = self.compute_greens_function(L, M, source_idx, device=device)
                if result is not None:
                    greens_values, residual = result
                    gt_greens = self._greens_gt_values.get(source_idx) if self._greens_gt_values else None
                    validation = self.validate_maximum_principle(
                        greens_values, source_idx, method_key,
                        vertices=vertices, faces=faces,
                        laplacian_residual_norm=residual,
                        gt_greens_function=gt_greens
                    )

                    if source_idx not in self._all_greens_results:
                        self._all_greens_results[source_idx] = {}
                    self._all_greens_results[source_idx][method_key] = validation

                    # Cache raw values for slider refresh
                    if source_idx not in self._all_greens_values:
                        self._all_greens_values[source_idx] = {}
                    self._all_greens_values[source_idx][method_key] = greens_values

                    if method_key == 'GT':
                        if self._greens_gt_values is None:
                            self._greens_gt_values = {}
                        self._greens_gt_values[source_idx] = greens_values

            greens_elapsed = time.perf_counter() - _t_greens_start
            timing_field = {'GT': 'gt_greens_solve_time', 'PRED': 'pred_greens_solve_time',
                            'Robust': 'robust_greens_solve_time', 'NeLo': 'nelo_greens_solve_time'}.get(method_key)
            if timing_field:
                setattr(self.timing_results, timing_field, greens_elapsed)

            self._aggregate_greens_metrics()

            # Update current_greens_results to first source for Polyscope display
            first_src = self._source_indices[0]
            if first_src in self._all_greens_results:
                self.current_greens_results = self._all_greens_results[first_src]

            # Update Polyscope visualization for first source
            display_src = self._source_indices[self._current_source_display_idx]
            mesh_structure = self.current_mesh_structure
            if mesh_structure is not None and not self._quantitative_mode:
                if display_src in self._all_greens_results and method_key in self._all_greens_results[display_src]:
                    # Recompute greens_values for display source
                    result = self.compute_greens_function(L, M, display_src, device=device)
                    if result is not None:
                        greens_values, _ = result
                        mesh_structure.add_scalar_quantity(
                            name=f"J1 Green's Function - {method_key}",
                            values=greens_values, enabled=False, cmap='coolwarm'
                        )

            agg = self._aggregated_greens_metrics.get(method_key)
            if agg:
                print(f"  {method_key}: GT corr={agg.greens_corr_with_gt_mean:.4f}±{agg.greens_corr_with_gt_std:.4f}, "
                      f"max principle pass={agg.greens_max_principle_pass_rate:.0%} ({agg.n_sources} sources)")

        except Exception as e:
            print(f"  [!] Green's function recomputation for {method_key} failed: {e}")
            import traceback
            traceback.print_exc()

    def _compute_geodesic_single_source(self, method_key: str, source_idx: int,
                                          vertices: np.ndarray, L, M,
                                          mesh_grad_op=None, mesh_face_areas=None) -> Optional[np.ndarray]:
        """Compute heat method geodesic distances from a single source for one method.

        Returns distances array (N,) or None if computation fails.
        """
        n = len(vertices)
        operator_mode = getattr(self.current_model, '_operator_mode', 'stiffness') if self.current_model else 'stiffness'

        if method_key == 'GT':
            if HAS_IGL and L is not None and M is not None and mesh_grad_op is not None:
                return compute_heat_geodesic_mesh(
                    L=L, M=M, grad_op=mesh_grad_op, face_areas=mesh_face_areas,
                    source_idx=source_idx, n_vertices=n
                )

        elif method_key == 'PRED':
            if L is not None and M is not None:
                if operator_mode == "gradient" and self.current_learned_gradient_op is not None:
                    return compute_heat_geodesic_learned(
                        S=L, M=M, G=self.current_learned_gradient_op,
                        source_idx=source_idx, n_vertices=n,
                        device=self.current_device
                    )
                elif self._pc_grad_op is not None and self._pc_div_op is not None:
                    return compute_heat_geodesic_pointcloud(
                        L=L, M=M, grad_op=self._pc_grad_op, div_op=self._pc_div_op,
                        source_idx=source_idx, n_vertices=n,
                        device=self.current_device
                    )

        elif method_key == 'Robust':
            if self._robust_geodesic_heat:
                if L is not None and M is not None and HAS_PCDIFF:
                    robust_k = self.reconstruction_settings.current_robust_k
                    robust_edge_index = knn_graph(vertices, k=robust_k)
                    robust_grad_op, robust_div_op = build_pointcloud_grad_div_operators(vertices, robust_edge_index)
                    return compute_heat_geodesic_pointcloud(
                        L=L, M=M, grad_op=robust_grad_op, div_op=robust_div_op,
                        source_idx=source_idx, n_vertices=n
                    )
            else:
                import potpourri3d as pp3d
                pc_solver = pp3d.PointCloudHeatSolver(vertices)
                return pc_solver.compute_distance(source_idx)

        elif method_key == 'NeLo':
            if L is not None and M is not None and self._pc_grad_op is not None:
                return compute_heat_geodesic_pointcloud(
                    L=L, M=M, grad_op=self._pc_grad_op, div_op=self._pc_div_op,
                    source_idx=source_idx, n_vertices=n,
                    device=self.current_device
                )

        return None

    def _update_geodesics_for_method(self, method_key: str):
        """Recompute heat method geodesics for a single method across ALL sources.

        Times gradient operator construction separately from solves.
        """
        if self.current_gt_data is None or not self._source_indices or not self._exact_geodesics:
            print(f"  [!] Multi-source geodesic state not available, skipping {method_key}")
            return

        vertices = self.current_gt_data['vertices'].astype(np.float64)
        n = len(vertices)
        L, M = self._get_method_L_M(method_key)

        result_key = method_key
        if method_key == "Robust" and not self._robust_geodesic_heat:
            result_key = "Robust (pp3d)"

        grad_op_elapsed = 0.0
        mesh_grad_op = None
        mesh_face_areas = None

        try:
            # Build gradient operator once (not per-source)
            if method_key == 'GT' and HAS_IGL and L is not None:
                faces = self.current_gt_data['faces'].astype(np.int32)
                _t_grad = time.perf_counter()
                mesh_grad_op = igl.grad(vertices, faces)
                mesh_face_areas = igl.doublearea(vertices, faces).flatten() / 2.0
                grad_op_elapsed = time.perf_counter() - _t_grad

            n_sources = len(self._source_indices)
            print(f"\nRecomputing Heat Method geodesic for {method_key} ({n_sources} sources)...")

            operator_mode = getattr(self.current_model, '_operator_mode', 'stiffness') if self.current_model else 'stiffness'

            _t_solve = time.perf_counter()

            # Use batch solve for PRED with gradient operator (factorize once, solve N times)
            if (method_key == 'PRED' and operator_mode == 'gradient'
                    and self.current_learned_gradient_op is not None and L is not None and M is not None):
                all_distances = compute_heat_geodesic_learned_batch(
                    S=L, M=M, G=self.current_learned_gradient_op,
                    source_indices=self._source_indices, n_vertices=n,
                    device=self.current_device,
                )
                for i, source_idx in enumerate(self._source_indices):
                    distances = all_distances[i]
                    exact_distances = self._exact_geodesics.get(source_idx)
                    if distances is not None and exact_distances is not None:
                        metrics = compute_geodesic_metrics(distances, exact_distances)
                        result = HeatMethodGeodesicResult(
                            method_name=result_key,
                            source_vertex_idx=source_idx,
                            num_vertices=n,
                            min_distance=float(distances.min()),
                            max_distance=float(distances.max()),
                            mean_distance=float(distances.mean()),
                            distance_at_source=float(distances[source_idx]),
                            correlation_with_reference=float(metrics.correlation),
                            mean_absolute_error=float(metrics.mae_normalized),
                            max_absolute_error=float(metrics.max_error_normalized),
                            relative_error_percent=float(metrics.mae_normalized * 100),
                            monotonicity_score=float(metrics.monotonicity),
                        )
                        if source_idx not in self._all_geodesic_results:
                            self._all_geodesic_results[source_idx] = {}
                        self._all_geodesic_results[source_idx][result_key] = result
                        if source_idx not in self._all_geodesic_distances:
                            self._all_geodesic_distances[source_idx] = {}
                        self._all_geodesic_distances[source_idx][result_key] = distances
            else:
                # Per-source loop for other methods (GT, Robust, NeLo, PRED stiffness mode)
                for source_idx in self._source_indices:
                    exact_distances = self._exact_geodesics.get(source_idx)
                    if exact_distances is None:
                        continue

                    distances = self._compute_geodesic_single_source(
                        method_key, source_idx, vertices, L, M,
                        mesh_grad_op=mesh_grad_op, mesh_face_areas=mesh_face_areas
                    )

                    if distances is not None:
                        metrics = compute_geodesic_metrics(distances, exact_distances)
                        result = HeatMethodGeodesicResult(
                            method_name=result_key,
                            source_vertex_idx=source_idx,
                            num_vertices=n,
                            min_distance=float(distances.min()),
                            max_distance=float(distances.max()),
                            mean_distance=float(distances.mean()),
                            distance_at_source=float(distances[source_idx]),
                            correlation_with_reference=float(metrics.correlation),
                            mean_absolute_error=float(metrics.mae_normalized),
                            max_absolute_error=float(metrics.max_error_normalized),
                            relative_error_percent=float(metrics.mae_normalized * 100),
                            monotonicity_score=float(metrics.monotonicity),
                        )

                        if source_idx not in self._all_geodesic_results:
                            self._all_geodesic_results[source_idx] = {}
                        self._all_geodesic_results[source_idx][result_key] = result

                        # Cache raw distances for slider refresh
                        if source_idx not in self._all_geodesic_distances:
                            self._all_geodesic_distances[source_idx] = {}
                        self._all_geodesic_distances[source_idx][result_key] = distances

            solve_elapsed = time.perf_counter() - _t_solve

            self._aggregate_geodesic_metrics()

            # Update current_heat_geodesic_results to first source for Polyscope
            first_src = self._source_indices[0]
            if first_src in self._all_geodesic_results:
                self.current_heat_geodesic_results = self._all_geodesic_results[first_src]

            # Update Polyscope visualization for display source
            display_src = self._source_indices[self._current_source_display_idx]
            mesh_structure = self.current_mesh_structure
            if mesh_structure is not None and not self._quantitative_mode:
                exact_distances = self._exact_geodesics.get(display_src)
                distances = self._compute_geodesic_single_source(
                    method_key, display_src, vertices, L, M,
                    mesh_grad_op=mesh_grad_op, mesh_face_areas=mesh_face_areas
                )
                if distances is not None and exact_distances is not None:
                    d_norm = normalize_distances(distances)
                    is_bad = distances.max() > 1e3
                    label_suffix = " [!]" if is_bad else ""
                    cmap = 'hot' if is_bad else 'viridis'
                    mesh_structure.add_scalar_quantity(
                        name=f"K1 Geodesic (norm) - {result_key}{label_suffix}",
                        values=d_norm, enabled=False, cmap=cmap
                    )
                    exact_norm = normalize_distances(exact_distances)
                    error = np.abs(d_norm - exact_norm)
                    mesh_structure.add_scalar_quantity(
                        name=f"K2 Geodesic Error - {result_key}",
                        values=error, enabled=False, cmap='reds'
                    )

            # Print aggregated summary
            agg = self._aggregated_geodesic_metrics.get(result_key)
            if agg:
                print(f"  {result_key}: corr={agg.corr_mean:.4f}±{agg.corr_std:.4f}, "
                      f"MAE={agg.mae_mean:.4f}±{agg.mae_std:.4f} ({agg.n_sources} sources)")

            # Store timing
            if grad_op_elapsed > 0:
                grad_op_field = {'GT': 'gt_grad_op_time', 'PRED': 'pred_grad_op_time',
                                 'NeLo': 'nelo_grad_op_time'}.get(method_key)
                if grad_op_field:
                    setattr(self.timing_results, grad_op_field, grad_op_elapsed)

            solve_field = {'GT': 'gt_geodesic_solve_time', 'PRED': 'pred_geodesic_solve_time',
                           'Robust': 'robust_geodesic_solve_time', 'NeLo': 'nelo_geodesic_solve_time'}.get(method_key)
            if solve_field:
                setattr(self.timing_results, solve_field, solve_elapsed)

        except Exception as e:
            print(f"  [!] Geodesic recomputation for {method_key} failed: {e}")
            import traceback
            traceback.print_exc()

    def _recompute_validations_after_k_change(self, changed_method: str):
        """Recompute downstream validations ONLY for the method whose k changed.

        This replaces the previous approach of recomputing everything for all
        methods, which was wasteful when only one method's k changed.

        Args:
            changed_method: The method whose k changed ('PRED', 'Robust', or 'NeLo').
        """
        if self.current_gt_data is None or self.current_inference_result is None:
            print("[!] No current data available for validation recomputation")
            return

        print(f"\n--- Recomputing validations for {changed_method} only ---")

        # Rebuild pcdiff operators if PRED's kNN changed (they depend on PRED connectivity)
        if changed_method == 'PRED' and HAS_PCDIFF and self.current_vertex_indices is not None:
            try:
                vertices = self.current_gt_data['vertices'].astype(np.float64)
                vertex_indices_np = self.current_vertex_indices.cpu().numpy()
                center_indices_np = self.current_center_indices.cpu().numpy()
                k = len(vertex_indices_np) // len(center_indices_np)
                edge_index = edge_index_from_knn_indices(vertex_indices_np, center_indices_np, k)
                self._pc_grad_op, self._pc_div_op = build_pointcloud_grad_div_operators(vertices, edge_index)
                print(f"  Rebuilt pcdiff operators for new PRED k={k}")
            except Exception as e:
                print(f"  [!] Failed to rebuild pcdiff operators: {e}")

        self._update_sparsity_for_method(changed_method)
        self._update_greens_for_method(changed_method)
        self._update_geodesics_for_method(changed_method)

        # Recompute new metrics (probe MSE, descriptors, compression)
        # These are cheap (no solves) and depend on L/M/eigenpairs which changed
        print(f"  Recomputing probe function MSE...")
        try:
            self._compute_probe_function_mse()
        except Exception as e:
            print(f"  [!] Probe function MSE failed: {e}")

        print(f"  Recomputing HKS/WKS descriptors...")
        try:
            # Recompute Weyl calibration (eigenvalues changed)
            self._weyl_calibration_factor = None
            self._weyl_calibrated_evals = None
            self._compute_weyl_calibration()
            self._compute_descriptor_comparison()
        except Exception as e:
            print(f"  [!] Descriptor comparison failed: {e}")

        print(f"  Recomputing spectral compression error...")
        try:
            self._compute_spectral_compression_error()
        except Exception as e:
            print(f"  [!] Spectral compression error failed: {e}")

    def _clear_stored_references(self):
        """Clear stored references from previous batch to free memory.

        This helps ensure consistent timing across batches by preventing
        memory accumulation from affecting performance.
        """
        # Clear GPU tensors
        self.current_stiffness_weights = None
        self.current_areas = None
        self._current_vertices_tensor = None
        self.current_vertex_indices = None
        self.current_center_indices = None
        self.current_batch_indices = None

        # Clear numpy arrays and dicts
        self.current_gt_data = None
        self.current_inference_result = None
        self.current_predicted_data = None
        self.current_original_vertices = None
        self.current_sparsity_stats = {}

        # Clear cached validation state
        self._source_indices = None
        self._exact_geodesics = None
        self._greens_gt_values = None
        self._all_greens_results = {}
        self._all_geodesic_results = {}
        self._all_greens_values = {}
        self._all_geodesic_distances = {}
        self._aggregated_geodesic_metrics = {}
        self._aggregated_greens_metrics = {}
        self._current_source_display_idx = 0
        self._pc_grad_op = None
        self._pc_div_op = None
        self._probe_function_results = {}
        self._descriptor_results = {}
        self._compression_results = {}
        self._weyl_calibration_factor = None
        self._weyl_calibrated_evals = None
        self._weight_distribution_stats = {}
        self._mesh_boundary_info = {}
        self._weight_nbr_sample_indices = None

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache after gc — only in visual mode.
        # In quantitative mode, keep allocator warm for fast subsequent inferences.
        if torch.cuda.is_available() and not self._quantitative_mode:
            torch.cuda.empty_cache()

    def _compute_predicted_vertex_areas(self, inference_result: Dict) -> Optional[np.ndarray]:
        """Compute predicted vertex areas based on current settings.

        Uses the areas predicted by the model's area head.
        """
        if not self.reconstruction_settings.use_pred_areas:
            return None

        if inference_result.get('areas') is not None:
            areas = inference_result['areas']
            predicted_vertex_areas = areas.cpu().numpy() if torch.is_tensor(areas) else areas
            print(f"Using predicted areas from model (range: [{predicted_vertex_areas.min():.6f}, {predicted_vertex_areas.max():.6f}])")
            return predicted_vertex_areas
        else:
            print(f"[!] No predicted areas available, falling back to standard reconstruction")
            return None

    def _compute_all_reconstructions(self, gt_data: Dict, inference_result: Dict) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Compute GT, predicted, robust-laplacian, and NeLo reconstructions."""
        gt_reconstructions = []
        pred_reconstructions = []
        robust_reconstructions = []
        nelo_reconstructions = []

        # Compute GT reconstructions (always use area weighting with PyFM areas)
        if gt_data.get('gt_eigenvectors') is not None:
            gt_reconstructions = self.compute_mesh_reconstruction(
                gt_data['vertices'],
                gt_data['gt_eigenvectors'],
                gt_data.get('gt_eigenvalues'),
                self.config.num_eigenvectors_to_show,
                vertex_areas=gt_data.get('vertex_areas')  # Always use PyFM areas
            )

        # Compute predicted reconstructions (respecting current setting)
        if inference_result['predicted_eigenvectors'] is not None:
            predicted_vertex_areas = self._compute_predicted_vertex_areas(inference_result)
            pred_reconstructions = self.compute_mesh_reconstruction(
                gt_data['vertices'],  # Use original vertices as reference
                inference_result['predicted_eigenvectors'],
                inference_result['predicted_eigenvalues'],
                self.config.num_eigenvectors_to_show,
                vertex_areas=predicted_vertex_areas  # None for standard, or predicted areas from model
            )

        # Compute robust-laplacian reconstructions (always use area weighting with robust areas)
        if gt_data.get('robust_eigenvectors') is not None:
            robust_reconstructions = self.compute_mesh_reconstruction(
                gt_data['vertices'],
                gt_data['robust_eigenvectors'],
                gt_data.get('robust_eigenvalues'),
                self.config.num_eigenvectors_to_show,
                vertex_areas=gt_data.get('robust_vertex_areas')  # Use robust-laplacian areas
            )

        # Compute NeLo reconstructions
        if self.current_nelo_eigenvectors is not None:
            nelo_vertex_areas = self.current_nelo_M.diagonal() if self.current_nelo_M is not None else None
            nelo_reconstructions = self.compute_mesh_reconstruction(
                gt_data['vertices'],
                self.current_nelo_eigenvectors,
                self.current_nelo_eigenvalues,
                self.config.num_eigenvectors_to_show,
                vertex_areas=nelo_vertex_areas
            )

        return gt_reconstructions, pred_reconstructions, robust_reconstructions, nelo_reconstructions

    def _update_mesh_reconstructions(self, gt_data: Dict, inference_result: Dict):
        """Complete pipeline for computing and visualizing mesh reconstructions."""
        print(f"Computing and visualizing mesh reconstructions...")

        # Compute reconstructions
        gt_reconstructions, pred_reconstructions, robust_reconstructions, nelo_reconstructions = self._compute_all_reconstructions(
            gt_data, inference_result
        )

        # Visualize reconstructions
        if gt_reconstructions or pred_reconstructions or robust_reconstructions or nelo_reconstructions:
            self.visualize_mesh_reconstructions(
                gt_data['faces'],
                gt_reconstructions,
                pred_reconstructions,
                robust_reconstructions,
                gt_data.get('gt_eigenvalues'),
                inference_result['predicted_eigenvalues'],
                gt_data.get('robust_eigenvalues'),
                nelo_reconstructions=nelo_reconstructions,
                nelo_eigenvalues=self.current_nelo_eigenvalues
            )
            print("[OK] Mesh reconstructions completed")
        else:
            print("[!] No reconstructions to visualize")

    def _recompute_knn_connectivity_for_k(self, new_k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute k-NN connectivity for new k value.

        Args:
            new_k: New number of neighbors per patch

        Returns:
            Tuple of (vertex_indices, center_indices, batch_indices) for new k
        """
        print(f"  Recomputing k-NN connectivity for k={new_k}...")

        vertices = self.current_original_vertices
        num_vertices = len(vertices)

        # Build k-NN index using cKDTree
        from scipy.spatial import cKDTree
        tree = cKDTree(vertices)
        _, neighbor_indices = tree.query(vertices, k=new_k + 1, workers=-1)  # (N, k+1)
        neighbor_indices_filtered = neighbor_indices[:, 1:]  # (N, k) — drop self

        # Create new connectivity tensors
        vertex_indices = torch.from_numpy(neighbor_indices_filtered.flatten()).long()  # Shape: (N*k,)
        center_indices = torch.from_numpy(np.arange(num_vertices)).long()  # Shape: (N,)
        batch_indices = torch.from_numpy(np.repeat(range(num_vertices), new_k)).long()  # Shape: (N*k,)

        print(f"  Created connectivity: {len(vertex_indices)} total points for {num_vertices} patches")
        return vertex_indices, center_indices, batch_indices

    def _recompute_pred_laplacian_with_k(self, new_k: int):
        """
        Recompute PRED stiffness/mass matrices and eigendata with new k.

        Args:
            new_k: New number of neighbors per patch
        """
        print(f"  Recomputing PRED matrices with k={new_k}...")

        # Get new k-NN connectivity
        new_vertex_indices, new_center_indices, new_batch_indices = self._recompute_knn_connectivity_for_k(new_k)

        # Clone stiffness weights for modification
        corrected_weights = self.current_stiffness_weights.clone()

        # Resize stiffness weights to match new k if necessary
        current_k = corrected_weights.shape[1]
        if new_k != current_k:
            if new_k < current_k:
                # Truncate to new_k
                corrected_weights = corrected_weights[:, :new_k]
                print(f"  Truncated stiffness weights from {current_k} to {new_k}")
            else:
                # Pad with zeros or repeat existing weights
                num_patches = corrected_weights.shape[0]
                padding_size = new_k - current_k
                # Use mean of existing weights for padding
                mean_weights = corrected_weights.mean(dim=1, keepdim=True)
                padding = mean_weights.expand(-1, padding_size)
                corrected_weights = torch.cat([corrected_weights, padding], dim=1)
                print(f"  Padded stiffness weights from {current_k} to {new_k} using mean weights")

        # Create attention mask (all True for uniform k)
        num_patches = corrected_weights.shape[0]
        attention_mask = torch.ones(num_patches, new_k, dtype=torch.bool, device=corrected_weights.device)

        # Assemble new stiffness and mass matrices
        top_k = self.reconstruction_settings.pred_top_k if self.reconstruction_settings.enable_top_k_pruning else None
        sym_policy = self.reconstruction_settings.symmetry_policy
        new_stiffness_matrix, new_mass_matrix = assemble_stiffness_and_mass_matrices(
            stiffness_weights=corrected_weights,
            areas=self.current_areas,
            attention_mask=attention_mask,
            vertex_indices=new_vertex_indices,
            center_indices=new_center_indices,
            batch_indices=new_batch_indices,
            top_k=top_k,
            symmetry_policy=sym_policy
        )

        print(f"  Assembled new stiffness matrix: {new_stiffness_matrix.shape} ({new_stiffness_matrix.nnz} non-zeros)")

        # Compute new eigendecomposition using generalized problem
        new_eigenvalues, new_eigenvectors = self.compute_eigendecomposition(
            new_stiffness_matrix, k=self.config.num_eigenvectors_to_show, mass_matrix=new_mass_matrix
        )

        if new_eigenvalues is not None:
            print(f"  Computed {len(new_eigenvalues)} new eigenvalues")
            print(f"  New eigenvalue range: [{new_eigenvalues[0]:.2e}, {new_eigenvalues[-1]:.6f}]")

            # Update current inference result
            self.current_inference_result['stiffness_matrix'] = new_stiffness_matrix
            self.current_inference_result['mass_matrix'] = new_mass_matrix
            self.current_inference_result['predicted_eigenvalues'] = new_eigenvalues
            self.current_inference_result['predicted_eigenvectors'] = new_eigenvectors
        else:
            print(f"  Failed to compute eigendecomposition for k={new_k}")

    def _extract_patches_for_mesh_with_k(self, vertices: np.ndarray, k: int) -> Data:
        """
        Extract k-NN patches from mesh vertices for model inference.

        Uses torch_geometric.nn.pool.knn_graph on GPU for fast neighbor search,
        falling back to sklearn on CPU if unavailable.

        Args:
            vertices: Mesh vertices of shape (N, 3)
            k: Number of nearest neighbors per patch

        Returns:
            Data object ready for model inference
        """
        from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData

        num_vertices = len(vertices)

        # Try GPU kNN via PyG (much faster for large meshes)
        device = self.current_device if self.current_device is not None else torch.device('cpu')
        use_gpu_knn = device.type == 'cuda'

        if use_gpu_knn:
            try:
                from torch_geometric.nn.pool import knn_graph as pyg_knn_graph

                # Move vertices to GPU
                pos_gpu = torch.from_numpy(vertices).float().to(device)

                # knn_graph returns edge_index (2, N*k): [0]=center, [1]=neighbor
                edge_index = pyg_knn_graph(pos_gpu, k=k, loop=False, flow='source_to_target')

                # Reshape neighbor indices to (N, k)
                # edge_index[1] contains neighbor vertex indices, grouped by center
                neighbor_indices_filtered = edge_index[1].reshape(num_vertices, k).cpu().numpy()

                if not getattr(self, '_logged_gpu_knn', False):
                    print("  [kNN] Using torch_geometric knn_graph on GPU")
                    self._logged_gpu_knn = True

            except (ImportError, RuntimeError) as e:
                if not getattr(self, '_logged_gpu_knn_fail', False):
                    print(f"  [kNN] GPU knn_graph failed ({e}), using sklearn CPU fallback")
                    self._logged_gpu_knn_fail = True
                use_gpu_knn = False

        if not use_gpu_knn:
            from scipy.spatial import cKDTree

            tree = cKDTree(vertices)
            _, neighbor_indices = tree.query(vertices, k=k + 1, workers=-1)  # (N, k+1)
            neighbor_indices_filtered = neighbor_indices[:, 1:]  # (N, k) — drop self

        # Extract neighbor positions and translate to origin
        all_neighbor_positions = vertices[neighbor_indices_filtered]  # (N, k, 3)
        center_positions_expanded = vertices[:, np.newaxis, :]  # (N, 1, 3)
        patch_positions = all_neighbor_positions - center_positions_expanded  # (N, k, 3)

        # Flatten for PyG
        all_positions = patch_positions.reshape(-1, 3)  # (N*k, 3)
        all_neighbor_indices = neighbor_indices_filtered.flatten()  # (N*k,)
        all_center_indices = np.arange(num_vertices)  # (N,)
        batch_indices = np.repeat(range(num_vertices), k)  # (N*k,)

        # Convert to tensors
        pos_tensor = torch.from_numpy(all_positions).float()
        features_tensor = pos_tensor.clone()  # XYZ as features
        patch_idx_tensor = torch.from_numpy(batch_indices).long()
        vertex_indices_tensor = torch.from_numpy(all_neighbor_indices).long()
        center_indices_tensor = torch.from_numpy(all_center_indices).long()

        data = MeshPatchData(
            pos=pos_tensor,
            x=features_tensor,
            patch_idx=patch_idx_tensor,
            vertex_indices=vertex_indices_tensor,
            center_indices=center_indices_tensor
        )

        return data

    def _compute_robust_laplacian_with_k(self, vertices: np.ndarray, k: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute robust-laplacian eigendecomposition using point_cloud_laplacian with k neighbors.

        This uses point cloud laplacian (k-NN based) instead of mesh laplacian,
        making it comparable to PRED which also uses k-NN connectivity.

        Args:
            vertices: Mesh vertices of shape (N, 3)
            k: Number of neighbors for point cloud laplacian

        Returns:
            Tuple of (eigenvalues, eigenvectors, vertex_areas) or (None, None, None) on failure
        """
        try:
            import robust_laplacian

            print(f"  Computing robust point_cloud_laplacian with k={k}...")

            # === TIME: Robust Laplacian assembly (k-NN + weight computation + matrix construction) ===
            t_robust_start = time.perf_counter()

            # Use point_cloud_laplacian with n_neighbors=k (same as PRED k-NN)
            L_robust, M_robust = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=k)

            t_robust_end = time.perf_counter()
            robust_laplacian_time = t_robust_end - t_robust_start

            # Store timing
            self.timing_results.robust_matrix_assembly_time = robust_laplacian_time
            self.timing_results.current_k = k
            print(f"  [TIMING] Robust matrix assembly: {robust_laplacian_time * 1000:.2f} ms")

            # Compute eigendecomposition (NOT timed - same for all methods)
            eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
                L_robust, self.config.num_eigenvectors_to_show, mass_matrix=M_robust
            )

            # Extract vertex areas from mass matrix diagonal
            vertex_areas = np.array(M_robust.diagonal()).flatten()

            print(f"  Robust eigenvalue range: [{eigenvalues[0]:.2e}, {eigenvalues[-1]:.6f}]")

            return eigenvalues, eigenvectors, vertex_areas

        except Exception as e:
            print(f"  Warning: Failed to compute robust-laplacian with k={k}: {e}")
            return None, None, None

    def _compute_gt_matrices_timed(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[float, Optional[scipy.sparse.csr_matrix], Optional[scipy.sparse.csr_matrix]]:
        """
        Compute GT Laplacian (cotangent) and mass matrices using igl, with timing.

        This times ONLY the matrix construction, not eigendecomposition,
        for fair comparison with PRED and Robust.

        Args:
            vertices: Mesh vertices of shape (N, 3)
            faces: Mesh faces of shape (F, 3)

        Returns:
            Tuple of (assembly_time_seconds, L_matrix, M_matrix) or (0, None, None) on failure
        """
        if not HAS_IGL:
            print("  [!] igl not available for GT matrix timing")
            return 0.0, None, None

        try:
            # Convert to float64 for igl
            V = vertices.astype(np.float64)
            F = faces.astype(np.int32)

            # === TIME: GT cotangent Laplacian + mass matrix assembly ===
            t_start = time.perf_counter()

            # Compute cotangent Laplacian (stiffness matrix)
            L = igl.cotmatrix(V, F)

            # Compute mass matrix (barycentric vertex areas)
            M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)

            t_end = time.perf_counter()
            assembly_time = t_end - t_start

            self.timing_results.gt_matrix_assembly_time = assembly_time
            print(f"  [TIMING] GT matrix assembly (igl.cotmatrix + massmatrix): {assembly_time * 1000:.2f} ms")

            return assembly_time, L, M

        except Exception as e:
            print(f"  Warning: Failed to compute GT matrices with igl: {e}")
            return 0.0, None, None

    def load_nelo_pipeline(self, ckpt_path: str, device: str = "cuda"):
        """
        Load NeLo from checkpoint using the inlined architecture.
        No external NeLo repo required — everything is self-contained above.
        """
        print(f"  Loading NeLo checkpoint: {ckpt_path}")
        pipeline = _NeLo_Pipeline.load_from_checkpoint(ckpt_path, map_location=device)
        pipeline.eval()
        pipeline.to(device)
        # Verify the two assumptions forward_inference hardcodes
        assert _nelo_global_config.gnn_input_signal == "all_one",             f"NeLo checkpoint uses gnn_input_signal={_nelo_global_config.gnn_input_signal!r}, but forward_inference hardcodes all_one"
        assert _nelo_global_config.use_vertex_mass == True,             f"NeLo checkpoint has use_vertex_mass=False, but forward_inference always calls mass_decoder"
        self.nelo_pipeline = pipeline
        print(f"  NeLo pipeline loaded (k will be set from config)")

    def perform_nelo_inference(self,
                               vertices: np.ndarray,
                               num_eigenvectors: int,
                               device) -> Optional[Dict[str, Any]]:
        """
        Run NeLo inference on an (N,3) vertex array using the inlined architecture.

        Timing breakdown mirrors PRED exactly:
          nelo_graph_tree_time      ↔  pred_patch_extraction_time   (KNN graph build)
          nelo_model_inference_time ↔  pred_model_inference_time    (forward pass)
          nelo_matrix_assembly_time ↔  pred_matrix_assembly_time    (sparse L, M)

        Returns dict with keys: L, M, eigenvalues, eigenvectors, timing — or None on failure.
        """
        if self.nelo_pipeline is None:
            return None

        import scipy.sparse as sp
        import scipy.sparse.linalg as sla

        device_str = str(device)
        N = vertices.shape[0]
        timing = {}

        # ---- sub-step 1: GraphTree construction (KNN build) ----
        t0 = time.perf_counter()
        graph_tree = _nelo_construct_graph_tree(vertices, k=self.nelo_k)
        graph_tree = graph_tree.to(device_str)
        if 'cuda' in device_str:
            torch.cuda.synchronize()
        timing['graph_tree'] = time.perf_counter() - t0
        print(f"  [NeLo] [TIMING] GraphTree (k={self.nelo_k}): {timing['graph_tree'] * 1000:.2f} ms")

        # ---- sub-step 2: Network forward pass ----
        t1 = time.perf_counter()
        with torch.no_grad():
            edge_w, mass, edge_index = self.nelo_pipeline.forward_inference(graph_tree)
        if 'cuda' in device_str:
            torch.cuda.synchronize()
        timing['forward'] = time.perf_counter() - t1
        print(f"  [NeLo] [TIMING] Forward pass: {timing['forward'] * 1000:.2f} ms")

        # ---- sub-step 3: Sparse (L, M) assembly ----
        t2 = time.perf_counter()
        rows = edge_index[0].cpu().numpy()
        cols = edge_index[1].cpu().numpy()
        # NeLo outputs non-positive weights (clamped with -relu(-x)).
        # Negate so w > 0, giving the standard positive semi-definite L = D - W.
        w    = -edge_w.squeeze().detach().cpu().numpy()
        mass_np = mass.squeeze().detach().cpu().numpy()

        diag_vals = np.zeros(N)
        np.add.at(diag_vals, rows, w)
        L = (sp.coo_matrix((-w, (rows, cols)), shape=(N, N)) + sp.diags(diag_vals)).tocsr()

        mass_mean = mass_np.mean()
        mass_rel = np.maximum(mass_np / mass_mean, 0.01) if mass_mean > 0 else np.ones(N)
        M = sp.diags(mass_rel).tocsr()

        timing['assembly'] = time.perf_counter() - t2
        timing['total']    = timing['graph_tree'] + timing['forward'] + timing['assembly']
        print(f"  [NeLo] [TIMING] Sparse assembly: {timing['assembly'] * 1000:.2f} ms")
        print(f"  [NeLo] [TIMING] Total (L,M):     {timing['total'] * 1000:.2f} ms")

        # ---- Eigendecomposition (using shared function, consistent with GT/PRED/Robust) ----
        eigenvalues, eigenvectors = None, None
        try:
            eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
                L, num_eigenvectors, mass_matrix=M
            )
            print(f"  [NeLo] Eigenvalue range: [{eigenvalues[0]:.2e}, {eigenvalues[-1]:.6f}]")
        except Exception as e:
            print(f"  [NeLo] eigendecomposition failed: {e}")

        return {
            'L': L,
            'M': M,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'timing': timing,
        }

    def _update_nelo_with_new_k(self, new_k: int):
        """Re-run NeLo inference with new k and update all stored state."""
        print(f"\n{'=' * 60}")
        print(f"UPDATING NeLo WITH NEW k={new_k}")
        print('=' * 60)

        if self.current_original_vertices is None or self.nelo_pipeline is None:
            print("[!] NeLo pipeline or vertices not available")
            return

        self.nelo_k = new_k
        nelo_result = self.perform_nelo_inference(
            self.current_original_vertices,
            self.config.num_eigenvectors_to_show,
            self.current_device,
        )
        if nelo_result is None:
            return

        self.current_nelo_L = nelo_result['L']
        self.current_nelo_M = nelo_result['M']
        self.current_nelo_eigenvectors = nelo_result['eigenvectors']
        self.current_nelo_eigenvalues  = nelo_result['eigenvalues']
        t = nelo_result['timing']
        self.timing_results.nelo_graph_tree_time      = t['graph_tree']
        self.timing_results.nelo_model_inference_time = t['forward']
        self.timing_results.nelo_matrix_assembly_time = t['assembly']
        self.timing_results.nelo_total_time           = t['total']
        self.timing_results.nelo_k                    = self.nelo_k

        # Update cosine similarities
        if self.current_gt_data is not None and nelo_result['eigenvectors'] is not None:
            self.current_cosine_similarities_nelo = self._compute_eigenvector_cosine_similarities(
                self.current_gt_data.get('gt_eigenvectors'),
                nelo_result['eigenvectors']
            )

        # Update visualizations (reconstructions include NeLo)
        if self._has_current_batch_data():
            self._remove_existing_reconstructions()
            self._update_mesh_reconstructions(self.current_gt_data, self.current_inference_result)

        # Recompute downstream validations for NeLo only
        print(f"Recomputing downstream validations for NeLo...")
        self._recompute_validations_after_k_change("NeLo")

        self._print_timing_summary()

    def _refresh_source_display(self):
        """Update Polyscope visualization when source slider changes. No recomputation."""
        if not self._source_indices:
            return
        src_idx = self._source_indices[self._current_source_display_idx]
        mesh = self.current_mesh_structure
        if mesh is None:
            return

        # Update Green's function coloring
        for method, values in self._all_greens_values.get(src_idx, {}).items():
            mesh.add_scalar_quantity(
                f"J1 Green's Function - {method}", values, enabled=False, cmap='coolwarm'
            )

        # Update geodesic coloring
        exact = self._exact_geodesics.get(src_idx) if self._exact_geodesics else None
        for method, dists in self._all_geodesic_distances.get(src_idx, {}).items():
            d_norm = normalize_distances(dists)
            is_bad = dists.max() > 1e3
            label_suffix = " [!]" if is_bad else ""
            cmap = 'hot' if is_bad else 'viridis'
            mesh.add_scalar_quantity(
                f"K1 Geodesic (norm) - {method}{label_suffix}",
                d_norm, enabled=False, cmap=cmap
            )
            if exact is not None:
                exact_norm = normalize_distances(exact)
                error = np.abs(d_norm - exact_norm)
                mesh.add_scalar_quantity(
                    f"K2 Geodesic Error - {method}", error, enabled=False, cmap='reds'
                )

        # Update exact geodesic visualization
        if exact is not None:
            mesh.add_scalar_quantity(
                "K1 Geodesic (norm) - Exact", normalize_distances(exact),
                enabled=False, cmap='viridis'
            )

        # Update per-source result dicts so UI tables refresh automatically
        self.current_greens_results = self._all_greens_results.get(src_idx, {})
        self.current_heat_geodesic_results = self._all_geodesic_results.get(src_idx, {})

        # Source point marker
        if self.current_original_vertices is not None:
            pos = self.current_original_vertices[src_idx:src_idx + 1]
            ps.register_point_cloud("Source Vertex", pos, radius=0.025, color=(0.6, 0.0, 0.8))

    # =========================================================================
    # WEYL'S LAW EIGENVALUE SELF-CALIBRATION
    # =========================================================================

    def _compute_weyl_calibration(self):
        """Compute Weyl's law self-calibration factor for PRED eigenvalues.

        For a closed 2-manifold of area A, Weyl's law gives:
            λ_k ~ 4πk / A

        The calibration factor α = median(λ_pred_k / λ_weyl_k) for k=1..K.
        Calibrated eigenvalues = λ_pred / α.

        Area source is controlled by reconstruction_settings.weyl_area_source:
        - "gt": GT surface area from PyFM vertex areas (requires mesh)
        - "pred": predicted surface area from model's area head (self-supervised)
        """
        if self.current_inference_result is None or self.current_gt_data is None:
            return

        pred_evals = self.current_inference_result.get('predicted_eigenvalues')
        if pred_evals is None:
            return

        pred_evals = pred_evals.cpu().numpy() if torch.is_tensor(pred_evals) else np.asarray(pred_evals)

        area_source = self.reconstruction_settings.weyl_area_source

        if area_source == "pred":
            # Self-supervised: use predicted areas from model
            pred_areas = self.current_inference_result.get('areas')
            if pred_areas is not None:
                pred_areas_np = pred_areas.cpu().numpy() if torch.is_tensor(pred_areas) else np.asarray(pred_areas)
                total_area = float(np.sum(pred_areas_np))
            else:
                print("  [!] Weyl calibration: no predicted areas available")
                return
        else:
            # GT area from PyFM vertex areas
            gt_areas = self.current_gt_data.get('vertex_areas')
            if gt_areas is not None:
                total_area = float(np.sum(gt_areas))
            else:
                gt_M = self.current_gt_data.get('gt_mass_matrix')
                if gt_M is not None:
                    total_area = float(gt_M.diagonal().sum())
                else:
                    print("  [!] Weyl calibration: no GT area available")
                    return

        if total_area <= 0:
            print(f"  [!] Weyl calibration: invalid total area {total_area}")
            return

        # Compute Weyl expected eigenvalues: λ_k = 4πk / A (skip k=0)
        K = len(pred_evals)
        k_indices = np.arange(1, K)  # k = 1, 2, ..., K-1
        weyl_evals = 4.0 * np.pi * k_indices / total_area

        # Only use positive PRED eigenvalues for robust median
        pred_positive = pred_evals[1:]  # skip λ_0
        valid_mask = pred_positive > 1e-10
        if valid_mask.sum() < 2:
            print("  [!] Weyl calibration: not enough positive PRED eigenvalues")
            return

        ratios = pred_positive[valid_mask] / weyl_evals[valid_mask]
        alpha = float(np.median(ratios))

        if alpha <= 0:
            print(f"  [!] Weyl calibration: invalid factor {alpha}")
            return

        self._weyl_calibration_factor = alpha
        self._weyl_calibrated_evals = pred_evals / alpha

        src_label = "PRED areas" if area_source == "pred" else "GT areas"
        print(f"\n  [Weyl Calibration] Area source: {src_label}, total area: {total_area:.6f}")
        print(f"  [Weyl Calibration] Factor α = {alpha:.2f} (PRED evals are {alpha:.1f}× Weyl expectation)")
        print(f"  [Weyl Calibration] Calibrated eigenvalue range: [{self._weyl_calibrated_evals[0]:.4e}, {self._weyl_calibrated_evals[-1]:.4f}]")
        if self.current_gt_data.get('gt_eigenvalues') is not None:
            gt_evals = self.current_gt_data['gt_eigenvalues']
            print(f"  [Weyl Calibration] GT eigenvalue range:         [{gt_evals[0]:.4e}, {gt_evals[-1]:.4f}]")

    def _apply_weyl_calibration(self):
        """Recompute HKS/WKS descriptors and probe cosine similarity with calibrated eigenvalues."""
        if self._weyl_calibration_factor is None:
            self._compute_weyl_calibration()

        if self._weyl_calibration_factor is None:
            return  # calibration failed

        print("\n  Recomputing metrics with Weyl-calibrated PRED eigenvalues...")
        # Recompute descriptors (HKS/WKS use eigenvalues directly)
        self._compute_descriptor_comparison()
        print("  [OK] Descriptors recomputed with calibrated eigenvalues")

    def _revert_weyl_calibration(self):
        """Revert to original PRED eigenvalues and recompute metrics."""
        # Don't clear _weyl_calibration_factor — keep it cached for quick re-enable
        print("\n  Reverting to original PRED eigenvalues...")
        self._compute_descriptor_comparison()
        print("  [OK] Descriptors recomputed with original eigenvalues")

    # =========================================================================
    # NEW METRICS: Probe Function MSE, HKS/WKS Descriptors, Spectral Compression
    # =========================================================================

    def _compute_probe_function_mse(self):
        """Compute Lf probe function MSE for all methods (NeLo paper Table 1 metric).

        Uses 112 probe functions: 64 spectral (scaled eigenvectors) + 42 trig + 6 polynomial.

        Reports two variants:
        - Raw MSE (NeLo paper convention): mean ||M^-1 Lf - M_gt^-1 L_gt f||^2, clipped at 1.0
        - Normalized MSE (scale-invariant): per-probe relative error, normalized by GT response magnitude
          This handles the case where eigenvalues are scaled differently but eigenvectors are correct.
        """
        if self.current_gt_data is None:
            return

        vertices = self.current_gt_data['vertices'].astype(np.float64)
        n = len(vertices)
        gt_evals = self.current_gt_data.get('gt_eigenvalues')
        gt_evecs = self.current_gt_data.get('gt_eigenvectors')
        if gt_evals is None or gt_evecs is None:
            print("  [!] GT eigenpairs not available, skipping probe function MSE")
            return

        # --- Build probe functions ---
        probes = []

        # 1) Spectral probes: first 64 non-constant eigenvectors, scaled by 1/(λ + 0.1)
        n_spectral = min(64, len(gt_evals) - 1)
        for i in range(1, n_spectral + 1):  # skip constant eigenvector
            f = gt_evecs[:, i] / (gt_evals[i] + 0.1)
            probes.append(f)

        # 2) Trigonometric probes: sin(k*t + phi) / (2k)
        coords = [vertices[:, 0], vertices[:, 1], vertices[:, 2]]
        for k_val in [1, 2, 4, 8, 16, 32, 64]:
            for coord in coords:
                for phi in [0.0, np.pi / 2.0]:
                    f = np.sin(k_val * coord + phi) / (2.0 * k_val)
                    probes.append(f)

        # 3) Polynomial probes: x, y, z, x^2, y^2, z^2
        for d in range(3):
            probes.append(vertices[:, d])
        for d in range(3):
            probes.append(vertices[:, d] ** 2)

        probes = np.column_stack(probes)  # (N, n_probes)
        n_probes = probes.shape[1]

        # --- Compute GT reference: M_gt^-1 L_gt f ---
        L_gt, M_gt = self._get_method_L_M('GT')
        if L_gt is None or M_gt is None:
            return

        M_gt_diag = np.array(M_gt.diagonal()).flatten()
        M_gt_inv_diag = np.where(M_gt_diag > 1e-15, 1.0 / M_gt_diag, 0.0)

        gt_Lf = L_gt @ probes  # (N, n_probes)
        gt_result = M_gt_inv_diag[:, None] * gt_Lf  # M_gt^-1 L_gt f

        # --- Evaluate each method ---
        self._probe_function_results = {}
        for method_key in ['GT', 'PRED', 'Robust', 'NeLo']:
            L, M = self._get_method_L_M(method_key)
            if L is None or M is None:
                continue

            try:
                M_diag = np.array(M.diagonal()).flatten()
                M_inv_diag = np.where(M_diag > 1e-15, 1.0 / M_diag, 0.0)

                method_Lf = L @ probes
                method_result = M_inv_diag[:, None] * method_Lf

                diff = method_result - gt_result  # (N, n_probes)

                # --- Raw MSE (NeLo paper convention) ---
                per_probe_mse = np.mean(diff ** 2, axis=0)  # (n_probes,)
                clipped_mse = np.minimum(per_probe_mse, 1.0)
                raw_mse = float(clipped_mse.mean())
                failure_rate = float(np.mean(per_probe_mse > 1.0))

                # --- Cosine similarity (truly scale-invariant) ---
                # Measures whether Lf produces the same *shape* of response, ignoring global scale
                cosine_sims = []
                for p in range(n_probes):
                    gt_col = gt_result[:, p]
                    m_col = method_result[:, p]
                    gt_norm = np.linalg.norm(gt_col)
                    m_norm = np.linalg.norm(m_col)
                    if gt_norm > 1e-15 and m_norm > 1e-15:
                        cosine_sims.append(float(np.dot(gt_col, m_col) / (gt_norm * m_norm)))
                mean_cosine = float(np.mean(cosine_sims)) if cosine_sims else 0.0

                self._probe_function_results[method_key] = {
                    'mse': raw_mse,
                    'cosine_similarity': mean_cosine,
                    'failure_rate': failure_rate,
                    'n_probes': n_probes,
                }
                print(f"  {method_key}: Lf MSE={raw_mse:.6f} (raw), cos_sim={mean_cosine:.6f}, "
                      f"R_MSE>1={failure_rate:.1%} ({n_probes} probes)")

            except Exception as e:
                print(f"  [!] Probe function MSE for {method_key} failed: {e}")

    def _compute_descriptor_comparison(self):
        """Compute HKS and WKS descriptor error vs GT for all methods."""
        if self.current_gt_data is None:
            return

        gt_evals = self.current_gt_data.get('gt_eigenvalues')
        gt_evecs = self.current_gt_data.get('gt_eigenvectors')
        if gt_evals is None or gt_evecs is None:
            return

        def compute_hks(eigenvalues, eigenvectors, t_values):
            """Compute HKS descriptor (N, n_scales) at given time scales."""
            evals = eigenvalues[1:]
            evecs = eigenvectors[:, 1:]

            # Filter out negative/zero eigenvalues
            pos_mask = evals > 1e-10
            evals = evals[pos_mask]
            evecs = evecs[:, pos_mask]

            k = min(len(evals), 50)
            if k < 2:
                return None
            evals = evals[:k]
            evecs = evecs[:, :k]

            # HKS(x, t) = Σ exp(-λ_i * t) * φ_i(x)^2
            phi_sq = evecs ** 2  # (N, k)
            exp_terms = np.exp(-evals[None, :] * t_values[:, None])  # (n_scales, k)
            hks = phi_sq @ exp_terms.T  # (N, n_scales)
            return hks

        def compute_wks(eigenvalues, eigenvectors, e_values, sigma):
            """Compute WKS descriptor (N, n_scales) at given energy scales."""
            evals = eigenvalues[1:]
            evecs = eigenvectors[:, 1:]

            # Filter out negative/zero eigenvalues
            pos_mask = evals > 1e-10
            evals = evals[pos_mask]
            evecs = evecs[:, pos_mask]

            k = min(len(evals), 50)
            if k < 2:
                return None
            evals = evals[:k]
            evecs = evecs[:, :k]

            log_evals = np.log(evals)

            # WKS(x, e) = Σ exp(-(e - ln(λ_i))^2 / 2σ^2) * φ_i(x)^2
            phi_sq = evecs ** 2  # (N, k)
            weights = np.exp(-((e_values[:, None] - log_evals[None, :]) ** 2) / (2 * sigma ** 2))  # (n_scales, k)
            wks = phi_sq @ weights.T  # (N, n_scales)
            return wks

        def compare_descriptors(desc_method, desc_gt):
            """Compute normalized L2 error and correlation between descriptor fields."""
            if desc_method is None or desc_gt is None:
                return None, None
            n_scales = desc_gt.shape[1]
            l2_errors = []
            correlations = []
            for s in range(n_scales):
                gt_col = desc_gt[:, s]
                m_col = desc_method[:, s]
                gt_norm = np.linalg.norm(gt_col)
                if gt_norm < 1e-15:
                    continue
                l2_errors.append(np.linalg.norm(m_col - gt_col) / gt_norm)
                if np.std(gt_col) > 1e-15 and np.std(m_col) > 1e-15:
                    correlations.append(float(np.corrcoef(m_col, gt_col)[0, 1]))
            mean_l2 = float(np.mean(l2_errors)) if l2_errors else 1.0
            mean_corr = float(np.mean(correlations)) if correlations else 0.0
            return mean_l2, mean_corr

        # Compute GT time/energy scales (used for ALL methods for fair comparison)
        gt_evals_pos = gt_evals[1:]
        gt_evals_pos = gt_evals_pos[gt_evals_pos > 1e-10]
        if len(gt_evals_pos) < 2:
            print("  [!] Not enough positive GT eigenvalues for HKS/WKS")
            return

        n_scales = 100
        # HKS time scales from GT eigenvalue range
        t_min = 4.0 * np.log(10.0) / gt_evals_pos[min(len(gt_evals_pos)-1, 49)]
        t_max = 4.0 * np.log(10.0) / gt_evals_pos[0]
        hks_t_values = np.exp(np.linspace(np.log(t_min), np.log(t_max), n_scales))

        # WKS energy scales from GT eigenvalue range
        log_gt_evals = np.log(gt_evals_pos[:min(len(gt_evals_pos), 50)])
        wks_e_min = log_gt_evals[0]
        wks_e_max = log_gt_evals[-1]
        wks_sigma = 7.0 * (wks_e_max - wks_e_min) / n_scales
        wks_e_values = np.linspace(wks_e_min, wks_e_max, n_scales)

        if wks_sigma < 1e-10:
            print("  [!] GT eigenvalue range too narrow for WKS")
            wks_e_values = None

        # GT descriptors (reference)
        hks_gt = compute_hks(gt_evals, gt_evecs, hks_t_values)
        wks_gt = compute_wks(gt_evals, gt_evecs, wks_e_values, wks_sigma) if wks_e_values is not None else None

        if hks_gt is None and wks_gt is None:
            print("  [!] Could not compute GT HKS/WKS (bad eigenvalues)")
            return

        self._descriptor_results = {}

        # Collect eigenpairs per method
        method_eigenpairs = {}
        pred_evals = self.current_inference_result.get('predicted_eigenvalues') if self.current_inference_result else None
        pred_evecs = self.current_inference_result.get('predicted_eigenvectors') if self.current_inference_result else None
        if pred_evals is not None and pred_evecs is not None:
            # Use Weyl-calibrated eigenvalues if enabled (eigenvectors unchanged)
            if self._weyl_calibrated_evals is not None and self.reconstruction_settings.enable_weyl_calibration:
                method_eigenpairs['PRED'] = (self._weyl_calibrated_evals, pred_evecs)
            else:
                method_eigenpairs['PRED'] = (pred_evals, pred_evecs)

        robust_evals = self.current_gt_data.get('robust_eigenvalues')
        robust_evecs = self.current_gt_data.get('robust_eigenvectors')
        if robust_evals is not None and robust_evecs is not None:
            method_eigenpairs['Robust'] = (robust_evals, robust_evecs)

        if self.current_nelo_eigenvalues is not None and self.current_nelo_eigenvectors is not None:
            method_eigenpairs['NeLo'] = (self.current_nelo_eigenvalues, self.current_nelo_eigenvectors)

        for method_key, (evals, evecs) in method_eigenpairs.items():
            try:
                evals_np = evals.cpu().numpy() if torch.is_tensor(evals) else np.asarray(evals)
                evecs_np = evecs.cpu().numpy() if torch.is_tensor(evecs) else np.asarray(evecs)

                # Use GT's time/energy scales for all methods
                hks_m = compute_hks(evals_np, evecs_np, hks_t_values)
                wks_m = compute_wks(evals_np, evecs_np, wks_e_values, wks_sigma) if wks_e_values is not None else None

                hks_l2, hks_corr = compare_descriptors(hks_m, hks_gt)
                wks_l2, wks_corr = compare_descriptors(wks_m, wks_gt)

                self._descriptor_results[method_key] = {
                    'hks_l2_error': hks_l2 if hks_l2 is not None else -1.0,
                    'hks_correlation': hks_corr if hks_corr is not None else 0.0,
                    'wks_l2_error': wks_l2 if wks_l2 is not None else -1.0,
                    'wks_correlation': wks_corr if wks_corr is not None else 0.0,
                }
                res = self._descriptor_results[method_key]
                print(f"  {method_key}: HKS L2={res['hks_l2_error']:.4f} corr={res['hks_correlation']:.4f}, "
                      f"WKS L2={res['wks_l2_error']:.4f} corr={res['wks_correlation']:.4f}")

            except Exception as e:
                print(f"  [!] Descriptor comparison for {method_key} failed: {e}")

    def _compute_spectral_compression_error(self):
        """Compute spectral mesh compression error at key eigenvector counts.

        For each method, project original vertices onto k eigenvectors and
        measure per-vertex L2 reconstruction error.
        """
        if self.current_gt_data is None:
            return

        vertices = self.current_gt_data['vertices'].astype(np.float64)
        gt_evecs = self.current_gt_data.get('gt_eigenvectors')
        if gt_evecs is None:
            return

        key_counts = [5, 10, 20, 50]

        def compute_reconstruction_error(eigenvectors, mass_matrix, verts, k_values):
            """Compute vertex L2 error for spectral reconstruction at each k."""
            evecs = eigenvectors.cpu().numpy() if torch.is_tensor(eigenvectors) else np.asarray(eigenvectors)
            if mass_matrix is not None:
                M_diag = np.array(mass_matrix.diagonal()).flatten() if scipy.sparse.issparse(mass_matrix) else np.diag(mass_matrix)
            else:
                M_diag = np.ones(len(verts))

            max_k = min(evecs.shape[1], max(k_values))
            # Coefficients: φᵀ M x for each coordinate
            M_verts = M_diag[:, None] * verts  # (N, 3)
            coeffs = evecs[:, :max_k].T @ M_verts  # (max_k, 3)

            results = {}
            for k in k_values:
                if k > evecs.shape[1]:
                    continue
                recon = evecs[:, :k] @ coeffs[:k]  # (N, 3)
                per_vertex_error = np.linalg.norm(verts - recon, axis=1)
                results[k] = {
                    'mean_l2': float(per_vertex_error.mean()),
                    'max_l2': float(per_vertex_error.max()),
                }
            return results

        self._compression_results = {}

        # GT
        L_gt, M_gt = self._get_method_L_M('GT')
        if gt_evecs is not None and M_gt is not None:
            try:
                self._compression_results['GT'] = compute_reconstruction_error(
                    gt_evecs, M_gt, vertices, key_counts)
            except Exception as e:
                print(f"  [!] GT compression error failed: {e}")

        # PRED
        pred_evecs = self.current_inference_result.get('predicted_eigenvectors') if self.current_inference_result else None
        pred_M = self.current_inference_result.get('mass_matrix') if self.current_inference_result else None
        if pred_evecs is not None:
            try:
                self._compression_results['PRED'] = compute_reconstruction_error(
                    pred_evecs, pred_M, vertices, key_counts)
            except Exception as e:
                print(f"  [!] PRED compression error failed: {e}")

        # Robust
        robust_evecs = self.current_gt_data.get('robust_eigenvectors')
        _, M_robust = self._get_method_L_M('Robust') if not self._skip_robust else (None, None)
        if robust_evecs is not None and M_robust is not None:
            try:
                self._compression_results['Robust'] = compute_reconstruction_error(
                    robust_evecs, M_robust, vertices, key_counts)
            except Exception as e:
                print(f"  [!] Robust compression error failed: {e}")

        # NeLo
        if self.current_nelo_eigenvectors is not None and self.current_nelo_M is not None:
            try:
                self._compression_results['NeLo'] = compute_reconstruction_error(
                    self.current_nelo_eigenvectors, self.current_nelo_M, vertices, key_counts)
            except Exception as e:
                print(f"  [!] NeLo compression error failed: {e}")

        # Print summary
        if self._compression_results:
            print(f"\n  {'Method':<12}", end="")
            for k in key_counts:
                print(f"  {'k=' + str(k):>8}", end="")
            print()
            for method, k_results in self._compression_results.items():
                print(f"  {method:<12}", end="")
                for k in key_counts:
                    if k in k_results:
                        print(f"  {k_results[k]['mean_l2']:>8.5f}", end="")
                    else:
                        print(f"  {'N/A':>8}", end="")
                print()

    def _print_multisource_summary(self):
        """Print aggregated metrics across all source vertices."""
        n_sources = len(self._source_indices) if self._source_indices else 0
        if n_sources == 0:
            return

        print(f"\n{'=' * 80}")
        print(f"MULTI-SOURCE VALIDATION SUMMARY ({n_sources} FPS-sampled sources)")
        print(f"{'=' * 80}")

        # Geodesic table
        if self._aggregated_geodesic_metrics:
            print(f"\nGEODESIC QUALITY (averaged over {n_sources} sources, normalized [0,1])")
            print(f"{'-' * 80}")
            print(f"{'Method':<16} {'Corr':>14} {'MAE':>14} {'MaxErr':>14} {'Mono':>14}")
            print(f"{'-' * 80}")
            for method, agg in self._aggregated_geodesic_metrics.items():
                print(f"{method:<16} {agg.corr_mean:>6.4f}±{agg.corr_std:<5.4f}"
                      f" {agg.mae_mean:>6.4f}±{agg.mae_std:<5.4f}"
                      f" {agg.max_err_mean:>6.4f}±{agg.max_err_std:<5.4f}"
                      f" {agg.mono_mean:>6.4f}±{agg.mono_std:<5.4f}")

        # Green's function table
        if self._aggregated_greens_metrics:
            print(f"\nGREEN'S FUNCTION QUALITY (averaged over {n_sources} sources)")
            print(f"{'-' * 80}")
            print(f"{'Method':<16} {'GT Corr':>14} {'Mono':>14} {'MaxPrinc Pass':>14}")
            print(f"{'-' * 80}")
            for method, agg in self._aggregated_greens_metrics.items():
                print(f"{method:<16} {agg.greens_corr_with_gt_mean:>6.4f}±{agg.greens_corr_with_gt_std:<5.4f}"
                      f" {agg.greens_mono_mean:>6.4f}±{agg.greens_mono_std:<5.4f}"
                      f" {agg.greens_max_principle_pass_rate:>12.0%}")

        print(f"{'=' * 80}\n")

    def _print_timing_summary(self):
        """Print timing comparison tables: assembly, geodesic E2E, Green's E2E."""
        t = self.timing_results

        # =====================================================================
        # TABLE 1: Laplacian Assembly (L, M)
        # =====================================================================
        print(f"\n{'=' * 70}")
        print("LAPLACIAN MATRIX ASSEMBLY TIMING COMPARISON")
        print(f"{'=' * 70}")
        print(f"Mesh: {t.num_vertices} vertices, {t.num_faces} faces, k={t.current_k}")
        print(f"{'-' * 70}")
        print(f"{'Method':<25} {'Time (ms)':<15} {'Notes'}")
        print(f"{'-' * 70}")

        # PRED breakdown
        print(f"{'PRED (Neural Network)':<25}")
        print(f"  {'k-NN extraction':<23} {t.pred_patch_extraction_time * 1000:>10.2f} ms")
        print(f"  {'Model inference':<23} {t.pred_model_inference_time * 1000:>10.2f} ms")
        print(f"  {'Matrix assembly':<23} {t.pred_matrix_assembly_time * 1000:>10.2f} ms")
        print(f"  {'TOTAL':<23} {t.pred_total_time * 1000:>10.2f} ms")

        print(f"{'-' * 70}")

        # NeLo breakdown
        if t.nelo_total_time > 0:
            print(f"{'NeLo (GNN)':<25}  (k={t.nelo_k})")
            print(f"  {'Graph tree (KNN)':<23} {t.nelo_graph_tree_time * 1000:>10.2f} ms")
            print(f"  {'Model inference':<23} {t.nelo_model_inference_time * 1000:>10.2f} ms")
            print(f"  {'Matrix assembly':<23} {t.nelo_matrix_assembly_time * 1000:>10.2f} ms")
            print(f"  {'TOTAL':<23} {t.nelo_total_time * 1000:>10.2f} ms")
        else:
            print(f"{'NeLo (GNN)':<25} N/A (not loaded)")

        print(f"{'-' * 70}")

        # Robust
        print(f"{'Robust (Point Cloud)':<25} {t.robust_matrix_assembly_time * 1000:>10.2f} ms   k-NN + weights + assembly")

        print(f"{'-' * 70}")

        # GT
        print(f"{'GT (Mesh Cotangent)':<25} {t.gt_matrix_assembly_time * 1000:>10.2f} ms   igl.cotmatrix + massmatrix")

        print(f"{'-' * 70}")

        # Speedup comparison
        if t.gt_matrix_assembly_time > 0:
            pred_vs_gt = t.pred_total_time / t.gt_matrix_assembly_time
            robust_vs_gt = t.robust_matrix_assembly_time / t.gt_matrix_assembly_time if t.robust_matrix_assembly_time > 0 else 0
            print(f"Relative to GT:  PRED = {pred_vs_gt:.2f}x,  Robust = {robust_vs_gt:.2f}x", end="")
            if t.nelo_total_time > 0:
                nelo_vs_gt = t.nelo_total_time / t.gt_matrix_assembly_time
                print(f",  NeLo = {nelo_vs_gt:.2f}x", end="")
            print()

        if t.robust_matrix_assembly_time > 0 and t.pred_total_time > 0:
            pred_vs_robust = t.pred_total_time / t.robust_matrix_assembly_time
            print(f"PRED vs Robust:  {pred_vs_robust:.2f}x")
        if t.nelo_total_time > 0 and t.robust_matrix_assembly_time > 0:
            nelo_vs_robust = t.nelo_total_time / t.robust_matrix_assembly_time
            print(f"NeLo vs Robust:  {nelo_vs_robust:.2f}x")
        if t.nelo_total_time > 0 and t.pred_total_time > 0:
            pred_vs_nelo = t.pred_total_time / t.nelo_total_time
            print(f"PRED vs NeLo:    {pred_vs_nelo:.2f}x")

        # =====================================================================
        # TABLE 2: Geodesic E2E (L,M assembly + grad operator + solve)
        # =====================================================================
        has_geodesic = (t.gt_geodesic_solve_time > 0 or t.pred_geodesic_solve_time > 0)
        if has_geodesic:
            n_src = len(self._source_indices) if self._source_indices else 1
            src_note = f" (avg over {n_src} sources)" if n_src > 1 else ""
            print(f"\n{'-' * 70}")
            print(f"GEODESIC E2E: L,M Assembly + Grad Operator + Solve{src_note}")
            print(f"{'-' * 70}")
            print(f"{'Method':<14} {'L,M (ms)':>10} {'Grad (ms)':>10} {'Solve (ms)':>10} {'TOTAL (ms)':>10}")
            print(f"{'-' * 70}")

            for label, lm_time, grad_time, solve_time in [
                ('GT',     t.gt_matrix_assembly_time,     t.gt_grad_op_time,   t.gt_geodesic_solve_time),
                ('PRED',   t.pred_total_time,             t.pred_grad_op_time, t.pred_geodesic_solve_time),
                ('NeLo',   t.nelo_total_time,             t.nelo_grad_op_time, t.nelo_geodesic_solve_time),
            ]:
                if solve_time > 0:
                    solve_avg = solve_time / n_src
                    total = lm_time + grad_time + solve_avg
                    print(f"{label:<14} {lm_time * 1000:>10.1f} {grad_time * 1000:>10.1f} {solve_avg * 1000:>10.1f} {total * 1000:>10.1f}")

            # Robust: pp3d is self-contained
            if t.robust_geodesic_solve_time > 0:
                solve_avg = t.robust_geodesic_solve_time / n_src
                if self._robust_geodesic_heat:
                    total = t.robust_matrix_assembly_time + solve_avg
                    print(f"{'Robust':<14} {t.robust_matrix_assembly_time * 1000:>10.1f} {'(incl.)':>10} {solve_avg * 1000:>10.1f} {total * 1000:>10.1f}")
                else:
                    print(f"{'Robust (pp3d)':<14} {'—':>10} {'—':>10} {solve_avg * 1000:>10.1f} {solve_avg * 1000:>10.1f}")

        # =====================================================================
        # TABLE 3: Green's Function E2E (L,M assembly + solve)
        # =====================================================================
        has_greens = (t.gt_greens_solve_time > 0 or t.pred_greens_solve_time > 0)
        if has_greens:
            n_src = len(self._source_indices) if self._source_indices else 1
            src_note = f" (avg over {n_src} sources)" if n_src > 1 else ""
            print(f"\n{'-' * 70}")
            print(f"GREEN'S FUNCTION E2E: L,M Assembly + Solve{src_note}")
            print(f"{'-' * 70}")
            print(f"{'Method':<14} {'L,M (ms)':>10} {'Solve (ms)':>10} {'TOTAL (ms)':>10}")
            print(f"{'-' * 70}")

            for label, lm_time, solve_time in [
                ('GT',     t.gt_matrix_assembly_time,     t.gt_greens_solve_time),
                ('PRED',   t.pred_total_time,             t.pred_greens_solve_time),
                ('NeLo',   t.nelo_total_time,             t.nelo_greens_solve_time),
                ('Robust', t.robust_matrix_assembly_time, t.robust_greens_solve_time),
            ]:
                if solve_time > 0:
                    solve_avg = solve_time / n_src
                    total = lm_time + solve_avg
                    print(f"{label:<14} {lm_time * 1000:>10.1f} {solve_avg * 1000:>10.1f} {total * 1000:>10.1f}")

        print(f"{'=' * 70}\n")

    def _update_pred_with_new_k(self, new_k: int):
        """
        Update PRED Laplacian with new k (re-extract patches + re-run inference).

        Args:
            new_k: New number of neighbors for PRED patches
        """
        print(f"\n{'=' * 60}")
        print(f"UPDATING PRED WITH NEW k={new_k}")
        print('=' * 60)

        if self.current_model is None or self.current_device is None:
            print("[!] Model or device not available for re-inference")
            return

        if self.current_original_vertices is None:
            print("[!] No mesh vertices available")
            return

        # STEP 1: Re-extract patches with new k
        print(f"STEP 1: Re-extracting patches with k={new_k}...")

        # === TIME: PRED patch extraction (k-NN search + data preparation) ===
        if self.current_device is not None and self.current_device.type == 'cuda':
            torch.cuda.synchronize()
        t_extraction_start = time.perf_counter()

        new_patch_data = build_patches_from_vertices(
            self._current_vertices_tensor, new_k, device=self.current_device
        )

        if self.current_device is not None and self.current_device.type == 'cuda':
            torch.cuda.synchronize()
        t_extraction_end = time.perf_counter()
        pred_extraction_time = t_extraction_end - t_extraction_start
        self.timing_results.pred_patch_extraction_time = pred_extraction_time

        print(f"  Extracted {len(new_patch_data.center_indices)} patches with {new_k} neighbors each")
        print(f"  [TIMING] PRED patch extraction: {pred_extraction_time * 1000:.2f} ms")

        # Update stored k-NN connectivity to match new k
        # (critical: downstream code like gradient operator assembly uses these)
        # Move to device since forward_result tensors (grad_coeffs etc.) live on GPU
        self.current_vertex_indices = new_patch_data.vertex_indices.to(self.current_device)
        self.current_center_indices = new_patch_data.center_indices.to(self.current_device)
        self.current_batch_indices = new_patch_data.patch_idx.to(self.current_device)

        # STEP 2: Re-run model inference (timing is inside perform_model_inference)
        print(f"STEP 2: Re-running model inference...")
        new_inference_result = self.perform_model_inference(
            self.current_model, new_patch_data, self.current_device
        )

        if new_inference_result['predicted_eigenvalues'] is None:
            print("[!] Failed to compute PRED eigendecomposition")
            return

        self.current_inference_result = new_inference_result
        print(f"  PRED eigenvalue range: [{new_inference_result['predicted_eigenvalues'][0]:.2e}, {new_inference_result['predicted_eigenvalues'][-1]:.6f}]")

        # Update stored raw weights and areas to stay in sync
        self.current_stiffness_weights = new_inference_result['stiffness_weights']
        self.current_areas = new_inference_result['areas']

        # Weight distribution diagnostic for new k
        mask = new_inference_result.get('attention_mask')
        if mask is None:
            mask = torch.ones_like(self.current_stiffness_weights, dtype=torch.bool)
        self._print_weight_distribution_diagnostic(self.current_stiffness_weights, mask)

        # Reassemble learned gradient operator if in gradient mode
        operator_mode = getattr(self.current_model, '_operator_mode', 'stiffness')
        new_forward_result = new_inference_result.get('forward_result')
        if operator_mode == "gradient" and new_forward_result is not None and new_forward_result.get('grad_coeffs') is not None:
            try:
                print("  Reassembling learned gradient operator G for new k...")
                _t_grad_op = time.perf_counter()
                batch_indices_for_g = self.current_batch_indices.clone()
                self.current_learned_gradient_op = assemble_gradient_operator(
                    grad_coeffs=new_forward_result['grad_coeffs'],
                    attention_mask=new_forward_result['attention_mask'],
                    vertex_indices=self.current_vertex_indices,
                    center_indices=self.current_center_indices,
                    batch_indices=batch_indices_for_g
                )
                self.timing_results.pred_grad_op_time = time.perf_counter() - _t_grad_op
                print(f"  Updated learned G: {self.current_learned_gradient_op.shape} ({self.current_learned_gradient_op.nnz} non-zeros)")
                print(f"  [TIMING] Gradient operator assembly: {self.timing_results.pred_grad_op_time * 1000:.2f} ms")
            except Exception as e:
                print(f"  [!] Failed to reassemble learned gradient operator: {e}")
                self.current_learned_gradient_op = None

        # Compute PRED total time
        pred_total_time = (self.timing_results.pred_patch_extraction_time +
                           self.timing_results.pred_model_inference_time +
                           self.timing_results.pred_matrix_assembly_time)
        self.timing_results.pred_total_time = pred_total_time

        # STEP 3: Recompute predicted quantities
        print(f"STEP 3: Recomputing predicted quantities...")
        new_predicted_data = self.compute_predicted_quantities_from_laplacian(
            new_inference_result['stiffness_matrix'],
            self.current_gt_data['vertices'],
            mass_matrix=new_inference_result.get('mass_matrix')
        )
        self.current_predicted_data = new_predicted_data

        # STEP 4: Update visualizations
        print(f"STEP 4: Updating visualizations...")
        self._remove_existing_reconstructions()
        self._update_mesh_reconstructions(self.current_gt_data, self.current_inference_result)

        # Update eigenvector visualizations on the mesh
        self._update_eigenvector_visualizations()

        # Refresh PRED curvature/normal visualizations (Polyscope overwrites same-named quantities)
        mesh_structure = self.current_mesh_structure
        if mesh_structure is not None:
            self.add_comprehensive_curvature_visualizations(
                mesh_structure, self.current_gt_data,
                self.current_predicted_data, self.current_inference_result
            )

        # Refresh weight neighborhood visualizations (same FPS samples, new k/weights)
        self._add_weight_neighborhood_visualization()

        # STEP 5: Recompute all downstream validations (Green's fn, geodesics, sparsity)
        print(f"STEP 5: Recomputing downstream validations for PRED...")
        self._recompute_validations_after_k_change("PRED")

        # Print timing summary
        self._print_timing_summary()

        print(f"\n[OK] Updated PRED with k={new_k}")
        print('=' * 60)

    def _update_robust_with_new_k(self, new_k: int):
        """
        Update Robust-laplacian with new k (independent of PRED k).

        Args:
            new_k: New number of neighbors for robust point_cloud_laplacian
        """
        print(f"\n{'=' * 60}")
        print(f"UPDATING ROBUST WITH NEW k={new_k}")
        print('=' * 60)

        if self.current_original_vertices is None:
            print("[!] No mesh vertices available")
            return

        # Recompute robust-laplacian with new k
        print(f"Recomputing robust-laplacian with k={new_k}...")
        robust_eigenvalues, robust_eigenvectors, robust_vertex_areas = self._compute_robust_laplacian_with_k(
            self.current_original_vertices, new_k
        )

        # Update gt_data with new robust-laplacian results
        self.current_gt_data['robust_eigenvalues'] = robust_eigenvalues
        self.current_gt_data['robust_eigenvectors'] = robust_eigenvectors
        self.current_gt_data['robust_vertex_areas'] = robust_vertex_areas

        # Update visualizations
        print(f"Updating visualizations...")
        self._remove_existing_reconstructions()
        self._update_mesh_reconstructions(self.current_gt_data, self.current_inference_result)

        # Update eigenvector visualizations on the mesh
        self._update_eigenvector_visualizations()

        # Recompute downstream validations for Robust only
        print(f"Recomputing downstream validations for Robust...")
        self._recompute_validations_after_k_change("Robust")

        print(f"\n[OK] Updated Robust with k={new_k}")
        print('=' * 60)

    def _update_eigenvector_visualizations(self):
        """Update eigenvector scalar fields on the mesh structure."""
        if self.current_mesh_structure is None:
            return

        # Remove old eigenvector quantities and add new ones
        # Note: Polyscope doesn't have a clean way to remove quantities,
        # so we just add new ones with updated values

        gt_eigenvalues = self.current_gt_data.get('gt_eigenvalues')
        gt_eigenvectors = self.current_gt_data.get('gt_eigenvectors')
        pred_eigenvalues = self.current_inference_result.get('predicted_eigenvalues')
        pred_eigenvectors = self.current_inference_result.get('predicted_eigenvectors')
        robust_eigenvalues = self.current_gt_data.get('robust_eigenvalues')
        robust_eigenvectors = self.current_gt_data.get('robust_eigenvectors')

        # Recompute cosine similarities
        cosine_similarities_pred = self._compute_eigenvector_cosine_similarities(
            gt_eigenvectors, pred_eigenvectors
        )
        cosine_similarities_robust = self._compute_eigenvector_cosine_similarities(
            gt_eigenvectors, robust_eigenvectors
        )
        cosine_similarities_nelo = self._compute_eigenvector_cosine_similarities(
            gt_eigenvectors, self.current_nelo_eigenvectors
        )

        # Update stored average for UI
        if cosine_similarities_pred is not None:
            num_to_show = min(self.config.num_eigenvectors_to_show, len(cosine_similarities_pred))
            self.current_avg_cosine_similarity = float(cosine_similarities_pred[:num_to_show].mean())

        # Update per-eigenvector cosine similarities for UI comparison table
        self.current_cosine_similarities_pred = cosine_similarities_pred
        self.current_cosine_similarities_robust = cosine_similarities_robust
        self.current_cosine_similarities_nelo = cosine_similarities_nelo

        # Print updated comparison
        print(f"\nUpdated eigenvector cosine similarities:")
        if cosine_similarities_pred is not None:
            print(f"  GT vs PRED mean: {cosine_similarities_pred.mean():.4f}")
        if cosine_similarities_robust is not None:
            print(f"  GT vs Robust mean: {cosine_similarities_robust.mean():.4f}")
        if cosine_similarities_nelo is not None:
            print(f"  GT vs NeLo mean: {cosine_similarities_nelo.mean():.4f}")

    def _recompute_and_update_reconstructions(self):
        """Re-compute and update mesh reconstructions with new settings."""
        if not self._has_current_batch_data():
            print("[!] No current batch data available for re-computation")
            return

        print("[*] Re-computing mesh reconstructions with new settings...")

        # Remove existing reconstruction structures
        self._remove_existing_reconstructions()

        # Re-compute and visualize with new settings
        self._update_mesh_reconstructions(
            self.current_gt_data, self.current_inference_result
        )

        print("[OK] Mesh reconstructions updated with new settings")

    def _remove_existing_reconstructions(self):
        """Remove existing reconstruction structures from polyscope."""
        try:
            # Remove tracked reconstruction structures
            for struct_name in self.reconstruction_structure_names:
                try:
                    ps.remove_surface_mesh(struct_name)
                    print(f"  Removed reconstruction structure: {struct_name}")
                except Exception as e:
                    print(f"  Warning: Could not remove structure {struct_name}: {e}")

            # Clear the tracking list
            self.reconstruction_structure_names.clear()
            print(f"  Cleared reconstruction structure tracking list")

        except Exception as e:
            print(f"  Warning: Failed to remove some reconstruction structures: {e}")

    def load_trained_model(self, ckpt_path: Path, device: torch.device, cfg: DictConfig) -> LaplacianTransformerModule:
        """
        Load trained LaplacianTransformerModule from checkpoint.

        Args:
            ckpt_path: Path to the checkpoint file
            device: Device to load the model on
            cfg: Hydra config containing model configuration

        Returns:
            Loaded model in evaluation mode
        """
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        try:
            print(f"Loading model checkpoint from: {ckpt_path}")

            # Extract model arguments from config
            model_cfg = cfg.model.module

            # Load model from checkpoint with inference-time settings:
            # - normalize_patch_features=True: normalize inputs to unit sphere (same as training)
            # - scale_areas_by_patch_size=False: don't scale areas back (Laplacian assembly uses normalized space)
            model = LaplacianTransformerModule.load_from_checkpoint(
                str(ckpt_path),
                map_location=device,
                normalize_patch_features=True,
                scale_areas_by_patch_size=True,
            )

            model.eval()
            model.to(device)

            # Disable gradient checkpointing for inference (if present)
            # This ensures we don't have unnecessary overhead from checkpointing logic
            self._disable_gradient_checkpointing(model)

            print(f"[OK] Model loaded successfully on {device}")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Input dim: {model._input_dim}")
            print(f"   Model dim: {model._d_model}")
            print(f"   Num eigenvalues: {model._num_eigenvalues}")

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {ckpt_path}: {e}")

    def _disable_gradient_checkpointing(self, model: torch.nn.Module):
        """
        Disable gradient checkpointing on the model for faster inference.

        Gradient checkpointing saves memory during training by recomputing activations,
        but adds overhead during inference when we don't need gradients.

        Args:
            model: The model to disable gradient checkpointing on
        """
        disabled_count = 0

        # Check for HuggingFace-style gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
            disabled_count += 1
            print("  Disabled HuggingFace-style gradient checkpointing")

        # Check for gradient_checkpointing attribute
        if hasattr(model, 'gradient_checkpointing'):
            model.gradient_checkpointing = False
            disabled_count += 1
            print("  Disabled gradient_checkpointing attribute")

        # Recursively check submodules for TransformerEncoder with checkpoint settings
        for name, module in model.named_modules():
            # PyTorch TransformerEncoder may have enable_nested_tensor which affects performance
            if hasattr(module, 'enable_nested_tensor'):
                # enable_nested_tensor=True is faster for inference with padding
                if not module.enable_nested_tensor:
                    print(f"  Note: {name} has enable_nested_tensor=False")

            # Check for any checkpoint-related attributes
            if hasattr(module, 'checkpoint'):
                module.checkpoint = False
                disabled_count += 1
            if hasattr(module, 'use_checkpoint'):
                module.use_checkpoint = False
                disabled_count += 1

        if disabled_count > 0:
            print(f"  Disabled {disabled_count} gradient checkpointing setting(s)")
        else:
            print("  No gradient checkpointing found (good for inference)")

    def _warmup_cuda(self, model: LaplacianTransformerModule, device: torch.device, k: int):
        """CUDA warmup — delegates to shared cuda_warmup() in utils."""
        cuda_warmup(model, device, k)

    def compute_eigendecomposition(self, stiffness_matrix: scipy.sparse.csr_matrix,
                                   k: int = 50,
                                   mass_matrix: scipy.sparse.csr_matrix = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of the Laplacian using generalized eigenvalue problem.

        Solves S @ v = lambda * M @ v where S is stiffness and M is mass matrix.

        Args:
            stiffness_matrix: Sparse symmetric stiffness matrix
            k: Number of eigenvalues to compute
            mass_matrix: Sparse diagonal mass matrix (optional, uses identity if None)

        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted in ascending order by eigenvalue.
            - eigenvalues: Array of shape (k,) with smallest k eigenvalues
            - eigenvectors: Array of shape (n, k) with corresponding eigenvectors
        """
        try:
            # Use the centralized eigendecomposition function from utils


            eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(
                stiffness_matrix, k, mass_matrix=mass_matrix
            )

            return eigenvalues, eigenvectors

        except Exception as e:
            print(f"Error computing eigendecomposition: {e}")
            return None, None

    def load_original_mesh_for_gt(self, mesh_file_path: str) -> Dict[str, Any]:
        """
        Load original mesh and compute ground-truth data using the same methods as ValidationMeshUploader.

        Args:
            mesh_file_path: Path to the mesh file

        Returns:
            Dictionary containing all GT quantities
        """
        print(f"Loading mesh for GT computation: {Path(mesh_file_path).name}")

        try:
            # Load mesh using trimesh
            mesh = trimesh.load(str(mesh_file_path))
            raw_vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)

            # Apply the same normalization as MeshDataset
            vertices = normalize_mesh_vertices(raw_vertices)

            # Update mesh with normalized vertices for normal computation
            mesh.vertices = vertices
            gt_vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)

            print(f"Mesh has {len(vertices)} vertices and {len(faces)} faces")
            print(f"Normalized vertices: center at origin, max distance = {np.linalg.norm(vertices, axis=1).max():.6f}")

        except Exception as e:
            raise RuntimeError(f"Failed to load mesh {mesh_file_path}: {e}")

        # Compute GT mean curvature using libigl if available
        gt_mean_curvature = None
        gt_mean_curvature_vector = None
        gt_gaussian_curvature = None
        gt_mesh_grad_op = None
        gt_face_areas = None
        if HAS_IGL:
            vertices_igl = vertices.astype(np.float64)
            faces_igl = faces.astype(np.int32)

            # Curvature computation — needed for metrics export in all modes
            try:
                print("Computing GT mean curvature using libigl...")

                # Compute principal curvatures using libigl
                _, _, principal_curvature1, principal_curvature2, _ = igl.principal_curvature(
                    vertices_igl, faces_igl
                )

                # Mean curvature is the average of principal curvatures: H = (k1 + k2) / 2
                gt_mean_curvature = (principal_curvature1 + principal_curvature2) / 2.0
                gt_mean_curvature = gt_mean_curvature.astype(np.float32)

                # Gaussian curvature: K = k1 * k2
                gt_gaussian_curvature = (principal_curvature1 * principal_curvature2).astype(np.float32)

                # GT mean curvature vector = GT normal * GT mean curvature
                gt_mean_curvature_vector = gt_vertex_normals * gt_mean_curvature[:, np.newaxis]

                print(f"GT mean curvature range: [{gt_mean_curvature.min():.6f}, {gt_mean_curvature.max():.6f}]")
                print(f"GT Gaussian curvature range: [{gt_gaussian_curvature.min():.6f}, {gt_gaussian_curvature.max():.6f}]")

            except Exception as e:
                print(f"Warning: Failed to compute GT curvatures with libigl: {e}")
                gt_mean_curvature = None
                gt_mean_curvature_vector = None
                gt_gaussian_curvature = None

            # Gradient operator — always needed (used for GT heat method geodesics)
            try:
                gt_mesh_grad_op = igl.grad(vertices_igl, faces_igl)
                gt_face_areas = (igl.doublearea(vertices_igl, faces_igl).flatten() / 2.0).astype(np.float64)
                print(f"GT mesh gradient operator: {gt_mesh_grad_op.shape}")
            except Exception as e:
                print(f"Warning: Failed to compute GT gradient operator: {e}")
                gt_mesh_grad_op = None
                gt_face_areas = None

        # Compute GT Laplacian eigendecomposition
        print("Computing GT Laplacian eigendecomposition...")
        gt_laplacian_time = 0.0
        try:
            import warnings

            # Create PyFM TriMesh object for L,M construction only
            pyfm_mesh = TriMesh(vertices, faces)

            t_gt_start = time.perf_counter()

            # Build cotangent Laplacian and area matrix via PyFM (no eigendecomposition)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pyfm_mesh.process(k=0, intrinsic=False, verbose=False)

            # Compute eigendecomposition using shared function (same as PRED/Robust/NeLo)
            gt_eigenvalues, gt_eigenvectors = compute_laplacian_eigendecomposition(
                pyfm_mesh.W, self.config.num_eigenvectors_to_show, mass_matrix=pyfm_mesh.A
            )

            t_gt_end = time.perf_counter()
            gt_laplacian_time = t_gt_end - t_gt_start

            self.timing_results.gt_laplacian_time = gt_laplacian_time
            print(f"[TIMING] GT Laplacian + eigen: {gt_laplacian_time * 1000:.2f} ms")

            vertex_areas = pyfm_mesh.vertex_areas

            print(f"Computed {len(gt_eigenvalues)} GT eigenvalues")
            print(f"GT eigenvalue range: [{gt_eigenvalues[0]:.2e}, {gt_eigenvalues[-1]:.6f}]")

        except Exception as e:
            print(f"Warning: Failed to compute GT eigendecomposition: {e}")
            gt_eigenvalues = None
            gt_eigenvectors = None
            vertex_areas = None


        # Note: robust-laplacian is computed later in process_batch with the actual k value
        # to ensure it uses point_cloud_laplacian with the same k as PRED

        return {
            'vertices': vertices,
            'faces': faces,
            'gt_vertex_normals': gt_vertex_normals,
            'gt_mean_curvature': gt_mean_curvature,
            'gt_mean_curvature_vector': gt_mean_curvature_vector,
            'gt_gaussian_curvature': gt_gaussian_curvature,
            'gt_mesh_grad_op': gt_mesh_grad_op,
            'gt_face_areas': gt_face_areas,
            'gt_eigenvalues': gt_eigenvalues,
            'gt_eigenvectors': gt_eigenvectors,
            'vertex_areas': vertex_areas,
            # Robust-laplacian data (computed later with actual k)
            'robust_eigenvalues': None,
            'robust_eigenvectors': None,
            'robust_vertex_areas': None
        }

    def _add_weight_neighborhood_visualization(self):
        """
        Visualize PRED stiffness weights on k-NN neighborhoods for FPS-sampled vertices.

        For each sampled vertex, registers:
        - A point cloud of its k-NN neighbors colored by stiffness weight (viridis)
        - A separate red point cloud for the center vertex

        Uses stored FPS sample indices (computed once per mesh, reused on k-change).
        """
        import polyscope as ps

        vertices = self.current_original_vertices
        weights = self.current_stiffness_weights  # (N, k)
        vertex_indices = self.current_vertex_indices  # (N*k,) flat
        center_indices = self.current_center_indices  # (N,)
        batch_indices = self.current_batch_indices    # (N*k,)

        if vertices is None or weights is None or vertex_indices is None:
            return

        # Move to CPU/numpy
        weights_np = weights.cpu().numpy() if torch.is_tensor(weights) else weights
        vertex_indices_np = vertex_indices.cpu().numpy() if torch.is_tensor(vertex_indices) else vertex_indices
        center_indices_np = center_indices.cpu().numpy() if torch.is_tensor(center_indices) else center_indices
        batch_indices_np = batch_indices.cpu().numpy() if torch.is_tensor(batch_indices) else batch_indices

        n_patches = weights_np.shape[0]
        k = weights_np.shape[1]

        # Compute batch sizes and starts to index into flat vertex_indices
        batch_sizes = np.bincount(batch_indices_np, minlength=n_patches)
        starts = np.concatenate([[0], np.cumsum(batch_sizes)[:-1]])

        # FPS sample indices: compute once per mesh, reuse on k-change
        if self._weight_nbr_sample_indices is None:
            self._weight_nbr_sample_indices = select_multiple_geodesic_sources(
                vertices, num_sources=self._weight_nbr_num_samples,
                method="farthest_point_sampling"
            )

        sample_indices = self._weight_nbr_sample_indices

        # First remove old neighborhood visualizations
        self._remove_weight_neighborhood_visualizations()

        # Global min/max for consistent color scale across all samples
        all_weights = weights_np.flatten()
        w_min, w_max = float(all_weights.min()), float(all_weights.max())

        print(f"\nAdding weight neighborhood visualizations for {len(sample_indices)} FPS-sampled vertices (k={k})...")

        for idx, center_vtx in enumerate(sample_indices):
            # Find the patch index for this center vertex
            # center_indices maps patch_i -> vertex_id, so we need to find patch with this vertex
            patch_matches = np.where(center_indices_np == center_vtx)[0]
            if len(patch_matches) == 0:
                continue
            patch_idx = patch_matches[0]

            # Get neighbor vertex indices for this patch
            start = starts[patch_idx]
            actual_k = batch_sizes[patch_idx]
            neighbor_vtx_indices = vertex_indices_np[start:start + actual_k]

            # Get neighbor positions and weights
            neighbor_positions = vertices[neighbor_vtx_indices]  # (actual_k, 3)
            patch_weights = weights_np[patch_idx, :actual_k]      # (actual_k,)

            # Register neighbor point cloud colored by weight
            nbr_name = f"W_Nbr {idx} (vtx {center_vtx})"
            pc = ps.register_point_cloud(
                name=nbr_name,
                points=neighbor_positions,
                radius=0.004,
                enabled=True
            )
            pc.add_scalar_quantity(
                name="stiffness_weight",
                values=patch_weights,
                enabled=True,
                cmap='viridis'
            )

            # Register center vertex as red dot
            center_name = f"W_Ctr {idx} (vtx {center_vtx})"
            ps.register_point_cloud(
                name=center_name,
                points=vertices[center_vtx:center_vtx + 1],
                radius=0.006,
                color=(1.0, 0.0, 0.0),
                enabled=True
            )

        print(f"  Added {len(sample_indices)} neighborhood visualizations")

    def _remove_weight_neighborhood_visualizations(self):
        """Remove all existing weight neighborhood point clouds."""
        import polyscope as ps
        try:
            all_structures = ps.get_all_structures()
        except Exception:
            return

        for kind_dict in all_structures.values():
            for name in list(kind_dict.keys()):
                if name.startswith("W_Nbr ") or name.startswith("W_Ctr "):
                    try:
                        ps.remove_point_cloud(name)
                    except Exception:
                        pass

    def _print_weight_distribution_diagnostic(
            self, stiffness_weights: torch.Tensor, attention_mask: torch.Tensor):
        """
        Print diagnostic statistics about the stiffness weight distribution.

        Helps determine whether the model concentrates weight on a few neighbors
        (sparse, good for pruning) or spreads it uniformly (dense, pruning loses info).

        Args:
            stiffness_weights: (num_patches, k) positive weights
            attention_mask: (num_patches, k) bool mask for valid entries
        """
        w = stiffness_weights.float()
        mask = attention_mask

        # Mask invalid entries
        w = w.masked_fill(~mask, 0.0)

        # Per-patch statistics
        k = w.shape[1]
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        w_norm = w / w_sum  # normalized to probability distribution

        # Entropy: 0 = all weight on one neighbor, log(k) = perfectly uniform
        log_w = torch.where(w_norm > 1e-12, w_norm.log(), torch.zeros_like(w_norm))
        entropy = -(w_norm * log_w).sum(dim=-1)  # (num_patches,)
        max_entropy = float(np.log(k))

        # Gini coefficient per patch: 0 = perfectly equal, 1 = all weight on one
        w_sorted, _ = w.sort(dim=-1)
        n = mask.sum(dim=-1).float()  # actual neighbors per patch
        cumw = w_sorted.cumsum(dim=-1)
        # Gini = 1 - 2 * (area under Lorenz curve) / (area under uniform)
        indices = torch.arange(1, k + 1, device=w.device).float().unsqueeze(0)
        gini = 1.0 - 2.0 * (cumw / w_sum).sum(dim=-1) / n + 1.0 / n

        # Top-k concentration: what fraction of total weight is in top-N neighbors
        w_desc, _ = w.sort(dim=-1, descending=True)
        total = w_sum.squeeze(-1)
        top1_frac = (w_desc[:, 0] / total) if k >= 1 else torch.zeros(1)
        top3_frac = (w_desc[:, :min(3, k)].sum(dim=-1) / total) if k >= 3 else top1_frac
        top6_frac = (w_desc[:, :min(6, k)].sum(dim=-1) / total) if k >= 6 else top3_frac

        # Weight ratio: max / min (among valid entries)
        w_valid = w.clone()
        w_valid[~mask] = float('inf')
        w_min_per_patch = w_valid.min(dim=-1).values
        w_valid[~mask] = float('-inf')
        w_max_per_patch = w_valid.max(dim=-1).values
        ratio = (w_max_per_patch / w_min_per_patch.clamp(min=1e-12))

        print(f"\n{'=' * 70}")
        print(f"STIFFNESS WEIGHT DISTRIBUTION (k={k})")
        print(f"{'=' * 70}")
        print(f"  Raw weights:       mean={w[mask].mean():.4f}, std={w[mask].std():.4f}, "
              f"min={w[mask].min():.4f}, max={w[mask].max():.4f}")
        print(f"  Max/min ratio:     mean={ratio.mean():.1f}, median={ratio.median():.1f}, "
              f"max={ratio.max():.1f}")
        print(f"  Entropy:           mean={entropy.mean():.3f} / {max_entropy:.3f} "
              f"({entropy.mean() / max_entropy * 100:.1f}% of uniform)")
        print(f"                     min={entropy.min():.3f}, max={entropy.max():.3f}")
        print(f"  Gini coefficient:  mean={gini.mean():.3f}, median={gini.median():.3f} "
              f"(0=equal, 1=concentrated)")
        print(f"  Weight concentration:")
        print(f"    Top-1 / total:   mean={top1_frac.mean():.3f}, median={top1_frac.median():.3f}")
        if k >= 3:
            print(f"    Top-3 / total:   mean={top3_frac.mean():.3f}, median={top3_frac.median():.3f}")
        if k >= 6:
            print(f"    Top-6 / total:   mean={top6_frac.mean():.3f}, median={top6_frac.median():.3f}")
        if k > 6:
            remaining = 1.0 - top6_frac.mean()
            print(f"    Remaining {k-6}:    mean={remaining:.3f} of total weight")

        # Per-position average weight (is 1st neighbor heavier than k-th?)
        # w_norm columns correspond to kNN order: col 0 = nearest, col k-1 = farthest
        per_pos_mean = w_norm.mean(dim=0)  # (k,)
        uniform_val = 1.0 / k
        print(f"  Per-position avg weight (kNN order, uniform={uniform_val:.4f}):")
        # Print in compact rows of 10
        for row_start in range(0, k, 10):
            row_end = min(row_start + 10, k)
            vals = [f"{per_pos_mean[i]:.4f}" for i in range(row_start, row_end)]
            labels = [f"{i:>2}" for i in range(row_start, row_end)]
            print(f"    pos [{row_start:>2}-{row_end-1:>2}]: {' '.join(vals)}")
        print(f"{'=' * 70}\n")

        # Store stats for CSV export
        self._weight_distribution_stats = {
            'weight_mean': float(w[mask].mean()),
            'weight_std': float(w[mask].std()),
            'weight_min': float(w[mask].min()),
            'weight_max': float(w[mask].max()),
            'max_min_ratio_mean': float(ratio.mean()),
            'max_min_ratio_median': float(ratio.median()),
            'max_min_ratio_max': float(ratio.max()),
            'entropy_mean': float(entropy.mean()),
            'entropy_max_possible': max_entropy,
            'entropy_ratio': float(entropy.mean() / max_entropy) if max_entropy > 0 else 0.0,
            'gini_mean': float(gini.mean()),
            'gini_median': float(gini.median()),
            'top1_frac_mean': float(top1_frac.mean()),
            'top3_frac_mean': float(top3_frac.mean()) if k >= 3 else float(top1_frac.mean()),
            'top6_frac_mean': float(top6_frac.mean()) if k >= 6 else float(top3_frac.mean()),
        }

    def perform_model_inference(self, model: LaplacianTransformerModule, batch_data: Data, device: torch.device) -> Dict[str, Any]:
        """
        Perform model inference and compute predicted quantities.

        Args:
            model: Trained LaplacianTransformerModule
            batch_data: Preprocessed batch data from MeshDataset
            device: Device for computation

        Returns:
            Dictionary containing predicted quantities, stiffness and mass matrices
        """
        print("Performing model inference...")

        # Log input size for debugging timing variations
        num_points = len(batch_data.pos)
        num_patches = len(batch_data.center_indices) if hasattr(batch_data, 'center_indices') else 'unknown'
        print(f"  Input size: {num_points} points, {num_patches} patches")

        # === TIME: Data transfer to device ===
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_transfer_start = time.perf_counter()

        batch_data = batch_data.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_transfer_end = time.perf_counter()
        transfer_time = t_transfer_end - t_transfer_start
        print(f"  [TIMING] Data transfer to GPU: {transfer_time * 1000:.2f} ms")

        # Determine mixed precision dtype (BF16 preferred on Ampere+, else FP16)
        use_amp = device.type == 'cuda'
        if use_amp:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
                print("  Using mixed precision: BF16")
            else:
                amp_dtype = torch.float16
                print("  Using mixed precision: FP16")

        with torch.no_grad():
            # === Single timed inference ===
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_inference_start = time.perf_counter()

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    forward_result = model._forward_pass(batch_data)
            else:
                forward_result = model._forward_pass(batch_data)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            pred_inference_time = time.perf_counter() - t_inference_start

            print(f"  [TIMING] Model inference: {pred_inference_time * 1000:.2f} ms")

            # === TIME: Result extraction and conversion ===
            t_extract_start = time.perf_counter()

            # Extract components from new forward result structure
            # Convert back to float32 for downstream processing (matrix assembly, eigendecomp)
            stiffness_weights = forward_result['stiffness_weights'].float()  # Shape: (batch_size, max_k)
            areas = forward_result['areas'].float()  # Shape: (batch_size,)
            attention_mask = forward_result['attention_mask']  # Shape: (batch_size, max_k)
            batch_sizes = forward_result['batch_sizes']  # Shape: (batch_size,)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_extract_end = time.perf_counter()
            extract_time = t_extract_end - t_extract_start

            print(f"Got stiffness weights shape: {stiffness_weights.shape}")
            print(f"Got areas shape: {areas.shape}")
            print(f"Got attention mask shape: {attention_mask.shape}")
            print(f"Got batch sizes: {batch_sizes}")
            print(f"Area statistics: mean={areas.mean():.6f}, std={areas.std():.6f}, min={areas.min():.6f}, max={areas.max():.6f}")

            # === WEIGHT DISTRIBUTION DIAGNOSTIC ===
            self._print_weight_distribution_diagnostic(stiffness_weights, attention_mask)

            # Use patch_idx if available (MeshDataset), otherwise use batch (synthetic)
            batch_indices = getattr(batch_data, 'patch_idx', batch_data.batch)

            # === TIME: Matrix assembly ===
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_assembly_start = time.perf_counter()

            # Assemble separate stiffness and mass matrices
            top_k = self.reconstruction_settings.pred_top_k if self.reconstruction_settings.enable_top_k_pruning else None
            sym_policy = self.reconstruction_settings.symmetry_policy
            stiffness_matrix, mass_matrix = assemble_stiffness_and_mass_matrices(
                stiffness_weights=stiffness_weights,
                areas=areas,
                attention_mask=attention_mask,
                vertex_indices=batch_data.vertex_indices,
                center_indices=batch_data.center_indices,
                batch_indices=batch_indices,
                top_k=top_k,
                symmetry_policy=sym_policy
            )

            t_assembly_end = time.perf_counter()
            pred_assembly_time = t_assembly_end - t_assembly_start

            print(f"Assembled stiffness matrix: {stiffness_matrix.shape} ({stiffness_matrix.nnz} non-zeros)")
            print(f"Assembled mass matrix: {mass_matrix.shape} (diagonal)")

            # Store timing results with detailed breakdown
            self.timing_results.pred_model_inference_time = pred_inference_time
            self.timing_results.pred_matrix_assembly_time = pred_assembly_time
            print(f"[TIMING] Data transfer:    {transfer_time * 1000:>8.2f} ms")
            print(f"[TIMING] Model inference:  {pred_inference_time * 1000:>8.2f} ms")
            print(f"[TIMING] Result extract:   {extract_time * 1000:>8.2f} ms")
            print(f"[TIMING] Matrix assembly:  {pred_assembly_time * 1000:>8.2f} ms")

            # Compute eigendecomposition using generalized eigenvalue problem
            predicted_eigenvalues, predicted_eigenvectors = self.compute_eigendecomposition(
                stiffness_matrix, k=self.config.num_eigenvectors_to_show, mass_matrix=mass_matrix
            )

            if predicted_eigenvalues is not None:
                print(f"Computed {len(predicted_eigenvalues)} predicted eigenvalues")
                print(f"Predicted eigenvalue range: [{predicted_eigenvalues[0]:.2e}, {predicted_eigenvalues[-1]:.6f}]")

        return {
            'stiffness_matrix': stiffness_matrix,
            'mass_matrix': mass_matrix,
            'predicted_eigenvalues': predicted_eigenvalues,
            'predicted_eigenvectors': predicted_eigenvectors,
            'stiffness_weights': stiffness_weights.detach(),  # keep as tensor (GPU or CPU)
            'areas': areas.detach(),
            'attention_mask': attention_mask.detach(),
            'batch_sizes': batch_sizes.detach(),
            'forward_result': forward_result  # Store complete forward result
        }

    def compute_predicted_quantities_from_laplacian(self, stiffness_matrix: scipy.sparse.csr_matrix,
                                                    vertices: np.ndarray,
                                                    mass_matrix: scipy.sparse.csr_matrix = None) -> Dict[str, np.ndarray]:
        """
        Compute predicted mean curvature vector and derived quantities from predicted matrices.

        The mean curvature vector is computed as: Delta_r = M^-1 @ S @ r = 2Hn

        Args:
            stiffness_matrix: Predicted stiffness matrix (scipy sparse)
            vertices: Mesh vertices array of shape (N, 3)
            mass_matrix: Predicted mass matrix (scipy sparse diagonal)

        Returns:
            Dictionary containing predicted quantities
        """
        print("Computing predicted quantities from stiffness/mass matrices...")

        try:
            # Compute S @ r first
            stiffness_times_vertices = stiffness_matrix @ vertices  # Shape: (N, 3)

            # If mass matrix is provided, compute M^-1 @ S @ r
            if mass_matrix is not None:
                # Extract diagonal of mass matrix and invert
                mass_diag = mass_matrix.diagonal()
                # Avoid division by zero
                mass_diag_inv = np.where(mass_diag > 1e-10, 1.0 / mass_diag, 0.0)

                # Apply M^-1 (element-wise division since M is diagonal)
                predicted_mean_curvature_vector = stiffness_times_vertices * mass_diag_inv[:, np.newaxis]
            else:
                # Without mass matrix, just use S @ r
                predicted_mean_curvature_vector = stiffness_times_vertices

            # Compute magnitudes (predicted mean curvature values)
            predicted_mean_curvature = np.linalg.norm(predicted_mean_curvature_vector, axis=1)  # Shape: (N,)

            # Compute predicted normals (normalized mean curvature vectors)
            predicted_normals = np.zeros_like(predicted_mean_curvature_vector)
            non_zero_mask = predicted_mean_curvature > 1e-10

            predicted_normals[non_zero_mask] = (
                    predicted_mean_curvature_vector[non_zero_mask] /
                    predicted_mean_curvature[non_zero_mask, np.newaxis]
            )

            # For zero curvature points, use a default normal (e.g., z-up)
            predicted_normals[~non_zero_mask] = np.array([0, 0, 1])

            print(f"Predicted mean curvature range: [{predicted_mean_curvature.min():.6f}, {predicted_mean_curvature.max():.6f}]")
            print(f"Zero curvature points: {(~non_zero_mask).sum()}/{len(predicted_mean_curvature)}")

            return {
                'predicted_mean_curvature_vector': predicted_mean_curvature_vector,  # Raw, unnormalized vectors
                'predicted_normals': predicted_normals,
                'predicted_mean_curvature': predicted_mean_curvature
            }

        except Exception as e:
            print(f"Error computing predicted quantities from matrices: {e}")
            return {}

    def _face_gradients_to_vertex_gradients(self, face_grads: np.ndarray, faces: np.ndarray,
                                             face_areas: np.ndarray, num_vertices: int) -> np.ndarray:
        """
        Average face-based gradient vectors to vertices, weighted by face area.

        Args:
            face_grads: Per-face gradient vectors (nF, 3)
            faces: Face indices (nF, 3)
            face_areas: Per-face areas (nF,)
            num_vertices: Number of vertices N

        Returns:
            Per-vertex gradient vectors (N, 3)
        """
        vertex_grads = np.zeros((num_vertices, 3), dtype=np.float64)
        vertex_area_sum = np.zeros(num_vertices, dtype=np.float64)

        # Accumulate area-weighted face gradients to each vertex
        for d in range(3):  # For each vertex of the triangle
            np.add.at(vertex_grads[:, 0], faces[:, d], face_areas * face_grads[:, 0])
            np.add.at(vertex_grads[:, 1], faces[:, d], face_areas * face_grads[:, 1])
            np.add.at(vertex_grads[:, 2], faces[:, d], face_areas * face_grads[:, 2])
            np.add.at(vertex_area_sum, faces[:, d], face_areas)

        # Normalize by total area at each vertex
        nonzero = vertex_area_sum > 1e-16
        vertex_grads[nonzero] /= vertex_area_sum[nonzero, np.newaxis]

        return vertex_grads.astype(np.float32)

    def _compute_gradient_of_scalar_field_mesh(self, scalar_field: np.ndarray,
                                                 gt_mesh_grad_op: scipy.sparse.csr_matrix,
                                                 faces: np.ndarray, face_areas: np.ndarray,
                                                 num_vertices: int) -> Optional[np.ndarray]:
        """
        Compute surface gradient of a scalar field using the GT mesh gradient operator (igl.grad).

        igl.grad returns a (3*nF, N) matrix. Applying it gives face-based gradients
        which are then averaged to vertices weighted by face area.

        Args:
            scalar_field: Per-vertex scalar (N,)
            gt_mesh_grad_op: igl.grad matrix (3*nF, N)
            faces: Face indices (nF, 3)
            face_areas: Per-face areas (nF,)
            num_vertices: Number of vertices N

        Returns:
            Per-vertex gradient vectors (N, 3) or None on failure
        """
        try:
            nF = len(faces)
            # Compute face-based gradients: (3*nF,) -> reshape to (nF, 3) in x,y,z blocks
            grad_flat = gt_mesh_grad_op @ scalar_field.astype(np.float64)
            face_grads = np.column_stack([
                grad_flat[:nF],      # x component
                grad_flat[nF:2*nF],  # y component
                grad_flat[2*nF:]     # z component
            ])
            return self._face_gradients_to_vertex_gradients(face_grads, faces, face_areas, num_vertices)
        except Exception as e:
            print(f"  [!] Failed to compute mesh gradient: {e}")
            return None

    def _compute_gradient_of_scalar_field_learned(self, scalar_field: np.ndarray,
                                                    gradient_operator: scipy.sparse.csr_matrix,
                                                    num_vertices: int) -> Optional[np.ndarray]:
        """
        Compute surface gradient of a scalar field using the learned gradient operator G.

        G is (3N, N). G @ f gives a (3N,) vector that reshapes to (N, 3) â€” per-vertex gradients.

        Args:
            scalar_field: Per-vertex scalar (N,)
            gradient_operator: Learned G matrix (3N, N)
            num_vertices: Number of vertices N

        Returns:
            Per-vertex gradient vectors (N, 3) or None on failure
        """
        try:
            grad_flat = gradient_operator @ scalar_field.astype(np.float64)
            return grad_flat.reshape(num_vertices, 3).astype(np.float32)
        except Exception as e:
            print(f"  [!] Failed to compute learned gradient: {e}")
            return None

    def compute_mesh_reconstruction(self, original_vertices: np.ndarray, eigenvectors: np.ndarray,
                                    eigenvalues: np.ndarray, max_eigenvectors: int,
                                    vertex_areas: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Reconstruct mesh geometry using progressive number of eigenvectors.

        Args:
            original_vertices: Original mesh vertices of shape (N, 3)
            eigenvectors: Eigenvectors of shape (N, k)
            eigenvalues: Eigenvalues of shape (k,)
            max_eigenvectors: Maximum number of eigenvectors to use
            vertex_areas: Optional vertex areas for area-weighted computation (GT case)

        Returns:
            List of reconstructed vertex arrays, one for each number of eigenvectors (1, 2, ..., max_eigenvectors)
        """
        if eigenvectors is None or eigenvalues is None:
            return []

        num_available = min(eigenvectors.shape[1], max_eigenvectors)
        reconstructed_meshes = []

        if vertex_areas is not None:
            print(f"Computing area-weighted mesh reconstruction with up to {num_available} eigenvectors...")
            reconstructed_meshes = self._compute_area_weighted_reconstruction(
                original_vertices, eigenvectors, num_available, vertex_areas
            )
        else:
            print(f"Computing standard mesh reconstruction with up to {num_available} eigenvectors...")
            reconstructed_meshes = self._compute_standard_reconstruction(
                original_vertices, eigenvectors, num_available
            )

        print(f"Generated {len(reconstructed_meshes)} reconstructed meshes")
        return reconstructed_meshes

    def _compute_area_weighted_reconstruction(self, original_vertices: np.ndarray, eigenvectors: np.ndarray,
                                              num_available: int, vertex_areas: np.ndarray) -> List[np.ndarray]:
        """
        Compute mesh reconstruction using area-weighted inner products (fully vectorized).

        For M-orthonormal eigenvectors from generalized eigenvalue problem (S v = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â» M v),
        the Gram matrix G = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦^T M ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ = I, so projection coefficients simplify to:
            c = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦^T M f
        and reconstruction is:
            f_l = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦_l c_l = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â£_{i=0}^{l-1} ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â _i (ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â _i^T M f)

        This vectorized implementation computes all progressive reconstructions efficiently
        using cumulative sums, avoiding Python loops entirely.

        Args:
            original_vertices: Original mesh vertices f in R^{n x 3}
            eigenvectors: Eigenvectors Phi in R^{n x k} (M-orthonormal: ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦^T M ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ = I)
            num_available: Number of available eigenvectors to use
            vertex_areas: Vertex areas a in R^n (diagonal of mass matrix M)

        Returns:
            List of reconstructed vertex arrays [f_1, f_2, ..., f_L] where f_l uses l eigenvectors
        """
        if num_available == 0:
            return []

        # Apply diagonal mass matrix efficiently: M @ f = diag(areas) @ f
        # No need to form full nÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Ân matrix - just element-wise multiplication
        M_f = vertex_areas[:, np.newaxis] * original_vertices  # (n, 3)

        # Compute all M-weighted coefficients at once: c = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦^T M f
        # Since ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ is M-orthonormal, these are the exact projection coefficients
        Phi = eigenvectors[:, :num_available]  # (n, L)
        coefficients = Phi.T @ M_f  # (L, 3)

        # Compute contribution from each eigenvector via broadcasting:
        # contribution_i = ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â _i ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â c_i^T (outer product, but c_i is a row vector)
        # Shape: (n, L, 1) * (1, L, 3) ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ (n, L, 3)
        contributions = Phi[:, :, np.newaxis] * coefficients[np.newaxis, :, :]

        # Cumulative sum along eigenvector axis gives progressive reconstructions:
        # cumulative[:, l, :] = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â£_{i=0}^{l} contribution_i = reconstruction using (l+1) eigenvectors
        cumulative = np.cumsum(contributions, axis=1)  # (n, L, 3)

        # Convert to list of (n, 3) arrays
        return [cumulative[:, i, :] for i in range(num_available)]

    def _compute_standard_reconstruction(self, original_vertices: np.ndarray, eigenvectors: np.ndarray,
                                         num_available: int) -> List[np.ndarray]:
        """
        Compute mesh reconstruction using standard Euclidean inner products (optimized).

        Solves the least squares problem for each l:
            min_c ||f - ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦_l c||_2^2

        Solution: c = (ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦_l^T ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦_l)^{-1} ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦_l^T f, then f_l = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦_l c

        Note: Eigenvectors from generalized EVP are M-orthonormal, not L2-orthonormal.
        Even after L2-renormalization, they are L2-normalized but NOT L2-orthogonal.
        So we must compute the actual Gram matrix G = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦^T ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦.

        Optimized by precomputing ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦^T ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ and ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦^T f once, then extracting submatrices.

        Args:
            original_vertices: Original mesh vertices of shape (N, 3)
            eigenvectors: Eigenvectors of shape (N, k)
            num_available: Number of available eigenvectors to use

        Returns:
            List of reconstructed vertex arrays [f_1, f_2, ..., f_L] where f_l uses l eigenvectors
        """
        if num_available == 0:
            return []

        Phi = eigenvectors[:, :num_available]  # (n, L)

        # Precompute full Gram matrix G_full = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦^T ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ and projection target b_full = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦^T f
        G_full = Phi.T @ Phi  # (L, L)
        b_full = Phi.T @ original_vertices  # (L, 3)

        reconstructed_meshes = []
        for l in range(1, num_available + 1):
            # Extract lÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Âl submatrix and lÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â3 subvector
            G_l = G_full[:l, :l]
            b_l = b_full[:l, :]

            # Solve G_l c = b_l for coefficients
            try:
                c = np.linalg.solve(G_l, b_l)  # (l, 3)
            except np.linalg.LinAlgError:
                c = np.linalg.pinv(G_l) @ b_l

            # Reconstruct: f_l = ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦_l c
            reconstructed_meshes.append(Phi[:, :l] @ c)

        return reconstructed_meshes

    def compute_eigenvector_correlations(self, gt_eigenvectors: np.ndarray, pred_eigenvectors: np.ndarray) -> np.ndarray:
        """Compute correlation matrix between ground-truth and predicted eigenvectors."""
        min_cols = min(gt_eigenvectors.shape[1], pred_eigenvectors.shape[1])
        correlation_matrix = np.zeros((min_cols, min_cols))

        for i in range(min_cols):
            for j in range(min_cols):
                # Compute absolute correlation (eigenvectors can have sign ambiguity)
                corr = np.abs(np.corrcoef(gt_eigenvectors[:, i], pred_eigenvectors[:, j])[0, 1])
                correlation_matrix[i, j] = corr

        return correlation_matrix

    def visualize_mesh_reconstructions(self, original_faces: np.ndarray, gt_reconstructions: List[np.ndarray],
                                       pred_reconstructions: List[np.ndarray], robust_reconstructions: List[np.ndarray],
                                       gt_eigenvalues: Optional[np.ndarray],
                                       pred_eigenvalues: Optional[np.ndarray],
                                       robust_eigenvalues: Optional[np.ndarray],
                                       nelo_reconstructions: Optional[List[np.ndarray]] = None,
                                       nelo_eigenvalues: Optional[np.ndarray] = None):
        """
        Visualize progressive mesh reconstructions using eigenvectors.
        GT (PyFM) on the right, PRED in front, Robust on the left, NeLo behind.
        """
        print("Adding mesh reconstruction visualizations...")
        nelo_reconstructions = nelo_reconstructions or []

        # Fixed positions for overlaid reconstructions
        gt_offset     = np.array([ 3.0, 0.0,  0.0])   # GT on the right
        pred_offset   = np.array([ 0.0, 0.0, -3.0])   # PRED in front
        robust_offset = np.array([-3.0, 0.0,  0.0])   # Robust on the left
        nelo_offset   = np.array([ 0.0, 0.0,  3.0])   # NeLo behind

        # Visualize GT (PyFM) reconstructions (all overlaid on the right)
        for i, gt_vertices in enumerate(gt_reconstructions):
            num_eigenvecs = i + 1
            gt_eigenval = gt_eigenvalues[i] if gt_eigenvalues is not None else 0.0

            # Position all GT reconstructions at the same location (right side)
            offset_vertices = gt_vertices + gt_offset

            mesh_name = f"GT-PyFM Recon {num_eigenvecs:02d} eigenvec (lambda={gt_eigenval:.3f})"

            try:
                gt_mesh = ps.register_surface_mesh(
                    name=mesh_name,
                    vertices=offset_vertices,
                    faces=original_faces,
                    enabled=(i == 0)  # Only enable the first one by default
                )

                # Track this structure for later removal
                self.reconstruction_structure_names.append(mesh_name)

                # Color GT reconstructions in blue tones with slight variation
                blue_intensity = 0.5 + 0.5 * (i / max(1, len(gt_reconstructions) - 1))
                mesh_color = np.array([0.2, 0.4, blue_intensity])
                vertex_colors = np.tile(mesh_color, (len(offset_vertices), 1))
                gt_mesh.add_color_quantity(
                    name="gt_color",
                    values=vertex_colors,
                    enabled=True
                )

            except Exception as e:
                print(f"Warning: Failed to visualize GT reconstruction {num_eigenvecs}: {e}")

        # Visualize predicted reconstructions (in front)
        for i, pred_vertices in enumerate(pred_reconstructions):
            num_eigenvecs = i + 1
            pred_eigenval = pred_eigenvalues[i] if pred_eigenvalues is not None else 0.0

            # Position all PRED reconstructions at the same location (front)
            offset_vertices = pred_vertices + pred_offset

            # Include reconstruction method in name
            method_suffix = " (Pred Areas)" if self.reconstruction_settings.use_pred_areas else " (Standard)"
            mesh_name = f"PRED Recon {num_eigenvecs:02d} eigenvec (lambda={pred_eigenval:.3f}){method_suffix}"

            try:
                pred_mesh = ps.register_surface_mesh(
                    name=mesh_name,
                    vertices=offset_vertices,
                    faces=original_faces,
                    enabled=(i == 0)  # Only enable the first one by default
                )

                # Track this structure for later removal
                self.reconstruction_structure_names.append(mesh_name)

                # Color predicted reconstructions in orange tones with slight variation
                orange_intensity = 0.5 + 0.5 * (i / max(1, len(pred_reconstructions) - 1))
                mesh_color = np.array([orange_intensity, 0.4, 0.2])
                vertex_colors = np.tile(mesh_color, (len(offset_vertices), 1))
                pred_mesh.add_color_quantity(
                    name="pred_color",
                    values=vertex_colors,
                    enabled=True
                )

            except Exception as e:
                print(f"Warning: Failed to visualize predicted reconstruction {num_eigenvecs}: {e}")

        # Visualize robust-laplacian reconstructions (all overlaid on the left)
        for i, robust_vertices in enumerate(robust_reconstructions):
            num_eigenvecs = i + 1
            robust_eigenval = robust_eigenvalues[i] if robust_eigenvalues is not None else 0.0

            # Position all robust reconstructions at the same location (left side)
            offset_vertices = robust_vertices + robust_offset

            mesh_name = f"Robust Recon {num_eigenvecs:02d} eigenvec (lambda={robust_eigenval:.3f})"

            try:
                robust_mesh = ps.register_surface_mesh(
                    name=mesh_name,
                    vertices=offset_vertices,
                    faces=original_faces,
                    enabled=(i == 0)  # Only enable the first one by default
                )

                # Track this structure for later removal
                self.reconstruction_structure_names.append(mesh_name)

                # Color robust reconstructions in green tones with slight variation
                green_intensity = 0.5 + 0.5 * (i / max(1, len(robust_reconstructions) - 1))
                mesh_color = np.array([0.2, green_intensity, 0.3])
                vertex_colors = np.tile(mesh_color, (len(offset_vertices), 1))
                robust_mesh.add_color_quantity(
                    name="robust_color",
                    values=vertex_colors,
                    enabled=True
                )

            except Exception as e:
                print(f"Warning: Failed to visualize robust reconstruction {num_eigenvecs}: {e}")

        # Visualize NeLo reconstructions (behind)
        for i, nelo_vertices in enumerate(nelo_reconstructions):
            num_eigenvecs = i + 1
            nelo_eigenval = nelo_eigenvalues[i] if nelo_eigenvalues is not None and i < len(nelo_eigenvalues) else 0.0

            offset_vertices = nelo_vertices + nelo_offset
            mesh_name = f"NeLo Recon {num_eigenvecs:02d} eigenvec (lambda={nelo_eigenval:.3f})"

            try:
                nelo_mesh = ps.register_surface_mesh(
                    name=mesh_name,
                    vertices=offset_vertices,
                    faces=original_faces,
                    enabled=(i == 0)
                )
                self.reconstruction_structure_names.append(mesh_name)

                # Purple tones for NeLo
                purple_intensity = 0.5 + 0.5 * (i / max(1, len(nelo_reconstructions) - 1))
                mesh_color = np.array([purple_intensity, 0.2, purple_intensity])
                vertex_colors = np.tile(mesh_color, (len(offset_vertices), 1))
                nelo_mesh.add_color_quantity(name="nelo_color", values=vertex_colors, enabled=True)

            except Exception as e:
                print(f"Warning: Failed to visualize NeLo reconstruction {num_eigenvecs}: {e}")

        print(f"Added {len(gt_reconstructions)} GT-PyFM, {len(pred_reconstructions)} PRED, {len(robust_reconstructions)} Robust, and {len(nelo_reconstructions)} NeLo mesh reconstructions")
        print("Toggle visibility to compare different numbers of eigenvectors")

    def print_eigenvalue_analysis(self, gt_eigenvalues: Optional[np.ndarray],
                                  predicted_eigenvalues: Optional[np.ndarray],
                                  mesh_name: str,
                                  robust_eigenvalues: Optional[np.ndarray] = None):
        """Print detailed eigenvalue comparison analysis."""
        print(f"\n" + "-" * 70)
        print(f"EIGENVALUE COMPARISON ANALYSIS - {mesh_name}")
        print("-" * 70)

        # Ground-truth (PyFM) analysis
        if gt_eigenvalues is not None:
            print("GT (PyFM) EIGENVALUES:")
            print(f"  Number of eigenvalues: {len(gt_eigenvalues)}")
            print(f"  First eigenvalue (should be ~0): {gt_eigenvalues[0]:.2e}")
            if len(gt_eigenvalues) > 1:
                print(f"  Second eigenvalue (Fiedler): {gt_eigenvalues[1]:.6f}")
                print(f"  Spectral gap: {gt_eigenvalues[1] - gt_eigenvalues[0]:.6f}")
            print(f"  Largest eigenvalue: {gt_eigenvalues[-1]:.6f}")

        # Predicted analysis
        if predicted_eigenvalues is not None:
            print("\nPREDICTED EIGENVALUES:")
            print(f"  Number of eigenvalues: {len(predicted_eigenvalues)}")
            print(f"  First eigenvalue (should be ~0): {predicted_eigenvalues[0]:.2e}")
            if len(predicted_eigenvalues) > 1:
                print(f"  Second eigenvalue (Fiedler): {predicted_eigenvalues[1]:.6f}")
                print(f"  Spectral gap: {predicted_eigenvalues[1] - predicted_eigenvalues[0]:.6f}")
            print(f"  Largest eigenvalue: {predicted_eigenvalues[-1]:.6f}")

        # Robust-laplacian analysis
        if robust_eigenvalues is not None:
            print("\nROBUST-LAPLACIAN EIGENVALUES:")
            print(f"  Number of eigenvalues: {len(robust_eigenvalues)}")
            print(f"  First eigenvalue (should be ~0): {robust_eigenvalues[0]:.2e}")
            if len(robust_eigenvalues) > 1:
                print(f"  Second eigenvalue (Fiedler): {robust_eigenvalues[1]:.6f}")
                print(f"  Spectral gap: {robust_eigenvalues[1] - robust_eigenvalues[0]:.6f}")
            print(f"  Largest eigenvalue: {robust_eigenvalues[-1]:.6f}")

        # Comparison metrics: PRED vs GT
        if gt_eigenvalues is not None and predicted_eigenvalues is not None:
            min_len = min(len(gt_eigenvalues), len(predicted_eigenvalues))
            if min_len > 0:
                gt_subset = gt_eigenvalues[:min_len]
                pred_subset = predicted_eigenvalues[:min_len]

                abs_errors = np.abs(pred_subset - gt_subset)

                print(f"\nPREDICTED VS GT (PyFM) COMPARISON:")
                print(f"  Mean absolute error: {abs_errors.mean():.6f}")
                print(f"  Max absolute error: {abs_errors.max():.6f}")
                print(f"  Correlation coefficient: {np.corrcoef(gt_subset, pred_subset)[0, 1]:.6f}")

        # Comparison metrics: Robust vs GT
        if gt_eigenvalues is not None and robust_eigenvalues is not None:
            min_len = min(len(gt_eigenvalues), len(robust_eigenvalues))
            if min_len > 0:
                gt_subset = gt_eigenvalues[:min_len]
                robust_subset = robust_eigenvalues[:min_len]

                abs_errors = np.abs(robust_subset - gt_subset)

                print(f"\nROBUST VS GT (PyFM) COMPARISON:")
                print(f"  Mean absolute error: {abs_errors.mean():.6f}")
                print(f"  Max absolute error: {abs_errors.max():.6f}")
                print(f"  Correlation coefficient: {np.corrcoef(gt_subset, robust_subset)[0, 1]:.6f}")

        print("-" * 70)

    def print_eigenvector_correlation_analysis(self, correlation_matrix: np.ndarray, comparison_name: str):
        """Print eigenvector correlation analysis."""
        print(f"\nEIGENVECTOR CORRELATION ANALYSIS ({comparison_name})")
        print("-" * 50)

        # Find best matches for each GT eigenvector
        print("Best matches for each GT eigenvector:")
        for i in range(min(8, correlation_matrix.shape[0])):
            best_match_idx = np.argmax(correlation_matrix[i, :])
            best_correlation = correlation_matrix[i, best_match_idx]
            print(f"  GT Eigenvector {i} -> Pred Eigenvector {best_match_idx}: {best_correlation:.4f}")

        # Overall statistics
        diagonal_corrs = np.diag(correlation_matrix)
        print(f"\nDiagonal correlations (GT_i vs PRED_i):")
        print(f"  Mean: {diagonal_corrs.mean():.4f}")
        print(f"  Min: {diagonal_corrs.min():.4f}")
        print(f"  Max: {diagonal_corrs.max():.4f}")

        well_aligned = diagonal_corrs > 0.8
        print(f"  Well-aligned (>0.8): {well_aligned.sum()}/{len(diagonal_corrs)}")

    def visualize_mesh(self, vertices: np.ndarray, vertex_normals: np.ndarray, faces: np.ndarray):
        """Visualize the base mesh."""
        # Register mesh as surface mesh
        if len(faces) > 0:
            mesh_surface = ps.register_surface_mesh(
                name="Mesh Surface",
                vertices=vertices,
                faces=faces,
                enabled=True
            )

            print(f"Registered mesh surface with {len(vertices)} vertices and {len(faces)} faces")
            return mesh_surface
        else:
            # Fallback to point cloud if no faces
            mesh_cloud = ps.register_point_cloud(
                name="Mesh",
                points=vertices,
                radius=self.config.point_radius,
                enabled=True
            )

            print(f"Registered mesh point cloud with {len(vertices)} vertices")
            return mesh_cloud

    def add_comprehensive_curvature_visualizations(self, mesh_structure,
                                                   gt_data: Dict[str, np.ndarray],
                                                   predicted_data: Dict[str, np.ndarray],
                                                   inference_result: Dict[str, Any] = None):
        """
        Add comprehensive curvature and normal visualizations to the mesh.

        Visualization naming scheme:
        - A: Mean Curvature - GT (scalar)
        - B: Mean Curvature - PRED (scalar)
        - C: Mean Curvature Vector - GT (raw Hn vector)
        - D: Mean Curvature Vector - PRED (raw Hn vector)
        - E0: Mesh Normals - GT (from mesh geometry)
        - E: Normal (from MCV) - GT (normalized Hn)
        - F: Normal (from MCV) - PRED (normalized Hn)
        - G1: Normal Alignment (GT MCV vs PRED MCV) (cosine similarity)
        - G2: Normal Alignment (PRED vs Mesh GT) (cosine similarity)
        - H: Vertex Areas - GT (from PyFM)
        - I: Vertex Areas - PRED (from model's area head)
        """
        print("Adding comprehensive curvature visualizations...")

        # === VERTEX AREAS ===
        if gt_data.get('vertex_areas') is not None:
            mesh_structure.add_scalar_quantity(
                name="H Vertex Areas - GT",
                values=gt_data['vertex_areas'],
                enabled=False,
                cmap='viridis'
            )

        if inference_result is not None and inference_result.get('areas') is not None:
            areas_np = inference_result['areas'].cpu().numpy() if torch.is_tensor(inference_result['areas']) else inference_result['areas']
            mesh_structure.add_scalar_quantity(
                name="I Vertex Areas - PRED",
                values=areas_np,
                enabled=False,
                cmap='viridis'
            )

        # === MEAN CURVATURE ===
        if gt_data.get('gt_mean_curvature') is not None:
            mesh_structure.add_scalar_quantity(
                name="A Mean Curvature - GT",
                values=gt_data['gt_mean_curvature'],
                enabled=False,
                cmap='plasma'
            )

            # Add absolute value of GT mean curvature
            mesh_structure.add_scalar_quantity(
                name="A2 |Mean Curvature| - GT",
                values=np.abs(gt_data['gt_mean_curvature']),
                enabled=False,
                cmap='viridis'
            )

        if predicted_data.get('predicted_mean_curvature') is not None:
            mesh_structure.add_scalar_quantity(
                name="B Mean Curvature - PRED",
                values=predicted_data['predicted_mean_curvature'],
                enabled=False,
                cmap='plasma'
            )

        # === MEAN CURVATURE VECTORS (RAW, UNNORMALIZED) ===
        if gt_data.get('gt_mean_curvature_vector') is not None:
            mesh_structure.add_vector_quantity(
                name="C Mean Curvature Vector - GT",
                values=gt_data['gt_mean_curvature_vector'] * self.vector_scales.get_vector_scale('gt_mean_curvature_vector'),
                enabled=False,
                color=(0.0, 1.0, 1.0),  # Cyan
                vectortype="ambient"
            )

        if predicted_data.get('predicted_mean_curvature_vector') is not None:
            mesh_structure.add_vector_quantity(
                name="D Mean Curvature Vector - PRED",
                values=predicted_data['predicted_mean_curvature_vector'] * self.vector_scales.get_vector_scale('predicted_mean_curvature_vector'),
                enabled=False,
                color=(1.0, 0.5, 0.0),  # Orange
                vectortype="ambient"
            )

        # === NORMALIZED MEAN CURVATURE VECTORS (SURFACE NORMALS) ===
        # These are the mean curvature vectors normalized to unit length,
        # representing the surface normal direction derived from the Laplacian

        # First, add GT mesh normals (from mesh geometry) for reference
        if gt_data.get('gt_vertex_normals') is not None:
            mesh_structure.add_vector_quantity(
                name="E0 Mesh Normals - GT",
                values=gt_data['gt_vertex_normals'] * self.vector_scales.get_vector_scale('gt_normals'),
                enabled=False,
                color=(0.0, 1.0, 0.0),  # Green
                vectortype="ambient"
            )

        if gt_data.get('gt_mean_curvature_vector') is not None:
            gt_mcv = gt_data['gt_mean_curvature_vector']
            gt_mcv_norm = np.linalg.norm(gt_mcv, axis=1, keepdims=True)
            gt_normals_from_mcv = np.where(gt_mcv_norm > 1e-10, gt_mcv / gt_mcv_norm, 0)

            mesh_structure.add_vector_quantity(
                name="E Normal (from MCV) - GT",
                values=gt_normals_from_mcv * self.vector_scales.get_vector_scale('gt_normals'),
                enabled=False,
                color=(0.0, 0.8, 0.8),  # Dark Cyan
                vectortype="ambient"
            )

        if predicted_data.get('predicted_normals') is not None:
            mesh_structure.add_vector_quantity(
                name="F Normal (from MCV) - PRED",
                values=predicted_data['predicted_normals'] * self.vector_scales.get_vector_scale('predicted_normals'),
                enabled=False,
                color=(0.8, 0.4, 0.0),  # Dark Orange
                vectortype="ambient"
            )

        # === COMPARISON METRICS FOR MEAN CURVATURE VECTORS ===
        gt_mean_curv_vector = gt_data.get('gt_mean_curvature_vector')
        pred_normals = predicted_data.get('predicted_normals')
        gt_mesh_normals = gt_data.get('gt_vertex_normals')

        # Mean curvature vector alignment comparison (normal alignment between GT MCV and PRED MCV)
        if gt_mean_curv_vector is not None and pred_normals is not None:
            # Normalize GT mean curvature vector
            gt_norm = np.linalg.norm(gt_mean_curv_vector, axis=1, keepdims=True)
            gt_normalized = np.where(gt_norm > 1e-10, gt_mean_curv_vector / gt_norm, 0)

            # Cosine similarity between normalized vectors (normal alignment)
            normal_alignment = np.sum(pred_normals * gt_normalized, axis=1)
            mesh_structure.add_scalar_quantity(
                name="G1 Normal Alignment (GT MCV vs PRED MCV)",
                values=normal_alignment,
                enabled=False,
                cmap='coolwarm'
            )

        # Alignment between predicted normals and GT mesh normals
        if pred_normals is not None and gt_mesh_normals is not None:
            # Cosine similarity between predicted normal and GT mesh normal
            pred_vs_mesh_alignment = np.sum(pred_normals * gt_mesh_normals, axis=1)
            mesh_structure.add_scalar_quantity(
                name="G2 Normal Alignment (PRED vs Mesh GT)",
                values=pred_vs_mesh_alignment,
                enabled=False,
                cmap='coolwarm'
            )

        # === GAUSSIAN CURVATURE (SCALAR) ===
        if gt_data.get('gt_gaussian_curvature') is not None:
            mesh_structure.add_scalar_quantity(
                name="A3 Gaussian Curvature - GT",
                values=gt_data['gt_gaussian_curvature'],
                enabled=False,
                cmap='plasma'
            )

        # === SURFACE GRADIENT VECTOR FIELDS OF CURVATURES ===
        # Compute grad(H) and grad(K) using both GT mesh gradient and learned gradient operators
        print("Computing curvature gradient vector fields...")

        vertices = gt_data['vertices']
        faces = gt_data['faces']
        num_vertices = len(vertices)
        gt_H = gt_data.get('gt_mean_curvature')
        gt_K = gt_data.get('gt_gaussian_curvature')
        gt_mesh_grad_op = gt_data.get('gt_mesh_grad_op')
        gt_face_areas = gt_data.get('gt_face_areas')
        learned_G = self.current_learned_gradient_op

        grad_scale = 0.01  # Scale factor for gradient vector visualization

        def _normalize_vectors(vecs: np.ndarray) -> np.ndarray:
            """Normalize vectors to unit length (zero-safe), so all arrows have constant length."""
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return np.where(norms > 1e-12, vecs / norms, 0.0)

        # --- GT mesh gradient operator (igl.grad, face-based -> vertex-averaged) ---
        if gt_mesh_grad_op is not None and gt_face_areas is not None:
            if gt_H is not None:
                grad_H_gt = self._compute_gradient_of_scalar_field_mesh(
                    gt_H, gt_mesh_grad_op, faces, gt_face_areas, num_vertices
                )
                if grad_H_gt is not None:
                    mesh_structure.add_vector_quantity(
                        name="J1 grad(H) - GT mesh",
                        values=_normalize_vectors(grad_H_gt) * grad_scale,
                        enabled=False,
                        color=(0.0, 0.8, 1.0),  # Light blue
                        vectortype="ambient"
                    )
                    print(f"  grad(H) GT mesh: |grad| range [{np.linalg.norm(grad_H_gt, axis=1).min():.4f}, {np.linalg.norm(grad_H_gt, axis=1).max():.4f}]")

            if gt_K is not None:
                grad_K_gt = self._compute_gradient_of_scalar_field_mesh(
                    gt_K, gt_mesh_grad_op, faces, gt_face_areas, num_vertices
                )
                if grad_K_gt is not None:
                    mesh_structure.add_vector_quantity(
                        name="J2 grad(K) - GT mesh",
                        values=_normalize_vectors(grad_K_gt) * grad_scale,
                        enabled=False,
                        color=(0.0, 1.0, 0.5),  # Green-cyan
                        vectortype="ambient"
                    )
                    print(f"  grad(K) GT mesh: |grad| range [{np.linalg.norm(grad_K_gt, axis=1).min():.4f}, {np.linalg.norm(grad_K_gt, axis=1).max():.4f}]")

        # --- Learned gradient operator G (vertex-based, gradient mode only) ---
        if learned_G is not None:
            if gt_H is not None:
                grad_H_pred = self._compute_gradient_of_scalar_field_learned(
                    gt_H, learned_G, num_vertices
                )
                if grad_H_pred is not None:
                    mesh_structure.add_vector_quantity(
                        name="K1 grad(H) - PRED learned G",
                        values=_normalize_vectors(grad_H_pred) * grad_scale,
                        enabled=False,
                        color=(1.0, 0.6, 0.0),  # Orange
                        vectortype="ambient"
                    )
                    print(f"  grad(H) PRED G: |grad| range [{np.linalg.norm(grad_H_pred, axis=1).min():.4f}, {np.linalg.norm(grad_H_pred, axis=1).max():.4f}]")

            if gt_K is not None:
                grad_K_pred = self._compute_gradient_of_scalar_field_learned(
                    gt_K, learned_G, num_vertices
                )
                if grad_K_pred is not None:
                    mesh_structure.add_vector_quantity(
                        name="K2 grad(K) - PRED learned G",
                        values=_normalize_vectors(grad_K_pred) * grad_scale,
                        enabled=False,
                        color=(1.0, 0.3, 0.3),  # Red-orange
                        vectortype="ambient"
                    )
                    print(f"  grad(K) PRED G: |grad| range [{np.linalg.norm(grad_K_pred, axis=1).min():.4f}, {np.linalg.norm(grad_K_pred, axis=1).max():.4f}]")

            # --- Gradient magnitude scalar fields (for easier visual comparison) ---
            if gt_H is not None and grad_H_pred is not None:
                mesh_structure.add_scalar_quantity(
                    name="K3 |grad(H)| - PRED learned G",
                    values=np.linalg.norm(grad_H_pred, axis=1),
                    enabled=False,
                    cmap='viridis'
                )
            if gt_K is not None and grad_K_pred is not None:
                mesh_structure.add_scalar_quantity(
                    name="K4 |grad(K)| - PRED learned G",
                    values=np.linalg.norm(grad_K_pred, axis=1),
                    enabled=False,
                    cmap='viridis'
                )
        elif self.current_learned_gradient_op is None:
            operator_mode = getattr(self.current_model, '_operator_mode', 'stiffness') if self.current_model else 'stiffness'
            if operator_mode != "gradient":
                print("  [i] Curvature gradient vector fields via learned G: skipped (model is in stiffness mode)")
            else:
                print("  [!] Learned gradient operator not available")

    def _compute_eigenvector_cosine_similarities(
            self,
            gt_eigenvectors: Optional[np.ndarray],
            pred_eigenvectors: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Compute cosine similarity between GT and predicted eigenvectors.

        Handles sign ambiguity by taking absolute value of dot product.

        Args:
            gt_eigenvectors: GT eigenvectors of shape (N, k_gt)
            pred_eigenvectors: Predicted eigenvectors of shape (N, k_pred)

        Returns:
            Array of cosine similarities for each eigenvector pair, or None if not computable
        """
        if gt_eigenvectors is None or pred_eigenvectors is None:
            return None

        num_pairs = min(gt_eigenvectors.shape[1], pred_eigenvectors.shape[1])
        cosine_similarities = np.zeros(num_pairs)

        for i in range(num_pairs):
            gt_vec = gt_eigenvectors[:, i]
            pred_vec = pred_eigenvectors[:, i]

            # Normalize vectors
            gt_norm = np.linalg.norm(gt_vec)
            pred_norm = np.linalg.norm(pred_vec)

            if gt_norm > 1e-8 and pred_norm > 1e-8:
                gt_vec_normalized = gt_vec / gt_norm
                pred_vec_normalized = pred_vec / pred_norm

                # Absolute value to handle sign ambiguity
                cosine_similarities[i] = np.abs(np.dot(gt_vec_normalized, pred_vec_normalized))
            else:
                cosine_similarities[i] = 0.0

        return cosine_similarities

    def visualize_comprehensive_eigenvectors(self, mesh_structure,
                                             gt_eigenvalues: Optional[np.ndarray], gt_eigenvectors: Optional[np.ndarray],
                                             pred_eigenvalues: Optional[np.ndarray], pred_eigenvectors: Optional[np.ndarray],
                                             robust_eigenvalues: Optional[np.ndarray] = None, robust_eigenvectors: Optional[np.ndarray] = None,
                                             nelo_eigenvalues: Optional[np.ndarray] = None, nelo_eigenvectors: Optional[np.ndarray] = None):
        """Add GT (PyFM), predicted, robust-laplacian, and NeLo eigenvector scalar fields to the mesh."""
        num_to_show = self.config.num_eigenvectors_to_show

        gt_available     = gt_eigenvectors.shape[1]     if gt_eigenvectors     is not None else 0
        pred_available   = pred_eigenvectors.shape[1]   if pred_eigenvectors   is not None else 0
        robust_available = robust_eigenvectors.shape[1] if robust_eigenvectors is not None else 0
        nelo_available   = nelo_eigenvectors.shape[1]   if nelo_eigenvectors   is not None else 0

        max_available = max(gt_available, pred_available, robust_available, nelo_available)
        num_to_show = min(num_to_show, max_available)

        print(f"Adding eigenvector visualization:")
        print(f"  GT (PyFM) eigenvectors available: {gt_available}")
        print(f"  Predicted eigenvectors available: {pred_available}")
        print(f"  Robust-laplacian eigenvectors available: {robust_available}")
        print(f"  NeLo eigenvectors available: {nelo_available}")
        print(f"  Showing: {num_to_show} eigenvectors per type")

        # Compute cosine similarities between GT and predicted eigenvectors
        cosine_similarities_pred = self._compute_eigenvector_cosine_similarities(
            gt_eigenvectors, pred_eigenvectors
        )

        # Compute cosine similarities between GT and robust eigenvectors
        cosine_similarities_robust = self._compute_eigenvector_cosine_similarities(
            gt_eigenvectors, robust_eigenvectors
        )

        # Compute cosine similarities between GT and NeLo eigenvectors
        cosine_similarities_nelo = self._compute_eigenvector_cosine_similarities(
            gt_eigenvectors, nelo_eigenvectors if nelo_eigenvectors is not None else self.current_nelo_eigenvectors
        )

        # Print cosine similarities to console
        if cosine_similarities_pred is not None or cosine_similarities_robust is not None or cosine_similarities_nelo is not None:
            print(f"\n" + "-" * 99)
            print("EIGENVECTOR COSINE SIMILARITIES")
            print("-" * 99)
            print(f"{'Index':<8} {'GT vs PRED':<14} {'GT vs NeLo':<14} {'GT vs Robust':<14} {'Description'}")
            print("-" * 99)
            for i in range(num_to_show):
                if i == 0:
                    desc = "constant"
                elif i == 1:
                    desc = "Fiedler"
                else:
                    desc = ""

                pred_sim   = cosine_similarities_pred[i]   if cosine_similarities_pred   is not None and i < len(cosine_similarities_pred)   else float('nan')
                robust_sim = cosine_similarities_robust[i] if cosine_similarities_robust is not None and i < len(cosine_similarities_robust) else float('nan')
                nelo_sim   = cosine_similarities_nelo[i]   if cosine_similarities_nelo   is not None and i < len(cosine_similarities_nelo)   else float('nan')

                print(f"{i:<8} {pred_sim:<14.6f} {nelo_sim:<14.6f} {robust_sim:<14.6f} {desc}")

            # Summary statistics
            print("-" * 99)
            if cosine_similarities_pred is not None:
                print(f"GT vs PRED   - Mean: {cosine_similarities_pred[:num_to_show].mean():.6f}, Min: {cosine_similarities_pred[:num_to_show].min():.6f}, Max: {cosine_similarities_pred[:num_to_show].max():.6f}")
            if cosine_similarities_nelo is not None:
                print(f"GT vs NeLo   - Mean: {cosine_similarities_nelo[:num_to_show].mean():.6f}, Min: {cosine_similarities_nelo[:num_to_show].min():.6f}, Max: {cosine_similarities_nelo[:num_to_show].max():.6f}")
            if cosine_similarities_robust is not None:
                print(f"GT vs Robust - Mean: {cosine_similarities_robust[:num_to_show].mean():.6f}, Min: {cosine_similarities_robust[:num_to_show].min():.6f}, Max: {cosine_similarities_robust[:num_to_show].max():.6f}")
            print("-" * 99 + "\n")

            # Store average cosine similarity for UI display (GT vs PRED)
            if cosine_similarities_pred is not None:
                self.current_avg_cosine_similarity = float(cosine_similarities_pred[:num_to_show].mean())

            # Store per-eigenvector cosine similarities for UI comparison table
            self.current_cosine_similarities_pred   = cosine_similarities_pred
            self.current_cosine_similarities_robust = cosine_similarities_robust
            self.current_cosine_similarities_nelo   = cosine_similarities_nelo

        # Add eigenvectors in groups
        for i in range(num_to_show):
            # Get cosine similarities for this index
            cos_sim_pred   = cosine_similarities_pred[i]   if cosine_similarities_pred   is not None and i < len(cosine_similarities_pred)   else None
            cos_sim_robust = cosine_similarities_robust[i] if cosine_similarities_robust is not None and i < len(cosine_similarities_robust) else None
            cos_sim_nelo   = cosine_similarities_nelo[i]   if cosine_similarities_nelo   is not None and i < len(cosine_similarities_nelo)   else None

            # Add GT (PyFM) eigenvector
            if gt_eigenvectors is not None and i < gt_available:
                gt_eigenvector = gt_eigenvectors[:, i]
                gt_eigenvalue = gt_eigenvalues[i] if gt_eigenvalues is not None else 0.0

                if i == 0:
                    gt_name = f"Eigenvector {i:02d}a GT-PyFM (lambda={gt_eigenvalue:.2e}, constant)"
                elif i == 1:
                    gt_name = f"Eigenvector {i:02d}a GT-PyFM (lambda={gt_eigenvalue:.6f}, Fiedler)"
                else:
                    gt_name = f"Eigenvector {i:02d}a GT-PyFM (lambda={gt_eigenvalue:.6f})"

                mesh_structure.add_scalar_quantity(
                    name=gt_name,
                    values=gt_eigenvector,
                    enabled=(i == 1),  # Enable GT Fiedler vector by default
                    cmap=self.config.colormap
                )

            # Add predicted eigenvector
            if pred_eigenvectors is not None and i < pred_available:
                pred_eigenvector = pred_eigenvectors[:, i]
                pred_eigenvalue = pred_eigenvalues[i] if pred_eigenvalues is not None else 0.0
                cos_str = f", cos={cos_sim_pred:.4f}" if cos_sim_pred is not None else ""

                if i == 0:
                    pred_name = f"Eigenvector {i:02d}b PRED (lambda={pred_eigenvalue:.2e}, constant{cos_str})"
                elif i == 1:
                    pred_name = f"Eigenvector {i:02d}b PRED (lambda={pred_eigenvalue:.6f}, Fiedler{cos_str})"
                else:
                    pred_name = f"Eigenvector {i:02d}b PRED (lambda={pred_eigenvalue:.6f}{cos_str})"

                mesh_structure.add_scalar_quantity(
                    name=pred_name,
                    values=pred_eigenvector,
                    enabled=False,
                    cmap=self.config.colormap
                )

            # Add robust-laplacian eigenvector
            if robust_eigenvectors is not None and i < robust_available:
                robust_eigenvector = robust_eigenvectors[:, i]
                robust_eigenvalue = robust_eigenvalues[i] if robust_eigenvalues is not None else 0.0
                cos_str = f", cos={cos_sim_robust:.4f}" if cos_sim_robust is not None else ""

                if i == 0:
                    robust_name = f"Eigenvector {i:02d}c Robust (lambda={robust_eigenvalue:.2e}, constant{cos_str})"
                elif i == 1:
                    robust_name = f"Eigenvector {i:02d}c Robust (lambda={robust_eigenvalue:.6f}, Fiedler{cos_str})"
                else:
                    robust_name = f"Eigenvector {i:02d}c Robust (lambda={robust_eigenvalue:.6f}{cos_str})"

                mesh_structure.add_scalar_quantity(
                    name=robust_name,
                    values=robust_eigenvector,
                    enabled=False,
                    cmap=self.config.colormap
                )

            # Add NeLo eigenvector
            if nelo_eigenvectors is not None and i < nelo_available:
                nelo_eigenvector = nelo_eigenvectors[:, i]
                nelo_eigenvalue  = nelo_eigenvalues[i] if nelo_eigenvalues is not None else 0.0
                cos_str = f", cos={cos_sim_nelo:.4f}" if cos_sim_nelo is not None else ""

                if i == 0:
                    nelo_name = f"Eigenvector {i:02d}d NeLo (lambda={nelo_eigenvalue:.2e}, constant{cos_str})"
                elif i == 1:
                    nelo_name = f"Eigenvector {i:02d}d NeLo (lambda={nelo_eigenvalue:.6f}, Fiedler{cos_str})"
                else:
                    nelo_name = f"Eigenvector {i:02d}d NeLo (lambda={nelo_eigenvalue:.6f}{cos_str})"

                mesh_structure.add_scalar_quantity(
                    name=nelo_name,
                    values=nelo_eigenvector,
                    enabled=False,
                    cmap=self.config.colormap
                )

    # =========================================================================
    # GREEN'S FUNCTION MAXIMUM PRINCIPLE VALIDATION
    # =========================================================================

    def compute_greens_function(
            self,
            laplacian_matrix: scipy.sparse.csr_matrix,
            mass_matrix: Optional[scipy.sparse.csr_matrix],
            source_vertex_idx: int,
            regularization: float = 1e-6,
            device: Optional[torch.device] = None
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Compute the harmonic Green's function by solving (L + ÃƒÅ½Ã‚ÂµM)g = ÃƒÅ½Ã‚Â´.

        The Green's function g satisfies Lg = ÃƒÅ½Ã‚Â´_source, where ÃƒÅ½Ã‚Â´ is a delta function
        at the source vertex. Since L is singular (constant null space), we use
        regularization: (L + ÃƒÅ½Ã‚ÂµM)g = ÃƒÅ½Ã‚Â´.

        After solving:
        1. Compute Laplacian residual on RAW solution (before normalization)
        2. Remove constant component (subtract weighted mean)
        3. Shift so min = 0
        4. Scale so max = 1

        The normalization makes visualization easier and doesn't affect:
        - argmax (maximum principle test)
        - correlation (monotonicity and GT comparison)
        - relative ordering (pairwise monotonicity)

        Args:
            laplacian_matrix: Sparse Laplacian matrix L (n x n), should be positive semi-definite
            mass_matrix: Sparse diagonal mass matrix M (n x n), or None for identity
            source_vertex_idx: Index of the source vertex
            regularization: Regularization parameter ÃƒÅ½Ã‚Âµ (default 1e-6)

        Returns:
            Tuple of (g, residual_norm) where:
            - g: Normalized Green's function values in [0, 1] range
            - residual_norm: ||Lg_raw - ÃƒÅ½Ã‚Â´|| / ||ÃƒÅ½Ã‚Â´|| computed before normalization
            Returns None on failure.
        """
        n = laplacian_matrix.shape[0]

        if source_vertex_idx < 0 or source_vertex_idx >= n:
            print(f"  [!] Invalid source vertex index: {source_vertex_idx}")
            return None

        try:
            # Convert to float64 for numerical stability
            L = laplacian_matrix.astype(np.float64)

            # Create delta function at source
            delta = np.zeros(n, dtype=np.float64)
            delta[source_vertex_idx] = 1.0

            # Use identity mass matrix if none provided
            if mass_matrix is None:
                M = scipy.sparse.eye(n, format='csr', dtype=np.float64)
            else:
                M = mass_matrix.astype(np.float64)

            # Ensure L is CSR format
            if not isinstance(L, scipy.sparse.csr_matrix):
                L = L.tocsr()

            # Estimate the scale of the Laplacian for adaptive regularization
            diag = np.array(L.diagonal()).flatten()
            L_scale = np.abs(diag).mean() if len(diag) > 0 else 1.0
            adaptive_reg = regularization * max(L_scale, 1e-4)

            print(f"    Using regularization: {adaptive_reg:.2e} (scale={L_scale:.2e})")

            # Ensure PSD convention before solving
            L = ensure_psd(L)
            A = L + adaptive_reg * M

            # Solve the linear system (L + reg*M is SPD)
            g_raw = sparse_solve(A, delta, device, spd=True)

            if not np.isfinite(g_raw).all():
                print(f"  [!] Solution contains NaN or Inf values")
                return None

            # =========================================================
            # Diagnostic: Raw solution statistics
            # =========================================================
            print(f"    Raw solution: min={g_raw.min():.4e}, max={g_raw.max():.4e}, "
                  f"range={g_raw.max() - g_raw.min():.4e}")
            print(f"    Raw value at source: {g_raw[source_vertex_idx]:.4e}")

            # Check if raw solution has the max at source (before any normalization)
            raw_max_idx = np.argmax(g_raw)
            if raw_max_idx != source_vertex_idx:
                print(f"    [!] WARNING: Raw max NOT at source! Max at vertex {raw_max_idx} "
                      f"(value={g_raw[raw_max_idx]:.4e} vs source={g_raw[source_vertex_idx]:.4e})")

            # =========================================================
            # Compute Laplacian residual BEFORE normalization
            # This measures how well Lg ÃƒÂ¢Ã¢â‚¬Â°Ã‹â€  ÃƒÅ½Ã‚Â´
            # =========================================================
            Lg = L @ g_raw
            residual = Lg - delta
            residual_norm = float(np.linalg.norm(residual) / np.linalg.norm(delta))

            # =========================================================
            # Now normalize for display (doesn't affect key metrics)
            # =========================================================

            # Step 1: Remove constant component (subtract weighted mean)
            M_diag = np.array(M.diagonal()).flatten()
            total_mass = M_diag.sum()
            if total_mass > 0:
                weighted_mean = (g_raw * M_diag).sum() / total_mass
                g = g_raw - weighted_mean
            else:
                g = g_raw - g_raw.mean()

            print(f"    After mean removal: min={g.min():.4e}, max={g.max():.4e}, "
                  f"range={g.max() - g.min():.4e}")

            # Step 2: Shift so min = 0
            g = g - g.min()

            # Step 3: Scale so max = 1 (if max > 0)
            if g.max() > 0:
                g = g / g.max()

            return g, residual_norm

        except Exception as e:
            print(f"  [!] Failed to compute Green's function: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compute_greens_function_batch(
            self,
            laplacian_matrix: scipy.sparse.csr_matrix,
            mass_matrix: Optional[scipy.sparse.csr_matrix],
            source_indices: List[int],
            regularization: float = 1e-6,
    ) -> Dict[int, Optional[Tuple[np.ndarray, float]]]:
        """
        Compute Green's function for multiple sources with one factorization.

        Factorizes (L + εM) ONCE, then solves for each source.

        Args:
            laplacian_matrix: Sparse Laplacian matrix L (n x n)
            mass_matrix: Sparse diagonal mass matrix M (n x n), or None for identity
            source_indices: List of source vertex indices
            regularization: Regularization parameter ε (default 1e-6)

        Returns:
            Dict mapping source_idx -> (normalized_greens, residual_norm), or None for failed sources.
        """
        from neural_local_laplacian.utils.geodesic_utils import sparse_factorize

        n = laplacian_matrix.shape[0]
        L = ensure_psd(laplacian_matrix.astype(np.float64))

        if mass_matrix is None:
            M = scipy.sparse.eye(n, format='csr', dtype=np.float64)
        else:
            M = mass_matrix.astype(np.float64)

        # Adaptive regularization based on Laplacian scale
        diag = np.array(L.diagonal()).flatten()
        L_scale = np.abs(diag).mean() if len(diag) > 0 else 1.0
        adaptive_reg = regularization * max(L_scale, 1e-4)

        A = L + adaptive_reg * M

        # Factorize ONCE
        try:
            _t_fact = time.perf_counter()
            factor = sparse_factorize(A)
            _t_fact_done = time.perf_counter()
            print(f"    Factorize: {(_t_fact_done - _t_fact)*1000:.1f} ms (backend: {factor._backend})")
        except Exception as e:
            print(f"  [!] Green's function batch factorization failed: {e}")
            return {src: None for src in source_indices}

        M_diag = np.array(M.diagonal()).flatten()
        total_mass = M_diag.sum()

        results = {}
        _t_solve_start = time.perf_counter()
        for source_idx in source_indices:
            try:
                delta = np.zeros(n, dtype=np.float64)
                delta[source_idx] = 1.0
                g_raw = factor.solve(delta)

                if not np.isfinite(g_raw).all():
                    results[source_idx] = None
                    continue

                # Residual before normalization
                Lg = L @ g_raw
                residual_norm = float(np.linalg.norm(Lg - delta) / np.linalg.norm(delta))

                # Normalize: remove mean, shift min=0, scale max=1
                if total_mass > 0:
                    g = g_raw - (g_raw * M_diag).sum() / total_mass
                else:
                    g = g_raw - g_raw.mean()
                g = g - g.min()
                if g.max() > 0:
                    g = g / g.max()

                results[source_idx] = (g, residual_norm)
            except Exception:
                results[source_idx] = None

        return results

    def validate_maximum_principle(
            self,
            greens_function: np.ndarray,
            source_vertex_idx: int,
            method_name: str,
            vertices: np.ndarray,
            faces: np.ndarray,
            laplacian_residual_norm: float = 0.0,
            gt_greens_function: np.ndarray = None
    ) -> GreensFunctionValidationResult:
        """
        Validate whether the Green's function satisfies the discrete maximum principle
        and compute additional quality metrics.

        The Green's function is normalized to [0, 1] range (min=0, max=1).
        This normalization does NOT affect:
        - argmax location (maximum principle test)
        - Pearson correlation (monotonicity and GT comparison)
        - Relative ordering (pairwise comparisons)

        Tests performed:
        1. PRIMARY: Is the maximum at the source vertex? (argmax invariant to normalization)
        2. SECONDARY: Does value decrease with GEODESIC distance? (uses compute_exact_geodesics with safety check)
        3. TERTIARY: Laplacian residual (computed before normalization, passed in)
        4. COMPARISON: Correlation with GT (invariant to normalization)

        Args:
            greens_function: Normalized Green's function values in [0, 1]
            source_vertex_idx: Index of the source vertex
            method_name: Name of the method ("GT", "PRED", or "Robust")
            vertices: Mesh vertices (n, 3) for computing distances
            faces: Mesh faces (m, 3) for geodesic distance computation
            laplacian_residual_norm: Pre-computed ||Lg - ÃƒÅ½Ã‚Â´|| / ||ÃƒÅ½Ã‚Â´|| (before normalization)
            gt_greens_function: Optional GT Green's function for comparison (also normalized)

        Returns:
            GreensFunctionValidationResult with all validation metrics
        """
        n = len(greens_function)
        g = greens_function

        # =====================================================================
        # Basic statistics (on normalized values)
        # =====================================================================
        min_val = float(g.min())  # Should be 0
        max_val = float(g.max())  # Should be 1
        mean_val = float(g.mean())
        value_at_source = float(g[source_vertex_idx])

        # =====================================================================
        # PRIMARY TEST: Maximum at source
        # argmax is INVARIANT to shifting and scaling!
        # =====================================================================
        max_idx = int(np.argmax(g))
        max_at_source = (max_idx == source_vertex_idx)

        # For normalized g, "violations" are vertices with g > value_at_source
        # If max is at source, value_at_source = 1.0 and there are no violations
        # If max is elsewhere, value_at_source < 1.0 and violations exist
        violation_mask = g > value_at_source + 1e-10
        num_violations = int(violation_mask.sum())

        if not max_at_source:
            worst_violation_vertex = max_idx
            worst_violation_value = float(g[max_idx])
        else:
            worst_violation_vertex = -1
            worst_violation_value = 0.0

        satisfies_principle = max_at_source

        # =====================================================================
        # SOURCE-CENTERED CHECK (Sharp & Crane style)
        # Shift so source = 0, then check all values are <= 0
        # This directly tests: "Is any vertex hotter than the source?"
        # =====================================================================
        g_source_centered = g - value_at_source  # Now g[source] = 0

        # All values should be <= 0 (source is the "hottest" point)
        positive_mask = g_source_centered > 1e-10  # Tolerance for numerical precision
        num_positive_vertices = int(positive_mask.sum())

        if num_positive_vertices > 0:
            # Find the worst violation
            positive_vertex_idx = int(np.argmax(g_source_centered))
            max_positive_value = float(g_source_centered[positive_vertex_idx])
        else:
            positive_vertex_idx = -1
            max_positive_value = float(g_source_centered.max())  # Should be ~0 or negative

        # =====================================================================
        # SECONDARY TEST: Monotonic decay with GEODESIC distance
        # The Green's function should decrease with geodesic distance from source,
        # NOT Euclidean distance (heat flows along the surface, not through air)
        # =====================================================================
        distance_correlation = 0.0
        monotonicity_score = 1.0
        distances = None  # Will store geodesic distances for later use

        # Try to compute geodesic distances using the shared protected function
        # (includes topology safety check to avoid segfaults on degenerate meshes)
        distances = compute_exact_geodesics(vertices, faces, source_vertex_idx)

        # Fallback to Euclidean distances if geodesic computation failed
        if distances is None or len(distances) != n:
            print(f"    [!] Exact geodesics unavailable, using Euclidean distances for monotonicity test")
            distances = np.linalg.norm(vertices - vertices[source_vertex_idx], axis=1)

        # Now compute correlation and monotonicity using the distances
        if distances is not None and len(distances) == n:
            # Exclude source vertex from correlation (distance = 0)
            mask = np.ones(n, dtype=bool)
            mask[source_vertex_idx] = False

            if mask.sum() > 1:
                g_masked = g[mask]
                dist_masked = distances[mask]

                # Filter out any infinite distances (disconnected vertices)
                finite_mask = np.isfinite(dist_masked)
                if finite_mask.sum() > 1:
                    g_finite = g_masked[finite_mask]
                    dist_finite = dist_masked[finite_mask]

                    # Correlation between g and -distance (should be positive: closer = higher g)
                    corr_matrix = np.corrcoef(g_finite, -dist_finite)
                    distance_correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

                    # Pairwise monotonicity: sample random pairs
                    num_finite = len(g_finite)
                    num_samples = min(10000, num_finite * (num_finite - 1) // 2)
                    if num_samples > 100:
                        # Get indices into the finite arrays
                        idx1 = np.random.randint(0, num_finite, num_samples)
                        idx2 = np.random.randint(0, num_finite, num_samples)
                        valid = idx1 != idx2
                        idx1, idx2 = idx1[valid], idx2[valid]

                        if len(idx1) > 0:
                            d1, d2 = dist_finite[idx1], dist_finite[idx2]
                            g1, g2 = g_finite[idx1], g_finite[idx2]

                            # If d1 < d2 (closer), then g1 should be > g2 (higher)
                            closer_is_higher = ((d1 < d2) & (g1 > g2)) | ((d2 < d1) & (g2 > g1))
                            not_tie = (np.abs(d1 - d2) > 1e-10) & (np.abs(g1 - g2) > 1e-10)

                            if not_tie.sum() > 0:
                                monotonicity_score = float(closer_is_higher[not_tie].sum() / not_tie.sum())

        # =====================================================================
        # TERTIARY TEST: Laplacian residual (passed in, computed before normalization)
        # =====================================================================
        # Already computed before normalization and passed as argument

        # =====================================================================
        # COMPARISON: Correlation with GT
        # Pearson correlation is INVARIANT to linear transforms!
        # =====================================================================
        correlation_with_gt = 0.0
        if gt_greens_function is not None and len(gt_greens_function) == n:
            corr_matrix = np.corrcoef(g, gt_greens_function)
            correlation_with_gt = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        # =====================================================================
        # DIAGNOSTIC: Min location and geodesic distance to it
        # The minimum should be at the geodesically farthest point from source
        # =====================================================================
        min_vertex_idx = int(np.argmin(g))
        geodesic_dist_to_min = 0.0
        max_geodesic_dist = 0.0

        # Use the distances array computed earlier (geodesic if available)
        if distances is not None and len(distances) == n:
            finite_distances = distances[np.isfinite(distances)]
            if len(finite_distances) > 0:
                max_geodesic_dist = float(np.max(finite_distances))
            if np.isfinite(distances[min_vertex_idx]):
                geodesic_dist_to_min = float(distances[min_vertex_idx])

        return GreensFunctionValidationResult(
            method_name=method_name,
            source_vertex_idx=source_vertex_idx,
            num_vertices=n,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            value_at_source=value_at_source,
            max_at_source=max_at_source,
            satisfies_maximum_principle=satisfies_principle,
            num_violations=num_violations,
            worst_violation_vertex=worst_violation_vertex,
            worst_violation_value=worst_violation_value,
            num_positive_vertices=num_positive_vertices,
            max_positive_value=max_positive_value,
            positive_vertex_idx=positive_vertex_idx,
            distance_correlation=distance_correlation,
            monotonicity_score=monotonicity_score,
            laplacian_residual_norm=laplacian_residual_norm,
            correlation_with_gt=correlation_with_gt,
            min_vertex_idx=min_vertex_idx,
            geodesic_dist_to_min=geodesic_dist_to_min,
            max_geodesic_dist=max_geodesic_dist
        )

    def select_source_vertex(
            self,
            vertices: np.ndarray,
            method: str = "centroid"
    ) -> int:
        """
        Select a source vertex for Green's function computation.

        Args:
            vertices: Mesh vertices (n, 3)
            method: Selection method - "centroid" (closest to center),
                    "random", or an integer index

        Returns:
            Index of selected source vertex
        """
        n = len(vertices)

        if method == "centroid":
            # Find vertex closest to centroid
            centroid = vertices.mean(axis=0)
            distances = np.linalg.norm(vertices - centroid, axis=1)
            return int(np.argmin(distances))
        elif method == "random":
            return int(np.random.randint(0, n))
        elif isinstance(method, int):
            return max(0, min(method, n - 1))
        else:
            # Default to centroid
            centroid = vertices.mean(axis=0)
            distances = np.linalg.norm(vertices - centroid, axis=1)
            return int(np.argmin(distances))

    def compute_and_visualize_greens_functions(
            self,
            mesh_structure,
            gt_data: Dict[str, Any],
            inference_result: Dict[str, Any],
            source_vertex_idx: Optional[int] = None
    ) -> Dict[str, GreensFunctionValidationResult]:
        """
        Compute and visualize Green's functions for GT, PRED, and Robust Laplacians.

        This validates the discrete maximum principle for each Laplacian:
        - Solves Lg = ÃƒÅ½Ã‚Â´_source for each method
        - Checks that g >= 0 everywhere (no negative values)
        - Checks that max(g) is at the source vertex

        Args:
            mesh_structure: Polyscope mesh structure for visualization
            gt_data: Dictionary containing GT Laplacian data
            inference_result: Dictionary containing PRED Laplacian data
            source_vertex_idx: Optional specific source vertex (default: centroid)

        Returns:
            Dictionary mapping method names to validation results
        """
        print(f"\n" + "=" * 70)
        print("GREEN'S FUNCTION MAXIMUM PRINCIPLE VALIDATION")
        print("=" * 70)

        vertices = gt_data['vertices']
        faces = gt_data['faces']
        n = len(vertices)

        # =====================================================================
        # Mesh diagnostics: check for potential issues
        # =====================================================================
        print("\nMesh diagnostics:")
        print(f"  Vertices: {n}, Faces: {len(faces)}")

        # Check for duplicate vertices (vertices at the same position)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(vertices)
            pairs = tree.query_pairs(r=1e-10)  # Find vertices within tiny distance
            if len(pairs) > 0:
                print(f"  [!] WARNING: {len(pairs)} pairs of duplicate/near-duplicate vertices detected!")
                print(f"      This can cause the mesh to appear closed but have topological holes.")
            else:
                print(f"  No duplicate vertices detected")
        except Exception as e:
            print(f"  [!] Could not check for duplicates: {e}")

        # Check for non-manifold edges (edges shared by != 2 faces)
        try:
            edge_face_count = {}
            for fi, f in enumerate(faces):
                for i in range(3):
                    e = tuple(sorted([f[i], f[(i + 1) % 3]]))
                    edge_face_count[e] = edge_face_count.get(e, 0) + 1

            non_manifold_edges = [(e, c) for e, c in edge_face_count.items() if c > 2]
            if len(non_manifold_edges) > 0:
                print(f"  [!] WARNING: {len(non_manifold_edges)} non-manifold edges (shared by >2 faces)!")

            # Edges with count 1 are boundary, count 2 are manifold interior
            boundary_count = sum(1 for c in edge_face_count.values() if c == 1)
            interior_count = sum(1 for c in edge_face_count.values() if c == 2)
            print(f"  Edge counts: {boundary_count} boundary, {interior_count} interior, {len(non_manifold_edges)} non-manifold")
        except Exception as e:
            print(f"  [!] Could not check edge manifoldness: {e}")

        # Check for connected components using scipy
        try:
            from scipy.sparse.csgraph import connected_components

            # Build adjacency matrix from faces
            rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                                   faces[:, 1], faces[:, 2], faces[:, 0]])
            cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                                   faces[:, 0], faces[:, 1], faces[:, 2]])
            data = np.ones(len(rows))
            adj = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

            n_components, labels = connected_components(adj, directed=False)
            print(f"  Connected components: {n_components}")

            if n_components > 1:
                component_sizes = [(labels == i).sum() for i in range(n_components)]
                print(f"  Component sizes: {sorted(component_sizes, reverse=True)}")
                print(f"  [!] WARNING: Mesh has multiple components - Green's function may be unreliable!")
        except Exception as e:
            print(f"  [!] Could not check connectivity: {e}")

        # Check for boundary vertices
        try:
            # An edge is boundary if it appears in only one face
            edges = {}
            for f in faces:
                for i in range(3):
                    e = tuple(sorted([f[i], f[(i + 1) % 3]]))
                    edges[e] = edges.get(e, 0) + 1

            boundary_edges = [e for e, count in edges.items() if count == 1]
            boundary_vertices = set()
            for e in boundary_edges:
                boundary_vertices.add(e[0])
                boundary_vertices.add(e[1])

            print(f"  Boundary edges: {len(boundary_edges)}")
            print(f"  Boundary vertices: {len(boundary_vertices)}")

            if len(boundary_edges) > 0:
                print(f"  [!] Mesh has boundaries (not closed)")

                # Count number of boundary loops
                # Build adjacency for boundary vertices only
                from collections import defaultdict
                boundary_adj = defaultdict(set)
                for e in boundary_edges:
                    boundary_adj[e[0]].add(e[1])
                    boundary_adj[e[1]].add(e[0])

                # Count connected components in boundary graph
                visited = set()
                num_loops = 0
                for start in boundary_vertices:
                    if start not in visited:
                        # BFS to find this loop
                        queue = [start]
                        while queue:
                            v = queue.pop()
                            if v in visited:
                                continue
                            visited.add(v)
                            for neighbor in boundary_adj[v]:
                                if neighbor not in visited:
                                    queue.append(neighbor)
                        num_loops += 1

                print(f"  Number of boundary loops (holes): {num_loops}")
                if num_loops > 1:
                    print(f"  [!] WARNING: Multiple boundary loops can cause patchy Green's function!")
            else:
                print(f"  Mesh is closed (no boundaries)")

            # Store boundary info for CSV export
            self._mesh_boundary_info = {
                'num_boundary_edges': len(boundary_edges),
                'num_boundary_vertices': len(boundary_vertices),
                'is_closed': len(boundary_edges) == 0,
                'num_boundary_loops': num_loops if len(boundary_edges) > 0 else 0,
            }
        except Exception as e:
            print(f"  [!] Could not check boundaries: {e}")

        # Store connected components count
        try:
            self._mesh_boundary_info['num_components'] = n_components
        except NameError:
            pass

        print("-" * 70)

        # Select source vertex
        if source_vertex_idx is None:
            source_vertex_idx = self.select_source_vertex(vertices, method="centroid")

        # Cache source vertex (backward compat — multi-source state is in _source_indices)
        # This source_vertex_idx is used within this function only

        source_pos = vertices[source_vertex_idx]
        print(f"Source vertex: {source_vertex_idx} at position ({source_pos[0]:.4f}, {source_pos[1]:.4f}, {source_pos[2]:.4f})")
        print(f"Number of vertices: {n}")
        print("-" * 70)

        results = {}
        greens_functions = {}

        # =====================================================================
        # 1. GT Green's Function (using PyFM/igl cotangent Laplacian)
        # =====================================================================
        print("\nComputing GT Green's function...")
        gt_greens = None
        gt_residual = 0.0
        L_gt = None

        if HAS_IGL:
            try:
                # Build GT Laplacian using igl
                V = vertices.astype(np.float64)
                F = faces.astype(np.int32)

                L_gt = igl.cotmatrix(V, F)
                M_gt = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)

                # Note: igl.cotmatrix returns negative semi-definite L (opposite sign convention)
                # We need to negate it to get positive semi-definite
                L_gt = -L_gt

                # Diagnostic: check Laplacian properties
                diag_gt = np.array(L_gt.diagonal()).flatten()
                print(f"  GT Laplacian: diag range [{diag_gt.min():.4f}, {diag_gt.max():.4f}], row sums ~ {np.abs(L_gt.sum(axis=1)).max():.2e}")

                _t_greens_start = time.perf_counter()
                result = self.compute_greens_function(L_gt, M_gt, source_vertex_idx)

                if result is not None:
                    gt_greens, gt_residual = result
                    results['GT'] = self.validate_maximum_principle(
                        gt_greens, source_vertex_idx, "GT (igl)",
                        vertices=vertices,
                        faces=faces,
                        laplacian_residual_norm=gt_residual,
                        gt_greens_function=None  # This IS the GT
                    )
                    greens_functions['GT'] = gt_greens
                    # Cache GT Green's function for per-method recomputation (multi-source dict)
                    if self._greens_gt_values is None:
                        self._greens_gt_values = {}
                    self._greens_gt_values[source_vertex_idx] = gt_greens
                    print(f"  GT Green's function: min={gt_greens.min():.6f}, max={gt_greens.max():.6f}")
                self.timing_results.gt_greens_solve_time = time.perf_counter() - _t_greens_start

            except Exception as e:
                print(f"  [!] Failed to compute GT Green's function: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  [!] igl not available, skipping GT Green's function")

        # =====================================================================
        # 2. PRED Green's Function (using neural network predicted Laplacian)
        # =====================================================================
        print("\nComputing PRED Green's function...")
        pred_greens = None
        pred_residual = 0.0
        L_pred = None

        if inference_result.get('stiffness_matrix') is not None:
            try:
                L_pred = inference_result['stiffness_matrix']
                M_pred = inference_result.get('mass_matrix')

                # Diagnostic: check Laplacian properties
                diag_pred = np.array(L_pred.diagonal()).flatten()
                print(f"  PRED Laplacian: diag range [{diag_pred.min():.4f}, {diag_pred.max():.4f}], row sums ~ {np.abs(L_pred.sum(axis=1)).max():.2e}")

                # Check if PRED Laplacian has correct sign (positive diagonal for positive semi-definite)
                if diag_pred.min() < 0:
                    print(f"  [!] WARNING: PRED Laplacian has negative diagonal entries - may indicate sign issue")

                _t_greens_start = time.perf_counter()
                result = self.compute_greens_function(L_pred, M_pred, source_vertex_idx,
                                                      device=self.current_device)

                if result is not None:
                    pred_greens, pred_residual = result
                    results['PRED'] = self.validate_maximum_principle(
                        pred_greens, source_vertex_idx, "PRED (Neural)",
                        vertices=vertices,
                        faces=faces,
                        laplacian_residual_norm=pred_residual,
                        gt_greens_function=gt_greens  # Compare to GT
                    )
                    greens_functions['PRED'] = pred_greens
                    print(f"  PRED Green's function: min={pred_greens.min():.6f}, max={pred_greens.max():.6f}")
                self.timing_results.pred_greens_solve_time = time.perf_counter() - _t_greens_start

            except Exception as e:
                print(f"  [!] Failed to compute PRED Green's function: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  [!] PRED stiffness matrix not available")

        # =====================================================================
        # 3. Robust Green's Function (using robust_laplacian point cloud)
        # =====================================================================
        print("\nComputing Robust Green's function...")
        robust_greens = None
        robust_residual = 0.0
        L_robust = None

        if not self._skip_robust:
            try:
                # Use the current robust k from settings
                k = self.reconstruction_settings.current_robust_k

                L_robust, M_robust = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=k)

                # Diagnostic: check Laplacian properties
                diag_robust = np.array(L_robust.diagonal()).flatten()
                print(f"  Robust Laplacian: diag range [{diag_robust.min():.4f}, {diag_robust.max():.4f}], row sums ~ {np.abs(L_robust.sum(axis=1)).max():.2e}")

                _t_greens_start = time.perf_counter()
                result = self.compute_greens_function(L_robust, M_robust, source_vertex_idx)

                if result is not None:
                    robust_greens, robust_residual = result
                    results['Robust'] = self.validate_maximum_principle(
                        robust_greens, source_vertex_idx, "Robust",
                        vertices=vertices,
                        faces=faces,
                        laplacian_residual_norm=robust_residual,
                        gt_greens_function=gt_greens  # Compare to GT
                    )
                    greens_functions['Robust'] = robust_greens
                    print(f"  Robust Green's function: min={robust_greens.min():.6f}, max={robust_greens.max():.6f}")
                self.timing_results.robust_greens_solve_time = time.perf_counter() - _t_greens_start

            except Exception as e:
                print(f"  [!] Failed to compute Robust Green's function: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  [!] Robust computation skipped (skip_robust=True)")

        # =====================================================================
        # 4. NeLo Green's Function
        # =====================================================================
        if self.current_nelo_L is not None:
            print("\nComputing NeLo Green's function...")
            try:
                _t_greens_start = time.perf_counter()
                result = self.compute_greens_function(self.current_nelo_L, self.current_nelo_M, source_vertex_idx,
                                                      device=self.current_device)
                if result is not None:
                    nelo_greens, nelo_residual = result
                    results['NeLo'] = self.validate_maximum_principle(
                        nelo_greens, source_vertex_idx, "NeLo",
                        vertices=vertices,
                        faces=faces,
                        laplacian_residual_norm=nelo_residual,
                        gt_greens_function=gt_greens
                    )
                    greens_functions['NeLo'] = nelo_greens
                    print(f"  NeLo Green's function: min={nelo_greens.min():.6f}, max={nelo_greens.max():.6f}")
                self.timing_results.nelo_greens_solve_time = time.perf_counter() - _t_greens_start
            except Exception as e:
                print(f"  [!] Failed to compute NeLo Green's function: {e}")
                import traceback
                traceback.print_exc()

        # =====================================================================
        # Cache raw Green's function values for slider refresh
        if source_vertex_idx is not None and greens_functions:
            if source_vertex_idx not in self._all_greens_values:
                self._all_greens_values[source_vertex_idx] = {}
            for mk, gv in greens_functions.items():
                self._all_greens_values[source_vertex_idx][mk] = gv

        # Print comparison table
        # =====================================================================
        print(f"\n" + "=" * 90)
        print("GREEN'S FUNCTION VALIDATION RESULTS")
        print(f"Source vertex: {source_vertex_idx}")
        print("=" * 90)

        # Primary test: Maximum principle
        print("\n[1] MAXIMUM PRINCIPLE TEST (Primary)")
        print(f"{'Method':<14} {'Val@Source':>10} {'Max Val':>10} {'Max@Source':^12} {'#Viol':>6} {'Status':>10}")
        print("-" * 74)
        for method_name, result in results.items():
            print(result)

        # Source-centered check (Sharp & Crane style)
        print("\n[1b] SOURCE-CENTERED CHECK (Sharp & Crane style: g - g_source <= 0?)")
        print(f"{'Method':<14} {'#Positive':>10} {'Max(g-g_src)':>14} {'Worst Vtx':>10} {'Status':>10}")
        print("-" * 62)
        for method_name, result in results.items():
            if result.num_positive_vertices == 0:
                status = "[OK] PASS"
            else:
                status = "[X] FAIL"
            print(f"{result.method_name:<14} {result.num_positive_vertices:>10} {result.max_positive_value:>14.6f} "
                  f"{result.positive_vertex_idx:>10} {status:>10}")

        # Secondary test: Monotonicity (using geodesic distances)
        print("\n[2] MONOTONICITY TEST (Does value decrease with GEODESIC distance from source?)")
        print(f"{'Method':<14} {'Geo Corr':>12} {'Mono Score':>12} {'Quality':>12}")
        print("-" * 54)
        for method_name, result in results.items():
            # Distance correlation should be positive (g correlates with -distance)
            corr = result.distance_correlation
            mono = result.monotonicity_score

            if corr > 0.8 and mono > 0.8:
                quality = "Excellent"
            elif corr > 0.5 and mono > 0.6:
                quality = "Good"
            elif corr > 0.2 and mono > 0.5:
                quality = "Fair"
            else:
                quality = "Poor"

            print(f"{result.method_name:<14} {corr:>12.4f} {mono:>12.4f} {quality:>12}")

        # Tertiary test: Laplacian residual
        print("\n[3] SMOOTHNESS TEST (Laplacian residual ||Lg - ÃƒÅ½Ã‚Â´|| / ||ÃƒÅ½Ã‚Â´||)")
        print(f"{'Method':<14} {'Residual':>12} {'Quality':>12}")
        print("-" * 40)
        for method_name, result in results.items():
            res = result.laplacian_residual_norm
            if res < 0.01:
                quality = "Excellent"
            elif res < 0.1:
                quality = "Good"
            elif res < 1.0:
                quality = "Fair"
            else:
                quality = "Poor"
            print(f"{result.method_name:<14} {res:>12.6f} {quality:>12}")

        # Comparison with GT
        if gt_greens is not None:
            print("\n[4] CORRELATION WITH GT")
            print(f"{'Method':<14} {'Correlation':>12}")
            print("-" * 28)
            for method_name, result in results.items():
                if method_name != 'GT':
                    print(f"{result.method_name:<14} {result.correlation_with_gt:>12.4f}")

        # Diagnostic: Min location geodesic distance
        print("\n[5] MIN LOCATION DIAGNOSTIC (Is minimum at geodesically farthest point?)")
        print(f"{'Method':<14} {'Min Vertex':>10} {'Geo Dist':>12} {'Max Geo Dist':>12} {'Ratio':>10}")
        print("-" * 62)
        for method_name, result in results.items():
            ratio = result.geodesic_dist_to_min / result.max_geodesic_dist if result.max_geodesic_dist > 0 else 0.0
            print(f"{result.method_name:<14} {result.min_vertex_idx:>10} {result.geodesic_dist_to_min:>12.4f} "
                  f"{result.max_geodesic_dist:>12.4f} {ratio:>10.2%}")

        print("\n" + "=" * 90)

        # Overall summary
        all_pass = all(r.satisfies_maximum_principle for r in results.values())
        if all_pass:
            print("[OK] All methods satisfy the discrete maximum principle")
        else:
            print("[X] Some methods VIOLATE the discrete maximum principle:")
            for method_name, result in results.items():
                if not result.satisfies_maximum_principle:
                    print(f"  - {result.method_name}: Max value {result.max_value:.4f} at vertex {result.worst_violation_vertex}, "
                          f"but source has only {result.value_at_source:.4f}")

        print("=" * 90)

        # =====================================================================
        # Add visualizations to mesh structure
        # =====================================================================
        if mesh_structure is not None and not self._quantitative_mode:
            print("\nAdding Green's function visualizations...")

            # Add source vertex indicator
            source_point = vertices[source_vertex_idx:source_vertex_idx + 1]
            ps.register_point_cloud(
                name="Green's Function Source",
                points=source_point,
                radius=0.02,
                color=(1.0, 0.0, 0.0),  # Red
                enabled=True
            )

            # Add boundary vertices visualization
            try:
                # Find boundary vertices
                edge_count = {}
                for f in faces:
                    for i in range(3):
                        e = tuple(sorted([f[i], f[(i + 1) % 3]]))
                        edge_count[e] = edge_count.get(e, 0) + 1

                boundary_verts = set()
                for e, count in edge_count.items():
                    if count == 1:  # Boundary edge
                        boundary_verts.add(e[0])
                        boundary_verts.add(e[1])

                if len(boundary_verts) > 0:
                    boundary_indices = np.array(list(boundary_verts))
                    boundary_points = vertices[boundary_indices]
                    ps.register_point_cloud(
                        name="Boundary Vertices",
                        points=boundary_points,
                        radius=0.005,
                        color=(1.0, 1.0, 0.0),  # Yellow
                        enabled=False  # Disabled by default
                    )
                    print(f"    Added boundary visualization: {len(boundary_verts)} vertices")

                    # Also add boundary as scalar on mesh
                    boundary_mask = np.zeros(n, dtype=np.float32)
                    boundary_mask[boundary_indices] = 1.0
                    mesh_structure.add_scalar_quantity(
                        name="J0 Boundary Vertices",
                        values=boundary_mask,
                        enabled=False,
                        cmap='reds'
                    )
            except Exception as e:
                print(f"    [!] Could not add boundary visualization: {e}")

            # Add Green's function scalar fields
            for method_name, g in greens_functions.items():
                # Main Green's function visualization (mean-centered values)
                mesh_structure.add_scalar_quantity(
                    name=f"J1 Green's Function - {method_name}",
                    values=g,
                    enabled=(method_name == "GT"),  # Enable GT by default
                    cmap='coolwarm'  # Diverging colormap since values are centered around 0
                )

                # Shifted version for easier viewing (min=0)
                g_shifted = g - g.min()
                mesh_structure.add_scalar_quantity(
                    name=f"J2 Green's Function (shifted) - {method_name}",
                    values=g_shifted,
                    enabled=False,
                    cmap='viridis'
                )

                # Highlight violations: vertices with value > source value
                result = results.get(method_name)
                if result is not None:
                    source_val = g[source_vertex_idx]
                    violation_mask = (g > source_val + 1e-10).astype(np.float32)
                    if violation_mask.sum() > 0:
                        mesh_structure.add_scalar_quantity(
                            name=f"J3 Max Principle Violations - {method_name}",
                            values=violation_mask,
                            enabled=False,
                            cmap='reds'
                        )
                        print(f"    [{method_name}] Added violation visualization: {int(violation_mask.sum())} vertices")
            # Add pairwise difference visualizations
            if 'GT' in greens_functions and 'PRED' in greens_functions:
                diff = greens_functions['PRED'] - greens_functions['GT']
                mesh_structure.add_scalar_quantity(
                    name="J4 Green's Fn Diff (PRED - GT)",
                    values=diff,
                    enabled=False,
                    cmap='coolwarm'
                )

            if 'GT' in greens_functions and 'Robust' in greens_functions:
                diff = greens_functions['Robust'] - greens_functions['GT']
                mesh_structure.add_scalar_quantity(
                    name="J5 Green's Fn Diff (Robust - GT)",
                    values=diff,
                    enabled=False,
                    cmap='coolwarm'
                )

            print(f"  Added {len(greens_functions)} Green's function visualizations")

        return results

    def validate_heat_method_geodesics_step9(
            self,
            vertices: np.ndarray,
            faces: np.ndarray,
            L_gt: scipy.sparse.spmatrix,
            M_gt: scipy.sparse.spmatrix,
            L_pred: scipy.sparse.spmatrix,
            M_pred: scipy.sparse.spmatrix,
            L_robust: scipy.sparse.spmatrix,
            M_robust: scipy.sparse.spmatrix,
            source_vertex_idx: int = None,
            mesh_structure=None,
            k_neighbors: int = 20
    ) -> Dict[str, HeatMethodGeodesicResult]:
        """
        STEP 9: Heat Method Geodesic Distance Validation

        Computes geodesic distances using the Heat Method (Crane et al. 2013)
        with different Laplacians and compares them.

        The Heat Method:
        1. Solve heat equation: (M + t*L) u = delta_source
        2. Compute normalized gradient: X = -grad(u) / |grad(u)|
        3. Solve Poisson: L @ phi = div(X)

        We use pcdiff for gradient/divergence operators and compare:
        - GT (cotangent Laplacian)
        - PRED (your neural network)
        - Robust (tufted cover)
        - Reference (potpourri3d or exact geodesic)
        """
        print("\n" + "=" * 70)
        print("STEP 9: HEAT METHOD GEODESIC DISTANCE VALIDATION")
        print("=" * 70)

        # Detect operator mode from the loaded model
        operator_mode = getattr(self.current_model, '_operator_mode', 'stiffness') if self.current_model else 'stiffness'
        print(f"  Model operator mode: {operator_mode}")

        if not HAS_PCDIFF and operator_mode == "stiffness":
            print("[!] pcdiff not available and model is in stiffness mode - skipping Heat Method geodesics")
            print("    Install with: pip install pcdiff")
            return {}

        n = len(vertices)
        vertices = vertices.astype(np.float64)

        # Select source vertex using shared function
        if source_vertex_idx is None:
            source_vertex_idx = select_geodesic_source_vertex(vertices, method="centroid")

        # Cache source vertex (backward compat — multi-source state is in _source_indices)
        # This source_vertex_idx is used within this function only

        source_pos = vertices[source_vertex_idx]
        print(f"Source vertex: {source_vertex_idx} at ({source_pos[0]:.4f}, {source_pos[1]:.4f}, {source_pos[2]:.4f})")

        # =====================================================================
        # Build gradient and divergence operators
        # IMPORTANT: We need DIFFERENT operators for mesh vs point cloud methods!
        # - GT (cotangent): Uses mesh topology -> igl.grad()
        # - PRED/Robust: Uses point cloud topology -> pcdiff
        # =====================================================================

        # Build MESH-based gradient operator using igl (for GT Laplacian)
        mesh_grad_op = None
        mesh_face_areas = None
        if HAS_IGL:
            try:
                V = vertices.astype(np.float64)
                F = faces.astype(np.int32)
                _t_gt_grad = time.perf_counter()
                mesh_grad_op = igl.grad(V, F)  # (3*nF x nV) matrix
                mesh_face_areas = igl.doublearea(V, F).flatten() / 2.0  # (nF,)
                self.timing_results.gt_grad_op_time = time.perf_counter() - _t_gt_grad
                print(f"  Mesh gradient operator (igl): {mesh_grad_op.shape}")
                print(f"  [TIMING] GT grad operator (igl.grad + doublearea): {self.timing_results.gt_grad_op_time * 1000:.2f} ms")
            except Exception as e:
                print(f"  [!] igl.grad() failed: {e}")

        # Build POINT CLOUD gradient/divergence operators using shared function
        # Each method gets operators matched to its own k-NN connectivity.

        # --- PRED operators (from PRED's k-NN) ---
        pc_grad_op = None
        pc_div_op = None
        if HAS_PCDIFF and self.current_vertex_indices is not None:
            try:
                print("\nBuilding point cloud gradient/divergence operators from PRED's k-NN...")

                # Use shared function to convert k-NN indices to edge_index
                vertex_indices_np = self.current_vertex_indices.cpu().numpy()
                center_indices_np = self.current_center_indices.cpu().numpy()
                k = len(vertex_indices_np) // len(center_indices_np)

                edge_index = edge_index_from_knn_indices(vertex_indices_np, center_indices_np, k)
                print(f"  Extracted edge_index: {edge_index.shape} (k={k})")

                # Use shared function to build operators
                pc_grad_op, pc_div_op = build_pointcloud_grad_div_operators(vertices, edge_index)
                print(f"  Point cloud gradient operator: {pc_grad_op.shape}")
                print(f"  Point cloud divergence operator: {pc_div_op.shape}")
            except Exception as e:
                print(f"  [!] pcdiff operators (PRED) failed: {e}")
                import traceback
                traceback.print_exc()
        elif HAS_PCDIFF:
            # Fallback: build k-NN from scratch (won't match PRED exactly)
            try:
                print("\nBuilding point cloud gradient/divergence operators (fresh k-NN)...")
                edge_index = knn_graph(vertices, k=k_neighbors)
                pc_grad_op, pc_div_op = build_pointcloud_grad_div_operators(vertices, edge_index)
                print(f"  Point cloud gradient operator: {pc_grad_op.shape}")
                print(f"  Point cloud divergence operator: {pc_div_op.shape}")
                print(f"  WARNING: Using fresh k-NN, may not match PRED's connectivity!")
            except Exception as e:
                print(f"  [!] pcdiff operators failed: {e}")

        # Cache PRED pcdiff operators for targeted per-method recomputation
        self._pc_grad_op = pc_grad_op
        self._pc_div_op = pc_div_op

        # --- NeLo operators (from NeLo's own k) ---
        nelo_grad_op = None
        nelo_div_op = None
        if HAS_PCDIFF and self.current_nelo_L is not None:
            try:
                nelo_k = self.nelo_k
                print(f"\nBuilding point cloud gradient/divergence operators for NeLo (k={nelo_k})...")
                _t_nelo_grad = time.perf_counter()
                nelo_edge_index = knn_graph(vertices, k=nelo_k)
                nelo_grad_op, nelo_div_op = build_pointcloud_grad_div_operators(vertices, nelo_edge_index)
                self.timing_results.nelo_grad_op_time = time.perf_counter() - _t_nelo_grad
                print(f"  NeLo gradient operator: {nelo_grad_op.shape}")
                print(f"  NeLo divergence operator: {nelo_div_op.shape}")
                print(f"  [TIMING] NeLo grad/div operators: {self.timing_results.nelo_grad_op_time * 1000:.2f} ms")
            except Exception as e:
                print(f"  [!] pcdiff operators (NeLo) failed: {e}")
                import traceback
                traceback.print_exc()

        # --- Robust operators (from Robust's own k, only when using heat method) ---
        robust_grad_op = None
        robust_div_op = None
        if HAS_PCDIFF and self._robust_geodesic_heat and L_robust is not None:
            try:
                robust_k = self.reconstruction_settings.current_robust_k
                print(f"\nBuilding point cloud gradient/divergence operators for Robust (k={robust_k})...")
                robust_edge_index = knn_graph(vertices, k=robust_k)
                robust_grad_op, robust_div_op = build_pointcloud_grad_div_operators(vertices, robust_edge_index)
                print(f"  Robust gradient operator: {robust_grad_op.shape}")
                print(f"  Robust divergence operator: {robust_div_op.shape}")
            except Exception as e:
                print(f"  [!] pcdiff operators (Robust) failed: {e}")
                import traceback
                traceback.print_exc()

        # =====================================================================
        # Compute EXACT geodesic distances (TRUE ground truth)
        # =====================================================================
        print("\nComputing EXACT geodesic distances (ground truth)...")

        # Use precomputed exact geodesics if available from Step 7.5
        exact_distances = None
        exact_method = None
        if self._exact_geodesics and source_vertex_idx in self._exact_geodesics:
            exact_distances = self._exact_geodesics[source_vertex_idx]
            exact_method = "pygeodesic/igl (precomputed)"
            print(f"  Using precomputed exact geodesics for source {source_vertex_idx}")
        else:
            # Compute from scratch
            exact_distances = compute_exact_geodesics(vertices, faces, source_vertex_idx)

        if exact_distances is not None and exact_method is None:
            exact_method = "pygeodesic/igl"
            print(f"  Exact geodesics computed successfully")
        elif exact_distances is None:
            # Fallback to potpourri3d heat method (approximate)
            try:
                import potpourri3d as pp3d
                solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces.astype(np.int32))
                exact_distances = solver.compute_distance(source_vertex_idx)
                exact_method = "potpourri3d (approximate)"
                print(f"  Exact geodesics: potpourri3d heat method (approximate)")
            except Exception as e:
                print(f"  [!] potpourri3d failed: {e}")
                # Final fallback: Euclidean
                exact_distances = np.linalg.norm(vertices - vertices[source_vertex_idx], axis=1)
                exact_method = "euclidean"
                print("  [!] Using Euclidean distance as fallback")

        print(f"  Exact geodesic range: [{exact_distances.min():.4f}, {exact_distances.max():.4f}]")

        # Store for comparison
        reference_distances = exact_distances
        reference_method = exact_method

        # Cache exact distances in multi-source structure
        if self._exact_geodesics is None:
            self._exact_geodesics = {}
        self._exact_geodesics[source_vertex_idx] = exact_distances

        # =====================================================================
        # NOTE: normalize_distances and compute_geodesic_metrics are imported
        # from geodesic_utils - no local definitions needed
        # =====================================================================

        # =====================================================================
        # Helper functions for Heat Method with different topologies
        # These wrap the shared functions from geodesic_utils with timing/logging
        # =====================================================================

        def compute_heat_geodesic_mesh_local(L, M, grad_op, face_areas, method_name, t=None, device=None):
            """Wrapper around shared compute_heat_geodesic_mesh with timing."""
            start_time = time.time()

            # Use shared function from geodesic_utils
            distances = compute_heat_geodesic_mesh(
                L=L, M=M, grad_op=grad_op, face_areas=face_areas,
                source_idx=source_vertex_idx, n_vertices=n, t=t, device=device
            )

            elapsed_ms = (time.time() - start_time) * 1000
            return distances, t, elapsed_ms

        def compute_heat_geodesic_pointcloud_local(L, M, grad_op, div_op, method_name, t=None, device=None):
            """Wrapper around shared compute_heat_geodesic_pointcloud with timing."""
            start_time = time.time()

            # Use shared function from geodesic_utils
            distances = compute_heat_geodesic_pointcloud(
                L=L, M=M, grad_op=grad_op, div_op=div_op,
                source_idx=source_vertex_idx, n_vertices=n, t=t, device=device
            )

            elapsed_ms = (time.time() - start_time) * 1000
            return distances, t, elapsed_ms

        # =====================================================================
        # Compute geodesics with each Laplacian
        # =====================================================================
        geodesic_distances = {}

        # GT Laplacian: Use mesh-based gradient (igl.grad)
        if L_gt is not None and M_gt is not None and mesh_grad_op is not None:
            print(f"\nComputing Heat Method geodesic with GT Laplacian (mesh gradient)...")
            distances, t, elapsed_ms = compute_heat_geodesic_mesh_local(
                L_gt, M_gt, mesh_grad_op, mesh_face_areas, "GT"
            )
            if distances is not None:
                geodesic_distances["GT"] = distances
                self.timing_results.gt_geodesic_solve_time = elapsed_ms / 1000.0
                print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                print(f"  Time: {elapsed_ms:.1f} ms")
        else:
            print(f"\n[GT] Mesh gradient not available, skipping")

        # PRED Laplacian: branch on operator mode
        if L_pred is not None and M_pred is not None:
            if operator_mode == "gradient":
                # === Gradient mode: self-consistent heat method using learned G ===
                print(f"\nComputing Heat Method geodesic with PRED Laplacian (learned gradient operator)...")
                try:
                    # Reuse the gradient operator already assembled in step 5
                    gradient_operator = self.current_learned_gradient_op
                    if gradient_operator is not None:
                        print(f"  Using cached gradient operator G: {gradient_operator.shape} ({gradient_operator.nnz} non-zeros)")

                        start_time = time.time()
                        distances = compute_heat_geodesic_learned(
                            S=L_pred, M=M_pred, G=gradient_operator,
                            source_idx=source_vertex_idx, n_vertices=n,
                            device=self.current_device
                        )
                        elapsed_ms = (time.time() - start_time) * 1000

                        if distances is not None:
                            geodesic_distances["PRED"] = distances
                            self.timing_results.pred_geodesic_solve_time = elapsed_ms / 1000.0
                            print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                            print(f"  Time: {elapsed_ms:.1f} ms")
                        else:
                            print(f"  [!] Learned heat method returned None")
                    else:
                        print(f"  [!] Learned gradient operator not available, skipping")
                except Exception as e:
                    print(f"  [!] Learned heat method failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # === Stiffness mode: frankenstein heat method (pcdiff grad/div) ===
                if pc_grad_op is not None and pc_div_op is not None:
                    print(f"\nComputing Heat Method geodesic with PRED Laplacian (matched k-NN)...")
                    distances, t, elapsed_ms = compute_heat_geodesic_pointcloud_local(
                        L_pred, M_pred, pc_grad_op, pc_div_op, "PRED",
                        device=self.current_device
                    )
                    if distances is not None:
                        geodesic_distances["PRED"] = distances
                        self.timing_results.pred_geodesic_solve_time = elapsed_ms / 1000.0
                        print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                        print(f"  Time: {elapsed_ms:.1f} ms")
                else:
                    print(f"\n[PRED] Point cloud gradient not available, skipping Heat Method")

        # Robust Laplacian geodesics: pp3d solver (default) or heat method with matched operators
        if L_robust is not None and M_robust is not None:
            if self._robust_geodesic_heat:
                # Heat method with Robust's own L, M and matched pcdiff operators
                if robust_grad_op is not None and robust_div_op is not None:
                    robust_k = self.reconstruction_settings.current_robust_k
                    print(f"\nComputing Heat Method geodesic with Robust Laplacian (matched k={robust_k})...")
                    distances, _t, elapsed_ms = compute_heat_geodesic_pointcloud_local(
                        L_robust, M_robust, robust_grad_op, robust_div_op, "Robust"
                    )
                    if distances is not None:
                        geodesic_distances["Robust"] = distances
                        self.timing_results.robust_geodesic_solve_time = elapsed_ms / 1000.0
                        print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                        print(f"  Time: {elapsed_ms:.1f} ms")
                else:
                    print(f"\n[Robust] Point cloud gradient not available, skipping Heat Method")
            else:
                # Default: potpourri3d point cloud solver
                print(f"\n[Robust] Using potpourri3d point cloud solver...")
                try:
                    import potpourri3d as pp3d
                    _t_geo_start = time.perf_counter()
                    pc_solver = pp3d.PointCloudHeatSolver(vertices)
                    robust_distances = pc_solver.compute_distance(source_vertex_idx)
                    _t_geo_elapsed = time.perf_counter() - _t_geo_start
                    self.timing_results.robust_geodesic_solve_time = _t_geo_elapsed
                    geodesic_distances["Robust (pp3d)"] = robust_distances
                    print(f"  Distance range: [{robust_distances.min():.4f}, {robust_distances.max():.4f}]")
                    print(f"  Time: {_t_geo_elapsed * 1000:.1f} ms")
                except Exception as e:
                    print(f"  [!] potpourri3d point cloud solver failed: {e}")

        # NeLo Laplacian: use point-cloud Heat Method with NeLo's own matched k-NN operators
        if self.current_nelo_L is not None and self.current_nelo_M is not None:
            if nelo_grad_op is not None and nelo_div_op is not None:
                print(f"\nComputing Heat Method geodesic with NeLo Laplacian (matched k={self.nelo_k})...")
                distances, _t, elapsed_ms = compute_heat_geodesic_pointcloud_local(
                    self.current_nelo_L, self.current_nelo_M, nelo_grad_op, nelo_div_op, "NeLo",
                    device=self.current_device
                )
                if distances is not None:
                    geodesic_distances["NeLo"] = distances
                    self.timing_results.nelo_geodesic_solve_time = elapsed_ms / 1000.0
                    print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                    print(f"  Time: {elapsed_ms:.1f} ms")
            else:
                print(f"\n[NeLo] Point cloud gradient not available, skipping Heat Method")

        # Add exact geodesics to the comparison set
        geodesic_distances["Exact"] = exact_distances

        # Cache raw geodesic distances for slider refresh
        if source_vertex_idx is not None:
            if source_vertex_idx not in self._all_geodesic_distances:
                self._all_geodesic_distances[source_vertex_idx] = {}
            for mk, dv in geodesic_distances.items():
                if mk != "Exact":
                    self._all_geodesic_distances[source_vertex_idx][mk] = dv

        # =====================================================================
        # Compute quality metrics for each method vs EXACT geodesics
        # All comparisons use NORMALIZED distances for fair comparison
        # =====================================================================
        print("\n" + "=" * 70)
        print("GEODESIC QUALITY METRICS (vs Exact Geodesics)")
        print(f"Ground truth: {exact_method}")
        print("All metrics computed on NORMALIZED [0,1] distances")
        print("=" * 70)

        results = {}
        unreasonable_methods = set()  # Track methods with unreasonable distance ranges

        for method_name, distances in geodesic_distances.items():
            if method_name == "Exact":
                continue  # Don't compare exact to itself

            # Flag unreasonable results (normalized mesh, geodesics should be O(1))
            if distances.max() > 1e3:
                unreasonable_methods.add(method_name)

            # Use the shared metrics function (returns GeodesicMetrics dataclass)
            metrics = compute_geodesic_metrics(distances, exact_distances)

            result = HeatMethodGeodesicResult(
                method_name=method_name,
                source_vertex_idx=source_vertex_idx,
                num_vertices=n,
                time_step=0.0,
                min_distance=float(distances.min()),
                max_distance=float(distances.max()),
                mean_distance=float(distances.mean()),
                distance_at_source=float(distances[source_vertex_idx]),
                correlation_with_reference=float(metrics.correlation),
                mean_absolute_error=float(metrics.mae_normalized),
                max_absolute_error=float(metrics.max_error_normalized),
                relative_error_percent=float(metrics.mae_normalized * 100),  # As percentage of [0,1] range
                monotonicity_score=float(metrics.monotonicity),
                computation_time_ms=0.0
            )
            results[method_name] = result

        # Print summary table
        print(f"\n{'Method':<20} {'Corr':>8} {'MAE':>8} {'MaxErr':>8} {'Mono':>8} {'Range':<20}")
        print("-" * 76)
        for method_name, distances in geodesic_distances.items():
            warn = " [!]" if method_name in unreasonable_methods else ""
            if method_name == "Exact":
                d_range = f"[{distances.min():.3f}, {distances.max():.3f}]"
                print(f"{method_name:<20} {'1.0000':>8} {'0.0000':>8} {'0.0000':>8} {'1.0000':>8} {d_range:<20}")
            else:
                r = results[method_name]
                d_range = f"[{distances.min():.3f}, {distances.max():.3f}]"
                print(f"{method_name + warn:<20} {r.correlation_with_reference:>8.4f} {r.mean_absolute_error:>8.4f} "
                      f"{r.max_absolute_error:>8.4f} {r.monotonicity_score:>8.4f} {d_range:<20}")

        if unreasonable_methods:
            print(f"\n  [!] = unreasonable distance range (max >> 1), metrics may be meaningless")

        # =====================================================================
        # Add visualizations
        # =====================================================================
        if mesh_structure is not None and not self._quantitative_mode:
            print("\nAdding geodesic visualizations...")

            # Add source vertex indicator
            source_point = vertices[source_vertex_idx:source_vertex_idx + 1]
            ps.register_point_cloud(
                name="K0 Geodesic Source",
                points=source_point,
                radius=0.02,
                color=(0.0, 1.0, 0.0),  # Green
                enabled=True
            )

            # Add geodesic distance visualizations (all normalized to [0,1])
            for method_name, distances in geodesic_distances.items():
                # Normalized distances (0 to 1) for fair visual comparison
                d_norm = normalize_distances(distances)

                # Use different colormap for unreasonable results
                is_bad = method_name in unreasonable_methods
                label_suffix = " [!]" if is_bad else ""
                cmap = 'hot' if is_bad else 'viridis'

                mesh_structure.add_scalar_quantity(
                    name=f"K1 Geodesic (norm) - {method_name}{label_suffix}",
                    values=d_norm,
                    enabled=(method_name == "Exact"),
                    cmap=cmap
                )

            # Add error visualizations (normalized error vs exact)
            exact_norm = normalize_distances(exact_distances)
            for method_name, distances in geodesic_distances.items():
                if method_name == 'Exact':
                    continue

                d_norm = normalize_distances(distances)
                error = np.abs(d_norm - exact_norm)
                mesh_structure.add_scalar_quantity(
                    name=f"K2 Geodesic Error - {method_name}",
                    values=error,
                    enabled=False,
                    cmap='reds'
                )

            print(f"  Added {len(geodesic_distances)} geodesic visualizations")

        return results

    def process_batch(self, model: LaplacianTransformerModule, batch_data, batch_idx: int, device: torch.device) -> bool:
        """Process a single batch through the complete pipeline.

        Returns:
            True if processing succeeded, False if the mesh was skipped.
        """
        if not self._verbose:
            # Suppress all stdout from the processing pipeline
            devnull = open(os.devnull, 'w')
            saved_stdout = sys.stdout
            sys.stdout = devnull
        try:
            return self._process_batch_impl(model, batch_data, batch_idx, device)
        finally:
            if not self._verbose:
                sys.stdout = saved_stdout
                devnull.close()

    def _process_batch_impl(self, model: LaplacianTransformerModule, batch_data, batch_idx: int, device: torch.device) -> bool:
        """Internal implementation of process_batch."""
        print(f"\n{'=' * 80}")
        print(f"PROCESSING BATCH {batch_idx + 1}")
        print('=' * 80)

        # Step-level timing tracker
        _step_times = {}
        _step_t0 = time.perf_counter()
        _current_step = "init"

        def _mark_step(name):
            nonlocal _step_t0, _current_step
            elapsed = time.perf_counter() - _step_t0
            _step_times[_current_step] = elapsed
            _current_step = name
            _step_t0 = time.perf_counter()

        # Clear previous visualization and tracking (only if polyscope is active)
        if not self._quantitative_mode:
            ps.remove_all_structures()
        self.reconstruction_structure_names.clear()  # Clear reconstruction tracking

        # Clear CUDA cache — only in visual mode where Polyscope needs the VRAM.
        # In quantitative mode, keep the caching allocator warm so subsequent meshes
        # with similar sizes reuse memory blocks (~12ms vs ~300ms per inference).
        if device.type == 'cuda':
            if not self._quantitative_mode:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Log GPU memory status
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            print(f"[GPU Memory] Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")

        # Clear stored references from previous batch to free memory
        # This helps ensure consistent timing across batches
        self._clear_stored_references()

        # STEP 1: Extract mesh data from batch
        _mark_step("step1_extract")
        print("STEP 1: Extracting mesh data from batch")

        # Handle case where batch_data is a list (from MeshDataset)
        if isinstance(batch_data, list):
            data = batch_data[0]  # Take the first Data object
        else:
            data = batch_data

        # Extract mesh metadata - handle both single values and lists
        def extract_first_if_list(attr_value):
            """Extract single value from potentially batched attribute."""
            if isinstance(attr_value, list):
                return attr_value[0] if len(attr_value) > 0 else None
            return attr_value

        mesh_file_path = extract_first_if_list(data.mesh_file_path)
        original_num_vertices = extract_first_if_list(data.original_num_vertices)

        print(f"Mesh file: {Path(mesh_file_path).name}")
        print(f"Original vertices: {original_num_vertices}")

        # New MeshDataset format: raw_vertices instead of patches
        num_vertices = int(data.num_mesh_vertices[0]) if hasattr(data, 'num_mesh_vertices') else len(data.raw_vertices)
        original_k = int(data.k_neighbors) if hasattr(data, 'k_neighbors') else 30
        print(f"Mesh vertices: {num_vertices}")
        print(f"Dataset k: {original_k}")

        # STEP 2: Load original mesh for GT computation
        _mark_step("step2_gt_load")
        print(f"\nSTEP 2: Loading original mesh for GT computation")
        gt_data = self.load_original_mesh_for_gt(mesh_file_path)

        # Check if GT eigendecomposition succeeded (PyFM can fail on degenerate meshes)
        if gt_data.get('gt_eigenvalues') is None:
            print(f"[X] GT eigendecomposition failed for {Path(mesh_file_path).name} "
                  f"(likely degenerate triangles), skipping this mesh")
            return False

        # STEP 2.5: Build patches with PRED k using GPU kNN
        pred_k = self._initial_k_pred
        _mark_step("step2_5_patches")
        print(f"\nSTEP 2.5: Extracting patches with k={pred_k} (for timing comparison)...")

        # Cache vertices tensor on GPU (reused for k-change updates)
        self._current_vertices_tensor = torch.from_numpy(gt_data['vertices']).float().to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_extraction_start = time.perf_counter()

        patch_data = build_patches_from_vertices(
            self._current_vertices_tensor, pred_k, device=device
        )

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_extraction_end = time.perf_counter()
        pred_extraction_time = t_extraction_end - t_extraction_start
        self.timing_results.pred_patch_extraction_time = pred_extraction_time
        print(f"  [TIMING] PRED patch extraction (k-NN): {pred_extraction_time * 1000:.2f} ms")

        # Store patch connectivity for pcdiff gradient/divergence operators (needed for geodesics)
        self.current_vertex_indices = patch_data.vertex_indices.to(device)
        self.current_center_indices = patch_data.center_indices.to(device)
        self.current_batch_indices = patch_data.patch_idx.to(device)

        # STEP 3: Model inference (use our extracted patches for consistent timing)
        _mark_step("step3_inference")
        print(f"\nSTEP 3: Model inference")
        inference_result = self.perform_model_inference(model, patch_data, device)

        if inference_result['predicted_eigenvalues'] is None:
            print("[X] Failed to compute eigendecomposition, skipping this batch")
            return False

        # Compute PRED total time
        self.timing_results.pred_total_time = (
                self.timing_results.pred_patch_extraction_time +
                self.timing_results.pred_model_inference_time +
                self.timing_results.pred_matrix_assembly_time
        )

        # Compute GT matrix assembly timing (for fair comparison, time only matrix construction)
        _mark_step("step3_5_gt_timing")
        print(f"\nSTEP 3.5: Computing GT matrix assembly timing...")
        self._compute_gt_matrices_timed(gt_data['vertices'], gt_data['faces'])

        # STEP 4: Compute predicted quantities
        _mark_step("step4_predicted")
        print(f"\nSTEP 4: Computing predicted quantities")
        predicted_data = self.compute_predicted_quantities_from_laplacian(
            inference_result['stiffness_matrix'], gt_data['vertices'],
            mass_matrix=inference_result.get('mass_matrix')
        )

        # STEP 5: Store current batch data for UI-driven re-computation
        self.current_gt_data = gt_data
        self.current_inference_result = inference_result
        self.current_predicted_data = predicted_data

        # Store raw data for k-NN slider updates
        self.current_stiffness_weights = inference_result['stiffness_weights']
        self.current_areas = inference_result['areas']
        self.current_original_vertices = gt_data['vertices']
        self.original_k = original_k  # Use the k calculated earlier
        self.current_vertex_indices = patch_data.vertex_indices.to(device)
        self.current_center_indices = patch_data.center_indices.to(device)
        self.current_batch_indices = patch_data.patch_idx.to(device)  # MeshPatchData uses patch_idx

        # Store model, device, and mesh info for k updates
        self.current_model = model
        self.current_device = device
        self.current_mesh_file_path = mesh_file_path
        self.current_faces = gt_data['faces']

        # Assemble learned gradient operator G if in gradient mode
        self.current_learned_gradient_op = None
        operator_mode = getattr(model, '_operator_mode', 'stiffness')
        forward_result = inference_result.get('forward_result')
        if operator_mode == "gradient" and forward_result is not None and forward_result.get('grad_coeffs') is not None:
            try:
                print("Assembling learned gradient operator G...")
                _t_grad_op = time.perf_counter()
                batch_indices_for_g = self.current_batch_indices.clone()
                self.current_learned_gradient_op = assemble_gradient_operator(
                    grad_coeffs=forward_result['grad_coeffs'],
                    attention_mask=forward_result['attention_mask'],
                    vertex_indices=self.current_vertex_indices,
                    center_indices=self.current_center_indices,
                    batch_indices=batch_indices_for_g
                )
                self.timing_results.pred_grad_op_time = time.perf_counter() - _t_grad_op
                print(f"  Learned G: {self.current_learned_gradient_op.shape} ({self.current_learned_gradient_op.nnz} non-zeros)")
                print(f"  [TIMING] Gradient operator assembly: {self.timing_results.pred_grad_op_time * 1000:.2f} ms")
            except Exception as e:
                print(f"  [!] Failed to assemble learned gradient operator: {e}")
                self.current_learned_gradient_op = None

        # Initialize PRED and Robust k sliders with per-method k from config
        self.reconstruction_settings.current_pred_k = self._initial_k_pred
        self.reconstruction_settings.current_robust_k = self._initial_k_robust
        print(f"Initialized PRED k={self._initial_k_pred}, Robust k={self._initial_k_robust}, NeLo k={self.nelo_k}")

        # Update timing results with mesh info (needed for UI display)
        self.timing_results.num_vertices = len(gt_data['vertices'])
        self.timing_results.num_faces = len(gt_data['faces'])
        self.timing_results.current_k = self._initial_k_pred

        # STEP 5.5: Compute robust-laplacian with point_cloud_laplacian using robust k
        robust_k = self._initial_k_robust
        _mark_step("step5_5_robust")
        print(f"\nSTEP 5.5: Computing robust-laplacian with k={robust_k}...")
        robust_eigenvalues, robust_eigenvectors, robust_vertex_areas = self._compute_robust_laplacian_with_k(
            gt_data['vertices'], robust_k
        )
        gt_data['robust_eigenvalues'] = robust_eigenvalues
        gt_data['robust_eigenvectors'] = robust_eigenvectors
        gt_data['robust_vertex_areas'] = robust_vertex_areas

        # STEP 5.6: Compute Laplacian sparsity comparison
        _mark_step("step5_6_sparsity")
        print(f"\nSTEP 5.6: Computing Laplacian sparsity comparison...")

        _mark_step("step5_7_nelo")
        # STEP 5.7: NeLo inference
        self.current_nelo_L = None
        self.current_nelo_M = None
        self.current_nelo_eigenvectors = None
        self.current_nelo_eigenvalues = None
        if self.nelo_pipeline is not None:
            print(f"\nSTEP 5.7: NeLo inference (k={self.nelo_k})...")
            nelo_result = self.perform_nelo_inference(
                gt_data['vertices'],
                self.config.num_eigenvectors_to_show,
                device,
            )
            if nelo_result is not None:
                self.current_nelo_L            = nelo_result['L']
                self.current_nelo_M            = nelo_result['M']
                self.current_nelo_eigenvectors = nelo_result['eigenvectors']
                self.current_nelo_eigenvalues  = nelo_result['eigenvalues']
                t_nelo = nelo_result['timing']
                self.timing_results.nelo_graph_tree_time      = t_nelo['graph_tree']
                self.timing_results.nelo_model_inference_time = t_nelo['forward']
                self.timing_results.nelo_matrix_assembly_time = t_nelo['assembly']
                self.timing_results.nelo_total_time           = t_nelo['total']
                self.timing_results.nelo_k                    = self.nelo_k
        else:
            print("\nSTEP 5.7: NeLo skipped (pipeline not loaded)")
        self._compute_and_store_all_sparsity_stats()
        _mark_step("step6_eigen_viz")
        if not self._quantitative_mode:
            print(f"\nSTEP 6: Creating comprehensive visualization")
            mesh_structure = self.visualize_mesh(gt_data['vertices'], gt_data['gt_vertex_normals'], gt_data['faces'])
            self.current_mesh_structure = mesh_structure
        else:
            print(f"\nSTEP 6: Skipping visualization (quantitative_mode=True)")
            mesh_structure = None

        # Print analysis (always do this - it's console output)
        if self.config.enable_eigenvalue_info:
            self.print_eigenvalue_analysis(
                gt_data.get('gt_eigenvalues'),
                inference_result['predicted_eigenvalues'],
                Path(mesh_file_path).name
            )

        # Print correlation analysis (always do this - it's console output)
        if self.config.enable_correlation_analysis:
            gt_eigenvecs = gt_data.get('gt_eigenvectors')
            if gt_eigenvecs is not None and inference_result['predicted_eigenvectors'] is not None:
                correlation_matrix = self.compute_eigenvector_correlations(
                    gt_eigenvecs, inference_result['predicted_eigenvectors']
                )
                self.print_eigenvector_correlation_analysis(correlation_matrix, "Predicted vs GT")

        # Compute eigenvector cosine similarities (always — needed for quantitative metrics)
        gt_eigvecs = gt_data.get('gt_eigenvectors')
        pred_eigvecs = inference_result.get('predicted_eigenvectors')
        robust_eigvecs = gt_data.get('robust_eigenvectors')
        nelo_eigvecs = self.current_nelo_eigenvectors

        self.current_cosine_similarities_pred = self._compute_eigenvector_cosine_similarities(
            gt_eigvecs, pred_eigvecs
        )
        self.current_cosine_similarities_robust = self._compute_eigenvector_cosine_similarities(
            gt_eigvecs, robust_eigvecs
        )
        self.current_cosine_similarities_nelo = self._compute_eigenvector_cosine_similarities(
            gt_eigvecs, nelo_eigvecs
        )

        if self.current_cosine_similarities_pred is not None:
            num_to_show = min(self.config.num_eigenvectors_to_show, len(self.current_cosine_similarities_pred))
            self.current_avg_cosine_similarity = float(self.current_cosine_similarities_pred[:num_to_show].mean())

        # Add visualizations (skip if in quantitative mode)
        if not self._quantitative_mode and mesh_structure is not None:
            self.visualize_comprehensive_eigenvectors(
                mesh_structure,
                gt_data.get('gt_eigenvalues'), gt_eigvecs,
                inference_result['predicted_eigenvalues'], pred_eigvecs,
                gt_data.get('robust_eigenvalues'), robust_eigvecs,
                self.current_nelo_eigenvalues, nelo_eigvecs
            )

            self.add_comprehensive_curvature_visualizations(mesh_structure, gt_data, predicted_data, inference_result)

            # Add weight neighborhood visualizations (FPS-sampled vertices)
            self._weight_nbr_sample_indices = None  # reset for new mesh
            self._add_weight_neighborhood_visualization()

            # Add mesh reconstructions using eigenvectors
            print(f"\nSTEP 7: Computing and visualizing mesh reconstructions")
            self._update_mesh_reconstructions(gt_data, inference_result)
        else:
            print(f"\nSTEP 7: Skipping mesh reconstructions (quantitative_mode=True)")

        # STEP 7.5: Select multiple source vertices for Green's + geodesic validation
        vertices_for_sources = gt_data['vertices'].astype(np.float64)
        self._source_indices = select_multiple_geodesic_sources(
            vertices_for_sources, num_sources=self.config.num_validation_sources, method="farthest_point_sampling", seed=42
        ).tolist()
        self._exact_geodesics = {}
        self._greens_gt_values = {}
        self._all_greens_results = {}
        self._all_geodesic_results = {}
        self._all_greens_values = {}
        self._all_geodesic_distances = {}
        self._current_source_display_idx = 0
        _mark_step("step7_5_sources")
        print(f"\nSelected {len(self._source_indices)} source vertices (FPS): {self._source_indices}")

        # Precompute exact geodesics for all sources
        print("Precomputing exact geodesics for all sources...")
        for src_idx in self._source_indices:
            exact = compute_exact_geodesics(vertices_for_sources, gt_data['faces'], int(src_idx))
            if exact is not None:
                self._exact_geodesics[src_idx] = exact
        print(f"  Exact geodesics computed for {len(self._exact_geodesics)}/{len(self._source_indices)} sources")

        # Register initial source point marker
        if not self._quantitative_mode and self._source_indices:
            first_src = self._source_indices[0]
            src_pos = gt_data['vertices'][first_src:first_src + 1]
            ps.register_point_cloud("Source Vertex", src_pos, radius=0.025, color=(0.6, 0.0, 0.8))

        # =====================================================================
        # STEP 8: Green's function validation (all sources, batched)
        # =====================================================================
        _mark_step("step8_greens")
        print(f"\nSTEP 8: Green's function validation ({len(self._source_indices)} sources)")

        try:
            all_sources = self._source_indices
            vertices_for_greens = gt_data['vertices']
            faces_for_greens = gt_data['faces']

            # Get L,M for each method
            method_LM = {}
            for mk in ['GT', 'PRED', 'Robust', 'NeLo']:
                method_LM[mk] = self._get_method_L_M(mk)

            # Batch Green's function: factorize once per method, solve all sources
            for method_key in ['GT', 'PRED', 'Robust', 'NeLo']:
                Lm, Mm = method_LM[method_key]
                if Lm is None:
                    continue

                _t = time.perf_counter()
                batch_results = self.compute_greens_function_batch(Lm, Mm, all_sources)
                elapsed = time.perf_counter() - _t

                # Store timing
                time_field = {'GT': 'gt_greens_solve_time', 'PRED': 'pred_greens_solve_time',
                              'Robust': 'robust_greens_solve_time', 'NeLo': 'nelo_greens_solve_time'}
                if method_key in time_field:
                    setattr(self.timing_results, time_field[method_key], elapsed)

                # Process results for each source
                for src_idx in all_sources:
                    result = batch_results.get(src_idx)
                    if result is None:
                        continue
                    greens_values, residual = result

                    # Store GT Green's for correlation comparison
                    if method_key == 'GT':
                        self._greens_gt_values[src_idx] = greens_values

                    # Validate maximum principle
                    gt_greens = self._greens_gt_values.get(src_idx)
                    validation = self.validate_maximum_principle(
                        greens_values, src_idx, method_key,
                        vertices=vertices_for_greens, faces=faces_for_greens,
                        laplacian_residual_norm=residual,
                        gt_greens_function=gt_greens
                    )
                    if src_idx not in self._all_greens_results:
                        self._all_greens_results[src_idx] = {}
                    self._all_greens_results[src_idx][method_key] = validation
                    if src_idx not in self._all_greens_values:
                        self._all_greens_values[src_idx] = {}
                    self._all_greens_values[src_idx][method_key] = greens_values

                n_ok = sum(1 for s in all_sources if batch_results.get(s) is not None)
                print(f"  {method_key}: {n_ok}/{len(all_sources)} sources in {elapsed*1000:.1f} ms")

            # Set current display results (first source)
            first_src = all_sources[0]
            if first_src in self._all_greens_results:
                self.current_greens_results = self._all_greens_results[first_src]

            # Aggregate
            self._aggregate_greens_metrics()

            # Visualization for first source (visual mode only)
            if not self._quantitative_mode and mesh_structure is not None:
                for method_key in ['GT', 'PRED', 'Robust', 'NeLo']:
                    greens_vals = (self._all_greens_values.get(first_src) or {}).get(method_key)
                    if greens_vals is not None:
                        mesh_structure.add_scalar_quantity(
                            name=f"J1 Green's Function - {method_key}",
                            values=greens_vals, enabled=False, cmap='viridis'
                        )

        except Exception as e:
            print(f"  [!] Green's function validation failed: {e}")
            import traceback
            traceback.print_exc()

        # =====================================================================
        # STEP 9: Heat Method Geodesic Distance Validation (all sources, batched)
        # =====================================================================
        _mark_step("step9_geodesic")
        print(f"\nSTEP 9: Heat Method geodesic distance validation ({len(self._source_indices)} sources)")

        try:
            vertices = gt_data['vertices']
            faces = gt_data['faces']
            n = len(vertices)
            all_sources = self._source_indices

            operator_mode = getattr(self.current_model, '_operator_mode', 'stiffness') if self.current_model else 'stiffness'

            # ---- Build gradient operators once ----
            gt_grad_div_fn = None
            gt_t = None
            L_gt_geo, M_gt_geo = None, None
            if HAS_IGL:
                V = vertices.astype(np.float64)
                F = faces.astype(np.int32)
                L_gt_geo = -igl.cotmatrix(V, F)
                M_gt_geo = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)
                mesh_grad_op = igl.grad(V, F)
                mesh_face_areas = igl.doublearea(V, F).flatten() / 2.0
                gt_grad_div_fn = make_grad_div_mesh(mesh_grad_op, mesh_face_areas)
                gt_t = (1.52 * np.sqrt(mesh_face_areas.mean())) ** 2

            # PRED: learned gradient operator or pointcloud fallback
            pred_grad_div_fn = None
            L_pred_geo = inference_result.get('stiffness_matrix')
            M_pred_geo = inference_result.get('mass_matrix')
            if L_pred_geo is not None and M_pred_geo is not None:
                if operator_mode == 'gradient' and self.current_learned_gradient_op is not None:
                    pred_grad_div_fn = make_grad_div_learned(self.current_learned_gradient_op, M_pred_geo)
                elif HAS_PCDIFF and self.current_vertex_indices is not None:
                    # Build pointcloud operators from PRED's k-NN
                    vertex_indices_np = self.current_vertex_indices.cpu().numpy()
                    center_indices_np = self.current_center_indices.cpu().numpy()
                    k_pred = len(vertex_indices_np) // len(center_indices_np)
                    edge_index = edge_index_from_knn_indices(vertex_indices_np, center_indices_np, k_pred)
                    pc_grad_op, pc_div_op = build_pointcloud_grad_div_operators(vertices, edge_index)
                    self._pc_grad_op = pc_grad_op
                    self._pc_div_op = pc_div_op
                    pred_grad_div_fn = make_grad_div_pointcloud(pc_grad_op, pc_div_op)

            # NeLo: build pointcloud operators from NeLo's own k-NN
            nelo_grad_div_fn = None
            L_nelo_geo = self.current_nelo_L
            M_nelo_geo = self.current_nelo_M
            if L_nelo_geo is not None and M_nelo_geo is not None and HAS_PCDIFF:
                try:
                    nelo_k = self.nelo_k
                    nelo_edge_index = knn_graph(vertices, k=nelo_k)
                    nelo_grad_op, nelo_div_op = build_pointcloud_grad_div_operators(vertices, nelo_edge_index)
                    nelo_grad_div_fn = make_grad_div_pointcloud(nelo_grad_op, nelo_div_op)
                except Exception as e:
                    print(f"  [!] NeLo grad/div operators failed: {e}")

            # ---- Batch geodesics for GT, PRED, NeLo ----
            geodesic_times = {}

            for method_key, S, M_mat, fn, t_override in [
                ('GT',   L_gt_geo,   M_gt_geo,   gt_grad_div_fn,   gt_t),
                ('PRED', L_pred_geo, M_pred_geo, pred_grad_div_fn, None),
                ('NeLo', L_nelo_geo, M_nelo_geo, nelo_grad_div_fn, None),
            ]:
                if S is None or M_mat is None or fn is None:
                    continue
                _t = time.perf_counter()
                results = compute_heat_geodesics_batch(S, M_mat, all_sources, n, fn, t=t_override)
                elapsed = time.perf_counter() - _t
                geodesic_times[method_key] = elapsed
                for i, src_idx in enumerate(all_sources):
                    if results[i] is not None:
                        if src_idx not in self._all_geodesic_distances:
                            self._all_geodesic_distances[src_idx] = {}
                        self._all_geodesic_distances[src_idx][method_key] = results[i]
                print(f"  {method_key}: {len(all_sources)} sources in {elapsed*1000:.1f} ms "
                      f"({elapsed/len(all_sources)*1000:.1f} ms/source)")

            # ---- Robust: pp3d per-source (has its own internal C++ solver) ----
            if not self._skip_robust:
                import potpourri3d as pp3d
                _t = time.perf_counter()
                pc_solver = pp3d.PointCloudHeatSolver(vertices)
                for src_idx in all_sources:
                    try:
                        d = pc_solver.compute_distance(src_idx)
                        if src_idx not in self._all_geodesic_distances:
                            self._all_geodesic_distances[src_idx] = {}
                        self._all_geodesic_distances[src_idx]['Robust (pp3d)'] = d
                    except Exception:
                        pass
                elapsed = time.perf_counter() - _t
                geodesic_times['Robust'] = elapsed
                print(f"  Robust (pp3d): {len(all_sources)} sources in {elapsed*1000:.1f} ms "
                      f"({elapsed/len(all_sources)*1000:.1f} ms/source)")

            # ---- Store timing ----
            if 'GT' in geodesic_times:
                self.timing_results.gt_geodesic_solve_time = geodesic_times['GT']
            if 'PRED' in geodesic_times:
                self.timing_results.pred_geodesic_solve_time = geodesic_times['PRED']
            if 'Robust' in geodesic_times:
                self.timing_results.robust_geodesic_solve_time = geodesic_times['Robust']
            if 'NeLo' in geodesic_times:
                self.timing_results.nelo_geodesic_solve_time = geodesic_times['NeLo']

            # ---- Compute metrics for all sources ----
            for src_idx in all_sources:
                exact_distances = self._exact_geodesics.get(src_idx)
                if exact_distances is None:
                    continue
                src_dists = self._all_geodesic_distances.get(src_idx, {})
                for method_key, distances in src_dists.items():
                    metrics = compute_geodesic_metrics(distances, exact_distances)
                    result = HeatMethodGeodesicResult(
                        method_name=method_key,
                        source_vertex_idx=src_idx,
                        num_vertices=n,
                        min_distance=float(distances.min()),
                        max_distance=float(distances.max()),
                        mean_distance=float(distances.mean()),
                        distance_at_source=float(distances[src_idx]),
                        correlation_with_reference=float(metrics.correlation),
                        mean_absolute_error=float(metrics.mae_normalized),
                        max_absolute_error=float(metrics.max_error_normalized),
                        relative_error_percent=float(metrics.mae_normalized * 100),
                        monotonicity_score=float(metrics.monotonicity),
                    )
                    if src_idx not in self._all_geodesic_results:
                        self._all_geodesic_results[src_idx] = {}
                    self._all_geodesic_results[src_idx][method_key] = result

            # ---- Aggregate and display ----
            self._aggregate_geodesic_metrics()
            self._aggregate_greens_metrics()
            first_src = self._source_indices[0]
            if first_src in self._all_geodesic_results:
                self.current_heat_geodesic_results = self._all_geodesic_results[first_src]

            self._print_multisource_summary()

            # Visualization for first source (visual mode only)
            if not self._quantitative_mode and mesh_structure is not None:
                for method_key in list((self._all_geodesic_distances.get(first_src) or {}).keys()):
                    distances = self._all_geodesic_distances[first_src][method_key]
                    exact_distances = self._exact_geodesics.get(first_src)
                    if distances is not None and exact_distances is not None:
                        d_norm = normalize_distances(distances)
                        is_bad = distances.max() > 1e3
                        label_suffix = " [!]" if is_bad else ""
                        cmap = 'hot' if is_bad else 'viridis'
                        mesh_structure.add_scalar_quantity(
                            name=f"K1 Geodesic (norm) - {method_key}{label_suffix}",
                            values=d_norm, enabled=False, cmap=cmap
                        )
                        exact_norm = normalize_distances(exact_distances)
                        error = np.abs(d_norm - exact_norm)
                        mesh_structure.add_scalar_quantity(
                            name=f"K2 Geodesic Error - {method_key}",
                            values=error, enabled=False, cmap='hot'
                        )

        except Exception as e:
            print(f"  [!] Heat Method geodesic validation failed: {e}")
            import traceback
            traceback.print_exc()

        # STEP 9.6: Probe function MSE (Lf error, NeLo paper Table 1 metric)
        _mark_step("step9_6_probe")
        print(f"\nSTEP 9.6: Probe function MSE (Lf error)")
        try:
            self._compute_probe_function_mse()
        except Exception as e:
            print(f"  [!] Probe function MSE failed: {e}")
            import traceback
            traceback.print_exc()

        # STEP 9.7: HKS / WKS descriptor comparison
        # Pre-compute Weyl calibration factor (cheap, always available for checkbox toggle)
        try:
            self._compute_weyl_calibration()
        except Exception as e:
            print(f"  [!] Weyl calibration failed: {e}")

        _mark_step("step9_7_descriptors")
        print(f"\nSTEP 9.7: Shape descriptor comparison (HKS / WKS)")
        try:
            self._compute_descriptor_comparison()
        except Exception as e:
            print(f"  [!] Descriptor comparison failed: {e}")
            import traceback
            traceback.print_exc()

        # STEP 9.8: Spectral mesh compression error
        _mark_step("step9_8_compression")
        print(f"\nSTEP 9.8: Spectral compression error")
        try:
            self._compute_spectral_compression_error()
        except Exception as e:
            print(f"  [!] Spectral compression error failed: {e}")
            import traceback
            traceback.print_exc()

        # Print timing summary to console
        self._print_timing_summary()

        # Print step-level timing breakdown
        _mark_step("done")
        print(f"\n{'=' * 60}")
        print(f"STEP TIMING BREAKDOWN")
        print(f"{'=' * 60}")
        total = sum(_step_times.values())
        for step_name, elapsed in _step_times.items():
            if step_name == "init":
                continue
            pct = (elapsed / total * 100) if total > 0 else 0
            print(f"  {step_name:<30s} {elapsed:>8.1f}s  ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<30s} {total:>8.1f}s")
        print(f"{'=' * 60}")

        print(f"\n[OK] Comprehensive visualization completed for {Path(mesh_file_path).name}")
        return True

    def _collect_mesh_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Collect all computed metrics for the current mesh into a flat dictionary.

        Called after process_batch in quantitative mode. Extracts metrics from
        the instance state that process_batch populated.

        Returns:
            Dictionary of metric_name -> value, or None if no data available.
        """
        if self.current_gt_data is None or self.current_inference_result is None:
            return None

        mesh_name = Path(self.current_mesh_file_path).name if self.current_mesh_file_path else "unknown"
        n_verts = len(self.current_gt_data['vertices'])
        n_faces = len(self.current_gt_data['faces'])

        metrics: Dict[str, Any] = {
            'mesh_name': mesh_name,
            'num_vertices': n_verts,
            'num_faces': n_faces,
            'k': self.original_k or 0,
        }

        # --- Timing ---
        t = self.timing_results
        metrics['pred_patch_extraction_ms'] = t.pred_patch_extraction_time * 1000
        metrics['pred_inference_ms'] = t.pred_model_inference_time * 1000
        metrics['pred_assembly_ms'] = t.pred_matrix_assembly_time * 1000
        metrics['pred_total_ms'] = t.pred_total_time * 1000
        metrics['robust_total_ms'] = t.robust_matrix_assembly_time * 1000
        metrics['gt_assembly_ms'] = t.gt_matrix_assembly_time * 1000
        if t.nelo_total_time > 0:
            metrics['nelo_graph_tree_ms'] = t.nelo_graph_tree_time * 1000
            metrics['nelo_inference_ms'] = t.nelo_model_inference_time * 1000
            metrics['nelo_assembly_ms'] = t.nelo_matrix_assembly_time * 1000
            metrics['nelo_total_ms'] = t.nelo_total_time * 1000

        # --- Downstream solve timings (per-source average) ---
        n_src = len(self._source_indices) if self._source_indices else 1
        if t.gt_geodesic_solve_time > 0:
            metrics['gt_geodesic_solve_avg_ms'] = t.gt_geodesic_solve_time / n_src * 1000
        if t.pred_geodesic_solve_time > 0:
            metrics['pred_geodesic_solve_avg_ms'] = t.pred_geodesic_solve_time / n_src * 1000
        if t.robust_geodesic_solve_time > 0:
            metrics['robust_geodesic_solve_avg_ms'] = t.robust_geodesic_solve_time / n_src * 1000
        if t.nelo_geodesic_solve_time > 0:
            metrics['nelo_geodesic_solve_avg_ms'] = t.nelo_geodesic_solve_time / n_src * 1000

        if t.gt_greens_solve_time > 0:
            metrics['gt_greens_solve_avg_ms'] = t.gt_greens_solve_time / n_src * 1000
        if t.pred_greens_solve_time > 0:
            metrics['pred_greens_solve_avg_ms'] = t.pred_greens_solve_time / n_src * 1000
        if t.robust_greens_solve_time > 0:
            metrics['robust_greens_solve_avg_ms'] = t.robust_greens_solve_time / n_src * 1000
        if t.nelo_greens_solve_time > 0:
            metrics['nelo_greens_solve_avg_ms'] = t.nelo_greens_solve_time / n_src * 1000

        # --- E2E timing: assembly + per-source avg downstream solve ---
        if t.pred_geodesic_solve_time > 0:
            metrics['pred_e2e_geodesic_ms'] = t.pred_total_time * 1000 + t.pred_geodesic_solve_time / n_src * 1000
        if t.pred_greens_solve_time > 0:
            metrics['pred_e2e_greens_ms'] = t.pred_total_time * 1000 + t.pred_greens_solve_time / n_src * 1000

        if t.robust_geodesic_solve_time > 0:
            if self._robust_geodesic_heat:
                metrics['robust_e2e_geodesic_ms'] = t.robust_matrix_assembly_time * 1000 + t.robust_geodesic_solve_time / n_src * 1000
            else:
                metrics['robust_e2e_geodesic_ms'] = t.robust_geodesic_solve_time / n_src * 1000
        if t.robust_greens_solve_time > 0:
            metrics['robust_e2e_greens_ms'] = t.robust_matrix_assembly_time * 1000 + t.robust_greens_solve_time / n_src * 1000

        if t.nelo_total_time > 0 and t.nelo_geodesic_solve_time > 0:
            metrics['nelo_e2e_geodesic_ms'] = t.nelo_total_time * 1000 + t.nelo_geodesic_solve_time / n_src * 1000
        if t.nelo_total_time > 0 and t.nelo_greens_solve_time > 0:
            metrics['nelo_e2e_greens_ms'] = t.nelo_total_time * 1000 + t.nelo_greens_solve_time / n_src * 1000

        if t.gt_geodesic_solve_time > 0:
            metrics['gt_e2e_geodesic_ms'] = t.gt_matrix_assembly_time * 1000 + t.gt_geodesic_solve_time / n_src * 1000
        if t.gt_greens_solve_time > 0:
            metrics['gt_e2e_greens_ms'] = t.gt_matrix_assembly_time * 1000 + t.gt_greens_solve_time / n_src * 1000

        # --- Eigenvector cosine similarity ---
        for method_label, sims in [
            ('pred', self.current_cosine_similarities_pred),
            ('robust', self.current_cosine_similarities_robust),
            ('nelo', self.current_cosine_similarities_nelo),
        ]:
            if sims is not None and len(sims) > 0:
                # Cumulative means at key counts
                for count in [5, 10, 20, 50]:
                    if count <= len(sims):
                        metrics[f'{method_label}_eigvec_cos_top{count}'] = float(sims[:count].mean())
                # Overall mean
                metrics[f'{method_label}_eigvec_cos_all'] = float(sims.mean())
                # Per-eigenvector cosine similarities
                for i, val in enumerate(sims):
                    metrics[f'{method_label}_eigvec_cos_{i}'] = float(val)

        # --- Eigenvalue relative error ---
        # We use two metrics:
        #   1. Spectrum-relative error: |pred_i - gt_i| / max(gt), normalized by spectral range
        #   2. Per-eigenvalue relative error: |pred_i - gt_i| / gt_i (only for eigenvalues above a threshold)
        gt_evals = self.current_gt_data.get('gt_eigenvalues')
        if gt_evals is not None:
            for method_label, evals in [
                ('pred', self.current_inference_result.get('predicted_eigenvalues')),
                ('robust', self.current_gt_data.get('robust_eigenvalues')),
                ('nelo', self.current_nelo_eigenvalues),
            ]:
                if evals is not None:
                    n_compare = min(len(gt_evals), len(evals))
                    # Skip first eigenvalue (always ~0)
                    if n_compare > 1:
                        gt_slice = gt_evals[1:n_compare]
                        pred_slice = evals[1:n_compare]

                        # Spectrum-relative error: normalize by spectral range (max eigenvalue)
                        spectral_range = np.abs(gt_slice).max()
                        if spectral_range > 1e-10:
                            spectrum_rel_err = np.abs(pred_slice - gt_slice) / spectral_range
                            metrics[f'{method_label}_eval_spectrum_rel_err_mean'] = float(spectrum_rel_err.mean())
                            metrics[f'{method_label}_eval_spectrum_rel_err_max'] = float(spectrum_rel_err.max())

                        # Per-eigenvalue relative error (skip near-zero eigenvalues)
                        # Only use eigenvalues > 1% of max to avoid division instability
                        threshold = spectral_range * 0.01 if spectral_range > 0 else 1e-6
                        stable_mask = np.abs(gt_slice) > threshold
                        if stable_mask.sum() > 0:
                            rel_err = np.abs(pred_slice[stable_mask] - gt_slice[stable_mask]) / np.abs(gt_slice[stable_mask])
                            metrics[f'{method_label}_eval_rel_err_mean'] = float(rel_err.mean())
                            metrics[f'{method_label}_eval_rel_err_max'] = float(rel_err.max())

        # --- Green's function maximum principle (multi-source aggregated) ---
        if self._aggregated_greens_metrics:
            n_src = len(self._source_indices) if self._source_indices else 1
            metrics['num_validation_sources'] = n_src
            for method, agg in self._aggregated_greens_metrics.items():
                prefix = method.lower().replace(' ', '_').replace('(', '').replace(')', '')
                metrics[f'{prefix}_greens_max_principle_pass_rate'] = agg.greens_max_principle_pass_rate
                metrics[f'{prefix}_greens_monotonicity_mean'] = agg.greens_mono_mean
                metrics[f'{prefix}_greens_monotonicity_std'] = agg.greens_mono_std
                metrics[f'{prefix}_greens_gt_corr_mean'] = agg.greens_corr_with_gt_mean
                metrics[f'{prefix}_greens_gt_corr_std'] = agg.greens_corr_with_gt_std
        elif self.current_greens_results is not None:
            # Fallback: single source
            metrics['num_validation_sources'] = 1
            for method_key, result in self.current_greens_results.items():
                prefix = method_key.lower().replace(' ', '_').replace('(', '').replace(')', '')
                metrics[f'{prefix}_greens_max_principle_pass_rate'] = 1.0 if result.satisfies_maximum_principle else 0.0
                metrics[f'{prefix}_greens_monotonicity_mean'] = result.monotonicity_score
                metrics[f'{prefix}_greens_gt_corr_mean'] = result.correlation_with_gt

        # --- Heat method geodesics (multi-source aggregated) ---
        if self._aggregated_geodesic_metrics:
            for method, agg in self._aggregated_geodesic_metrics.items():
                prefix = method.lower().replace(' ', '_').replace('(', '').replace(')', '')
                metrics[f'{prefix}_geodesic_corr_mean'] = agg.corr_mean
                metrics[f'{prefix}_geodesic_corr_std'] = agg.corr_std
                metrics[f'{prefix}_geodesic_mae_mean'] = agg.mae_mean
                metrics[f'{prefix}_geodesic_mae_std'] = agg.mae_std
                metrics[f'{prefix}_geodesic_max_err_mean'] = agg.max_err_mean
                metrics[f'{prefix}_geodesic_max_err_std'] = agg.max_err_std
                metrics[f'{prefix}_geodesic_monotonicity_mean'] = agg.mono_mean
                metrics[f'{prefix}_geodesic_monotonicity_std'] = agg.mono_std
        elif self.current_heat_geodesic_results is not None:
            # Fallback: single source
            for method_key, result in self.current_heat_geodesic_results.items():
                prefix = method_key.lower().replace(' ', '_').replace('(', '').replace(')', '')
                metrics[f'{prefix}_geodesic_corr_mean'] = result.correlation_with_reference
                metrics[f'{prefix}_geodesic_mae_mean'] = result.mean_absolute_error
                metrics[f'{prefix}_geodesic_max_err_mean'] = result.max_absolute_error
                metrics[f'{prefix}_geodesic_monotonicity_mean'] = result.monotonicity_score

        # --- Probe Function MSE (Lf error) ---
        for method, res in self._probe_function_results.items():
            prefix = method.lower().replace(' ', '_').replace('(', '').replace(')', '')
            metrics[f'{prefix}_probe_mse'] = res['mse']
            metrics[f'{prefix}_probe_cosine_sim'] = res.get('cosine_similarity', 0.0)
            metrics[f'{prefix}_probe_failure_rate'] = res['failure_rate']

        # --- HKS / WKS Descriptor Error ---
        for method, res in self._descriptor_results.items():
            prefix = method.lower().replace(' ', '_').replace('(', '').replace(')', '')
            metrics[f'{prefix}_hks_l2_error'] = res['hks_l2_error']
            metrics[f'{prefix}_hks_correlation'] = res['hks_correlation']
            metrics[f'{prefix}_wks_l2_error'] = res['wks_l2_error']
            metrics[f'{prefix}_wks_correlation'] = res['wks_correlation']

        # --- Weyl Calibration ---
        if self._weyl_calibration_factor is not None:
            metrics['pred_weyl_calibration_factor'] = self._weyl_calibration_factor
            metrics['pred_weyl_area_source'] = self.reconstruction_settings.weyl_area_source

        # --- Spectral Compression Error ---
        for method, k_res in self._compression_results.items():
            prefix = method.lower().replace(' ', '_').replace('(', '').replace(')', '')
            for k, vals in k_res.items():
                metrics[f'{prefix}_compression_mean_k{k}'] = vals['mean_l2']
                metrics[f'{prefix}_compression_max_k{k}'] = vals['max_l2']

        # --- Sparsity ---
        for method_key, stats in self.current_sparsity_stats.items():
            prefix = method_key.lower().replace(' ', '_').replace('(', '').replace(')', '')
            metrics[f'{prefix}_nnz'] = stats.nnz
            metrics[f'{prefix}_density_pct'] = stats.density_percent
            metrics[f'{prefix}_avg_nnz_per_row'] = stats.avg_nnz_per_row

        # --- Weight Distribution Stats ---
        if self._weight_distribution_stats:
            for key, val in self._weight_distribution_stats.items():
                metrics[f'pred_{key}'] = val

        # --- Mesh Boundary Info ---
        if self._mesh_boundary_info:
            for key, val in self._mesh_boundary_info.items():
                if isinstance(val, bool):
                    metrics[key] = int(val)
                else:
                    metrics[key] = val

        # --- Curvature Comparison (GT vs PRED) ---
        gt_H = self.current_gt_data.get('gt_mean_curvature') if self.current_gt_data else None
        gt_K = self.current_gt_data.get('gt_gaussian_curvature') if self.current_gt_data else None
        pred_H = self.current_predicted_data.get('predicted_mean_curvature') if self.current_predicted_data else None

        if gt_H is not None:
            metrics['gt_mean_curv_min'] = float(gt_H.min())
            metrics['gt_mean_curv_max'] = float(gt_H.max())
            metrics['gt_mean_curv_mean'] = float(np.mean(gt_H))
            metrics['gt_mean_curv_std'] = float(np.std(gt_H))
        if gt_K is not None:
            metrics['gt_gauss_curv_min'] = float(gt_K.min())
            metrics['gt_gauss_curv_max'] = float(gt_K.max())
            metrics['gt_gauss_curv_mean'] = float(np.mean(gt_K))
            metrics['gt_gauss_curv_std'] = float(np.std(gt_K))
        if pred_H is not None:
            metrics['pred_mean_curv_min'] = float(pred_H.min())
            metrics['pred_mean_curv_max'] = float(pred_H.max())
            metrics['pred_mean_curv_mean'] = float(np.mean(pred_H))
            metrics['pred_mean_curv_std'] = float(np.std(pred_H))
        # Curvature correlation (GT vs PRED)
        if gt_H is not None and pred_H is not None:
            valid = np.isfinite(gt_H) & np.isfinite(pred_H)
            if valid.sum() > 2:
                corr = np.corrcoef(np.abs(gt_H[valid]), pred_H[valid])[0, 1]
                metrics['mean_curv_magnitude_corr'] = float(corr) if np.isfinite(corr) else 0.0

        # --- Vertex Area Comparison (GT vs PRED) ---
        gt_areas = self.current_gt_data.get('vertex_areas') if self.current_gt_data else None
        pred_areas_raw = self.current_inference_result.get('areas') if self.current_inference_result else None

        if gt_areas is not None:
            metrics['gt_area_total'] = float(np.sum(gt_areas))
            metrics['gt_area_mean'] = float(np.mean(gt_areas))
            metrics['gt_area_std'] = float(np.std(gt_areas))
            metrics['gt_area_min'] = float(np.min(gt_areas))
            metrics['gt_area_max'] = float(np.max(gt_areas))

        if pred_areas_raw is not None:
            pred_areas_np = pred_areas_raw.cpu().numpy() if torch.is_tensor(pred_areas_raw) else np.asarray(pred_areas_raw)
            metrics['pred_area_total'] = float(np.sum(pred_areas_np))
            metrics['pred_area_mean'] = float(np.mean(pred_areas_np))
            metrics['pred_area_std'] = float(np.std(pred_areas_np))
            metrics['pred_area_min'] = float(np.min(pred_areas_np))
            metrics['pred_area_max'] = float(np.max(pred_areas_np))

            # Area ratio (PRED total / GT total)
            if gt_areas is not None:
                gt_total = float(np.sum(gt_areas))
                if gt_total > 1e-15:
                    metrics['area_total_ratio_pred_gt'] = float(np.sum(pred_areas_np)) / gt_total
                # Per-vertex area correlation
                if len(pred_areas_np) == len(gt_areas):
                    corr = np.corrcoef(pred_areas_np, gt_areas)[0, 1]
                    metrics['area_correlation'] = float(corr) if np.isfinite(corr) else 0.0

        # --- Raw Eigenvalue Stats ---
        gt_evals = self.current_gt_data.get('gt_eigenvalues') if self.current_gt_data else None
        pred_evals = self.current_inference_result.get('predicted_eigenvalues') if self.current_inference_result else None
        robust_evals = self.current_gt_data.get('robust_eigenvalues') if self.current_gt_data else None
        nelo_evals = self.current_nelo_eigenvalues

        for label, evals in [('gt', gt_evals), ('pred', pred_evals), ('robust', robust_evals), ('nelo', nelo_evals)]:
            if evals is not None:
                evals_np = evals.cpu().numpy() if torch.is_tensor(evals) else np.asarray(evals)
                if len(evals_np) > 1:
                    metrics[f'{label}_eval_0'] = float(evals_np[0])
                    metrics[f'{label}_eval_1'] = float(evals_np[1])  # Fiedler
                    metrics[f'{label}_eval_max'] = float(evals_np[-1])
                    metrics[f'{label}_eval_median'] = float(np.median(evals_np[1:]))
                    # Spectral gap
                    metrics[f'{label}_spectral_gap'] = float(evals_np[1] - evals_np[0])

        # Eigenvalue scale ratio (PRED / GT)
        if gt_evals is not None and pred_evals is not None:
            gt_np = gt_evals.cpu().numpy() if torch.is_tensor(gt_evals) else np.asarray(gt_evals)
            pred_np = pred_evals.cpu().numpy() if torch.is_tensor(pred_evals) else np.asarray(pred_evals)
            if len(gt_np) > 1 and len(pred_np) > 1:
                # Median ratio of positive eigenvalues (skip λ_0)
                gt_pos = gt_np[1:]
                pred_pos = pred_np[1:]
                valid = (gt_pos > 1e-10) & (pred_pos > 1e-10)
                if valid.sum() > 0:
                    ratios = pred_pos[valid] / gt_pos[valid]
                    metrics['eval_scale_ratio_median'] = float(np.median(ratios))
                    metrics['eval_scale_ratio_mean'] = float(np.mean(ratios))

        return metrics

    def _aggregate_and_save_results(self, all_metrics: List[Dict[str, Any]], csv_path: Path):
        """
        Aggregate per-mesh metrics, print summary table, and write CSV.

        Args:
            all_metrics: List of per-mesh metric dictionaries from _collect_mesh_metrics.
            csv_path: Path to write the CSV file.
        """
        import csv

        if not all_metrics:
            print("[!] No metrics to aggregate")
            return

        n_meshes = len(all_metrics)
        print(f"\n{'=' * 100}")
        print(f"QUANTITATIVE RESULTS SUMMARY ({n_meshes} meshes)")
        print(f"{'=' * 100}")

        # Collect all numeric keys (skip mesh_name)
        numeric_keys = []
        for key in all_metrics[0].keys():
            if key == 'mesh_name':
                continue
            # Check if this key has numeric values in at least one mesh
            values = [m.get(key) for m in all_metrics if m.get(key) is not None]
            if len(values) > 0 and isinstance(values[0], (int, float, np.integer, np.floating)):
                numeric_keys.append(key)

        # Compute aggregated stats
        agg_stats: Dict[str, Dict[str, float]] = {}
        for key in numeric_keys:
            values = np.array([m[key] for m in all_metrics if key in m and m[key] is not None])
            if len(values) == 0:
                continue
            agg_stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'count': len(values),
            }

        # Print summary table grouped by category
        def print_section(title: str, key_prefix: str):
            """Print a section of the summary table for keys matching prefix."""
            matching = [(k, v) for k, v in agg_stats.items() if k.startswith(key_prefix)]
            if not matching:
                return
            print(f"\n  {title}:")
            print(f"  {'Metric':<45} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10} {'N':>4}")
            print(f"  {'-' * 99}")
            for key, stats in matching:
                print(f"  {key:<45} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                      f"{stats['min']:>10.4f} {stats['max']:>10.4f} {stats['median']:>10.4f} {stats['count']:>4}")

        def print_section_filter(title: str, filter_fn):
            """Print a section of the summary table for keys matching filter function."""
            matching = [(k, v) for k, v in agg_stats.items() if filter_fn(k)]
            if not matching:
                return
            print(f"\n  {title}:")
            print(f"  {'Metric':<45} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10} {'N':>4}")
            print(f"  {'-' * 99}")
            for key, stats in matching:
                print(f"  {key:<45} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                      f"{stats['min']:>10.4f} {stats['max']:>10.4f} {stats['median']:>10.4f} {stats['count']:>4}")

        # Mesh info
        print_section("Mesh Info", "num_")
        print_section("Mesh Info", "k")

        # Timing
        print_section_filter("Timing (ms)",
            lambda k: k.endswith('_ms'))

        # Eigenvector alignment
        for method in ['pred', 'robust', 'nelo']:
            print_section(f"Eigenvector Cosine Similarity ({method.upper()})", f"{method}_eigvec")

        # Eigenvalue errors
        for method in ['pred', 'robust', 'nelo']:
            print_section(f"Eigenvalue Relative Error ({method.upper()})", f"{method}_eval")

        # Green's function
        for method in ['gt', 'pred', 'robust', 'nelo']:
            print_section(f"Green's Function ({method.upper()})", f"{method}_greens")

        # Geodesics
        for method in ['gt', 'pred', 'robust', 'nelo']:
            print_section(f"Geodesics ({method.upper()})", f"{method}_geodesic")

        # Probe functions (Lf error)
        for method in ['gt', 'pred', 'robust', 'nelo']:
            print_section(f"Probe Functions ({method.upper()})", f"{method}_probe")

        # HKS / WKS Descriptors
        for method in ['gt', 'pred', 'robust', 'nelo']:
            print_section_filter(f"Shape Descriptors ({method.upper()})",
                lambda k, m=method: k.startswith(f"{m}_hks") or k.startswith(f"{m}_wks"))

        # Spectral Compression
        for method in ['gt', 'pred', 'robust', 'nelo']:
            print_section(f"Spectral Compression ({method.upper()})", f"{method}_compression")

        # Sparsity
        print_section_filter("Sparsity",
            lambda k: k.endswith('_nnz') or k.endswith('_density_pct') or k.endswith('_avg_nnz_per_row'))

        # Vertex Areas
        print_section_filter("Vertex Areas",
            lambda k: '_area_' in k or k.startswith('area_'))

        # Curvature
        print_section_filter("Curvature",
            lambda k: '_curv_' in k or k.startswith('mean_curv_'))

        # Weyl calibration
        print_section("Weyl Calibration", "pred_weyl")

        # Eigenvalue scale
        print_section("Eigenvalue Scale", "eval_scale")

        print(f"\n{'=' * 100}")

        # --- Write per-mesh CSV ---
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Gather all possible columns across all meshes
        all_keys = []
        seen = set()
        for m in all_metrics:
            for k in m.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        # Write per-mesh rows
        per_mesh_path = csv_path
        with open(per_mesh_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(m)

        print(f"\n[OK] Per-mesh CSV written to: {per_mesh_path}")

        # Write aggregated summary CSV
        summary_path = csv_path.with_name(csv_path.stem + '_summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'mean', 'std', 'min', 'max', 'median', 'count'])
            for key, stats in agg_stats.items():
                writer.writerow([
                    key, stats['mean'], stats['std'],
                    stats['min'], stats['max'], stats['median'], stats['count']
                ])

        print(f"[OK] Summary CSV written to: {summary_path}")

        # --- Per-eigenvector mean cosine similarity table ---
        self._print_per_eigenvector_table(all_metrics, agg_stats)

        # --- Generate comparison charts ---
        charts_dir = csv_path.parent / 'charts'
        charts_dir.mkdir(parents=True, exist_ok=True)
        self._generate_comparison_charts(all_metrics, agg_stats, charts_dir)

    def _print_per_eigenvector_table(self, all_metrics: List[Dict[str, Any]],
                                      agg_stats: Dict[str, Dict[str, float]]):
        """Print per-eigenvector mean cosine similarity table to console."""
        methods = ['pred', 'robust', 'nelo']
        method_labels = {'pred': 'PRED', 'robust': 'Robust', 'nelo': 'NeLo'}

        # Find max eigenvector index across all methods
        max_idx = 0
        for m in methods:
            for key in agg_stats:
                if key.startswith(f'{m}_eigvec_cos_') and not any(
                    key.endswith(s) for s in ['all', 'top5', 'top10', 'top20', 'top50']
                ):
                    try:
                        idx = int(key.split('_')[-1])
                        max_idx = max(max_idx, idx)
                    except ValueError:
                        pass

        if max_idx == 0:
            print("\n  No per-eigenvector data available")
            return

        print(f"\n  Per-Eigenvector Mean |cos| Similarity (averaged over {len(all_metrics)} meshes):")
        header = f"  {'Eigvec':<8}"
        for m in methods:
            label = method_labels.get(m, m)
            header += f"  {label:>10}"
        print(header)
        print(f"  {'-' * (8 + 12 * len(methods))}")

        for i in range(max_idx + 1):
            row = f"  {i:<8}"
            for m in methods:
                key = f'{m}_eigvec_cos_{i}'
                if key in agg_stats:
                    row += f"  {agg_stats[key]['mean']:>10.4f}"
                else:
                    row += f"  {'N/A':>10}"
            print(row)

        # Overall means
        print(f"  {'-' * (8 + 12 * len(methods))}")
        row = f"  {'MEAN':<8}"
        for m in methods:
            key = f'{m}_eigvec_cos_all'
            if key in agg_stats:
                row += f"  {agg_stats[key]['mean']:>10.4f}"
            else:
                row += f"  {'N/A':>10}"
        print(row)

    def _generate_comparison_charts(self, all_metrics: List[Dict[str, Any]],
                                     agg_stats: Dict[str, Dict[str, float]],
                                     charts_dir: Path):
        """Generate comparison charts and save to charts_dir."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("[!] matplotlib not available, skipping chart generation")
            return

        methods = ['pred', 'robust', 'nelo']
        method_labels = {'pred': 'PRED', 'robust': 'Robust', 'nelo': 'NeLo'}
        method_colors = {'pred': '#E07020', 'robust': '#2090D0', 'nelo': '#9040E0'}

        n_meshes = len(all_metrics)

        # =====================================================================
        # Chart 1: Per-eigenvector mean cosine similarity (line plot)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 6))

        for m in methods:
            # Collect per-eigenvector means
            indices = []
            means = []
            stds = []
            for key, stats in sorted(agg_stats.items()):
                if key.startswith(f'{m}_eigvec_cos_') and not any(
                    key.endswith(s) for s in ['all', 'top5', 'top10', 'top20', 'top50']
                ):
                    try:
                        idx = int(key.split('_')[-1])
                        indices.append(idx)
                        means.append(stats['mean'])
                        stds.append(stats['std'])
                    except ValueError:
                        pass

            if indices:
                indices = np.array(indices)
                means = np.array(means)
                stds = np.array(stds)
                order = np.argsort(indices)
                indices, means, stds = indices[order], means[order], stds[order]

                label = method_labels.get(m, m)
                color = method_colors.get(m, None)
                ax.plot(indices, means, label=label, color=color, linewidth=2)
                ax.fill_between(indices, means - stds, np.minimum(means + stds, 1.0),
                                alpha=0.15, color=color)

        ax.set_xlabel('Eigenvector Index', fontsize=12)
        ax.set_ylabel('Mean |cos| Similarity', fontsize=12)
        ax.set_title(f'Per-Eigenvector Cosine Similarity (n={n_meshes} meshes)', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        chart_path = charts_dir / 'eigenvector_cosine_similarity.png'
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Chart saved: {chart_path}")

        # =====================================================================
        # Chart 2: Cumulative mean cosine similarity (line plot)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 6))

        for m in methods:
            # Build cumulative mean from per-eigenvector data
            per_eig = {}
            for key, stats in agg_stats.items():
                if key.startswith(f'{m}_eigvec_cos_') and not any(
                    key.endswith(s) for s in ['all', 'top5', 'top10', 'top20', 'top50']
                ):
                    try:
                        idx = int(key.split('_')[-1])
                        per_eig[idx] = stats['mean']
                    except ValueError:
                        pass

            if per_eig:
                max_idx = max(per_eig.keys())
                vals = [per_eig.get(i, 0.0) for i in range(max_idx + 1)]
                cumulative_means = np.cumsum(vals) / np.arange(1, max_idx + 2)

                label = method_labels.get(m, m)
                color = method_colors.get(m, None)
                ax.plot(range(max_idx + 1), cumulative_means, label=label,
                        color=color, linewidth=2)

        ax.set_xlabel('Number of Eigenvectors (top-k)', fontsize=12)
        ax.set_ylabel('Cumulative Mean |cos| Similarity', fontsize=12)
        ax.set_title(f'Cumulative Eigenvector Alignment (n={n_meshes} meshes)', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        chart_path = charts_dir / 'cumulative_cosine_similarity.png'
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Chart saved: {chart_path}")

        # =====================================================================
        # Chart 3: Geodesic quality comparison (grouped bar chart)
        # =====================================================================
        geodesic_metrics = ['geodesic_corr', 'geodesic_mae', 'geodesic_monotonicity']
        geodesic_labels = ['Correlation', 'MAE', 'Monotonicity']
        geo_methods = ['gt', 'pred', 'nelo', 'robust']
        geo_colors = {'gt': '#30B030', 'pred': '#E07020', 'nelo': '#9040E0', 'robust': '#2090D0'}

        fig, axes = plt.subplots(1, len(geodesic_metrics), figsize=(5 * len(geodesic_metrics), 6))
        if len(geodesic_metrics) == 1:
            axes = [axes]

        for ax_idx, (metric_suffix, metric_label) in enumerate(zip(geodesic_metrics, geodesic_labels)):
            ax = axes[ax_idx]
            bar_vals = []
            bar_errs = []
            bar_labels = []
            bar_colors_list = []

            for m in geo_methods:
                # Try both key patterns for Robust (pp3d vs heat mode)
                key = f'{m}_{metric_suffix}'
                if key not in agg_stats and m == 'robust':
                    key = f'robust_pp3d_{metric_suffix}'
                if key in agg_stats:
                    bar_vals.append(agg_stats[key]['mean'])
                    bar_errs.append(agg_stats[key]['std'])
                    bar_labels.append(method_labels.get(m, m.upper()))
                    bar_colors_list.append(geo_colors.get(m, '#888888'))

            if bar_vals:
                x = np.arange(len(bar_vals))
                ax.bar(x, bar_vals, yerr=bar_errs, capsize=4,
                       color=bar_colors_list, alpha=0.85, edgecolor='black', linewidth=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(bar_labels, fontsize=10)
                ax.set_title(metric_label, fontsize=12)
                ax.set_ylim(bottom=0)
                ax.grid(True, axis='y', alpha=0.3)

        fig.suptitle(f'Geodesic Quality Metrics (n={n_meshes} meshes)', fontsize=14)
        fig.tight_layout()
        chart_path = charts_dir / 'geodesic_comparison.png'
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Chart saved: {chart_path}")

        # =====================================================================
        # Chart 4: Eigenvalue relative error comparison (bar chart)
        # =====================================================================
        # Chart 4: Eigenvalue error comparison (bar chart — two metrics)
        # =====================================================================
        eval_metric_pairs = [
            ('_eval_spectrum_rel_err_mean', 'Spectrum-Relative Error\n(normalized by λ_max)'),
            ('_eval_rel_err_mean', 'Per-Eigenvalue Relative Error\n(stable eigenvalues only)'),
        ]

        fig, axes = plt.subplots(1, len(eval_metric_pairs), figsize=(6 * len(eval_metric_pairs), 6))
        if len(eval_metric_pairs) == 1:
            axes = [axes]

        for ax_idx, (metric_suffix, metric_title) in enumerate(eval_metric_pairs):
            ax = axes[ax_idx]
            bar_vals = []
            bar_errs = []
            bar_labels = []
            bar_colors_list = []

            for m in methods:
                key = f'{m}{metric_suffix}'
                if key in agg_stats:
                    mean_val = agg_stats[key]['mean']
                    std_val = agg_stats[key]['std']
                    bar_vals.append(mean_val)
                    # Clamp lower error bar so bar doesn't go negative
                    bar_errs.append(min(std_val, mean_val))
                    bar_labels.append(method_labels.get(m, m))
                    bar_colors_list.append(method_colors.get(m, '#888888'))

            if bar_vals:
                x = np.arange(len(bar_vals))
                ax.bar(x, bar_vals, yerr=bar_errs, capsize=4,
                       color=bar_colors_list, alpha=0.85, edgecolor='black', linewidth=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(bar_labels, fontsize=11)
                ax.set_title(metric_title, fontsize=11)
                ax.set_ylim(bottom=0)
                ax.grid(True, axis='y', alpha=0.3)

        fig.suptitle(f'Eigenvalue Relative Error (n={n_meshes} meshes)', fontsize=14)
        fig.tight_layout()
        chart_path = charts_dir / 'eigenvalue_error.png'
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Chart saved: {chart_path}")

        # =====================================================================
        # Chart 5: Timing comparison (bar chart)
        # =====================================================================
        timing_keys = {
            'pred': 'pred_total_ms',
            'robust': 'robust_total_ms',
            'nelo': 'nelo_total_ms',
            'gt': 'gt_assembly_ms',
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        bar_vals = []
        bar_errs = []
        bar_labels = []
        bar_colors_list = []

        for m, key in timing_keys.items():
            if key in agg_stats:
                bar_vals.append(agg_stats[key]['mean'])
                bar_errs.append(agg_stats[key]['std'])
                bar_labels.append(method_labels.get(m, m.upper()))
                bar_colors_list.append(
                    {'gt': '#30B030', 'pred': '#E07020', 'nelo': '#9040E0', 'robust': '#2090D0'}.get(m, '#888888')
                )

        if bar_vals:
            x = np.arange(len(bar_vals))
            ax.bar(x, bar_vals, yerr=bar_errs, capsize=4,
                   color=bar_colors_list, alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, fontsize=11)
            ax.set_ylabel('Time (ms)', fontsize=12)
            ax.set_title(f'Laplacian Assembly Timing (n={n_meshes} meshes)', fontsize=14)
            ax.set_ylim(bottom=0)
            ax.grid(True, axis='y', alpha=0.3)
            fig.tight_layout()
            chart_path = charts_dir / 'timing_comparison.png'
            fig.savefig(chart_path, dpi=150)
            print(f"[OK] Chart saved: {chart_path}")
        plt.close(fig)

        # =====================================================================
        # Chart 6: Green's function metrics (grouped bar chart)
        # =====================================================================
        greens_metrics = ['greens_monotonicity', 'greens_distance_corr']
        greens_labels = ['Monotonicity', 'Distance Correlation']
        greens_method_map = {
            'gt': 'gt',
            'pred': 'pred',
            'robust': 'robust',
            'nelo': 'nelo',
        }

        fig, axes = plt.subplots(1, len(greens_metrics), figsize=(5 * len(greens_metrics), 6))
        if len(greens_metrics) == 1:
            axes = [axes]

        for ax_idx, (metric_suffix, metric_label) in enumerate(zip(greens_metrics, greens_labels)):
            ax = axes[ax_idx]
            bar_vals = []
            bar_errs = []
            bar_labels = []
            bar_colors_list = []

            for m in geo_methods:
                g_key = greens_method_map.get(m, m)
                key = f'{g_key}_{metric_suffix}'
                if key in agg_stats:
                    bar_vals.append(agg_stats[key]['mean'])
                    bar_errs.append(agg_stats[key]['std'])
                    bar_labels.append(method_labels.get(m, m.upper()))
                    bar_colors_list.append(geo_colors.get(m, '#888888'))

            if bar_vals:
                x = np.arange(len(bar_vals))
                ax.bar(x, bar_vals, yerr=bar_errs, capsize=4,
                       color=bar_colors_list, alpha=0.85, edgecolor='black', linewidth=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(bar_labels, fontsize=10)
                ax.set_title(metric_label, fontsize=12)
                ax.set_ylim(bottom=0)
                ax.grid(True, axis='y', alpha=0.3)

        fig.suptitle(f"Green's Function Metrics (n={n_meshes} meshes)", fontsize=14)
        fig.tight_layout()
        chart_path = charts_dir / 'greens_function_comparison.png'
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Chart saved: {chart_path}")

        # =====================================================================
        # Chart 7: E2E timing comparison (assembly + downstream solve)
        # =====================================================================
        e2e_tasks = [
            ('e2e_geodesic_ms', "E2E Geodesic\n(assembly + heat method)"),
            ('e2e_greens_ms', "E2E Green's Function\n(assembly + solve)"),
        ]
        all_methods_e2e = ['gt', 'pred', 'nelo', 'robust']

        fig, axes = plt.subplots(1, len(e2e_tasks), figsize=(6 * len(e2e_tasks), 6))
        if len(e2e_tasks) == 1:
            axes = [axes]

        for ax_idx, (metric_suffix, metric_title) in enumerate(e2e_tasks):
            ax = axes[ax_idx]
            bar_vals = []
            bar_errs = []
            bar_labels = []
            bar_colors_list = []

            for m in all_methods_e2e:
                key = f'{m}_{metric_suffix}'
                if key in agg_stats:
                    bar_vals.append(agg_stats[key]['mean'])
                    bar_errs.append(agg_stats[key]['std'])
                    bar_labels.append(method_labels.get(m, m.upper()))
                    bar_colors_list.append(geo_colors.get(m, '#888888'))

            if bar_vals:
                x = np.arange(len(bar_vals))
                ax.bar(x, bar_vals, yerr=bar_errs, capsize=4,
                       color=bar_colors_list, alpha=0.85, edgecolor='black', linewidth=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(bar_labels, fontsize=11)
                ax.set_ylabel('Time (ms)', fontsize=12)
                ax.set_title(metric_title, fontsize=11)
                ax.set_ylim(bottom=0)
                ax.grid(True, axis='y', alpha=0.3)

        fig.suptitle(f'End-to-End Timing: Assembly + Downstream Solve (n={n_meshes} meshes)', fontsize=14)
        fig.tight_layout()
        chart_path = charts_dir / 'e2e_timing_comparison.png'
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Chart saved: {chart_path}")

    def run_dataset_iteration(self, cfg: DictConfig):
        """Run visualization on all meshes in the dataset."""
        print(f"\n{'=' * 80}")
        print("REAL-TIME EIGENANALYSIS VISUALIZATION")
        print("=" * 80)

        # Get and validate checkpoint path from config
        if not hasattr(cfg, 'ckpt_path') or cfg.ckpt_path is None:
            raise ValueError("ckpt_path parameter is required. Please specify the path to the model checkpoint.")

        ckpt_path = Path(cfg.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        print(f"Checkpoint: {ckpt_path}")
        print("=" * 80)

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Check for diagnostic mode (disable optimizations to debug timing issues)
        diagnostic_mode = getattr(cfg, 'diagnostic_mode', False)
        skip_robust = getattr(cfg, 'skip_robust', False)  # Debug: skip robust computation
        quantitative_mode = getattr(cfg, 'quantitative_mode', False)  # Headless metrics-only mode
        robust_geodesic_heat = getattr(cfg, 'robust_geodesic_heat', False)  # Use heat method for Robust geodesics

        if diagnostic_mode:
            print("\n" + "!" * 80)
            print("DIAGNOSTIC MODE ENABLED - Optimizations disabled for debugging")
            print("!" * 80 + "\n")

        if skip_robust:
            print("[DEBUG] skip_robust=True: Robust-laplacian computation will be skipped")

        if robust_geodesic_heat:
            print("[INFO] robust_geodesic_heat=true: Robust geodesics will use heat method with matched pcdiff operators")

        if quantitative_mode:
            print("\n" + "=" * 80)
            print("QUANTITATIVE MODE — Polyscope disabled, collecting metrics for CSV output")
            print("=" * 80 + "\n")

        # Store flags for use in process_batch
        self._diagnostic_mode = diagnostic_mode
        self._skip_robust = skip_robust
        self._quantitative_mode = quantitative_mode
        self._robust_geodesic_heat = robust_geodesic_heat
        self._verbose = getattr(cfg, 'verbose', True)  # +verbose=False to suppress per-mesh output

        # Load trained model
        model = self.load_trained_model(ckpt_path, device, cfg)

        # Load NeLo pipeline if checkpoint path provided
        nelo_ckpt = getattr(cfg, 'nelo_ckpt_path', None)
        if nelo_ckpt is not None:
            print(f"\nLoading NeLo pipeline from: {nelo_ckpt}")
            try:
                self.load_nelo_pipeline(str(nelo_ckpt), device=str(device))
                print("[OK] NeLo pipeline ready")
            except Exception as e:
                print(f"[!] Failed to load NeLo pipeline: {e} — NeLo will be skipped")
        else:
            print("\n[INFO] nelo_ckpt_path not set — NeLo comparison will be skipped")

        # Read per-method kNN sizes from globals config (fallback to globals.k)
        default_k = getattr(cfg.globals, 'k', 30)
        self._initial_k_pred = getattr(cfg.globals, 'k_pred', None) or default_k
        self._initial_k_robust = getattr(cfg.globals, 'k_robust', None) or default_k
        self._initial_k_nelo = getattr(cfg.globals, 'k_nelo', None) or default_k
        self.nelo_k = self._initial_k_nelo
        print(f"Per-method kNN: PRED={self._initial_k_pred}, Robust={self._initial_k_robust}, NeLo={self._initial_k_nelo}")

        # Set random seed for reproducibility
        pl.seed_everything(cfg.globals.seed)

        # Initialize data module and loader BEFORE warmup so we can get actual k
        data_module = hydra.utils.instantiate(cfg.data_module)
        data_loader = data_module.val_dataloader()

        # Handle case where val_dataloader returns a list of dataloaders
        if isinstance(data_loader, list):
            data_loader = data_loader[0]  # Take the first validation dataloader

        # Peek at first batch to get actual k value for warmup
        first_batch = next(iter(data_loader))
        if isinstance(first_batch, list):
            first_data = first_batch[0]
        else:
            first_data = first_batch
        actual_k = int(first_data.k_neighbors) if hasattr(first_data, 'k_neighbors') else 30
        print(f"Detected k={actual_k} from dataset")

        # CUDA warmup: forward pass + assembly + gradient op (eliminates first-mesh cold start)
        self._warmup_cuda(model, device, k=self._initial_k_pred)

        # Re-create dataloader since we consumed it (reset iterator)
        # Also reset seed to ensure same data order
        pl.seed_everything(cfg.globals.seed)
        data_loader = data_module.val_dataloader()
        if isinstance(data_loader, list):
            data_loader = data_loader[0]

        print(f"DataLoader ready with batch size: {data_loader.batch_size}")

        # Setup polyscope with UI callback (only if NOT in quantitative mode)
        if not quantitative_mode:
            self.setup_polyscope()
        else:
            print("[quantitative_mode] Skipping polyscope setup")

        # Collect per-mesh metrics when in quantitative mode
        all_mesh_metrics: List[Dict[str, Any]] = []

        # Process all batches
        for batch_idx, batch_data in enumerate(data_loader):
            print(f"\n[>] Processing batch {batch_idx + 1}")

            try:
                success = self.process_batch(model, batch_data, batch_idx, device)

                if not quantitative_mode:
                    if success:
                        print(f"\nVisualization ready for batch {batch_idx + 1}. Use the 'Reconstruction Settings' window to control PRED reconstruction method.")
                        print("Close window to continue to next batch.")
                        ps.show()

                        # Free GPU memory held by polyscope before next batch
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                else:
                    if success:
                        # Collect metrics for this mesh
                        mesh_metrics = self._collect_mesh_metrics()
                        if mesh_metrics is not None:
                            all_mesh_metrics.append(mesh_metrics)
                        print(f"\n[quantitative_mode] Collected metrics for batch {batch_idx + 1}")
                    else:
                        print(f"\n[quantitative_mode] Skipped batch {batch_idx + 1} (processing failed)")

            except Exception as e:
                print(f"[X] Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()

                if not quantitative_mode:
                    user_input = input("Continue to next batch? (y/n): ").strip().lower()
                    if user_input != 'y':
                        break
                # In quantitative mode, always continue to next batch

        # Aggregate and save results in quantitative mode
        if quantitative_mode and len(all_mesh_metrics) > 0:
            output_dir = getattr(cfg, 'output_dir', '.')
            csv_path = Path(output_dir) / 'quantitative_results.csv'
            self._aggregate_and_save_results(all_mesh_metrics, csv_path)

        print(f"\n[OK] Completed processing all batches!")


@hydra.main(version_base="1.2", config_path='./visualization_config')
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration."""

    # Create visualization config
    vis_config = VisualizationConfig(
        point_radius=0.005,
        num_eigenvectors_to_show=getattr(cfg.globals, 'num_eigenvectors', 60),
        colormap='coolwarm',
        enable_eigenvalue_info=True,
        enable_correlation_analysis=True,
        num_validation_sources=getattr(cfg.globals, 'num_validation_sources', 5)
    )

    # Check dependencies
    print("Checking dependencies...")
    try:
        from pyFM.mesh import TriMesh
        print("[OK] PyFM available")
    except ImportError:
        raise ImportError("PyFM is required for GT eigendecomposition. Install with: pip install pyFM")

    if HAS_IGL:
        print("[OK] libigl available")
    else:
        print("[!] libigl not available - GT mean curvature will be skipped")

    # Create visualizer and run
    visualizer = RealTimeEigenanalysisVisualizer(config=vis_config)
    visualizer.run_dataset_iteration(cfg)


if __name__ == "__main__":
    main()