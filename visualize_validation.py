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
import sys
import time

import numpy as np
import torch
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
    compute_laplacian_eigendecomposition
)
from neural_local_laplacian.utils.geodesic_utils import (
    compute_heat_geodesic_pointcloud,
    compute_heat_geodesic_learned,
    compute_heat_geodesic_mesh,
    compute_geodesic_metrics,
    compute_exact_geodesics,
    normalize_distances,
    build_pointcloud_grad_div_operators,
    edge_index_from_knn_indices,
    select_geodesic_source_vertex,
    select_multiple_geodesic_sources,
    compute_multisource_geodesic_metrics,
)
from torch_geometric.data import Data, Batch


@dataclass
class VisualizationConfig:
    """Configuration for eigenanalysis visualization."""
    point_radius: float = 0.005
    show_wireframe: bool = False
    colormap: str = 'coolwarm'
    num_eigenvectors_to_show: int = 8
    enable_eigenvalue_info: bool = True
    enable_correlation_analysis: bool = True


@dataclass
class ReconstructionSettings:
    """Settings for mesh reconstruction visualization."""
    use_pred_areas: bool = True  # Required for M-orthonormal eigenvectors from generalized EVP
    current_pred_k: int = 20  # Will be updated with actual k from dataset
    current_robust_k: int = 20  # Independent k for robust-laplacian


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

    # GT (mesh-based cotangent) timing
    gt_matrix_assembly_time: float = 0.0  # igl.cotmatrix + massmatrix

    # Robust (k-NN point cloud) timing
    robust_matrix_assembly_time: float = 0.0  # point_cloud_laplacian (k-NN + weights + assembly)

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
        status = "ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ PASS" if self.satisfies_maximum_principle else "ÃƒÂ¢Ã…â€œÃ¢â‚¬â€ FAIL"
        max_status = "ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“" if self.max_at_source else "ÃƒÂ¢Ã…â€œÃ¢â‚¬â€"
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

        # NEW: Timing results for Laplacian assembly comparison
        self.timing_results = LaplacianTimingResults()

        # NEW: Green's function validation results
        self.current_greens_results = None

        # NEW: Heat Method geodesic validation results
        self.current_heat_geodesic_results = None

        # NEW: Laplacian sparsity comparison stats
        self.current_sparsity_stats: Dict[str, LaplacianSparsityStats] = {}

        # Debug flags (set from config in run_dataset_iteration)
        self._diagnostic_mode = False
        self._skip_robust = False
        self._skip_visualization = False

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
        else:
            psim.Text("k-NN input: (no data loaded)")

        psim.Text("(Press Enter to apply changes)")

        psim.Text("")
        psim.Text("Current Settings:")
        if self.reconstruction_settings.use_pred_areas:
            psim.TextColored((0.0, 1.0, 0.0, 1.0), "[OK] Using predicted areas from model")
            psim.Text("  (Area-weighted reconstruction for PRED)")
        else:
            psim.TextColored((1.0, 0.5, 0.0, 1.0), "[o] Using standard Euclidean reconstruction")
            psim.Text("  (Standard L2 projection for PRED)")

        if self.original_k is not None:
            psim.Text(f"Original k: {self.original_k}, PRED k: {self.reconstruction_settings.current_pred_k}, Robust k: {self.reconstruction_settings.current_robust_k}")

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

            if t.robust_matrix_assembly_time > 0 and t.pred_total_time > 0:
                pred_vs_robust = t.pred_total_time / t.robust_matrix_assembly_time
                psim.Text(f"PRED vs Robust: {pred_vs_robust:.2f}x")
        else:
            psim.Text("(No timing data yet)")

        # === GREEN'S FUNCTION MAXIMUM PRINCIPLE VALIDATION ===
        psim.Text("")
        psim.Separator()
        psim.Text("Maximum Principle Validation:")

        if self.current_greens_results is not None and len(self.current_greens_results) > 0:
            # Show results for each method
            for method_name, result in self.current_greens_results.items():
                if result.satisfies_maximum_principle:
                    psim.TextColored((0.0, 1.0, 0.0, 1.0), f"  {method_name}: ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ PASS")
                else:
                    psim.TextColored((1.0, 0.0, 0.0, 1.0), f"  {method_name}: ÃƒÂ¢Ã…â€œÃ¢â‚¬â€ FAIL")
                    if result.num_violations > 0:
                        psim.Text(f"    {result.num_violations} vertices above source")
                    if not result.max_at_source:
                        psim.Text(f"    Max at vertex {result.worst_violation_vertex}")
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

                # Color code by correlation quality
                if corr > 0.99:
                    color = (0.0, 1.0, 0.0, 1.0)  # Green
                elif corr > 0.95:
                    color = (1.0, 1.0, 0.0, 1.0)  # Yellow
                else:
                    color = (1.0, 0.3, 0.0, 1.0)  # Orange-red

                psim.TextColored(color, f"  {method_name:<16} {corr:>7.4f} {mae:>7.4f} {max_err:>7.4f} {mono:>7.4f}")
        else:
            psim.Text("(Not computed yet)")

        # === EIGENVECTOR CUMULATIVE COSINE SIMILARITY COMPARISON ===
        psim.Text("")
        psim.Separator()
        psim.Text("Eigenvector Mean |cos| Similarity (cumulative):")

        pred_sims = self.current_cosine_similarities_pred
        robust_sims = self.current_cosine_similarities_robust

        if pred_sims is not None or robust_sims is not None:
            max_available = 0
            if pred_sims is not None:
                max_available = max(max_available, len(pred_sims))
            if robust_sims is not None:
                max_available = max(max_available, len(robust_sims))

            # Header
            psim.TextColored((0.7, 0.7, 0.7, 1.0), f"  {'#Eigvec':>8}  {'PRED':>8}  {'Robust':>8}  {'Delta':>8}")
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
                pred_mean = float(pred_sims[:count].mean()) if pred_sims is not None and count <= len(pred_sims) else None
                robust_mean = float(robust_sims[:count].mean()) if robust_sims is not None and count <= len(robust_sims) else None

                pred_str = f"{pred_mean:>8.4f}" if pred_mean is not None else f"{'N/A':>8}"
                robust_str = f"{robust_mean:>8.4f}" if robust_mean is not None else f"{'N/A':>8}"

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

                psim.TextColored(color, f"  {count:>8}  {pred_str}  {robust_str}  {delta_str}")
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

    def _recompute_validations_after_k_change(self):
        """Recompute all downstream validations after a k change.

        This re-runs Green's function validation (step 8), heat method geodesics (step 9),
        sparsity stats, and updates all UI-facing results.
        """
        if self.current_gt_data is None or self.current_inference_result is None:
            print("[!] No current data available for validation recomputation")
            return

        vertices = self.current_gt_data['vertices']
        faces = self.current_gt_data['faces']
        mesh_structure = self.current_mesh_structure

        # --- Recompute sparsity stats ---
        print("\nRecomputing sparsity stats...")
        self._compute_and_store_all_sparsity_stats()

        # --- Recompute Green's function validation (Step 8) ---
        print("\nRecomputing Green's function validation...")
        try:
            greens_results = self.compute_and_visualize_greens_functions(
                mesh_structure, self.current_gt_data, self.current_inference_result
            )
            self.current_greens_results = greens_results
        except Exception as e:
            print(f"  [!] Green's function recomputation failed: {e}")
            import traceback
            traceback.print_exc()

        # --- Recompute Heat Method geodesics (Step 9) ---
        print("\nRecomputing Heat Method geodesics...")
        try:
            # Build Laplacians
            L_gt, M_gt = None, None
            L_pred, M_pred = None, None
            L_robust, M_robust = None, None

            if HAS_IGL:
                V = vertices.astype(np.float64)
                F = faces.astype(np.int32)
                L_gt = -igl.cotmatrix(V, F)
                M_gt = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)

            if self.current_inference_result.get('stiffness_matrix') is not None:
                L_pred = self.current_inference_result['stiffness_matrix']
                M_pred = self.current_inference_result.get('mass_matrix')

            if not self._skip_robust:
                k = self.reconstruction_settings.current_robust_k
                L_robust, M_robust = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=k)

            heat_geodesic_results = self.validate_heat_method_geodesics_step9(
                vertices=vertices,
                faces=faces,
                L_gt=L_gt, M_gt=M_gt,
                L_pred=L_pred, M_pred=M_pred,
                L_robust=L_robust, M_robust=M_robust,
                source_vertex_idx=None,
                mesh_structure=mesh_structure,
                k_neighbors=self.original_k if self.original_k else 20
            )
            self.current_heat_geodesic_results = heat_geodesic_results
        except Exception as e:
            print(f"  [!] Heat Method geodesic recomputation failed: {e}")
            import traceback
            traceback.print_exc()

    def _clear_stored_references(self):
        """Clear stored references from previous batch to free memory.

        This helps ensure consistent timing across batches by preventing
        memory accumulation from affecting performance.
        """
        # Clear GPU tensors
        self.current_stiffness_weights = None
        self.current_areas = None
        self.current_vertex_indices = None
        self.current_center_indices = None
        self.current_batch_indices = None

        # Clear numpy arrays and dicts
        self.current_gt_data = None
        self.current_inference_result = None
        self.current_predicted_data = None
        self.current_original_vertices = None
        self.current_sparsity_stats = {}

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache again after gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_predicted_vertex_areas(self, inference_result: Dict) -> Optional[np.ndarray]:
        """Compute predicted vertex areas based on current settings.

        Uses the areas predicted by the model's area head.
        """
        if not self.reconstruction_settings.use_pred_areas:
            return None

        if inference_result.get('areas') is not None:
            predicted_vertex_areas = inference_result['areas']
            print(f"Using predicted areas from model (range: [{predicted_vertex_areas.min():.6f}, {predicted_vertex_areas.max():.6f}])")
            return predicted_vertex_areas
        else:
            print(f"[!] No predicted areas available, falling back to standard reconstruction")
            return None

    def _compute_all_reconstructions(self, gt_data: Dict, inference_result: Dict) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Compute GT, predicted, and robust-laplacian reconstructions."""
        gt_reconstructions = []
        pred_reconstructions = []
        robust_reconstructions = []

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

        return gt_reconstructions, pred_reconstructions, robust_reconstructions

    def _update_mesh_reconstructions(self, gt_data: Dict, inference_result: Dict):
        """Complete pipeline for computing and visualizing mesh reconstructions."""
        print(f"Computing and visualizing mesh reconstructions...")

        # Compute reconstructions
        gt_reconstructions, pred_reconstructions, robust_reconstructions = self._compute_all_reconstructions(
            gt_data, inference_result
        )

        # Visualize reconstructions
        if gt_reconstructions or pred_reconstructions or robust_reconstructions:
            self.visualize_mesh_reconstructions(
                gt_data['faces'],
                gt_reconstructions,
                pred_reconstructions,
                robust_reconstructions,
                gt_data.get('gt_eigenvalues'),
                inference_result['predicted_eigenvalues'],
                gt_data.get('robust_eigenvalues')
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

        # Build k-NN index for the entire mesh
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=new_k + 1, algorithm='auto').fit(vertices)

        # Get k+1 nearest neighbors for ALL vertices at once
        distances, neighbor_indices = nbrs.kneighbors(vertices)  # Shape: (N, k+1)

        # Vectorized removal of center point from neighbors
        center_positions = np.arange(num_vertices)[:, np.newaxis]  # Shape: (N, 1)
        is_center_mask = neighbor_indices == center_positions  # Shape: (N, k+1)

        # Create mask to keep only non-center neighbors
        keep_mask = ~is_center_mask  # Shape: (N, k+1)

        # For each row, we want to keep the first k True values in keep_mask
        keep_positions = np.cumsum(keep_mask, axis=1)  # Shape: (N, k+1)
        final_mask = (keep_positions <= new_k) & keep_mask  # Shape: (N, k+1)

        # Extract neighbor indices using the mask
        neighbor_indices_flat = neighbor_indices[final_mask]  # Shape: (N*k,)
        neighbor_indices_filtered = neighbor_indices_flat.reshape(num_vertices, new_k)  # Shape: (N, k)

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
        new_stiffness_matrix, new_mass_matrix = assemble_stiffness_and_mass_matrices(
            stiffness_weights=corrected_weights,
            areas=self.current_areas,
            attention_mask=attention_mask,
            vertex_indices=new_vertex_indices,
            center_indices=new_center_indices,
            batch_indices=new_batch_indices
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

        This replicates MeshDataset._extract_all_patches logic with configurable k.

        Args:
            vertices: Mesh vertices of shape (N, 3)
            k: Number of nearest neighbors per patch

        Returns:
            Data object ready for model inference
        """
        from sklearn.neighbors import NearestNeighbors
        from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData

        num_vertices = len(vertices)

        # Build k-NN index
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(vertices)
        distances, neighbor_indices = nbrs.kneighbors(vertices)  # Shape: (N, k+1)

        # Remove center point from neighbors (vectorized)
        center_positions = np.arange(num_vertices)[:, np.newaxis]
        is_center_mask = neighbor_indices == center_positions
        keep_mask = ~is_center_mask
        keep_positions = np.cumsum(keep_mask, axis=1)
        final_mask = (keep_positions <= k) & keep_mask

        neighbor_indices_flat = neighbor_indices[final_mask]
        neighbor_indices_filtered = neighbor_indices_flat.reshape(num_vertices, k)

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

    def _print_timing_summary(self):
        """Print a summary table comparing matrix assembly times for all methods."""
        t = self.timing_results

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

        # Robust
        print(f"{'Robust (Point Cloud)':<25} {t.robust_matrix_assembly_time * 1000:>10.2f} ms   k-NN + weights + assembly")

        print(f"{'-' * 70}")

        # GT
        print(f"{'GT (Mesh Cotangent)':<25} {t.gt_matrix_assembly_time * 1000:>10.2f} ms   igl.cotmatrix + massmatrix")

        print(f"{'-' * 70}")

        # Speedup comparison (if GT is non-zero)
        if t.gt_matrix_assembly_time > 0:
            pred_vs_gt = t.pred_total_time / t.gt_matrix_assembly_time
            robust_vs_gt = t.robust_matrix_assembly_time / t.gt_matrix_assembly_time if t.robust_matrix_assembly_time > 0 else 0
            print(f"Relative to GT:  PRED = {pred_vs_gt:.2f}x,  Robust = {robust_vs_gt:.2f}x")

        if t.robust_matrix_assembly_time > 0 and t.pred_total_time > 0:
            pred_vs_robust = t.pred_total_time / t.robust_matrix_assembly_time
            print(f"PRED vs Robust:  {pred_vs_robust:.2f}x")

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
        t_extraction_start = time.perf_counter()

        new_patch_data = self._extract_patches_for_mesh_with_k(
            self.current_original_vertices, new_k
        )

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
        self.current_stiffness_weights = torch.from_numpy(new_inference_result['stiffness_weights'])
        self.current_areas = torch.from_numpy(new_inference_result['areas'])

        # Reassemble learned gradient operator if in gradient mode
        operator_mode = getattr(self.current_model, '_operator_mode', 'stiffness')
        new_forward_result = new_inference_result.get('forward_result')
        if operator_mode == "gradient" and new_forward_result is not None and new_forward_result.get('grad_coeffs') is not None:
            try:
                print("  Reassembling learned gradient operator G for new k...")
                batch_indices_for_g = self.current_batch_indices.clone()
                self.current_learned_gradient_op = assemble_gradient_operator(
                    grad_coeffs=new_forward_result['grad_coeffs'],
                    attention_mask=new_forward_result['attention_mask'],
                    vertex_indices=self.current_vertex_indices,
                    center_indices=self.current_center_indices,
                    batch_indices=batch_indices_for_g
                )
                print(f"  Updated learned G: {self.current_learned_gradient_op.shape} ({self.current_learned_gradient_op.nnz} non-zeros)")
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

        # STEP 5: Recompute all downstream validations (Green's fn, geodesics, sparsity)
        print(f"STEP 5: Recomputing downstream validations...")
        self._recompute_validations_after_k_change()

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

        # Recompute all downstream validations (Green's fn, geodesics, sparsity)
        print(f"Recomputing downstream validations...")
        self._recompute_validations_after_k_change()

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

        # Update stored average for UI
        if cosine_similarities_pred is not None:
            num_to_show = min(self.config.num_eigenvectors_to_show, len(cosine_similarities_pred))
            self.current_avg_cosine_similarity = float(cosine_similarities_pred[:num_to_show].mean())

        # Update per-eigenvector cosine similarities for UI comparison table
        self.current_cosine_similarities_pred = cosine_similarities_pred
        self.current_cosine_similarities_robust = cosine_similarities_robust

        # Print updated comparison
        print(f"\nUpdated eigenvector cosine similarities:")
        if cosine_similarities_pred is not None:
            print(f"  GT vs PRED mean: {cosine_similarities_pred.mean():.4f}")
        if cosine_similarities_robust is not None:
            print(f"  GT vs Robust mean: {cosine_similarities_robust.mean():.4f}")

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

    def load_trained_model(self, ckpt_path: Path, device: torch.device, cfg: DictConfig, use_torch_compile: bool = True, diagnostic_mode: bool = False) -> LaplacianTransformerModule:
        """
        Load trained LaplacianTransformerModule from checkpoint.

        Args:
            ckpt_path: Path to the checkpoint file
            device: Device to load the model on
            cfg: Hydra config containing model configuration
            use_torch_compile: Whether to use torch.compile() for faster inference
            diagnostic_mode: If True, disables optimizations for debugging timing issues

        Returns:
            Loaded model in evaluation mode
        """
        if diagnostic_mode:
            print("[DIAGNOSTIC MODE] Disabling torch.compile for debugging")
            use_torch_compile = False

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

            # Apply torch.compile() for faster inference (PyTorch 2.0+)
            # Use dynamic=True to handle varying mesh sizes without recompilation
            # Use fullgraph=False to allow graph breaks for better compatibility
            if use_torch_compile:
                try:
                    print("Applying torch.compile() for optimized inference...")
                    # dynamic=True: Allows varying input shapes without recompilation
                    # fullgraph=False: Allows graph breaks for better compatibility with varying shapes
                    model = torch.compile(model, mode="default", dynamic=True, fullgraph=False)
                    print("[OK] torch.compile() applied (dynamic shapes enabled)")
                except Exception as e:
                    print(f"[!] torch.compile() failed, using eager mode: {e}")

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

    def _warmup_model(self, model: LaplacianTransformerModule, device: torch.device, k: int = 30, num_warmup: int = 3):
        """
        Warmup the model to trigger torch.compile() compilation.

        With dynamic=True, we warm up with multiple input sizes to ensure
        the compiler generates efficient code for varying shapes.

        Args:
            model: The model to warm up
            device: Device to run on
            k: Number of neighbors (should match actual data!)
            num_warmup: Number of warmup iterations per size
        """
        print(f"Warming up model (k={k}, dynamic shapes)...")

        from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData

        # Warmup with different BATCH sizes but SAME k as actual data
        # This ensures torch.compile sees the right sequence length
        # Include sizes close to actual mesh sizes (e.g., 9999, 10002)
        warmup_patch_counts = [1000, 5000, 9999, 10000, 10002]

        # Determine mixed precision dtype (must match inference for torch.compile)
        use_amp = device.type == 'cuda'
        if use_amp:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        try:
            with torch.no_grad():
                for num_patches in warmup_patch_counts:
                    dummy_data = MeshPatchData(
                        pos=torch.randn(num_patches * k, 3, device=device),
                        x=torch.randn(num_patches * k, 3, device=device),
                        patch_idx=torch.arange(num_patches, device=device).repeat_interleave(k),
                        vertex_indices=torch.randint(0, num_patches, (num_patches * k,), device=device),
                        center_indices=torch.arange(num_patches, device=device)
                    )

                    for i in range(num_warmup):
                        if device.type == 'cuda':
                            torch.cuda.synchronize()

                        # Use same mixed precision as inference
                        if use_amp:
                            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                                _ = model._forward_pass(dummy_data)
                        else:
                            _ = model._forward_pass(dummy_data)

                        if device.type == 'cuda':
                            torch.cuda.synchronize()

                    print(f"  Warmed up with {num_patches} patches, k={k}")

            # Clear cache after warmup
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            print(f"[OK] Model warmup complete")
        except Exception as e:
            print(f"[!] Model warmup failed (this is OK if not using torch.compile): {e}")

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
            # Note: eigsh returns M-orthonormal eigenvectors (ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦^T M ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ = I)
            # We preserve this property for correct area-weighted reconstruction
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
            try:
                print("Computing GT mean curvature using libigl...")
                # Convert to float64 for libigl (more stable)
                vertices_igl = vertices.astype(np.float64)
                faces_igl = faces.astype(np.int32)

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

                # Build GT mesh gradient operator (face-based, 3*nF x N)
                gt_mesh_grad_op = igl.grad(vertices_igl, faces_igl)
                gt_face_areas = (igl.doublearea(vertices_igl, faces_igl).flatten() / 2.0).astype(np.float64)
                print(f"GT mesh gradient operator: {gt_mesh_grad_op.shape}")

            except Exception as e:
                print(f"Warning: Failed to compute GT curvatures with libigl: {e}")
                gt_mean_curvature = None
                gt_mean_curvature_vector = None
                gt_gaussian_curvature = None
                gt_mesh_grad_op = None
                gt_face_areas = None

        # Compute GT Laplacian eigendecomposition using PyFM
        print("Computing GT Laplacian eigendecomposition using PyFM...")
        gt_laplacian_time = 0.0
        try:
            # Create PyFM TriMesh object
            pyfm_mesh = TriMesh(vertices, faces)

            # === TIME: GT Laplacian assembly (PyFM uses cotangent weights from mesh) ===
            # Note: process() computes Laplacian AND eigendecomposition together
            # We time the whole call but the Laplacian assembly is O(N) while eigen is O(N*k)
            t_gt_start = time.perf_counter()

            # Process the mesh and compute the Laplacian spectrum
            pyfm_mesh.process(k=self.config.num_eigenvectors_to_show, intrinsic=False, verbose=False)

            t_gt_end = time.perf_counter()
            gt_laplacian_time = t_gt_end - t_gt_start

            # Store timing (note: includes eigendecomposition)
            self.timing_results.gt_laplacian_time = gt_laplacian_time
            print(f"[TIMING] GT (PyFM) Laplacian + eigen: {gt_laplacian_time * 1000:.2f} ms")

            # Retrieve eigenvalues, eigenfunctions, and vertex areas
            # Note: PyFM returns M-orthonormal eigenvectors (ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦^T M ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ = I)
            # We preserve this property for correct area-weighted reconstruction
            gt_eigenvalues = pyfm_mesh.eigenvalues
            gt_eigenvectors = pyfm_mesh.eigenvectors
            vertex_areas = pyfm_mesh.vertex_areas

            print(f"Computed {len(gt_eigenvalues)} GT eigenvalues")
            print(f"GT eigenvalue range: [{gt_eigenvalues[0]:.2e}, {gt_eigenvalues[-1]:.6f}]")

        except Exception as e:
            print(f"Warning: Failed to compute GT eigendecomposition: {e}")
            gt_eigenvalues = None
            gt_eigenvectors = None
            vertex_areas = None

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
            # === DIAGNOSTIC: Run inference multiple times to check consistency ===
            inference_times = []
            num_runs = 3  # Run multiple times to check if first run is slow

            for run_idx in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t_inference_start = time.perf_counter()

                # Forward pass with mixed precision for faster inference
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        forward_result = model._forward_pass(batch_data)
                else:
                    forward_result = model._forward_pass(batch_data)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t_inference_end = time.perf_counter()
                inference_times.append(t_inference_end - t_inference_start)

            # Report all inference times
            print(f"  [DIAGNOSTIC] Inference times: {[f'{t * 1000:.1f}ms' for t in inference_times]}")

            # Use the minimum time as the "true" inference time (excludes one-time overhead)
            pred_inference_time = min(inference_times)
            print(f"  [TIMING] Model inference (best of {num_runs}): {pred_inference_time * 1000:.2f} ms")

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

            # Use patch_idx if available (MeshDataset), otherwise use batch (synthetic)
            batch_indices = getattr(batch_data, 'patch_idx', batch_data.batch)

            # === TIME: Matrix assembly ===
            t_assembly_start = time.perf_counter()

            # Assemble separate stiffness and mass matrices
            stiffness_matrix, mass_matrix = assemble_stiffness_and_mass_matrices(
                stiffness_weights=stiffness_weights,
                areas=areas,
                attention_mask=attention_mask,
                vertex_indices=batch_data.vertex_indices,
                center_indices=batch_data.center_indices,
                batch_indices=batch_indices
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
            'stiffness_weights': stiffness_weights.cpu().numpy(),
            'areas': areas.cpu().numpy(),
            'attention_mask': attention_mask.cpu().numpy(),
            'batch_sizes': batch_sizes.cpu().numpy(),
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
                                       robust_eigenvalues: Optional[np.ndarray]):
        """
        Visualize progressive mesh reconstructions using eigenvectors.
        GT (PyFM) on the right, PRED in the center-left, Robust on the far left.

        Args:
            original_faces: Mesh faces for topology
            gt_reconstructions: List of GT (PyFM) reconstructed vertices
            pred_reconstructions: List of predicted reconstructed vertices
            robust_reconstructions: List of robust-laplacian reconstructed vertices
            gt_eigenvalues: GT eigenvalues for labeling
            pred_eigenvalues: Predicted eigenvalues for labeling
            robust_eigenvalues: Robust-laplacian eigenvalues for labeling
        """
        print("Adding mesh reconstruction visualizations...")

        # Fixed positions for overlaid reconstructions
        gt_offset = np.array([3.0, 0.0, 0.0])  # GT (PyFM) reconstructions on the right
        pred_offset = np.array([0.0, 0.0, -3.0])  # PRED reconstructions in front
        robust_offset = np.array([-3.0, 0.0, 0.0])  # Robust reconstructions on the left

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

        print(f"Added {len(gt_reconstructions)} GT-PyFM, {len(pred_reconstructions)} PRED, and {len(robust_reconstructions)} Robust mesh reconstructions")
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
            mesh_structure.add_scalar_quantity(
                name="I Vertex Areas - PRED",
                values=inference_result['areas'],
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
                                             robust_eigenvalues: Optional[np.ndarray] = None, robust_eigenvectors: Optional[np.ndarray] = None):
        """Add GT (PyFM), predicted, and robust-laplacian eigenvector scalar fields to the mesh."""
        num_to_show = self.config.num_eigenvectors_to_show

        # Determine how many eigenvectors we can show for each type
        gt_available = gt_eigenvectors.shape[1] if gt_eigenvectors is not None else 0
        pred_available = pred_eigenvectors.shape[1] if pred_eigenvectors is not None else 0
        robust_available = robust_eigenvectors.shape[1] if robust_eigenvectors is not None else 0

        max_available = max(gt_available, pred_available, robust_available)
        num_to_show = min(num_to_show, max_available)

        print(f"Adding eigenvector visualization:")
        print(f"  GT (PyFM) eigenvectors available: {gt_available}")
        print(f"  Predicted eigenvectors available: {pred_available}")
        print(f"  Robust-laplacian eigenvectors available: {robust_available}")
        print(f"  Showing: {num_to_show} eigenvectors per type")

        # Compute cosine similarities between GT and predicted eigenvectors
        cosine_similarities_pred = self._compute_eigenvector_cosine_similarities(
            gt_eigenvectors, pred_eigenvectors
        )

        # Compute cosine similarities between GT and robust eigenvectors
        cosine_similarities_robust = self._compute_eigenvector_cosine_similarities(
            gt_eigenvectors, robust_eigenvectors
        )

        # Print cosine similarities to console
        if cosine_similarities_pred is not None or cosine_similarities_robust is not None:
            print(f"\n" + "-" * 85)
            print("EIGENVECTOR COSINE SIMILARITIES")
            print("-" * 85)
            print(f"{'Index':<8} {'GT vs PRED':<14} {'GT vs Robust':<14} {'Description'}")
            print("-" * 85)
            for i in range(num_to_show):
                if i == 0:
                    desc = "constant"
                elif i == 1:
                    desc = "Fiedler"
                else:
                    desc = ""

                pred_sim = cosine_similarities_pred[i] if cosine_similarities_pred is not None and i < len(cosine_similarities_pred) else float('nan')
                robust_sim = cosine_similarities_robust[i] if cosine_similarities_robust is not None and i < len(cosine_similarities_robust) else float('nan')

                print(f"{i:<8} {pred_sim:<14.6f} {robust_sim:<14.6f} {desc}")

            # Summary statistics
            print("-" * 85)
            if cosine_similarities_pred is not None:
                print(f"GT vs PRED - Mean: {cosine_similarities_pred[:num_to_show].mean():.6f}, Min: {cosine_similarities_pred[:num_to_show].min():.6f}, Max: {cosine_similarities_pred[:num_to_show].max():.6f}")
            if cosine_similarities_robust is not None:
                print(f"GT vs Robust - Mean: {cosine_similarities_robust[:num_to_show].mean():.6f}, Min: {cosine_similarities_robust[:num_to_show].min():.6f}, Max: {cosine_similarities_robust[:num_to_show].max():.6f}")
            print("-" * 85 + "\n")

            # Store average cosine similarity for UI display (GT vs PRED)
            if cosine_similarities_pred is not None:
                self.current_avg_cosine_similarity = float(cosine_similarities_pred[:num_to_show].mean())

            # Store per-eigenvector cosine similarities for UI comparison table
            self.current_cosine_similarities_pred = cosine_similarities_pred
            self.current_cosine_similarities_robust = cosine_similarities_robust

        # Add eigenvectors in groups
        for i in range(num_to_show):
            # Get cosine similarities for this index
            cos_sim_pred = cosine_similarities_pred[i] if cosine_similarities_pred is not None and i < len(cosine_similarities_pred) else None
            cos_sim_robust = cosine_similarities_robust[i] if cosine_similarities_robust is not None and i < len(cosine_similarities_robust) else None

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

    # =========================================================================
    # GREEN'S FUNCTION MAXIMUM PRINCIPLE VALIDATION
    # =========================================================================

    def compute_greens_function(
            self,
            laplacian_matrix: scipy.sparse.csr_matrix,
            mass_matrix: Optional[scipy.sparse.csr_matrix],
            source_vertex_idx: int,
            regularization: float = 1e-6
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

            # Regularized system: (L + ÃƒÅ½Ã‚ÂµM)g = ÃƒÅ½Ã‚Â´
            A = L + adaptive_reg * M

            # Solve the linear system
            g_raw = scipy.sparse.linalg.spsolve(A, delta)

            # Check for NaN or Inf
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
        2. SECONDARY: Does value decrease with GEODESIC distance? (uses igl.exact_geodesic)
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

        # Try to compute geodesic distances using various methods
        # Priority: 1) pygeodesic (fast, robust), 2) igl.heat_geodesic, 3) igl.exact_geodesic, 4) Euclidean

        geodesic_computed = False

        # Method 1: Try pygeodesic (fastest and most robust)
        try:
            import pygeodesic.geodesic as geodesic

            V = vertices.astype(np.float64)
            F = faces.astype(np.int32)

            # Create geodesic algorithm object
            geoalg = geodesic.PyGeodesicAlgorithmExact(V, F)

            # Compute distances from source to all vertices
            distances, _ = geoalg.geodesicDistances(np.array([source_vertex_idx]), None)

            if distances is not None and len(distances) == n:
                geodesic_computed = True
            else:
                raise ValueError(f"pygeodesic returned invalid result")

        except ImportError:
            pass  # pygeodesic not installed, try next method
        except Exception as e:
            print(f"    [!] pygeodesic failed: {e}")

        # Method 2: Try igl heat geodesic
        if not geodesic_computed and HAS_IGL:
            try:
                V = vertices.astype(np.float64)
                F = faces.astype(np.int32)

                # Compute mean edge length for time parameter
                edges = np.vstack([
                    F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]
                ])
                edge_lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
                mean_edge = edge_lengths.mean()
                t = mean_edge ** 2  # Default time parameter

                # Source vertex
                gamma = np.array([source_vertex_idx], dtype=np.int32)

                # Compute heat geodesic distances
                distances = igl.heat_geodesic(V, F, t, gamma)

                if distances is not None and len(distances) == n:
                    geodesic_computed = True
                else:
                    raise ValueError("heat_geodesic returned invalid result")

            except Exception as e:
                print(f"    [!] igl.heat_geodesic failed: {e}")

        # Method 3: Try igl exact geodesic
        if not geodesic_computed and HAS_IGL:
            try:
                V = vertices.astype(np.float64)
                F = faces.astype(np.int32)

                VS = np.array([source_vertex_idx], dtype=np.int32)
                VT = np.arange(n, dtype=np.int32)
                distances = igl.exact_geodesic(V, F, VS, VT)

                if distances is not None and len(distances) == n:
                    geodesic_computed = True
                else:
                    raise ValueError("exact_geodesic returned invalid result")

            except Exception as e:
                print(f"    [!] igl.exact_geodesic failed: {e}")

        # Method 4: Fall back to Euclidean distances
        if not geodesic_computed:
            print(f"    [!] All geodesic methods failed, using Euclidean distances")
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
        except Exception as e:
            print(f"  [!] Could not check boundaries: {e}")

        print("-" * 70)

        # Select source vertex
        if source_vertex_idx is None:
            source_vertex_idx = self.select_source_vertex(vertices, method="centroid")

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
                    print(f"  GT Green's function: min={gt_greens.min():.6f}, max={gt_greens.max():.6f}")

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

                result = self.compute_greens_function(L_pred, M_pred, source_vertex_idx)

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
                # Use the same k as PRED for fair comparison
                k = self.original_k if self.original_k is not None else 30

                L_robust, M_robust = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=k)

                # Diagnostic: check Laplacian properties
                diag_robust = np.array(L_robust.diagonal()).flatten()
                print(f"  Robust Laplacian: diag range [{diag_robust.min():.4f}, {diag_robust.max():.4f}], row sums ~ {np.abs(L_robust.sum(axis=1)).max():.2e}")

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

            except Exception as e:
                print(f"  [!] Failed to compute Robust Green's function: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  [!] Robust computation skipped (skip_robust=True)")

        # =====================================================================
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
                status = "ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ PASS"
            else:
                status = "ÃƒÂ¢Ã…â€œÃ¢â‚¬â€ FAIL"
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
            print("ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ All methods satisfy the discrete maximum principle")
        else:
            print("ÃƒÂ¢Ã…â€œÃ¢â‚¬â€ Some methods VIOLATE the discrete maximum principle:")
            for method_name, result in results.items():
                if not result.satisfies_maximum_principle:
                    print(f"  - {result.method_name}: Max value {result.max_value:.4f} at vertex {result.worst_violation_vertex}, "
                          f"but source has only {result.value_at_source:.4f}")

        print("=" * 90)

        # =====================================================================
        # Add visualizations to mesh structure
        # =====================================================================
        if mesh_structure is not None and not self._skip_visualization:
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
                mesh_grad_op = igl.grad(V, F)  # (3*nF x nV) matrix
                mesh_face_areas = igl.doublearea(V, F).flatten() / 2.0  # (nF,)
                print(f"  Mesh gradient operator (igl): {mesh_grad_op.shape}")
            except Exception as e:
                print(f"  [!] igl.grad() failed: {e}")

        # Build POINT CLOUD gradient/divergence operators using shared function
        # IMPORTANT: Use the SAME k-NN connectivity as PRED's Laplacian!
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
                print(f"  [!] pcdiff operators failed: {e}")
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

        # =====================================================================
        # Compute EXACT geodesic distances (TRUE ground truth)
        # =====================================================================
        print("\nComputing EXACT geodesic distances (ground truth)...")

        # Use shared function for exact geodesics
        exact_distances = compute_exact_geodesics(vertices, faces, source_vertex_idx)

        if exact_distances is not None:
            exact_method = "pygeodesic/igl"
            print(f"  Exact geodesics computed successfully")
        else:
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

        # =====================================================================
        # NOTE: normalize_distances and compute_geodesic_metrics are imported
        # from geodesic_utils - no local definitions needed
        # =====================================================================

        # =====================================================================
        # Helper functions for Heat Method with different topologies
        # These wrap the shared functions from geodesic_utils with timing/logging
        # =====================================================================

        def compute_heat_geodesic_mesh_local(L, M, grad_op, face_areas, method_name, t=None):
            """Wrapper around shared compute_heat_geodesic_mesh with timing."""
            start_time = time.time()

            # Use shared function from geodesic_utils
            distances = compute_heat_geodesic_mesh(
                L=L, M=M, grad_op=grad_op, face_areas=face_areas,
                source_idx=source_vertex_idx, n_vertices=n, t=t
            )

            elapsed_ms = (time.time() - start_time) * 1000
            return distances, t, elapsed_ms

        def compute_heat_geodesic_pointcloud_local(L, M, grad_op, div_op, method_name, t=None):
            """Wrapper around shared compute_heat_geodesic_pointcloud with timing."""
            start_time = time.time()

            # Use shared function from geodesic_utils
            distances = compute_heat_geodesic_pointcloud(
                L=L, M=M, grad_op=grad_op, div_op=div_op,
                source_idx=source_vertex_idx, n_vertices=n, t=t
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
                    forward_result = self.current_inference_result.get('forward_result') if self.current_inference_result else None
                    if forward_result is not None and forward_result.get('grad_coeffs') is not None:
                        batch_indices = getattr(self.current_batch_indices, 'clone', lambda: self.current_batch_indices)()
                        gradient_operator = assemble_gradient_operator(
                            grad_coeffs=forward_result['grad_coeffs'],
                            attention_mask=forward_result['attention_mask'],
                            vertex_indices=self.current_vertex_indices,
                            center_indices=self.current_center_indices,
                            batch_indices=batch_indices if torch.is_tensor(batch_indices) else torch.tensor(batch_indices)
                        )
                        print(f"  Assembled gradient operator G: {gradient_operator.shape} ({gradient_operator.nnz} non-zeros)")

                        start_time = time.time()
                        distances = compute_heat_geodesic_learned(
                            S=L_pred, M=M_pred, G=gradient_operator,
                            source_idx=source_vertex_idx, n_vertices=n
                        )
                        elapsed_ms = (time.time() - start_time) * 1000

                        if distances is not None:
                            geodesic_distances["PRED"] = distances
                            print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                            print(f"  Time: {elapsed_ms:.1f} ms")
                        else:
                            print(f"  [!] Learned heat method returned None")
                    else:
                        print(f"  [!] grad_coeffs not available in forward_result, skipping")
                except Exception as e:
                    print(f"  [!] Learned heat method failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # === Stiffness mode: frankenstein heat method (pcdiff grad/div) ===
                if pc_grad_op is not None and pc_div_op is not None:
                    print(f"\nComputing Heat Method geodesic with PRED Laplacian (matched k-NN)...")
                    distances, t, elapsed_ms = compute_heat_geodesic_pointcloud_local(
                        L_pred, M_pred, pc_grad_op, pc_div_op, "PRED"
                    )
                    if distances is not None:
                        geodesic_distances["PRED"] = distances
                        print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                        print(f"  Time: {elapsed_ms:.1f} ms")
                else:
                    print(f"\n[PRED] Point cloud gradient not available, skipping Heat Method")

        # Robust Laplacian: Cannot use matched k-NN (Robust builds its own tufted cover)
        # Use potpourri3d point cloud solver for visualization instead
        if L_robust is not None and M_robust is not None:
            print(f"\n[Robust] Heat Method skipped (uses internal tufted cover, not k-NN)")
            print(f"         Using potpourri3d point cloud solver for visualization...")
            try:
                import potpourri3d as pp3d
                pc_solver = pp3d.PointCloudHeatSolver(vertices)
                robust_distances = pc_solver.compute_distance(source_vertex_idx)
                geodesic_distances["Robust (pp3d)"] = robust_distances
                print(f"  Distance range: [{robust_distances.min():.4f}, {robust_distances.max():.4f}]")
            except Exception as e:
                print(f"  [!] potpourri3d point cloud solver failed: {e}")

        # Add exact geodesics to the comparison set
        geodesic_distances["Exact"] = exact_distances

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

        for method_name, distances in geodesic_distances.items():
            if method_name == "Exact":
                continue  # Don't compare exact to itself

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
            if method_name == "Exact":
                d_range = f"[{distances.min():.3f}, {distances.max():.3f}]"
                print(f"{method_name:<20} {'1.0000':>8} {'0.0000':>8} {'0.0000':>8} {'1.0000':>8} {d_range:<20}")
            else:
                r = results[method_name]
                d_range = f"[{distances.min():.3f}, {distances.max():.3f}]"
                print(f"{method_name:<20} {r.correlation_with_reference:>8.4f} {r.mean_absolute_error:>8.4f} "
                      f"{r.max_absolute_error:>8.4f} {r.monotonicity_score:>8.4f} {d_range:<20}")

        # =====================================================================
        # Add visualizations
        # =====================================================================
        if mesh_structure is not None and not self._skip_visualization:
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

                mesh_structure.add_scalar_quantity(
                    name=f"K1 Geodesic (norm) - {method_name}",
                    values=d_norm,
                    enabled=(method_name == "Exact"),
                    cmap='viridis'
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

    def process_batch(self, model: LaplacianTransformerModule, batch_data, batch_idx: int, device: torch.device):
        """Process a single batch through the complete pipeline."""
        print(f"\n{'=' * 80}")
        print(f"PROCESSING BATCH {batch_idx + 1}")
        print('=' * 80)

        # Clear previous visualization and tracking (only if polyscope is active)
        if not self._skip_visualization:
            ps.remove_all_structures()
        self.reconstruction_structure_names.clear()  # Clear reconstruction tracking

        # Clear CUDA cache to ensure consistent memory state across meshes
        # This prevents memory fragmentation from affecting timing
        if device.type == 'cuda':
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
        print(f"Total patch points: {len(data.pos)}")
        print(f"Number of patches: {len(data.center_indices)}")

        # Calculate original k from dataset
        original_k = len(data.pos) // len(data.center_indices)
        print(f"Original k (from dataset): {original_k}")

        # STEP 2: Load original mesh for GT computation
        print(f"\nSTEP 2: Loading original mesh for GT computation")
        gt_data = self.load_original_mesh_for_gt(mesh_file_path)

        # STEP 2.5: Re-extract patches with timing for fair comparison
        # (MeshDataset does k-NN during data loading, but we need to time it for comparison
        # since robust_laplacian includes k-NN search in its timing)
        print(f"\nSTEP 2.5: Extracting patches with k={original_k} (for timing comparison)...")
        t_extraction_start = time.perf_counter()

        patch_data = self._extract_patches_for_mesh_with_k(gt_data['vertices'], original_k)

        t_extraction_end = time.perf_counter()
        pred_extraction_time = t_extraction_end - t_extraction_start
        self.timing_results.pred_patch_extraction_time = pred_extraction_time
        print(f"  [TIMING] PRED patch extraction (k-NN): {pred_extraction_time * 1000:.2f} ms")

        # STEP 3: Model inference (use our extracted patches for consistent timing)
        print(f"\nSTEP 3: Model inference")
        inference_result = self.perform_model_inference(model, patch_data, device)

        if inference_result['predicted_eigenvalues'] is None:
            print("[X] Failed to compute eigendecomposition, skipping this batch")
            return

        # Compute PRED total time
        self.timing_results.pred_total_time = (
                self.timing_results.pred_patch_extraction_time +
                self.timing_results.pred_model_inference_time +
                self.timing_results.pred_matrix_assembly_time
        )

        # Compute GT matrix assembly timing (for fair comparison, time only matrix construction)
        print(f"\nSTEP 3.5: Computing GT matrix assembly timing...")
        self._compute_gt_matrices_timed(gt_data['vertices'], gt_data['faces'])

        # STEP 4: Compute predicted quantities
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
        self.current_stiffness_weights = torch.from_numpy(inference_result['stiffness_weights'])
        self.current_areas = torch.from_numpy(inference_result['areas'])
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
                batch_indices_for_g = self.current_batch_indices.clone()
                self.current_learned_gradient_op = assemble_gradient_operator(
                    grad_coeffs=forward_result['grad_coeffs'],
                    attention_mask=forward_result['attention_mask'],
                    vertex_indices=self.current_vertex_indices,
                    center_indices=self.current_center_indices,
                    batch_indices=batch_indices_for_g
                )
                print(f"  Learned G: {self.current_learned_gradient_op.shape} ({self.current_learned_gradient_op.nnz} non-zeros)")
            except Exception as e:
                print(f"  [!] Failed to assemble learned gradient operator: {e}")
                self.current_learned_gradient_op = None

        # Initialize PRED and Robust k sliders with original k
        self.reconstruction_settings.current_pred_k = self.original_k
        self.reconstruction_settings.current_robust_k = self.original_k
        print(f"Initialized PRED k={self.original_k}, Robust k={self.original_k}")

        # Update timing results with mesh info (needed for UI display)
        self.timing_results.num_vertices = len(gt_data['vertices'])
        self.timing_results.num_faces = len(gt_data['faces'])
        self.timing_results.current_k = self.original_k

        # STEP 5.5: Compute robust-laplacian with point_cloud_laplacian using same k as PRED
        print(f"\nSTEP 5.5: Computing robust-laplacian with k={self.original_k}...")
        robust_eigenvalues, robust_eigenvectors, robust_vertex_areas = self._compute_robust_laplacian_with_k(
            gt_data['vertices'], self.original_k
        )
        gt_data['robust_eigenvalues'] = robust_eigenvalues
        gt_data['robust_eigenvectors'] = robust_eigenvectors
        gt_data['robust_vertex_areas'] = robust_vertex_areas

        # STEP 5.6: Compute Laplacian sparsity comparison
        print(f"\nSTEP 5.6: Computing Laplacian sparsity comparison...")
        self._compute_and_store_all_sparsity_stats()

        # STEP 6: Visualization (skip if not needed)
        if not self._skip_visualization:
            print(f"\nSTEP 6: Creating comprehensive visualization")
            mesh_structure = self.visualize_mesh(gt_data['vertices'], gt_data['gt_vertex_normals'], gt_data['faces'])
            self.current_mesh_structure = mesh_structure
        else:
            print(f"\nSTEP 6: Skipping visualization (skip_visualization=True)")
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

        # Add visualizations (skip if not needed)
        if not self._skip_visualization and mesh_structure is not None:
            self.visualize_comprehensive_eigenvectors(
                mesh_structure,
                gt_data.get('gt_eigenvalues'), gt_data.get('gt_eigenvectors'),
                inference_result['predicted_eigenvalues'], inference_result['predicted_eigenvectors'],
                gt_data.get('robust_eigenvalues'), gt_data.get('robust_eigenvectors')
            )

            self.add_comprehensive_curvature_visualizations(mesh_structure, gt_data, predicted_data, inference_result)

            # Add mesh reconstructions using eigenvectors
            print(f"\nSTEP 7: Computing and visualizing mesh reconstructions")
            self._update_mesh_reconstructions(gt_data, inference_result)
        else:
            print(f"\nSTEP 7: Skipping mesh reconstructions (skip_visualization=True)")

        # STEP 8: Green's function maximum principle validation
        print(f"\nSTEP 8: Green's function maximum principle validation")
        L_gt = None
        M_gt = None
        L_pred = None
        M_pred = None
        L_robust = None
        M_robust = None
        try:
            greens_results = self.compute_and_visualize_greens_functions(
                mesh_structure, gt_data, inference_result
            )
            # Store results for potential UI display
            self.current_greens_results = greens_results

            # Extract Laplacians for Step 9 (they're built in step 8)
            # We'll rebuild them here for Step 9 since they're local to compute_and_visualize_greens_functions
        except Exception as e:
            print(f"  [!] Green's function validation failed: {e}")
            import traceback
            traceback.print_exc()

        # STEP 9: Heat Method Geodesic Distance Validation
        print(f"\nSTEP 9: Heat Method geodesic distance validation")
        try:
            vertices = gt_data['vertices']
            faces = gt_data['faces']

            # Build Laplacians for Step 9
            # GT Laplacian
            if HAS_IGL:
                V = vertices.astype(np.float64)
                F = faces.astype(np.int32)
                L_gt = -igl.cotmatrix(V, F)  # Negate for positive semi-definite
                M_gt = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)

            # PRED Laplacian
            if inference_result.get('stiffness_matrix') is not None:
                L_pred = inference_result['stiffness_matrix']
                M_pred = inference_result.get('mass_matrix')

            # Robust Laplacian
            if not self._skip_robust:
                k = self.original_k if self.original_k is not None else 30
                L_robust, M_robust = robust_laplacian.point_cloud_laplacian(vertices, n_neighbors=k)

            # Run Heat Method geodesic validation
            heat_geodesic_results = self.validate_heat_method_geodesics_step9(
                vertices=vertices,
                faces=faces,
                L_gt=L_gt,
                M_gt=M_gt,
                L_pred=L_pred,
                M_pred=M_pred,
                L_robust=L_robust,
                M_robust=M_robust,
                source_vertex_idx=None,  # Will auto-select centroid
                mesh_structure=mesh_structure,
                k_neighbors=self.original_k if self.original_k else 20
            )
            self.current_heat_geodesic_results = heat_geodesic_results

        except Exception as e:
            print(f"  [!] Heat Method geodesic validation failed: {e}")
            import traceback
            traceback.print_exc()

        # Print timing summary to console
        self._print_timing_summary()

        print(f"\n[OK] Comprehensive visualization completed for {Path(mesh_file_path).name}")

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
        skip_visualization = getattr(cfg, 'skip_visualization', False)  # Debug: skip polyscope

        if diagnostic_mode:
            print("\n" + "!" * 80)
            print("DIAGNOSTIC MODE ENABLED - Optimizations disabled for debugging")
            print("!" * 80 + "\n")

        if skip_robust:
            print("[DEBUG] skip_robust=True: Robust-laplacian computation will be skipped")

        if skip_visualization:
            print("[DEBUG] skip_visualization=True: Polyscope visualization will be skipped")

        # Store flags for use in process_batch
        self._diagnostic_mode = diagnostic_mode
        self._skip_robust = skip_robust
        self._skip_visualization = skip_visualization

        # Load trained model
        model = self.load_trained_model(ckpt_path, device, cfg, diagnostic_mode=diagnostic_mode)

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
        actual_k = len(first_data.pos) // len(first_data.center_indices)
        print(f"Detected k={actual_k} from dataset")

        # Warmup for torch.compile with ACTUAL k from dataset
        if not diagnostic_mode:
            self._warmup_model(model, device, k=actual_k)
        else:
            print("[DIAGNOSTIC] Skipping warmup")

        # Re-create dataloader since we consumed it (reset iterator)
        # Also reset seed to ensure same data order
        pl.seed_everything(cfg.globals.seed)
        data_loader = data_module.val_dataloader()
        if isinstance(data_loader, list):
            data_loader = data_loader[0]

        print(f"DataLoader ready with batch size: {data_loader.batch_size}")

        # Setup polyscope with UI callback (only if visualization is enabled)
        if not skip_visualization:
            self.setup_polyscope()
        else:
            print("[skip_visualization=True] Skipping polyscope setup")

        # Process all batches
        for batch_idx, batch_data in enumerate(data_loader):
            print(f"\n[>] Processing batch {batch_idx + 1}")

            try:
                self.process_batch(model, batch_data, batch_idx, device)

                if not skip_visualization:
                    print(f"\nVisualization ready for batch {batch_idx + 1}. Use the 'Reconstruction Settings' window to control PRED reconstruction method.")
                    print("Close window to continue to next batch.")
                    ps.show()

                    # === CLEANUP after polyscope visualization ===
                    # Polyscope uses OpenGL which can interfere with CUDA
                    # Force a full CUDA synchronization and cache clear to restore GPU state
                    if device.type == 'cuda':
                        print("[DEBUG] Cleaning up GPU state after polyscope...")
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                        # Do a quick warmup inference to restore torch.compile state
                        print("[DEBUG] Re-warming model after polyscope...")
                        from neural_local_laplacian.datasets.mesh_datasets import MeshPatchData
                        with torch.no_grad():
                            dummy = MeshPatchData(
                                pos=torch.randn(5000 * actual_k, 3, device=device),
                                x=torch.randn(5000 * actual_k, 3, device=device),
                                patch_idx=torch.arange(5000, device=device).repeat_interleave(actual_k),
                                vertex_indices=torch.randint(0, 5000, (5000 * actual_k,), device=device),
                                center_indices=torch.arange(5000, device=device)
                            )
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                                _ = model._forward_pass(dummy)
                            torch.cuda.synchronize()
                        del dummy
                        torch.cuda.empty_cache()
                        print("[DEBUG] Re-warm complete")
                else:
                    print(f"\n[skip_visualization=True] Skipping polyscope for batch {batch_idx + 1}")

            except Exception as e:
                print(f"[X] Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()

                user_input = input("Continue to next batch? (y/n): ").strip().lower()
                if user_input != 'y':
                    break

        print(f"\n[OK] Completed processing all batches!")


@hydra.main(version_base="1.2", config_path='./visualization_config')
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration."""

    # Create visualization config
    vis_config = VisualizationConfig(
        point_radius=0.005,
        num_eigenvectors_to_show=60,
        colormap='coolwarm',
        enable_eigenvalue_info=True,
        enable_correlation_analysis=True
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