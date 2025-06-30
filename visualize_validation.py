#!/usr/bin/env python3
"""
Eigenanalysis Visualization from Validation Data with Ground-Truth Comparison

This script loads validation data pkl files from a folder and visualizes:
1. The mesh point cloud
2. Ground-truth Laplacian eigenvectors as scalar fields on the mesh
3. Predicted Laplacian eigenvectors as scalar fields on the mesh
4. Eigenvalue comparison and analysis
5. Eigenvector correlation analysis
6. Predicted Laplacian matrix analysis

Updated to process all pkl files in a folder sequentially.
"""

import argparse
import pickle
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import glob

import numpy as np
import torch
import polyscope as ps


@dataclass
class VisualizationConfig:
    """Configuration for eigenanalysis visualization."""
    point_radius: float = 0.005
    show_wireframe: bool = False
    colormap: str = 'coolwarm'
    num_eigenvectors_to_show: int = 8
    enable_eigenvalue_info: bool = True
    enable_correlation_analysis: bool = True


class EigenanalysisVisualizer:
    """Visualizer for Laplacian eigenanalysis results with ground-truth comparison."""

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.validation_data = None
        self.current_batch_idx = 0

    def setup_polyscope(self):
        """Initialize and configure polyscope."""
        ps.init()
        ps.set_up_dir("z_up")
        ps.look_at(camera_location=[2.0, 2.0, 2.0], target=[0, 0, 0])
        ps.set_ground_plane_mode("none")
        ps.set_background_color((0.05, 0.05, 0.05))  # Dark background

    def load_validation_data(self, pkl_file_path: Path) -> Dict[str, Any]:
        """Load validation data from pickle file."""
        try:
            if pkl_file_path.suffix == '.zip':
                # Extract and load from zip file
                return self._load_from_zip(pkl_file_path)
            else:
                # Load directly from pickle file
                with open(pkl_file_path, 'rb') as f:
                    data = pickle.load(f)
                return data
        except Exception as e:
            raise RuntimeError(f"Failed to load validation data from {pkl_file_path}: {e}")

    def _load_from_zip(self, zip_file_path: Path) -> Dict[str, Any]:
        """Load validation data from zip file (W&B artifact)."""
        temp_dir = Path("temp_extraction")
        temp_dir.mkdir(exist_ok=True)

        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zipf:
                zipf.extractall(temp_dir)

            # Find the validation data pickle file
            pkl_files = list(temp_dir.glob("*_validation_data.pkl"))
            if not pkl_files:
                raise FileNotFoundError("No validation data pickle file found in zip")

            # Load the first validation data file found
            with open(pkl_files[0], 'rb') as f:
                data = pickle.load(f)

            return data

        finally:
            # Cleanup temp directory
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _reconstruct_laplacian_matrix(self, laplacian_data: Dict) -> Optional[Any]:
        """
        Reconstruct scipy sparse matrix from saved Laplacian data.

        Args:
            laplacian_data: Dictionary containing sparse matrix data

        Returns:
            Reconstructed scipy sparse matrix or None if data is invalid
        """
        if laplacian_data is None:
            return None

        try:
            import scipy.sparse

            # Reconstruct CSR matrix from saved components
            matrix = scipy.sparse.csr_matrix(
                (laplacian_data['data'],
                 laplacian_data['indices'],
                 laplacian_data['indptr']),
                shape=laplacian_data['shape'],
                dtype=laplacian_data['dtype']
            )

            print(f"  📊 Reconstructed Laplacian matrix: {matrix.shape}")
            print(f"      Non-zero entries: {matrix.nnz}")
            print(f"      Matrix format: {laplacian_data.get('matrix_format', 'unknown')}")

            return matrix

        except Exception as e:
            print(f"❌ Error reconstructing Laplacian matrix: {e}")
            return None

    def _compute_mean_curvature_from_laplacian(self, laplacian_matrix, vertices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute mean curvature vector and derived quantities from predicted Laplacian matrix.

        For a surface embedded in 3D, the Laplacian of position gives the mean curvature vector:
        Δr = H * n̂, where H is mean curvature and n̂ is the unit normal.

        Args:
            laplacian_matrix: Predicted Laplacian matrix (scipy sparse)
            vertices: Mesh vertices array of shape (N, 3)

        Returns:
            Dictionary containing:
            - mean_curvature_vector: (N, 3) - Δr = H * n̂
            - predicted_normals: (N, 3) - normalized directions of mean curvature vectors
            - predicted_mean_curvature: (N,) - magnitudes |Δr|
        """
        try:
            print(f"  🔄 Computing mean curvature from Laplacian...")

            # Apply Laplacian to vertex positions: Δr = L @ r
            mean_curvature_vector = laplacian_matrix @ vertices  # Shape: (N, 3)

            # Compute magnitudes (predicted mean curvature values)
            predicted_mean_curvature = np.linalg.norm(mean_curvature_vector, axis=1)  # Shape: (N,)

            # Compute predicted normals (normalized mean curvature vectors)
            # Handle zero vectors to avoid division by zero
            predicted_normals = np.zeros_like(mean_curvature_vector)
            non_zero_mask = predicted_mean_curvature > 1e-10

            predicted_normals[non_zero_mask] = (
                    mean_curvature_vector[non_zero_mask] /
                    predicted_mean_curvature[non_zero_mask, np.newaxis]
            )

            # For zero curvature points, use a default normal (e.g., z-up)
            predicted_normals[~non_zero_mask] = np.array([0, 0, 1])

            print(f"      Mean curvature range: [{predicted_mean_curvature.min():.6f}, {predicted_mean_curvature.max():.6f}]")
            print(f"      Mean curvature mean: {predicted_mean_curvature.mean():.6f}")
            print(f"      Zero curvature points: {(~non_zero_mask).sum()}/{len(predicted_mean_curvature)}")

            return {
                'mean_curvature_vector': mean_curvature_vector,
                'predicted_normals': predicted_normals,
                'predicted_mean_curvature': predicted_mean_curvature
            }

        except Exception as e:
            print(f"  ❌ Error computing mean curvature from Laplacian: {e}")
            return None

    def _add_curvature_visualizations(self, mesh_structure, curvature_data: Dict[str, np.ndarray],
                                      gt_normals: np.ndarray = None, gt_mean_curvature: np.ndarray = None):
        """
        Add predicted curvature and normal visualizations to the mesh with controlled ordering.

        Args:
            mesh_structure: Polyscope mesh structure
            curvature_data: Dictionary from _compute_mean_curvature_from_laplacian
            gt_normals: Ground-truth normals for comparison (optional)
            gt_mean_curvature: Ground-truth mean curvature for comparison (optional)
        """
        if curvature_data is None:
            return

        print(f"  🎨 Adding curvature visualizations...")

        # Use alphabetical prefixes to control ordering in polyscope UI

        # === MEAN CURVATURE (grouped together) ===
        # Add GT mean curvature first (if available) with 'A' prefix
        if gt_mean_curvature is not None:
            mesh_structure.add_scalar_quantity(
                name="A Mean Curvature - GT",
                values=gt_mean_curvature,
                enabled=False,
                cmap='plasma'  # Same colormap as predicted for comparison
            )

        # Add predicted mean curvature with 'B' prefix to come after GT
        mesh_structure.add_scalar_quantity(
            name="B Mean Curvature - PRED",
            values=curvature_data['predicted_mean_curvature'],
            enabled=False,
            cmap='plasma'  # Different colormap to distinguish from eigenvectors
        )

        # === NORMALS (grouped together) ===
        # Add GT normals first (if available) with 'C' prefix
        if gt_normals is not None:
            mesh_structure.add_vector_quantity(
                name="C Normals - GT",
                values=gt_normals * 0.05,  # Scale for visibility
                enabled=False,
                color=(0.0, 1.0, 1.0),  # Cyan color for GT normals
                vectortype="ambient"
            )

        # Add predicted normals with 'D' prefix to come after GT normals
        mesh_structure.add_vector_quantity(
            name="D Normals - PRED",
            values=curvature_data['predicted_normals'] * 0.05,  # Scale for visibility
            enabled=False,
            color=(1.0, 0.5, 0.0),  # Orange color for predicted normals
            vectortype="ambient"
        )

        # === OTHER QUANTITIES ===
        # Add mean curvature vector field with 'E' prefix
        mesh_structure.add_vector_quantity(
            name="E Mean Curvature Vector - PRED",
            values=curvature_data['mean_curvature_vector'] * 0.1,  # Scale for visibility
            enabled=False,
            color=(1.0, 0.0, 0.5),  # Pink color for curvature vectors
            vectortype="ambient"
        )

        # === COMPARISON METRICS ===
        # If ground-truth normals are available, add comparison metrics with 'F' prefix
        if gt_normals is not None:
            # Compute normal alignment (dot product)
            normal_alignment = np.sum(
                curvature_data['predicted_normals'] * gt_normals, axis=1
            )

            mesh_structure.add_scalar_quantity(
                name="F Normal Alignment (PRED vs GT)",
                values=normal_alignment,
                enabled=False,
                cmap='coolwarm'  # MODIFIED: Changed from 'RdBu' to 'coolwarm'
            )

            # Compute angular differences
            # Clamp to avoid numerical issues with arccos
            normal_alignment_clamped = np.clip(normal_alignment, -1.0, 1.0)
            angular_differences = np.arccos(np.abs(normal_alignment_clamped)) * 180 / np.pi

            mesh_structure.add_scalar_quantity(
                name="G Normal Angular Error (degrees)",
                values=angular_differences,
                enabled=False,
                cmap='coolwarm'  # MODIFIED: Changed from 'hot' to 'coolwarm'
            )

            print(f"      Normal alignment mean: {normal_alignment.mean():.4f}")
            print(f"      Angular error mean: {angular_differences.mean():.2f}°")
            print(f"      Angular error std: {angular_differences.std():.2f}°")

        # Print summary
        if gt_mean_curvature is not None:
            print(f"      Added GT mean curvature scalar field")
        print(f"      Added predicted mean curvature scalar field")
        if gt_normals is not None:
            print(f"      Added GT normals vector field")
        print(f"      Added predicted normals vector field")
        print(f"      Added mean curvature vector field")

    def _validate_batch_consistency(self, batch_result: Dict) -> bool:
        """
        Validate that batch result has consistent mesh and eigendata.

        Args:
            batch_result: Single validation batch result

        Returns:
            True if consistent, False otherwise
        """
        if 'mesh_data' not in batch_result:
            print("❌ Error: No mesh_data in batch result")
            return False

        mesh_data = batch_result['mesh_data']
        eigendata = batch_result.get('eigendata', {})

        # Check validation status from ValidationMeshUploader
        validation_status = mesh_data.get('validation_status', 'unknown')
        if validation_status != 'consistent':
            print(f"⚠️  Warning: Batch marked as '{validation_status}' during validation")
            return False

        # Double-check dimensions if eigendata is present
        if eigendata and 'predicted_eigenvectors' in eigendata and eigendata['predicted_eigenvectors'] is not None:
            mesh_vertices = len(mesh_data['vertices'])
            eigen_vertices = eigendata['predicted_eigenvectors'].shape[0]

            if mesh_vertices != eigen_vertices:
                print(f"❌ ERROR: Dimension mismatch detected in visualization!")
                print(f"  Mesh vertices: {mesh_vertices}")
                print(f"  Eigendata vertices: {eigen_vertices}")
                print(f"  Mesh file: {mesh_data.get('mesh_file_path', 'Unknown')}")
                print(f"  Validation status: {validation_status}")
                return False

            print(f"✅ Consistency validated: {mesh_vertices} vertices match between mesh and eigendata")

        return True

    def print_data_summary(self, data: Dict[str, Any]):
        """Print summary of loaded validation data with consistency information."""
        print("\n" + "=" * 60)
        print("VALIDATION DATA SUMMARY")
        print("=" * 60)
        print(f"Epoch: {data['epoch']}")
        print(f"Rank: {data['rank']}")
        print(f"Number of validation batches: {data['num_batches']}")
        print(f"Validation results: {len(data['validation_results'])}")

        if data['validation_results']:
            # Analyze consistency status across all results
            consistent_count = 0
            inconsistent_count = 0
            mesh_files = set()
            dimension_mismatches = []
            laplacian_count = 0

            for i, result in enumerate(data['validation_results']):
                if 'mesh_data' in result:
                    mesh_data = result['mesh_data']
                    mesh_files.add(mesh_data.get('mesh_file_path', 'Unknown'))

                    # Check validation status
                    status = mesh_data.get('validation_status', 'unknown')
                    if status == 'consistent':
                        consistent_count += 1
                    else:
                        inconsistent_count += 1

                    # Check for dimension mismatches
                    eigendata = result.get('eigendata', {})
                    if eigendata and 'predicted_eigenvectors' in eigendata:
                        mesh_vertices = len(mesh_data['vertices'])
                        eigen_vertices = eigendata['predicted_eigenvectors'].shape[0]
                        if mesh_vertices != eigen_vertices:
                            dimension_mismatches.append({
                                'batch_idx': i,
                                'mesh_vertices': mesh_vertices,
                                'eigen_vertices': eigen_vertices,
                                'mesh_file': mesh_data.get('mesh_file_path', 'Unknown')
                            })

                # Check for predicted Laplacian matrices
                if 'predicted_laplacian' in result and result['predicted_laplacian'] is not None:
                    laplacian_count += 1

            # Show summary of first result's mesh data
            first_result = data['validation_results'][0]
            if 'mesh_data' in first_result:
                mesh_data = first_result['mesh_data']
                print(f"First result mesh file: {mesh_data.get('mesh_file_path', 'Unknown')}")
                print(f"First result mesh vertices: {mesh_data['vertices'].shape if 'vertices' in mesh_data else 'N/A'}")
                print(f"First result mesh faces: {mesh_data['faces'].shape if 'faces' in mesh_data else 'N/A'}")
                if 'gt_eigenvalues' in mesh_data:
                    print(f"First result GT eigenvalues: {mesh_data['gt_eigenvalues'].shape}")

                # Show validation metadata if available
                if 'validation_status' in mesh_data:
                    print(f"First result validation status: {mesh_data['validation_status']}")
                if 'processed_vertices' in mesh_data:
                    print(f"First result processed vertices: {mesh_data['processed_vertices']}")

            # Check for predicted Laplacian in first result
            if 'predicted_laplacian' in first_result:
                laplacian_data = first_result['predicted_laplacian']
                if laplacian_data is not None:
                    print(f"First result predicted Laplacian: {laplacian_data['shape']} ({laplacian_data['nnz']} non-zeros)")
                else:
                    print(f"First result predicted Laplacian: None")

            print(f"\nConsistency Analysis:")
            print(f"  Consistent results: {consistent_count}")
            print(f"  Inconsistent results: {inconsistent_count}")
            print(f"  Unique mesh files used: {len(mesh_files)}")
            print(f"  Results with predicted Laplacians: {laplacian_count}/{len(data['validation_results'])}")

            if dimension_mismatches:
                print(f"  ❌ DIMENSION MISMATCHES DETECTED: {len(dimension_mismatches)}")
                for mismatch in dimension_mismatches[:3]:  # Show first 3
                    print(f"    Batch {mismatch['batch_idx']}: {mismatch['mesh_vertices']} mesh vs {mismatch['eigen_vertices']} eigen vertices")
                if len(dimension_mismatches) > 3:
                    print(f"    ... and {len(dimension_mismatches) - 3} more")
            else:
                print(f"  ✅ No dimension mismatches detected")

            if len(mesh_files) <= 3:  # Show mesh files if not too many
                for i, mesh_file in enumerate(sorted(mesh_files)):
                    print(f"  {i + 1}. {Path(mesh_file).name}")

            first_result = data['validation_results'][0]
            print(f"\nFirst batch metrics: {list(first_result['metrics'].keys())}")

            if 'eigendata' in first_result and first_result['eigendata']:
                eigendata = first_result['eigendata']
                print(f"Predicted eigenvalues shape: {eigendata['predicted_eigenvalues'].shape if 'predicted_eigenvalues' in eigendata else 'N/A'}")
                print(f"Predicted eigenvectors shape: {eigendata['predicted_eigenvectors'].shape if 'predicted_eigenvectors' in eigendata else 'N/A'}")
                print(f"Matrix size: {eigendata.get('matrix_size', 'N/A')}")

        print("=" * 60)

    def _get_mesh_data_for_batch(self, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get mesh data for a specific batch.

        Args:
            batch_idx: Index of the batch

        Returns:
            Dictionary containing mesh vertices, faces, normals, and ground-truth eigendecomposition
        """
        if batch_idx >= len(self.validation_data['validation_results']):
            raise IndexError(f"Batch index {batch_idx} out of range")

        batch_result = self.validation_data['validation_results'][batch_idx]
        if 'mesh_data' not in batch_result:
            raise ValueError(f"No mesh data found for batch {batch_idx}")

        mesh_data = batch_result['mesh_data']

        return {
            'vertices': mesh_data['vertices'],
            'vertex_normals': mesh_data['vertex_normals'],
            'faces': mesh_data['faces'],
            'gt_eigenvalues': mesh_data['gt_eigenvalues'],
            'gt_eigenvectors': mesh_data['gt_eigenvectors'],
            'mesh_file_path': mesh_data.get('mesh_file_path', 'Unknown'),
            'num_vertices': mesh_data.get('num_vertices', len(mesh_data['vertices'])),
            'num_faces': mesh_data.get('num_faces', len(mesh_data['faces']))
        }

    def print_eigenvalue_analysis(self, gt_eigenvalues: torch.Tensor, pred_eigenvalues: torch.Tensor, batch_idx: int):
        """Print detailed eigenvalue comparison analysis."""
        gt_eigenvals_np = gt_eigenvalues.cpu().numpy()
        pred_eigenvals_np = pred_eigenvalues.cpu().numpy()

        print(f"\n" + "-" * 70)
        print(f"EIGENVALUE COMPARISON ANALYSIS - Batch {batch_idx}")
        print("-" * 70)

        # Ground-truth analysis
        print("GROUND-TRUTH EIGENVALUES:")
        print(f"  Number of eigenvalues: {len(gt_eigenvals_np)}")
        print(f"  First eigenvalue (should be ~0): {gt_eigenvals_np[0]:.2e}")
        if len(gt_eigenvals_np) > 1:
            print(f"  Second eigenvalue (Fiedler): {gt_eigenvals_np[1]:.6f}")
            print(f"  Spectral gap: {gt_eigenvals_np[1] - gt_eigenvals_np[0]:.6f}")
        print(f"  Largest eigenvalue: {gt_eigenvals_np[-1]:.6f}")
        print(f"  Mean eigenvalue: {gt_eigenvals_np.mean():.6f}")

        # Predicted analysis
        print("\nPREDICTED EIGENVALUES:")
        print(f"  Number of eigenvalues: {len(pred_eigenvals_np)}")
        print(f"  First eigenvalue (should be ~0): {pred_eigenvals_np[0]:.2e}")
        if len(pred_eigenvals_np) > 1:
            print(f"  Second eigenvalue (Fiedler): {pred_eigenvals_np[1]:.6f}")
            print(f"  Spectral gap: {pred_eigenvals_np[1] - pred_eigenvals_np[0]:.6f}")
        print(f"  Largest eigenvalue: {pred_eigenvals_np[-1]:.6f}")
        print(f"  Mean eigenvalue: {pred_eigenvals_np.mean():.6f}")

        # Comparison metrics
        min_len = min(len(gt_eigenvals_np), len(pred_eigenvals_np))
        if min_len > 0:
            gt_subset = gt_eigenvals_np[:min_len]
            pred_subset = pred_eigenvals_np[:min_len]

            # Compute errors
            abs_errors = np.abs(pred_subset - gt_subset)
            rel_errors = abs_errors / (np.abs(gt_subset) + 1e-10)

            print(f"\nCOMPARISON METRICS (first {min_len} eigenvalues):")
            print(f"  Mean absolute error: {abs_errors.mean():.6f}")
            print(f"  Max absolute error: {abs_errors.max():.6f}")
            print(f"  Mean relative error: {rel_errors.mean():.6f}")
            print(f"  Max relative error: {rel_errors.max():.6f}")
            print(f"  Correlation coefficient: {np.corrcoef(gt_subset, pred_subset)[0, 1]:.6f}")

        # Check positive semi-definiteness
        print("\nPOSITIVE SEMI-DEFINITENESS CHECK:")
        if gt_eigenvals_np[0] < -1e-10:
            print(f"  ⚠️  GT: First eigenvalue is significantly negative ({gt_eigenvals_np[0]:.2e})")
        else:
            print("  ✅ GT: Laplacian appears to be positive semi-definite")

        if pred_eigenvals_np[0] < -1e-10:
            print(f"  ⚠️  PRED: First eigenvalue is significantly negative ({pred_eigenvals_np[0]:.2e})")
        else:
            print("  ✅ PRED: Laplacian appears to be positive semi-definite")

        print("-" * 70)

    def compute_eigenvector_correlations(self, gt_eigenvectors: torch.Tensor, pred_eigenvectors: torch.Tensor) -> np.ndarray:
        """Compute correlation matrix between ground-truth and predicted eigenvectors."""
        gt_eigenvecs_np = gt_eigenvectors.cpu().numpy()
        pred_eigenvecs_np = pred_eigenvectors.cpu().numpy()

        min_cols = min(gt_eigenvecs_np.shape[1], pred_eigenvecs_np.shape[1])
        correlation_matrix = np.zeros((min_cols, min_cols))

        for i in range(min_cols):
            for j in range(min_cols):
                # Compute absolute correlation (eigenvectors can have sign ambiguity)
                corr = np.abs(np.corrcoef(gt_eigenvecs_np[:, i], pred_eigenvecs_np[:, j])[0, 1])
                correlation_matrix[i, j] = corr

        return correlation_matrix

    def print_eigenvector_correlation_analysis(self, correlation_matrix: np.ndarray, batch_idx: int):
        """Print eigenvector correlation analysis."""
        print(f"\n" + "-" * 50)
        print(f"EIGENVECTOR CORRELATION ANALYSIS - Batch {batch_idx}")
        print("-" * 50)

        # Find best matches for each GT eigenvector
        print("Best matches for each GT eigenvector:")
        for i in range(min(8, correlation_matrix.shape[0])):  # Show first 8
            best_match_idx = np.argmax(correlation_matrix[i, :])
            best_correlation = correlation_matrix[i, best_match_idx]
            print(f"  GT Eigenvector {i} ↔ Pred Eigenvector {best_match_idx}: {best_correlation:.4f}")

        # Overall statistics
        diagonal_corrs = np.diag(correlation_matrix)
        print(f"\nDiagonal correlations (GT_i vs PRED_i):")
        print(f"  Mean: {diagonal_corrs.mean():.4f}")
        print(f"  Min: {diagonal_corrs.min():.4f}")
        print(f"  Max: {diagonal_corrs.max():.4f}")

        # Check if eigenvectors are well-aligned
        well_aligned = diagonal_corrs > 0.8
        print(f"  Well-aligned (>0.8): {well_aligned.sum()}/{len(diagonal_corrs)}")

        print("-" * 50)

    def visualize_mesh(self, vertices: torch.Tensor, vertex_normals: torch.Tensor, faces: torch.Tensor):
        """Visualize the base mesh point cloud."""
        vertices_np = vertices.cpu().numpy()
        normals_np = vertex_normals.cpu().numpy()
        faces_np = faces.cpu().numpy()

        # Register mesh as surface mesh (with faces)
        if len(faces_np) > 0:
            mesh_surface = ps.register_surface_mesh(
                name="Mesh Surface",
                vertices=vertices_np,
                faces=faces_np,
                enabled=True
            )

            # Add vertex normals
            mesh_surface.add_vector_quantity(
                name="normals",
                values=normals_np * 0.05,  # Scale for visibility
                enabled=False,  # Disabled by default
                color=(0.0, 1.0, 1.0),
                vectortype="ambient"
            )

            print(f"Registered mesh surface with {len(vertices_np)} vertices and {len(faces_np)} faces")
            return mesh_surface
        else:
            # Fallback to point cloud if no faces
            mesh_cloud = ps.register_point_cloud(
                name="Mesh",
                points=vertices_np,
                radius=self.config.point_radius,
                enabled=True
            )

            # Add vertex normals
            mesh_cloud.add_vector_quantity(
                name="normals",
                values=normals_np * 0.05,  # Scale for visibility
                enabled=False,  # Disabled by default
                color=(0.0, 1.0, 1.0),
                vectortype="ambient"
            )

            print(f"Registered mesh point cloud with {len(vertices_np)} vertices")
            return mesh_cloud

    def visualize_eigenvectors(self, mesh_structure, gt_eigenvectors: torch.Tensor, gt_eigenvalues: torch.Tensor,
                               pred_eigenvectors: Optional[torch.Tensor] = None, pred_eigenvalues: Optional[torch.Tensor] = None):
        """Add both ground-truth and predicted eigenvector scalar fields to the mesh, organized for easy comparison."""
        gt_eigenvecs_np = gt_eigenvectors.cpu().numpy()
        gt_eigenvals_np = gt_eigenvalues.cpu().numpy()

        num_to_show = min(self.config.num_eigenvectors_to_show, gt_eigenvecs_np.shape[1])

        # Check if we have predicted data
        has_predicted = pred_eigenvectors is not None and pred_eigenvalues is not None

        if has_predicted:
            pred_eigenvecs_np = pred_eigenvectors.cpu().numpy()
            pred_eigenvals_np = pred_eigenvalues.cpu().numpy()
            num_pred_to_show = min(num_to_show, pred_eigenvecs_np.shape[1])
            print(f"Adding {num_to_show} GT and {num_pred_to_show} predicted eigenvector pairs...")
        else:
            print(f"Adding {num_to_show} ground-truth eigenvector scalar fields...")

        # Add eigenvectors in pairs using numerical prefixes to control alphabetical ordering
        for i in range(num_to_show):
            # Add GT eigenvector with numerical prefix
            gt_eigenvector = gt_eigenvecs_np[:, i]
            gt_eigenvalue = gt_eigenvals_np[i]

            # Create descriptive name for GT with zero-padded numbering and prefix for ordering
            if i == 0:
                gt_name = f"Eigenvector {i:02d}a GT (λ={gt_eigenvalue:.2e}, constant)"
            elif i == 1:
                gt_name = f"Eigenvector {i:02d}a GT (λ={gt_eigenvalue:.6f}, Fiedler)"
            else:
                gt_name = f"Eigenvector {i:02d}a GT (λ={gt_eigenvalue:.6f})"

            # Add GT as scalar quantity
            mesh_structure.add_scalar_quantity(
                name=gt_name,
                values=gt_eigenvector,
                enabled=(i == 1),  # Enable GT Fiedler vector by default
                cmap=self.config.colormap
            )

            print(f"  {gt_name}: range=[{gt_eigenvector.min():.4f}, {gt_eigenvector.max():.4f}]")

            # Add corresponding predicted eigenvector if available
            if has_predicted and i < num_pred_to_show:
                pred_eigenvector = pred_eigenvecs_np[:, i]
                pred_eigenvalue = pred_eigenvals_np[i]

                # Create descriptive name for predictions with 'b' suffix to come after GT
                if i == 0:
                    pred_name = f"Eigenvector {i:02d}b PRED (λ={pred_eigenvalue:.2e}, constant)"
                elif i == 1:
                    pred_name = f"Eigenvector {i:02d}b PRED (λ={pred_eigenvalue:.6f}, Fiedler)"
                else:
                    pred_name = f"Eigenvector {i:02d}b PRED (λ={pred_eigenvalue:.6f})"

                # Add PRED as scalar quantity (will be placed right after corresponding GT due to alphabetical sorting)
                mesh_structure.add_scalar_quantity(
                    name=pred_name,
                    values=pred_eigenvector,
                    enabled=False,  # Disabled by default
                    cmap=self.config.colormap  # Same colormap as GT for easy comparison
                )

                print(f"  {pred_name}: range=[{pred_eigenvector.min():.4f}, {pred_eigenvector.max():.4f}]")

    def visualize_batch(self, batch_idx: int = 0):
        """Visualize eigenanalysis results for a specific batch with ground-truth comparison."""
        if not self.validation_data:
            print("No validation data loaded!")
            return

        if batch_idx >= len(self.validation_data['validation_results']):
            print(f"Batch index {batch_idx} out of range (max: {len(self.validation_data['validation_results']) - 1})")
            return

        self.current_batch_idx = batch_idx
        batch_result = self.validation_data['validation_results'][batch_idx]

        print(f"\nVisualizing batch {batch_idx} with ground-truth comparison...")

        # CRITICAL: Validate consistency before visualization
        if not self._validate_batch_consistency(batch_result):
            print(f"❌ Skipping visualization for batch {batch_idx} due to inconsistency")
            print("   This batch has mismatched mesh and eigendata dimensions.")
            print("   Check ValidationMeshUploader logs for details.")
            return

        # Clear previous visualization
        ps.remove_all_structures()

        # Get mesh data for this specific batch
        try:
            mesh_data = self._get_mesh_data_for_batch(batch_idx)
        except Exception as e:
            print(f"Error getting mesh data for batch {batch_idx}: {e}")
            return

        vertices = mesh_data['vertices']
        vertex_normals = mesh_data['vertex_normals']
        faces = mesh_data['faces']
        gt_eigenvalues = mesh_data['gt_eigenvalues']
        gt_eigenvectors = mesh_data['gt_eigenvectors']

        print(f"Mesh file: {mesh_data['mesh_file_path']}")
        print(f"Mesh has {mesh_data['num_vertices']} vertices and {mesh_data['num_faces']} faces")

        # Show validation metadata if available
        if 'validation_status' in mesh_data:
            print(f"Validation status: {mesh_data['validation_status']}")
        if 'processed_vertices' in mesh_data:
            print(f"Processed vertices: {mesh_data['processed_vertices']}")
        if 'mesh_idx' in mesh_data:
            print(f"Original mesh index: {mesh_data['mesh_idx']}")

        # Show predicted Laplacian matrix info if available
        if 'predicted_laplacian' in batch_result:
            laplacian_data = batch_result['predicted_laplacian']
            if laplacian_data is not None:
                print(f"Predicted Laplacian: {laplacian_data['shape']} ({laplacian_data['nnz']} non-zeros)")

                # Reconstruct and analyze the matrix
                laplacian_matrix = self._reconstruct_laplacian_matrix(laplacian_data)
                if laplacian_matrix is not None:
                    # Show some basic matrix properties
                    print(f"  Matrix properties:")
                    print(f"    Symmetry check: {abs(laplacian_matrix - laplacian_matrix.T).max():.2e}")

                    # Check row sums (should be close to zero for Laplacian)
                    row_sums = np.array(laplacian_matrix.sum(axis=1)).flatten()
                    print(f"    Row sum range: [{row_sums.min():.2e}, {row_sums.max():.2e}]")
                    print(f"    Row sum mean: {row_sums.mean():.2e}")

                    # CRITICAL ADDITION: Compute mean curvature from Laplacian
                    vertices_np = vertices.cpu().numpy()
                    curvature_data = self._compute_mean_curvature_from_laplacian(laplacian_matrix, vertices_np)

                else:
                    curvature_data = None
            else:
                print(f"Predicted Laplacian: None")
                curvature_data = None
        else:
            curvature_data = None

        # Visualize base mesh
        mesh_structure = self.visualize_mesh(vertices, vertex_normals, faces)

        # Get predicted eigenanalysis results if available
        pred_eigenvalues = None
        pred_eigenvectors = None

        if 'eigendata' in batch_result and batch_result['eigendata']:
            eigendata = batch_result['eigendata']
            if 'predicted_eigenvalues' in eigendata and 'predicted_eigenvectors' in eigendata:
                pred_eigenvalues = eigendata['predicted_eigenvalues']
                pred_eigenvectors = eigendata['predicted_eigenvectors']

                # Additional consistency check during visualization
                if pred_eigenvectors is not None:
                    eigen_vertices = pred_eigenvectors.shape[0]
                    mesh_vertices = len(vertices)
                    print(f"Final consistency check: Mesh({mesh_vertices}) == Eigen({eigen_vertices}) -> {mesh_vertices == eigen_vertices}")

        # Print eigenvalue analysis
        if self.config.enable_eigenvalue_info and pred_eigenvalues is not None:
            self.print_eigenvalue_analysis(gt_eigenvalues, pred_eigenvalues, batch_idx)

        # Print eigenvector correlation analysis
        if self.config.enable_correlation_analysis and pred_eigenvectors is not None:
            correlation_matrix = self.compute_eigenvector_correlations(gt_eigenvectors, pred_eigenvectors)
            self.print_eigenvector_correlation_analysis(correlation_matrix, batch_idx)

        # Visualize eigenvectors (both GT and predicted)
        self.visualize_eigenvectors(
            mesh_structure,
            gt_eigenvectors,
            gt_eigenvalues,
            pred_eigenvectors,
            pred_eigenvalues
        )

        # CRITICAL ADDITION: Add curvature visualizations from predicted Laplacian
        if curvature_data is not None:
            gt_normals_np = vertex_normals.cpu().numpy()

            # Extract GT mean curvature from mesh data (computed during mesh loading with libigl)
            gt_mean_curvature_np = None
            if 'mesh_data' in batch_result:
                mesh_data_batch = batch_result['mesh_data']
                if 'gt_mean_curvature' in mesh_data_batch and mesh_data_batch['gt_mean_curvature'] is not None:
                    gt_mean_curvature_np = mesh_data_batch['gt_mean_curvature'].cpu().numpy()
                    print(f"  📐 Using cached GT mean curvature: range=[{gt_mean_curvature_np.min():.6f}, {gt_mean_curvature_np.max():.6f}]")
                else:
                    print(f"  ⚠️  GT mean curvature not available (libigl might not be installed or computation failed)")

            self._add_curvature_visualizations(
                mesh_structure,
                curvature_data,
                gt_normals_np,
                gt_mean_curvature_np
            )
            print(f"✅ Added predicted curvature and normal visualizations")

        # Print batch metrics
        metrics = batch_result['metrics']
        print(f"\nBatch {batch_idx} metrics:")
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value}")

    def run_automatic_iteration(self, pkl_file_path: Path):
        """Run visualization that automatically iterates through all validation results."""
        # Load validation data
        print(f"Loading validation data from: {pkl_file_path}")
        self.validation_data = self.load_validation_data(pkl_file_path)

        # Print summary
        self.print_data_summary(self.validation_data)

        if not self.validation_data['validation_results']:
            print("No validation results found in data!")
            return

        num_batches = len(self.validation_data['validation_results'])
        print(f"\n{'=' * 60}")
        print("AUTOMATIC ITERATION MODE")
        print("=" * 60)
        print(f"Will iterate through {num_batches} validation results")
        print("Close each window to proceed to the next result")
        print("Features:")
        print("  - Ground-truth eigenvectors: Eigenvector XXa GT")
        print("  - Predicted eigenvectors: Eigenvector XXb PRED")
        print("  - GT and PRED mean curvature side-by-side")
        print("  - GT and PRED normals side-by-side")
        print("  - Both use the same colormap for easy comparison")
        print("  - Eigenvalue comparison and correlation analysis in terminal")
        print("  - Each result shows its own mesh data")
        print("  - Predicted Laplacian matrix analysis included")
        print("  - Mean curvature and normal predictions from Laplacian")
        print("  - Normal alignment analysis (GT vs Predicted)")
        print("=" * 60)

        # Iterate through all validation results
        for batch_idx in range(num_batches):
            print(f"\n🔍 Showing validation result {batch_idx + 1}/{num_batches}")

            # Setup polyscope for this iteration
            self.setup_polyscope()

            # Visualize this batch
            self.visualize_batch(batch_idx)

            print(f"Window opened for result {batch_idx + 1}. Close window to continue to next result.")

            # Show visualization (blocks until window is closed)
            ps.show()

            # Clear structures for next iteration
            ps.remove_all_structures()

        print(f"\n✅ Completed visualization of all {num_batches} validation results!")

    def find_pkl_files(self, folder_path: Path) -> List[Path]:
        """Find all pkl files in the given folder."""
        pkl_files = []

        # Look for direct pkl files
        pkl_files.extend(folder_path.glob("*.pkl"))

        # Look for zip files that might contain pkl files
        zip_files = list(folder_path.glob("*.zip"))

        # Sort all files for consistent ordering
        all_files = sorted(pkl_files + zip_files)

        if not all_files:
            raise FileNotFoundError(f"No .pkl or .zip files found in {folder_path}")

        return all_files

    def process_folder(self, folder_path: Path):
        """Process all pkl files in the given folder."""
        print(f"\n{'=' * 80}")
        print(f"PROCESSING FOLDER: {folder_path}")
        print("=" * 80)

        # Find all pkl/zip files
        pkl_files = self.find_pkl_files(folder_path)

        print(f"Found {len(pkl_files)} validation files:")
        for i, file_path in enumerate(pkl_files, 1):
            print(f"  {i}. {file_path.name}")

        # Process each file
        for file_idx, pkl_file_path in enumerate(pkl_files, 1):
            print(f"\n{'*' * 60}")
            print(f"PROCESSING FILE {file_idx}/{len(pkl_files)}: {pkl_file_path.name}")
            print("*" * 60)

            try:
                # Load and visualize this file
                self.run_automatic_iteration(pkl_file_path)

                # Ask user if they want to continue to next file
                if file_idx < len(pkl_files):
                    print(f"\n{'─' * 60}")
                    print(f"Completed processing {pkl_file_path.name}")
                    print(f"Next: {pkl_files[file_idx].name}")

                    user_input = input("\nPress Enter to continue to next file, or 'q' to quit: ").strip().lower()
                    if user_input == 'q':
                        print("Visualization stopped by user.")
                        break

            except Exception as e:
                print(f"❌ Error processing {pkl_file_path.name}: {e}")
                import traceback
                traceback.print_exc()

                # Ask if user wants to continue despite error
                user_input = input("\nError occurred. Press Enter to continue to next file, or 'q' to quit: ").strip().lower()
                if user_input == 'q':
                    break

        print(f"\n✅ Completed processing all files in {folder_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Visualize Laplacian eigenanalysis with ground-truth comparison")
    parser.add_argument("folder_path", type=str, help="Path to folder containing validation data pickle files")
    parser.add_argument("--point_radius", type=float, default=0.005, help="Point radius for mesh visualization")
    parser.add_argument("--num_eigenvectors", type=int, default=40, help="Number of eigenvectors to show")
    parser.add_argument("--colormap", type=str, default='coolwarm', help="Colormap for GT eigenvectors")
    parser.add_argument("--disable_correlation", action='store_true', help="Disable eigenvector correlation analysis")

    args = parser.parse_args()

    # Validate folder path
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Create visualization config
    vis_config = VisualizationConfig(
        point_radius=args.point_radius,
        num_eigenvectors_to_show=args.num_eigenvectors,
        colormap=args.colormap,
        enable_correlation_analysis=not args.disable_correlation
    )

    # Create visualizer and process folder
    visualizer = EigenanalysisVisualizer(config=vis_config)
    visualizer.process_folder(folder_path)


if __name__ == "__main__":
    main()