#!/usr/bin/env python3
"""
Eigenanalysis Visualization from Validation Data with Ground-Truth Comparison

This script loads validation data pkl files from a folder and visualizes:
1. The mesh point cloud
2. Ground-truth Laplacian eigenvectors as scalar fields on the mesh (computed locally)
3. Original predicted Laplacian eigenvectors as scalar fields on the mesh
4. Corrected predicted Laplacian eigenvectors (after normal orientation correction)
5. Eigenvalue comparison and analysis
6. Eigenvector correlation analysis
7. Normal orientation comparison (GT vs Original vs Corrected)

Updated to process all pkl files in a folder sequentially and compute GT data locally.
"""

import argparse
import pickle
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import glob

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

# For eigendecomposition
import scipy.sparse
import scipy.sparse.linalg


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
    """Visualizer for Laplacian eigenanalysis results with ground-truth comparison and normal correction."""

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

    def _compute_gt_data_locally(self, vertices: np.ndarray, faces: np.ndarray, mesh_file_path: str, k: int = 50) -> Dict[str, np.ndarray]:
        """
        Compute all ground-truth data locally using the same methods as ValidationMeshUploader.

        Args:
            vertices: Mesh vertices array of shape (N, 3)
            faces: Mesh faces array of shape (F, 3)
            mesh_file_path: Path to mesh file (for logging)
            k: Number of eigenvalues to compute

        Returns:
            Dictionary containing all GT quantities
        """
        print(f"Computing GT data locally for: {Path(mesh_file_path).name}")
        print(f"Mesh has {len(vertices)} vertices and {len(faces)} faces")

        # Compute GT normals using trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        gt_vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)
        print(f"Computed GT vertex normals")

        # Compute GT mean curvature using libigl if available
        gt_mean_curvature = None
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

                print(f"GT mean curvature range: [{gt_mean_curvature.min():.6f}, {gt_mean_curvature.max():.6f}]")
                print(f"GT mean curvature mean: {gt_mean_curvature.mean():.6f}")

            except Exception as e:
                print(f"Warning: Failed to compute GT mean curvature with libigl: {e}")
                gt_mean_curvature = None
        else:
            print("libigl not available, skipping GT mean curvature computation")

        # Compute GT Laplacian eigendecomposition using PyFM
        print("Computing GT Laplacian eigendecomposition using PyFM...")
        try:
            # Create PyFM TriMesh object
            pyfm_mesh = TriMesh(vertices, faces)

            # Process the mesh and compute the Laplacian spectrum
            pyfm_mesh.process(k=k, intrinsic=False, verbose=False)

            # Retrieve eigenvalues, eigenfunctions, and vertex areas
            gt_eigenvalues = pyfm_mesh.eigenvalues  # Shape: (k,)
            gt_eigenvectors = pyfm_mesh.eigenvectors  # Shape: (num_vertices, k)
            vertex_areas = pyfm_mesh.vertex_areas  # Shape: (num_vertices,)

            print(f"Computed {len(gt_eigenvalues)} GT eigenvalues")
            print(f"GT eigenvalue range: [{gt_eigenvalues[0]:.2e}, {gt_eigenvalues[-1]:.6f}]")

        except Exception as e:
            print(f"Warning: Failed to compute GT eigendecomposition: {e}")
            gt_eigenvalues = None
            gt_eigenvectors = None
            vertex_areas = None

        return {
            'gt_vertex_normals': gt_vertex_normals,
            'gt_mean_curvature': gt_mean_curvature,
            'gt_eigenvalues': gt_eigenvalues,
            'gt_eigenvectors': gt_eigenvectors,
            'vertex_areas': vertex_areas
        }

    def _orient_normals_consistently_with_open3d(self, vertices: np.ndarray, faces: np.ndarray,
                                                 predicted_normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use Open3D PointCloud to orient normals consistently and determine which normals were flipped.

        Args:
            vertices: Mesh vertices array of shape (N, 3)
            faces: Mesh faces array of shape (F, 3) (not used, kept for interface compatibility)
            predicted_normals: Original predicted normals of shape (N, 3)

        Returns:
            Tuple of (oriented_normals, flipped_mask)
            - oriented_normals: Consistently oriented normals of shape (N, 3)
            - flipped_mask: Boolean array of shape (N,) indicating which vertices had normals flipped
        """
        print("Applying consistent normal orientation using Open3D PointCloud...")

        try:
            # Create Open3D point cloud
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(vertices)
            o3d_pcd.normals = o3d.utility.Vector3dVector(predicted_normals)

            # Orient normals consistently using point cloud method
            # This will automatically determine a consistent orientation
            o3d_pcd.orient_normals_consistent_tangent_plane(k=30)

            # Extract consistently oriented normals
            oriented_normals = np.asarray(o3d_pcd.normals)

            # Determine which normals were flipped by comparing with original predictions
            # Use dot product: if dot < 0, the normal was flipped
            dot_products = np.sum(predicted_normals * oriented_normals, axis=1)
            flipped_mask = dot_products < 0

            num_flipped = np.sum(flipped_mask)
            print(f"Open3D PointCloud normal orientation completed:")
            print(f"  Total vertices: {len(vertices)}")
            print(f"  Normals flipped: {num_flipped} ({100 * num_flipped / len(vertices):.1f}%)")

            return oriented_normals, flipped_mask

        except Exception as e:
            print(f"Warning: Open3D PointCloud normal orientation failed: {e}")
            print("Using original predicted normals without orientation correction")
            return predicted_normals, np.zeros(len(predicted_normals), dtype=bool)

    def _create_corrected_laplacian(self, laplacian_matrix: scipy.sparse.csr_matrix,
                                    flipped_mask: np.ndarray) -> scipy.sparse.csr_matrix:
        """
        Create corrected Laplacian matrix by flipping rows for vertices with flipped normals.
        Uses vectorized operations for efficiency.

        Args:
            laplacian_matrix: Original Laplacian matrix
            flipped_mask: Boolean array indicating which vertices had normals flipped

        Returns:
            Corrected Laplacian matrix with flipped rows
        """
        print("Creating corrected Laplacian matrix...")

        # Find indices of flipped vertices
        flipped_indices = np.where(flipped_mask)[0]

        if len(flipped_indices) == 0:
            print("No normals were flipped, returning original Laplacian matrix")
            return laplacian_matrix.copy()

        print(f"Vectorized flipping of Laplacian rows for {len(flipped_indices)} vertices...")

        # Create sign matrix for vectorized row flipping
        # +1 for non-flipped vertices, -1 for flipped vertices
        sign_vector = np.ones(laplacian_matrix.shape[0])
        sign_vector[flipped_mask] = -1

        # Convert to diagonal sparse matrix for efficient multiplication
        sign_matrix = scipy.sparse.diags(sign_vector, format='csr')

        # Vectorized row flipping: S @ L (only left multiplication)
        # This flips only the rows, not the columns
        corrected_laplacian = sign_matrix @ laplacian_matrix

        print(f"Vectorized Laplacian row correction completed successfully")
        print(f"  Flipped {len(flipped_indices)} rows out of {laplacian_matrix.shape[0]} total rows")

        return corrected_laplacian

    def _compute_corrected_quantities(self, corrected_laplacian: scipy.sparse.csr_matrix,
                                      vertices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute corrected mean curvature normals and magnitudes using the corrected Laplacian.

        Args:
            corrected_laplacian: Corrected Laplacian matrix
            vertices: Mesh vertices array of shape (N, 3)

        Returns:
            Dictionary containing corrected quantities
        """
        print("Computing corrected quantities from corrected Laplacian...")

        # Apply corrected Laplacian to vertex positions: Δr = L @ r
        corrected_mean_curvature_vector = corrected_laplacian @ vertices  # Shape: (N, 3)

        # Compute magnitudes (corrected mean curvature values)
        corrected_predicted_mean_curvature = np.linalg.norm(corrected_mean_curvature_vector, axis=1)  # Shape: (N,)

        # Compute corrected normals (normalized mean curvature vectors)
        corrected_predicted_normals = np.zeros_like(corrected_mean_curvature_vector)
        non_zero_mask = corrected_predicted_mean_curvature > 1e-10

        corrected_predicted_normals[non_zero_mask] = (
                corrected_mean_curvature_vector[non_zero_mask] /
                corrected_predicted_mean_curvature[non_zero_mask, np.newaxis]
        )

        # For zero curvature points, use a default normal (e.g., z-up)
        corrected_predicted_normals[~non_zero_mask] = np.array([0, 0, 1])

        print(f"Corrected mean curvature range: [{corrected_predicted_mean_curvature.min():.6f}, {corrected_predicted_mean_curvature.max():.6f}]")
        print(f"Corrected mean curvature mean: {corrected_predicted_mean_curvature.mean():.6f}")
        print(f"Corrected zero curvature points: {(~non_zero_mask).sum()}/{len(corrected_predicted_mean_curvature)}")

        return {
            'corrected_mean_curvature_vector': corrected_mean_curvature_vector,
            'corrected_predicted_normals': corrected_predicted_normals,
            'corrected_predicted_mean_curvature': corrected_predicted_mean_curvature
        }

    def _recompute_eigendecomposition(self, laplacian_matrix: scipy.sparse.csr_matrix, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recompute eigendecomposition using the same method as ValidationMeshUploader.

        Args:
            laplacian_matrix: Laplacian matrix (original or corrected)
            k: Number of eigenvalues to compute

        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted in ascending order by eigenvalue
        """
        print(f"Computing eigendecomposition using shift-invert mode...")

        try:
            # Use the exact same method as ValidationMeshUploader._compute_eigendecomposition
            # Use shift-invert mode with sigma=-0.01 to find eigenvalues closest to 0
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(laplacian_matrix, k=k, sigma=-0.01)

            # Sort eigenvalues and eigenvectors in ascending order
            sort_indices = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sort_indices]
            eigenvectors = eigenvectors[:, sort_indices]

            print(f"Eigendecomposition completed: {k} eigenvalues computed")
            print(f"Eigenvalue range: [{eigenvalues[0]:.2e}, {eigenvalues[-1]:.6f}]")

            return eigenvalues, eigenvectors

        except Exception as e:
            print(f"Error computing eigendecomposition: {e}")
            return None, None

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

    def _compute_original_quantities_from_laplacian(self, laplacian_matrix, vertices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute original mean curvature vector and derived quantities from predicted Laplacian matrix.

        For a surface embedded in 3D, the Laplacian of position gives the mean curvature vector:
        Δr = H * n̂, where H is mean curvature and n̂ is the unit normal.

        Args:
            laplacian_matrix: Predicted Laplacian matrix (scipy sparse)
            vertices: Mesh vertices array of shape (N, 3)

        Returns:
            Dictionary containing original predicted quantities
        """
        try:
            print(f"  🔄 Computing original quantities from Laplacian...")

            # Apply Laplacian to vertex positions: Δr = L @ r
            original_mean_curvature_vector = laplacian_matrix @ vertices  # Shape: (N, 3)

            # Compute magnitudes (original predicted mean curvature values)
            original_predicted_mean_curvature = np.linalg.norm(original_mean_curvature_vector, axis=1)  # Shape: (N,)

            # Compute original predicted normals (normalized mean curvature vectors)
            original_predicted_normals = np.zeros_like(original_mean_curvature_vector)
            non_zero_mask = original_predicted_mean_curvature > 1e-10

            original_predicted_normals[non_zero_mask] = (
                    original_mean_curvature_vector[non_zero_mask] /
                    original_predicted_mean_curvature[non_zero_mask, np.newaxis]
            )

            # For zero curvature points, use a default normal (e.g., z-up)
            original_predicted_normals[~non_zero_mask] = np.array([0, 0, 1])

            print(f"      Original mean curvature range: [{original_predicted_mean_curvature.min():.6f}, {original_predicted_mean_curvature.max():.6f}]")
            print(f"      Original mean curvature mean: {original_predicted_mean_curvature.mean():.6f}")
            print(f"      Original zero curvature points: {(~non_zero_mask).sum()}/{len(original_predicted_mean_curvature)}")

            return {
                'original_mean_curvature_vector': original_mean_curvature_vector,
                'original_predicted_normals': original_predicted_normals,
                'original_predicted_mean_curvature': original_predicted_mean_curvature
            }

        except Exception as e:
            print(f"  ❌ Error computing original quantities from Laplacian: {e}")
            return None

    def _add_comprehensive_curvature_visualizations(self, mesh_structure,
                                                    gt_data: Dict[str, np.ndarray],
                                                    original_data: Dict[str, np.ndarray],
                                                    corrected_data: Dict[str, np.ndarray]):
        """
        Add comprehensive curvature and normal visualizations to the mesh with controlled ordering.
        Shows GT, Original Predicted, and Corrected Predicted quantities side-by-side.

        Args:
            mesh_structure: Polyscope mesh structure
            gt_data: Dictionary containing GT quantities
            original_data: Dictionary containing original predicted quantities
            corrected_data: Dictionary containing corrected predicted quantities
        """
        print(f"  🎨 Adding comprehensive curvature visualizations...")

        # Use alphabetical prefixes to control ordering in polyscope UI

        # === MEAN CURVATURE (grouped together) ===
        # GT mean curvature (if available)
        if gt_data.get('gt_mean_curvature') is not None:
            mesh_structure.add_scalar_quantity(
                name="A Mean Curvature - GT",
                values=gt_data['gt_mean_curvature'],
                enabled=False,
                cmap='plasma'
            )
            print(f"      Added GT mean curvature scalar field")

        # Original predicted mean curvature
        if original_data.get('original_predicted_mean_curvature') is not None:
            mesh_structure.add_scalar_quantity(
                name="B Mean Curvature - PRED Original",
                values=original_data['original_predicted_mean_curvature'],
                enabled=False,
                cmap='plasma'
            )
            print(f"      Added original predicted mean curvature scalar field")

        # Corrected predicted mean curvature
        if corrected_data.get('corrected_predicted_mean_curvature') is not None:
            mesh_structure.add_scalar_quantity(
                name="B2 Mean Curvature - PRED Corrected",
                values=corrected_data['corrected_predicted_mean_curvature'],
                enabled=False,
                cmap='plasma'
            )
            print(f"      Added corrected predicted mean curvature scalar field")

        # === NORMALS (grouped together) ===
        # GT normals (if available)
        if gt_data.get('gt_vertex_normals') is not None:
            mesh_structure.add_vector_quantity(
                name="C Normals - GT",
                values=gt_data['gt_vertex_normals'] * 0.05,  # Scale for visibility
                enabled=False,
                color=(0.0, 1.0, 1.0),  # Cyan color for GT normals
                vectortype="ambient"
            )
            print(f"      Added GT normals vector field")

        # Original predicted normals
        if original_data.get('original_predicted_normals') is not None:
            mesh_structure.add_vector_quantity(
                name="D Normals - PRED Original",
                values=original_data['original_predicted_normals'] * 0.05,  # Scale for visibility
                enabled=False,
                color=(1.0, 0.5, 0.0),  # Orange color for original predicted normals
                vectortype="ambient"
            )
            print(f"      Added original predicted normals vector field")

        # Corrected predicted normals
        if corrected_data.get('corrected_predicted_normals') is not None:
            mesh_structure.add_vector_quantity(
                name="D2 Normals - PRED Corrected",
                values=corrected_data['corrected_predicted_normals'] * 0.05,  # Scale for visibility
                enabled=False,
                color=(0.5, 1.0, 0.0),  # Light green color for corrected predicted normals
                vectortype="ambient"
            )
            print(f"      Added corrected predicted normals vector field")

        # === MEAN CURVATURE VECTORS ===
        # Original mean curvature vector field
        if original_data.get('original_mean_curvature_vector') is not None:
            mesh_structure.add_vector_quantity(
                name="E Mean Curvature Vector - PRED Original",
                values=original_data['original_mean_curvature_vector'] * 0.1,  # Scale for visibility
                enabled=False,
                color=(1.0, 0.0, 0.5),  # Pink color for original curvature vectors
                vectortype="ambient"
            )
            print(f"      Added original mean curvature vector field")

        # Corrected mean curvature vector field
        if corrected_data.get('corrected_mean_curvature_vector') is not None:
            mesh_structure.add_vector_quantity(
                name="E2 Mean Curvature Vector - PRED Corrected",
                values=corrected_data['corrected_mean_curvature_vector'] * 0.1,  # Scale for visibility
                enabled=False,
                color=(0.5, 0.0, 1.0),  # Purple color for corrected curvature vectors
                vectortype="ambient"
            )
            print(f"      Added corrected mean curvature vector field")

        # === COMPARISON METRICS ===
        gt_normals = gt_data.get('gt_vertex_normals')
        original_normals = original_data.get('original_predicted_normals')
        corrected_normals = corrected_data.get('corrected_predicted_normals')

        # Original vs GT comparison
        if gt_normals is not None and original_normals is not None:
            # Compute normal alignment (dot product)
            original_normal_alignment = np.sum(original_normals * gt_normals, axis=1)

            mesh_structure.add_scalar_quantity(
                name="F Normal Alignment (Original vs GT)",
                values=original_normal_alignment,
                enabled=False,
                cmap='coolwarm'
            )

            # Compute angular differences
            original_normal_alignment_clamped = np.clip(original_normal_alignment, -1.0, 1.0)
            original_angular_differences = np.arccos(np.abs(original_normal_alignment_clamped)) * 180 / np.pi

            mesh_structure.add_scalar_quantity(
                name="G Normal Angular Error Original (degrees)",
                values=original_angular_differences,
                enabled=False,
                cmap='coolwarm'
            )

            print(f"      Original normal alignment mean: {original_normal_alignment.mean():.4f}")
            print(f"      Original angular error mean: {original_angular_differences.mean():.2f}°")

        # Corrected vs GT comparison
        if gt_normals is not None and corrected_normals is not None:
            # Compute normal alignment (dot product)
            corrected_normal_alignment = np.sum(corrected_normals * gt_normals, axis=1)

            mesh_structure.add_scalar_quantity(
                name="F2 Normal Alignment (Corrected vs GT)",
                values=corrected_normal_alignment,
                enabled=False,
                cmap='coolwarm'
            )

            # Compute angular differences
            corrected_normal_alignment_clamped = np.clip(corrected_normal_alignment, -1.0, 1.0)
            corrected_angular_differences = np.arccos(np.abs(corrected_normal_alignment_clamped)) * 180 / np.pi

            mesh_structure.add_scalar_quantity(
                name="G2 Normal Angular Error Corrected (degrees)",
                values=corrected_angular_differences,
                enabled=False,
                cmap='coolwarm'
            )

            print(f"      Corrected normal alignment mean: {corrected_normal_alignment.mean():.4f}")
            print(f"      Corrected angular error mean: {corrected_angular_differences.mean():.2f}°")

            # Print comparison statistics
            if original_normals is not None:
                original_mean_error = original_angular_differences.mean()
                corrected_mean_error = corrected_angular_differences.mean()
                improvement = original_mean_error - corrected_mean_error
                print(f"      Normal orientation improvement: {improvement:.2f}° reduction in mean angular error")

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
            Dictionary containing mesh vertices, faces, and metadata
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
            'mesh_file_path': mesh_data.get('mesh_file_path', 'Unknown'),
            'num_vertices': mesh_data.get('num_vertices', len(mesh_data['vertices'])),
            'num_faces': mesh_data.get('num_faces', len(mesh_data['faces']))
        }

    def print_eigenvalue_analysis(self, gt_eigenvalues: Optional[np.ndarray],
                                  original_pred_eigenvalues: Optional[np.ndarray],
                                  corrected_pred_eigenvalues: Optional[np.ndarray],
                                  batch_idx: int):
        """Print detailed eigenvalue comparison analysis for GT, Original, and Corrected predictions."""
        print(f"\n" + "-" * 70)
        print(f"COMPREHENSIVE EIGENVALUE COMPARISON ANALYSIS - Batch {batch_idx}")
        print("-" * 70)

        # Ground-truth analysis
        if gt_eigenvalues is not None:
            print("GROUND-TRUTH EIGENVALUES:")
            print(f"  Number of eigenvalues: {len(gt_eigenvalues)}")
            print(f"  First eigenvalue (should be ~0): {gt_eigenvalues[0]:.2e}")
            if len(gt_eigenvalues) > 1:
                print(f"  Second eigenvalue (Fiedler): {gt_eigenvalues[1]:.6f}")
                print(f"  Spectral gap: {gt_eigenvalues[1] - gt_eigenvalues[0]:.6f}")
            print(f"  Largest eigenvalue: {gt_eigenvalues[-1]:.6f}")
            print(f"  Mean eigenvalue: {gt_eigenvalues.mean():.6f}")
        else:
            print("GROUND-TRUTH EIGENVALUES: Not available")

        # Original predicted analysis
        if original_pred_eigenvalues is not None:
            print("\nORIGINAL PREDICTED EIGENVALUES:")
            print(f"  Number of eigenvalues: {len(original_pred_eigenvalues)}")
            print(f"  First eigenvalue (should be ~0): {original_pred_eigenvalues[0]:.2e}")
            if len(original_pred_eigenvalues) > 1:
                print(f"  Second eigenvalue (Fiedler): {original_pred_eigenvalues[1]:.6f}")
                print(f"  Spectral gap: {original_pred_eigenvalues[1] - original_pred_eigenvalues[0]:.6f}")
            print(f"  Largest eigenvalue: {original_pred_eigenvalues[-1]:.6f}")
            print(f"  Mean eigenvalue: {original_pred_eigenvalues.mean():.6f}")
        else:
            print("\nORIGINAL PREDICTED EIGENVALUES: Not available")

        # Corrected predicted analysis
        if corrected_pred_eigenvalues is not None:
            print("\nCORRECTED PREDICTED EIGENVALUES:")
            print(f"  Number of eigenvalues: {len(corrected_pred_eigenvalues)}")
            print(f"  First eigenvalue (should be ~0): {corrected_pred_eigenvalues[0]:.2e}")
            if len(corrected_pred_eigenvalues) > 1:
                print(f"  Second eigenvalue (Fiedler): {corrected_pred_eigenvalues[1]:.6f}")
                print(f"  Spectral gap: {corrected_pred_eigenvalues[1] - corrected_pred_eigenvalues[0]:.6f}")
            print(f"  Largest eigenvalue: {corrected_pred_eigenvalues[-1]:.6f}")
            print(f"  Mean eigenvalue: {corrected_pred_eigenvalues.mean():.6f}")
        else:
            print("\nCORRECTED PREDICTED EIGENVALUES: Not available")

        # Comparison metrics
        if gt_eigenvalues is not None:
            eigenvalue_sets = [
                ("Original vs GT", original_pred_eigenvalues),
                ("Corrected vs GT", corrected_pred_eigenvalues)
            ]

            for comparison_name, pred_eigenvals in eigenvalue_sets:
                if pred_eigenvals is not None:
                    min_len = min(len(gt_eigenvalues), len(pred_eigenvals))
                    if min_len > 0:
                        gt_subset = gt_eigenvalues[:min_len]
                        pred_subset = pred_eigenvals[:min_len]

                        # Compute errors
                        abs_errors = np.abs(pred_subset - gt_subset)
                        rel_errors = abs_errors / (np.abs(gt_subset) + 1e-10)

                        print(f"\n{comparison_name.upper()} COMPARISON METRICS (first {min_len} eigenvalues):")
                        print(f"  Mean absolute error: {abs_errors.mean():.6f}")
                        print(f"  Max absolute error: {abs_errors.max():.6f}")
                        print(f"  Mean relative error: {rel_errors.mean():.6f}")
                        print(f"  Max relative error: {rel_errors.max():.6f}")
                        print(f"  Correlation coefficient: {np.corrcoef(gt_subset, pred_subset)[0, 1]:.6f}")

        # Positive semi-definiteness check
        print("\nPOSITIVE SEMI-DEFINITENESS CHECK:")
        for name, eigenvals in [("GT", gt_eigenvalues), ("Original PRED", original_pred_eigenvalues), ("Corrected PRED", corrected_pred_eigenvalues)]:
            if eigenvals is not None:
                if eigenvals[0] < -1e-10:
                    print(f"  ⚠️  {name}: First eigenvalue is significantly negative ({eigenvals[0]:.2e})")
                else:
                    print(f"  ✅ {name}: Laplacian appears to be positive semi-definite")

        print("-" * 70)

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

    def print_eigenvector_correlation_analysis(self, correlation_matrix: np.ndarray, comparison_name: str, batch_idx: int):
        """Print eigenvector correlation analysis."""
        print(f"\n" + "-" * 50)
        print(f"EIGENVECTOR CORRELATION ANALYSIS ({comparison_name}) - Batch {batch_idx}")
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

    def visualize_comprehensive_eigenvectors(self, mesh_structure,
                                             gt_eigenvalues: Optional[np.ndarray], gt_eigenvectors: Optional[np.ndarray],
                                             original_pred_eigenvalues: Optional[np.ndarray], original_pred_eigenvectors: Optional[np.ndarray],
                                             corrected_pred_eigenvalues: Optional[np.ndarray], corrected_pred_eigenvectors: Optional[np.ndarray]):
        """Add GT, original predicted, and corrected predicted eigenvector scalar fields to the mesh."""
        num_to_show = self.config.num_eigenvectors_to_show

        # Determine how many eigenvectors we can show for each type
        gt_available = gt_eigenvectors.shape[1] if gt_eigenvectors is not None else 0
        original_available = original_pred_eigenvectors.shape[1] if original_pred_eigenvectors is not None else 0
        corrected_available = corrected_pred_eigenvectors.shape[1] if corrected_pred_eigenvectors is not None else 0

        max_available = max(gt_available, original_available, corrected_available)
        num_to_show = min(num_to_show, max_available)

        print(f"Adding comprehensive eigenvector visualization:")
        print(f"  GT eigenvectors available: {gt_available}")
        print(f"  Original predicted eigenvectors available: {original_available}")
        print(f"  Corrected predicted eigenvectors available: {corrected_available}")
        print(f"  Showing: {num_to_show} eigenvectors per type")

        # Add eigenvectors in groups using numerical prefixes to control alphabetical ordering
        for i in range(num_to_show):
            # Add GT eigenvector
            if gt_eigenvectors is not None and i < gt_available:
                gt_eigenvector = gt_eigenvectors[:, i]
                gt_eigenvalue = gt_eigenvalues[i] if gt_eigenvalues is not None else 0.0

                if i == 0:
                    gt_name = f"Eigenvector {i:02d}a GT (λ={gt_eigenvalue:.2e}, constant)"
                elif i == 1:
                    gt_name = f"Eigenvector {i:02d}a GT (λ={gt_eigenvalue:.6f}, Fiedler)"
                else:
                    gt_name = f"Eigenvector {i:02d}a GT (λ={gt_eigenvalue:.6f})"

                mesh_structure.add_scalar_quantity(
                    name=gt_name,
                    values=gt_eigenvector,
                    enabled=(i == 1),  # Enable GT Fiedler vector by default
                    cmap=self.config.colormap
                )
                print(f"  {gt_name}: range=[{gt_eigenvector.min():.4f}, {gt_eigenvector.max():.4f}]")

            # Add original predicted eigenvector
            if original_pred_eigenvectors is not None and i < original_available:
                original_eigenvector = original_pred_eigenvectors[:, i]
                original_eigenvalue = original_pred_eigenvalues[i] if original_pred_eigenvalues is not None else 0.0

                if i == 0:
                    original_name = f"Eigenvector {i:02d}b PRED Original (λ={original_eigenvalue:.2e}, constant)"
                elif i == 1:
                    original_name = f"Eigenvector {i:02d}b PRED Original (λ={original_eigenvalue:.6f}, Fiedler)"
                else:
                    original_name = f"Eigenvector {i:02d}b PRED Original (λ={original_eigenvalue:.6f})"

                mesh_structure.add_scalar_quantity(
                    name=original_name,
                    values=original_eigenvector,
                    enabled=False,
                    cmap=self.config.colormap
                )
                print(f"  {original_name}: range=[{original_eigenvector.min():.4f}, {original_eigenvector.max():.4f}]")

            # Add corrected predicted eigenvector
            if corrected_pred_eigenvectors is not None and i < corrected_available:
                corrected_eigenvector = corrected_pred_eigenvectors[:, i]
                corrected_eigenvalue = corrected_pred_eigenvalues[i] if corrected_pred_eigenvalues is not None else 0.0

                if i == 0:
                    corrected_name = f"Eigenvector {i:02d}c PRED Corrected (λ={corrected_eigenvalue:.2e}, constant)"
                elif i == 1:
                    corrected_name = f"Eigenvector {i:02d}c PRED Corrected (λ={corrected_eigenvalue:.6f}, Fiedler)"
                else:
                    corrected_name = f"Eigenvector {i:02d}c PRED Corrected (λ={corrected_eigenvalue:.6f})"

                mesh_structure.add_scalar_quantity(
                    name=corrected_name,
                    values=corrected_eigenvector,
                    enabled=False,
                    cmap=self.config.colormap
                )
                print(f"  {corrected_name}: range=[{corrected_eigenvector.min():.4f}, {corrected_eigenvector.max():.4f}]")

    def visualize_batch(self, batch_idx: int = 0):
        """Visualize comprehensive eigenanalysis results for a specific batch with GT, original, and corrected comparisons."""
        if not self.validation_data:
            print("No validation data loaded!")
            return

        if batch_idx >= len(self.validation_data['validation_results']):
            print(f"Batch index {batch_idx} out of range (max: {len(self.validation_data['validation_results']) - 1})")
            return

        self.current_batch_idx = batch_idx
        batch_result = self.validation_data['validation_results'][batch_idx]

        print(f"\nVisualizing comprehensive analysis for batch {batch_idx}...")

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
        mesh_file_path = mesh_data['mesh_file_path']

        print(f"Mesh file: {mesh_file_path}")
        print(f"Mesh has {mesh_data['num_vertices']} vertices and {mesh_data['num_faces']} faces")

        # Convert to numpy for processing
        vertices_np = vertices.cpu().numpy()
        faces_np = faces.cpu().numpy()

        # STEP 1: Compute GT data locally (instead of using pkl data)
        print(f"\n{'=' * 60}")
        print("STEP 1: Computing GT data locally")
        print('=' * 60)
        gt_data = self._compute_gt_data_locally(vertices_np, faces_np, mesh_file_path, k=self.config.num_eigenvectors_to_show)

        # STEP 2: Extract original predicted Laplacian matrix and compute original quantities
        print(f"\n{'=' * 60}")
        print("STEP 2: Processing original predicted Laplacian")
        print('=' * 60)

        original_laplacian_matrix = None
        original_data = {}
        original_eigenvalues = None
        original_eigenvectors = None

        if 'predicted_laplacian' in batch_result:
            laplacian_data = batch_result['predicted_laplacian']
            if laplacian_data is not None:
                print(f"Found predicted Laplacian: {laplacian_data['shape']} ({laplacian_data['nnz']} non-zeros)")

                # Reconstruct the matrix
                original_laplacian_matrix = self._reconstruct_laplacian_matrix(laplacian_data)

                if original_laplacian_matrix is not None:
                    # Compute original quantities from Laplacian
                    original_data = self._compute_original_quantities_from_laplacian(original_laplacian_matrix, vertices_np)

                    # Get original eigendecomposition from batch data
                    if 'eigendata' in batch_result and batch_result['eigendata']:
                        eigendata = batch_result['eigendata']
                        if 'predicted_eigenvalues' in eigendata and eigendata['predicted_eigenvalues'] is not None:
                            original_eigenvalues = eigendata['predicted_eigenvalues']
                            if isinstance(original_eigenvalues, torch.Tensor):
                                original_eigenvalues = original_eigenvalues.cpu().numpy()
                        if 'predicted_eigenvectors' in eigendata and eigendata['predicted_eigenvectors'] is not None:
                            original_eigenvectors = eigendata['predicted_eigenvectors']
                            if isinstance(original_eigenvectors, torch.Tensor):
                                original_eigenvectors = original_eigenvectors.cpu().numpy()
            else:
                print(f"No predicted Laplacian found in batch data")
        else:
            print(f"No predicted Laplacian data in batch result")

        # STEP 3: Apply normal orientation correction
        print(f"\n{'=' * 60}")
        print("STEP 3: Applying normal orientation correction")
        print('=' * 60)

        corrected_data = {}
        corrected_eigenvalues = None
        corrected_eigenvectors = None

        if original_data and 'original_predicted_normals' in original_data:
            # Apply Open3D normal orientation
            oriented_normals, flipped_mask = self._orient_normals_consistently_with_open3d(
                vertices_np, faces_np, original_data['original_predicted_normals']
            )

            # STEP 4: Create corrected Laplacian matrix
            print(f"\n{'=' * 60}")
            print("STEP 4: Creating corrected Laplacian matrix")
            print('=' * 60)

            if original_laplacian_matrix is not None:
                corrected_laplacian_matrix = self._create_corrected_laplacian(original_laplacian_matrix, flipped_mask)

                # STEP 5: Compute corrected quantities
                print(f"\n{'=' * 60}")
                print("STEP 5: Computing corrected quantities")
                print('=' * 60)

                # Compute corrected mean curvature quantities
                corrected_data = self._compute_corrected_quantities(corrected_laplacian_matrix, vertices_np)

                # Recompute eigendecomposition for corrected Laplacian
                corrected_eigenvalues, corrected_eigenvectors = self._recompute_eigendecomposition(
                    corrected_laplacian_matrix, k=self.config.num_eigenvectors_to_show
                )

        # STEP 6: Visualize comprehensive results
        print(f"\n{'=' * 60}")
        print("STEP 6: Creating comprehensive visualization")
        print('=' * 60)

        # Visualize base mesh
        mesh_structure = self.visualize_mesh(vertices, vertex_normals, faces)

        # Print comprehensive eigenvalue analysis
        if self.config.enable_eigenvalue_info:
            self.print_eigenvalue_analysis(
                gt_data.get('gt_eigenvalues'),
                original_eigenvalues,
                corrected_eigenvalues,
                batch_idx
            )

        # Print eigenvector correlation analysis
        if self.config.enable_correlation_analysis:
            gt_eigenvecs = gt_data.get('gt_eigenvectors')
            if gt_eigenvecs is not None:
                if original_eigenvectors is not None:
                    original_correlation_matrix = self.compute_eigenvector_correlations(gt_eigenvecs, original_eigenvectors)
                    self.print_eigenvector_correlation_analysis(original_correlation_matrix, "Original vs GT", batch_idx)

                if corrected_eigenvectors is not None:
                    corrected_correlation_matrix = self.compute_eigenvector_correlations(gt_eigenvecs, corrected_eigenvectors)
                    self.print_eigenvector_correlation_analysis(corrected_correlation_matrix, "Corrected vs GT", batch_idx)

        # Visualize comprehensive eigenvectors (GT, Original, Corrected)
        self.visualize_comprehensive_eigenvectors(
            mesh_structure,
            gt_data.get('gt_eigenvalues'), gt_data.get('gt_eigenvectors'),
            original_eigenvalues, original_eigenvectors,
            corrected_eigenvalues, corrected_eigenvectors
        )

        # Add comprehensive curvature and normal visualizations
        self._add_comprehensive_curvature_visualizations(
            mesh_structure,
            gt_data,
            original_data,
            corrected_data
        )

        # Print batch metrics
        metrics = batch_result['metrics']
        print(f"\nBatch {batch_idx} metrics:")
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value}")

        print(f"\n✅ Comprehensive visualization completed for batch {batch_idx}")
        print(f"Available quantities:")
        print(f"  - GT: {'✓' if gt_data.get('gt_eigenvalues') is not None else '✗'} eigendecomposition, {'✓' if gt_data.get('gt_mean_curvature') is not None else '✗'} mean curvature")
        print(f"  - Original predicted: {'✓' if original_eigenvalues is not None else '✗'} eigendecomposition, {'✓' if original_data else '✗'} curvature/normals")
        print(f"  - Corrected predicted: {'✓' if corrected_eigenvalues is not None else '✗'} eigendecomposition, {'✓' if corrected_data else '✗'} curvature/normals")

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
        print("COMPREHENSIVE AUTOMATIC ITERATION MODE")
        print("=" * 60)
        print(f"Will iterate through {num_batches} validation results")
        print("Close each window to proceed to the next result")
        print("Features:")
        print("  - Ground-truth data computed locally (not from pkl)")
        print("  - GT eigenvectors: Eigenvector XXa GT")
        print("  - Original predicted eigenvectors: Eigenvector XXb PRED Original")
        print("  - Corrected predicted eigenvectors: Eigenvector XXc PRED Corrected")
        print("  - GT, Original, and Corrected mean curvature side-by-side")
        print("  - GT, Original, and Corrected normals side-by-side")
        print("  - Normal orientation correction using Open3D")
        print("  - Laplacian matrix correction based on flipped normals")
        print("  - Comprehensive eigenvalue and correlation analysis")
        print("  - Each result shows its own locally computed GT data")
        print("=" * 60)

        # Iterate through all validation results
        for batch_idx in range(num_batches):
            print(f"\n🔍 Showing comprehensive analysis for result {batch_idx + 1}/{num_batches}")

            # Setup polyscope for this iteration
            self.setup_polyscope()

            # Visualize this batch
            self.visualize_batch(batch_idx)

            print(f"Window opened for result {batch_idx + 1}. Close window to continue to next result.")

            # Show visualization (blocks until window is closed)
            ps.show()

            # Clear structures for next iteration
            ps.remove_all_structures()

        print(f"\n✅ Completed comprehensive visualization of all {num_batches} validation results!")

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
    parser = argparse.ArgumentParser(description="Comprehensive Laplacian eigenanalysis visualization with GT computation and normal correction")
    parser.add_argument("folder_path", type=str, help="Path to folder containing validation data pickle files")
    parser.add_argument("--point_radius", type=float, default=0.005, help="Point radius for mesh visualization")
    parser.add_argument("--num_eigenvectors", type=int, default=40, help="Number of eigenvectors to show")
    parser.add_argument("--colormap", type=str, default='coolwarm', help="Colormap for eigenvectors")
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

    # Check dependencies
    print("Checking dependencies...")
    try:
        import open3d as o3d
        print("✓ Open3D available")
    except ImportError:
        raise ImportError("Open3D is required for normal orientation. Install with: pip install open3d")

    try:
        from pyFM.mesh import TriMesh
        print("✓ PyFM available")
    except ImportError:
        raise ImportError("PyFM is required for GT eigendecomposition. Install with: pip install pyFM")

    if HAS_IGL:
        print("✓ libigl available")
    else:
        print("⚠ libigl not available - GT mean curvature will be skipped")

    # Create visualizer and process folder
    visualizer = EigenanalysisVisualizer(config=vis_config)
    visualizer.process_folder(folder_path)


if __name__ == "__main__":
    main()