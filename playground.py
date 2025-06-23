import numpy as np
import torch
from scipy.spatial.transform import Rotation
from neural_local_laplacian.datasets.synthetic_datasets import (
    PolynomialSurfaceDataset,
    RandomGridSampler,
    PoseType,
    CoeffGenerationMethod,
    DifferentialGeometryComponent
)
from neural_local_laplacian.utils.features import RISPFeatureExtractor
from neural_local_laplacian.utils.utils import compute_risp_features


def create_dataset():
    """Create PolynomialSurfaceDataset with the specified config."""

    # Create grid sampler
    grid_sampler = RandomGridSampler(num_points_range=30)

    # Create feature extractor
    feature_extractor = RISPFeatureExtractor(
        k=20,
        num_fourier_features=256,
        use_fourier=False
    )

    # Create dataset
    dataset = PolynomialSurfaceDataset(
        epoch_size=1000,
        grid_radius_range=[0.1, 1.0],
        grid_offset_range=[-0.01, 0.01],
        points_scale_range=[1, 1.1],
        coefficient_scale_range=[0.8, 0.95],
        order_range=[2, 7],
        grid_samplers=[grid_sampler],
        feature_extractor=feature_extractor,
        diff_geom_at_origin_only=True,
        conv_k_nearest=None,
        pose_type=PoseType.RANDOM_ROTATION,
        coeff_generation_method=CoeffGenerationMethod.UNIFORM,
        diff_geom_components=[DifferentialGeometryComponent.MEAN_CURVATURE],
        seed=42
    )

    return dataset


def extract_surface_data(surface_data):
    """Extract positions and normals from surface data object."""
    # surface_data is a list containing a single surface
    surface = surface_data[0]

    # Extract positions and normal
    positions = surface.pos.detach().cpu().numpy()  # Shape: (N, 3)
    normal = surface.normal.detach().cpu().numpy()  # Shape: (1, 3) due to diff_geom_at_origin_only=True

    return positions, normal


def apply_rotation_to_surface_data(positions, normal, rotation_matrix):
    """Apply rotation to positions and normal."""
    positions_rotated = positions @ rotation_matrix.T
    normal_rotated = normal @ rotation_matrix.T
    return positions_rotated, normal_rotated


def risp_rotation_invariance_with_dataset():
    """Test RISP rotation invariance using PolynomialSurfaceDataset."""
    print("Testing RISP rotation invariance with PolynomialSurfaceDataset...")

    # Create dataset
    dataset = create_dataset()

    # Generate a surface
    surface_data = dataset.get(0)  # Get first sample

    # Extract positions and normal
    positions, normal = extract_surface_data(surface_data)

    print(f"Surface positions shape: {positions.shape}")
    print(f"Surface normal shape: {normal.shape}")
    print(f"Normal vector: {normal[0]}")

    # Compute original RISP features using utils.py implementation
    risp_original = compute_risp_features(points=positions, normals=normal)
    print(f"Original RISP shape: {risp_original.shape}")

    # Apply max pooling as requested
    risp_original_maxpool = np.max(risp_original, axis=0)  # Shape: (14,)
    print(f"Original RISP max-pooled shape: {risp_original_maxpool.shape}")

    # Test multiple random rotations
    num_tests = 5
    max_error = 0.0

    print(f"\nTesting {num_tests} random rotations...")

    for i in range(num_tests):
        # Generate random rotation
        rotation = Rotation.random(random_state=i + 42).as_matrix()

        # Apply rotation to surface
        positions_rotated, normal_rotated = apply_rotation_to_surface_data(
            positions, normal, rotation
        )

        # Compute RISP features for rotated surface
        risp_rotated = compute_risp_features(points=positions_rotated, normals=normal_rotated)

        # Apply max pooling
        risp_rotated_maxpool = np.max(risp_rotated, axis=0)  # Shape: (14,)

        # Compute maximum absolute difference
        max_diff = np.max(np.abs(risp_original_maxpool - risp_rotated_maxpool))
        max_error = max(max_error, max_diff)

        print(f"Test {i + 1}: Max difference = {max_diff:.2e}")

        # Debug: show first few features
        if i == 0:
            print(f"  Original max-pooled RISP[:5]: {risp_original_maxpool[:5]}")
            print(f"  Rotated max-pooled RISP[:5]:  {risp_rotated_maxpool[:5]}")

    # Check if features are rotation invariant (within numerical precision)
    tolerance = 1e-10
    is_invariant = max_error < tolerance

    print(f"\nRotation Invariance Test Results:")
    print(f"Maximum error across all tests: {max_error:.2e}")
    print(f"Tolerance: {tolerance}")
    print(f"Result: {'PASS' if is_invariant else 'FAIL'}")

    if not is_invariant:
        print("⚠️  RISP features are NOT rotation invariant!")
        print("This indicates issues with the implementation.")

        # Try a more relaxed tolerance
        relaxed_tolerance = 1e-6
        if max_error < relaxed_tolerance:
            print(f"✓ However, features are rotation invariant within relaxed tolerance ({relaxed_tolerance})")
            print("  This might be acceptable for practical purposes.")
    else:
        print("✅ RISP features are rotation invariant within numerical precision!")

    return is_invariant, max_error


def risp_with_multiple_surfaces():
    """Test RISP on multiple different surfaces from the dataset."""
    print("\n" + "=" * 60)
    print("Testing RISP rotation invariance on multiple surfaces...")

    dataset = create_dataset()

    all_errors = []
    num_surfaces = 3

    for surface_idx in range(num_surfaces):
        print(f"\nSurface {surface_idx + 1}/{num_surfaces}:")

        # Generate different surfaces by resetting RNG
        dataset.reset_rng()
        for _ in range(surface_idx):  # Skip to get different surface
            dataset._rng = np.random.default_rng(seed=42 + surface_idx)

        surface_data = dataset.get(surface_idx)
        positions, normal = extract_surface_data(surface_data)

        # Test rotation invariance
        risp_original = compute_risp_features(points=positions, normals=normal)
        risp_original_maxpool = np.max(risp_original, axis=0)

        # Test with one rotation
        rotation = Rotation.random(random_state=surface_idx + 100).as_matrix()
        positions_rotated, normal_rotated = apply_rotation_to_surface_data(
            positions, normal, rotation
        )

        risp_rotated = compute_risp_features(points=positions_rotated, normals=normal_rotated)
        risp_rotated_maxpool = np.max(risp_rotated, axis=0)

        error = np.max(np.abs(risp_original_maxpool - risp_rotated_maxpool))
        all_errors.append(error)

        print(f"  Surface shape: {positions.shape}")
        print(f"  Normal: {normal[0]}")
        print(f"  Rotation error: {error:.2e}")

    avg_error = np.mean(all_errors)
    max_error = np.max(all_errors)

    print(f"\nSummary across {num_surfaces} surfaces:")
    print(f"Average error: {avg_error:.2e}")
    print(f"Maximum error: {max_error:.2e}")

    return max_error < 1e-10


def main():
    """Main function to run all RISP rotation invariance tests."""
    print("Starting RISP Rotation Invariance Testing")
    print("=" * 60)

    try:
        # Test single surface
        print("PHASE 1: Testing single surface with multiple rotations...")
        is_invariant, max_error = risp_rotation_invariance_with_dataset()

        # Test multiple surfaces
        print("\nPHASE 2: Testing multiple different surfaces...")
        multi_surface_pass = risp_with_multiple_surfaces()

        # Final results
        print(f"\n" + "=" * 60)
        print("FINAL RESULTS:")
        print(f"Single surface test: {'PASS' if is_invariant else 'FAIL'} (max error: {max_error:.2e})")
        print(f"Multiple surface test: {'PASS' if multi_surface_pass else 'FAIL'}")

        if is_invariant and multi_surface_pass:
            print("🎉 All tests PASSED! RISP features are rotation invariant.")
            return True
        else:
            print("❌ Some tests FAILED. RISP implementation needs review.")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)