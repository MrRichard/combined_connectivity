#!/usr/bin/env python3
"""
Test matrix I/O functions.

Validates loading, saving, and conversion of connectivity matrices.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "connectivity_shared" / "src"))

from connectivity_shared import (
    load_connectivity_matrix,
    save_connectivity_matrix,
    convert_legacy_csv,
    validate_matrix,
    get_matrix_stats,
)
from generate_test_data import generate_synthetic_connectivity_matrix, generate_labels


def test_save_and_load_csv():
    """Test saving and loading CSV with headers."""
    print("\n=== Testing CSV Save/Load ===")

    # Create test data
    n_rois = 50
    matrix = generate_synthetic_connectivity_matrix(n_rois=n_rois, density=0.3, seed=42)
    labels = generate_labels(n_rois)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)

    save_connectivity_matrix(matrix, labels, csv_path, modality="test", atlas="synthetic")

    # Check files exist
    assert csv_path.exists(), "CSV file not created"
    labels_file = csv_path.with_suffix('.labels.txt')
    assert labels_file.exists(), "Labels file not created"

    # Load back
    loaded_matrix, loaded_labels = load_connectivity_matrix(csv_path)

    # Verify
    print(f"  Original shape: {matrix.shape}")
    print(f"  Loaded shape: {loaded_matrix.shape}")
    print(f"  Labels match: {labels == loaded_labels}")

    assert np.allclose(matrix, loaded_matrix), "Matrix values don't match"
    assert labels == loaded_labels, "Labels don't match"

    # Cleanup
    csv_path.unlink()
    labels_file.unlink()

    print("  CSV Save/Load: PASSED")
    return True


def test_load_legacy_csv():
    """Test loading headerless CSV (tck2connectome style)."""
    print("\n=== Testing Legacy CSV Load ===")

    # Create headerless CSV
    n_rois = 30
    matrix = generate_synthetic_connectivity_matrix(n_rois=n_rois, density=0.3, seed=123)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
    np.savetxt(csv_path, matrix, delimiter=',')

    # Load without labels
    loaded_matrix, loaded_labels = load_connectivity_matrix(csv_path)

    print(f"  Matrix shape: {loaded_matrix.shape}")
    print(f"  Generated labels: {loaded_labels[:3]}...")

    assert np.allclose(matrix, loaded_matrix), "Matrix values don't match"
    assert len(loaded_labels) == n_rois, "Wrong number of labels"
    assert loaded_labels[0] == "ROI_000", "Labels should be generic"

    # Cleanup
    csv_path.unlink()

    print("  Legacy CSV Load: PASSED")
    return True


def test_convert_legacy_csv():
    """Test converting headerless CSV to new format."""
    print("\n=== Testing Legacy CSV Conversion ===")

    # Create headerless CSV
    n_rois = 25
    matrix = generate_synthetic_connectivity_matrix(n_rois=n_rois, density=0.4, seed=456)
    labels = generate_labels(n_rois, prefix="BNA")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        legacy_path = Path(f.name)
    np.savetxt(legacy_path, matrix, delimiter=',')

    # Create labels file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
        labels_path = Path(f.name)
        for i, label in enumerate(labels):
            f.write(f"{i}\t{label}\n")

    # Convert
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = Path(f.name)

    convert_legacy_csv(
        legacy_path,
        labels_path,
        output_path,
        modality="dwi",
        atlas="brainnetome",
    )

    # Load converted file
    loaded_matrix, loaded_labels = load_connectivity_matrix(output_path)

    print(f"  Converted matrix shape: {loaded_matrix.shape}")
    print(f"  Converted labels: {loaded_labels[:3]}...")

    assert np.allclose(matrix, loaded_matrix), "Conversion changed matrix values"
    assert loaded_labels == labels, "Conversion changed labels"

    # Cleanup
    legacy_path.unlink()
    labels_path.unlink()
    output_path.unlink()
    output_path.with_suffix('.labels.txt').unlink()

    print("  Legacy CSV Conversion: PASSED")
    return True


def test_validate_matrix():
    """Test matrix validation."""
    print("\n=== Testing Matrix Validation ===")

    # Valid matrix
    valid = generate_synthetic_connectivity_matrix(n_rois=20, density=0.3, seed=789)
    is_valid, warnings = validate_matrix(valid)
    print(f"  Valid matrix: {is_valid}, warnings: {warnings}")
    assert is_valid, "Valid matrix should pass"
    assert len(warnings) == 0, "Valid matrix should have no warnings"

    # Non-square matrix
    non_square = np.random.rand(10, 15)
    is_valid, warnings = validate_matrix(non_square)
    print(f"  Non-square: {is_valid}, warnings: {warnings}")
    assert not is_valid, "Non-square should fail"

    # Asymmetric matrix
    asymmetric = np.random.rand(10, 10)
    is_valid, warnings = validate_matrix(asymmetric, check_symmetric=True)
    print(f"  Asymmetric: {is_valid}, warnings: {warnings}")
    assert len(warnings) > 0, "Asymmetric should have warning"

    # Non-zero diagonal
    nonzero_diag = generate_synthetic_connectivity_matrix(n_rois=10, seed=101)
    np.fill_diagonal(nonzero_diag, 1.0)
    is_valid, warnings = validate_matrix(nonzero_diag, check_diagonal=True)
    print(f"  Non-zero diagonal: {is_valid}, warnings: {warnings}")
    assert len(warnings) > 0, "Non-zero diagonal should have warning"

    # Matrix with NaN
    with_nan = generate_synthetic_connectivity_matrix(n_rois=10, seed=202)
    with_nan[5, 5] = np.nan
    is_valid, warnings = validate_matrix(with_nan)
    print(f"  With NaN: {is_valid}, warnings: {warnings}")
    assert not is_valid, "Matrix with NaN should fail"

    print("  Matrix Validation: PASSED")
    return True


def test_matrix_stats():
    """Test matrix statistics."""
    print("\n=== Testing Matrix Stats ===")

    matrix = generate_synthetic_connectivity_matrix(n_rois=50, density=0.3, seed=303)
    stats = get_matrix_stats(matrix)

    print(f"  Stats keys: {list(stats.keys())}")
    print(f"  n_nodes: {stats['n_nodes']}")
    print(f"  density: {stats['density']:.4f}")
    print(f"  mean: {stats['mean']:.4f}")

    assert stats['n_nodes'] == 50, "Wrong node count"
    assert 0 < stats['density'] < 1, "Density out of range"
    assert stats['n_nonzero_edges'] > 0, "Should have edges"

    print("  Matrix Stats: PASSED")
    return True


def test_npy_format():
    """Test NumPy .npy format support."""
    print("\n=== Testing NPY Format ===")

    n_rois = 40
    matrix = generate_synthetic_connectivity_matrix(n_rois=n_rois, density=0.3, seed=404)
    labels = generate_labels(n_rois)

    # Save as npy
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_path = Path(f.name)
    np.save(npy_path, matrix)

    # Save labels file
    labels_file = npy_path.with_suffix('.labels.txt')
    with open(labels_file, 'w') as f:
        for i, label in enumerate(labels):
            f.write(f"{i}\t{label}\n")

    # Load back
    loaded_matrix, loaded_labels = load_connectivity_matrix(npy_path)

    print(f"  Loaded shape: {loaded_matrix.shape}")
    print(f"  Labels: {loaded_labels[:3]}...")

    assert np.allclose(matrix, loaded_matrix), "Matrix values don't match"
    assert loaded_labels == labels, "Labels don't match"

    # Cleanup
    npy_path.unlink()
    labels_file.unlink()

    print("  NPY Format: PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("MATRIX I/O VALIDATION TESTS")
    print("=" * 60)

    tests = [
        ("CSV Save/Load", test_save_and_load_csv),
        ("Legacy CSV Load", test_load_legacy_csv),
        ("Legacy CSV Conversion", test_convert_legacy_csv),
        ("Matrix Validation", test_validate_matrix),
        ("Matrix Stats", test_matrix_stats),
        ("NPY Format", test_npy_format),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")

    n_passed = sum(1 for _, s in results if s == "PASSED")
    print(f"\n  Total: {n_passed}/{len(tests)} passed")

    return n_passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
