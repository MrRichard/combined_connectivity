#!/usr/bin/env python3
"""
Test QC visualization functions.

Validates that visualization functions work with synthetic data.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "connectivity_shared" / "src"))

from connectivity_shared import QC_VIZ_AVAILABLE
from generate_test_data import generate_synthetic_connectivity_matrix, generate_labels


def test_connectivity_heatmap():
    """Test connectivity matrix heatmap generation."""
    print("\n=== Testing Connectivity Heatmap ===")

    if not QC_VIZ_AVAILABLE:
        print("  SKIPPED: matplotlib not available")
        return True

    from connectivity_shared import QCVisualizer

    # Create test data
    matrix = generate_synthetic_connectivity_matrix(n_rois=50, density=0.3, seed=42)
    labels = generate_labels(50)

    with tempfile.TemporaryDirectory() as tmpdir:
        viz = QCVisualizer(output_dir=tmpdir)

        # Test heatmap creation
        output_path = viz.create_connectivity_heatmap(
            matrix,
            labels=labels,
            output_filename="test_heatmap.png",
            title="Test Connectivity Matrix"
        )

        assert output_path.exists(), "Heatmap file not created"
        print(f"  Created: {output_path}")
        print(f"  File size: {output_path.stat().st_size} bytes")

    print("  Connectivity Heatmap: PASSED")
    return True


def test_edge_histogram():
    """Test edge distribution histogram."""
    print("\n=== Testing Edge Histogram ===")

    if not QC_VIZ_AVAILABLE:
        print("  SKIPPED: matplotlib not available")
        return True

    from connectivity_shared import QCVisualizer

    matrix = generate_synthetic_connectivity_matrix(n_rois=50, density=0.3, seed=123)

    with tempfile.TemporaryDirectory() as tmpdir:
        viz = QCVisualizer(output_dir=tmpdir)

        output_path = viz.create_edge_histogram(
            matrix,
            output_filename="test_histogram.png",
            title="Test Edge Distribution"
        )

        assert output_path.exists(), "Histogram file not created"
        print(f"  Created: {output_path}")

    print("  Edge Histogram: PASSED")
    return True


def test_degree_distribution():
    """Test degree distribution plot."""
    print("\n=== Testing Degree Distribution ===")

    if not QC_VIZ_AVAILABLE:
        print("  SKIPPED: matplotlib not available")
        return True

    from connectivity_shared import QCVisualizer

    matrix = generate_synthetic_connectivity_matrix(n_rois=100, density=0.2, seed=456)
    labels = generate_labels(100)

    with tempfile.TemporaryDirectory() as tmpdir:
        viz = QCVisualizer(output_dir=tmpdir)

        output_path = viz.create_degree_distribution(
            matrix,
            labels=labels,
            output_filename="test_degree.png",
            title="Test Degree Distribution"
        )

        assert output_path.exists(), "Degree distribution file not created"
        print(f"  Created: {output_path}")

    print("  Degree Distribution: PASSED")
    return True


def test_qc_summary():
    """Test comprehensive QC summary figure."""
    print("\n=== Testing QC Summary ===")

    if not QC_VIZ_AVAILABLE:
        print("  SKIPPED: matplotlib not available")
        return True

    from connectivity_shared import QCVisualizer, GraphMetrics

    matrix = generate_synthetic_connectivity_matrix(n_rois=50, density=0.3, seed=789)
    labels = generate_labels(50)

    # Compute metrics
    gm = GraphMetrics(n_random_networks=10, seed=42)
    result = gm.compute_all(matrix, labels)

    with tempfile.TemporaryDirectory() as tmpdir:
        viz = QCVisualizer(output_dir=tmpdir)

        output_path = viz.create_qc_summary(
            matrix,
            labels=labels,
            metrics=result.global_metrics,
            output_filename="test_summary.png",
            title="Test QC Summary",
            modality="fmri"
        )

        assert output_path.exists(), "QC summary file not created"
        print(f"  Created: {output_path}")
        print(f"  File size: {output_path.stat().st_size} bytes")

    print("  QC Summary: PASSED")
    return True


def test_empty_matrix():
    """Test visualization with edge cases."""
    print("\n=== Testing Edge Cases ===")

    if not QC_VIZ_AVAILABLE:
        print("  SKIPPED: matplotlib not available")
        return True

    from connectivity_shared import QCVisualizer

    # Sparse matrix
    sparse_matrix = generate_synthetic_connectivity_matrix(n_rois=20, density=0.05, seed=101)

    with tempfile.TemporaryDirectory() as tmpdir:
        viz = QCVisualizer(output_dir=tmpdir)

        # Should handle sparse matrix
        output_path = viz.create_connectivity_heatmap(
            sparse_matrix,
            output_filename="sparse_heatmap.png"
        )
        assert output_path.exists(), "Sparse heatmap not created"
        print("  Sparse matrix: OK")

        # Very small matrix
        small_matrix = generate_synthetic_connectivity_matrix(n_rois=5, density=0.5, seed=202)
        output_path = viz.create_qc_summary(
            small_matrix,
            output_filename="small_summary.png"
        )
        assert output_path.exists(), "Small matrix summary not created"
        print("  Small matrix: OK")

    print("  Edge Cases: PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("QC VISUALIZATION VALIDATION TESTS")
    print("=" * 60)

    if not QC_VIZ_AVAILABLE:
        print("\nWARNING: QC visualization not available (matplotlib not installed)")
        print("Tests will be skipped.\n")

    tests = [
        ("Connectivity Heatmap", test_connectivity_heatmap),
        ("Edge Histogram", test_edge_histogram),
        ("Degree Distribution", test_degree_distribution),
        ("QC Summary", test_qc_summary),
        ("Edge Cases", test_empty_matrix),
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
