#!/usr/bin/env python3
"""
Test graph metrics implementation.

Validates that the NetworkX-based GraphMetrics class produces
sensible results on synthetic data.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "connectivity_shared" / "src"))

from connectivity_shared import GraphMetrics, GraphMetricsResult
from generate_test_data import generate_synthetic_connectivity_matrix, generate_labels


def test_basic_metrics():
    """Test basic connectivity metrics."""
    print("\n=== Testing Basic Metrics ===")

    # Create test matrix
    matrix = generate_synthetic_connectivity_matrix(n_rois=50, density=0.3, seed=42)
    labels = generate_labels(50)

    # Compute metrics
    gm = GraphMetrics(n_random_networks=20, seed=42)
    result = gm.compute_all(matrix, labels, modality="test", atlas="synthetic")

    # Check result type
    assert isinstance(result, GraphMetricsResult), "Result should be GraphMetricsResult"

    # Check basic metrics
    metrics = result.global_metrics
    print(f"  n_nodes: {metrics['n_nodes']}")
    assert metrics['n_nodes'] == 50, "Should have 50 nodes"

    print(f"  total_connections: {metrics['total_connections']}")
    assert metrics['total_connections'] > 0, "Should have connections"

    print(f"  connection_density: {metrics['connection_density']:.4f}")
    assert 0 < metrics['connection_density'] < 1, "Density should be between 0 and 1"

    print(f"  sparsity: {metrics['sparsity']:.4f}")
    assert np.isclose(metrics['sparsity'], 1 - metrics['connection_density'], rtol=1e-5), "Sparsity should be ~(1 - density)"

    print("  Basic metrics: PASSED")
    return True


def test_clustering_metrics():
    """Test clustering coefficient calculations."""
    print("\n=== Testing Clustering Metrics ===")

    matrix = generate_synthetic_connectivity_matrix(n_rois=50, density=0.4, seed=123)

    gm = GraphMetrics(n_random_networks=10, seed=42)
    result = gm.compute_all(matrix)

    metrics = result.global_metrics

    print(f"  binary_clustering_coefficient: {metrics['binary_clustering_coefficient']:.4f}")
    print(f"  weighted_clustering_coefficient: {metrics['weighted_clustering_coefficient']:.4f}")

    # Clustering should be between 0 and 1
    assert 0 <= metrics['binary_clustering_coefficient'] <= 1, "Binary clustering out of range"
    assert 0 <= metrics['weighted_clustering_coefficient'] <= 1, "Weighted clustering out of range"

    print("  Clustering metrics: PASSED")
    return True


def test_efficiency_metrics():
    """Test efficiency calculations."""
    print("\n=== Testing Efficiency Metrics ===")

    # Use higher density to ensure connected graph
    matrix = generate_synthetic_connectivity_matrix(n_rois=30, density=0.5, seed=456)

    gm = GraphMetrics(n_random_networks=10, seed=42)
    result = gm.compute_all(matrix)

    metrics = result.global_metrics

    print(f"  global_efficiency: {metrics['global_efficiency']:.4f}")
    print(f"  local_efficiency: {metrics['local_efficiency']:.4f}")

    if metrics['characteristic_path_length'] is not None:
        print(f"  characteristic_path_length: {metrics['characteristic_path_length']:.4f}")
    else:
        print("  characteristic_path_length: None (disconnected graph)")

    # Efficiency should be between 0 and 1
    assert 0 <= metrics['global_efficiency'] <= 1, "Global efficiency out of range"
    assert 0 <= metrics['local_efficiency'] <= 1, "Local efficiency out of range"

    print("  Efficiency metrics: PASSED")
    return True


def test_small_world_metrics():
    """Test small-world calculations."""
    print("\n=== Testing Small-World Metrics ===")

    matrix = generate_synthetic_connectivity_matrix(n_rois=50, density=0.3, seed=789)

    gm = GraphMetrics(n_random_networks=20, seed=42)
    result = gm.compute_all(matrix)

    metrics = result.global_metrics

    print(f"  small_worldness: {metrics['small_worldness']:.4f}")
    print(f"  normalized_clustering (gamma): {metrics['normalized_clustering']:.4f}")
    print(f"  normalized_path_length (lambda): {metrics['normalized_path_length']:.4f}")

    # Small-worldness should be positive for most networks
    # (typically > 1 for small-world networks)
    assert metrics['small_worldness'] >= 0, "Small-worldness should be non-negative"

    print("  Small-world metrics: PASSED")
    return True


def test_topology_metrics():
    """Test topology metrics (assortativity, modularity, rich-club)."""
    print("\n=== Testing Topology Metrics ===")

    matrix = generate_synthetic_connectivity_matrix(n_rois=50, density=0.3, seed=101)

    gm = GraphMetrics(n_random_networks=10, seed=42)
    result = gm.compute_all(matrix)

    metrics = result.global_metrics

    print(f"  assortativity: {metrics['assortativity']:.4f}")
    print(f"  modularity: {metrics['modularity']:.4f}")
    print(f"  n_communities: {metrics['n_communities']}")
    print(f"  rich_club_coefficient: {metrics['rich_club_coefficient']:.4f}")

    # Assortativity ranges from -1 to 1
    assert -1 <= metrics['assortativity'] <= 1, "Assortativity out of range"

    # Modularity ranges from -0.5 to 1
    assert -0.5 <= metrics['modularity'] <= 1, "Modularity out of range"

    # Should find at least 1 community
    assert metrics['n_communities'] >= 1, "Should find at least 1 community"

    print("  Topology metrics: PASSED")
    return True


def test_nodal_metrics():
    """Test per-node metrics."""
    print("\n=== Testing Nodal Metrics ===")

    n_rois = 30
    matrix = generate_synthetic_connectivity_matrix(n_rois=n_rois, density=0.4, seed=202)
    labels = generate_labels(n_rois)

    gm = GraphMetrics()
    result = gm.compute_all(matrix, labels)

    nodal = result.nodal_metrics

    print(f"  Nodal metrics computed for {len(nodal['labels'])} nodes")
    print(f"  Metrics: {list(nodal.keys())}")

    # Check all nodal metrics have correct length
    for key, values in nodal.items():
        assert len(values) == n_rois, f"Nodal metric {key} has wrong length"

    # Check degree is non-negative integers
    assert all(d >= 0 for d in nodal['degree']), "Degree should be non-negative"

    # Check clustering is in [0, 1]
    assert all(0 <= c <= 1 for c in nodal['clustering']), "Clustering should be in [0,1]"

    # Convert to DataFrame
    df = gm.nodal_metrics_to_dataframe(nodal)
    print(f"  DataFrame shape: {df.shape}")

    print("  Nodal metrics: PASSED")
    return True


def test_json_serialization():
    """Test JSON export."""
    print("\n=== Testing JSON Serialization ===")

    import tempfile
    import json

    matrix = generate_synthetic_connectivity_matrix(n_rois=20, density=0.3, seed=303)

    gm = GraphMetrics(n_random_networks=5, seed=42)
    result = gm.compute_all(
        matrix,
        modality="dwi",
        atlas="brainnetome",
        subject_id="sub-01",
        session_id="ses-01",
    )

    # Save to JSON
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = Path(f.name)

    result.to_json(json_path)

    # Load and validate
    with open(json_path, 'r') as f:
        loaded = json.load(f)

    print(f"  JSON keys: {list(loaded.keys())}")
    print(f"  Metadata: {loaded['metadata']}")
    print(f"  Global metrics count: {len(loaded['global_metrics'])}")

    assert 'metadata' in loaded, "Missing metadata"
    assert 'global_metrics' in loaded, "Missing global_metrics"
    assert loaded['metadata']['modality'] == 'dwi', "Wrong modality"

    # Cleanup
    json_path.unlink()

    print("  JSON serialization: PASSED")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")

    gm = GraphMetrics(n_random_networks=5, seed=42)

    # Empty matrix
    print("  Testing empty matrix...")
    empty = np.zeros((10, 10))
    result = gm.compute_all(empty)
    assert result.global_metrics['total_connections'] == 0, "Empty matrix should have 0 connections"
    print("    Empty matrix: OK")

    # Very sparse matrix
    print("  Testing sparse matrix...")
    sparse = generate_synthetic_connectivity_matrix(n_rois=50, density=0.05, seed=404)
    result = gm.compute_all(sparse)
    assert result.global_metrics['n_nodes'] == 50, "Sparse matrix should preserve node count"
    print("    Sparse matrix: OK")

    # Small matrix
    print("  Testing small matrix...")
    small = generate_synthetic_connectivity_matrix(n_rois=5, density=0.5, seed=505)
    result = gm.compute_all(small)
    assert result.global_metrics['n_nodes'] == 5, "Small matrix should work"
    print("    Small matrix: OK")

    print("  Edge cases: PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("GRAPH METRICS VALIDATION TESTS")
    print("=" * 60)

    tests = [
        ("Basic Metrics", test_basic_metrics),
        ("Clustering Metrics", test_clustering_metrics),
        ("Efficiency Metrics", test_efficiency_metrics),
        ("Small-World Metrics", test_small_world_metrics),
        ("Topology Metrics", test_topology_metrics),
        ("Nodal Metrics", test_nodal_metrics),
        ("JSON Serialization", test_json_serialization),
        ("Edge Cases", test_edge_cases),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"  ERROR: {e}")
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
