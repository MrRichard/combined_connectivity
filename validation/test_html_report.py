#!/usr/bin/env python3
"""
Test HTML report generation.

Validates the HTML report generator with synthetic data.
"""

import sys
from pathlib import Path
import tempfile
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "connectivity_shared" / "src"))

from connectivity_shared import (
    HTMLReportGenerator,
    generate_connectivity_report,
    QC_VIZ_AVAILABLE,
)


def test_basic_report_generation():
    """Test basic HTML report generation."""
    print("\n=== Testing Basic Report Generation ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        report = HTMLReportGenerator(output_dir=tmpdir)

        # Add metadata
        report.add_metadata(
            subject_id="sub-01",
            session_id="ses-01",
            modality="fMRI",
            atlas="Brainnetome",
            n_rois=246,
        )

        # Add some metrics
        report.add_metrics('global', {
            'n_nodes': 246,
            'n_edges': 5000,
            'connection_density': 0.165,
            'global_efficiency': 0.52,
            'clustering_coefficient': 0.45,
        })

        # Generate
        output_path = report.generate("test_report.html")

        assert output_path.exists(), "Report file not created"
        content = output_path.read_text()

        # Check content
        assert "sub-01" in content, "Subject ID not in report"
        assert "Brainnetome" in content, "Atlas not in report"
        assert "0.52" in content, "Efficiency metric not in report"

        print(f"  Created: {output_path}")
        print(f"  File size: {output_path.stat().st_size} bytes")

    print("  Basic Report Generation: PASSED")
    return True


def test_report_with_tables():
    """Test report with tables."""
    print("\n=== Testing Report with Tables ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        report = HTMLReportGenerator(output_dir=tmpdir)

        report.add_metadata(subject_id="sub-02", modality="DWI")

        # Add nodal metrics as table
        nodal_data = [
            {'roi': 'ROI_001', 'degree': 45, 'clustering': 0.52, 'centrality': 0.08},
            {'roi': 'ROI_002', 'degree': 38, 'clustering': 0.48, 'centrality': 0.05},
            {'roi': 'ROI_003', 'degree': 62, 'clustering': 0.55, 'centrality': 0.12},
        ]
        report.add_table(
            title="Nodal Metrics (Top 3)",
            data=nodal_data,
            columns=['roi', 'degree', 'clustering', 'centrality'],
            section="Nodal Analysis",
        )

        output_path = report.generate("table_report.html")

        content = output_path.read_text()
        assert "ROI_001" in content, "ROI not in table"
        assert "0.52" in content, "Clustering not in table"
        assert "Nodal Analysis" in content, "Section title not found"

        print(f"  Created: {output_path}")

    print("  Report with Tables: PASSED")
    return True


def test_report_with_warnings():
    """Test report with warnings and info messages."""
    print("\n=== Testing Report with Warnings ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        report = HTMLReportGenerator(output_dir=tmpdir)

        report.add_metadata(subject_id="sub-03", modality="fMRI")

        report.add_warning("High motion detected: mean FD = 0.8mm")
        report.add_info("Processing completed successfully")

        output_path = report.generate("warning_report.html")

        content = output_path.read_text()
        assert "High motion" in content, "Warning not in report"
        assert "warning-box" in content, "Warning styling not found"
        assert "Processing completed" in content, "Info not in report"

        print(f"  Created: {output_path}")

    print("  Report with Warnings: PASSED")
    return True


def test_report_with_images():
    """Test report with embedded images."""
    print("\n=== Testing Report with Images ===")

    if not QC_VIZ_AVAILABLE:
        print("  SKIPPED: matplotlib not available")
        return True

    from connectivity_shared import QCVisualizer
    from generate_test_data import generate_synthetic_connectivity_matrix

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate test images
        viz = QCVisualizer(output_dir=tmpdir)
        matrix = generate_synthetic_connectivity_matrix(n_rois=50, density=0.3)

        heatmap_path = viz.create_connectivity_heatmap(
            matrix,
            output_filename="heatmap.png"
        )

        histogram_path = viz.create_edge_histogram(
            matrix,
            output_filename="histogram.png"
        )

        # Create report with images
        report = HTMLReportGenerator(output_dir=tmpdir, embed_images=True)
        report.add_metadata(subject_id="sub-04", modality="fMRI")

        report.add_image(
            "heatmap",
            heatmap_path,
            caption="Connectivity Matrix Heatmap",
            section="Quality Control"
        )
        report.add_image(
            "histogram",
            histogram_path,
            caption="Edge Weight Distribution",
            section="Quality Control"
        )

        output_path = report.generate("image_report.html")

        content = output_path.read_text()
        assert "data:image/png;base64" in content, "Image not embedded"
        assert "Connectivity Matrix Heatmap" in content, "Caption not found"

        print(f"  Created: {output_path}")
        print(f"  File size: {output_path.stat().st_size} bytes")

    print("  Report with Images: PASSED")
    return True


def test_convenience_function():
    """Test the generate_connectivity_report convenience function."""
    print("\n=== Testing Convenience Function ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock matrix file
        matrix_path = tmpdir / "connectivity.csv"
        matrix_path.write_text("roi1,roi2\n0,0.5\n0.5,0")

        # Create mock metrics file
        metrics_path = tmpdir / "metrics.json"
        metrics = {
            "metadata": {"atlas_name": "test_atlas"},
            "global_metrics": {
                "n_nodes": 2,
                "n_edges": 1,
                "connection_density": 0.5,
            }
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

        # Generate report
        output_path = generate_connectivity_report(
            matrix_path=matrix_path,
            metrics_path=metrics_path,
            subject_id="sub-05",
            modality="fmri",
            atlas="test_atlas",
        )

        assert output_path.exists(), "Report not created"
        content = output_path.read_text()
        assert "sub-05" in content, "Subject ID not in report"
        assert "test_atlas" in content, "Atlas not in report"

        print(f"  Created: {output_path}")

    print("  Convenience Function: PASSED")
    return True


def test_custom_sections():
    """Test adding custom HTML sections."""
    print("\n=== Testing Custom Sections ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        report = HTMLReportGenerator(output_dir=tmpdir)

        report.add_metadata(subject_id="sub-06", modality="DWI")

        # Add custom section with HTML
        report.add_custom_section(
            title="Processing Notes",
            content="""
            <p>This subject was processed with the following parameters:</p>
            <ul>
                <li>Tractography: iFOD2 + ACT</li>
                <li>Streamlines: 10M generated, 1M SIFT-filtered</li>
            </ul>
            """,
            order=50,
        )

        output_path = report.generate("custom_report.html")

        content = output_path.read_text()
        assert "Processing Notes" in content, "Custom section title not found"
        assert "iFOD2" in content, "Custom content not found"

        print(f"  Created: {output_path}")

    print("  Custom Sections: PASSED")
    return True


def test_metric_display_names():
    """Test custom display names for metrics."""
    print("\n=== Testing Metric Display Names ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        report = HTMLReportGenerator(output_dir=tmpdir)

        report.add_metadata(subject_id="sub-07")

        report.add_metrics(
            'network',
            {
                'char_path_len': 2.5,
                'sw_sigma': 1.8,
            },
            display_names={
                'char_path_len': 'Characteristic Path Length',
                'sw_sigma': 'Small-World σ',
            }
        )

        output_path = report.generate("display_names_report.html")

        content = output_path.read_text()
        assert "Characteristic Path Length" in content, "Display name not used"
        assert "Small-World" in content, "Display name not used"

        print(f"  Created: {output_path}")

    print("  Metric Display Names: PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("HTML REPORT VALIDATION TESTS")
    print("=" * 60)

    tests = [
        ("Basic Report Generation", test_basic_report_generation),
        ("Report with Tables", test_report_with_tables),
        ("Report with Warnings", test_report_with_warnings),
        ("Report with Images", test_report_with_images),
        ("Convenience Function", test_convenience_function),
        ("Custom Sections", test_custom_sections),
        ("Metric Display Names", test_metric_display_names),
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
