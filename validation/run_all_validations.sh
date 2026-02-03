#!/bin/bash
# Run all validation tests for connectivity-shared package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")/connectivity_shared"

echo "=============================================="
echo "CONNECTIVITY-SHARED VALIDATION SUITE"
echo "=============================================="
echo ""
echo "Package directory: $PACKAGE_DIR"
echo "Validation directory: $SCRIPT_DIR"
echo ""

# Check if package is installed
echo "Checking package installation..."
if python -c "import connectivity_shared" 2>/dev/null; then
    echo "  Package is installed"
else
    echo "  Package not installed. Installing in development mode..."
    pip install -e "$PACKAGE_DIR"
fi

echo ""
echo "----------------------------------------------"
echo "Test 1: Matrix I/O"
echo "----------------------------------------------"
python "$SCRIPT_DIR/test_matrix_io.py"

echo ""
echo "----------------------------------------------"
echo "Test 2: Graph Metrics"
echo "----------------------------------------------"
python "$SCRIPT_DIR/test_graph_metrics.py"

echo ""
echo "----------------------------------------------"
echo "Test 3: QC Visualization"
echo "----------------------------------------------"
python "$SCRIPT_DIR/test_qc_visualization.py"

echo ""
echo "----------------------------------------------"
echo "Test 4: Atlas Labels"
echo "----------------------------------------------"
python "$SCRIPT_DIR/test_atlas_labels.py"

echo ""
echo "----------------------------------------------"
echo "Test 5: Atlas Consistency"
echo "----------------------------------------------"
python "$SCRIPT_DIR/test_atlas_consistency.py" --test-only

echo ""
echo "----------------------------------------------"
echo "Test 6: HTML Reports"
echo "----------------------------------------------"
python "$SCRIPT_DIR/test_html_report.py"

echo ""
echo "=============================================="
echo "ALL VALIDATIONS COMPLETE"
echo "=============================================="
