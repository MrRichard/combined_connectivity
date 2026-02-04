"""
connectivity-shared: Shared utilities for neuroimaging connectivity analysis.

This package provides common functionality for both DTI/NODDI diffusion
and resting-state fMRI connectivity pipelines, including:

- FreeSurfer reconstruction detection and atlas handling
- Connectivity matrix I/O with standardized formats
- Graph theory metrics using NetworkX
- QC visualization utilities
- Atlas label definitions
"""

__version__ = "0.1.0"

from .freesurfer_detector import FreeSurferDetector, FreeSurferInfo, detect_freesurfer
from .matrix_io import (
    load_connectivity_matrix,
    save_connectivity_matrix,
    convert_legacy_csv,
    validate_matrix,
    get_matrix_stats,
)
from .graph_metrics import GraphMetrics, GraphMetricsResult
from .atlas_labels import (
    get_atlas_labels,
    get_atlas_info,
    save_labels_file,
    load_labels_file,
    verify_labels_consistency,
    generate_canonical_labels_file,
    BRAINNETOME_LABELS,
    FREESURFER_DK_LABELS,
)
from .html_report import (
    HTMLReportGenerator,
    generate_connectivity_report,
)
from .dicom_to_bids import DicomToBIDS, BIDSConversionResult
from .brain_extraction import BrainExtractor, BrainExtractionResult

# QC visualization (optional - requires matplotlib)
try:
    from .qc_visualization import QCVisualizer, create_registration_check
    QC_VIZ_AVAILABLE = True
except ImportError:
    QC_VIZ_AVAILABLE = False
    QCVisualizer = None
    create_registration_check = None

__all__ = [
    "__version__",
    # FreeSurfer
    "FreeSurferDetector",
    "FreeSurferInfo",
    "detect_freesurfer",
    # Matrix I/O
    "load_connectivity_matrix",
    "save_connectivity_matrix",
    "convert_legacy_csv",
    "validate_matrix",
    "get_matrix_stats",
    # Graph Metrics
    "GraphMetrics",
    "GraphMetricsResult",
    # Atlas Labels
    "get_atlas_labels",
    "get_atlas_info",
    "save_labels_file",
    "load_labels_file",
    "verify_labels_consistency",
    "generate_canonical_labels_file",
    "BRAINNETOME_LABELS",
    "FREESURFER_DK_LABELS",
    # HTML Reports
    "HTMLReportGenerator",
    "generate_connectivity_report",
    # QC Visualization
    "QCVisualizer",
    "create_registration_check",
    "QC_VIZ_AVAILABLE",
    # DICOM-to-BIDS
    "DicomToBIDS",
    "BIDSConversionResult",
    # Brain Extraction
    "BrainExtractor",
    "BrainExtractionResult",
]
