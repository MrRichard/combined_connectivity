# Pipeline Usage Guide

This guide covers running the connectivity pipelines, from installation through processing data and generating reports.

## Table of Contents

1. [Installation](#installation)
2. [Running Validation Tests](#running-validation-tests)
3. [Diffusion Pipeline (DWI)](#diffusion-pipeline-dwi)
4. [fMRI Pipeline](#fmri-pipeline)
5. [Using the Shared Package](#using-the-shared-package)
6. [Generating QC Reports](#generating-qc-reports)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- SLURM (for HPC job submission)
- Singularity (for containerized processing)

### Step 1: Clone the Repository

```bash
git clone <repository-url> combined_connectivity
cd combined_connectivity
```

### Step 2: Install the Shared Package

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install the shared package in development mode
pip install -e ./connectivity_shared

# Install optional visualization dependencies
pip install matplotlib nilearn
```

### Step 3: Verify Installation

```bash
python -c "import connectivity_shared; print(connectivity_shared.__version__)"
# Should print: 0.1.0
```

---

## Running Validation Tests

Before processing data, verify that all components work correctly:

```bash
# Run all validation tests
bash validation/run_all_validations.sh

# Or run individual test suites
python validation/test_matrix_io.py
python validation/test_graph_metrics.py
python validation/test_qc_visualization.py
python validation/test_atlas_labels.py
python validation/test_atlas_consistency.py
python validation/test_html_report.py
```

Expected output: **37/37 tests passed**

### Testing with Synthetic Data

Generate synthetic test data for development:

```python
from validation.generate_test_data import (
    generate_synthetic_connectivity_matrix,
    create_mock_freesurfer_structure,
    generate_labels,
)

# Create a synthetic 246x246 connectivity matrix
matrix = generate_synthetic_connectivity_matrix(
    n_rois=246,
    density=0.2,
    seed=42
)

# Create mock FreeSurfer directory for testing
fs_path = create_mock_freesurfer_structure(
    base_dir="/tmp/test_fs",
    subject_id="sub-01",
    version="7.4.1"
)
```

---

## Diffusion Pipeline (DWI)

### Overview

The diffusion pipeline processes DTI/NODDI data through:
1. DICOM to NIfTI conversion
2. Preprocessing (denoising, distortion correction)
3. Tractography (iFOD2 + ACT)
4. Connectome generation

### Configuration

Edit `mrtrix3_demon_addon/config.json`:

```json
{
    "account": "your-slurm-account",
    "partition": "your-partition",
    "time": "8:00:00",
    "templates": "/path/to/templates/",
    "mrtrix3_sif": "container/mrtrix3_with_ants.sif"
}
```

### Running the Pipeline

```bash
cd mrtrix3_demon_addon

# Process a single subject
python run_pipeline.py \
    --input /path/to/dicom/subject01 \
    --output /path/to/output/subject01 \
    --commands enhanced_commands_dti.json

# Dry run (generate scripts without submitting)
python run_pipeline.py \
    --input /path/to/dicom/subject01 \
    --output /path/to/output/subject01 \
    --commands enhanced_commands_dti.json \
    --dry-run
```

### Command Specifications

| File | Use Case |
|------|----------|
| `enhanced_commands_dti.json` | Single-shell DTI (42 steps) |
| `enhanced_commands_ms.json` | Multi-shell diffusion |
| `commands_NHP_ms.json` | Non-human primate |

### Output Structure

```
output/subject01/
├── connectome_Brainnetome_counts.csv    # Streamline counts
├── connectome_Brainnetome_scaled.csv    # Volume-normalized
├── standardized_connectome_report.json  # Graph metrics
├── dwi_Brainnetome.nii.gz              # Atlas in native space
└── sift_1M_*.tck                       # Filtered tractogram
```

---

## fMRI Pipeline

### Overview

The fMRI pipeline processes resting-state BOLD data through:
1. BIDS conversion
2. fMRIPrep preprocessing
3. Signal denoising
4. ROI extraction
5. Connectivity computation
6. Graph metrics

### Configuration

Edit `nilearn_RSB_analysis_pipeline/configs/pipeline_config.yaml`:

```yaml
# Key settings to update
template:
  output_space: "MNI152NLin2009cAsym"
  resolution: "res-2"

atlas:
  name: "schaefer_2018"
  n_rois: 400

paths:
  project_root: "/path/to/your/project/"
  fmriprep_container: "/path/to/fmriprep.sif"
  fs_license: "/path/to/freesurfer_license.txt"

slurm:
  account: "your-account"
  partition: "your-partition"
```

### Running the Full Pipeline

```bash
cd nilearn_RSB_analysis_pipeline

# Run full pipeline for a subject
python scripts/submit_pipeline.py \
    --subject 01 \
    --session 1 \
    --config configs/pipeline_config.yaml

# Dry run
python scripts/submit_pipeline.py \
    --subject 01 \
    --session 1 \
    --config configs/pipeline_config.yaml \
    --dry-run
```

### Running Individual Steps

```bash
# Skip BIDS conversion and fMRIPrep (use existing derivatives)
python scripts/submit_pipeline.py \
    --subject 01 \
    --start-step 3 \
    --end-step 6 \
    --fmriprep-derivatives /path/to/derivatives \
    --config configs/pipeline_config.yaml

# Run steps locally (no SLURM)
python scripts/03_denoise_signals.py --subject 01 --session 1 --config configs/pipeline_config.yaml
python scripts/04_extract_roi_signals.py --subject 01 --session 1 --config configs/pipeline_config.yaml
python scripts/05_compute_connectivity.py --subject 01 --session 1 --config configs/pipeline_config.yaml
python scripts/06_graph_metrics.py --subject 01 --session 1 --config configs/pipeline_config.yaml
```

### Atlas Options

| Atlas | Configuration | ROIs |
|-------|--------------|------|
| Schaefer 2018 | `name: "schaefer_2018"`, `n_rois: 400` | 100-1000 |
| Brainnetome | `name: "custom"`, `atlas_file: "/path/to/Brainnetome.nii.gz"` | 246 |
| FreeSurfer DK | `name: "freesurfer_dk"` | 84 |
| FreeSurfer Destrieux | `name: "freesurfer_destrieux"` | 164 |

### Output Structure

```
data/outputs/sub-01/ses-01/
├── timeseries/
│   └── sub-01_ses-01_task-rest_desc-roi_timeseries.csv
├── connectivity/
│   ├── sub-01_ses-01_task-rest_desc-correlation_matrix.csv
│   └── sub-01_ses-01_task-rest_desc-correlation_matrix.npy
├── graph_metrics/
│   ├── sub-01_ses-01_task-rest_desc-graph_metrics.json
│   └── sub-01_ses-01_task-rest_desc-nodal_metrics.csv
└── qc/
    └── sub-01_ses-01_task-rest_desc-qc_summary.png
```

---

## Using the Shared Package

### Loading Connectivity Matrices

```python
from connectivity_shared import load_connectivity_matrix, save_connectivity_matrix

# Load a matrix (auto-detects format)
matrix, labels = load_connectivity_matrix("connectivity.csv")
print(f"Matrix shape: {matrix.shape}")
print(f"ROI labels: {labels[:5]}")

# Save with standardized format
save_connectivity_matrix(
    matrix,
    "output_connectivity.csv",
    labels=labels,
    metadata={"atlas": "brainnetome", "subject": "sub-01"}
)
```

### Computing Graph Metrics

```python
from connectivity_shared import GraphMetrics

# Initialize with options
gm = GraphMetrics(
    threshold=0.0,           # Threshold for binarization
    n_random_networks=100,   # For small-world calculation
    seed=42
)

# Compute all metrics
result = gm.compute_all(matrix, labels)

# Access results
print(f"Global efficiency: {result.global_metrics['global_efficiency']:.3f}")
print(f"Small-worldness: {result.global_metrics['small_worldness_sigma']:.3f}")

# Save to JSON
result.to_json("graph_metrics.json")
```

### Working with Atlas Labels

```python
from connectivity_shared import (
    get_atlas_labels,
    get_atlas_info,
    verify_labels_consistency,
)

# Get canonical labels for an atlas
labels = get_atlas_labels("brainnetome")
print(f"Brainnetome has {len(labels)} ROIs")

# Get atlas metadata
info = get_atlas_info("brainnetome")
print(f"Description: {info['description']}")

# Verify consistency between label lists
consistent, details = verify_labels_consistency(labels1, labels2)
if not consistent:
    print(f"Mismatch: {details['only_in_1']}")
```

### FreeSurfer Detection

```python
from connectivity_shared import FreeSurferDetector

detector = FreeSurferDetector()
info = detector.detect("/path/to/subjects/sub-01")

if info.found:
    print(f"FreeSurfer version: {info.version}")
    print(f"Available atlases: {info.available_atlases}")
else:
    print(f"Warning: {info.warning}")
```

---

## Generating QC Reports

### Using QCVisualizer

```python
from connectivity_shared import QCVisualizer, QC_VIZ_AVAILABLE

if QC_VIZ_AVAILABLE:
    viz = QCVisualizer(output_dir="./qc_images")

    # Create individual plots
    viz.create_connectivity_heatmap(
        matrix,
        labels=labels,
        output_filename="heatmap.png",
        title="Connectivity Matrix"
    )

    viz.create_edge_histogram(matrix, output_filename="histogram.png")

    viz.create_degree_distribution(
        matrix,
        labels=labels,
        output_filename="degree.png"
    )

    # Create comprehensive summary
    viz.create_qc_summary(
        matrix,
        labels=labels,
        metrics=result.global_metrics,
        output_filename="qc_summary.png",
        modality="fmri"
    )
```

### Generating HTML Reports

```python
from connectivity_shared import HTMLReportGenerator

report = HTMLReportGenerator(output_dir="./reports")

# Add metadata
report.add_metadata(
    subject_id="sub-01",
    session_id="ses-01",
    modality="fMRI",
    atlas="Brainnetome",
    n_rois=246,
)

# Add metrics
report.add_metrics('global', {
    'n_nodes': 246,
    'n_edges': 5000,
    'global_efficiency': 0.52,
    'clustering_coefficient': 0.45,
})

# Add images
report.add_image(
    "heatmap",
    "./qc_images/heatmap.png",
    caption="Connectivity Matrix"
)

# Add warnings if needed
report.add_warning("High motion detected: mean FD = 0.8mm")

# Generate report
output_path = report.generate("sub-01_qc_report.html")
print(f"Report saved to: {output_path}")
```

### Quick Report Generation

```python
from connectivity_shared import generate_connectivity_report

# One-line report generation
report_path = generate_connectivity_report(
    matrix_path="connectivity.csv",
    metrics_path="graph_metrics.json",
    qc_images={
        "heatmap": "./qc/heatmap.png",
        "summary": "./qc/summary.png",
    },
    subject_id="sub-01",
    modality="fmri",
    atlas="brainnetome",
)
```

---

## Troubleshooting

### Common Issues

#### "connectivity_shared not found"

```bash
# Ensure the package is installed
pip install -e ./connectivity_shared

# Check installation
python -c "import connectivity_shared"
```

#### Matrix loading errors

```python
# For headerless CSV (legacy DWI output)
from connectivity_shared import convert_legacy_csv

new_path = convert_legacy_csv(
    "old_connectome.csv",
    "new_connectome.csv",
    atlas="brainnetome"
)
```

#### FreeSurfer not detected

```python
from connectivity_shared import FreeSurferDetector

detector = FreeSurferDetector(
    version_priority=['7', '8', '5.3'],  # Check these versions
    include_template_atlases=True         # Always include Brainnetome
)
info = detector.detect(subjects_dir)
print(info.warning)  # Check for specific issues
```

#### Template space mismatch

```bash
# Verify atlas compatibility between pipelines
python validation/test_atlas_consistency.py \
    --dwi-atlas /path/to/dwi/Brainnetome.nii.gz \
    --fmri-atlas /path/to/fmri/rBrainnetome.nii.gz
```

### SLURM Job Issues

```bash
# Check job status
squeue -u $USER

# View job details
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed

# Check logs
cat logs/<job_name>_<job_id>.log
```

### Getting Help

1. Check the validation tests for usage examples
2. Review session documentation in `docs/`
3. Open an issue on the repository

---

## Quick Reference

### File Formats

| Extension | Description | Tool |
|-----------|-------------|------|
| `.csv` | Connectivity matrix with headers | `load_connectivity_matrix()` |
| `.npy` | NumPy array (no headers) | `load_connectivity_matrix()` |
| `.json` | Graph metrics | `GraphMetricsResult.to_json()` |
| `.html` | QC report | `HTMLReportGenerator.generate()` |

### Key Functions

```python
# Matrix I/O
load_connectivity_matrix(path, labels_file=None)
save_connectivity_matrix(matrix, path, labels=None, metadata=None)
convert_legacy_csv(input_path, output_path, atlas=None)

# Graph Metrics
GraphMetrics(threshold=0.0, n_random_networks=100)
gm.compute_all(matrix, labels)
result.to_json(path)

# Atlas Labels
get_atlas_labels(atlas_name)  # 'brainnetome', 'freesurfer_dk', 'destrieux'
get_atlas_info(atlas_name)
verify_labels_consistency(labels1, labels2)

# FreeSurfer
FreeSurferDetector().detect(subjects_dir)

# Visualization
QCVisualizer(output_dir).create_qc_summary(matrix, labels, metrics)
HTMLReportGenerator(output_dir).generate(filename)
```
