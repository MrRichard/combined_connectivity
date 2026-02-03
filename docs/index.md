# Combined Connectivity Documentation

Welcome to the Combined Connectivity documentation. This system processes functional (fMRI) and structural (DWI) neuroimaging data to generate brain connectivity matrices and graph theory metrics.

## Quick Links

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture.md) | System architecture with UML diagrams |
| [Usage Guide](usage_guide.md) | Installation, configuration, and usage |
| [Version Control Guide](version_control_guide.md) | Git workflow and release management |

## UML Diagrams

Visual documentation of the system architecture:

### System Architecture
![System Architecture](svg/architecture.svg)
[PlantUML Source](uml/architecture.puml)

### Class Diagrams

| Diagram | Description |
|---------|-------------|
| ![Shared Package](svg/class_shared.svg) | connectivity_shared package classes |
| ![fMRI Pipeline](svg/class_fmri_pipeline.svg) | fMRI pipeline classes |
| ![DWI Pipeline](svg/class_dwi_pipeline.svg) | DWI pipeline classes |

### Behavioral Diagrams

| Diagram | Description |
|---------|-------------|
| ![Sequence Diagram](svg/sequence_fmri.svg) | fMRI pipeline execution sequence |
| ![State Machine](svg/state_pipeline.svg) | Pipeline state transitions |
| ![Data Flow](svg/data_flow.svg) | Data flow through the system |
| ![Use Cases](svg/use_case.svg) | System use cases |

## Project Structure

```
combined_connectivity/
├── connectivity_shared/              # Shared Python package
│   └── src/connectivity_shared/
│       ├── graph_metrics.py          # NetworkX-based metrics
│       ├── matrix_io.py              # Matrix I/O utilities
│       ├── freesurfer_detector.py    # FreeSurfer detection
│       ├── atlas_labels.py           # Atlas label definitions
│       ├── qc_visualization.py       # QC image generation
│       └── html_report.py            # HTML report generator
│
├── nilearn_RSB_analysis_pipeline/    # fMRI pipeline
│   ├── configs/pipeline_config.yaml  # Master configuration
│   └── scripts/
│       ├── 01-07_*.py               # Processing steps
│       └── submit_pipeline.py        # SLURM orchestrator
│
├── mrtrix3_demon_addon/              # DWI pipeline
│   ├── run_pipeline.py               # Main orchestrator
│   ├── Connectome.py                 # Graph metrics
│   └── ImageTypeChecker.py           # Image classification
│
├── validation/                       # Test suite
│   └── test_*.py                     # Validation tests
│
└── docs/                             # Documentation
    ├── index.md                      # This file
    ├── architecture.md               # Architecture overview
    ├── usage_guide.md                # Usage instructions
    ├── version_control_guide.md      # Version control
    ├── uml/                          # PlantUML sources
    └── svg/                          # Rendered diagrams
```

## Getting Started

### Prerequisites
- Python 3.8+
- SLURM (for HPC deployment)
- Singularity (for fMRIPrep)

### Installation

```bash
# Clone repository
git clone <repository-url> combined_connectivity
cd combined_connectivity

# Install shared package
pip install -e ./connectivity_shared

# Verify installation
python -c "import connectivity_shared; print(connectivity_shared.__version__)"
```

### Run Validation Tests

```bash
bash validation/run_all_validations.sh
```

### Process Data

```bash
# fMRI Pipeline
python scripts/submit_pipeline.py --subject 01 --session 1 --config configs/pipeline_config.yaml

# DWI Pipeline
python run_pipeline.py --input /path/to/dicom --output /path/to/output
```

## Pipeline Comparison

| Feature | fMRI Pipeline | DWI Pipeline |
|---------|--------------|--------------|
| Input | DICOM/BIDS | DICOM/NIfTI |
| Preprocessing | fMRIPrep | MRtrix3 |
| Connectivity | Correlation | Streamline counts |
| Default Atlas | Schaefer 2018 | Brainnetome |
| Graph Metrics | shared.GraphMetrics | shared.GraphMetrics |
| Output Format | CSV + JSON | CSV + JSON |

## Support

- Run validation tests to verify installation
- Check the [Usage Guide](usage_guide.md) for detailed instructions
- Review logs in `logs/` directory for debugging
- Open an issue on the repository for bugs or feature requests
