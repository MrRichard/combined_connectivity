# Combined Connectivity Analysis

Harmonized neuroimaging pipelines for ROI-level connectivity analysis, supporting both structural (DTI/NODDI) and functional (resting-state fMRI) modalities.

## Overview

This project provides a unified framework for brain connectivity analysis across two imaging modalities:

| Modality | Pipeline | Method |
|----------|----------|--------|
| **Diffusion (DWI)** | `mrtrix3_demon_addon/` | Tractography-based structural connectivity |
| **Functional (fMRI)** | `nilearn_RSB_analysis_pipeline/` | Correlation-based functional connectivity |

Both pipelines share common utilities via the `connectivity_shared` Python package, ensuring consistent outputs for cross-modal comparison.

## Project Structure

```
combined_connectivity/
├── connectivity_shared/          # Shared Python package
│   └── src/connectivity_shared/
│       ├── freesurfer_detector.py   # FreeSurfer detection
│       ├── matrix_io.py             # Connectivity matrix I/O
│       ├── graph_metrics.py         # NetworkX graph theory
│       ├── atlas_labels.py          # Canonical atlas labels
│       ├── qc_visualization.py      # QC image generation
│       └── html_report.py           # HTML report generator
├── mrtrix3_demon_addon/          # Diffusion pipeline
├── nilearn_RSB_analysis_pipeline/ # fMRI pipeline
├── validation/                   # Test suite
│   ├── test_matrix_io.py
│   ├── test_graph_metrics.py
│   ├── test_qc_visualization.py
│   ├── test_atlas_labels.py
│   ├── test_atlas_consistency.py
│   ├── test_html_report.py
│   └── run_all_validations.sh
└── docs/                         # Documentation
    ├── session_2026-01-30_initial_exploration.md
    ├── version_control_guide.md
    └── usage_guide.md
```

## Quick Start

### 1. Install the Shared Package

```bash
# Development installation (recommended)
pip install -e ./connectivity_shared

# Or install dependencies only
pip install numpy pandas scipy networkx nibabel matplotlib
```

### 2. Run Validation Tests

```bash
bash validation/run_all_validations.sh
```

### 3. Choose Your Pipeline

- **For diffusion data**: See `mrtrix3_demon_addon/README.md`
- **For fMRI data**: See `nilearn_RSB_analysis_pipeline/README.md`
- **For pipeline usage**: See `docs/usage_guide.md`

## Key Features

### Harmonized Outputs

Both pipelines produce consistent output formats:

- **Connectivity matrices**: CSV with ROI headers (self-documenting)
- **Graph metrics**: Standardized JSON with 27+ metrics
- **QC reports**: HTML with embedded visualizations

### Supported Atlases

| Atlas | ROIs | Availability |
|-------|------|--------------|
| Brainnetome | 246 | Both pipelines (template-based) |
| FreeSurfer DK | 84 | Both pipelines (requires recon) |
| FreeSurfer Destrieux | 164 | Both pipelines (requires recon) |
| Schaefer 2018 | 100-1000 | fMRI pipeline only |

### Graph Metrics

Computed using NetworkX for consistency:

**Global Metrics:**
- Network size (nodes, edges, density)
- Clustering coefficient (binary & weighted)
- Global/local efficiency
- Characteristic path length
- Small-worldness (sigma, omega)
- Modularity, assortativity
- Rich club coefficient

**Nodal Metrics:**
- Degree, strength
- Clustering coefficient
- Betweenness centrality
- Local efficiency

## Documentation

| Document | Description |
|----------|-------------|
| [Usage Guide](docs/usage_guide.md) | Running pipelines and processing data |
| [Version Control Guide](docs/version_control_guide.md) | Git workflow and dependency management |
| [Session Notes](docs/session_2026-01-30_initial_exploration.md) | Development history and decisions |

## Requirements

### Core Dependencies
- Python 3.8+
- NumPy, Pandas, SciPy
- NetworkX (graph metrics)
- NiBabel (neuroimaging I/O)

### Optional Dependencies
- Matplotlib (QC visualization)
- nilearn (atlas fetching, fMRI processing)

### Pipeline-Specific
- **DWI**: MRtrix3, ANTs, FSL (via Singularity)
- **fMRI**: fMRIPrep (via Singularity)
- **Both**: SLURM for HPC job submission

## Template Spaces

The pipelines use different MNI template spaces:

| Pipeline | Template | Notes |
|----------|----------|-------|
| fMRI | `MNI152NLin2009cAsym` | TemplateFlow standard |
| DWI | `mni_icbm152_nlin_sym_09a` | ICBM symmetric |

For cross-modal comparison, use `validation/test_atlas_consistency.py` to verify atlas alignment.