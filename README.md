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
│       ├── dicom_to_bids.py         # DICOM → BIDS conversion
│       ├── brain_extraction.py      # FSL BET brain mask wrapper
│       ├── freesurfer_detector.py   # FreeSurfer detection
│       ├── matrix_io.py             # Connectivity matrix I/O
│       ├── graph_metrics.py         # NetworkX graph theory
│       ├── atlas_labels.py          # Canonical atlas labels
│       ├── qc_visualization.py      # QC image generation
│       └── html_report.py           # HTML report generator
├── mrtrix3_demon_addon/          # Diffusion pipeline
├── nilearn_RSB_analysis_pipeline/ # fMRI pipeline
├── validation/                   # Test suite
│   └── run_all_validations.sh
└── docs/                         # Documentation
    └── version_control_guide.md
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

### 3. Convert DICOM to BIDS

Both pipelines start from raw DICOM data. The shared `DicomToBIDS` converter classifies and organises your data into a standard BIDS layout, ready for either pipeline.

```python
from connectivity_shared import DicomToBIDS

converter = DicomToBIDS()
result = converter.convert(
    dicom_dir="/path/to/raw/dicoms",
    output_dir="/path/to/bids_output",
    subject_id="01",
    session_id="1",          # optional
)

print(result.modalities_found)  # e.g. ['anat', 'dwi', 'fmap', 'func']
print(result.files)             # {modality: [bids_paths]}
print(result.warnings)          # any issues encountered
```

The converter automatically:
- Runs `dcm2niix` on the DICOM directory
- Classifies each series by its JSON sidecar (DWI, BOLD, T1w, FLAIR, fieldmaps, etc.)
- Filters out SBRef images
- Organises files into BIDS directories with correct naming
- Copies bvec/bval files alongside DWI data
- Adds `IntendedFor` metadata to fieldmap sidecars
- Writes `dataset_description.json`

You can customise the series description patterns used for classification:

```python
converter = DicomToBIDS(config={
    "series_patterns": {
        "dwi":       ["NODDI", "DTI", "DWI", "DKI"],
        "func":      ["BOLD", "resting", "rest", "fMRI"],
        "anat_t1w":  ["tfl3d", "T1w", "MPRAGE"],
        "anat_flair": ["FLAIR", "spcir"],
        "fmap":      ["FIELD_MAP", "fieldmap", "FieldMap"],
    }
})
```

### 4. (Optional) Brain Extraction

If you need a skull-stripped T1w before running a pipeline, the shared `BrainExtractor` wraps FSL BET:

```python
from connectivity_shared import BrainExtractor

extractor = BrainExtractor(fractional_intensity=0.5)

# From a BIDS dataset (auto-finds the T1w)
result = extractor.extract_from_bids(
    bids_dir="/path/to/bids_output",
    subject_id="01",
    session_id="1",
)

# Or from a specific file
result = extractor.extract(
    t1w_path="/path/to/sub-01_T1w.nii.gz",
    output_dir="/path/to/output",
)

print(result.brain_path)   # skull-stripped brain
print(result.mask_path)    # binary brain mask
```

Requires FSL to be installed and `bet` on your PATH.

### 5. Launch a Pipeline

**fMRI pipeline** -- The fMRI pipeline has its own BIDS conversion step built in (Step 1), so you can either use the shared converter above or let the pipeline handle it. Either way, the pipeline expects raw DICOMs under `data/raw/sub-{id}/` or an existing BIDS dataset under `data/bids/`:

```bash
python nilearn_RSB_analysis_pipeline/scripts/submit_pipeline.py \
    --subject 01 --session 1 --config configs/pipeline_config.yaml
```

**DWI pipeline** -- The DWI pipeline uses its own `ImageTypeChecker` to convert DICOMs into an `mrtrix3_inputs/` folder. Point it at a subject folder containing a `tmp/` directory with raw DICOMs:

```bash
cd mrtrix3_demon_addon && python run_pipeline.py \
    subject_name /path/to/subject config.json enhanced_commands_dti.json --human
```

If you used the shared `DicomToBIDS` converter first, the DWI pipeline will still run its own `ImageTypeChecker` step on the raw data. The shared converter is most useful when you want a standardised BIDS dataset for archival, QC, or feeding into other tools (e.g. fMRIPrep directly).

### Pipeline-Specific Documentation

- **For diffusion data**: See `mrtrix3_demon_addon/README.md`
- **For fMRI data**: See `nilearn_RSB_analysis_pipeline/README.md`

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
| [Version Control Guide](docs/version_control_guide.md) | Git workflow and dependency management |

## Requirements

### Core Dependencies
- Python 3.8+
- NumPy, Pandas, SciPy
- NetworkX (graph metrics)
- NiBabel (neuroimaging I/O)

### Optional Dependencies
- Matplotlib (QC visualization)
- nilearn (atlas fetching, fMRI processing)

### Preprocessing
- dcm2niix (DICOM-to-NIfTI conversion, used by `DicomToBIDS`)
- FSL (brain extraction via `BrainExtractor`, optional)

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