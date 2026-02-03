# Combined Connectivity Harmonization - Session 1: Initial Exploration

**Date:** 2026-01-30
**Session Type:** Planning and exploration
**Plan File:** `/home/richard/.claude/plans/stateful-rolling-pretzel.md`

---

## Session Summary

This session established the foundation for harmonizing two neuroimaging pipelines:
1. **Diffusion (DTI/NODDI)** via MRtrix3 - structural connectivity
2. **Resting-State BOLD** via nilearn/fmriprep - functional connectivity

---

## Codebase Exploration Findings

### Diffusion Pipeline (`mrtrix3_demon_addon/`)

**Main Files:**
- `run_pipeline.py` (818 lines) - Main orchestrator
- `run_pipeline_legacy.py` (49K) - Legacy version with extended functionality
- `ImageTypeChecker.py` - DICOM classification and validation
- `SlurmBatch.py` - SLURM job generation
- `scripts/generate_standardized_report.py` - Graph metrics (pure NumPy)
- `scripts/connectome_aggregator.py` - Multi-subject aggregation

**Command Specifications (JSON):**
- `enhanced_commands_dti.json` - Single-shell DTI (42 steps)
- `enhanced_commands_ms.json` - Multi-shell diffusion
- `commands_NHP_ms.json` - Non-human primate variant

**Processing Flow:**
1. DICOM → NIfTI conversion via dcm2niix
2. Denoising (MP-PCA), Gibbs correction
3. Distortion correction (fieldmap or reverse phase encoding)
4. Bias field correction (ANTs N4)
5. Brain masking (dwi2mask or pre-existing)
6. Response function estimation (Tournier method)
7. FOD estimation (CSD)
8. Anatomical processing and coregistration (FSL FLIRT)
9. 5-tissue-type segmentation
10. Tractography (10M streamlines, iFOD2 + ACT)
11. SIFT filtering (1M streamlines)
12. MNI template registration (ANTs)
13. Atlas transformation to native space
14. Connectome generation (tck2connectome)
15. Graph metrics calculation

**Atlases Supported:**
- Brainnetome (246 ROIs) - template-based, all species
- FreeSurfer DK (84 ROIs) - human only, requires aparc+aseg.mgz
- FreeSurfer Destrieux (164 ROIs) - human only, requires aparc.a2009s+aseg.mgz

**FreeSurfer Detection:** `run_pipeline.py:325-431`
- Detects versions: 8.0, 7.x, 5.3 (priority order)
- Functions: `detect_freesurfer_version()`, `find_freesurfer_files()`

**Graph Metrics:** `scripts/generate_standardized_report.py`
- `PureNumpyGraphMetrics` class (no external dependencies)
- Metrics: clustering, efficiency, path length, small-worldness, assortativity

**Output Format:**
- Connectivity matrices: CSV (headerless)
- Graph metrics: JSON (`standardized_connectome_report.json`)

---

### RS-fMRI Pipeline (`nilearn_RSB_analysis_pipeline/`)

**Main Files:**
- `scripts/submit_pipeline.py` (424 lines) - SLURM job orchestrator
- `scripts/01_validate_and_convert.py` (917 lines) - BIDS conversion
- `scripts/02_run_fmriprep.py` (700 lines) - fmriprep execution
- `scripts/03_denoise_signals.py` (893 lines) - Confound regression
- `scripts/04_extract_roi_signals.py` (686 lines) - ROI extraction
- `scripts/05_compute_connectivity.py` (493 lines) - Correlation matrices
- `scripts/06_graph_metrics.py` (957 lines) - Graph theory analysis
- `scripts/07_visualize_rois.py` (594 lines) - QC visualization

**Configuration:** `configs/pipeline_config.yaml` (YAML-based)

**Processing Flow:**
1. DICOM → BIDS conversion via dcm2niix
2. fMRIPrep preprocessing (Singularity container)
3. Signal denoising (confound regression, bandpass filtering, scrubbing)
4. ROI time series extraction
5. Correlation matrix computation (Pearson/partial)
6. Fisher z-transformation
7. Thresholding (proportional, absolute, or adaptive)
8. Graph theory analysis
9. QC visualization

**Atlases Supported:**
- Schaefer 2018 (100-1000 ROIs) - default
- Custom atlas support (Brainnetome configured)
- Destrieux, AAL, Harvard-Oxford

**FreeSurfer Integration:** `submit_pipeline.py:105-128`
- Reuses existing recons via fmriprep `--fs-subjects-dir`
- Function: `check_for_existing_recons()`

**Graph Metrics:** `scripts/06_graph_metrics.py`
- `GraphAnalyzer` class (NetworkX-based)
- Optional BCT (Brain Connectivity Toolbox) support

**Output Format:**
- Connectivity matrices: .npy (primary), .csv (secondary)
- Graph metrics: JSON + YAML

---

## User Decisions (Confirmed)

| Question | Decision |
|----------|----------|
| Implementation priority | Output formats first, then QC |
| NHP support for fMRI | Human only (NHP in diffusion only) |
| FreeSurfer atlases in fMRI | Important - add DK and Destrieux |
| Graph metrics library | NetworkX (consistent across both) |

---

## Harmonization Plan Summary

### Phase 1: Shared Utilities (Weeks 1-2)
- Create `shared/` module with:
  - `freesurfer_detector.py` - Unified FS detection
  - `matrix_io.py` - Connectivity matrix I/O
  - `graph_metrics.py` - NetworkX-based metrics
  - `naming_convention.py` - BIDS-style naming

### Phase 2: Output Harmonization (Weeks 3-4)
- Standardize connectivity matrix format: CSV with headers
- Unify graph metrics JSON schema
- Integrate shared modules into both pipelines

### Phase 3: QC Harmonization (Weeks 5-6)
- Standardize atlas overlay images
- Create unified HTML QC reports
- Add QC generation to diffusion pipeline

### Phase 4: Input Integration (Weeks 7-8)
- Deploy unified FreeSurfer detection
- Add FreeSurfer atlas support to fMRI pipeline
- Documentation and testing

---

## Key Files to Modify

### Diffusion Pipeline
| File | Modification |
|------|--------------|
| `run_pipeline.py` | Integrate shared FreeSurfer detection |
| `scripts/generate_standardized_report.py` | Replace with shared NetworkX metrics |
| Connectome outputs | Add CSV headers |

### fMRI Pipeline
| File | Modification |
|------|--------------|
| `scripts/05_compute_connectivity.py` | Standardize CSV format |
| `scripts/06_graph_metrics.py` | Use shared NetworkX module |
| `scripts/04_extract_roi_signals.py` | Add FreeSurfer atlas support |
| `scripts/07_visualize_rois.py` | Use shared QC visualization |

---

## New Files to Create

```
combined_connectivity/
├── shared/
│   ├── __init__.py
│   ├── freesurfer_detector.py
│   ├── matrix_io.py
│   ├── graph_metrics.py
│   ├── qc_visualization.py
│   ├── naming_convention.py
│   └── schemas/
│       └── graph_metrics_schema.json
├── validation/
│   ├── test_matrix_io.py
│   ├── test_graph_metrics.py
│   ├── test_freesurfer_detection.py
│   ├── generate_test_data.py
│   └── run_all_validations.sh
└── docs/
    └── (this file and future session docs)
```

---

## Validation Strategy

Since no test data is available (privacy constraints):
1. Generate synthetic connectivity matrices with known properties
2. Create mock FreeSurfer directory structures
3. Validate NetworkX implementation produces expected results
4. Cross-validate output formats between pipelines

Validation scripts will be in `validation/` folder with instructions for running and returning results.

---

## Progress Made

### Phase 1 Implementation - COMPLETED

1. **Created `connectivity_shared` Python package:**
   ```
   connectivity_shared/
   ├── pyproject.toml
   ├── README.md
   └── src/connectivity_shared/
       ├── __init__.py
       ├── freesurfer_detector.py   # Unified FS detection
       ├── matrix_io.py             # Matrix I/O with CSV headers
       └── graph_metrics.py         # NetworkX graph metrics (27 metrics)
   ```

2. **Key Features Implemented:**
   - `FreeSurferDetector` class with support for FS 5.3, 7.x, 8.0
   - `load_connectivity_matrix()` / `save_connectivity_matrix()` with headers
   - `convert_legacy_csv()` for DWI pipeline migration
   - `GraphMetrics` class computing 27 global metrics + nodal metrics
   - JSON serialization for metrics export

3. **Validation Suite Created:**
   ```
   validation/
   ├── generate_test_data.py    # Synthetic data generation
   ├── test_matrix_io.py        # Matrix I/O tests
   ├── test_graph_metrics.py    # Graph metrics tests
   └── run_all_validations.sh   # Run all tests
   ```

4. **Package Successfully Installed and Tested**

### Phase 2: Pipeline Integration - COMPLETED

1. **fMRI Pipeline Integration** (`scripts/06_graph_metrics.py`):
   - Added `connectivity_shared` import with graceful fallback
   - Modified `run()` method to use `SharedGraphMetrics` when available
   - Outputs new standardized JSON file alongside backward-compatible formats
   - Tested and verified working

2. **DWI Pipeline Integration** (`scripts/generate_standardized_report.py`):
   - Added `connectivity_shared` import with fallback to `PureNumpyGraphMetrics`
   - Modified `ConnectomeReporter` class to detect and use shared module
   - `calculate_graph_metrics()` now uses NetworkX-based shared implementation
   - Reports include `metrics_implementation` field for tracking
   - Tested and verified working

### Files Modified

**fMRI Pipeline:**
- `nilearn_RSB_analysis_pipeline/scripts/06_graph_metrics.py`
  - Lines 1-35: Added imports for connectivity_shared
  - Lines 863-920: Modified `run()` to use shared module

**DWI Pipeline:**
- `mrtrix3_demon_addon/scripts/generate_standardized_report.py`
  - Lines 1-20: Added imports for connectivity_shared
  - Lines 361-385: Modified `ConnectomeReporter.__init__`
  - Lines 502-520: Modified `calculate_graph_metrics()`

### Validation Tests - ALL PASSED

```
Matrix I/O Tests: 6/6 passed
  - CSV Save/Load
  - Legacy CSV Load
  - Legacy CSV Conversion
  - Matrix Validation
  - Matrix Stats
  - NPY Format

Graph Metrics Tests: 8/8 passed
  - Basic Metrics
  - Clustering Metrics
  - Efficiency Metrics
  - Small-World Metrics
  - Topology Metrics
  - Nodal Metrics
  - JSON Serialization
  - Edge Cases
```

---

## Template Space Analysis

### Critical Finding: Different Template Spaces

The two pipelines use **different MNI template spaces**:

| Pipeline | Template Space | Description |
|----------|---------------|-------------|
| **fMRI** | `MNI152NLin2009cAsym` | Asymmetric ICBM 2009c template (TemplateFlow standard) |
| **DWI** | `mni_icbm152_nlin_sym_09a` | Symmetric ICBM 2009a template |

### Brainnetome Atlas Files

| Pipeline | Atlas Location |
|----------|---------------|
| fMRI | `/isilon/.../nilearn_RSB_analysis_pipeline/templates/rBrainnetome.nii.gz` |
| DWI | `/isilon/.../mrtrix3_demon_addon/templates/mni_icbm152_nlin_sym_09a/Brainnetome.nii.gz` |

### Implications

1. **Coordinate Systems**: Both templates are in MNI space, but coordinates may differ slightly
2. **Atlas Labels**: The Brainnetome atlas (246 ROIs) should have identical label values in both files
3. **ROI Boundaries**: At 2mm resolution, ROI boundaries should be nearly identical
4. **Cross-Modal Comparison**: For direct DWI/fMRI matrix comparison, verification is needed that:
   - Same ROI indices correspond to same anatomical regions
   - Label ordering is consistent

### Recommendations

1. **Verify atlas consistency**: Add validation script to compare atlas files
2. **Document label mappings**: Create canonical Brainnetome label file for both pipelines
3. **For cross-modal studies**: Either:
   - Use the same atlas file transformed to each space
   - Verify label correspondence between existing files

---

## Session 1 Continued - QC Visualization Module

Added `qc_visualization.py` to `connectivity_shared` package:

**Features:**
- `QCVisualizer` class with output directory management
- `create_connectivity_heatmap()` - matrix visualization with ROI labels
- `create_edge_histogram()` - edge weight distribution plots
- `create_degree_distribution()` - network topology visualization
- `create_qc_summary()` - comprehensive 4-panel QC figure with metrics
- `create_atlas_overlay()` - atlas parcellation on anatomical (requires nilearn)
- `create_registration_check()` - registration quality visualization

**Validation:** All 5 QC visualization tests pass.

---

## Session 1 Continued - Atlas Labels Module

Added `atlas_labels.py` to `connectivity_shared` package:

**Features:**
- `BRAINNETOME_LABELS` - canonical 246 ROI labels with anatomical names
- `FREESURFER_DK_LABELS` - canonical 84 ROI labels matching MRtrix3 ordering
- `get_atlas_labels(atlas_name)` - retrieve labels by name (with aliases)
- `get_atlas_info(atlas_name)` - get atlas metadata (n_rois, description, etc.)
- `verify_labels_consistency(labels1, labels2)` - compare label lists
- `generate_canonical_labels_file(atlas_name, output_dir)` - create standard file

**Atlas Consistency Validation:**
- Created `validation/test_atlas_consistency.py`
- `compare_atlas_labels()` - compare label sets between atlases
- `compare_atlas_volumes()` - compare voxel counts per ROI
- `compute_overlap_matrix()` - compute Dice overlap between atlases
- Supports loading real NIfTI atlas files for comparison

---

## Session 1 Continued - HTML Report Generator

Added `html_report.py` to `connectivity_shared` package:

**Features:**
- `HTMLReportGenerator` class for building comprehensive QC reports
- Embedded images (base64) for fully self-contained HTML files
- Metrics displayed as visual cards with responsive grid layout
- Tables with sorting and proper formatting
- Warning/info boxes for QC flags
- Custom sections for pipeline-specific content
- Modern CSS with responsive design

**Convenience Function:**
- `generate_connectivity_report()` - one-line report generation from matrix + metrics files

---

## Validation Suite Summary

All 37 tests pass across 6 test suites:

| Test Suite | Tests | Status |
|------------|-------|--------|
| Matrix I/O | 6 | PASSED |
| Graph Metrics | 8 | PASSED |
| QC Visualization | 5 | PASSED |
| Atlas Labels | 7 | PASSED |
| Atlas Consistency | 4 | PASSED |
| HTML Reports | 7 | PASSED |

---

## Next Steps

1. ~~Add QC visualization module to `connectivity_shared`~~ DONE
2. ~~Add FreeSurfer atlas support to fMRI pipeline~~ DONE (DK and Destrieux added)
3. ~~Create atlas label definitions and consistency validation~~ DONE
4. ~~Create unified HTML report generator~~ DONE
5. Integrate HTML reports into pipeline outputs
6. **For real data:** Use `test_atlas_consistency.py --dwi-atlas PATH --fmri-atlas PATH` to verify Brainnetome files match
7. Test with real data when available

---

## Reference: Agent IDs for Resumption

If needed, these agent IDs can resume exploration:
- Diffusion pipeline exploration: `a5ff8b0`
- RS-fMRI pipeline exploration: `a35cb44`
- Project structure exploration: `a1e3711`
- Harmonization planning: `afe5fca`
