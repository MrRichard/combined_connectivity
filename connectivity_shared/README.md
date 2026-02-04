# connectivity-shared

Shared utilities for DTI/NODDI diffusion and resting-state fMRI connectivity analysis pipelines.

## Installation

### Development (editable install)
```bash
cd /path/to/connectivity_shared
pip install -e .
```

### With visualization dependencies
```bash
pip install -e ".[viz]"
```

### With all dependencies (including dev tools)
```bash
pip install -e ".[all]"
```

## Usage

### FreeSurfer Detection
```python
from connectivity_shared import FreeSurferDetector

detector = FreeSurferDetector()
fs_info = detector.detect("/path/to/subject/folder")

if fs_info.found:
    print(f"FreeSurfer version: {fs_info.version}")
    print(f"Available atlases: {fs_info.available_atlases}")
```

### Connectivity Matrix I/O
```python
from connectivity_shared import load_connectivity_matrix, save_connectivity_matrix

# Load matrix with labels
matrix, labels = load_connectivity_matrix("connectome.csv")

# Save with standardized format (CSV with headers)
save_connectivity_matrix(matrix, labels, "output.csv")
```

### Graph Metrics
```python
from connectivity_shared import GraphMetrics

gm = GraphMetrics()
metrics = gm.compute_all(connectivity_matrix, labels)

# Access specific metrics
print(f"Global efficiency: {metrics['global_metrics']['global_efficiency']}")
print(f"Clustering coefficient: {metrics['global_metrics']['mean_clustering_coefficient']}")
```

## Components

| Module | Description |
|--------|-------------|
| `freesurfer_detector` | Detect FreeSurfer reconstructions and available atlases |
| `matrix_io` | Load/save connectivity matrices with standardized format |
| `graph_metrics` | NetworkX-based graph theory metrics |
| `qc_visualization` | QC image generation (atlas overlays, heatmaps) |
| `naming_convention` | BIDS-style naming utilities |

## Integration with Pipelines

### mrtrix3_demon_addon (Diffusion)
Add to `requirements.txt`:
```
connectivity-shared @ file:///path/to/connectivity_shared
```

### nilearn_RSB_analysis_pipeline (fMRI)
Add to requirements or `pipeline_config.yaml`:
```
connectivity-shared @ file:///path/to/connectivity_shared
```

## Version History

- **0.1.0** - Initial release with FreeSurfer detection, matrix I/O, and graph metrics
