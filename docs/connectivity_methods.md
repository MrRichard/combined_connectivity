# Connectivity Analysis Methods

This document describes the mathematical and computational methods used for functional connectivity analysis in the Combined Connectivity framework.

## Table of Contents

1. [Overview](#overview)
2. [ROI-Level Connectivity (Step 5)](#roi-level-connectivity-step-5)
3. [Hippocampal Voxel-Wise Connectivity (Step 5b)](#hippocampal-voxel-wise-connectivity-step-5b)
4. [Graph Theory Metrics](#graph-theory-metrics)
5. [References](#references)

---

## Overview

The framework computes functional connectivity at two spatial scales:

| Analysis Type | Spatial Unit | Typical Size | Use Case |
|--------------|--------------|--------------|----------|
| ROI-Level (Step 5) | Atlas parcels | 84-400 ROIs | Whole-brain network analysis |
| Voxel-Level (Step 5b) | Individual voxels | 500-1500 voxels/hemisphere | Intra-regional connectivity patterns |

Both approaches use the same mathematical framework but differ in spatial granularity.

---

## ROI-Level Connectivity (Step 5)

### Input Data

- **Denoised BOLD timeseries**: 4D NIfTI from Step 3 (denoising)
- **Atlas parcellation**: Brainnetome (246 ROIs), FreeSurfer DK (84 ROIs), or custom

### Signal Extraction

Mean timeseries are extracted from each atlas parcel using nilearn's `NiftiLabelsMasker`:

```python
from nilearn.maskers import NiftiLabelsMasker

masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
roi_timeseries = masker.fit_transform(bold_img)  # Shape: (n_timepoints, n_rois)
```

The `standardize=True` parameter z-scores each ROI's timeseries (zero mean, unit variance).

### Connectivity Measure

Functional connectivity is computed as **Pearson correlation** between ROI timeseries:

$$
r_{ij} = \frac{\sum_{t=1}^{T}(x_i(t) - \bar{x}_i)(x_j(t) - \bar{x}_j)}{\sqrt{\sum_{t=1}^{T}(x_i(t) - \bar{x}_i)^2} \cdot \sqrt{\sum_{t=1}^{T}(x_j(t) - \bar{x}_j)^2}}
$$

Where:
- $x_i(t)$ is the signal from ROI $i$ at time $t$
- $T$ is the number of timepoints
- $\bar{x}_i$ is the mean signal of ROI $i$

Implementation uses nilearn's `ConnectivityMeasure`:

```python
from nilearn.connectome import ConnectivityMeasure

conn_measure = ConnectivityMeasure(kind='correlation')
connectivity_matrix = conn_measure.fit_transform([roi_timeseries])[0]
```

### Fisher Z-Transformation

To normalize the distribution of correlation values (which are bounded [-1, 1]), we apply the **Fisher z-transformation**:

$$
z_{ij} = \text{arctanh}(r_{ij}) = \frac{1}{2} \ln\left(\frac{1 + r_{ij}}{1 - r_{ij}}\right)
$$

This transforms correlations to an approximately normal distribution, which is important for:
- Statistical comparisons across subjects
- Averaging connectivity values
- Applying parametric statistical tests

Values are clipped to [-0.99999, 0.99999] before transformation to avoid infinite values.

### Network Thresholding

Raw connectivity matrices are dense (all ROIs connected). Thresholding creates sparse networks for graph analysis.

#### Proportional Thresholding (Default)

Retains the top $p$% of connections by absolute weight:

```python
# Keep top 10% of connections
threshold = np.percentile(abs_matrix[upper_triangle], 90)
thresholded_matrix[abs_matrix < threshold] = 0
```

**Default**: `threshold_value: 0.1` (top 10% retained)

#### Absolute Thresholding

Retains connections above a fixed correlation value:

```python
thresholded_matrix[abs_matrix < threshold_value] = 0
```

#### Adaptive Thresholding

Uses mean + k*std of connection weights:

$$
\theta = \mu + k \cdot \sigma
$$

Where $\mu$ and $\sigma$ are the mean and standard deviation of upper-triangle values.

### Output Files

| File | Description |
|------|-------------|
| `*_desc-connectivity_matrix.npy` | Full correlation matrix (Fisher-z if enabled) |
| `*_desc-connectivity_matrix.csv` | Same as above, with ROI labels |
| `*_desc-thresholded_connectivity.npy` | Sparse thresholded matrix |
| `*_desc-connectivity_stats.yaml` | Summary statistics |

---

## Hippocampal Voxel-Wise Connectivity (Step 5b)

### Purpose

Analyzes fine-grained connectivity patterns **within** the hippocampus at voxel resolution. This captures intra-hippocampal organization that is lost in ROI-level analysis where the entire hippocampus is a single node.

### Mask Extraction

Hippocampal masks are extracted from the atlas based on label values:

| Atlas | Left Hippocampus | Right Hippocampus |
|-------|------------------|-------------------|
| Brainnetome | Labels 215, 217 (Hipp_L_2_1, Hipp_L_2_2) | Labels 216, 218 (Hipp_R_2_1, Hipp_R_2_2) |
| FreeSurfer DK | Label 17 (Left-Hippocampus) | Label 53 (Right-Hippocampus) |

```python
# Create binary mask from atlas labels
left_mask = np.isin(atlas_data, [215, 217]).astype(np.uint8)
```

### Voxel Timeseries Extraction

Uses nilearn's `NiftiMasker` for voxel-level extraction:

```python
from nilearn.maskers import NiftiMasker

masker = NiftiMasker(mask_img=hippocampus_mask, standardize=True)
voxel_timeseries = masker.fit_transform(bold_img)  # Shape: (n_timepoints, n_voxels)
```

Typical voxel counts at 2mm resolution:
- Left hippocampus: 500-800 voxels
- Right hippocampus: 500-800 voxels

### Voxel Quality Control

Each voxel is assessed for signal quality:

| Metric | Description | Exclusion Criterion |
|--------|-------------|---------------------|
| `std_signal` | Temporal standard deviation | Zero variance |
| `snr` | Signal-to-noise ratio | Below 10th percentile |
| `nan_count` | NaN values | Any NaN |
| `autocorr_lag1` | Temporal autocorrelation | Diagnostic only |

Voxels with zero variance or NaN values are excluded before connectivity computation.

### Intra-Hippocampal Connectivity

The same methods as ROI-level connectivity are applied within each hemisphere:

1. **Correlation matrix**: $n_{voxels} \times n_{voxels}$ Pearson correlations
2. **Fisher z-transform**: Normalize correlation distribution
3. **Thresholding**: Create sparse graph (inherits settings from Step 5)

Matrix dimensions:
- ~700 voxels per hemisphere → ~490,000 unique connections
- Memory: ~4 MB per hemisphere (float32)

### Graph Metrics on Voxel Networks

The `connectivity_shared.GraphMetrics` class computes identical metrics as ROI-level analysis, but nodes represent voxels instead of atlas parcels.

**Global metrics** characterize the hippocampal network topology:
- Small-worldness, modularity, efficiency

**Nodal metrics** identify functional hubs within the hippocampus:
- Degree, betweenness centrality, clustering coefficient

### Nodal Metric NIfTI Maps

Nodal metrics are projected back to brain space as 3D NIfTI volumes:

```python
# Map nodal values back to voxel positions
output_data = np.zeros(mask_shape)
for i, (x, y, z) in enumerate(voxel_indices):
    output_data[x, y, z] = nodal_values[i]
```

This enables visualization of hub locations and spatial patterns in standard neuroimaging viewers.

### Output Structure

```
hippocampal_connectivity/
├── masks/
│   ├── sub-{id}_desc-leftHippocampus_mask.nii.gz
│   ├── sub-{id}_desc-rightHippocampus_mask.nii.gz
│   └── sub-{id}_desc-hippocampus_maskInfo.yaml
├── left_hpc/
│   ├── sub-{id}_desc-leftHpc_voxelTimeseries.npy    # (T, V) array
│   ├── sub-{id}_desc-leftHpc_voxelCoords.npy        # (V, 3) MNI coords
│   ├── sub-{id}_desc-leftHpc_voxelQuality.csv       # QC metrics
│   ├── sub-{id}_desc-leftHpc_connectivity.npy       # Full matrix
│   ├── sub-{id}_desc-leftHpc_thresholded_connectivity.npy
│   ├── sub-{id}_desc-leftHpc_graphMetrics.json      # Global metrics
│   ├── sub-{id}_desc-leftHpc_nodalMetrics.csv       # Per-voxel metrics
│   ├── nodal_maps/
│   │   ├── sub-{id}_desc-leftHpc_degree.nii.gz
│   │   ├── sub-{id}_desc-leftHpc_betweenness.nii.gz
│   │   └── ...
│   └── qc_plots/
├── right_hpc/
│   └── [same structure]
└── sub-{id}_desc-hippocampal_summary.yaml
```

---

## Graph Theory Metrics

Both ROI-level and voxel-level analyses use the shared `GraphMetrics` class from `connectivity_shared`.

### Global Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Global Efficiency** | $E_{glob} = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{1}{d_{ij}}$ | Information integration capacity |
| **Local Efficiency** | Average efficiency of local subgraphs | Fault tolerance |
| **Clustering Coefficient** | Fraction of closed triangles | Local segregation |
| **Characteristic Path Length** | Average shortest path length | Network integration |
| **Small-Worldness** | $\sigma = \frac{C/C_{rand}}{L/L_{rand}}$ | Balance of segregation and integration |
| **Modularity** | Quality of community structure | Functional segregation |
| **Assortativity** | Degree correlation | Hub connectivity pattern |

### Nodal Metrics

| Metric | Description | Identifies |
|--------|-------------|------------|
| **Degree** | Number of connections | Hub nodes |
| **Strength** | Sum of connection weights | Weighted hubs |
| **Betweenness Centrality** | Fraction of shortest paths through node | Connector hubs |
| **Clustering Coefficient** | Local clustering around node | Provincial hubs |
| **Local Efficiency** | Efficiency of node's neighborhood | Resilience contribution |

### Small-World Analysis

Networks are compared to random networks with matched degree distribution:

1. Generate 100 Erdos-Renyi random networks with same edge density
2. Compute mean clustering ($C_{rand}$) and path length ($L_{rand}$)
3. Calculate normalized metrics:
   - $\gamma = C_{real} / C_{rand}$ (normalized clustering)
   - $\lambda = L_{real} / L_{rand}$ (normalized path length)
   - $\sigma = \gamma / \lambda$ (small-worldness)

A network is considered "small-world" if $\sigma > 1$ (typically $\sigma > 1.5$).

---

## Configuration Reference

### Connectivity Settings (pipeline_config.yaml)

```yaml
connectivity:
  method: "correlation"           # correlation, partial_correlation
  fisher_z: true                  # Apply Fisher z-transform
  threshold_method: "proportional" # absolute, proportional, adaptive
  threshold_value: 0.1            # Top 10% connections retained

hippocampal_connectivity:
  enabled: true
  hippocampal_labels:
    brainnetome:
      left: [215, 217]
      right: [216, 218]
    freesurfer_dk:
      left_labels: [17]
      right_labels: [53]
  threshold_method: null          # null = inherit from connectivity
  threshold_value: null
  fisher_z: null
  graph_metrics:
    n_random_networks: 100
    nodal_nifti_maps:
      - degree
      - strength
      - betweenness_centrality
      - clustering
      - local_efficiency
```

---

## References

### Methods

1. **Pearson Correlation**: Zalesky A, Fornito A, Bullmore ET. (2010). Network-based statistic. *NeuroImage*.

2. **Fisher Z-Transform**: Fisher RA. (1915). Frequency distribution of the values of the correlation coefficient. *Biometrika*.

3. **Graph Metrics**: Rubinov M, Sporns O. (2010). Complex network measures of brain connectivity. *NeuroImage*.

4. **Small-World Networks**: Watts DJ, Strogatz SH. (1998). Collective dynamics of small-world networks. *Nature*.

### Atlases

5. **Brainnetome**: Fan L, et al. (2016). The Human Brainnetome Atlas. *Cerebral Cortex*.

6. **Desikan-Killiany**: Desikan RS, et al. (2006). An automated labeling system for subdividing the human cerebral cortex. *NeuroImage*.

### Software

7. **nilearn**: Abraham A, et al. (2014). Machine learning for neuroimaging with scikit-learn. *Frontiers in Neuroinformatics*.

8. **NetworkX**: Hagberg AA, Schult DA, Swart PJ. (2008). Exploring network structure with NetworkX. *SciPy Conference*.
