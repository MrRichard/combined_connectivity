#!/usr/bin/env python3
"""
Generate synthetic test data for validation without real neuroimaging data.

Creates:
- Synthetic connectivity matrices with known properties
- Mock FreeSurfer directory structures
- Test label files
"""

import numpy as np
from pathlib import Path
import json


def generate_synthetic_connectivity_matrix(
    n_rois: int = 100,
    density: float = 0.3,
    seed: int = 42,
    symmetric: bool = True,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """
    Generate a random symmetric connectivity matrix.

    Args:
        n_rois: Number of ROIs (matrix size)
        density: Proportion of non-zero connections (0 to 1)
        seed: Random seed for reproducibility
        symmetric: Whether to make matrix symmetric
        zero_diagonal: Whether to set diagonal to zero

    Returns:
        Connectivity matrix as numpy array
    """
    np.random.seed(seed)

    # Generate random values
    matrix = np.random.rand(n_rois, n_rois)

    # Apply density threshold
    threshold = 1 - density
    matrix[matrix < threshold] = 0

    # Make symmetric
    if symmetric:
        matrix = (matrix + matrix.T) / 2

    # Zero diagonal
    if zero_diagonal:
        np.fill_diagonal(matrix, 0)

    return matrix


def generate_synthetic_timeseries(
    n_rois: int = 100,
    n_timepoints: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic ROI time series with known correlation structure.

    Creates time series where nearby ROIs have higher correlation.

    Args:
        n_rois: Number of ROIs
        n_timepoints: Number of time points
        seed: Random seed

    Returns:
        Time series matrix (n_timepoints x n_rois)
    """
    np.random.seed(seed)

    # Start with random noise
    timeseries = np.random.randn(n_timepoints, n_rois)

    # Add some shared signal to create correlations
    n_components = 10
    components = np.random.randn(n_timepoints, n_components)

    for i in range(n_rois):
        # Each ROI loads on different components
        loadings = np.random.randn(n_components) * 0.5
        timeseries[:, i] += components @ loadings

    return timeseries


def generate_labels(
    n_rois: int = 100,
    prefix: str = "ROI",
) -> list:
    """Generate ROI labels."""
    return [f"{prefix}_{i:03d}" for i in range(n_rois)]


def generate_mock_freesurfer_structure(
    output_dir: Path,
    subject_name: str = "test_subject",
    version: str = "FreeSurfer7",
    include_destrieux: bool = True,
) -> Path:
    """
    Create directory structure mimicking FreeSurfer output.

    Note: Creates placeholder files, not actual MGZ data.

    Args:
        output_dir: Base directory
        subject_name: Subject name for nested structure
        version: FreeSurfer version directory name
        include_destrieux: Whether to include Destrieux atlas file

    Returns:
        Path to FreeSurfer subject directory
    """
    output_dir = Path(output_dir)

    if version == "freesurfer8.0":
        fs_dir = output_dir / version / subject_name
    else:
        fs_dir = output_dir / version

    # Create directory structure
    mri_dir = fs_dir / "mri"
    surf_dir = fs_dir / "surf"
    label_dir = fs_dir / "label"
    stats_dir = fs_dir / "stats"

    for d in [mri_dir, surf_dir, label_dir, stats_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create placeholder files
    placeholder_files = [
        mri_dir / "aparc+aseg.mgz",
        mri_dir / "aparc.DKTatlas+aseg.mgz",
        mri_dir / "brain.mgz",
        mri_dir / "orig.mgz",
        mri_dir / "wm.mgz",
        mri_dir / "aseg.mgz",
    ]

    if include_destrieux:
        placeholder_files.append(mri_dir / "aparc.a2009s+aseg.mgz")

    for f in placeholder_files:
        f.touch()

    return fs_dir


def generate_test_dataset(
    output_dir: Path,
    n_subjects: int = 3,
    n_rois: int = 100,
) -> dict:
    """
    Generate a complete test dataset.

    Args:
        output_dir: Base output directory
        n_subjects: Number of test subjects
        n_rois: Number of ROIs per subject

    Returns:
        Dictionary with paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = {
        'subjects': [],
        'labels_file': None,
        'config': {},
    }

    # Generate shared labels
    labels = generate_labels(n_rois)
    labels_file = output_dir / "test_labels.txt"
    with open(labels_file, 'w') as f:
        f.write("# Test ROI labels\n")
        for i, label in enumerate(labels):
            f.write(f"{i}\t{label}\n")
    dataset['labels_file'] = str(labels_file)

    # Generate per-subject data
    for i in range(n_subjects):
        subject_id = f"sub-{i+1:02d}"
        subject_dir = output_dir / subject_id
        subject_dir.mkdir(exist_ok=True)

        subject_data = {'id': subject_id, 'path': str(subject_dir)}

        # Generate connectivity matrix
        matrix = generate_synthetic_connectivity_matrix(
            n_rois=n_rois,
            density=0.3,
            seed=42 + i,
        )
        matrix_file = subject_dir / f"{subject_id}_connectivity.csv"
        np.savetxt(matrix_file, matrix, delimiter=',')
        subject_data['connectivity_matrix'] = str(matrix_file)

        # Generate FreeSurfer structure
        fs_dir = generate_mock_freesurfer_structure(
            subject_dir,
            subject_name=subject_id,
            version="FreeSurfer7",
        )
        subject_data['freesurfer_path'] = str(fs_dir)

        dataset['subjects'].append(subject_data)

    # Save dataset manifest
    manifest_file = output_dir / "test_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated test dataset at: {output_dir}")
    print(f"  - {n_subjects} subjects")
    print(f"  - {n_rois} ROIs")
    print(f"  - Labels file: {labels_file}")
    print(f"  - Manifest: {manifest_file}")

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_data",
        help="Output directory",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=3,
        help="Number of test subjects",
    )
    parser.add_argument(
        "--n-rois",
        type=int,
        default=100,
        help="Number of ROIs",
    )

    args = parser.parse_args()

    generate_test_dataset(
        output_dir=Path(args.output_dir),
        n_subjects=args.n_subjects,
        n_rois=args.n_rois,
    )
