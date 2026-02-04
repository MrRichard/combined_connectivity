"""
Connectivity matrix I/O with standardized formats.

Handles loading and saving connectivity matrices with consistent format:
- CSV with row and column headers (ROI labels)
- Symmetric matrices with zero diagonal
- Support for legacy headerless CSV (from tck2connectome)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_connectivity_matrix(
    filepath: Union[str, Path],
    labels_file: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load a connectivity matrix from file.

    Supports multiple formats:
    - CSV with headers (pandas DataFrame style)
    - CSV without headers (legacy tck2connectome output)
    - NumPy .npy files

    Args:
        filepath: Path to connectivity matrix file
        labels_file: Optional path to labels file (for headerless CSV)
            If not provided and CSV has no headers, generic labels are generated.

    Returns:
        Tuple of (matrix as np.ndarray, list of ROI labels)

    Example:
        >>> matrix, labels = load_connectivity_matrix("connectome.csv")
        >>> print(f"Matrix shape: {matrix.shape}")
        >>> print(f"First 5 labels: {labels[:5]}")
    """
    filepath = Path(filepath)

    if filepath.suffix == '.npy':
        return _load_npy(filepath, labels_file)
    elif filepath.suffix == '.csv':
        return _load_csv(filepath, labels_file)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def _load_npy(
    filepath: Path,
    labels_file: Optional[Path],
) -> Tuple[np.ndarray, List[str]]:
    """Load matrix from NumPy file."""
    matrix = np.load(filepath)

    if labels_file:
        labels = _load_labels(labels_file)
    else:
        # Try to find labels file with same name
        labels_path = filepath.with_suffix('.labels.txt')
        if labels_path.exists():
            labels = _load_labels(labels_path)
        else:
            labels = [f"ROI_{i:03d}" for i in range(matrix.shape[0])]
            logger.warning(f"No labels file found, using generic labels")

    return matrix, labels


def _load_csv(
    filepath: Path,
    labels_file: Optional[Path],
) -> Tuple[np.ndarray, List[str]]:
    """Load matrix from CSV file."""
    # First, try loading without headers (tck2connectome style - pure numeric)
    # This should be tried first because it's the most restrictive
    try:
        matrix = np.loadtxt(filepath, delimiter=',')
        # If we get here, it's a headerless numeric CSV
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            if labels_file:
                labels = _load_labels(labels_file)
            else:
                labels = [f"ROI_{i:03d}" for i in range(matrix.shape[0])]
                logger.info(f"Loaded headerless CSV: {matrix.shape}")
            return matrix, labels
    except Exception as e:
        logger.debug(f"Not a headerless numeric CSV: {e}")

    # Try loading with headers (pandas style with index column)
    try:
        df = pd.read_csv(filepath, index_col=0)
        # Check if we successfully loaded a square matrix with labels
        if df.shape[0] == df.shape[1] and df.shape[0] > 0:
            # Check if column names look like labels (not just integers)
            first_col = str(df.columns[0])
            if not first_col.replace('_', '').replace('-', '').isdigit():
                # Has string headers
                labels = list(df.columns)
                matrix = df.values.astype(float)
                logger.info(f"Loaded CSV with headers: {matrix.shape}, {len(labels)} labels")
                return matrix, labels
    except Exception as e:
        logger.debug(f"Could not load as pandas DataFrame with index: {e}")

    # Try loading without index column (but still with headers)
    try:
        df = pd.read_csv(filepath)
        # Check if first column looks like labels
        first_col_name = str(df.columns[0])
        if first_col_name == '' or first_col_name == 'Unnamed: 0':
            # First column is row labels, drop it
            df = df.iloc[:, 1:]
        # Check if it's square
        if df.shape[0] == df.shape[1] and df.shape[0] > 0:
            labels = list(df.columns)
            matrix = df.values.astype(float)
            logger.info(f"Loaded CSV (alternative format): {matrix.shape}")
            return matrix, labels
    except Exception as e:
        logger.debug(f"Could not load as pandas DataFrame: {e}")

    raise ValueError(f"Could not load CSV file - unrecognized format")


def _load_labels(filepath: Union[str, Path]) -> List[str]:
    """Load ROI labels from file."""
    filepath = Path(filepath)
    labels = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle tab-separated format (index\tlabel)
                parts = line.split('\t')
                if len(parts) >= 2:
                    labels.append(parts[1])
                else:
                    labels.append(parts[0])

    return labels


def save_connectivity_matrix(
    matrix: np.ndarray,
    labels: List[str],
    filepath: Union[str, Path],
    modality: Optional[str] = None,
    atlas: Optional[str] = None,
) -> None:
    """
    Save connectivity matrix in standardized format.

    Saves as CSV with row and column headers for self-documentation.
    Also saves a companion .labels.txt file.

    Args:
        matrix: Connectivity matrix (n_rois x n_rois)
        labels: List of ROI labels
        filepath: Output file path
        modality: Optional modality tag ('dwi' or 'fmri') for metadata
        atlas: Optional atlas name for metadata

    Example:
        >>> save_connectivity_matrix(matrix, labels, "output.csv", modality="dwi")
    """
    filepath = Path(filepath)

    if len(labels) != matrix.shape[0]:
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match "
            f"matrix dimension ({matrix.shape[0]})"
        )

    # Create DataFrame with labels
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    # Save CSV
    df.to_csv(filepath)
    logger.info(f"Saved connectivity matrix: {filepath}")

    # Save companion labels file
    labels_file = filepath.with_suffix('.labels.txt')
    with open(labels_file, 'w') as f:
        f.write(f"# ROI labels for {filepath.name}\n")
        if modality:
            f.write(f"# Modality: {modality}\n")
        if atlas:
            f.write(f"# Atlas: {atlas}\n")
        f.write("# Format: index<tab>label\n")
        for i, label in enumerate(labels):
            f.write(f"{i}\t{label}\n")

    logger.info(f"Saved labels file: {labels_file}")


def convert_legacy_csv(
    input_csv: Union[str, Path],
    labels: Union[List[str], str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    modality: Optional[str] = None,
    atlas: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert legacy headerless CSV to standardized format.

    Args:
        input_csv: Path to headerless CSV (e.g., from tck2connectome)
        labels: List of labels, or path to labels file
        output_csv: Output path (if None, overwrites input)
        modality: Optional modality tag for metadata
        atlas: Optional atlas name for metadata

    Returns:
        Tuple of (matrix, labels)

    Example:
        >>> # Convert tck2connectome output
        >>> matrix, labels = convert_legacy_csv(
        ...     "connectome_counts.csv",
        ...     "brainnetome_labels.txt",
        ...     "connectome_counts_harmonized.csv",
        ...     modality="dwi",
        ...     atlas="brainnetome"
        ... )
    """
    input_csv = Path(input_csv)

    # Load labels
    if isinstance(labels, (str, Path)):
        label_list = _load_labels(labels)
    else:
        label_list = labels

    # Load matrix
    matrix = np.loadtxt(input_csv, delimiter=',')

    if len(label_list) != matrix.shape[0]:
        raise ValueError(
            f"Number of labels ({len(label_list)}) doesn't match "
            f"matrix dimension ({matrix.shape[0]})"
        )

    # Determine output path
    if output_csv is None:
        output_csv = input_csv
    else:
        output_csv = Path(output_csv)

    # Save in standardized format
    save_connectivity_matrix(matrix, label_list, output_csv, modality, atlas)

    return matrix, label_list


def validate_matrix(
    matrix: np.ndarray,
    check_symmetric: bool = True,
    check_diagonal: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate a connectivity matrix.

    Args:
        matrix: Connectivity matrix to validate
        check_symmetric: Check if matrix is symmetric
        check_diagonal: Check if diagonal is zero

    Returns:
        Tuple of (is_valid, list of warning messages)
    """
    warnings = []
    is_valid = True

    # Check shape
    if matrix.ndim != 2:
        warnings.append(f"Matrix is not 2D: {matrix.ndim}D")
        is_valid = False
        return is_valid, warnings

    if matrix.shape[0] != matrix.shape[1]:
        warnings.append(f"Matrix is not square: {matrix.shape}")
        is_valid = False
        return is_valid, warnings

    # Check for NaN/Inf
    if np.any(np.isnan(matrix)):
        n_nan = np.sum(np.isnan(matrix))
        warnings.append(f"Matrix contains {n_nan} NaN values")
        is_valid = False

    if np.any(np.isinf(matrix)):
        n_inf = np.sum(np.isinf(matrix))
        warnings.append(f"Matrix contains {n_inf} Inf values")
        is_valid = False

    # Check symmetry
    if check_symmetric:
        if not np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-8):
            max_diff = np.max(np.abs(matrix - matrix.T))
            warnings.append(f"Matrix is not symmetric (max diff: {max_diff:.2e})")

    # Check diagonal
    if check_diagonal:
        if not np.allclose(np.diag(matrix), 0, atol=1e-8):
            max_diag = np.max(np.abs(np.diag(matrix)))
            warnings.append(f"Diagonal is not zero (max: {max_diag:.2e})")

    return is_valid, warnings


def get_matrix_stats(matrix: np.ndarray) -> dict:
    """
    Get basic statistics for a connectivity matrix.

    Args:
        matrix: Connectivity matrix

    Returns:
        Dictionary with statistics
    """
    # Get upper triangle (excluding diagonal) for undirected networks
    triu_indices = np.triu_indices_from(matrix, k=1)
    values = matrix[triu_indices]

    # Compute statistics
    stats = {
        'n_nodes': matrix.shape[0],
        'n_possible_edges': len(values),
        'n_nonzero_edges': np.sum(values != 0),
        'density': np.sum(values != 0) / len(values) if len(values) > 0 else 0,
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
    }

    # Add percentiles
    for p in [25, 75, 90, 95, 99]:
        stats[f'percentile_{p}'] = float(np.percentile(values, p))

    return stats
