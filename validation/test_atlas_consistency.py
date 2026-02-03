#!/usr/bin/env python3
"""
Atlas Consistency Validation.

Compares atlas files between DWI and fMRI pipelines to ensure
label mappings are consistent for cross-modal analysis.

Usage:
    python test_atlas_consistency.py [--dwi-atlas PATH] [--fmri-atlas PATH]

If paths are not provided, uses synthetic test data.
"""

import sys
from pathlib import Path
import argparse
import numpy as np

# Check for nibabel availability
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


def create_synthetic_atlas(n_rois: int = 246, shape: tuple = (91, 109, 91)) -> np.ndarray:
    """Create a synthetic atlas for testing.

    Creates a 3D volume with ROI labels distributed across the volume.

    Parameters
    ----------
    n_rois : int
        Number of ROIs (default 246 for Brainnetome)
    shape : tuple
        Volume dimensions

    Returns
    -------
    np.ndarray
        3D array with integer ROI labels
    """
    atlas = np.zeros(shape, dtype=np.int32)

    # Create spherical ROIs distributed through the volume
    np.random.seed(42)  # For reproducibility

    # Generate ROI centers
    margin = 5
    centers = []
    for _ in range(n_rois):
        x = np.random.randint(margin, shape[0] - margin)
        y = np.random.randint(margin, shape[1] - margin)
        z = np.random.randint(margin, shape[2] - margin)
        centers.append((x, y, z))

    # Create coordinate grids
    xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]

    # Assign voxels to nearest center (Voronoi-like)
    for roi_idx, (cx, cy, cz) in enumerate(centers):
        # Create a small sphere for each ROI
        radius = 3
        mask = ((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2) <= radius**2
        atlas[mask] = roi_idx + 1  # ROI labels start at 1

    return atlas


def compare_atlas_labels(atlas1: np.ndarray, atlas2: np.ndarray) -> dict:
    """Compare label sets between two atlases.

    Parameters
    ----------
    atlas1 : np.ndarray
        First atlas volume
    atlas2 : np.ndarray
        Second atlas volume

    Returns
    -------
    dict
        Comparison results including:
        - labels_atlas1: unique labels in atlas 1
        - labels_atlas2: unique labels in atlas 2
        - common_labels: labels present in both
        - only_in_atlas1: labels only in atlas 1
        - only_in_atlas2: labels only in atlas 2
        - n_rois_match: whether ROI counts match (excluding background)
    """
    labels1 = set(np.unique(atlas1)) - {0}  # Exclude background
    labels2 = set(np.unique(atlas2)) - {0}

    return {
        'n_rois_atlas1': len(labels1),
        'n_rois_atlas2': len(labels2),
        'labels_atlas1': sorted(labels1),
        'labels_atlas2': sorted(labels2),
        'common_labels': sorted(labels1 & labels2),
        'only_in_atlas1': sorted(labels1 - labels2),
        'only_in_atlas2': sorted(labels2 - labels1),
        'n_rois_match': len(labels1) == len(labels2),
        'labels_identical': labels1 == labels2,
    }


def compare_atlas_volumes(atlas1: np.ndarray, atlas2: np.ndarray) -> dict:
    """Compare voxel counts per ROI between atlases.

    Parameters
    ----------
    atlas1 : np.ndarray
        First atlas volume (must match shape of atlas2)
    atlas2 : np.ndarray
        Second atlas volume

    Returns
    -------
    dict
        Volume comparison results
    """
    if atlas1.shape != atlas2.shape:
        return {
            'same_shape': False,
            'shape_atlas1': atlas1.shape,
            'shape_atlas2': atlas2.shape,
            'error': 'Atlases have different dimensions - cannot compare volumes directly'
        }

    # Get common labels
    labels1 = set(np.unique(atlas1)) - {0}
    labels2 = set(np.unique(atlas2)) - {0}
    common = labels1 & labels2

    # Count voxels per ROI
    volume_comparison = {}
    for label in sorted(common):
        count1 = np.sum(atlas1 == label)
        count2 = np.sum(atlas2 == label)
        volume_comparison[int(label)] = {
            'atlas1_voxels': int(count1),
            'atlas2_voxels': int(count2),
            'difference': int(count2 - count1),
            'ratio': float(count2 / count1) if count1 > 0 else float('inf')
        }

    # Summary statistics
    diffs = [v['difference'] for v in volume_comparison.values()]
    ratios = [v['ratio'] for v in volume_comparison.values() if v['ratio'] != float('inf')]

    return {
        'same_shape': True,
        'shape': atlas1.shape,
        'n_common_rois': len(common),
        'volume_comparison': volume_comparison,
        'mean_difference': float(np.mean(diffs)) if diffs else 0,
        'max_difference': int(np.max(np.abs(diffs))) if diffs else 0,
        'mean_ratio': float(np.mean(ratios)) if ratios else 1.0,
        'volumes_identical': all(d == 0 for d in diffs),
    }


def compute_overlap_matrix(atlas1: np.ndarray, atlas2: np.ndarray) -> np.ndarray:
    """Compute overlap matrix between atlases.

    For each ROI in atlas1, computes overlap with all ROIs in atlas2.

    Parameters
    ----------
    atlas1 : np.ndarray
        First atlas volume
    atlas2 : np.ndarray
        Second atlas volume (same shape)

    Returns
    -------
    np.ndarray
        Overlap matrix of shape (n_rois_atlas1, n_rois_atlas2)
        Values represent Dice coefficient between ROIs
    """
    if atlas1.shape != atlas2.shape:
        raise ValueError("Atlases must have same shape")

    labels1 = sorted(set(np.unique(atlas1)) - {0})
    labels2 = sorted(set(np.unique(atlas2)) - {0})

    overlap = np.zeros((len(labels1), len(labels2)))

    for i, l1 in enumerate(labels1):
        mask1 = atlas1 == l1
        for j, l2 in enumerate(labels2):
            mask2 = atlas2 == l2
            intersection = np.sum(mask1 & mask2)
            union = np.sum(mask1 | mask2)
            if union > 0:
                overlap[i, j] = 2 * intersection / (np.sum(mask1) + np.sum(mask2))  # Dice

    return overlap, labels1, labels2


def test_synthetic_comparison():
    """Test atlas comparison with synthetic data."""
    print("\n=== Testing Synthetic Atlas Comparison ===")

    # Create two slightly different atlases
    atlas1 = create_synthetic_atlas(n_rois=50, shape=(50, 50, 50))

    # Create second atlas with small perturbation
    atlas2 = atlas1.copy()
    # Add a few extra voxels to some ROIs
    atlas2[10:12, 10:12, 10:12] = 5

    # Compare labels
    label_results = compare_atlas_labels(atlas1, atlas2)
    print(f"  Atlas 1 ROIs: {label_results['n_rois_atlas1']}")
    print(f"  Atlas 2 ROIs: {label_results['n_rois_atlas2']}")
    print(f"  Labels identical: {label_results['labels_identical']}")

    # Compare volumes
    volume_results = compare_atlas_volumes(atlas1, atlas2)
    print(f"  Same shape: {volume_results['same_shape']}")
    print(f"  Volumes identical: {volume_results['volumes_identical']}")
    print(f"  Mean volume difference: {volume_results['mean_difference']:.1f} voxels")

    print("  Synthetic Comparison: PASSED")
    return True


def test_identical_atlases():
    """Test that identical atlases are detected correctly."""
    print("\n=== Testing Identical Atlas Detection ===")

    atlas1 = create_synthetic_atlas(n_rois=100)
    atlas2 = atlas1.copy()

    label_results = compare_atlas_labels(atlas1, atlas2)
    volume_results = compare_atlas_volumes(atlas1, atlas2)

    assert label_results['labels_identical'], "Labels should be identical"
    assert volume_results['volumes_identical'], "Volumes should be identical"

    print("  Labels identical: True")
    print("  Volumes identical: True")
    print("  Identical Atlas Detection: PASSED")
    return True


def test_different_roi_counts():
    """Test detection of different ROI counts."""
    print("\n=== Testing Different ROI Count Detection ===")

    atlas1 = create_synthetic_atlas(n_rois=100)
    atlas2 = create_synthetic_atlas(n_rois=84)  # Like DK atlas

    label_results = compare_atlas_labels(atlas1, atlas2)

    assert not label_results['n_rois_match'], "Should detect ROI count mismatch"
    print(f"  Atlas 1: {label_results['n_rois_atlas1']} ROIs")
    print(f"  Atlas 2: {label_results['n_rois_atlas2']} ROIs")
    print(f"  Match: {label_results['n_rois_match']}")
    print("  Different ROI Count Detection: PASSED")
    return True


def test_overlap_computation():
    """Test overlap matrix computation."""
    print("\n=== Testing Overlap Matrix Computation ===")

    atlas1 = create_synthetic_atlas(n_rois=20, shape=(30, 30, 30))
    atlas2 = atlas1.copy()

    overlap, labels1, labels2 = compute_overlap_matrix(atlas1, atlas2)

    # Identical atlases should have perfect diagonal overlap
    diagonal = np.diag(overlap)
    off_diagonal = overlap - np.diag(diagonal)

    print(f"  Overlap matrix shape: {overlap.shape}")
    print(f"  Mean diagonal Dice: {np.mean(diagonal):.3f}")
    print(f"  Max off-diagonal: {np.max(off_diagonal):.3f}")

    assert np.allclose(diagonal, 1.0), "Diagonal should be 1.0 for identical atlases"
    print("  Overlap Matrix Computation: PASSED")
    return True


def compare_real_atlases(dwi_atlas_path: str, fmri_atlas_path: str):
    """Compare real atlas files from both pipelines.

    Parameters
    ----------
    dwi_atlas_path : str
        Path to DWI pipeline Brainnetome atlas
    fmri_atlas_path : str
        Path to fMRI pipeline Brainnetome atlas
    """
    if not NIBABEL_AVAILABLE:
        print("ERROR: nibabel required to load NIfTI files")
        return False

    print("\n=== Comparing Real Atlas Files ===")
    print(f"  DWI atlas:  {dwi_atlas_path}")
    print(f"  fMRI atlas: {fmri_atlas_path}")

    # Load atlases
    dwi_img = nib.load(dwi_atlas_path)
    fmri_img = nib.load(fmri_atlas_path)

    dwi_data = np.asarray(dwi_img.dataobj)
    fmri_data = np.asarray(fmri_img.dataobj)

    # Compare headers
    print("\n  Header Information:")
    print(f"    DWI shape:  {dwi_data.shape}")
    print(f"    fMRI shape: {fmri_data.shape}")
    print(f"    DWI affine:\n{dwi_img.affine}")
    print(f"    fMRI affine:\n{fmri_img.affine}")

    # Compare labels
    print("\n  Label Comparison:")
    label_results = compare_atlas_labels(dwi_data, fmri_data)
    print(f"    DWI ROIs:  {label_results['n_rois_atlas1']}")
    print(f"    fMRI ROIs: {label_results['n_rois_atlas2']}")
    print(f"    Labels match: {label_results['labels_identical']}")

    if label_results['only_in_atlas1']:
        print(f"    Only in DWI:  {label_results['only_in_atlas1'][:10]}...")
    if label_results['only_in_atlas2']:
        print(f"    Only in fMRI: {label_results['only_in_atlas2'][:10]}...")

    # Compare volumes if same shape
    if dwi_data.shape == fmri_data.shape:
        print("\n  Volume Comparison:")
        volume_results = compare_atlas_volumes(dwi_data, fmri_data)
        print(f"    Volumes identical: {volume_results['volumes_identical']}")
        print(f"    Mean voxel difference: {volume_results['mean_difference']:.1f}")
        print(f"    Max voxel difference: {volume_results['max_difference']}")
    else:
        print("\n  Volume Comparison: SKIPPED (different shapes)")
        print("  Note: Atlases in different template spaces - direct comparison not meaningful")

    return True


def run_all_tests():
    """Run all atlas consistency tests."""
    print("=" * 60)
    print("ATLAS CONSISTENCY VALIDATION TESTS")
    print("=" * 60)

    tests = [
        ("Synthetic Comparison", test_synthetic_comparison),
        ("Identical Detection", test_identical_atlases),
        ("Different ROI Count", test_different_roi_counts),
        ("Overlap Computation", test_overlap_computation),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")

    n_passed = sum(1 for _, s in results if s == "PASSED")
    print(f"\n  Total: {n_passed}/{len(tests)} passed")

    return n_passed == len(tests)


def main():
    parser = argparse.ArgumentParser(
        description="Compare atlas files between DWI and fMRI pipelines"
    )
    parser.add_argument(
        "--dwi-atlas",
        type=str,
        help="Path to DWI pipeline Brainnetome atlas"
    )
    parser.add_argument(
        "--fmri-atlas",
        type=str,
        help="Path to fMRI pipeline Brainnetome atlas"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run synthetic tests only (no real files)"
    )

    args = parser.parse_args()

    # Run synthetic tests
    success = run_all_tests()

    # Compare real atlases if provided
    if args.dwi_atlas and args.fmri_atlas:
        if Path(args.dwi_atlas).exists() and Path(args.fmri_atlas).exists():
            compare_real_atlases(args.dwi_atlas, args.fmri_atlas)
        else:
            print("\nWARNING: One or both atlas files not found")
            if not Path(args.dwi_atlas).exists():
                print(f"  Missing: {args.dwi_atlas}")
            if not Path(args.fmri_atlas).exists():
                print(f"  Missing: {args.fmri_atlas}")
    elif not args.test_only:
        print("\n" + "-" * 60)
        print("To compare real atlas files, use:")
        print("  python test_atlas_consistency.py \\")
        print("    --dwi-atlas /path/to/dwi/Brainnetome.nii.gz \\")
        print("    --fmri-atlas /path/to/fmri/rBrainnetome.nii.gz")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
