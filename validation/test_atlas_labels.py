#!/usr/bin/env python3
"""
Test atlas labels functionality.

Validates the canonical atlas label definitions and utilities.
"""

import sys
from pathlib import Path
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "connectivity_shared" / "src"))

from connectivity_shared import (
    get_atlas_labels,
    get_atlas_info,
    save_labels_file,
    load_labels_file,
    verify_labels_consistency,
    generate_canonical_labels_file,
    BRAINNETOME_LABELS,
    FREESURFER_DK_LABELS,
)


def test_brainnetome_labels():
    """Test Brainnetome atlas labels."""
    print("\n=== Testing Brainnetome Labels ===")

    labels = get_atlas_labels('brainnetome')

    assert len(labels) == 246, f"Expected 246 ROIs, got {len(labels)}"
    print(f"  ROI count: {len(labels)}")

    # Check first and last labels
    assert labels[0] == "SFG_L_7_1", f"First label mismatch: {labels[0]}"
    assert labels[-1] == "Tha_R_8_8", f"Last label mismatch: {labels[-1]}"
    print(f"  First label: {labels[0]}")
    print(f"  Last label: {labels[-1]}")

    # Check bilateral structure (L/R alternating)
    n_left = sum(1 for l in labels if '_L_' in l)
    n_right = sum(1 for l in labels if '_R_' in l)
    assert n_left == n_right, f"Hemisphere imbalance: L={n_left}, R={n_right}"
    print(f"  Left/Right balance: {n_left}/{n_right}")

    # Check canonical constant matches function
    assert labels == BRAINNETOME_LABELS, "Constant mismatch with function"
    print("  Constant matches function output")

    print("  Brainnetome Labels: PASSED")
    return True


def test_freesurfer_dk_labels():
    """Test FreeSurfer Desikan-Killiany labels."""
    print("\n=== Testing FreeSurfer DK Labels ===")

    labels = get_atlas_labels('freesurfer_dk')

    assert len(labels) == 84, f"Expected 84 ROIs, got {len(labels)}"
    print(f"  ROI count: {len(labels)}")

    # Check structure
    cortical = [l for l in labels if l.startswith('ctx-')]
    subcortical = [l for l in labels if not l.startswith('ctx-')]
    print(f"  Cortical regions: {len(cortical)}")
    print(f"  Subcortical regions: {len(subcortical)}")

    assert len(cortical) == 68, f"Expected 68 cortical, got {len(cortical)}"
    assert len(subcortical) == 16, f"Expected 16 subcortical, got {len(subcortical)}"

    # Check canonical constant
    assert labels == FREESURFER_DK_LABELS, "Constant mismatch"

    print("  FreeSurfer DK Labels: PASSED")
    return True


def test_atlas_info():
    """Test atlas info retrieval."""
    print("\n=== Testing Atlas Info ===")

    info = get_atlas_info('brainnetome')
    assert info['n_rois'] == 246
    assert info['includes_subcortical'] is True
    print(f"  Brainnetome: {info['n_rois']} ROIs, subcortical={info['includes_subcortical']}")

    info = get_atlas_info('freesurfer_dk')
    assert info['n_rois'] == 84
    print(f"  FreeSurfer DK: {info['n_rois']} ROIs")

    # Test aliases
    info1 = get_atlas_info('bna')
    info2 = get_atlas_info('brainnetome')
    assert info1 == info2, "Alias 'bna' should work"
    print("  Aliases work correctly")

    print("  Atlas Info: PASSED")
    return True


def test_save_load_labels():
    """Test saving and loading labels files."""
    print("\n=== Testing Save/Load Labels ===")

    labels = get_atlas_labels('brainnetome')

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_labels.txt"

        # Save
        save_labels_file(labels, output_path, atlas_name="brainnetome")
        assert output_path.exists(), "Labels file not created"
        print(f"  Saved {len(labels)} labels")

        # Load
        loaded = load_labels_file(output_path)
        assert loaded == labels, "Loaded labels don't match original"
        print(f"  Loaded {len(loaded)} labels")

    print("  Save/Load Labels: PASSED")
    return True


def test_verify_consistency():
    """Test label consistency verification."""
    print("\n=== Testing Label Consistency Verification ===")

    labels1 = get_atlas_labels('brainnetome')
    labels2 = labels1.copy()

    # Identical labels should be consistent
    consistent, details = verify_labels_consistency(labels1, labels2)
    assert consistent, "Identical labels should be consistent"
    assert details['order_match'], "Order should match"
    print("  Identical labels: consistent=True")

    # Modified labels should be detected
    labels2[0] = "MODIFIED_LABEL"
    consistent, details = verify_labels_consistency(labels1, labels2)
    assert not consistent, "Modified labels should be inconsistent"
    assert len(details['only_in_1']) == 1
    assert len(details['only_in_2']) == 1
    print("  Modified labels: consistent=False, detected differences")

    # Different count should be detected
    labels3 = labels1[:100]
    consistent, details = verify_labels_consistency(labels1, labels3)
    assert not consistent, "Different counts should be inconsistent"
    assert not details['count_match']
    print("  Different count: consistent=False, count_match=False")

    print("  Label Consistency Verification: PASSED")
    return True


def test_generate_canonical_file():
    """Test canonical labels file generation."""
    print("\n=== Testing Canonical File Generation ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Generate Brainnetome
        path = generate_canonical_labels_file('brainnetome', output_dir)
        assert path.exists(), "Brainnetome labels file not created"
        assert path.name == "brainnetome_labels.txt"
        print(f"  Generated: {path.name}")

        # Verify content
        loaded = load_labels_file(path)
        assert len(loaded) == 246
        print(f"  Verified: 246 labels")

        # Generate DK
        path = generate_canonical_labels_file('dk', output_dir)
        assert path.exists(), "DK labels file not created"
        assert path.name == "freesurfer_dk_labels.txt"
        print(f"  Generated: {path.name}")

    print("  Canonical File Generation: PASSED")
    return True


def test_name_aliases():
    """Test that different atlas name variants work."""
    print("\n=== Testing Name Aliases ===")

    # Brainnetome variants
    for name in ['brainnetome', 'bna', 'brainnetome246']:
        labels = get_atlas_labels(name)
        assert len(labels) == 246, f"'{name}' should return 246 labels"
    print("  Brainnetome aliases: OK")

    # DK variants
    for name in ['freesurfer_dk', 'dk', 'desikan_killiany', 'desikan-killiany']:
        labels = get_atlas_labels(name)
        assert len(labels) == 84, f"'{name}' should return 84 labels"
    print("  DK aliases: OK")

    print("  Name Aliases: PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("ATLAS LABELS VALIDATION TESTS")
    print("=" * 60)

    tests = [
        ("Brainnetome Labels", test_brainnetome_labels),
        ("FreeSurfer DK Labels", test_freesurfer_dk_labels),
        ("Atlas Info", test_atlas_info),
        ("Save/Load Labels", test_save_load_labels),
        ("Consistency Verification", test_verify_consistency),
        ("Canonical File Generation", test_generate_canonical_file),
        ("Name Aliases", test_name_aliases),
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


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
