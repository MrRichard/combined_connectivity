#!/usr/bin/env python3
"""
Test DICOM-to-BIDS conversion module.

Uses synthetic JSON sidecars (no real DICOMs needed for classification tests).
"""

import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "connectivity_shared" / "src"))

from connectivity_shared.dicom_to_bids import DicomToBIDS, BIDSConversionResult


def _write_sidecar(tmp_dir: Path, name: str, content: dict) -> Path:
    """Write a synthetic JSON sidecar and matching empty NIfTI stub."""
    json_path = tmp_dir / f"{name}.json"
    with open(json_path, "w") as fh:
        json.dump(content, fh)
    # Create a zero-byte NIfTI stub so _find_nifti_for_json can locate it
    nii_path = tmp_dir / f"{name}.nii.gz"
    nii_path.touch()
    return json_path


# ------------------------------------------------------------------
# Classification tests
# ------------------------------------------------------------------

passed = 0
failed = 0
total = 0


def _report(name, ok, detail=""):
    global passed, failed, total
    total += 1
    if ok:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}  {detail}")


def test_classify_dwi_ap():
    """DWI with j- PE direction -> modality='dwi', pe='AP'."""
    print("\n--- test_classify_dwi_ap ---")
    converter = DicomToBIDS()
    with tempfile.TemporaryDirectory() as tmp:
        jp = _write_sidecar(Path(tmp), "dwi_ap", {
            "SeriesDescription": "DTI_AP_b1000",
            "PhaseEncodingDirection": "j-",
            "EchoTime": 0.089,
        })
        result = converter.classify_file(jp)
        _report("modality is dwi", result["modality"] == "dwi", f"got {result['modality']}")
        _report("suffix is dwi", result["suffix"] == "dwi", f"got {result['suffix']}")
        _report("pe_direction is AP", result["pe_direction"] == "AP", f"got {result['pe_direction']}")
        _report("pe_direction_raw is j-", result["pe_direction_raw"] == "j-", f"got {result['pe_direction_raw']}")


def test_classify_dwi_pa():
    """DWI with j PE direction -> modality='dwi', pe='PA'."""
    print("\n--- test_classify_dwi_pa ---")
    converter = DicomToBIDS()
    with tempfile.TemporaryDirectory() as tmp:
        jp = _write_sidecar(Path(tmp), "dwi_pa", {
            "SeriesDescription": "NODDI_PA",
            "PhaseEncodingDirection": "j",
            "EchoTime": 0.089,
        })
        result = converter.classify_file(jp)
        _report("modality is dwi", result["modality"] == "dwi", f"got {result['modality']}")
        _report("pe_direction is PA", result["pe_direction"] == "PA", f"got {result['pe_direction']}")


def test_classify_bold():
    """BOLD functional -> modality='func', suffix='bold'."""
    print("\n--- test_classify_bold ---")
    converter = DicomToBIDS()
    with tempfile.TemporaryDirectory() as tmp:
        jp = _write_sidecar(Path(tmp), "bold", {
            "SeriesDescription": "resting_state_BOLD",
            "PhaseEncodingDirection": "j-",
            "RepetitionTime": 2.0,
        })
        result = converter.classify_file(jp)
        _report("modality is func", result["modality"] == "func", f"got {result['modality']}")
        _report("suffix is bold", result["suffix"] == "bold", f"got {result['suffix']}")
        _report("is_sbref is False", result["is_sbref"] is False, f"got {result['is_sbref']}")


def test_classify_sbref():
    """SBRef filtering -> is_sbref=True."""
    print("\n--- test_classify_sbref ---")
    converter = DicomToBIDS()
    with tempfile.TemporaryDirectory() as tmp:
        jp = _write_sidecar(Path(tmp), "sbref", {
            "SeriesDescription": "resting_SBRef",
            "PhaseEncodingDirection": "j-",
        })
        result = converter.classify_file(jp)
        _report("is_sbref is True", result["is_sbref"] is True, f"got {result['is_sbref']}")


def test_classify_t1w():
    """T1w anatomical -> modality='anat', suffix='T1w'."""
    print("\n--- test_classify_t1w ---")
    converter = DicomToBIDS()
    with tempfile.TemporaryDirectory() as tmp:
        jp = _write_sidecar(Path(tmp), "t1w", {
            "SeriesDescription": "tfl3d116_sag_T1w",
        })
        result = converter.classify_file(jp)
        _report("modality is anat", result["modality"] == "anat", f"got {result['modality']}")
        _report("suffix is T1w", result["suffix"] == "T1w", f"got {result['suffix']}")


def test_classify_flair():
    """FLAIR -> modality='anat', suffix='FLAIR'."""
    print("\n--- test_classify_flair ---")
    converter = DicomToBIDS()
    with tempfile.TemporaryDirectory() as tmp:
        jp = _write_sidecar(Path(tmp), "flair", {
            "SeriesDescription": "spcir_sag_FLAIR",
        })
        result = converter.classify_file(jp)
        _report("modality is anat", result["modality"] == "anat", f"got {result['modality']}")
        _report("suffix is FLAIR", result["suffix"] == "FLAIR", f"got {result['suffix']}")


def test_classify_fieldmap_magnitude():
    """Fieldmap magnitude by echo time."""
    print("\n--- test_classify_fieldmap_magnitude ---")
    converter = DicomToBIDS()
    with tempfile.TemporaryDirectory() as tmp:
        jp = _write_sidecar(Path(tmp), "mag1", {
            "SeriesDescription": "GRE_FIELD_MAPPING",
            "ImageType": ["ORIGINAL", "PRIMARY", "MAGNITUDE"],
            "EchoTime": 0.00246,
        })
        result = converter.classify_file(jp)
        _report("modality is fmap", result["modality"] == "fmap", f"got {result['modality']}")
        _report("suffix is magnitude1", result["suffix"] == "magnitude1", f"got {result['suffix']}")

    with tempfile.TemporaryDirectory() as tmp:
        jp = _write_sidecar(Path(tmp), "mag2", {
            "SeriesDescription": "GRE_FIELD_MAPPING",
            "ImageType": ["ORIGINAL", "PRIMARY", "MAGNITUDE"],
            "EchoTime": 0.00738,
        })
        result = converter.classify_file(jp)
        _report("suffix is magnitude2", result["suffix"] == "magnitude2", f"got {result['suffix']}")


def test_classify_fieldmap_phasediff():
    """Fieldmap phase difference."""
    print("\n--- test_classify_fieldmap_phasediff ---")
    converter = DicomToBIDS()
    with tempfile.TemporaryDirectory() as tmp:
        jp = _write_sidecar(Path(tmp), "phasediff", {
            "SeriesDescription": "GRE_FIELD_MAPPING",
            "ImageType": ["ORIGINAL", "PRIMARY", "PHASE"],
            "EchoTime": 0.00738,
        })
        result = converter.classify_file(jp)
        _report("modality is fmap", result["modality"] == "fmap", f"got {result['modality']}")
        _report("suffix is phasediff", result["suffix"] == "phasediff", f"got {result['suffix']}")


def test_bids_directory_structure():
    """Verify correct BIDS directory organisation output."""
    print("\n--- test_bids_directory_structure ---")
    converter = DicomToBIDS()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        src.mkdir()
        out = tmp / "bids"

        # Write synthetic classified files
        _write_sidecar(src, "t1", {
            "SeriesDescription": "MPRAGE_sag",
        })
        _write_sidecar(src, "bold", {
            "SeriesDescription": "resting_BOLD",
            "PhaseEncodingDirection": "j-",
            "RepetitionTime": 2.0,
        })
        _write_sidecar(src, "dwi", {
            "SeriesDescription": "DTI_AP",
            "PhaseEncodingDirection": "j-",
        })

        # Classify
        json_files = sorted(src.glob("*.json"))
        classified = []
        for jp in json_files:
            info = converter.classify_file(jp)
            info["json_path"] = jp
            info["nii_path"] = converter._find_nifti_for_json(jp)
            classified.append(info)

        bids_files = converter._organize_bids(classified, out, "01", "1")

        # Check directory structure
        _report("anat dir exists", (out / "sub-01" / "ses-1" / "anat").is_dir())
        _report("func dir exists", (out / "sub-01" / "ses-1" / "func").is_dir())
        _report("dwi dir exists", (out / "sub-01" / "ses-1" / "dwi").is_dir())
        _report("has anat files", len(bids_files.get("anat", [])) > 0, f"got {len(bids_files.get('anat', []))}")
        _report("has func files", len(bids_files.get("func", [])) > 0, f"got {len(bids_files.get('func', []))}")
        _report("has dwi files", len(bids_files.get("dwi", [])) > 0, f"got {len(bids_files.get('dwi', []))}")


def test_dataset_description():
    """Verify dataset_description.json creation."""
    print("\n--- test_dataset_description ---")
    converter = DicomToBIDS()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        converter._write_dataset_description(tmp)
        desc_path = tmp / "dataset_description.json"
        _report("file exists", desc_path.exists())

        with open(desc_path) as fh:
            desc = json.load(fh)
        _report("has Name", "Name" in desc, f"keys: {list(desc.keys())}")
        _report("has BIDSVersion", "BIDSVersion" in desc)
        _report("BIDSVersion is string", isinstance(desc["BIDSVersion"], str))


def test_intended_for_metadata():
    """Verify IntendedFor field in fieldmap JSON."""
    print("\n--- test_intended_for_metadata ---")
    converter = DicomToBIDS()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Create a fake fieldmap JSON
        fmap_json = tmp / "fmap.json"
        with open(fmap_json, "w") as fh:
            json.dump({"EchoTime": 0.005}, fh)

        bold_path = Path("/fake/sub-01/ses-1/func/sub-01_ses-1_task-rest_bold.nii.gz")

        converter._add_intended_for(fmap_json, bold_path, "01", "1")

        with open(fmap_json) as fh:
            updated = json.load(fh)

        _report("IntendedFor present", "IntendedFor" in updated)
        expected = "ses-1/func/sub-01_ses-1_task-rest_bold.nii.gz"
        _report("IntendedFor correct", updated.get("IntendedFor") == expected,
                f"got {updated.get('IntendedFor')}")


def test_custom_series_patterns():
    """Verify config-driven pattern matching."""
    print("\n--- test_custom_series_patterns ---")
    custom_config = {
        "series_patterns": {
            "dwi": ["MY_CUSTOM_DWI"],
            "func": ["MY_CUSTOM_FUNC"],
            "anat_t1w": ["MY_CUSTOM_T1"],
            "anat_flair": ["MY_FLAIR"],
            "fmap": ["MY_FIELDMAP"],
        }
    }
    converter = DicomToBIDS(config=custom_config)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jp = _write_sidecar(tmp, "custom_dwi", {
            "SeriesDescription": "MY_CUSTOM_DWI_seq1",
            "PhaseEncodingDirection": "j-",
        })
        result = converter.classify_file(jp)
        _report("custom DWI detected", result["modality"] == "dwi", f"got {result['modality']}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jp = _write_sidecar(tmp, "custom_func", {
            "SeriesDescription": "MY_CUSTOM_FUNC_run1",
            "RepetitionTime": 2.0,
        })
        result = converter.classify_file(jp)
        _report("custom BOLD detected", result["modality"] == "func", f"got {result['modality']}")

    # Original patterns should NOT match with custom config
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jp = _write_sidecar(tmp, "old_pattern", {
            "SeriesDescription": "DTI_AP_b1000",
            "PhaseEncodingDirection": "j-",
        })
        result = converter.classify_file(jp)
        _report("default DTI NOT matched", result["modality"] != "dwi", f"got {result['modality']}")


def main():
    print("=" * 50)
    print("DICOM-TO-BIDS CONVERSION TESTS")
    print("=" * 50)

    test_classify_dwi_ap()
    test_classify_dwi_pa()
    test_classify_bold()
    test_classify_sbref()
    test_classify_t1w()
    test_classify_flair()
    test_classify_fieldmap_magnitude()
    test_classify_fieldmap_phasediff()
    test_bids_directory_structure()
    test_dataset_description()
    test_intended_for_metadata()
    test_custom_series_patterns()

    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
