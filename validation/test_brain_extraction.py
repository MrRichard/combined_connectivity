#!/usr/bin/env python3
"""
Test brain extraction module.

Tests that require FSL BET are skipped if it is not available on the system.
"""

import shutil
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "connectivity_shared" / "src"))

from connectivity_shared.brain_extraction import BrainExtractor, BrainExtractionResult

passed = 0
failed = 0
skipped = 0
total = 0

BET_AVAILABLE = shutil.which("bet") is not None


def _report(name, ok, detail=""):
    global passed, failed, total
    total += 1
    if ok:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}  {detail}")


def _skip(name, reason="FSL BET not available"):
    global skipped, total
    total += 1
    skipped += 1
    print(f"  SKIP: {name} ({reason})")


def test_brain_extractor_init():
    """Default parameters."""
    print("\n--- test_brain_extractor_init ---")
    extractor = BrainExtractor()
    _report("default bet_path", extractor.bet_path == "bet", f"got {extractor.bet_path}")
    _report("default frac_intensity", extractor.fractional_intensity == 0.5,
            f"got {extractor.fractional_intensity}")


def test_brain_extractor_custom_params():
    """Custom fractional intensity."""
    print("\n--- test_brain_extractor_custom_params ---")
    extractor = BrainExtractor(bet_path="/usr/local/bin/bet", fractional_intensity=0.3)
    _report("custom bet_path", extractor.bet_path == "/usr/local/bin/bet",
            f"got {extractor.bet_path}")
    _report("custom frac_intensity", extractor.fractional_intensity == 0.3,
            f"got {extractor.fractional_intensity}")


def test_bet_command_construction():
    """Verify correct bet command is built."""
    print("\n--- test_bet_command_construction ---")
    extractor = BrainExtractor(fractional_intensity=0.35)

    t1w = Path("/data/sub-01/anat/sub-01_T1w.nii.gz")
    output_base = Path("/out/sub-01_T1w_brain")

    # With robust
    cmd = extractor.build_command(t1w, output_base, robust=True)
    _report("cmd starts with bet", cmd[0] == "bet", f"got {cmd[0]}")
    _report("input path correct", cmd[1] == str(t1w))
    _report("output path correct", cmd[2] == str(output_base))
    _report("-m flag present", "-m" in cmd)
    _report("-f flag present", "-f" in cmd)
    f_idx = cmd.index("-f")
    _report("frac value correct", cmd[f_idx + 1] == "0.35", f"got {cmd[f_idx + 1]}")
    _report("-R flag present (robust)", "-R" in cmd)

    # Without robust
    cmd_nr = extractor.build_command(t1w, output_base, robust=False)
    _report("-R flag absent (no robust)", "-R" not in cmd_nr)


def test_bids_t1w_discovery():
    """Find T1w in BIDS structure."""
    print("\n--- test_bids_t1w_discovery ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Create BIDS structure with a T1w file
        anat_dir = tmp / "sub-01" / "ses-1" / "anat"
        anat_dir.mkdir(parents=True)
        t1w = anat_dir / "sub-01_ses-1_T1w.nii.gz"
        t1w.touch()

        extractor = BrainExtractor()

        # Test that extract_from_bids finds the file and raises RuntimeError
        # (because BET is probably not available) or succeeds
        try:
            result = extractor.extract_from_bids(tmp, "01", session_id="1")
            # If BET is available and a real NIfTI was provided, this would work
            _report("extract_from_bids returned result", isinstance(result, BrainExtractionResult))
        except RuntimeError as e:
            # Expected if BET is not available
            _report("raises RuntimeError when BET unavailable", "not found" in str(e).lower() or "failed" in str(e).lower(),
                    f"got: {e}")
        except FileNotFoundError:
            _report("T1w discovery failed", False, "FileNotFoundError - T1w should have been found")


def test_result_dataclass():
    """BrainExtractionResult fields."""
    print("\n--- test_result_dataclass ---")
    result = BrainExtractionResult(
        mask_path=Path("/out/mask.nii.gz"),
        brain_path=Path("/out/brain.nii.gz"),
        input_path=Path("/in/t1w.nii.gz"),
        method="fsl_bet",
        parameters={"fractional_intensity": 0.5, "robust": True},
    )
    _report("mask_path correct", result.mask_path == Path("/out/mask.nii.gz"))
    _report("brain_path correct", result.brain_path == Path("/out/brain.nii.gz"))
    _report("input_path correct", result.input_path == Path("/in/t1w.nii.gz"))
    _report("method correct", result.method == "fsl_bet")
    _report("parameters has frac_intensity", result.parameters.get("fractional_intensity") == 0.5)
    _report("parameters has robust", result.parameters.get("robust") is True)


def test_bet_not_available():
    """Graceful error when BET not found."""
    print("\n--- test_bet_not_available ---")
    extractor = BrainExtractor(bet_path="nonexistent_bet_binary_xyz")
    _report("check_bet_available returns False", extractor.check_bet_available() is False)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake_t1 = tmp / "t1w.nii.gz"
        fake_t1.touch()

        try:
            extractor.extract(fake_t1, tmp)
            _report("should have raised RuntimeError", False)
        except RuntimeError as e:
            _report("raises RuntimeError", True)
            _report("error mentions not found", "not found" in str(e).lower(),
                    f"got: {e}")


def main():
    print("=" * 50)
    print("BRAIN EXTRACTION TESTS")
    print("=" * 50)

    if BET_AVAILABLE:
        print("FSL BET: available")
    else:
        print("FSL BET: not available (some tests will adjust)")

    test_brain_extractor_init()
    test_brain_extractor_custom_params()
    test_bet_command_construction()
    test_bids_t1w_discovery()
    test_result_dataclass()
    test_bet_not_available()

    print("\n" + "=" * 50)
    status_parts = [f"{passed}/{total} passed"]
    if failed:
        status_parts.append(f"{failed} failed")
    if skipped:
        status_parts.append(f"{skipped} skipped")
    print(f"RESULTS: {', '.join(status_parts)}")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
