"""
FSL BET wrapper for T1w brain mask creation.

Provides a simple interface to FSL's BET (Brain Extraction Tool) for
skull-stripping T1-weighted images, with BIDS-aware convenience methods.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BrainExtractionResult:
    """Result of a brain extraction operation."""
    mask_path: Path              # Binary brain mask NIfTI
    brain_path: Path             # Skull-stripped brain NIfTI
    input_path: Path             # Original T1w path
    method: str                  # e.g. 'fsl_bet'
    parameters: dict = field(default_factory=dict)


class BrainExtractor:
    """FSL BET wrapper for T1w brain mask creation.

    Args:
        bet_path: Path to FSL ``bet`` executable.
        fractional_intensity: BET ``-f`` parameter (0-1, smaller = larger brain estimate).
    """

    def __init__(
        self,
        bet_path: str = "bet",
        fractional_intensity: float = 0.5,
    ):
        self.bet_path = bet_path
        self.fractional_intensity = fractional_intensity

    def extract(
        self,
        t1w_path: Path,
        output_dir: Path,
        output_prefix: Optional[str] = None,
        robust: bool = True,
    ) -> BrainExtractionResult:
        """Run FSL BET on a T1w image.

        Args:
            t1w_path: Path to T1w NIfTI.
            output_dir: Where to save outputs.
            output_prefix: Filename prefix (default: derived from input).
            robust: Use BET ``-R`` for robust brain centre estimation.

        Returns:
            :class:`BrainExtractionResult` with paths to brain and mask files.

        Raises:
            FileNotFoundError: If the input T1w file does not exist.
            RuntimeError: If FSL BET is not available or fails.
        """
        t1w_path = Path(t1w_path)
        output_dir = Path(output_dir)

        if not t1w_path.exists():
            raise FileNotFoundError(f"T1w file not found: {t1w_path}")

        if not self.check_bet_available():
            raise RuntimeError(
                f"FSL BET not found at '{self.bet_path}'. "
                "Ensure FSL is installed and in your PATH."
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        if output_prefix is None:
            # Derive prefix from input filename
            stem = t1w_path.name
            for ext in (".nii.gz", ".nii"):
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            output_prefix = f"{stem}_brain"

        output_base = output_dir / output_prefix

        cmd = self.build_command(t1w_path, output_base, robust=robust)
        logger.info("Running BET: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            if result.stdout:
                logger.debug("BET stdout: %s", result.stdout)
            if result.stderr:
                logger.debug("BET stderr: %s", result.stderr)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"BET failed with exit code {exc.returncode}: {exc.stderr}"
            ) from exc

        brain_path = Path(f"{output_base}.nii.gz")
        mask_path = Path(f"{output_base}_mask.nii.gz")

        # BET may produce files without the _mask suffix depending on version
        if not brain_path.exists():
            raise RuntimeError(f"BET did not produce expected output: {brain_path}")

        parameters = {
            "fractional_intensity": self.fractional_intensity,
            "robust": robust,
            "bet_path": self.bet_path,
        }

        return BrainExtractionResult(
            mask_path=mask_path,
            brain_path=brain_path,
            input_path=t1w_path,
            method="fsl_bet",
            parameters=parameters,
        )

    def extract_from_bids(
        self,
        bids_dir: Path,
        subject_id: str,
        session_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> BrainExtractionResult:
        """Find T1w in BIDS structure and run extraction.

        Searches ``{bids_dir}/sub-{id}/[ses-{id}/]anat/sub-{id}*_T1w.nii.gz``

        Args:
            bids_dir: Root of BIDS dataset.
            subject_id: Subject identifier (without ``sub-`` prefix).
            session_id: Session identifier (without ``ses-`` prefix), or *None*.
            output_dir: Where to save outputs. Defaults to ``derivatives/brain_extraction``
                        under the BIDS root.

        Returns:
            :class:`BrainExtractionResult`

        Raises:
            FileNotFoundError: If no T1w file is found.
        """
        bids_dir = Path(bids_dir)
        sub_label = f"sub-{subject_id}"

        if session_id:
            anat_dir = bids_dir / sub_label / f"ses-{session_id}" / "anat"
        else:
            anat_dir = bids_dir / sub_label / "anat"

        # Search for T1w
        t1w_files = sorted(anat_dir.glob(f"{sub_label}*_T1w.nii.gz"))
        if not t1w_files:
            t1w_files = sorted(anat_dir.glob(f"{sub_label}*_T1w.nii"))
        if not t1w_files:
            raise FileNotFoundError(
                f"No T1w file found in {anat_dir}. "
                f"Expected pattern: {sub_label}*_T1w.nii.gz"
            )

        t1w_path = t1w_files[0]
        logger.info("Found T1w: %s", t1w_path)

        if output_dir is None:
            output_dir = bids_dir / "derivatives" / "brain_extraction" / sub_label
            if session_id:
                output_dir = output_dir / f"ses-{session_id}"
            output_dir = output_dir / "anat"

        prefix = f"{sub_label}_T1w_brain"
        return self.extract(t1w_path, output_dir, output_prefix=prefix)

    def build_command(
        self, t1w_path: Path, output_base: Path, robust: bool = True
    ) -> list:
        """Build the BET command list.

        Args:
            t1w_path: Input T1w NIfTI.
            output_base: Output path base (without extension).
            robust: Whether to use ``-R`` flag.

        Returns:
            Command as a list of strings.
        """
        cmd = [
            self.bet_path,
            str(t1w_path),
            str(output_base),
            "-m",  # generate brain mask
            "-f", str(self.fractional_intensity),
        ]
        if robust:
            cmd.append("-R")
        return cmd

    def check_bet_available(self) -> bool:
        """Check whether FSL BET is available on the system."""
        return shutil.which(self.bet_path) is not None
