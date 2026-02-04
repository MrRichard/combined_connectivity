"""
Unified DICOM-to-BIDS conversion for both DWI and fMRI modalities.

Config-driven classification using series description patterns, with logic
extracted from both the DWI pipeline (ImageTypeChecker.py) and fMRI pipeline
(01_validate_and_convert.py).
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default series description patterns for classifying dcm2niix output
DEFAULT_SERIES_PATTERNS = {
    "dwi": ["NODDI", "DTI", "DWI", "DKI"],
    "func": ["BOLD", "resting", "rest", "fMRI", "eyes open", "task"],
    "anat_t1w": ["tfl3d", "T1w", "MPRAGE", "t1_mprage"],
    "anat_flair": ["FLAIR", "spcir"],
    "anat_t2w": ["T2w", "spc2"],
    "fmap": ["FIELD_MAP", "fieldmap", "FieldMap", "FIELD_MAPPING",
             "B0MAP", "B0_MAP", "GRE_FIELD_MAPPING", "DISTORTION"],
}

# Phase encoding direction mapping (dcm2niix convention -> BIDS label)
PE_DIRECTION_MAP = {
    "j-": "AP",   # anterior-posterior
    "j": "PA",    # posterior-anterior
    "i-": "RL",   # right-left
    "i": "LR",    # left-right
    "k-": "IS",   # inferior-superior
    "k": "SI",    # superior-inferior
}


@dataclass
class BIDSConversionResult:
    """Result of a DICOM-to-BIDS conversion."""
    bids_dir: Path
    subject_id: str
    session_id: Optional[str]
    modalities_found: List[str] = field(default_factory=list)
    files: Dict[str, List[Path]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class DicomToBIDS:
    """Unified DICOM-to-BIDS converter for both DWI and fMRI data.

    Args:
        config: dict with ``series_patterns`` for classification.
                Defaults to built-in patterns if *None*.
    """

    def __init__(self, config: Optional[dict] = None):
        if config and "series_patterns" in config:
            self.series_patterns = config["series_patterns"]
        else:
            self.series_patterns = DEFAULT_SERIES_PATTERNS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        dicom_dir: Path,
        output_dir: Path,
        subject_id: str,
        session_id: Optional[str] = None,
        dcm2niix_path: str = "dcm2niix",
    ) -> BIDSConversionResult:
        """Full DICOM-to-BIDS conversion pipeline.

        1. Run *dcm2niix* on *dicom_dir* -> temp NIfTI + JSON sidecars
        2. Classify each JSON sidecar to determine modality
        3. Organise into BIDS directory structure
        4. Write ``dataset_description.json``
        5. Add ``IntendedFor`` fields to fieldmap JSON sidecars
        6. Return :class:`BIDSConversionResult`
        """
        dicom_dir = Path(dicom_dir)
        output_dir = Path(output_dir)
        warnings: List[str] = []

        # Step 1 — run dcm2niix into a temp directory
        tmp_dir = output_dir / "tmp_dcm2niix"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            json_paths = self._run_dcm2niix(dicom_dir, tmp_dir, dcm2niix_path)
        except Exception as exc:
            warnings.append(f"dcm2niix failed: {exc}")
            return BIDSConversionResult(
                bids_dir=output_dir,
                subject_id=subject_id,
                session_id=session_id,
                warnings=warnings,
            )

        # Step 2 — classify each sidecar
        classified: List[dict] = []
        for jp in json_paths:
            try:
                info = self.classify_file(jp)
                if info is None:
                    warnings.append(f"Skipped unrecognised series: {jp.name}")
                    continue
                info["json_path"] = jp
                # Locate the corresponding NIfTI
                nii_path = self._find_nifti_for_json(jp)
                info["nii_path"] = nii_path
                classified.append(info)
            except Exception as exc:
                warnings.append(f"Classification failed for {jp.name}: {exc}")

        # Step 3 — organise into BIDS
        bids_files = self._organize_bids(classified, output_dir, subject_id, session_id)

        # Step 4 — dataset_description.json
        self._write_dataset_description(output_dir)

        # Step 5 — IntendedFor on fieldmap sidecars
        bold_paths = bids_files.get("func", [])
        fmap_paths = bids_files.get("fmap", [])
        if bold_paths and fmap_paths:
            # Pick the first BOLD file as the IntendedFor target
            bold_target = bold_paths[0]
            for fmap_path in fmap_paths:
                fmap_json = fmap_path.with_name(
                    fmap_path.name.replace(".nii.gz", ".json").replace(".nii", ".json")
                )
                if fmap_json.exists():
                    self._add_intended_for(fmap_json, bold_target, subject_id, session_id)

        # Clean up temp directory
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

        modalities_found = sorted(bids_files.keys())

        return BIDSConversionResult(
            bids_dir=output_dir,
            subject_id=subject_id,
            session_id=session_id,
            modalities_found=modalities_found,
            files=bids_files,
            warnings=warnings,
        )

    def classify_file(self, json_sidecar: Path) -> Optional[dict]:
        """Classify a single NIfTI by its JSON sidecar.

        Returns a dict with keys, or *None* if the series is unrecognised
        (e.g. perfusion, localizers, derived sequences):

            - ``modality``: ``'dwi'`` | ``'func'`` | ``'anat'`` | ``'fmap'``
            - ``suffix``: BIDS suffix (``'dwi'``, ``'bold'``, ``'sbref'``,
              ``'T1w'``, ``'T2w'``, ``'FLAIR'``, ``'epi'``, ``'phasediff'``,
              ``'magnitude1'``, ``'magnitude2'``)
            - ``pe_direction``: ``'AP'`` | ``'PA'`` | ``'LR'`` | ``'RL'`` | *None*
            - ``pe_direction_raw``: raw dcm2niix value (``'j-'``, ``'j'``, ...)
            - ``is_sbref``: *bool*
            - ``metadata``: dict of relevant sidecar fields
        """
        json_sidecar = Path(json_sidecar)
        sidecar = self._read_json(json_sidecar)

        series_desc = sidecar.get("SeriesDescription", "")
        protocol_name = sidecar.get("ProtocolName", "")
        image_type = sidecar.get("ImageType", [])
        image_type_str = str(image_type).upper()
        pe_raw = sidecar.get("PhaseEncodingDirection", None)
        pe_direction = PE_DIRECTION_MAP.get(pe_raw) if pe_raw else None
        echo_time = sidecar.get("EchoTime")
        readout_time = sidecar.get("ReadoutTime")
        repetition_time = sidecar.get("RepetitionTime")

        metadata = {}
        if echo_time is not None:
            metadata["echo_time"] = echo_time
        if readout_time is not None:
            metadata["readout_time"] = readout_time
        if repetition_time is not None:
            metadata["repetition_time"] = repetition_time

        # --- SBRef detection ---
        is_sbref = self._is_sbref(sidecar, json_sidecar)

        # --- Classification cascade ---

        # 1. Fieldmaps (magnitude / phasediff / EPI)
        fmap_result = self._classify_fieldmap(
            series_desc, image_type_str, echo_time, sidecar
        )
        if fmap_result:
            return {
                "modality": "fmap",
                "suffix": fmap_result,
                "pe_direction": pe_direction,
                "pe_direction_raw": pe_raw,
                "is_sbref": False,
                "metadata": metadata,
            }

        # 2. DWI
        if self._matches_patterns(series_desc, self.series_patterns.get("dwi", [])):
            suffix = "sbref" if is_sbref else "dwi"
            return {
                "modality": "dwi",
                "suffix": suffix,
                "pe_direction": pe_direction,
                "pe_direction_raw": pe_raw,
                "is_sbref": is_sbref,
                "metadata": metadata,
            }

        # 3. Functional (BOLD)
        if self._matches_patterns(
            series_desc, self.series_patterns.get("func", [])
        ) or self._matches_patterns(
            protocol_name, self.series_patterns.get("func", [])
        ):
            suffix = "sbref" if is_sbref else "bold"
            return {
                "modality": "func",
                "suffix": suffix,
                "pe_direction": pe_direction,
                "pe_direction_raw": pe_raw,
                "is_sbref": is_sbref,
                "metadata": metadata,
            }

        # 4. Anatomical — T1w
        if self._matches_patterns(
            series_desc, self.series_patterns.get("anat_t1w", [])
        ):
            return {
                "modality": "anat",
                "suffix": "T1w",
                "pe_direction": pe_direction,
                "pe_direction_raw": pe_raw,
                "is_sbref": False,
                "metadata": metadata,
            }

        # 5. Anatomical — FLAIR
        if self._matches_patterns(
            series_desc, self.series_patterns.get("anat_flair", [])
        ):
            return {
                "modality": "anat",
                "suffix": "FLAIR",
                "pe_direction": pe_direction,
                "pe_direction_raw": pe_raw,
                "is_sbref": False,
                "metadata": metadata,
            }

        # 6. Anatomical — T2w
        if self._matches_patterns(
            series_desc, self.series_patterns.get("anat_t2w", [])
        ):
            return {
                "modality": "anat",
                "suffix": "T2w",
                "pe_direction": pe_direction,
                "pe_direction_raw": pe_raw,
                "is_sbref": False,
                "metadata": metadata,
            }

        # 7. BidsGuess fallback (from dcm2niix) — only accept known modalities
        bids_guess = sidecar.get("BidsGuess", [])
        if bids_guess and len(bids_guess) >= 2:
            guess_modality = bids_guess[0].strip("/")
            guess_suffix = bids_guess[1].strip("_")
            if guess_modality in ("anat", "func", "dwi", "fmap"):
                return {
                    "modality": guess_modality,
                    "suffix": guess_suffix,
                    "pe_direction": pe_direction,
                    "pe_direction_raw": pe_raw,
                    "is_sbref": is_sbref,
                    "metadata": metadata,
                }

        # 8. Unrecognised — skip (perfusion, localizers, derived, etc.)
        logger.info(
            "Skipping unrecognised series: %s (SeriesDescription=%r, ProtocolName=%r)",
            json_sidecar.name, series_desc, protocol_name,
        )
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_dcm2niix(
        self, dicom_dir: Path, output_dir: Path, dcm2niix_path: str
    ) -> List[Path]:
        """Run dcm2niix with standard flags. Returns list of JSON sidecar paths."""
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            dcm2niix_path,
            "-b", "y",     # generate BIDS sidecar
            "-m", "y",     # merge 2D slices
            "-z", "y",     # gzip output
            "-f", "%p_%t_%s",  # filename pattern
            "-o", str(output_dir),
            str(dicom_dir),
        ]
        logger.info("Running dcm2niix: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            logger.debug("dcm2niix stdout: %s", result.stdout)
        if result.stderr:
            logger.debug("dcm2niix stderr: %s", result.stderr)

        return sorted(output_dir.glob("*.json"))

    def _organize_bids(
        self,
        classified_files: List[dict],
        output_dir: Path,
        subject_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, List[Path]]:
        """Move/rename classified files into BIDS directory structure."""
        sub_label = f"sub-{subject_id}"
        ses_label = f"ses-{session_id}" if session_id else None

        # Build base path
        if ses_label:
            base_dir = output_dir / sub_label / ses_label
        else:
            base_dir = output_dir / sub_label

        # Track run numbers per modality+suffix combination
        run_counters: Dict[str, int] = {}
        result: Dict[str, List[Path]] = {}

        for info in classified_files:
            modality = info["modality"]
            suffix = info["suffix"]
            pe_dir = info.get("pe_direction")
            is_sbref = info.get("is_sbref", False)
            nii_path = info.get("nii_path")
            json_path = info.get("json_path")

            if nii_path is None:
                continue

            # Skip SBRef files
            if is_sbref:
                logger.info("Skipping SBRef file: %s", nii_path.name)
                continue

            # Create modality directory
            mod_dir = base_dir / modality
            mod_dir.mkdir(parents=True, exist_ok=True)

            # Build BIDS filename
            parts = [sub_label]
            if ses_label:
                parts.append(ses_label)

            # Add task entity for BOLD
            if modality == "func" and suffix == "bold":
                parts.append("task-rest")

            # Add direction entity
            if pe_dir:
                parts.append(f"dir-{pe_dir}")

            # Handle run numbers for duplicates
            counter_key = f"{modality}_{suffix}_{pe_dir or 'nodir'}"
            run_counters[counter_key] = run_counters.get(counter_key, 0) + 1
            if run_counters[counter_key] > 1:
                parts.append(f"run-{run_counters[counter_key]:02d}")

            parts.append(suffix)
            bids_stem = "_".join(parts)

            # Determine extension
            nii_path = Path(nii_path)
            ext = ".nii.gz" if nii_path.name.endswith(".nii.gz") else ".nii"

            dest_nii = mod_dir / f"{bids_stem}{ext}"
            shutil.copy2(nii_path, dest_nii)
            logger.info("Copied %s -> %s", nii_path.name, dest_nii)

            # Copy JSON sidecar
            if json_path and Path(json_path).exists():
                dest_json = mod_dir / f"{bids_stem}.json"
                shutil.copy2(json_path, dest_json)

            # Copy bvec/bval for DWI
            if modality == "dwi":
                for bids_ext in (".bvec", ".bval"):
                    src = nii_path.with_name(
                        nii_path.name.replace(".nii.gz", bids_ext).replace(".nii", bids_ext)
                    )
                    if src.exists():
                        dest = mod_dir / f"{bids_stem}{bids_ext}"
                        shutil.copy2(src, dest)

            result.setdefault(modality, []).append(dest_nii)

        return result

    def _add_intended_for(
        self,
        fmap_json: Path,
        bold_path: Path,
        subject_id: str,
        session_id: Optional[str] = None,
    ):
        """Add IntendedFor field to a fieldmap JSON sidecar."""
        try:
            sidecar = self._read_json(fmap_json)

            # Build BIDS-relative IntendedFor path
            if session_id:
                intended = f"ses-{session_id}/func/{bold_path.name}"
            else:
                intended = f"func/{bold_path.name}"

            sidecar["IntendedFor"] = intended

            with open(fmap_json, "w", encoding="utf-8") as fh:
                json.dump(sidecar, fh, indent=2)

            logger.info("Added IntendedFor to %s -> %s", fmap_json.name, intended)
        except Exception as exc:
            logger.warning("Failed to add IntendedFor to %s: %s", fmap_json.name, exc)

    def _write_dataset_description(self, bids_dir: Path):
        """Write minimal ``dataset_description.json``."""
        desc_path = bids_dir / "dataset_description.json"
        if desc_path.exists():
            return

        description = {
            "Name": "Combined Connectivity Dataset",
            "BIDSVersion": "1.6.0",
            "DatasetType": "raw",
            "GeneratedBy": [
                {
                    "Name": "connectivity_shared.dicom_to_bids",
                    "Version": "0.1.0",
                }
            ],
        }
        with open(desc_path, "w", encoding="utf-8") as fh:
            json.dump(description, fh, indent=2)
        logger.info("Wrote dataset_description.json to %s", bids_dir)

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Case-insensitive check whether *text* contains any of *patterns*."""
        text_lower = text.lower()
        return any(p.lower() in text_lower for p in patterns)

    def _classify_fieldmap(
        self,
        series_desc: str,
        image_type_str: str,
        echo_time: Optional[float],
        sidecar: dict,
    ) -> Optional[str]:
        """Classify fieldmap images. Returns BIDS suffix or None."""
        fmap_patterns = self.series_patterns.get("fmap", [])
        if not self._matches_patterns(series_desc, fmap_patterns):
            return None

        has_magnitude = "MAGNITUDE" in image_type_str
        has_phase = "PHASE" in image_type_str

        if has_phase or "PHASEDIFF" in series_desc.upper():
            return "phasediff"

        if has_magnitude:
            # Distinguish magnitude1 vs magnitude2 by echo time
            if echo_time is not None and echo_time < 0.005:
                return "magnitude1"
            else:
                return "magnitude2"

        # Generic fieldmap / EPI
        pe_raw = sidecar.get("PhaseEncodingDirection")
        if pe_raw:
            return "epi"

        return "fieldmap"

    def _is_sbref(self, sidecar: dict, json_path: Path) -> bool:
        """Detect whether a sidecar corresponds to an SBRef image."""
        series_desc = sidecar.get("SeriesDescription", "").lower()
        protocol_name = sidecar.get("ProtocolName", "").lower()
        sequence_name = sidecar.get("SequenceName", "").lower()

        sbref_indicators = ["sbref", "sb_ref", "single_band_ref"]
        for indicator in sbref_indicators:
            if indicator in series_desc or indicator in protocol_name or indicator in sequence_name:
                return True

        # Volume-count fallback: SBRef typically has <= 5 volumes
        nii_path = self._find_nifti_for_json(json_path)
        if nii_path and nii_path.exists():
            try:
                import nibabel as nib
                img = nib.load(nii_path)
                if len(img.shape) >= 4 and img.shape[3] <= 5:
                    return True
            except Exception:
                pass

        return False

    # ------------------------------------------------------------------
    # File-system helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_nifti_for_json(json_path: Path) -> Optional[Path]:
        """Find the NIfTI file corresponding to a JSON sidecar."""
        json_path = Path(json_path)
        gz = json_path.with_suffix(".nii.gz")
        # with_suffix replaces only the last suffix, so .json -> .nii.gz needs
        # a manual approach for the .nii.gz case
        stem = json_path.with_suffix("")  # strip .json
        gz = stem.with_suffix(".nii.gz")
        nii = stem.with_suffix(".nii")
        if gz.exists():
            return gz
        if nii.exists():
            return nii
        return None

    @staticmethod
    def _read_json(path: Path) -> dict:
        """Read a JSON file, trying multiple encodings."""
        for encoding in ("utf-8", "utf-8-sig", "iso-8859-1", "windows-1252"):
            try:
                with open(path, "r", encoding=encoding) as fh:
                    return json.load(fh)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        # Last-resort: ignore bad bytes
        with open(path, "rb") as fh:
            raw = fh.read().replace(b"\xa0", b" ")
            return json.loads(raw.decode("utf-8", errors="ignore"))
