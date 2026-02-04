"""
Unified FreeSurfer reconstruction detection for connectivity pipelines.

Consolidates detection logic from:
- mrtrix3_demon_addon/run_pipeline.py (detect_freesurfer_version, find_freesurfer_files)
- nilearn_RSB_analysis_pipeline/scripts/submit_pipeline.py (check_for_existing_recons)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class FreeSurferInfo:
    """Information about detected FreeSurfer reconstruction."""

    found: bool = False
    version: Optional[str] = None
    path: Optional[Path] = None
    available_atlases: List[str] = field(default_factory=list)
    files: Dict[str, Path] = field(default_factory=dict)
    warning: Optional[str] = None

    def __bool__(self):
        """Allow using FreeSurferInfo directly in boolean context."""
        return self.found


class FreeSurferDetector:
    """
    Detect FreeSurfer reconstructions and available atlases.

    Supports multiple FreeSurfer versions and directory structures:
    - FreeSurfer 8.0: {subject}/freesurfer8.0/{subject_name}/
    - FreeSurfer 7.x: {subject}/FreeSurfer7/
    - FreeSurfer 5.3: {subject}/FreeSurfer/
    - BIDS derivatives: {project}/data/recons/sub-{id}/ses-{session}/

    Example:
        >>> detector = FreeSurferDetector()
        >>> fs_info = detector.detect("/path/to/subject")
        >>> if fs_info:
        ...     print(f"Found FreeSurfer {fs_info.version}")
        ...     print(f"Available atlases: {fs_info.available_atlases}")
    """

    # Version configurations with priority (lower = higher priority)
    VERSIONS = {
        'freesurfer7.x': {
            'priority': 1,
            'subdirs': ['freesurfer7.x', 'FreeSurfer7', 'freesurfer7'],
            'nested_subject': False,
        },
        'freesurfer8.0': {
            'priority': 2,
            'subdirs': ['freesurfer8.0', 'FreeSurfer8'],
            'nested_subject': True,  # Has subject name as subdirectory
        },
        'freesurfer5.3': {
            'priority': 3,
            'subdirs': ['FreeSurfer', 'freesurfer'],
            'nested_subject': False,
        },
    }

    # Required and optional FreeSurfer files
    FS_FILES = {
        # Required for basic operation
        'aparc_aseg': 'mri/aparc+aseg.mgz',
        'brain': 'mri/brain.mgz',
        'orig': 'mri/orig.mgz',
        # Optional atlas files
        'aparc_dk': 'mri/aparc.DKTatlas+aseg.mgz',
        'aparc_destrieux': 'mri/aparc.a2009s+aseg.mgz',
        # Additional useful files
        'wm': 'mri/wm.mgz',
        'aseg': 'mri/aseg.mgz',
        'ribbon': 'mri/ribbon.mgz',
    }

    # Atlas mapping: file key -> atlas name
    ATLAS_MAPPING = {
        'aparc_aseg': 'FreeSurfer_DK',
        'aparc_dk': 'FreeSurfer_DKT',
        'aparc_destrieux': 'FreeSurfer_Destrieux',
    }

    def __init__(self, include_template_atlases: bool = True):
        """
        Initialize detector.

        Args:
            include_template_atlases: If True, always include 'Brainnetome' in
                available atlases (template-based, no FreeSurfer needed)
        """
        self.include_template_atlases = include_template_atlases

    def detect(
        self,
        subject_path: str | Path,
        subject_name: Optional[str] = None,
        is_nhp: bool = False,
    ) -> FreeSurferInfo:
        """
        Detect FreeSurfer reconstruction for a subject.

        Args:
            subject_path: Path to subject folder
            subject_name: Subject name (for nested directories). If None,
                uses the subject_path folder name.
            is_nhp: If True, skip FreeSurfer detection (NHPs use template only)

        Returns:
            FreeSurferInfo with detection results
        """
        subject_path = Path(subject_path)

        if subject_name is None:
            subject_name = subject_path.name

        # NHPs don't use FreeSurfer
        if is_nhp:
            return FreeSurferInfo(
                found=False,
                available_atlases=['Brainnetome'] if self.include_template_atlases else [],
                warning="NHP subject - using template-based parcellation only"
            )

        # Try each version in priority order
        detected = self._detect_by_version(subject_path, subject_name)

        if detected:
            return detected

        # Try BIDS-style recons directory
        detected = self._detect_bids_recons(subject_path, subject_name)

        if detected:
            return detected

        # Nothing found
        atlases = ['Brainnetome'] if self.include_template_atlases else []
        return FreeSurferInfo(
            found=False,
            available_atlases=atlases,
            warning="No FreeSurfer reconstruction found - using template-based parcellation only"
        )

    def _detect_by_version(
        self,
        subject_path: Path,
        subject_name: str,
    ) -> Optional[FreeSurferInfo]:
        """Try detecting FreeSurfer by version directories."""

        # Sort versions by priority
        sorted_versions = sorted(
            self.VERSIONS.items(),
            key=lambda x: x[1]['priority']
        )

        for version, config in sorted_versions:
            for subdir in config['subdirs']:
                if config['nested_subject']:
                    # Version 8.0 style: freesurfer8.0/{subject_name}/
                    fs_path = subject_path / subdir / subject_name
                else:
                    # Version 7/5.3 style: FreeSurfer7/
                    fs_path = subject_path / subdir

                if fs_path.exists():
                    info = self._validate_and_build_info(fs_path, version)
                    if info and info.found:
                        return info

        return None

    def _detect_bids_recons(
        self,
        subject_path: Path,
        subject_name: str,
    ) -> Optional[FreeSurferInfo]:
        """Try detecting FreeSurfer in BIDS derivatives structure."""

        # Try common BIDS patterns
        patterns = [
            subject_path / "data" / "recons" / f"sub-{subject_name}",
            subject_path / "derivatives" / "freesurfer" / f"sub-{subject_name}",
            subject_path.parent / "derivatives" / "freesurfer" / f"sub-{subject_name}",
        ]

        for base_path in patterns:
            if not base_path.exists():
                continue

            # Check for session subdirectories
            session_dirs = sorted([
                d for d in base_path.iterdir()
                if d.is_dir() and d.name.startswith('ses-')
            ])

            if session_dirs:
                # Use first available session
                fs_path = session_dirs[0]
            else:
                fs_path = base_path

            info = self._validate_and_build_info(fs_path, 'bids')
            if info and info.found:
                return info

        return None

    def _validate_and_build_info(
        self,
        fs_path: Path,
        version: str,
    ) -> Optional[FreeSurferInfo]:
        """Validate FreeSurfer directory and build info object."""

        # Check for required aparc+aseg.mgz
        aparc_aseg = fs_path / self.FS_FILES['aparc_aseg']
        if not aparc_aseg.exists():
            return None

        # Find all available files
        files = {}
        for key, relative_path in self.FS_FILES.items():
            full_path = fs_path / relative_path
            if full_path.exists():
                files[key] = full_path

        # Determine available atlases
        atlases = []
        if self.include_template_atlases:
            atlases.append('Brainnetome')

        for file_key, atlas_name in self.ATLAS_MAPPING.items():
            if file_key in files:
                atlases.append(atlas_name)

        # Generate warning for older versions
        warning = None
        if version == 'freesurfer5.3':
            warning = "WARNING: Using FreeSurfer 5.3 - strongly recommend upgrading to FreeSurfer 7+ for better results"

        return FreeSurferInfo(
            found=True,
            version=version,
            path=fs_path,
            available_atlases=atlases,
            files=files,
            warning=warning,
        )

    def get_parcellation_file(
        self,
        fs_info: FreeSurferInfo,
        atlas: str,
    ) -> Optional[Path]:
        """
        Get the parcellation file path for a specific atlas.

        Args:
            fs_info: FreeSurferInfo from detect()
            atlas: Atlas name ('FreeSurfer_DK', 'FreeSurfer_Destrieux', etc.)

        Returns:
            Path to parcellation file, or None if not available
        """
        if not fs_info.found:
            return None

        # Reverse mapping: atlas name -> file key
        atlas_to_file = {v: k for k, v in self.ATLAS_MAPPING.items()}

        file_key = atlas_to_file.get(atlas)
        if file_key and file_key in fs_info.files:
            return fs_info.files[file_key]

        return None


def detect_freesurfer(
    subject_path: str | Path,
    subject_name: Optional[str] = None,
    is_nhp: bool = False,
) -> FreeSurferInfo:
    """
    Convenience function for FreeSurfer detection.

    Args:
        subject_path: Path to subject folder
        subject_name: Subject name (optional)
        is_nhp: If True, skip detection (NHP uses template only)

    Returns:
        FreeSurferInfo with detection results
    """
    detector = FreeSurferDetector()
    return detector.detect(subject_path, subject_name, is_nhp)
