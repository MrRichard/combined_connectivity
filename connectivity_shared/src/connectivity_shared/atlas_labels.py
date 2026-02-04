"""
Atlas label definitions for connectivity analysis.

Provides canonical label lists for standard atlases used in both
DWI and fMRI connectivity pipelines.
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Brainnetome Atlas (246 ROIs)
# Based on: Fan et al. (2016) The Human Brainnetome Atlas
BRAINNETOME_LABELS = [
    # Superior Frontal Gyrus (SFG) - 14 regions (1-14)
    "SFG_L_7_1", "SFG_R_7_1", "SFG_L_7_2", "SFG_R_7_2",
    "SFG_L_7_3", "SFG_R_7_3", "SFG_L_7_4", "SFG_R_7_4",
    "SFG_L_7_5", "SFG_R_7_5", "SFG_L_7_6", "SFG_R_7_6",
    "SFG_L_7_7", "SFG_R_7_7",
    # Middle Frontal Gyrus (MFG) - 14 regions (15-28)
    "MFG_L_7_1", "MFG_R_7_1", "MFG_L_7_2", "MFG_R_7_2",
    "MFG_L_7_3", "MFG_R_7_3", "MFG_L_7_4", "MFG_R_7_4",
    "MFG_L_7_5", "MFG_R_7_5", "MFG_L_7_6", "MFG_R_7_6",
    "MFG_L_7_7", "MFG_R_7_7",
    # Inferior Frontal Gyrus (IFG) - 12 regions (29-40)
    "IFG_L_6_1", "IFG_R_6_1", "IFG_L_6_2", "IFG_R_6_2",
    "IFG_L_6_3", "IFG_R_6_3", "IFG_L_6_4", "IFG_R_6_4",
    "IFG_L_6_5", "IFG_R_6_5", "IFG_L_6_6", "IFG_R_6_6",
    # Orbital Gyrus (OrG) - 12 regions (41-52)
    "OrG_L_6_1", "OrG_R_6_1", "OrG_L_6_2", "OrG_R_6_2",
    "OrG_L_6_3", "OrG_R_6_3", "OrG_L_6_4", "OrG_R_6_4",
    "OrG_L_6_5", "OrG_R_6_5", "OrG_L_6_6", "OrG_R_6_6",
    # Precentral Gyrus (PrG) - 12 regions (53-64)
    "PrG_L_6_1", "PrG_R_6_1", "PrG_L_6_2", "PrG_R_6_2",
    "PrG_L_6_3", "PrG_R_6_3", "PrG_L_6_4", "PrG_R_6_4",
    "PrG_L_6_5", "PrG_R_6_5", "PrG_L_6_6", "PrG_R_6_6",
    # Paracentral Lobule (PCL) - 4 regions (65-68)
    "PCL_L_2_1", "PCL_R_2_1", "PCL_L_2_2", "PCL_R_2_2",
    # Short Insular Gyri (ShInG) - 4 regions (69-72)
    "ShInG_L_2_1", "ShInG_R_2_1", "ShInG_L_2_2", "ShInG_R_2_2",
    # Long Insular Gyri (LoInG) - 4 regions (73-76)
    "LoInG_L_2_1", "LoInG_R_2_1", "LoInG_L_2_2", "LoInG_R_2_2",
    # Postcentral Gyrus (PoG) - 8 regions (77-84)
    "PoG_L_4_1", "PoG_R_4_1", "PoG_L_4_2", "PoG_R_4_2",
    "PoG_L_4_3", "PoG_R_4_3", "PoG_L_4_4", "PoG_R_4_4",
    # Superior Parietal Lobule (SPL) - 10 regions (85-94)
    "SPL_L_5_1", "SPL_R_5_1", "SPL_L_5_2", "SPL_R_5_2",
    "SPL_L_5_3", "SPL_R_5_3", "SPL_L_5_4", "SPL_R_5_4",
    "SPL_L_5_5", "SPL_R_5_5",
    # Inferior Parietal Lobule (IPL) - 12 regions (95-106)
    "IPL_L_6_1", "IPL_R_6_1", "IPL_L_6_2", "IPL_R_6_2",
    "IPL_L_6_3", "IPL_R_6_3", "IPL_L_6_4", "IPL_R_6_4",
    "IPL_L_6_5", "IPL_R_6_5", "IPL_L_6_6", "IPL_R_6_6",
    # Precuneus (Pcun) - 8 regions (107-114)
    "Pcun_L_4_1", "Pcun_R_4_1", "Pcun_L_4_2", "Pcun_R_4_2",
    "Pcun_L_4_3", "Pcun_R_4_3", "Pcun_L_4_4", "Pcun_R_4_4",
    # Cingulate Gyrus (CG) - 14 regions (115-128)
    "CG_L_7_1", "CG_R_7_1", "CG_L_7_2", "CG_R_7_2",
    "CG_L_7_3", "CG_R_7_3", "CG_L_7_4", "CG_R_7_4",
    "CG_L_7_5", "CG_R_7_5", "CG_L_7_6", "CG_R_7_6",
    "CG_L_7_7", "CG_R_7_7",
    # Medial Frontal Cortex (MFC) - 8 regions (129-136)
    "MFC_L_4_1", "MFC_R_4_1", "MFC_L_4_2", "MFC_R_4_2",
    "MFC_L_4_3", "MFC_R_4_3", "MFC_L_4_4", "MFC_R_4_4",
    # Superior Temporal Gyrus (STG) - 12 regions (137-148)
    "STG_L_6_1", "STG_R_6_1", "STG_L_6_2", "STG_R_6_2",
    "STG_L_6_3", "STG_R_6_3", "STG_L_6_4", "STG_R_6_4",
    "STG_L_6_5", "STG_R_6_5", "STG_L_6_6", "STG_R_6_6",
    # Middle Temporal Gyrus (MTG) - 8 regions (149-156)
    "MTG_L_4_1", "MTG_R_4_1", "MTG_L_4_2", "MTG_R_4_2",
    "MTG_L_4_3", "MTG_R_4_3", "MTG_L_4_4", "MTG_R_4_4",
    # Inferior Temporal Gyrus (ITG) - 14 regions (157-170)
    "ITG_L_7_1", "ITG_R_7_1", "ITG_L_7_2", "ITG_R_7_2",
    "ITG_L_7_3", "ITG_R_7_3", "ITG_L_7_4", "ITG_R_7_4",
    "ITG_L_7_5", "ITG_R_7_5", "ITG_L_7_6", "ITG_R_7_6",
    "ITG_L_7_7", "ITG_R_7_7",
    # Fusiform Gyrus (FuG) - 6 regions (171-176)
    "FuG_L_3_1", "FuG_R_3_1", "FuG_L_3_2", "FuG_R_3_2",
    "FuG_L_3_3", "FuG_R_3_3",
    # Parahippocampal Gyrus (PhG) - 12 regions (177-188)
    "PhG_L_6_1", "PhG_R_6_1", "PhG_L_6_2", "PhG_R_6_2",
    "PhG_L_6_3", "PhG_R_6_3", "PhG_L_6_4", "PhG_R_6_4",
    "PhG_L_6_5", "PhG_R_6_5", "PhG_L_6_6", "PhG_R_6_6",
    # Posterior Superior Temporal Sulcus (pSTS) - 4 regions (189-192)
    "pSTS_L_2_1", "pSTS_R_2_1", "pSTS_L_2_2", "pSTS_R_2_2",
    # Occipital Lobe - Lateral (LOcC) - 8 regions (193-200)
    "LOcC_L_4_1", "LOcC_R_4_1", "LOcC_L_4_2", "LOcC_R_4_2",
    "LOcC_L_4_3", "LOcC_R_4_3", "LOcC_L_4_4", "LOcC_R_4_4",
    # Occipital Lobe - Medial (MOcC) - 10 regions (201-210)
    "MOcC_L_5_1", "MOcC_R_5_1", "MOcC_L_5_2", "MOcC_R_5_2",
    "MOcC_L_5_3", "MOcC_R_5_3", "MOcC_L_5_4", "MOcC_R_5_4",
    "MOcC_L_5_5", "MOcC_R_5_5",
    # Amygdala (Amyg) - 4 regions (211-214)
    "Amyg_L_2_1", "Amyg_R_2_1", "Amyg_L_2_2", "Amyg_R_2_2",
    # Hippocampus (Hipp) - 4 regions (215-218)
    "Hipp_L_2_1", "Hipp_R_2_1", "Hipp_L_2_2", "Hipp_R_2_2",
    # Basal Ganglia - Striatum (BG) - 12 regions (219-230)
    "BG_L_6_1", "BG_R_6_1", "BG_L_6_2", "BG_R_6_2",
    "BG_L_6_3", "BG_R_6_3", "BG_L_6_4", "BG_R_6_4",
    "BG_L_6_5", "BG_R_6_5", "BG_L_6_6", "BG_R_6_6",
    # Thalamus (Tha) - 16 regions (231-246)
    "Tha_L_8_1", "Tha_R_8_1", "Tha_L_8_2", "Tha_R_8_2",
    "Tha_L_8_3", "Tha_R_8_3", "Tha_L_8_4", "Tha_R_8_4",
    "Tha_L_8_5", "Tha_R_8_5", "Tha_L_8_6", "Tha_R_8_6",
    "Tha_L_8_7", "Tha_R_8_7", "Tha_L_8_8", "Tha_R_8_8",
]

# FreeSurfer Desikan-Killiany Atlas (84 ROIs including subcortical)
# Matches MRtrix3 LUT ordering for compatibility
FREESURFER_DK_LABELS = [
    # Left hemisphere cortical (1-34)
    "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate",
    "ctx-lh-caudalmiddlefrontal", "ctx-lh-cuneus",
    "ctx-lh-entorhinal", "ctx-lh-fusiform",
    "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal",
    "ctx-lh-isthmuscingulate", "ctx-lh-lateraloccipital",
    "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal", "ctx-lh-middletemporal",
    "ctx-lh-parahippocampal", "ctx-lh-paracentral",
    "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis",
    "ctx-lh-parstriangularis", "ctx-lh-pericalcarine",
    "ctx-lh-postcentral", "ctx-lh-posteriorcingulate",
    "ctx-lh-precentral", "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate", "ctx-lh-rostralmiddlefrontal",
    "ctx-lh-superiorfrontal", "ctx-lh-superiorparietal",
    "ctx-lh-superiortemporal", "ctx-lh-supramarginal",
    "ctx-lh-frontalpole", "ctx-lh-temporalpole",
    "ctx-lh-transversetemporal", "ctx-lh-insula",
    # Right hemisphere cortical (35-68)
    "ctx-rh-bankssts", "ctx-rh-caudalanteriorcingulate",
    "ctx-rh-caudalmiddlefrontal", "ctx-rh-cuneus",
    "ctx-rh-entorhinal", "ctx-rh-fusiform",
    "ctx-rh-inferiorparietal", "ctx-rh-inferiortemporal",
    "ctx-rh-isthmuscingulate", "ctx-rh-lateraloccipital",
    "ctx-rh-lateralorbitofrontal", "ctx-rh-lingual",
    "ctx-rh-medialorbitofrontal", "ctx-rh-middletemporal",
    "ctx-rh-parahippocampal", "ctx-rh-paracentral",
    "ctx-rh-parsopercularis", "ctx-rh-parsorbitalis",
    "ctx-rh-parstriangularis", "ctx-rh-pericalcarine",
    "ctx-rh-postcentral", "ctx-rh-posteriorcingulate",
    "ctx-rh-precentral", "ctx-rh-precuneus",
    "ctx-rh-rostralanteriorcingulate", "ctx-rh-rostralmiddlefrontal",
    "ctx-rh-superiorfrontal", "ctx-rh-superiorparietal",
    "ctx-rh-superiortemporal", "ctx-rh-supramarginal",
    "ctx-rh-frontalpole", "ctx-rh-temporalpole",
    "ctx-rh-transversetemporal", "ctx-rh-insula",
    # Subcortical (69-84)
    "Left-Thalamus-Proper", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
    "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area", "Left-VentralDC",
    "Right-Thalamus-Proper", "Right-Caudate", "Right-Putamen", "Right-Pallidum",
    "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area", "Right-VentralDC",
]

# FreeSurfer Destrieux Atlas (164 ROIs)
# Uses nilearn's fetch_atlas_destrieux_2009
FREESURFER_DESTRIEUX_N_ROIS = 164


def get_atlas_labels(atlas_name: str) -> List[str]:
    """Get canonical labels for a standard atlas.

    Parameters
    ----------
    atlas_name : str
        Atlas name: 'brainnetome', 'freesurfer_dk', 'dk', 'destrieux'

    Returns
    -------
    List[str]
        List of ROI labels in canonical order

    Raises
    ------
    ValueError
        If atlas name is not recognized
    """
    atlas_lower = atlas_name.lower()

    if atlas_lower in ('brainnetome', 'bna', 'brainnetome246'):
        return BRAINNETOME_LABELS.copy()
    elif atlas_lower in ('freesurfer_dk', 'dk', 'desikan_killiany', 'desikan-killiany'):
        return FREESURFER_DK_LABELS.copy()
    elif atlas_lower in ('destrieux', 'freesurfer_destrieux', 'a2009s'):
        # Destrieux labels come from nilearn at runtime
        try:
            from nilearn.datasets import fetch_atlas_destrieux_2009
            atlas = fetch_atlas_destrieux_2009(legacy_format=False)
            return list(atlas['labels'])
        except ImportError:
            raise ValueError(
                "nilearn required for Destrieux atlas labels. "
                "Install with: pip install nilearn"
            )
    else:
        raise ValueError(
            f"Unknown atlas: {atlas_name}. "
            f"Supported: brainnetome, freesurfer_dk, destrieux"
        )


def get_atlas_info(atlas_name: str) -> Dict:
    """Get information about an atlas.

    Parameters
    ----------
    atlas_name : str
        Atlas name

    Returns
    -------
    dict
        Atlas information including n_rois, description, reference
    """
    atlas_lower = atlas_name.lower()

    info = {
        'brainnetome': {
            'n_rois': 246,
            'description': 'Brainnetome Atlas - connectivity-based parcellation',
            'reference': 'Fan et al. (2016) Cerebral Cortex',
            'hemispheres': 'bilateral',
            'includes_subcortical': True,
        },
        'freesurfer_dk': {
            'n_rois': 84,
            'description': 'Desikan-Killiany Atlas - FreeSurfer cortical + subcortical',
            'reference': 'Desikan et al. (2006) NeuroImage',
            'hemispheres': 'bilateral',
            'includes_subcortical': True,
        },
        'destrieux': {
            'n_rois': 164,
            'description': 'Destrieux Atlas - FreeSurfer sulcogyral parcellation',
            'reference': 'Destrieux et al. (2010) NeuroImage',
            'hemispheres': 'bilateral',
            'includes_subcortical': False,
        },
    }

    # Normalize name
    if atlas_lower in ('bna', 'brainnetome246'):
        atlas_lower = 'brainnetome'
    elif atlas_lower in ('dk', 'desikan_killiany', 'desikan-killiany'):
        atlas_lower = 'freesurfer_dk'
    elif atlas_lower in ('freesurfer_destrieux', 'a2009s'):
        atlas_lower = 'destrieux'

    if atlas_lower not in info:
        raise ValueError(f"Unknown atlas: {atlas_name}")

    return info[atlas_lower]


def save_labels_file(
    labels: List[str],
    output_path: Path,
    atlas_name: Optional[str] = None,
) -> Path:
    """Save labels to a text file.

    Parameters
    ----------
    labels : List[str]
        List of ROI labels
    output_path : Path
        Output file path
    atlas_name : str, optional
        Atlas name to include in header

    Returns
    -------
    Path
        Path to saved file
    """
    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        if atlas_name:
            f.write(f"# {atlas_name} atlas labels ({len(labels)} ROIs)\n")
        for label in labels:
            f.write(f"{label}\n")

    logger.info(f"Saved {len(labels)} labels to {output_path}")
    return output_path


def load_labels_file(filepath: Path) -> List[str]:
    """Load labels from a text file.

    Parameters
    ----------
    filepath : Path
        Path to labels file

    Returns
    -------
    List[str]
        List of labels
    """
    filepath = Path(filepath)
    labels = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                labels.append(line)

    return labels


def verify_labels_consistency(
    labels1: List[str],
    labels2: List[str],
) -> Tuple[bool, Dict]:
    """Verify that two label lists are consistent.

    Parameters
    ----------
    labels1 : List[str]
        First label list
    labels2 : List[str]
        Second label list

    Returns
    -------
    Tuple[bool, Dict]
        (consistent, details)
    """
    set1 = set(labels1)
    set2 = set(labels2)

    details = {
        'n_labels1': len(labels1),
        'n_labels2': len(labels2),
        'count_match': len(labels1) == len(labels2),
        'order_match': labels1 == labels2,
        'set_match': set1 == set2,
        'only_in_1': list(set1 - set2),
        'only_in_2': list(set2 - set1),
    }

    # Check for order mismatches even if sets match
    if details['set_match'] and not details['order_match']:
        mismatches = []
        for i, (l1, l2) in enumerate(zip(labels1, labels2)):
            if l1 != l2:
                mismatches.append({'index': i, 'label1': l1, 'label2': l2})
        details['order_mismatches'] = mismatches[:10]  # First 10

    consistent = details['count_match'] and details['set_match']
    return consistent, details


def generate_canonical_labels_file(
    atlas_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate a canonical labels file for an atlas.

    Parameters
    ----------
    atlas_name : str
        Atlas name
    output_dir : Path, optional
        Output directory (default: current directory)

    Returns
    -------
    Path
        Path to generated file
    """
    labels = get_atlas_labels(atlas_name)
    info = get_atlas_info(atlas_name)

    if output_dir is None:
        output_dir = Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize filename
    atlas_lower = atlas_name.lower()
    if atlas_lower in ('brainnetome', 'bna', 'brainnetome246'):
        filename = 'brainnetome_labels.txt'
    elif atlas_lower in ('freesurfer_dk', 'dk', 'desikan_killiany'):
        filename = 'freesurfer_dk_labels.txt'
    elif atlas_lower in ('destrieux', 'freesurfer_destrieux', 'a2009s'):
        filename = 'destrieux_labels.txt'
    else:
        filename = f'{atlas_name}_labels.txt'

    output_path = output_dir / filename
    return save_labels_file(labels, output_path, atlas_name)
