"""
Unified QC visualization for connectivity pipelines.

Provides standardized visualization functions for:
- Atlas overlays on anatomical/functional data
- Registration quality checks
- Connectivity matrix heatmaps
- Network visualization

Both DWI and fMRI pipelines can use these to produce consistent QC images.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - visualization functions disabled")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from nilearn import plotting as niplot
    from nilearn import image as niimage
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    logger.info("nilearn not available - using basic visualization")


def check_viz_dependencies():
    """Check if visualization dependencies are available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
    return True


class QCVisualizer:
    """
    Generate standardized QC images for connectivity analysis.

    Example:
        >>> viz = QCVisualizer(output_dir="/path/to/qc")
        >>> viz.create_atlas_overlay(background_img, atlas_img, "sub-01_atlas_overlay.png")
        >>> viz.create_connectivity_heatmap(matrix, labels, "sub-01_connectivity.png")
    """

    # Standard figure settings
    DPI = 150
    FIGSIZE_OVERLAY = (12, 8)
    FIGSIZE_MATRIX = (10, 10)
    FIGSIZE_SUMMARY = (16, 12)

    # Color settings
    ATLAS_CMAP = 'tab20'
    MATRIX_CMAP = 'RdBu_r'
    BACKGROUND_COLOR = 'black'

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        dpi: int = 150,
    ):
        """
        Initialize QC visualizer.

        Args:
            output_dir: Default output directory for images
            dpi: Resolution for output images
        """
        check_viz_dependencies()
        self.output_dir = Path(output_dir) if output_dir else None
        self.dpi = dpi

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_path(self, filename: str) -> Path:
        """Get full output path for a file."""
        if self.output_dir:
            return self.output_dir / filename
        return Path(filename)

    def create_atlas_overlay(
        self,
        background: Union[str, Path, 'nib.Nifti1Image'],
        atlas: Union[str, Path, 'nib.Nifti1Image'],
        output_filename: str,
        title: Optional[str] = None,
        alpha: float = 0.6,
        slice_coords: Optional[Dict[str, int]] = None,
    ) -> Path:
        """
        Create atlas overlay on background image.

        Args:
            background: Background image (anatomical or mean functional)
            atlas: Atlas/parcellation image
            output_filename: Output filename
            title: Optional title for the figure
            alpha: Transparency for atlas overlay (0-1)
            slice_coords: Optional dict with 'x', 'y', 'z' slice coordinates

        Returns:
            Path to saved image
        """
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel required for atlas overlay")

        # Load images if paths provided
        if isinstance(background, (str, Path)):
            background = nib.load(str(background))
        if isinstance(atlas, (str, Path)):
            atlas = nib.load(str(atlas))

        output_path = self._get_output_path(output_filename)

        # Use nilearn if available for nice visualization
        if NILEARN_AVAILABLE:
            return self._create_atlas_overlay_nilearn(
                background, atlas, output_path, title, alpha
            )
        else:
            return self._create_atlas_overlay_basic(
                background, atlas, output_path, title, alpha, slice_coords
            )

    def _create_atlas_overlay_nilearn(
        self,
        background: 'nib.Nifti1Image',
        atlas: 'nib.Nifti1Image',
        output_path: Path,
        title: Optional[str],
        alpha: float,
    ) -> Path:
        """Create atlas overlay using nilearn plotting."""
        fig, axes = plt.subplots(2, 3, figsize=self.FIGSIZE_OVERLAY)
        fig.patch.set_facecolor('black')

        display_modes = ['x', 'y', 'z']

        # Row 1: Background only
        for i, mode in enumerate(display_modes):
            ax = axes[0, i]
            ax.set_facecolor('black')
            try:
                display = niplot.plot_anat(
                    background,
                    display_mode=mode,
                    cut_coords=1,
                    axes=ax,
                    annotate=False,
                )
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', color='white')
                ax.axis('off')

        # Row 2: With atlas overlay
        for i, mode in enumerate(display_modes):
            ax = axes[1, i]
            ax.set_facecolor('black')
            try:
                display = niplot.plot_roi(
                    atlas,
                    bg_img=background,
                    display_mode=mode,
                    cut_coords=1,
                    axes=ax,
                    alpha=alpha,
                    annotate=False,
                )
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', color='white')
                ax.axis('off')

        if title:
            fig.suptitle(title, color='white', fontsize=14, y=0.98)

        # Add row labels
        fig.text(0.02, 0.75, 'Background', rotation=90, va='center', color='white', fontsize=12)
        fig.text(0.02, 0.25, 'Atlas Overlay', rotation=90, va='center', color='white', fontsize=12)

        plt.tight_layout(rect=[0.03, 0, 1, 0.96])
        plt.savefig(output_path, dpi=self.dpi, facecolor='black', bbox_inches='tight')
        plt.close()

        logger.info(f"Saved atlas overlay: {output_path}")
        return output_path

    def _create_atlas_overlay_basic(
        self,
        background: 'nib.Nifti1Image',
        atlas: 'nib.Nifti1Image',
        output_path: Path,
        title: Optional[str],
        alpha: float,
        slice_coords: Optional[Dict[str, int]],
    ) -> Path:
        """Create atlas overlay using basic matplotlib (no nilearn)."""
        bg_data = background.get_fdata()
        atlas_data = atlas.get_fdata()

        # Get middle slices if not specified
        if slice_coords is None:
            slice_coords = {
                'x': bg_data.shape[0] // 2,
                'y': bg_data.shape[1] // 2,
                'z': bg_data.shape[2] // 2,
            }

        fig, axes = plt.subplots(2, 3, figsize=self.FIGSIZE_OVERLAY)
        fig.patch.set_facecolor('black')

        slices = [
            ('Sagittal', bg_data[slice_coords['x'], :, :].T, atlas_data[slice_coords['x'], :, :].T),
            ('Coronal', bg_data[:, slice_coords['y'], :].T, atlas_data[:, slice_coords['y'], :].T),
            ('Axial', bg_data[:, :, slice_coords['z']].T, atlas_data[:, :, slice_coords['z']].T),
        ]

        for i, (name, bg_slice, atlas_slice) in enumerate(slices):
            # Row 1: Background only
            axes[0, i].imshow(bg_slice, cmap='gray', origin='lower')
            axes[0, i].set_title(name, color='white')
            axes[0, i].axis('off')

            # Row 2: With overlay
            axes[1, i].imshow(bg_slice, cmap='gray', origin='lower')
            masked_atlas = np.ma.masked_where(atlas_slice == 0, atlas_slice)
            axes[1, i].imshow(masked_atlas, cmap=self.ATLAS_CMAP, alpha=alpha, origin='lower')
            axes[1, i].axis('off')

        if title:
            fig.suptitle(title, color='white', fontsize=14)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, facecolor='black', bbox_inches='tight')
        plt.close()

        logger.info(f"Saved atlas overlay: {output_path}")
        return output_path

    def create_connectivity_heatmap(
        self,
        matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        output_filename: str = "connectivity_matrix.png",
        title: Optional[str] = None,
        show_labels: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        symmetric_cmap: bool = True,
    ) -> Path:
        """
        Create connectivity matrix heatmap.

        Args:
            matrix: Connectivity matrix (n_rois x n_rois)
            labels: Optional ROI labels
            output_filename: Output filename
            title: Optional title
            show_labels: Whether to show ROI labels on axes
            vmin, vmax: Color scale limits
            symmetric_cmap: If True, center colormap at 0

        Returns:
            Path to saved image
        """
        output_path = self._get_output_path(output_filename)

        fig, ax = plt.subplots(figsize=self.FIGSIZE_MATRIX)

        # Determine color limits
        if symmetric_cmap and vmin is None and vmax is None:
            abs_max = np.nanmax(np.abs(matrix))
            vmin, vmax = -abs_max, abs_max

        im = ax.imshow(matrix, cmap=self.MATRIX_CMAP, vmin=vmin, vmax=vmax)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Connection Strength', fontsize=10)

        # Add labels if requested and not too many
        if show_labels and labels and len(labels) <= 50:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
        elif labels:
            # Just show tick marks at intervals
            n_ticks = min(10, len(labels))
            tick_indices = np.linspace(0, len(labels) - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_yticks(tick_indices)
            ax.set_xticklabels([labels[i] for i in tick_indices], rotation=90, fontsize=8)
            ax.set_yticklabels([labels[i] for i in tick_indices], fontsize=8)

        if title:
            ax.set_title(title, fontsize=12)
        ax.set_xlabel('ROI', fontsize=10)
        ax.set_ylabel('ROI', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved connectivity heatmap: {output_path}")
        return output_path

    def create_edge_histogram(
        self,
        matrix: np.ndarray,
        output_filename: str = "edge_distribution.png",
        title: Optional[str] = None,
        bins: int = 50,
    ) -> Path:
        """
        Create histogram of edge weights.

        Args:
            matrix: Connectivity matrix
            output_filename: Output filename
            title: Optional title
            bins: Number of histogram bins

        Returns:
            Path to saved image
        """
        output_path = self._get_output_path(output_filename)

        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(matrix, k=1)
        values = matrix[triu_indices]
        nonzero_values = values[values != 0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: All edges
        axes[0].hist(values, bins=bins, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Edge Weight')
        axes[0].set_ylabel('Count')
        axes[0].set_title('All Edges (including zeros)')
        axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

        # Right: Non-zero edges only
        if len(nonzero_values) > 0:
            axes[1].hist(nonzero_values, bins=bins, edgecolor='black', alpha=0.7, color='orange')
            axes[1].set_xlabel('Edge Weight')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'Non-zero Edges (n={len(nonzero_values)})')

            # Add statistics
            stats_text = f'Mean: {np.mean(nonzero_values):.3f}\nStd: {np.std(nonzero_values):.3f}\nMedian: {np.median(nonzero_values):.3f}'
            axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[1].text(0.5, 0.5, 'No non-zero edges', ha='center', va='center')
            axes[1].set_title('Non-zero Edges')

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved edge histogram: {output_path}")
        return output_path

    def create_degree_distribution(
        self,
        matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        output_filename: str = "degree_distribution.png",
        title: Optional[str] = None,
    ) -> Path:
        """
        Create degree distribution plot.

        Args:
            matrix: Connectivity matrix
            labels: Optional ROI labels
            output_filename: Output filename
            title: Optional title

        Returns:
            Path to saved image
        """
        output_path = self._get_output_path(output_filename)

        # Calculate degree (binary) and strength (weighted)
        binary_matrix = (matrix != 0).astype(int)
        degree = np.sum(binary_matrix, axis=1)
        strength = np.sum(np.abs(matrix), axis=1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Left: Degree histogram
        axes[0].hist(degree, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Degree')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Degree Distribution')
        axes[0].axvline(x=np.mean(degree), color='red', linestyle='--', label=f'Mean: {np.mean(degree):.1f}')
        axes[0].legend()

        # Middle: Strength histogram
        axes[1].hist(strength, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel('Strength')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Strength Distribution')
        axes[1].axvline(x=np.mean(strength), color='red', linestyle='--', label=f'Mean: {np.mean(strength):.1f}')
        axes[1].legend()

        # Right: Top 10 hubs by degree
        top_indices = np.argsort(degree)[-10:][::-1]
        top_degrees = degree[top_indices]
        if labels:
            top_labels = [labels[i] if i < len(labels) else f'ROI_{i}' for i in top_indices]
        else:
            top_labels = [f'ROI_{i}' for i in top_indices]

        y_pos = np.arange(len(top_indices))
        axes[2].barh(y_pos, top_degrees, color='green', alpha=0.7)
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(top_labels, fontsize=8)
        axes[2].set_xlabel('Degree')
        axes[2].set_title('Top 10 Hubs (by degree)')
        axes[2].invert_yaxis()

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved degree distribution: {output_path}")
        return output_path

    def create_qc_summary(
        self,
        matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        metrics: Optional[Dict] = None,
        output_filename: str = "qc_summary.png",
        title: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> Path:
        """
        Create comprehensive QC summary figure.

        Args:
            matrix: Connectivity matrix
            labels: Optional ROI labels
            metrics: Optional dict of graph metrics to display
            output_filename: Output filename
            title: Optional title
            modality: 'dwi' or 'fmri' for modality-specific formatting

        Returns:
            Path to saved image
        """
        output_path = self._get_output_path(output_filename)

        fig = plt.figure(figsize=self.FIGSIZE_SUMMARY)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Top left: Connectivity matrix
        ax1 = fig.add_subplot(gs[0, 0])
        abs_max = np.nanmax(np.abs(matrix))
        im = ax1.imshow(matrix, cmap=self.MATRIX_CMAP, vmin=-abs_max, vmax=abs_max)
        ax1.set_title('Connectivity Matrix')
        ax1.set_xlabel('ROI')
        ax1.set_ylabel('ROI')
        plt.colorbar(im, ax=ax1, shrink=0.8)

        # Top middle: Edge distribution
        ax2 = fig.add_subplot(gs[0, 1])
        triu_indices = np.triu_indices_from(matrix, k=1)
        values = matrix[triu_indices]
        nonzero = values[values != 0]
        if len(nonzero) > 0:
            ax2.hist(nonzero, bins=30, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Edge Weight')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Edge Distribution (n={len(nonzero)})')

        # Top right: Degree distribution
        ax3 = fig.add_subplot(gs[0, 2])
        degree = np.sum(matrix != 0, axis=1)
        ax3.hist(degree, bins=20, edgecolor='black', alpha=0.7, color='orange')
        ax3.set_xlabel('Degree')
        ax3.set_ylabel('Count')
        ax3.set_title('Degree Distribution')
        ax3.axvline(x=np.mean(degree), color='red', linestyle='--')

        # Bottom left: Metrics table
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')
        if metrics:
            # Select key metrics to display
            display_metrics = [
                ('Nodes', metrics.get('n_nodes', 'N/A')),
                ('Edges', metrics.get('total_connections', 'N/A')),
                ('Density', f"{metrics.get('connection_density', 0):.4f}"),
                ('Clustering', f"{metrics.get('mean_clustering_coefficient', 0):.4f}"),
                ('Global Eff.', f"{metrics.get('global_efficiency', 0):.4f}"),
                ('Local Eff.', f"{metrics.get('local_efficiency', 0):.4f}"),
                ('Small-world', f"{metrics.get('small_worldness', 0):.4f}"),
                ('Modularity', f"{metrics.get('modularity', 0):.4f}"),
            ]

            table_data = [[name, str(val)] for name, val in display_metrics]
            table = ax4.table(
                cellText=table_data,
                colLabels=['Metric', 'Value'],
                loc='center',
                cellLoc='left',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax4.set_title('Graph Metrics', fontsize=12, pad=20)
        else:
            ax4.text(0.5, 0.5, 'No metrics provided', ha='center', va='center')

        # Bottom middle: Network density by node
        ax5 = fig.add_subplot(gs[1, 1])
        strength = np.sum(np.abs(matrix), axis=1)
        ax5.bar(range(len(strength)), strength, alpha=0.7)
        ax5.set_xlabel('Node Index')
        ax5.set_ylabel('Node Strength')
        ax5.set_title('Node Strength Distribution')

        # Bottom right: Info text
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        info_text = [
            f"Matrix Shape: {matrix.shape}",
            f"Non-zero edges: {np.sum(matrix != 0) // 2}",
            f"Mean edge weight: {np.mean(nonzero):.4f}" if len(nonzero) > 0 else "Mean edge weight: N/A",
            f"Max edge weight: {np.max(np.abs(matrix)):.4f}",
            f"Mean degree: {np.mean(degree):.1f}",
            f"Max degree: {np.max(degree)}",
        ]
        if modality:
            info_text.insert(0, f"Modality: {modality.upper()}")

        ax6.text(0.1, 0.9, '\n'.join(info_text), transform=ax6.transAxes,
                verticalalignment='top', fontsize=10, family='monospace')
        ax6.set_title('Summary Statistics', fontsize=12)

        if title:
            fig.suptitle(title, fontsize=14, y=0.98)

        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved QC summary: {output_path}")
        return output_path


def create_registration_check(
    fixed_img: Union[str, Path],
    moving_img: Union[str, Path],
    output_filename: str,
    output_dir: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> Path:
    """
    Create registration quality check image showing overlay of two images.

    Args:
        fixed_img: Path to fixed/reference image
        moving_img: Path to moving/registered image
        output_filename: Output filename
        output_dir: Output directory
        title: Optional title

    Returns:
        Path to saved image
    """
    check_viz_dependencies()

    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel required for registration check")

    fixed = nib.load(str(fixed_img))
    moving = nib.load(str(moving_img))

    output_path = Path(output_dir) / output_filename if output_dir else Path(output_filename)

    if NILEARN_AVAILABLE:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.patch.set_facecolor('black')

        for i, mode in enumerate(['x', 'y', 'z']):
            # Row 1: Fixed image
            niplot.plot_anat(fixed, display_mode=mode, cut_coords=1,
                           axes=axes[0, i], annotate=False, title=f'Fixed ({mode})')
            # Row 2: Edge overlay
            niplot.plot_anat(fixed, display_mode=mode, cut_coords=1,
                           axes=axes[1, i], annotate=False)
            try:
                niplot.plot_roi(moving, bg_img=fixed, display_mode=mode, cut_coords=1,
                              axes=axes[1, i], alpha=0.5, annotate=False)
            except:
                pass

        if title:
            fig.suptitle(title, color='white', fontsize=14)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor='black', bbox_inches='tight')
        plt.close()
    else:
        # Basic matplotlib version
        fixed_data = fixed.get_fdata()
        moving_data = moving.get_fdata()

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        mid = [s // 2 for s in fixed_data.shape]

        slices = [
            fixed_data[mid[0], :, :].T,
            fixed_data[:, mid[1], :].T,
            fixed_data[:, :, mid[2]].T,
        ]

        for i, sl in enumerate(slices):
            axes[0, i].imshow(sl, cmap='gray', origin='lower')
            axes[0, i].axis('off')

            axes[1, i].imshow(sl, cmap='gray', origin='lower')
            # Add moving image contours if shapes match
            axes[1, i].axis('off')

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Saved registration check: {output_path}")
    return output_path
