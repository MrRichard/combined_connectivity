"""
HTML Report Generator for Connectivity Analysis.

Generates comprehensive HTML reports combining QC visualizations,
metrics, and processing information for both DWI and fMRI pipelines.
"""

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def _encode_image(image_path: Path) -> str:
    """Encode an image file as base64 for embedding in HTML."""
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    suffix = image_path.suffix.lower()
    mime = 'image/png' if suffix == '.png' else 'image/jpeg'
    return f"data:{mime};base64,{data}"


def _format_value(value) -> str:
    """Format a metric value for display."""
    if isinstance(value, float):
        if abs(value) < 0.001:
            return f"{value:.2e}"
        return f"{value:.4f}"
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, bool):
        return "Yes" if value else "No"
    elif value is None:
        return "N/A"
    return str(value)


# CSS styles for the report
REPORT_CSS = """
<style>
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --success-color: #27ae60;
    --warning-color: #f39c12;
    --danger-color: #e74c3c;
    --light-gray: #ecf0f1;
    --dark-gray: #7f8c8d;
}

* {
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f6fa;
}

.header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    padding: 30px;
    border-radius: 10px;
    margin-bottom: 30px;
}

.header h1 {
    margin: 0 0 10px 0;
    font-size: 2em;
}

.header .subtitle {
    opacity: 0.9;
    font-size: 1.1em;
}

.metadata {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin-top: 20px;
}

.metadata-item {
    background: rgba(255,255,255,0.1);
    padding: 10px 15px;
    border-radius: 5px;
}

.metadata-label {
    font-size: 0.85em;
    opacity: 0.8;
}

.metadata-value {
    font-weight: 600;
}

.section {
    background: white;
    border-radius: 10px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.section h2 {
    color: var(--primary-color);
    border-bottom: 2px solid var(--light-gray);
    padding-bottom: 10px;
    margin-top: 0;
}

.section h3 {
    color: var(--secondary-color);
    margin-top: 25px;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
}

.metric-card {
    background: var(--light-gray);
    padding: 15px;
    border-radius: 8px;
    text-align: center;
}

.metric-value {
    font-size: 1.5em;
    font-weight: 700;
    color: var(--primary-color);
}

.metric-label {
    font-size: 0.85em;
    color: var(--dark-gray);
    margin-top: 5px;
}

.image-container {
    text-align: center;
    margin: 20px 0;
}

.image-container img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.image-caption {
    color: var(--dark-gray);
    font-size: 0.9em;
    margin-top: 10px;
}

.image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid var(--light-gray);
}

th {
    background: var(--light-gray);
    font-weight: 600;
    color: var(--primary-color);
}

tr:hover {
    background: #f8f9fa;
}

.status-pass {
    color: var(--success-color);
    font-weight: 600;
}

.status-warn {
    color: var(--warning-color);
    font-weight: 600;
}

.status-fail {
    color: var(--danger-color);
    font-weight: 600;
}

.warning-box {
    background: #fff3cd;
    border-left: 4px solid var(--warning-color);
    padding: 15px 20px;
    border-radius: 0 8px 8px 0;
    margin: 15px 0;
}

.info-box {
    background: #e7f3ff;
    border-left: 4px solid var(--secondary-color);
    padding: 15px 20px;
    border-radius: 0 8px 8px 0;
    margin: 15px 0;
}

.footer {
    text-align: center;
    color: var(--dark-gray);
    font-size: 0.9em;
    margin-top: 30px;
    padding: 20px;
}

.footer a {
    color: var(--secondary-color);
    text-decoration: none;
}

@media (max-width: 768px) {
    .image-grid {
        grid-template-columns: 1fr;
    }
    .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
</style>
"""


class HTMLReportGenerator:
    """Generate HTML reports for connectivity analysis QC.

    Creates comprehensive, self-contained HTML reports that include
    embedded images, metrics tables, and processing metadata.

    Parameters
    ----------
    output_dir : Path or str
        Directory for output files
    embed_images : bool
        If True, embed images as base64 (default True for portability)

    Examples
    --------
    >>> from connectivity_shared import HTMLReportGenerator
    >>> report = HTMLReportGenerator(output_dir="./reports")
    >>> report.add_metadata(subject_id="sub-01", modality="fmri")
    >>> report.add_metrics("global", {"efficiency": 0.65, "clustering": 0.45})
    >>> report.add_image("heatmap", "./qc/heatmap.png", "Connectivity Matrix")
    >>> report.generate("sub-01_qc_report.html")
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        embed_images: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embed_images = embed_images

        # Report content storage
        self.metadata: Dict[str, str] = {}
        self.metrics: Dict[str, Dict] = {}
        self.images: List[Dict] = []
        self.tables: List[Dict] = []
        self.warnings: List[str] = []
        self.info_messages: List[str] = []
        self.custom_sections: List[Dict] = []

    def add_metadata(self, **kwargs) -> None:
        """Add metadata fields to the report.

        Parameters
        ----------
        **kwargs
            Key-value pairs for metadata (e.g., subject_id, session, modality)
        """
        self.metadata.update(kwargs)

    def add_metrics(
        self,
        category: str,
        metrics: Dict,
        display_names: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add metrics to the report.

        Parameters
        ----------
        category : str
            Category name (e.g., 'global', 'network', 'qc')
        metrics : dict
            Dictionary of metric name -> value
        display_names : dict, optional
            Mapping of metric names to display labels
        """
        if category not in self.metrics:
            self.metrics[category] = {}
        self.metrics[category].update(metrics)

        if display_names:
            if '_display_names' not in self.metrics[category]:
                self.metrics[category]['_display_names'] = {}
            self.metrics[category]['_display_names'].update(display_names)

    def add_image(
        self,
        name: str,
        path: Union[str, Path],
        caption: Optional[str] = None,
        section: Optional[str] = None,
    ) -> None:
        """Add an image to the report.

        Parameters
        ----------
        name : str
            Image identifier
        path : str or Path
            Path to image file
        caption : str, optional
            Caption to display under image
        section : str, optional
            Section to place image in (default: 'Images')
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Image not found: {path}")
            return

        self.images.append({
            'name': name,
            'path': path,
            'caption': caption or name,
            'section': section or 'Images',
        })

    def add_table(
        self,
        title: str,
        data: List[Dict],
        columns: Optional[List[str]] = None,
        section: Optional[str] = None,
    ) -> None:
        """Add a table to the report.

        Parameters
        ----------
        title : str
            Table title
        data : list of dict
            Table data as list of row dictionaries
        columns : list of str, optional
            Column order (default: keys from first row)
        section : str, optional
            Section to place table in
        """
        if not data:
            return

        if columns is None:
            columns = list(data[0].keys())

        self.tables.append({
            'title': title,
            'data': data,
            'columns': columns,
            'section': section or 'Tables',
        })

    def add_warning(self, message: str) -> None:
        """Add a warning message to the report."""
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        """Add an info message to the report."""
        self.info_messages.append(message)

    def add_custom_section(
        self,
        title: str,
        content: str,
        order: int = 100,
    ) -> None:
        """Add a custom HTML section.

        Parameters
        ----------
        title : str
            Section title
        content : str
            HTML content for the section
        order : int
            Order in report (lower = earlier)
        """
        self.custom_sections.append({
            'title': title,
            'content': content,
            'order': order,
        })

    def _generate_header(self) -> str:
        """Generate the report header HTML."""
        title = self.metadata.get('title', 'Connectivity Analysis Report')
        subtitle = self.metadata.get('subtitle', '')

        # Build metadata display
        meta_items = []
        display_fields = [
            ('subject_id', 'Subject'),
            ('session_id', 'Session'),
            ('modality', 'Modality'),
            ('atlas', 'Atlas'),
            ('n_rois', 'ROIs'),
            ('processing_date', 'Processed'),
        ]

        for key, label in display_fields:
            if key in self.metadata:
                meta_items.append(f'''
                    <div class="metadata-item">
                        <div class="metadata-label">{label}</div>
                        <div class="metadata-value">{self.metadata[key]}</div>
                    </div>
                ''')

        return f'''
        <div class="header">
            <h1>{title}</h1>
            <div class="subtitle">{subtitle}</div>
            <div class="metadata">
                {''.join(meta_items)}
            </div>
        </div>
        '''

    def _generate_metrics_section(self) -> str:
        """Generate the metrics section HTML."""
        if not self.metrics:
            return ''

        sections = []
        for category, metrics in self.metrics.items():
            if category.startswith('_'):
                continue

            display_names = metrics.get('_display_names', {})

            cards = []
            for name, value in metrics.items():
                if name.startswith('_'):
                    continue
                label = display_names.get(name, name.replace('_', ' ').title())
                cards.append(f'''
                    <div class="metric-card">
                        <div class="metric-value">{_format_value(value)}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                ''')

            sections.append(f'''
                <h3>{category.replace('_', ' ').title()}</h3>
                <div class="metrics-grid">
                    {''.join(cards)}
                </div>
            ''')

        return f'''
        <div class="section">
            <h2>Metrics</h2>
            {''.join(sections)}
        </div>
        '''

    def _generate_images_section(self) -> str:
        """Generate the images section HTML."""
        if not self.images:
            return ''

        # Group images by section
        sections: Dict[str, List] = {}
        for img in self.images:
            section = img['section']
            if section not in sections:
                sections[section] = []
            sections[section].append(img)

        html_sections = []
        for section_name, images in sections.items():
            image_html = []
            for img in images:
                if self.embed_images:
                    src = _encode_image(img['path'])
                else:
                    src = str(img['path'])

                image_html.append(f'''
                    <div class="image-container">
                        <img src="{src}" alt="{img['name']}">
                        <div class="image-caption">{img['caption']}</div>
                    </div>
                ''')

            html_sections.append(f'''
                <div class="section">
                    <h2>{section_name}</h2>
                    <div class="image-grid">
                        {''.join(image_html)}
                    </div>
                </div>
            ''')

        return ''.join(html_sections)

    def _generate_tables_section(self) -> str:
        """Generate the tables section HTML."""
        if not self.tables:
            return ''

        # Group tables by section
        sections: Dict[str, List] = {}
        for table in self.tables:
            section = table['section']
            if section not in sections:
                sections[section] = []
            sections[section].append(table)

        html_sections = []
        for section_name, tables in sections.items():
            table_html = []
            for table in tables:
                headers = ''.join(f'<th>{col}</th>' for col in table['columns'])
                rows = []
                for row in table['data']:
                    cells = ''.join(
                        f'<td>{_format_value(row.get(col, ""))}</td>'
                        for col in table['columns']
                    )
                    rows.append(f'<tr>{cells}</tr>')

                table_html.append(f'''
                    <h3>{table['title']}</h3>
                    <table>
                        <thead><tr>{headers}</tr></thead>
                        <tbody>{''.join(rows)}</tbody>
                    </table>
                ''')

            html_sections.append(f'''
                <div class="section">
                    <h2>{section_name}</h2>
                    {''.join(table_html)}
                </div>
            ''')

        return ''.join(html_sections)

    def _generate_warnings(self) -> str:
        """Generate warnings and info boxes."""
        html = []

        for warning in self.warnings:
            html.append(f'<div class="warning-box">{warning}</div>')

        for info in self.info_messages:
            html.append(f'<div class="info-box">{info}</div>')

        if html:
            return f'''
            <div class="section">
                <h2>Notices</h2>
                {''.join(html)}
            </div>
            '''
        return ''

    def _generate_footer(self) -> str:
        """Generate the report footer."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f'''
        <div class="footer">
            <p>Generated on {timestamp}</p>
            <p>Created with <a href="https://github.com/anthropics/connectivity-shared">connectivity-shared</a></p>
        </div>
        '''

    def generate(
        self,
        filename: str = "qc_report.html",
        title: Optional[str] = None,
    ) -> Path:
        """Generate the HTML report.

        Parameters
        ----------
        filename : str
            Output filename
        title : str, optional
            Report title (overrides metadata title)

        Returns
        -------
        Path
            Path to generated report
        """
        if title:
            self.metadata['title'] = title

        if 'title' not in self.metadata:
            subject = self.metadata.get('subject_id', 'Unknown')
            modality = self.metadata.get('modality', 'connectivity')
            self.metadata['title'] = f"QC Report: {subject} - {modality}"

        # Sort custom sections by order
        custom_html = ''
        if self.custom_sections:
            sorted_sections = sorted(self.custom_sections, key=lambda x: x['order'])
            for section in sorted_sections:
                custom_html += f'''
                <div class="section">
                    <h2>{section['title']}</h2>
                    {section['content']}
                </div>
                '''

        # Assemble the report
        html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.metadata['title']}</title>
    {REPORT_CSS}
</head>
<body>
    {self._generate_header()}
    {self._generate_warnings()}
    {self._generate_metrics_section()}
    {self._generate_images_section()}
    {self._generate_tables_section()}
    {custom_html}
    {self._generate_footer()}
</body>
</html>
'''

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"Generated report: {output_path}")
        return output_path


def generate_connectivity_report(
    matrix_path: Path,
    metrics_path: Optional[Path] = None,
    qc_images: Optional[Dict[str, Path]] = None,
    output_path: Optional[Path] = None,
    subject_id: str = "unknown",
    session_id: Optional[str] = None,
    modality: str = "connectivity",
    atlas: str = "unknown",
    **kwargs,
) -> Path:
    """Generate a complete connectivity QC report.

    Convenience function that loads data and generates a full report.

    Parameters
    ----------
    matrix_path : Path
        Path to connectivity matrix file
    metrics_path : Path, optional
        Path to graph metrics JSON file
    qc_images : dict, optional
        Dictionary mapping image names to paths
    output_path : Path, optional
        Output path (default: same dir as matrix)
    subject_id : str
        Subject identifier
    session_id : str, optional
        Session identifier
    modality : str
        Modality (dwi or fmri)
    atlas : str
        Atlas name
    **kwargs
        Additional metadata fields

    Returns
    -------
    Path
        Path to generated report
    """
    matrix_path = Path(matrix_path)
    if output_path is None:
        output_path = matrix_path.parent / f"{subject_id}_{modality}_qc_report.html"
    output_path = Path(output_path)

    report = HTMLReportGenerator(output_dir=output_path.parent)

    # Add metadata
    report.add_metadata(
        subject_id=subject_id,
        modality=modality.upper(),
        atlas=atlas,
        processing_date=datetime.now().strftime('%Y-%m-%d'),
        **kwargs,
    )
    if session_id:
        report.add_metadata(session_id=session_id)

    # Load and add metrics
    if metrics_path and Path(metrics_path).exists():
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        if 'global_metrics' in metrics_data:
            report.add_metrics('global', metrics_data['global_metrics'])

        if 'metadata' in metrics_data:
            report.add_metadata(**metrics_data['metadata'])

    # Add QC images
    if qc_images:
        for name, path in qc_images.items():
            if Path(path).exists():
                report.add_image(
                    name=name,
                    path=path,
                    caption=name.replace('_', ' ').title(),
                    section='Quality Control',
                )

    return report.generate(filename=output_path.name)
