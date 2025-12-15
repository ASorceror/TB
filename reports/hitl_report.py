"""
Blueprint Processor V4.2.1 - HITL Report Generator
Creates CSV and HTML reports for human-in-the-loop review.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import REVIEW_THRESHOLD

logger = logging.getLogger(__name__)


class HITLReportGenerator:
    """
    Generates Human-In-The-Loop review reports.

    Creates two types of reports:
    1. CSV: Machine-readable format for bulk corrections
    2. HTML: Human-readable format with visual indicators

    Reports include:
    - Sheets needing review (confidence < threshold)
    - Cross-reference conflicts
    - Extraction method breakdown
    - Suggested corrections
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the report generator.

        Args:
            output_dir: Directory for output reports (default: output/reports)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'output' / 'reports'
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(
        self,
        results: List[Dict[str, Any]],
        stats: Dict[str, Any],
        pdf_name: str = "batch"
    ) -> Dict[str, Path]:
        """
        Generate both CSV and HTML reports.

        Args:
            results: List of extraction results
            stats: Extraction statistics
            pdf_name: Name for report files

        Returns:
            Dict with paths to generated reports
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"hitl_{pdf_name}_{timestamp}"

        # Separate sheets needing review
        needs_review = [r for r in results if r.get('needs_review', False)]
        all_sheets = results

        # Generate reports
        csv_path = self._generate_csv(needs_review, all_sheets, stats, base_name)
        html_path = self._generate_html(needs_review, all_sheets, stats, base_name)

        logger.info(f"Generated HITL reports: {csv_path.name}, {html_path.name}")

        return {
            'csv': csv_path,
            'html': html_path,
            'needs_review_count': len(needs_review),
            'total_count': len(all_sheets),
        }

    def _generate_csv(
        self,
        needs_review: List[Dict[str, Any]],
        all_sheets: List[Dict[str, Any]],
        stats: Dict[str, Any],
        base_name: str
    ) -> Path:
        """Generate CSV report for bulk corrections."""
        csv_path = self.output_dir / f"{base_name}.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'pdf_filename',
                'page_number',
                'pdf_hash',
                'sheet_number',
                'extracted_title',
                'confidence',
                'method',
                'needs_review',
                'review_reason',
                'suggested_title',
                'corrected_title',  # Empty column for human input
            ])

            # Write all sheets (review items first)
            for sheet in needs_review + [s for s in all_sheets if s not in needs_review]:
                # Determine review reason
                review_reason = self._get_review_reason(sheet)

                # Get suggested title from cross-reference if available
                suggested = sheet.get('extraction_details', {}).get('index_title', '')

                writer.writerow([
                    sheet.get('pdf_filename', ''),
                    sheet.get('page_number', ''),
                    sheet.get('pdf_hash', ''),
                    sheet.get('sheet_number', ''),
                    sheet.get('sheet_title', ''),
                    f"{sheet.get('title_confidence', 0):.2f}",
                    sheet.get('title_method', ''),
                    'YES' if sheet.get('needs_review', False) else 'NO',
                    review_reason,
                    suggested,
                    '',  # Empty for human correction
                ])

        return csv_path

    def _generate_html(
        self,
        needs_review: List[Dict[str, Any]],
        all_sheets: List[Dict[str, Any]],
        stats: Dict[str, Any],
        base_name: str
    ) -> Path:
        """Generate HTML report for visual review."""
        html_path = self.output_dir / f"{base_name}.html"

        # Calculate statistics
        total = len(all_sheets)
        review_count = len(needs_review)
        method_counts = stats.get('by_method', {})

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HITL Review Report - {base_name}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; margin-bottom: 10px; }}
        .subtitle {{ color: #7f8c8d; margin-bottom: 30px; }}

        /* Stats cards */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{ margin: 0 0 10px 0; color: #7f8c8d; font-size: 14px; }}
        .stat-card .value {{ font-size: 32px; font-weight: bold; }}
        .stat-card.warning .value {{ color: #e74c3c; }}
        .stat-card.success .value {{ color: #27ae60; }}

        /* Method breakdown */
        .method-bar {{
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
            display: flex;
        }}
        .method-bar div {{ height: 100%; }}
        .method-index {{ background: #27ae60; }}
        .method-spatial {{ background: #3498db; }}
        .method-vision {{ background: #9b59b6; }}
        .method-pattern {{ background: #f39c12; }}
        .method-failed {{ background: #e74c3c; }}

        /* Tables */
        .section {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section h2 {{ margin-top: 0; color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #7f8c8d; text-transform: uppercase; font-size: 12px; }}
        tr:hover {{ background: #f8f9fa; }}

        /* Confidence indicators */
        .confidence {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 500;
            font-size: 12px;
        }}
        .confidence.high {{ background: #d4edda; color: #155724; }}
        .confidence.medium {{ background: #fff3cd; color: #856404; }}
        .confidence.low {{ background: #f8d7da; color: #721c24; }}

        /* Method badges */
        .method {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
        }}
        .method.drawing_index, .method.drawing_index_xref {{ background: #d4edda; color: #155724; }}
        .method.spatial {{ background: #d1ecf1; color: #0c5460; }}
        .method.vision_api {{ background: #e2d5f1; color: #4a235a; }}
        .method.pattern {{ background: #fff3cd; color: #856404; }}

        /* Review flags */
        .review-flag {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .review-flag.yes {{ background: #f8d7da; color: #721c24; }}
        .review-flag.no {{ background: #d4edda; color: #155724; }}

        /* Legend */
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 10px;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }}

        /* Print styles */
        @media print {{
            body {{ background: white; }}
            .section {{ box-shadow: none; border: 1px solid #ddd; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>HITL Review Report</h1>
        <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | PDF Hash: {stats.get('pdf_hash', 'N/A')}</p>

        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Sheets</h3>
                <div class="value">{total}</div>
            </div>
            <div class="stat-card {'warning' if review_count > 0 else 'success'}">
                <h3>Needs Review</h3>
                <div class="value">{review_count}</div>
            </div>
            <div class="stat-card success">
                <h3>Success Rate</h3>
                <div class="value">{stats.get('success_rate', 0)*100:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>From Index</h3>
                <div class="value">{method_counts.get('drawing_index', 0)}</div>
            </div>
        </div>

        <!-- Method Breakdown -->
        <div class="section">
            <h2>Extraction Methods</h2>
            <div class="method-bar">
                {self._method_bar_segment(method_counts.get('drawing_index', 0), total, 'index')}
                {self._method_bar_segment(method_counts.get('spatial', 0), total, 'spatial')}
                {self._method_bar_segment(method_counts.get('vision_api', 0), total, 'vision')}
                {self._method_bar_segment(method_counts.get('pattern', 0), total, 'pattern')}
                {self._method_bar_segment(method_counts.get('failed', 0), total, 'failed')}
            </div>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot method-index"></div> Drawing Index ({method_counts.get('drawing_index', 0)})</div>
                <div class="legend-item"><div class="legend-dot method-spatial"></div> Spatial ({method_counts.get('spatial', 0)})</div>
                <div class="legend-item"><div class="legend-dot method-vision"></div> Vision API ({method_counts.get('vision_api', 0)})</div>
                <div class="legend-item"><div class="legend-dot method-pattern"></div> Pattern ({method_counts.get('pattern', 0)})</div>
                <div class="legend-item"><div class="legend-dot method-failed"></div> Failed ({method_counts.get('failed', 0)})</div>
            </div>
        </div>

        <!-- Needs Review Table -->
        {self._review_table_html(needs_review) if needs_review else '<div class="section"><h2>Sheets Needing Review</h2><p>No sheets require review.</p></div>'}

        <!-- All Sheets Table -->
        <div class="section">
            <h2>All Extracted Sheets</h2>
            <table>
                <thead>
                    <tr>
                        <th>PDF</th>
                        <th>Page</th>
                        <th>Sheet #</th>
                        <th>Title</th>
                        <th>Confidence</th>
                        <th>Method</th>
                        <th>Review</th>
                    </tr>
                </thead>
                <tbody>
"""
        for sheet in all_sheets:
            confidence = sheet.get('title_confidence', 0)
            conf_class = 'high' if confidence >= 0.85 else ('medium' if confidence >= 0.70 else 'low')
            method = sheet.get('title_method', '') or 'none'
            needs_rev = sheet.get('needs_review', False)

            html += f"""                    <tr>
                        <td>{sheet.get('pdf_filename', '')}</td>
                        <td>{sheet.get('page_number', '')}</td>
                        <td><strong>{sheet.get('sheet_number', '') or '-'}</strong></td>
                        <td>{sheet.get('sheet_title', '') or '-'}</td>
                        <td><span class="confidence {conf_class}">{confidence:.0%}</span></td>
                        <td><span class="method {method}">{method}</span></td>
                        <td><span class="review-flag {'yes' if needs_rev else 'no'}">{'YES' if needs_rev else 'NO'}</span></td>
                    </tr>
"""

        html += """                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return html_path

    def _method_bar_segment(self, count: int, total: int, method: str) -> str:
        """Generate a segment for the method breakdown bar."""
        if total == 0 or count == 0:
            return ''
        pct = (count / total) * 100
        return f'<div class="method-{method}" style="width: {pct}%"></div>'

    def _review_table_html(self, needs_review: List[Dict[str, Any]]) -> str:
        """Generate the review table HTML."""
        html = """<div class="section">
            <h2>Sheets Needing Review ({count})</h2>
            <table>
                <thead>
                    <tr>
                        <th>PDF</th>
                        <th>Page</th>
                        <th>Sheet #</th>
                        <th>Extracted Title</th>
                        <th>Suggested Title</th>
                        <th>Reason</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
""".format(count=len(needs_review))

        for sheet in needs_review:
            reason = self._get_review_reason(sheet)
            suggested = sheet.get('extraction_details', {}).get('index_title', '')
            confidence = sheet.get('title_confidence', 0)

            html += f"""                    <tr>
                        <td>{sheet.get('pdf_filename', '')}</td>
                        <td>{sheet.get('page_number', '')}</td>
                        <td><strong>{sheet.get('sheet_number', '') or '-'}</strong></td>
                        <td>{sheet.get('sheet_title', '') or '-'}</td>
                        <td><em>{suggested or '-'}</em></td>
                        <td>{reason}</td>
                        <td><span class="confidence low">{confidence:.0%}</span></td>
                    </tr>
"""

        html += """                </tbody>
            </table>
        </div>"""

        return html

    def _get_review_reason(self, sheet: Dict[str, Any]) -> str:
        """Determine why a sheet needs review."""
        details = sheet.get('extraction_details', {})

        if details.get('xref_conflict'):
            return 'Cross-ref conflict'
        if sheet.get('title_confidence', 0) < REVIEW_THRESHOLD:
            return 'Low confidence'
        if not sheet.get('sheet_title'):
            return 'No title extracted'
        if sheet.get('title_method') == 'pattern':
            return 'Pattern fallback'
        return 'Manual review requested'

    def import_corrections(self, csv_path: Path) -> List[Dict[str, Any]]:
        """
        Import corrections from a filled-in CSV file.

        Args:
            csv_path: Path to CSV with corrections

        Returns:
            List of correction records
        """
        corrections = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                corrected = row.get('corrected_title', '').strip()
                if corrected:
                    corrections.append({
                        'pdf_filename': row.get('pdf_filename'),
                        'page_number': int(row.get('page_number', 0)),
                        'pdf_hash': row.get('pdf_hash'),
                        'sheet_number': row.get('sheet_number'),
                        'original_title': row.get('extracted_title'),
                        'corrected_title': corrected,
                    })

        logger.info(f"Imported {len(corrections)} corrections from {csv_path}")
        return corrections
