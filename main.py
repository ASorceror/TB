"""
Blueprint Processor V4.4 - Main Entry Point
CLI interface for processing blueprints.

Usage:
    python main.py process <path>      Process single PDF or folder
    python main.py info <pdf_path>     Show info about a PDF
    python main.py stats               Show database statistics
    python main.py review <csv_path>   Import corrections from HITL CSV
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm

from constants import LOG_FILENAME, EXTRACTION_METHODS, CONFIDENCE_LEVELS, DEFAULT_DPI
from core.pdf_handler import PDFHandler
from core.page_normalizer import PageNormalizer
from core.ocr_engine import OCREngine
from core.sheet_title_extractor import SheetTitleExtractor
from validation.validator import Validator
from database.operations import DatabaseOperations
from reports.hitl_report import HITLReportGenerator


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILENAME

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('blueprint_processor')


class BlueprintProcessor:
    """Main processor class that orchestrates all components."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize processor with all components."""
        self.project_root = Path(__file__).parent
        self.telemetry_dir = self.project_root / 'output' / 'telemetry'
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.normalizer = PageNormalizer(telemetry_dir=self.telemetry_dir)
        self.ocr_engine = OCREngine(telemetry_dir=self.telemetry_dir)
        self.validator = Validator()
        self.db = DatabaseOperations(db_path)

        # V4.2.1: New sheet title extractor and HITL report generator
        self.title_extractor = SheetTitleExtractor(telemetry_dir=self.telemetry_dir)
        self.hitl_reporter = HITLReportGenerator()

        # Setup logging
        self.logger = setup_logging(self.project_root / 'logs')

    def process_page(self, handler: PDFHandler, page_num: int,
                     pdf_filename: str, pdf_hash: str = None) -> Dict[str, Any]:
        """
        Process a single page from a PDF using the four-layer extractor.

        Args:
            handler: PDFHandler instance
            page_num: Page number (0-indexed)
            pdf_filename: Name of the PDF file
            pdf_hash: SHA-256 hash of PDF for stable identification

        Returns:
            Dict with extracted and validated data
        """
        result = {
            'pdf_filename': pdf_filename,
            'pdf_hash': pdf_hash,
            'page_number': page_num + 1,  # Store as 1-indexed
            'sheet_number': None,
            'project_number': None,
            'sheet_title': None,
            'date': None,
            'scale': None,
            'discipline': None,
            'confidence': CONFIDENCE_LEVELS['LOW'],
            'extraction_method': None,
            'title_confidence': 0.0,
            'title_method': None,
            'needs_review': 1,
            'is_valid': 0,
            'errors': [],
            'extraction_details': {},
        }

        try:
            analysis = handler.analyze_page(page_num)
            is_vector = analysis['recommendation'] == EXTRACTION_METHODS['VECTOR_PDF']
            result['extraction_method'] = 'vector' if is_vector else 'ocr'

            page_image = handler.get_page_image(page_num, dpi=DEFAULT_DPI)
            normalized_image, orientation_info = self.normalizer.normalize(page_image)

            if is_vector:
                text = handler.get_page_text(page_num)
            else:
                text = self.ocr_engine.ocr_image(normalized_image)

            extracted = self.title_extractor.extract_title(
                pdf_handler=handler,
                page_num=page_num,
                page_image=normalized_image,
                text=text,
                original_page_image=page_image,  # V4.4: Pass original for rotation fallback
                orientation_info=orientation_info,  # V4.4: Pass rotation info
            )

            extraction_details = extracted.get('extraction_details', {}) or {}
            if orientation_info:
                extraction_details.setdefault('orientation', orientation_info)
            result['extraction_details'] = extraction_details

            for field in ['sheet_number', 'project_number', 'sheet_title', 'date', 'scale', 'discipline']:
                result[field] = extracted.get(field)

            result['title_confidence'] = extracted.get('title_confidence', 0.0)
            result['title_method'] = extracted.get('title_method')
            result['needs_review'] = 1 if extracted.get('needs_review', True) else 0
            result['is_cover_sheet'] = extracted.get('is_cover_sheet', False)

            if result.get('project_number') and not result.get('project_number_source'):
                result['project_number_source'] = 'extracted'

            validation = self.validator.validate(result)
            result['is_valid'] = 1 if validation['is_valid'] else 0
            result['errors'] = validation['errors']

            if result['sheet_number'] and result['project_number']:
                if validation['is_valid']:
                    result['confidence'] = CONFIDENCE_LEVELS['HIGH']
                else:
                    result['confidence'] = CONFIDENCE_LEVELS['MEDIUM']

        except Exception as e:
            result['errors'].append(str(e))
            self.logger.error(f"Error processing page {page_num + 1} of {pdf_filename}: {e}")

        return result


    def apply_project_number_fallback(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        If a page is missing project_number, use the most common value from other pages.

        This works because all pages in a blueprint set share the same project number.

        Args:
            results: List of extraction results for all pages

        Returns:
            Updated results with fallback project numbers applied
        """
        from collections import Counter

        # Collect all non-null project numbers
        project_numbers = [r['project_number'] for r in results if r.get('project_number')]

        if not project_numbers:
            return results  # No project numbers found at all

        # Find the most common project number
        most_common, count = Counter(project_numbers).most_common(1)[0]
        confidence = count / max(1, len(project_numbers))
        if confidence < 0.6:
            self.logger.warning('Project number fallback skipped (insufficient agreement across pages)')
            return results

        # Apply to pages that are missing it
        for result in results:
            if result.get('project_number') is None:
                result['project_number'] = most_common
                result['project_number_source'] = 'fallback'

        return results

    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Process all pages of a PDF using V4.2.1 layered extraction.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of results for each page
        """
        results = []
        pdf_filename = pdf_path.name

        self.logger.info(f"Processing: {pdf_filename}")

        try:
            self.title_extractor.reset_for_new_pdf()
            pdf_hash = self.title_extractor.compute_pdf_hash(pdf_path)
            self.logger.info(f"PDF hash: {pdf_hash}")

            with PDFHandler(pdf_path) as handler:
                index_map = self.title_extractor.parse_drawing_index(handler)
                if index_map:
                    self.logger.info(f"Drawing index: {len(index_map)} entries")

                for page_num in range(handler.page_count):
                    result = self.process_page(handler, page_num, pdf_filename, pdf_hash)
                    results.append(result)

            results = self.apply_project_number_fallback(results)

            non_db_fields = {'errors', 'extraction_details', 'project_number_source', 'is_cover_sheet'}
            for result in results:
                db_data = {k: v for k, v in result.items() if k not in non_db_fields}
                self.db.upsert_sheet(db_data)

        except Exception as e:
            self.logger.error(f"Error processing {pdf_filename}: {e}")
            results.append({
                'pdf_filename': pdf_filename,
                'page_number': 0,
                'errors': [str(e)],
            })

        return results



    def process_folder(self, folder_path: Path) -> Dict[str, Any]:
        """
        Process all PDFs in a folder.

        Args:
            folder_path: Path to folder

        Returns:
            Summary dict with all results
        """
        # Find all PDFs
        pdf_files = list(folder_path.glob('*.pdf'))

        if not pdf_files:
            self.logger.warning(f"No PDF files found in {folder_path}")
            return {'summary': {'total': 0}, 'sheets': []}

        self.logger.info(f"Found {len(pdf_files)} PDF files")

        # Start processing run
        run = self.db.start_processing_run(str(folder_path))

        all_results = []
        success_count = 0
        error_count = 0

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            results = self.process_pdf(pdf_path)
            all_results.extend(results)

            for result in results:
                if result.get('is_valid', 0) == 1:
                    success_count += 1
                else:
                    error_count += 1

        # Complete processing run
        self.db.complete_processing_run(
            run.id,
            pdf_count=len(pdf_files),
            page_count=len(all_results),
            success_count=success_count,
            error_count=error_count
        )

        summary = self._summarize_results(all_results, len(pdf_files), success_count, error_count)
        return {
            'summary': summary,
            'sheets': all_results,
        }

    def process(self, path: Path) -> Dict[str, Any]:
        """
        Process a file or folder.

        Args:
            path: Path to PDF file or folder

        Returns:
            Processing results
        """
        if path.is_file():
            results = self.process_pdf(path)
            summary = self._summarize_results(results, 1)
            return {
                'summary': summary,
                'sheets': results,
            }
        elif path.is_dir():
            return self.process_folder(path)
        else:
            raise ValueError(f"Path does not exist: {path}")

    def _summarize_results(self, sheets: List[Dict[str, Any]], total_pdfs: int,
                            success_count: Optional[int] = None, error_count: Optional[int] = None) -> Dict[str, Any]:
        """Build summary statistics for reports."""
        if success_count is None:
            success_count = sum(1 for r in sheets if r.get('is_valid', 0) == 1)
        if error_count is None:
            error_count = len(sheets) - success_count

        missing_titles = sum(1 for r in sheets if not (r.get('sheet_title') or '').strip())
        missing_sheet_numbers = sum(1 for r in sheets if not (r.get('sheet_number') or '').strip())
        fallback_projects = sum(1 for r in sheets if r.get('project_number_source') == 'fallback')

        return {
            'total_pdfs': total_pdfs,
            'total_pages': len(sheets),
            'success_count': success_count,
            'error_count': error_count,
            'missing_sheet_titles': missing_titles,
            'missing_sheet_numbers': missing_sheet_numbers,
            'fallback_project_numbers': fallback_projects,
        }


    def generate_report(self, results: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
        """
        Generate JSON, HTML, and HITL review reports.

        Args:
            results: Processing results
            output_dir: Directory to save reports

        Returns:
            Dict with paths to all generated reports
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON report
        json_path = output_dir / f'report_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        # HTML report
        html_path = output_dir / f'report_{timestamp}.html'
        html_content = self._generate_html_report(results)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # V4.2.1: Generate HITL review reports
        sheets = results.get('sheets', [])
        stats = {
            'total_pages': len(sheets),
            'by_method': self._count_methods(sheets),
            'success_rate': self._calc_success_rate(sheets),
            'pdf_hash': sheets[0].get('pdf_hash') if sheets else None,
        }
        hitl_reports = self.hitl_reporter.generate_reports(sheets, stats, f"batch_{timestamp}")

        return {
            'json': json_path,
            'html': html_path,
            'hitl_csv': hitl_reports['csv'],
            'hitl_html': hitl_reports['html'],
            'needs_review_count': hitl_reports['needs_review_count'],
        }

    def _count_methods(self, sheets: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count extraction methods used."""
        counts = {'drawing_index': 0, 'spatial': 0, 'vision_api': 0, 'pattern': 0, 'failed': 0}
        for sheet in sheets:
            method = sheet.get('title_method')
            if method in counts:
                counts[method] += 1
            elif method == 'drawing_index_xref':
                counts['drawing_index'] += 1
            elif method is None:
                counts['failed'] += 1
        return counts

    def _calc_success_rate(self, sheets: List[Dict[str, Any]]) -> float:
        """Calculate extraction success rate."""
        if not sheets:
            return 0.0
        success = sum(1 for s in sheets if s.get('title_method') is not None)
        return success / len(sheets)

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        summary = results.get('summary', {})
        sheets = results.get('sheets', [])

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Blueprint Processing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .valid {{ color: green; }}
        .invalid {{ color: red; }}
        @media print {{ body {{ margin: 0; }} }}
    </style>
</head>
<body>
    <h1>Blueprint Processing Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p>Total PDFs: {summary.get('total_pdfs', 0)}</p>
        <p>Total Pages: {summary.get('total_pages', 0)}</p>
        <p>Successful: <span class="valid">{summary.get('success_count', 0)}</span></p>
        <p>Errors: <span class="invalid">{summary.get('error_count', 0)}</span></p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <h2>Extracted Sheets</h2>
    <table>
        <tr>
            <th>PDF</th>
            <th>Page</th>
            <th>Sheet #</th>
            <th>Project #</th>
            <th>Date</th>
            <th>Discipline</th>
            <th>Method</th>
            <th>Valid</th>
        </tr>
"""
        for sheet in sheets:
            valid_class = 'valid' if sheet.get('is_valid', 0) == 1 else 'invalid'
            valid_text = 'Yes' if sheet.get('is_valid', 0) == 1 else 'No'
            html += f"""        <tr>
            <td>{sheet.get('pdf_filename', '')}</td>
            <td>{sheet.get('page_number', '')}</td>
            <td>{sheet.get('sheet_number', '') or '-'}</td>
            <td>{sheet.get('project_number', '') or '-'}</td>
            <td>{sheet.get('date', '') or '-'}</td>
            <td>{sheet.get('discipline', '') or '-'}</td>
            <td>{sheet.get('extraction_method', '') or '-'}</td>
            <td class="{valid_class}">{valid_text}</td>
        </tr>
"""

        html += """    </table>
</body>
</html>"""

        return html


def cmd_process(args):
    """Process command handler."""
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    processor = BlueprintProcessor()
    results = processor.process(path)

    # Generate reports
    output_dir = Path(__file__).parent / 'output'
    reports = processor.generate_report(results, output_dir)

    # Print summary
    summary = results['summary']
    print(f"\nProcessing Complete! (Blueprint Processor V4.4)")
    print(f"  PDFs processed: {summary['total_pdfs']}")
    print(f"  Pages processed: {summary['total_pages']}")
    print(f"  Successful extractions: {summary['success_count']}")
    print(f"  Errors: {summary['error_count']}")
    print(f"  Needs HITL Review: {reports['needs_review_count']}")
    print(f"\nReports saved to:")
    print(f"  JSON: {reports['json']}")
    print(f"  HTML: {reports['html']}")
    print(f"  HITL CSV: {reports['hitl_csv']}")
    print(f"  HITL HTML: {reports['hitl_html']}")


def cmd_info(args):
    """Info command handler."""
    path = Path(args.pdf_path)
    if not path.exists():
        print(f"Error: File does not exist: {path}")
        sys.exit(1)

    with PDFHandler(path) as handler:
        print(f"\nPDF Information: {path.name}")
        print(f"  Page count: {handler.page_count}")

        for i in range(handler.page_count):
            analysis = handler.analyze_page(i)
            print(f"\n  Page {i + 1}:")
            print(f"    Has text: {analysis['has_text']}")
            print(f"    Text length: {analysis['text_length']}")
            print(f"    Is scanned: {analysis['is_scanned']}")
            print(f"    Recommendation: {analysis['recommendation']}")


def cmd_stats(args):
    """Stats command handler."""
    db = DatabaseOperations()

    total = db.count_sheets()
    runs = db.get_processing_runs(5)

    print(f"\nDatabase Statistics (V4.4):")
    print(f"  Total sheets: {total}")
    print(f"\nRecent Processing Runs:")

    for run in runs:
        print(f"  - {run.started_at}: {run.pdf_count} PDFs, {run.page_count} pages, status={run.status}")


def cmd_review(args):
    """Import corrections from HITL review CSV."""
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: File does not exist: {csv_path}")
        sys.exit(1)

    reporter = HITLReportGenerator()
    corrections = reporter.import_corrections(csv_path)

    if not corrections:
        print("No corrections found in CSV file.")
        sys.exit(0)

    print(f"\nImporting {len(corrections)} corrections...")

    db = DatabaseOperations()
    updated = 0

    for correction in corrections:
        # Find and update the sheet
        sheet = db.get_sheet(
            correction['pdf_filename'],
            correction['page_number']
        )
        if sheet:
            db.upsert_sheet({
                'pdf_filename': correction['pdf_filename'],
                'page_number': correction['page_number'],
                'sheet_title': correction['corrected_title'],
                'title_confidence': 1.0,  # Human-verified
                'title_method': 'hitl_correction',
                'needs_review': 0,
            })
            updated += 1
            print(f"  Updated: {correction['pdf_filename']} p{correction['page_number']}: {correction['corrected_title']}")

    print(f"\nImport complete: {updated}/{len(corrections)} sheets updated.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Blueprint Processor V4.4 - Extract data from blueprint PDFs'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Process command
    process_parser = subparsers.add_parser('process', help='Process PDF(s)')
    process_parser.add_argument('path', help='Path to PDF file or folder')
    process_parser.set_defaults(func=cmd_process)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show PDF info')
    info_parser.add_argument('pdf_path', help='Path to PDF file')
    info_parser.set_defaults(func=cmd_info)

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database stats')
    stats_parser.set_defaults(func=cmd_stats)

    # Review command (V4.2.1)
    review_parser = subparsers.add_parser('review', help='Import corrections from HITL CSV')
    review_parser.add_argument('csv_path', help='Path to corrections CSV file')
    review_parser.set_defaults(func=cmd_review)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
