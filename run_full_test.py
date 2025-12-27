"""
Full Test Run Script - Blueprint Processor V4.7.1
Runs on all PDFs in test folder with detailed logging and result packaging.
"""

import sys
import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime

# Add blueprint_processor to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.sheet_title_extractor import SheetTitleExtractor

# Create timestamped output directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(__file__).parent / "output" / f"full_test_run_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
log_file = OUTPUT_DIR / "extraction.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_test_pdfs(test_folder: Path) -> list:
    """Get list of PDFs to test (exclude Test subfolder and _5pages versions)."""
    pdfs = []
    for pdf_path in test_folder.glob("*.pdf"):
        # Skip 5-page test versions if full version exists
        if "_5pages" in pdf_path.name:
            full_name = pdf_path.name.replace("_5pages", "")
            if (test_folder / full_name).exists():
                logger.info(f"Skipping {pdf_path.name} (full version exists)")
                continue
        pdfs.append(pdf_path)
    return sorted(pdfs)


def process_pdf(pdf_path: Path, extractor: SheetTitleExtractor) -> dict:
    """Process a single PDF and return results."""
    logger.info(f"=" * 60)
    logger.info(f"Processing: {pdf_path.name}")
    logger.info(f"=" * 60)

    results = {
        'pdf_name': pdf_path.name,
        'pdf_path': str(pdf_path),
        'pdf_hash': None,
        'page_count': 0,
        'pages': [],
        'drawing_index_entries': 0,
        'success': False,
        'error': None,
    }

    try:
        extractor.reset_for_new_pdf()
        results['pdf_hash'] = extractor.compute_pdf_hash(pdf_path)

        with PDFHandler(pdf_path) as handler:
            results['page_count'] = handler.page_count
            logger.info(f"Page count: {handler.page_count}")

            # Parse drawing index
            drawing_index = extractor.parse_drawing_index(handler)
            results['drawing_index_entries'] = len(drawing_index)
            logger.info(f"Drawing index entries: {len(drawing_index)}")

            # Process each page
            for page_num in range(handler.page_count):
                logger.info(f"--- Page {page_num + 1}/{handler.page_count} ---")

                page_image = handler.get_page_image(page_num, dpi=200)
                result = extractor.extract_title(handler, page_num, page_image)

                page_result = {
                    'page_number': page_num + 1,
                    'sheet_number': result.get('sheet_number'),
                    'sheet_title': result.get('sheet_title'),
                    'title_confidence': result.get('title_confidence'),
                    'title_method': result.get('title_method'),
                    'needs_review': result.get('needs_review', False),
                }
                results['pages'].append(page_result)

                logger.info(f"  Sheet: {page_result['sheet_number']}")
                logger.info(f"  Title: {page_result['sheet_title']}")
                logger.info(f"  Method: {page_result['title_method']}")
                logger.info(f"  Confidence: {page_result['title_confidence']}")

        results['success'] = True

    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        results['error'] = str(e)

    return results


def generate_summary_csv(all_results: list, output_path: Path):
    """Generate a summary CSV with all page results."""
    csv_path = output_path / "summary.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pdf_filename', 'page_number', 'pdf_hash', 'sheet_number',
            'sheet_title', 'confidence', 'method', 'needs_review'
        ])

        for result in all_results:
            for page in result.get('pages', []):
                writer.writerow([
                    result['pdf_name'],
                    page['page_number'],
                    result['pdf_hash'],
                    page['sheet_number'] or '',
                    page['sheet_title'] or '',
                    f"{page['title_confidence']:.2f}" if page['title_confidence'] else '',
                    page['title_method'] or '',
                    'YES' if page['needs_review'] else 'NO',
                ])

    logger.info(f"Summary CSV written to: {csv_path}")
    return csv_path


def generate_statistics(all_results: list) -> dict:
    """Generate extraction statistics."""
    stats = {
        'total_pdfs': len(all_results),
        'successful_pdfs': sum(1 for r in all_results if r['success']),
        'failed_pdfs': sum(1 for r in all_results if not r['success']),
        'total_pages': sum(r['page_count'] for r in all_results),
        'pages_with_sheet_number': 0,
        'pages_without_sheet_number': 0,
        'pages_needing_review': 0,
        'methods': {},
        'confidence_distribution': {
            'high_90_100': 0,
            'medium_70_90': 0,
            'low_50_70': 0,
            'very_low_0_50': 0,
        },
        'per_pdf_stats': [],
    }

    for result in all_results:
        pdf_stat = {
            'pdf_name': result['pdf_name'],
            'page_count': result['page_count'],
            'success': result['success'],
            'sheets_extracted': 0,
            'sheets_missing': 0,
        }

        for page in result.get('pages', []):
            if page['sheet_number']:
                stats['pages_with_sheet_number'] += 1
                pdf_stat['sheets_extracted'] += 1
            else:
                stats['pages_without_sheet_number'] += 1
                pdf_stat['sheets_missing'] += 1

            if page['needs_review']:
                stats['pages_needing_review'] += 1

            method = page['title_method'] or 'none'
            stats['methods'][method] = stats['methods'].get(method, 0) + 1

            conf = page['title_confidence'] or 0
            if conf >= 0.9:
                stats['confidence_distribution']['high_90_100'] += 1
            elif conf >= 0.7:
                stats['confidence_distribution']['medium_70_90'] += 1
            elif conf >= 0.5:
                stats['confidence_distribution']['low_50_70'] += 1
            else:
                stats['confidence_distribution']['very_low_0_50'] += 1

        stats['per_pdf_stats'].append(pdf_stat)

    # Calculate percentages
    if stats['total_pages'] > 0:
        stats['sheet_extraction_rate'] = stats['pages_with_sheet_number'] / stats['total_pages'] * 100
    else:
        stats['sheet_extraction_rate'] = 0

    return stats


def main():
    """Main entry point."""
    test_folder = Path("C:/Full Set")

    logger.info("=" * 70)
    logger.info("Blueprint Processor V4.7.1 - Full Test Run")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 70)

    # Get PDFs to process
    pdfs = get_test_pdfs(test_folder)
    logger.info(f"Found {len(pdfs)} PDFs to process")

    for i, pdf in enumerate(pdfs, 1):
        logger.info(f"  {i}. {pdf.name}")

    # Process all PDFs
    extractor = SheetTitleExtractor()
    all_results = []

    for i, pdf_path in enumerate(pdfs, 1):
        logger.info(f"\n[{i}/{len(pdfs)}] Processing {pdf_path.name}")
        result = process_pdf(pdf_path, extractor)
        all_results.append(result)

        # Save intermediate results
        with open(OUTPUT_DIR / "results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # Generate summary CSV
    generate_summary_csv(all_results, OUTPUT_DIR)

    # Generate and save statistics
    stats = generate_statistics(all_results)
    with open(OUTPUT_DIR / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total PDFs processed: {stats['total_pdfs']}")
    logger.info(f"Successful: {stats['successful_pdfs']}")
    logger.info(f"Failed: {stats['failed_pdfs']}")
    logger.info(f"Total pages: {stats['total_pages']}")
    logger.info(f"Pages with sheet number: {stats['pages_with_sheet_number']} ({stats['sheet_extraction_rate']:.1f}%)")
    logger.info(f"Pages without sheet number: {stats['pages_without_sheet_number']}")
    logger.info(f"Pages needing review: {stats['pages_needing_review']}")

    logger.info("\nExtraction methods used:")
    for method, count in sorted(stats['methods'].items(), key=lambda x: -x[1]):
        logger.info(f"  {method}: {count}")

    logger.info("\nConfidence distribution:")
    for level, count in stats['confidence_distribution'].items():
        logger.info(f"  {level}: {count}")

    logger.info("\nPer-PDF results:")
    for pdf_stat in stats['per_pdf_stats']:
        status = "OK" if pdf_stat['success'] else "FAILED"
        extracted = pdf_stat['sheets_extracted']
        total = pdf_stat['page_count']
        pct = (extracted / total * 100) if total > 0 else 0
        logger.info(f"  {pdf_stat['pdf_name']}: {extracted}/{total} ({pct:.0f}%) [{status}]")

    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info("Files:")
    logger.info(f"  - extraction.log (detailed log)")
    logger.info(f"  - results.json (full results)")
    logger.info(f"  - summary.csv (page-level summary)")
    logger.info(f"  - statistics.json (aggregate stats)")

    return stats


if __name__ == "__main__":
    stats = main()

    # Exit with error if any PDFs failed
    if stats['failed_pdfs'] > 0:
        sys.exit(1)
