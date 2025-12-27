"""
Full Test Run Script V2 - Blueprint Processor V4.7.1
Optimized for faster processing with real-time progress output.
"""

import sys
import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Add blueprint_processor to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.sheet_title_extractor import SheetTitleExtractor

# Create timestamped output directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(__file__).parent / "output" / f"full_test_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure file logging only (console output manually controlled)
log_file = OUTPUT_DIR / "extraction.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Set up root logger
logging.root.setLevel(logging.DEBUG)
logging.root.addHandler(file_handler)

# Suppress noisy loggers
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('fitz').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def get_test_pdfs(test_folder: Path) -> list:
    """Get list of PDFs to test (exclude Test subfolder and _5pages versions)."""
    pdfs = []
    for pdf_path in test_folder.glob("*.pdf"):
        # Skip 5-page test versions if full version exists
        if "_5pages" in pdf_path.name:
            full_name = pdf_path.name.replace("_5pages", "")
            if (test_folder / full_name).exists():
                continue
        pdfs.append(pdf_path)
    return sorted(pdfs)


def process_pdf(pdf_path: Path, extractor: SheetTitleExtractor) -> dict:
    """Process a single PDF and return results."""
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

            # Parse drawing index
            drawing_index = extractor.parse_drawing_index(handler)
            results['drawing_index_entries'] = len(drawing_index)

            # Process each page
            for page_num in range(handler.page_count):
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

                logger.info(f"Page {page_num+1}: sheet={page_result['sheet_number']}, "
                           f"method={page_result['title_method']}, conf={page_result['title_confidence']}")

        results['success'] = True

    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        results['error'] = str(e)
        traceback.print_exc()

    return results


def main():
    """Main entry point."""
    test_folder = Path("C:/Full Set")

    print("=" * 70)
    print("Blueprint Processor V4.7.1 - Full Test Run")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)
    sys.stdout.flush()

    # Get PDFs to process
    pdfs = get_test_pdfs(test_folder)
    print(f"\nFound {len(pdfs)} PDFs to process:")
    for i, pdf in enumerate(pdfs, 1):
        print(f"  {i:2}. {pdf.name}")
    print()
    sys.stdout.flush()

    # Process all PDFs
    extractor = SheetTitleExtractor()
    all_results = []
    total_pages = 0
    pages_with_sheet = 0

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i:2}/{len(pdfs)}] {pdf_path.name}...", end=" ", flush=True)

        start_time = datetime.now()
        result = process_pdf(pdf_path, extractor)
        elapsed = (datetime.now() - start_time).total_seconds()

        all_results.append(result)

        # Count results
        pdf_pages = len(result.get('pages', []))
        pdf_sheets = sum(1 for p in result.get('pages', []) if p['sheet_number'])
        total_pages += pdf_pages
        pages_with_sheet += pdf_sheets

        status = "OK" if result['success'] else "FAILED"
        pct = (pdf_sheets / pdf_pages * 100) if pdf_pages > 0 else 0
        print(f"{pdf_pages} pages, {pdf_sheets} sheets ({pct:.0f}%), {elapsed:.1f}s [{status}]", flush=True)

    # Save results
    print("\nSaving results...")

    # Full JSON results
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary CSV
    csv_path = OUTPUT_DIR / "summary.csv"
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
                    (page['sheet_title'] or '')[:100],  # Truncate long titles
                    f"{page['title_confidence']:.2f}" if page['title_confidence'] else '',
                    page['title_method'] or '',
                    'YES' if page['needs_review'] else 'NO',
                ])

    # Statistics
    stats = {
        'timestamp': TIMESTAMP,
        'total_pdfs': len(all_results),
        'successful_pdfs': sum(1 for r in all_results if r['success']),
        'failed_pdfs': sum(1 for r in all_results if not r['success']),
        'total_pages': total_pages,
        'pages_with_sheet_number': pages_with_sheet,
        'extraction_rate': (pages_with_sheet / total_pages * 100) if total_pages > 0 else 0,
        'methods': {},
        'per_pdf': [],
    }

    for result in all_results:
        for page in result.get('pages', []):
            method = page['title_method'] or 'none'
            stats['methods'][method] = stats['methods'].get(method, 0) + 1

        pdf_pages = len(result.get('pages', []))
        pdf_sheets = sum(1 for p in result.get('pages', []) if p['sheet_number'])
        stats['per_pdf'].append({
            'name': result['pdf_name'],
            'pages': pdf_pages,
            'sheets': pdf_sheets,
            'rate': (pdf_sheets / pdf_pages * 100) if pdf_pages > 0 else 0,
            'success': result['success'],
        })

    with open(OUTPUT_DIR / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total PDFs: {stats['total_pdfs']} ({stats['successful_pdfs']} OK, {stats['failed_pdfs']} failed)")
    print(f"Total pages: {stats['total_pages']}")
    print(f"Sheets extracted: {stats['pages_with_sheet_number']} ({stats['extraction_rate']:.1f}%)")

    print(f"\nMethods used:")
    for method, count in sorted(stats['methods'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_pages'] * 100 if stats['total_pages'] > 0 else 0
        print(f"  {method}: {count} ({pct:.1f}%)")

    print(f"\nOutput files:")
    print(f"  {OUTPUT_DIR / 'extraction.log'}")
    print(f"  {OUTPUT_DIR / 'results.json'}")
    print(f"  {OUTPUT_DIR / 'summary.csv'}")
    print(f"  {OUTPUT_DIR / 'statistics.json'}")

    return stats


if __name__ == "__main__":
    stats = main()
    sys.exit(0 if stats['failed_pdfs'] == 0 else 1)
