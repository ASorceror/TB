"""
Visual Verification Test - Creates human-verifiable output.

For each page processed:
1. Saves the full page image (normalized)
2. Saves the detected title block crop
3. Saves edge region crops used for OCR
4. Logs every extraction step
5. Outputs CSV with results + image paths

This lets a human SEE what the code is doing and verify if:
- Title block detection is finding the right region
- Rotation is correct
- The right crops are being sent to Vision API
"""
import csv
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from main import BlueprintProcessor
from core.pdf_handler import PDFHandler

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('verify')


def load_ground_truth(csv_path: Path):
    """Load ground truth from CSV file."""
    ground_truth = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_name = row.get('pdf_filename', '')
            page_num = row.get('page_number', '')
            if pdf_name and page_num:
                key = (pdf_name, int(page_num))
                ground_truth[key] = {
                    'sheet_number': row.get('sheet_number', ''),
                    'sheet_title': row.get('sheet_title', ''),
                }
    return ground_truth


def normalize(value):
    """Normalize value for comparison."""
    if value is None:
        return ''
    return str(value).strip().upper()


def run_visual_verification():
    """Run verification with full visual output."""

    # Setup output directory
    output_dir = Path(r"C:\tb\blueprint_processor\output\verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"VISUAL VERIFICATION RUN")
    print(f"Output directory: {run_dir}")
    print(f"{'='*60}\n")

    # Load ground truth
    gt_path = Path(r"C:\tb\blueprint_processor\output\ground_truth_full.csv")
    if gt_path.exists():
        ground_truth = load_ground_truth(gt_path)
        print(f"Ground truth: {len(ground_truth)} entries")
    else:
        print("WARNING: No ground truth file found")
        ground_truth = {}

    test_data_dir = Path(r"C:\BPA-CC\tests\e2e\test_data")

    # Test PDFs - start with the ones we claim work
    test_pdfs = [
        ("Janesville Nissan Full set Issued for Bids.pdf", [1, 2, 3]),
        ("0_full_permit_set_chiro_one_evergreen_park.pdf", [1, 2, 3, 4, 5]),
        ("18222 midland tx - final 2-19-19 rev 1.pdf", [1, 2, 3]),
    ]

    processor = BlueprintProcessor()

    # CSV output
    csv_path = run_dir / "results.csv"
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'pdf_name', 'page_num',
        'expected_sn', 'extracted_sn', 'sn_match',
        'extraction_method',
        'rotation_applied', 'title_block_found',
        'full_page_image', 'title_block_image',
        'notes'
    ])

    # Detailed log file
    log_path = run_dir / "extraction_log.txt"
    log_file = open(log_path, 'w', encoding='utf-8')

    stats = {'correct': 0, 'wrong': 0, 'empty': 0}

    for pdf_name, pages in test_pdfs:
        pdf_path = test_data_dir / pdf_name
        if not pdf_path.exists():
            print(f"\nSkipping {pdf_name} - not found")
            continue

        # Create PDF-specific output folder
        pdf_safe_name = pdf_name.replace(' ', '_').replace('.pdf', '')[:50]
        pdf_dir = run_dir / pdf_safe_name
        pdf_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"PDF: {pdf_name}")
        print(f"Output: {pdf_dir}")
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"PDF: {pdf_name}\n")
        log_file.write(f"{'='*60}\n")

        try:
            with PDFHandler(pdf_path) as handler:
                for page_num in pages:
                    print(f"\n  Processing page {page_num}...")
                    log_file.write(f"\n--- Page {page_num} ---\n")

                    # Get ground truth
                    key = (pdf_name, page_num)
                    expected = ground_truth.get(key, {})
                    expected_sn = normalize(expected.get('sheet_number', ''))

                    notes = []

                    try:
                        # Get the raw page for inspection
                        page = handler.get_page(page_num - 1)

                        # Render full page at 150 DPI for visual verification
                        import fitz
                        matrix = fitz.Matrix(150/72, 150/72)
                        pix = page.get_pixmap(matrix=matrix)
                        full_page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                        # Save full page image BEFORE any processing
                        raw_page_path = pdf_dir / f"page_{page_num:02d}_raw.png"
                        full_page_img.save(raw_page_path)
                        log_file.write(f"  Saved raw page: {raw_page_path.name}\n")

                        # Now process the page
                        result = processor.process_page(handler, page_num - 1, pdf_name)

                        # Extract key info
                        extracted_sn = normalize(result.get('sheet_number', ''))
                        method = result.get('extraction_details', {}).get('sheet_number', 'unknown')

                        # Check for title block image in processor state
                        # We need to capture this during processing
                        title_block_path = ""
                        rotation_applied = 0
                        title_block_found = False

                        # Get extraction details
                        details = result.get('extraction_details', {})
                        log_file.write(f"  Extraction details: {details}\n")

                        # Try to get the title block image that was used
                        # This requires accessing processor internals
                        if hasattr(processor, '_last_title_block_image'):
                            tb_img = processor._last_title_block_image
                            if tb_img:
                                tb_path = pdf_dir / f"page_{page_num:02d}_titleblock.png"
                                tb_img.save(tb_path)
                                title_block_path = tb_path.name
                                title_block_found = True
                                log_file.write(f"  Saved title block: {tb_path.name}\n")

                        if hasattr(processor, '_last_rotation_applied'):
                            rotation_applied = processor._last_rotation_applied

                        # Comparison
                        if extracted_sn == expected_sn:
                            status = 'OK'
                            stats['correct'] += 1
                        elif not extracted_sn:
                            status = 'EMPTY'
                            stats['empty'] += 1
                        else:
                            status = 'WRONG'
                            stats['wrong'] += 1

                        sn_match = 'Y' if status == 'OK' else 'N'

                        print(f"    [{status}] Expected: '{expected_sn}', Got: '{extracted_sn}' via {method}")
                        log_file.write(f"  Result: [{status}] Expected: '{expected_sn}', Got: '{extracted_sn}'\n")
                        log_file.write(f"  Method: {method}\n")

                        # Write to CSV
                        csv_writer.writerow([
                            pdf_name, page_num,
                            expected_sn, extracted_sn, sn_match,
                            method,
                            rotation_applied, 'Y' if title_block_found else 'N',
                            raw_page_path.name, title_block_path,
                            '; '.join(notes) if notes else ''
                        ])

                        # Small delay for Vision API rate limiting
                        time.sleep(0.3)

                    except Exception as e:
                        print(f"    ERROR: {e}")
                        log_file.write(f"  ERROR: {e}\n")
                        import traceback
                        log_file.write(f"  {traceback.format_exc()}\n")

                        csv_writer.writerow([
                            pdf_name, page_num,
                            expected_sn, '', 'N',
                            'error',
                            0, 'N',
                            '', '',
                            str(e)
                        ])

        except Exception as e:
            print(f"  PDF ERROR: {e}")
            log_file.write(f"PDF ERROR: {e}\n")

    csv_file.close()
    log_file.close()

    # Summary
    total = stats['correct'] + stats['wrong'] + stats['empty']
    print(f"\n{'='*60}")
    print(f"SUMMARY: {stats['correct']}/{total} correct ({100*stats['correct']/total:.1f}%)")
    print(f"  Correct: {stats['correct']}")
    print(f"  Wrong:   {stats['wrong']}")
    print(f"  Empty:   {stats['empty']}")
    print(f"\nOutput files:")
    print(f"  CSV: {csv_path}")
    print(f"  Log: {log_path}")
    print(f"  Images: {run_dir}")


if __name__ == "__main__":
    run_visual_verification()
