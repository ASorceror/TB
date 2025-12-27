"""
Minimal test - test just one PDF, one page at a time.
Uses the same flow as main.py but with detailed logging.
"""
import sys
import logging
from pathlib import Path

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

sys.path.insert(0, str(Path(__file__).parent))

from main import BlueprintProcessor
from core.pdf_handler import PDFHandler

# Ground truth for test cases
GROUND_TRUTH = {
    ("Janesville Nissan Full set Issued for Bids.pdf", 1): {"sn": "T-1", "title": "COVER SHEET"},
    ("Janesville Nissan Full set Issued for Bids.pdf", 2): {"sn": "A-1", "title": ""},
    ("Janesville Nissan Full set Issued for Bids.pdf", 3): {"sn": "D-1", "title": "DEMO FLOOR PLAN"},
    ("Janesville Nissan Full set Issued for Bids.pdf", 4): {"sn": "A-2.1", "title": "FLOOR PLAN/ HALL TYPES"},
    ("Janesville Nissan Full set Issued for Bids.pdf", 5): {"sn": "A-2.2", "title": "ENLARGED FLOOR PLAN MISC. DETAILS"},
    ("0_full_permit_set_chiro_one_evergreen_park.pdf", 1): {"sn": "A-1", "title": "GENERAL NOTES/LEGEND, CODE MATRIX, SITE/KEY PLAN"},
    ("0_full_permit_set_chiro_one_evergreen_park.pdf", 2): {"sn": "A-2", "title": "FLOOR PLAN"},
    ("0_full_permit_set_chiro_one_evergreen_park.pdf", 3): {"sn": "A-3", "title": "REFLECTED CEILING PLAN"},
    ("0_full_permit_set_chiro_one_evergreen_park.pdf", 4): {"sn": "A-4", "title": "FURNITURE/FINISH PLAN"},
    ("0_full_permit_set_chiro_one_evergreen_park.pdf", 5): {"sn": "A-5", "title": "WALL SECTIONS"},
    ("18222 midland tx - final 2-19-19 rev 1.pdf", 1): {"sn": "A0.0", "title": "COVER SHEET"},
    ("18222 midland tx - final 2-19-19 rev 1.pdf", 2): {"sn": "A0.1", "title": "SITE PLAN"},
    ("18222 midland tx - final 2-19-19 rev 1.pdf", 3): {"sn": "A1.1", "title": "DEMOLITION MEZZANINE PLAN"},
}


def test_pdfs():
    """Test extraction on multiple PDFs/pages."""
    test_data_dir = Path(r"C:\BPA-CC\tests\e2e\test_data")
    processor = BlueprintProcessor()

    # Group tests by PDF
    tests_by_pdf = {}
    for (pdf_name, page_num), expected in GROUND_TRUTH.items():
        if pdf_name not in tests_by_pdf:
            tests_by_pdf[pdf_name] = []
        tests_by_pdf[pdf_name].append((page_num, expected))

    stats = {'correct': 0, 'wrong': 0, 'empty': 0}

    for pdf_name, tests in tests_by_pdf.items():
        pdf_path = test_data_dir / pdf_name
        if not pdf_path.exists():
            print(f"\nSkipping {pdf_name} - not found")
            continue

        print(f"\n{'='*60}")
        print(f"PDF: {pdf_name}")

        try:
            with PDFHandler(pdf_path) as handler:
                for page_num, expected in sorted(tests):
                    try:
                        # process_page uses 0-indexed
                        result = processor.process_page(handler, page_num - 1, pdf_name)

                        extracted_sn = (result.get('sheet_number') or '').upper().strip()
                        expected_sn = expected['sn'].upper().strip()
                        method = result.get('extraction_details', {}).get('sheet_number', 'unknown')

                        if extracted_sn == expected_sn:
                            status = 'OK'
                            stats['correct'] += 1
                        elif not extracted_sn:
                            status = 'EMPTY'
                            stats['empty'] += 1
                        else:
                            status = 'WRONG'
                            stats['wrong'] += 1

                        print(f"  Page {page_num}: [{status}] got '{extracted_sn or '(empty)'}' via {method}, expected '{expected_sn}'")

                    except Exception as e:
                        print(f"  Page {page_num}: ERROR - {e}")

        except Exception as e:
            print(f"  PDF ERROR: {e}")

    # Summary
    total = stats['correct'] + stats['wrong'] + stats['empty']
    print(f"\n{'='*60}")
    print(f"SUMMARY: {stats['correct']}/{total} correct ({100*stats['correct']/total:.1f}%)")
    print(f"  Correct: {stats['correct']}")
    print(f"  Wrong:   {stats['wrong']}")
    print(f"  Empty:   {stats['empty']}")


if __name__ == "__main__":
    test_pdfs()
