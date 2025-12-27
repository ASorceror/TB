"""
Quick single-page test to validate extraction on specific PDFs and pages.
Uses BlueprintProcessor from main.py which has the process_page method.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import BlueprintProcessor
from core.pdf_handler import PDFHandler

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

def run_quick_test():
    """Run quick test on sample pages from each PDF."""
    gt_path = Path(r"C:\tb\blueprint_processor\output\ground_truth_full.csv")
    ground_truth = load_ground_truth(gt_path)
    print(f"Ground truth: {len(ground_truth)} entries")

    test_data_dir = Path(r"C:\BPA-CC\tests\e2e\test_data")

    # Get list of unique PDFs from ground truth
    pdf_names = sorted(set(pdf for pdf, _ in ground_truth.keys()))
    print(f"PDFs in ground truth: {len(pdf_names)}")

    processor = BlueprintProcessor()

    stats = {
        'sn_correct': 0,
        'sn_empty': 0,
        'sn_wrong': 0,
        'title_correct': 0,
        'title_empty': 0,
        'title_wrong': 0,
        'total': 0,
    }

    failures = []

    for pdf_name in pdf_names:
        pdf_path = test_data_dir / pdf_name
        if not pdf_path.exists():
            print(f"\nSkipping {pdf_name} - file not found")
            continue

        # Get pages with ground truth for this PDF
        gt_pages = sorted([page for (pn, page) in ground_truth.keys() if pn == pdf_name])

        # Test first 3 pages only for speed
        sample_pages = gt_pages[:3]

        print(f"\n{'='*60}")
        print(f"PDF: {pdf_name}")
        print(f"Testing pages: {sample_pages}")

        try:
            # Open PDF once for all pages
            with PDFHandler(pdf_path) as handler:
                for page_num in sample_pages:
                    key = (pdf_name, page_num)
                    expected = ground_truth[key]

                    try:
                        # process_page uses 0-indexed pages
                        result = processor.process_page(handler, page_num - 1, pdf_name)
                        if result is None:
                            print(f"  Page {page_num}: ERROR - no result")
                            continue

                        stats['total'] += 1

                        # Sheet number comparison
                        extracted_sn = normalize(result.get('sheet_number', ''))
                        expected_sn = normalize(expected.get('sheet_number', ''))
                        sn_method = result.get('extraction_details', {}).get('sheet_number', 'unknown')

                        if extracted_sn == expected_sn:
                            sn_status = 'CORRECT'
                            stats['sn_correct'] += 1
                        elif not extracted_sn:
                            sn_status = 'EMPTY'
                            stats['sn_empty'] += 1
                            failures.append((pdf_name, page_num, 'sheet_number', expected_sn, extracted_sn, sn_method))
                        else:
                            sn_status = 'WRONG'
                            stats['sn_wrong'] += 1
                            failures.append((pdf_name, page_num, 'sheet_number', expected_sn, extracted_sn, sn_method))

                        # Sheet title comparison
                        extracted_title = normalize(result.get('sheet_title', ''))
                        expected_title = normalize(expected.get('sheet_title', ''))
                        title_method = result.get('extraction_details', {}).get('sheet_title', 'unknown')

                        if extracted_title == expected_title:
                            title_status = 'CORRECT'
                            stats['title_correct'] += 1
                        elif not extracted_title:
                            title_status = 'EMPTY'
                            stats['title_empty'] += 1
                        else:
                            title_status = 'WRONG'
                            stats['title_wrong'] += 1

                        print(f"  Page {page_num}:")
                        print(f"    SN:    {sn_status} - got '{extracted_sn or '(empty)'}' via {sn_method}, expected '{expected_sn}'")
                        print(f"    Title: {title_status} - got '{extracted_title[:40] if extracted_title else '(empty)'}...' via {title_method}")

                    except Exception as e:
                        print(f"  Page {page_num}: ERROR - {e}")
                        import traceback
                        traceback.print_exc()

        except Exception as e:
            print(f"  PDF ERROR: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total = stats['total']
    if total > 0:
        print(f"\nSheet Numbers:")
        print(f"  Correct: {stats['sn_correct']}/{total} ({100*stats['sn_correct']/total:.1f}%)")
        print(f"  Empty:   {stats['sn_empty']}/{total} ({100*stats['sn_empty']/total:.1f}%)")
        print(f"  Wrong:   {stats['sn_wrong']}/{total} ({100*stats['sn_wrong']/total:.1f}%)")

        print(f"\nSheet Titles:")
        print(f"  Correct: {stats['title_correct']}/{total} ({100*stats['title_correct']/total:.1f}%)")
        print(f"  Empty:   {stats['title_empty']}/{total} ({100*stats['title_empty']/total:.1f}%)")
        print(f"  Wrong:   {stats['title_wrong']}/{total} ({100*stats['title_wrong']/total:.1f}%)")

    # Print failure details
    if failures:
        print(f"\n{'='*60}")
        print("SHEET NUMBER FAILURES")
        print("="*60)
        for pdf, page, field, expected, got, method in failures:
            print(f"  {pdf}:{page} - expected '{expected}', got '{got}' via {method}")

if __name__ == "__main__":
    run_quick_test()
