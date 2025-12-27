"""
Expanded test - tests more pages per PDF to verify extraction accuracy.
"""
import csv
import sys
import time
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


def run_expanded_test():
    """Run expanded test on more pages per PDF."""
    gt_path = Path(r"C:\tb\blueprint_processor\output\ground_truth_full.csv")
    ground_truth = load_ground_truth(gt_path)
    print(f"Ground truth: {len(ground_truth)} entries")

    test_data_dir = Path(r"C:\BPA-CC\tests\e2e\test_data")

    # Get list of unique PDFs from ground truth
    pdf_names = sorted(set(pdf for pdf, _ in ground_truth.keys()))
    print(f"PDFs in ground truth: {len(pdf_names)}")

    processor = BlueprintProcessor()

    stats = {'correct': 0, 'wrong': 0, 'empty': 0}
    failures = []

    for pdf_name in pdf_names:
        pdf_path = test_data_dir / pdf_name
        if not pdf_path.exists():
            print(f"\nSkipping {pdf_name} - not found")
            continue

        # Get pages with ground truth for this PDF
        gt_pages = sorted([page for (pn, page) in ground_truth.keys() if pn == pdf_name])

        # Test first 5 pages (expanded from 3)
        sample_pages = gt_pages[:5]

        print(f"\n{'='*60}")
        print(f"PDF: {pdf_name}")
        print(f"Testing pages: {sample_pages}")

        try:
            with PDFHandler(pdf_path) as handler:
                for page_num in sample_pages:
                    key = (pdf_name, page_num)
                    expected = ground_truth[key]

                    try:
                        result = processor.process_page(handler, page_num - 1, pdf_name)

                        extracted_sn = normalize(result.get('sheet_number', ''))
                        expected_sn = normalize(expected.get('sheet_number', ''))
                        method = result.get('extraction_details', {}).get('sheet_number', 'unknown')

                        if extracted_sn == expected_sn:
                            status = 'OK'
                            stats['correct'] += 1
                        elif not extracted_sn:
                            status = 'EMPTY'
                            stats['empty'] += 1
                            failures.append((pdf_name, page_num, expected_sn, extracted_sn, method))
                        else:
                            status = 'WRONG'
                            stats['wrong'] += 1
                            failures.append((pdf_name, page_num, expected_sn, extracted_sn, method))

                        print(f"  Page {page_num}: [{status}] got '{extracted_sn or '(empty)'}' via {method}, expected '{expected_sn}'")

                        # Small delay to avoid rate limiting
                        time.sleep(0.5)

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

    if failures:
        print(f"\n{'='*60}")
        print("FAILURES")
        print("="*60)
        for pdf, page, expected, got, method in failures:
            print(f"  {pdf}:{page} - expected '{expected}', got '{got or '(empty)'}' via {method}")


if __name__ == "__main__":
    run_expanded_test()
