"""
Fast diagnostic - only tests small PDFs (<50 pages) for quick validation.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
import fitz

sys.path.insert(0, str(Path(__file__).parent))

from main import BlueprintProcessor

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

def run_diagnostic():
    """Run quick diagnostic on small PDFs only."""
    # Load ground truth
    gt_path = Path(r"C:\tb\blueprint_processor\output\ground_truth_full.csv")
    ground_truth = load_ground_truth(gt_path)
    print(f"Ground truth: {len(ground_truth)} entries")

    # Find PDFs
    test_data_dir = Path(r"C:\BPA-CC\tests\e2e\test_data")
    pdf_files = sorted(list(test_data_dir.glob("*.pdf")))
    print(f"Found {len(pdf_files)} PDFs total")

    # Filter to small PDFs only
    small_pdfs = []
    for pdf_path in pdf_files:
        try:
            with fitz.open(pdf_path) as doc:
                if doc.page_count <= 50:
                    small_pdfs.append((pdf_path, doc.page_count))
        except Exception as e:
            print(f"Error reading {pdf_path.name}: {e}")

    print(f"Testing {len(small_pdfs)} small PDFs (<= 50 pages)")
    for pdf_path, page_count in small_pdfs:
        print(f"  {pdf_path.name}: {page_count} pages")

    # Process PDFs
    processor = BlueprintProcessor()

    total_correct = 0
    total_tested = 0
    total_empty = 0
    total_wrong = 0

    for pdf_path, page_count in small_pdfs:
        pdf_name = pdf_path.name
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_name} ({page_count} pages)")

        # Find which pages have ground truth
        gt_pages = [page_num for (pn, page_num) in ground_truth.keys() if pn == pdf_name]
        if not gt_pages:
            print("  No ground truth entries")
            continue

        sample_pages = sorted(gt_pages)[:5]  # First 5 pages with ground truth
        print(f"  Testing pages: {sample_pages}")

        try:
            all_results = processor.process_pdf(pdf_path)

            for page_result in all_results:
                page_num = page_result.get('page_number', 0)

                if page_num not in sample_pages:
                    continue

                key = (pdf_name, page_num)
                expected = ground_truth.get(key, {})

                extracted_sn = normalize(page_result.get('sheet_number', ''))
                expected_sn = normalize(expected.get('sheet_number', ''))
                method = page_result.get('extraction_details', {}).get('sheet_number', 'unknown')

                total_tested += 1

                if extracted_sn == expected_sn:
                    status = 'CORRECT'
                    total_correct += 1
                elif not extracted_sn:
                    status = 'EMPTY'
                    total_empty += 1
                else:
                    status = 'WRONG'
                    total_wrong += 1

                print(f"  Page {page_num}: {status}")
                print(f"    Extracted: '{extracted_sn or '(empty)'}' via {method}")
                print(f"    Expected:  '{expected_sn or '(empty)'}'")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "="*60)
    print("FAST DIAGNOSTIC SUMMARY")
    print("="*60)

    if total_tested > 0:
        print(f"\nOverall: {total_correct}/{total_tested} correct ({100*total_correct/total_tested:.1f}%)")
        print(f"  Empty extractions: {total_empty} ({100*total_empty/total_tested:.1f}%)")
        print(f"  Wrong extractions: {total_wrong} ({100*total_wrong/total_tested:.1f}%)")
    else:
        print("No pages tested!")

if __name__ == "__main__":
    run_diagnostic()
