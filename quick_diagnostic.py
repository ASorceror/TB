"""
Quick diagnostic test - samples 3 pages from each PDF to identify failure patterns.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

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
    """Run quick diagnostic on sample pages from each PDF."""
    # Load ground truth
    gt_path = Path(r"C:\tb\blueprint_processor\output\ground_truth_full.csv")
    ground_truth = load_ground_truth(gt_path)
    print(f"Ground truth: {len(ground_truth)} entries")

    # Find PDFs
    test_data_dir = Path(r"C:\BPA-CC\tests\e2e\test_data")
    pdf_files = sorted(list(test_data_dir.glob("*.pdf")))
    print(f"Found {len(pdf_files)} PDFs")

    # Get count of pages with ground truth per PDF
    pdf_gt_counts = defaultdict(int)
    for (pdf_name, page_num) in ground_truth.keys():
        pdf_gt_counts[pdf_name] += 1

    print("\nGround truth pages per PDF:")
    for pdf_name, count in sorted(pdf_gt_counts.items()):
        print(f"  {pdf_name[:40]}: {count} pages")

    # Process sample pages from each PDF
    processor = BlueprintProcessor()

    results_by_pdf = {}

    for pdf_path in pdf_files:
        pdf_name = pdf_path.name
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_name}")

        # Find which pages have ground truth for this PDF
        gt_pages = [page_num for (pn, page_num) in ground_truth.keys() if pn == pdf_name]
        if not gt_pages:
            print("  No ground truth entries")
            continue

        # Sample: first 3 pages with ground truth
        sample_pages = sorted(gt_pages)[:3]
        print(f"  Sampling pages: {sample_pages}")

        pdf_results = {
            'correct': 0,
            'total': 0,
            'empty': 0,
            'wrong': 0,
            'details': []
        }

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

                pdf_results['total'] += 1

                if extracted_sn == expected_sn:
                    status = 'CORRECT'
                    pdf_results['correct'] += 1
                elif not extracted_sn:
                    status = 'EMPTY'
                    pdf_results['empty'] += 1
                else:
                    status = 'WRONG'
                    pdf_results['wrong'] += 1

                detail = {
                    'page': page_num,
                    'extracted': extracted_sn or '(empty)',
                    'expected': expected_sn or '(empty)',
                    'method': method,
                    'status': status
                }
                pdf_results['details'].append(detail)

                print(f"  Page {page_num}: {status}")
                print(f"    Extracted: '{extracted_sn or '(empty)'}' via {method}")
                print(f"    Expected:  '{expected_sn or '(empty)'}'")

        except Exception as e:
            print(f"  ERROR: {e}")
            pdf_results['error'] = str(e)

        results_by_pdf[pdf_name] = pdf_results

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    total_correct = sum(r['correct'] for r in results_by_pdf.values())
    total_tested = sum(r['total'] for r in results_by_pdf.values())
    total_empty = sum(r['empty'] for r in results_by_pdf.values())
    total_wrong = sum(r['wrong'] for r in results_by_pdf.values())

    print(f"\nOverall: {total_correct}/{total_tested} correct ({100*total_correct/total_tested:.1f}%)")
    print(f"  Empty extractions: {total_empty}")
    print(f"  Wrong extractions: {total_wrong}")

    # Categorize PDFs by success rate
    print("\nPDFs by success rate:")

    working = []
    partial = []
    failing = []

    for pdf_name, results in sorted(results_by_pdf.items()):
        if results['total'] == 0:
            continue
        rate = results['correct'] / results['total']
        if rate >= 0.9:
            working.append((pdf_name, results))
        elif rate >= 0.3:
            partial.append((pdf_name, results))
        else:
            failing.append((pdf_name, results))

    print(f"\n  WORKING (>=90%): {len(working)}")
    for pdf_name, r in working:
        print(f"    {pdf_name[:40]}: {r['correct']}/{r['total']}")

    print(f"\n  PARTIAL (30-90%): {len(partial)}")
    for pdf_name, r in partial:
        print(f"    {pdf_name[:40]}: {r['correct']}/{r['total']}")

    print(f"\n  FAILING (<30%): {len(failing)}")
    for pdf_name, r in failing:
        print(f"    {pdf_name[:40]}: {r['correct']}/{r['total']} (empty={r['empty']}, wrong={r['wrong']})")
        for d in r['details']:
            print(f"      p{d['page']}: '{d['extracted']}' vs '{d['expected']}' [{d['method']}]")

if __name__ == "__main__":
    run_diagnostic()
