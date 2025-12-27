"""
V5.0 Accuracy Test - Tests extraction on all available PDFs with ground truth.
"""

import csv
import sys
from pathlib import Path
from tqdm import tqdm

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

def run_accuracy_test():
    """Run accuracy test on all PDFs."""
    # Load ground truth
    gt_path = Path(r"C:\tb\blueprint_processor\output\ground_truth_full.csv")
    if not gt_path.exists():
        print(f"Ground truth file not found: {gt_path}")
        return

    ground_truth = load_ground_truth(gt_path)
    print(f"Ground truth: {len(ground_truth)} entries")

    # Find PDFs
    test_data_dir = Path(r"C:\BPA-CC\tests\e2e\test_data")
    pdf_files = list(test_data_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process")

    # Process PDFs
    processor = BlueprintProcessor()

    sheet_num_correct = 0
    sheet_num_total = 0
    sheet_title_correct = 0
    sheet_title_total = 0

    mismatches = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            results = processor.process_pdf(pdf_path)

            for page_result in results:
                page_num = page_result.get('page_number', 0)
                key = (pdf_path.name, page_num)

                if key not in ground_truth:
                    continue

                expected = ground_truth[key]

                # Sheet number
                extracted_sn = normalize(page_result.get('sheet_number', ''))
                expected_sn = normalize(expected.get('sheet_number', ''))
                sheet_num_total += 1
                if extracted_sn == expected_sn:
                    sheet_num_correct += 1
                else:
                    mismatches.append({
                        'pdf': pdf_path.name[:25],
                        'page': page_num,
                        'field': 'sheet_number',
                        'got': extracted_sn or '(empty)',
                        'expected': expected_sn or '(empty)',
                    })

                # Sheet title
                extracted_st = normalize(page_result.get('sheet_title', ''))
                expected_st = normalize(expected.get('sheet_title', ''))
                sheet_title_total += 1
                if extracted_st == expected_st or (expected_st and expected_st in extracted_st) or (extracted_st and extracted_st in expected_st):
                    sheet_title_correct += 1

        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

    # Print results
    print()
    print("=" * 80)
    print("V5.0 ACCURACY RESULTS")
    print("=" * 80)

    sn_acc = sheet_num_correct / sheet_num_total * 100 if sheet_num_total > 0 else 0
    st_acc = sheet_title_correct / sheet_title_total * 100 if sheet_title_total > 0 else 0

    print(f"Sheet Number Accuracy: {sheet_num_correct}/{sheet_num_total} = {sn_acc:.1f}%")
    print(f"Sheet Title Accuracy:  {sheet_title_correct}/{sheet_title_total} = {st_acc:.1f}%")

    # Show first 20 mismatches
    print(f"\nSheet Number Mismatches ({len(mismatches)}):")
    for m in mismatches[:20]:
        print(f"  {m['pdf']} p{m['page']:2d}: Got '{m['got']}' vs Expected '{m['expected']}'")
    if len(mismatches) > 20:
        print(f"  ... and {len(mismatches) - 20} more")

if __name__ == "__main__":
    run_accuracy_test()
