"""
Extract from ALL Saved Crops

Runs extraction on all 2,435 title block crops.
Saves progress incrementally so it can be resumed if interrupted.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from core.vision_extractor import VisionExtractor


def extract_all_crops(crops_dir: Path):
    """Extract from all saved crops with progress saving."""

    print("="*80)
    print("EXTRACTION FROM ALL SAVED CROPS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    extractor = VisionExtractor()
    if not extractor.is_available():
        print("ERROR: Vision API not available")
        return

    # Get all PDF folders
    pdf_folders = sorted([f for f in crops_dir.iterdir() if f.is_dir()])

    # Load existing progress if any
    progress_file = crops_dir / "extraction_progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            all_results = json.load(f)
        print(f"Resuming from existing progress file")
        completed_pdfs = {r['pdf'] for r in all_results if r.get('completed')}
    else:
        all_results = []
        completed_pdfs = set()

    total_extracted = 0
    total_pages = 0

    for pdf_idx, pdf_folder in enumerate(pdf_folders, 1):
        pdf_name = pdf_folder.name

        # Skip if already completed
        if pdf_name in completed_pdfs:
            print(f"\n[{pdf_idx}/{len(pdf_folders)}] {pdf_name} - ALREADY DONE, skipping")
            # Count for totals
            for r in all_results:
                if r['pdf'] == pdf_name:
                    total_pages += len(r.get('pages', []))
                    total_extracted += sum(1 for p in r.get('pages', []) if p.get('sheet_number'))
            continue

        print(f"\n[{pdf_idx}/{len(pdf_folders)}] {pdf_name}")

        # Get crop files
        crop_files = sorted(pdf_folder.glob("page_*_titleblock.png"))

        if not crop_files:
            print("  No crops found")
            continue

        print(f"  Processing {len(crop_files)} pages...")

        pdf_result = {
            'pdf': pdf_name,
            'pages': [],
            'completed': False
        }

        for crop_idx, crop_path in enumerate(crop_files, 1):
            page_num = int(crop_path.stem.split('_')[1])

            # Progress indicator
            if crop_idx % 10 == 0 or crop_idx == len(crop_files):
                print(f"    {crop_idx}/{len(crop_files)} pages...")

            try:
                crop_img = Image.open(crop_path)

                # Extract sheet number
                number_result = extractor.extract_sheet_number(crop_img)
                sheet_number = number_result.get('sheet_number')

                # Extract sheet title
                title_result = extractor.extract_title(crop_img)
                sheet_title = title_result.get('title')

                pdf_result['pages'].append({
                    'page': page_num,
                    'sheet_number': sheet_number,
                    'sheet_title': sheet_title,
                    'number_conf': number_result.get('confidence', 0),
                    'title_conf': title_result.get('confidence', 0)
                })

                if sheet_number:
                    total_extracted += 1
                total_pages += 1

            except Exception as e:
                print(f"    Page {page_num}: ERROR - {e}")
                pdf_result['pages'].append({
                    'page': page_num,
                    'error': str(e)
                })
                total_pages += 1

        pdf_result['completed'] = True

        # Update results
        # Remove old entry if exists
        all_results = [r for r in all_results if r['pdf'] != pdf_name]
        all_results.append(pdf_result)

        # Save progress after each PDF
        with open(progress_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Print PDF summary
        numbers = sum(1 for p in pdf_result['pages'] if p.get('sheet_number'))
        titles = sum(1 for p in pdf_result['pages'] if p.get('sheet_title'))
        print(f"  Done: {numbers}/{len(crop_files)} numbers, {titles}/{len(crop_files)} titles")

        extractor.reset_pdf_counter()

    # Final summary
    print("\n" + "="*100)
    print("FINAL EXTRACTION SUMMARY")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)

    grand_total_pages = 0
    grand_total_numbers = 0
    grand_total_titles = 0

    print(f"\n{'PDF':<45} {'Pages':>7} {'Numbers':>10} {'Titles':>10}")
    print("-"*80)

    for r in sorted(all_results, key=lambda x: x['pdf']):
        pages = len(r.get('pages', []))
        numbers = sum(1 for p in r.get('pages', []) if p.get('sheet_number'))
        titles = sum(1 for p in r.get('pages', []) if p.get('sheet_title'))

        grand_total_pages += pages
        grand_total_numbers += numbers
        grand_total_titles += titles

        pct_num = f"{numbers/pages*100:.0f}%" if pages > 0 else "N/A"
        pct_tit = f"{titles/pages*100:.0f}%" if pages > 0 else "N/A"

        print(f"{r['pdf'][:44]:<45} {pages:>7} {numbers:>6} ({pct_num:>3}) {titles:>6} ({pct_tit:>3})")

    print("-"*80)
    print(f"{'TOTAL':<45} {grand_total_pages:>7} {grand_total_numbers:>10} {grand_total_titles:>10}")

    if grand_total_pages > 0:
        print(f"\nSheet Numbers: {grand_total_numbers}/{grand_total_pages} ({grand_total_numbers/grand_total_pages*100:.1f}%)")
        print(f"Sheet Titles:  {grand_total_titles}/{grand_total_pages} ({grand_total_titles/grand_total_pages*100:.1f}%)")

    # Save final results
    final_results_path = crops_dir / "final_extraction_results.json"
    with open(final_results_path, 'w') as f:
        json.dump({
            'completed_at': datetime.now().isoformat(),
            'total_pages': grand_total_pages,
            'total_numbers': grand_total_numbers,
            'total_titles': grand_total_titles,
            'results': all_results
        }, f, indent=2)

    print(f"\nFinal results saved to: {final_results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract from all saved crops")
    parser.add_argument('--crops-dir', type=str, required=True,
                        help='Directory containing crop folders')
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    extract_all_crops(crops_dir)
