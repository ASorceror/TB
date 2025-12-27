"""
Extract from Saved Crops

Runs extraction on saved title block crops (from crop-only run).
Samples N pages from each PDF to test extraction accuracy.
"""

import sys
import json
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from core.vision_extractor import VisionExtractor


def extract_from_crops(crops_dir: Path, samples_per_pdf: int = 5):
    """Extract from saved crops."""

    print("="*80)
    print(f"EXTRACTION FROM SAVED CROPS")
    print(f"Samples per PDF: {samples_per_pdf}")
    print("="*80)

    extractor = VisionExtractor()
    if not extractor.is_available():
        print("ERROR: Vision API not available")
        return

    # Get all PDF folders
    pdf_folders = sorted([f for f in crops_dir.iterdir() if f.is_dir()])

    all_results = []

    for pdf_idx, pdf_folder in enumerate(pdf_folders, 1):
        pdf_name = pdf_folder.name
        print(f"\n[{pdf_idx}/{len(pdf_folders)}] {pdf_name}")

        # Get crop files
        crop_files = sorted(pdf_folder.glob("page_*_titleblock.png"))

        if not crop_files:
            print("  No crops found")
            continue

        # Sample evenly across the PDF
        total = len(crop_files)
        if total <= samples_per_pdf:
            samples = crop_files
        else:
            step = total // samples_per_pdf
            samples = [crop_files[i * step] for i in range(samples_per_pdf)]

        print(f"  Total pages: {total}, Sampling: {len(samples)}")

        pdf_results = []

        for crop_path in samples:
            page_num = int(crop_path.stem.split('_')[1])

            try:
                crop_img = Image.open(crop_path)

                # Extract sheet number
                number_result = extractor.extract_sheet_number(crop_img)
                sheet_number = number_result.get('sheet_number')

                # Extract sheet title
                title_result = extractor.extract_title(crop_img)
                sheet_title = title_result.get('title')

                pdf_results.append({
                    'page': page_num,
                    'sheet_number': sheet_number,
                    'sheet_title': sheet_title
                })

                print(f"    Page {page_num:3d}: {sheet_number or 'N/A':>10} | {(sheet_title[:35] + '...') if sheet_title and len(sheet_title) > 35 else (sheet_title or 'N/A')}")

            except Exception as e:
                print(f"    Page {page_num}: ERROR - {e}")

        all_results.append({
            'pdf': pdf_name,
            'total_pages': total,
            'samples': pdf_results
        })

        extractor.reset_pdf_counter()

    # Summary
    print("\n" + "="*100)
    print("EXTRACTION SUMMARY")
    print("="*100)

    total_samples = 0
    total_numbers = 0
    total_titles = 0

    for r in all_results:
        samples = r.get('samples', [])
        numbers = sum(1 for s in samples if s.get('sheet_number'))
        titles = sum(1 for s in samples if s.get('sheet_title'))

        total_samples += len(samples)
        total_numbers += numbers
        total_titles += titles

        print(f"{r['pdf'][:45]:<45} {len(samples):>3} samples: {numbers}/{len(samples)} numbers, {titles}/{len(samples)} titles")

    print("-"*100)
    print(f"TOTAL: {total_samples} samples")
    print(f"Sheet Numbers: {total_numbers}/{total_samples} ({total_numbers/total_samples*100:.1f}%)")
    print(f"Sheet Titles:  {total_titles}/{total_samples} ({total_titles/total_samples*100:.1f}%)")

    # Save results
    results_path = crops_dir / "extraction_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    crops_dir = Path(r"C:\tb\blueprint_processor\output\complete_extraction\20251227_004829")
    extract_from_crops(crops_dir, samples_per_pdf=3)
