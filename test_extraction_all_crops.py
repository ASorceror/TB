"""
Test Extraction on ALL Title Block Crops
"""

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from core.vision_extractor import VisionExtractor


def test_all_crops():
    """Test extraction on all title block crops."""

    crop_dir = Path(r"C:\tb\blueprint_processor\output\full_pipeline")
    crop_files = sorted(crop_dir.glob("*_titleblock.png"))

    print("="*90)
    print("EXTRACTION TEST - ALL CROPS")
    print(f"Testing {len(crop_files)} title block crops")
    print("="*90)

    extractor = VisionExtractor()

    if not extractor.is_available():
        print("ERROR: Vision API not available")
        return

    results = []

    for i, crop_path in enumerate(crop_files, 1):
        pdf_name = crop_path.stem.replace("_titleblock", "")
        print(f"\n[{i}/{len(crop_files)}] {pdf_name[:45]}...")

        try:
            crop_img = Image.open(crop_path)

            # Extract both
            title_result = extractor.extract_title(crop_img)
            number_result = extractor.extract_sheet_number(crop_img)

            sheet_title = title_result.get('title')
            sheet_number = number_result.get('sheet_number')

            print(f"    Sheet #: {sheet_number or 'N/A'}")
            print(f"    Title: {(sheet_title[:40] + '...') if sheet_title and len(sheet_title) > 40 else (sheet_title or 'N/A')}")

            results.append({
                'pdf': pdf_name[:40],
                'title': sheet_title,
                'number': sheet_number
            })

        except Exception as e:
            print(f"    ERROR: {e}")

    # Summary table
    print("\n" + "="*100)
    print("FULL EXTRACTION RESULTS")
    print("="*100)
    print(f"{'PDF':<42} {'Sheet #':>10} {'Sheet Title':<40}")
    print("-"*100)

    for r in results:
        title = (r['title'][:37] + "...") if r['title'] and len(r['title']) > 40 else (r['title'] or "---")
        number = r['number'] or "---"
        print(f"{r['pdf']:<42} {number:>10} {title:<40}")

    # Statistics
    titles_found = sum(1 for r in results if r['title'])
    numbers_found = sum(1 for r in results if r['number'])
    total = len(results)

    print("-"*100)
    print(f"\nSHEET NUMBERS: {numbers_found}/{total} ({numbers_found/total*100:.0f}%)")
    print(f"SHEET TITLES:  {titles_found}/{total} ({titles_found/total*100:.0f}%)")

    if numbers_found == total:
        print("\n✓ PERFECT sheet number extraction!")


if __name__ == "__main__":
    test_all_crops()
