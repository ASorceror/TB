"""
Complete Extraction Test - All Pages, All PDFs

Processes every page of every PDF in the test folder:
1. Detect title block boundary (using sample pages)
2. Crop title block from ALL pages
3. Extract sheet number and title from each crop
4. Save all crops and results

Usage:
    python run_complete_extraction.py              # Crop + Extract (slow)
    python run_complete_extraction.py --crop-only  # Just save crops (fast)
    python run_complete_extraction.py --max-pages 50  # Limit pages per PDF
    python run_complete_extraction.py --input-dir "path/to/pdfs"  # Custom input folder
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.title_block_detector import TitleBlockDetector
from core.vision_extractor import VisionExtractor


def run_complete_extraction(crop_only=False, max_pages=None, input_dir=None):
    """Run complete extraction on all pages of all PDFs."""

    test_dir = Path(input_dir) if input_dir else Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")
    output_base = Path(r"C:\tb\blueprint_processor\output\complete_extraction")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all PDFs
    all_pdfs = sorted(test_dir.glob("*.pdf"))

    mode = "CROP ONLY" if crop_only else "CROP + EXTRACT"
    print("="*80)
    print(f"COMPLETE EXTRACTION TEST - {mode}")
    print(f"PDFs: {len(all_pdfs)}")
    if max_pages:
        print(f"Max pages per PDF: {max_pages}")
    print(f"Output: {output_dir}")
    print("="*80)

    # Initialize components
    detector = TitleBlockDetector(use_ai_refinement=False)  # Fast mode
    extractor = None

    if not crop_only:
        extractor = VisionExtractor()
        if not extractor.is_available():
            print("WARNING: Vision API not available - running in crop-only mode")
            crop_only = True

    all_results = []
    total_pages_processed = 0
    total_extractions = 0

    for pdf_idx, pdf_path in enumerate(all_pdfs, 1):
        pdf_name = pdf_path.name
        safe_name = pdf_name.replace(' ', '_').replace('.pdf', '')[:30]

        print(f"\n{'='*70}")
        print(f"[{pdf_idx}/{len(all_pdfs)}] {pdf_name}")

        # Create output folder for this PDF
        pdf_output_dir = output_dir / safe_name
        pdf_output_dir.mkdir(exist_ok=True)

        pdf_results = {
            'pdf': pdf_name,
            'pages': []
        }

        try:
            with PDFHandler(pdf_path) as handler:
                total_pages = handler.page_count
                print(f"  Total pages: {total_pages}")

                # Step 1: Detect title block using sample pages
                print("  Detecting title block boundary...")
                if total_pages > 5:
                    sample_indices = [1, 3, 5]  # 0-indexed: pages 2, 4, 6
                else:
                    sample_indices = [1, 2] if total_pages >= 3 else [0]

                sample_images = []
                for idx in sample_indices:
                    if idx < total_pages:
                        img = handler.get_page_image(idx, dpi=100)
                        sample_images.append(img)

                detection = detector.detect(sample_images, strategy='balanced')
                x1 = detection['x1']
                print(f"  Title block x1: {x1:.3f} ({detection['width_pct']*100:.1f}% width)")

                pdf_results['title_block_x1'] = x1
                pdf_results['title_block_width_pct'] = detection['width_pct']

                # Step 2: Process ALL pages (or up to max_pages)
                pages_to_process = total_pages
                if max_pages and max_pages < total_pages:
                    pages_to_process = max_pages
                    print(f"  Processing {pages_to_process} of {total_pages} pages (limited)...")
                else:
                    print(f"  Processing all {total_pages} pages...")

                for page_num in range(1, pages_to_process + 1):
                    page_idx = page_num - 1

                    # Progress indicator
                    if page_num % 10 == 0 or page_num == pages_to_process:
                        print(f"    Page {page_num}/{total_pages}...")

                    try:
                        # Get page image at higher DPI for extraction
                        page_img = handler.get_page_image(page_idx, dpi=150)
                        width, height = page_img.size

                        # Crop title block
                        x1_px = int(x1 * width)
                        title_block_crop = page_img.crop((x1_px, 0, width, height))

                        # Save crop
                        crop_filename = f"page_{page_num:03d}_titleblock.png"
                        crop_path = pdf_output_dir / crop_filename
                        title_block_crop.save(crop_path)

                        # Extract sheet number and title (if not crop_only mode)
                        page_result = {
                            'page': page_num,
                            'crop_file': crop_filename,
                            'crop_size': title_block_crop.size
                        }

                        if not crop_only and extractor and extractor.is_available():
                            # Extract sheet number
                            number_result = extractor.extract_sheet_number(title_block_crop)
                            page_result['sheet_number'] = number_result.get('sheet_number')
                            page_result['number_confidence'] = number_result.get('confidence', 0)

                            # Extract sheet title
                            title_result = extractor.extract_title(title_block_crop)
                            page_result['sheet_title'] = title_result.get('title')
                            page_result['title_confidence'] = title_result.get('confidence', 0)

                            if page_result['sheet_number']:
                                total_extractions += 1

                        pdf_results['pages'].append(page_result)
                        total_pages_processed += 1

                    except Exception as e:
                        print(f"    ERROR on page {page_num}: {e}")
                        pdf_results['pages'].append({
                            'page': page_num,
                            'error': str(e)
                        })

                # Print PDF summary
                extracted_numbers = sum(1 for p in pdf_results['pages'] if p.get('sheet_number'))
                extracted_titles = sum(1 for p in pdf_results['pages'] if p.get('sheet_title'))
                print(f"  Results: {extracted_numbers}/{total_pages} sheet numbers, {extracted_titles}/{total_pages} titles")

                # Save PDF results
                pdf_json_path = pdf_output_dir / "results.json"
                with open(pdf_json_path, 'w') as f:
                    json.dump(pdf_results, f, indent=2)

        except Exception as e:
            print(f"  PDF ERROR: {e}")
            import traceback
            traceback.print_exc()
            pdf_results['error'] = str(e)

        all_results.append(pdf_results)
        if extractor:
            extractor.reset_pdf_counter()  # Reset rate limit counter

    # Save master results
    master_results_path = output_dir / "all_results.json"
    with open(master_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print final summary
    print("\n" + "="*80)
    print("COMPLETE EXTRACTION SUMMARY")
    print("="*80)

    print(f"\n{'PDF':<45} {'Pages':>7} {'Numbers':>10} {'Titles':>10}")
    print("-"*80)

    total_pages_all = 0
    total_numbers = 0
    total_titles = 0

    for r in all_results:
        if 'error' in r and 'pages' not in r:
            print(f"{r['pdf'][:44]:<45} ERROR")
            continue

        pages = len(r.get('pages', []))
        numbers = sum(1 for p in r.get('pages', []) if p.get('sheet_number'))
        titles = sum(1 for p in r.get('pages', []) if p.get('sheet_title'))

        total_pages_all += pages
        total_numbers += numbers
        total_titles += titles

        print(f"{r['pdf'][:44]:<45} {pages:>7} {numbers:>10} {titles:>10}")

    print("-"*80)
    print(f"{'TOTAL':<45} {total_pages_all:>7} {total_numbers:>10} {total_titles:>10}")

    if total_pages_all > 0:
        print(f"\nSheet Number Extraction: {total_numbers}/{total_pages_all} ({total_numbers/total_pages_all*100:.1f}%)")
        print(f"Sheet Title Extraction:  {total_titles}/{total_pages_all} ({total_titles/total_pages_all*100:.1f}%)")

    print(f"\nResults saved to: {output_dir}")
    print(f"Total crops saved: {total_pages_processed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete extraction test")
    parser.add_argument('--crop-only', action='store_true',
                        help='Only save crops, skip extraction (fast)')
    parser.add_argument('--max-pages', type=int, default=None,
                        help='Maximum pages to process per PDF')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Input directory containing PDFs')
    args = parser.parse_args()

    run_complete_extraction(crop_only=args.crop_only, max_pages=args.max_pages, input_dir=args.input_dir)
