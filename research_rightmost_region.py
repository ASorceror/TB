"""
Title Block Detection - Focus on Rightmost Region ONLY

Key insight:
- Title blocks are ALWAYS on the RIGHT EDGE (rightmost 8-20% of page)
- Drawing tables/schedules can be anywhere on the page
- By ONLY analyzing the rightmost region, we avoid drawing coincidences

Approach:
1. Extract edges from rightmost 25% of each page
2. Apply majority vote within that region
3. Find leftmost consistent structure = title block left border
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler


def find_title_block_rightmost(pdf_path: Path, sample_pages: list = None, dpi: int = 100,
                                right_portion: float = 0.25, min_agreement: float = 0.6):
    """
    Find title block by analyzing only the rightmost portion of each page.

    Args:
        right_portion: Fraction of page width to analyze (0.25 = rightmost 25%)
        min_agreement: Fraction of pages that must agree for majority vote
    """
    output_dir = Path(r"C:\tb\blueprint_processor\output\research")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_path.stem[:30].replace(' ', '_')

    with PDFHandler(pdf_path) as handler:
        total_pages = handler.page_count

        if sample_pages is None:
            start = 2 if total_pages > 5 else 1
            if total_pages > 20:
                sample_pages = [2, 5, 8, 12, 15]
            else:
                sample_pages = list(range(start, min(start + 5, total_pages + 1)))

        print(f"\n{'='*60}")
        print(f"PDF: {pdf_path.name}")
        print(f"Total pages: {total_pages}")
        print(f"Sample pages: {sample_pages}")
        print(f"Analyzing rightmost: {right_portion*100:.0f}%")

        # Step 1: Render and extract edges from RIGHTMOST region only
        edge_images = []
        target_size = None
        region_start = None

        for page_num in sample_pages:
            img = handler.get_page_image(page_num - 1, dpi=dpi)

            if target_size is None:
                target_size = img.size
                full_width = target_size[0]
                region_start = int(full_width * (1 - right_portion))
            elif img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)

            gray = img.convert('L')
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)

            # ONLY keep rightmost region
            edges_right = edges_array[:, region_start:]
            binary = (edges_right > 25).astype(np.uint8)
            edge_images.append(binary)

        num_pages = len(edge_images)
        print(f"  Processed {num_pages} pages")
        print(f"  Region: columns {region_start} to {full_width} (rightmost {right_portion*100:.0f}%)")

        # Step 2: Majority vote within the rightmost region
        stack = np.stack(edge_images, axis=0)
        agreement = np.sum(stack, axis=0)

        height, region_width = agreement.shape

        min_pages = int(np.ceil(num_pages * min_agreement))
        majority_edges = (agreement >= min_pages).astype(np.uint8) * 255

        print(f"  Majority threshold: {min_pages} out of {num_pages} pages")
        print(f"  Pixels with majority agreement: {np.sum(majority_edges > 0)}")

        # Step 3: Calculate column edge counts within this region
        col_edge_count = np.sum(majority_edges > 128, axis=0)

        # Smooth
        from scipy.ndimage import uniform_filter1d
        col_smoothed = uniform_filter1d(col_edge_count.astype(float), size=5)

        # Step 4: Find the LEFTMOST column with significant consistent structure
        # Use a simple threshold: > 5% of page height
        threshold = height * 0.03  # 3% of height

        # Find first column exceeding threshold
        x1_local = None
        for i in range(len(col_smoothed)):
            if col_smoothed[i] > threshold:
                x1_local = i
                break

        if x1_local is not None:
            # Convert local coordinate to full page coordinate
            x1 = region_start + x1_local

            # Find x2 (rightmost consistent structure)
            x2_local = len(col_smoothed) - 1
            for i in range(len(col_smoothed) - 1, x1_local, -1):
                if col_smoothed[i] > threshold:
                    x2_local = i
                    break
            x2 = region_start + x2_local

            x1_pct = x1 / full_width
            x2_pct = x2 / full_width
            width_pct = (x2 - x1) / full_width

            print(f"\n  DETECTED TITLE BLOCK:")
            print(f"    Local x1: {x1_local} (in rightmost region)")
            print(f"    Global x1: {x1} pixels = {x1_pct:.3f}")
            print(f"    Global x2: {x2} pixels = {x2_pct:.3f}")
            print(f"    Width: {width_pct*100:.1f}% of page")

            # Save visualization
            first_page = handler.get_page_image(sample_pages[0] - 1, dpi=dpi)
            if first_page.size != target_size:
                first_page = first_page.resize(target_size, Image.Resampling.LANCZOS)

            result_img = first_page.convert('RGB')
            draw = ImageDraw.Draw(result_img)

            # Blue dashed line showing search region start
            for y in range(0, height, 20):
                draw.line([(region_start, y), (region_start, min(y+10, height))], fill='blue', width=1)

            # Green line at detected left boundary
            draw.line([(x1, 0), (x1, height)], fill='green', width=3)

            # Red box for detected title block
            draw.rectangle([x1, 0, x2, height], outline='red', width=2)

            detection_path = output_dir / f"{pdf_name}_rightmost_detected.png"
            result_img.save(detection_path)
            print(f"  Saved: {detection_path}")

            # Save majority edges from region
            majority_img = Image.fromarray(majority_edges)
            majority_path = output_dir / f"{pdf_name}_rightmost_majority.png"
            majority_img.save(majority_path)
            print(f"  Saved: {majority_path}")

            return {
                'bbox_percent': {
                    'x1': x1_pct,
                    'y1': 0.0,
                    'x2': x2_pct,
                    'y2': 1.0
                },
                'width_percent': width_pct,
            }
        else:
            print("  WARNING: Could not detect title block")
            return None


def main():
    """Test rightmost region detection."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
        "Janesville Nissan Full set Issued for Bids.pdf",
    ]

    print("="*70)
    print("RIGHTMOST REGION: Title Block Detection (rightmost 25%)")
    print("="*70)

    results = []

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if pdf_path.exists():
            try:
                result = find_title_block_rightmost(pdf_path, right_portion=0.25)
                if result:
                    results.append({
                        'pdf': pdf_name[:40],
                        'x1': result['bbox_percent']['x1'],
                        'width': result['width_percent']
                    })
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - RIGHTMOST REGION (25%)")
    print("="*70)
    print(f"{'PDF':<45} {'x1':>8} {'Width':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['pdf']:<45} {r['x1']:>7.3f} {r['width']*100:>7.1f}%")

    print("\nOutput saved to: C:\\tb\\blueprint_processor\\output\\research")


if __name__ == "__main__":
    main()
