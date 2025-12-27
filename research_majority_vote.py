"""
Title Block Detection via Majority Vote Edge Detection

Problem: "Edges on ALL pages" is too strict - slight variations cause edges to disappear.
Solution: Use MAJORITY VOTE - edges on MOST pages (e.g., 3 out of 5).

This is more robust to:
- Slight PDF rendering differences
- Anti-aliasing variations
- Compression artifacts
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler


def find_title_block_majority_vote(pdf_path: Path, sample_pages: list = None, dpi: int = 100,
                                    min_agreement: float = 0.6):
    """
    Find title block using majority vote edge detection.

    Args:
        min_agreement: Fraction of pages that must agree (0.6 = 60% = 3 out of 5)
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
        print(f"Min agreement: {min_agreement*100:.0f}%")

        # Step 1: Render and extract edges
        edge_images = []
        target_size = None

        for page_num in sample_pages:
            img = handler.get_page_image(page_num - 1, dpi=dpi)

            if target_size is None:
                target_size = img.size
            elif img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)

            gray = img.convert('L')
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)
            binary = (edges_array > 25).astype(np.uint8)  # 0 or 1
            edge_images.append(binary)

        num_pages = len(edge_images)
        print(f"  Processed {num_pages} pages")

        # Step 2: Calculate agreement at each pixel
        # Sum across pages - value 0-N indicating how many pages have edge at that pixel
        stack = np.stack(edge_images, axis=0)
        agreement = np.sum(stack, axis=0)  # 0 to num_pages

        height, width = agreement.shape

        # Step 3: Apply majority vote threshold
        min_pages = int(np.ceil(num_pages * min_agreement))
        majority_edges = (agreement >= min_pages).astype(np.uint8) * 255

        print(f"  Majority threshold: {min_pages} out of {num_pages} pages")
        print(f"  Pixels with majority agreement: {np.sum(majority_edges > 0)}")

        # Save majority edges
        majority_img = Image.fromarray(majority_edges)
        majority_path = output_dir / f"{pdf_name}_majority_edges.png"
        majority_img.save(majority_path)
        print(f"  Saved: {majority_path}")

        # Compare with strict AND (all pages)
        all_edges = (agreement == num_pages).astype(np.uint8) * 255
        print(f"  Pixels with ALL agreement: {np.sum(all_edges > 0)}")

        # Step 4: Calculate column edge counts using majority edges
        col_edge_count = np.sum(majority_edges > 128, axis=0)

        # Smooth to reduce noise
        from scipy.ndimage import uniform_filter1d
        col_smoothed = uniform_filter1d(col_edge_count.astype(float), size=10)

        # Step 5: Find transition point (same as before)
        left_portion = col_smoothed[:int(width * 0.5)]
        baseline = np.median(left_portion)
        noise_level = np.percentile(left_portion, 95)
        threshold = max(noise_level * 2, 20)

        print(f"\n  Baseline: {baseline:.1f}, Noise: {noise_level:.1f}, Threshold: {threshold:.1f}")

        search_start = int(width * 0.60)
        search_end = int(width * 0.97)
        window_size = 15

        transition_col = None
        for i in range(search_start, search_end - window_size):
            window = col_smoothed[i:i + window_size]
            if np.all(window > threshold):
                transition_col = i
                break

        if transition_col is not None:
            x1 = transition_col

            # Find x2
            for i in range(search_end - 1, x1, -1):
                if col_smoothed[i] > threshold:
                    x2 = i
                    break
            else:
                x2 = search_end - 1

            x1_pct = x1 / width
            x2_pct = x2 / width
            width_pct = (x2 - x1) / width

            print(f"\n  DETECTED TITLE BLOCK:")
            print(f"    Transition at column: {transition_col}")
            print(f"    Left edge (x1): {x1_pct:.3f}")
            print(f"    Right edge (x2): {x2_pct:.3f}")
            print(f"    Width: {width_pct*100:.1f}% of page")

            # Save visualization
            first_page = handler.get_page_image(sample_pages[0] - 1, dpi=dpi)
            if first_page.size != target_size:
                first_page = first_page.resize(target_size, Image.Resampling.LANCZOS)

            result_img = first_page.convert('RGB')
            draw = ImageDraw.Draw(result_img)

            draw.line([(x1, 0), (x1, height)], fill='green', width=3)
            draw.rectangle([x1, 0, x2, height], outline='red', width=2)

            detection_path = output_dir / f"{pdf_name}_majority_detected.png"
            result_img.save(detection_path)
            print(f"  Saved: {detection_path}")

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
            print("  WARNING: Could not detect transition")
            return None


def main():
    """Test majority vote detection."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
        "Janesville Nissan Full set Issued for Bids.pdf",
    ]

    print("="*70)
    print("MAJORITY VOTE: Title Block Detection (60% agreement)")
    print("="*70)

    results = []

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if pdf_path.exists():
            try:
                result = find_title_block_majority_vote(pdf_path, min_agreement=0.6)
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
    print("SUMMARY - MAJORITY VOTE (60%)")
    print("="*70)
    print(f"{'PDF':<45} {'x1':>8} {'Width':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['pdf']:<45} {r['x1']:>7.3f} {r['width']*100:>7.1f}%")

    print("\nOutput saved to: C:\\tb\\blueprint_processor\\output\\research")


if __name__ == "__main__":
    main()
