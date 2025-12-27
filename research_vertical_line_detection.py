"""
Title Block Detection via Vertical Line Detection

Key insight: The LEFT BORDER of the title block is a VERTICAL LINE
that spans most of the page height.

Approach:
1. Get common edges (structure present on ALL pages)
2. For each column in the right portion, calculate TOTAL edge pixels
   (not requiring continuity - lines can be broken by horizontal dividers)
3. Find columns with high total edge count (indicating a vertical line)
4. The LEFTMOST such column = title block left border
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler


def find_title_block_vertical_lines(pdf_path: Path, sample_pages: list = None, dpi: int = 100):
    """
    Find title block by detecting vertical lines (left border).
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
            binary = (edges_array > 25).astype(np.uint8) * 255
            edge_images.append(binary)

        print(f"  Processed {len(edge_images)} pages")

        # Step 2: Find common edges
        stack = np.stack(edge_images, axis=0)
        common_edges = np.all(stack > 128, axis=0).astype(np.uint8) * 255

        height, width = common_edges.shape

        # Step 3: Calculate edge count per column (total edge pixels)
        col_edge_count = np.sum(common_edges > 128, axis=0)

        # Step 4: Search in right portion (60% to 97% of page width)
        # Start from 60% to avoid page border artifacts
        search_start = int(width * 0.60)
        search_end = int(width * 0.97)

        right_counts = col_edge_count[search_start:search_end]

        # Step 5: Find columns with significant vertical line presence
        # A vertical line should have edge pixels spanning a good portion of height
        # But they don't need to be continuous (can be broken by horizontal lines)

        # Calculate statistics for the right portion
        mean_count = np.mean(right_counts)
        std_count = np.std(right_counts)
        max_count = np.max(right_counts)

        print(f"\n  Column edge statistics (right 40% of page):")
        print(f"    Mean: {mean_count:.1f}")
        print(f"    Std: {std_count:.1f}")
        print(f"    Max: {max_count:.1f}")

        # Find peaks - columns with significantly above-average edge counts
        # These are likely vertical lines
        threshold = mean_count + std_count  # At least 1 std above mean

        # Find all columns above threshold
        above_threshold = right_counts > threshold
        peak_cols = np.where(above_threshold)[0]

        if len(peak_cols) > 0:
            # Find the LEFTMOST peak (first vertical line from left in search area)
            first_peak = peak_cols[0]
            x1 = search_start + first_peak

            # Find all significant peaks to determine x2
            last_peak = peak_cols[-1]
            x2 = search_start + last_peak

            # Calculate percentages
            x1_pct = x1 / width
            x2_pct = x2 / width
            width_pct = (x2 - x1) / width

            print(f"\n  Found {len(peak_cols)} columns above threshold")
            print(f"  First peak at column {first_peak} (absolute: {x1})")
            print(f"  Edge count at first peak: {right_counts[first_peak]:.0f}")

            print(f"\n  DETECTED TITLE BLOCK:")
            print(f"    Left edge (x1): {x1} pixels = {x1_pct:.3f}")
            print(f"    Right edge (x2): {x2} pixels = {x2_pct:.3f}")
            print(f"    Width: {width_pct*100:.1f}% of page")

            # Save visualization
            first_page = handler.get_page_image(sample_pages[0] - 1, dpi=dpi)
            if first_page.size != target_size:
                first_page = first_page.resize(target_size, Image.Resampling.LANCZOS)

            result_img = first_page.convert('RGB')
            draw = ImageDraw.Draw(result_img)

            # Green line at left boundary
            draw.line([(x1, 0), (x1, height)], fill='green', width=3)

            # Red box for full title block
            draw.rectangle([x1, 0, x2, height], outline='red', width=2)

            detection_path = output_dir / f"{pdf_name}_vertical_line_detected.png"
            result_img.save(detection_path)
            print(f"  Saved: {detection_path}")

            # Save column profile plot
            save_column_profile(col_edge_count, search_start, search_end, x1, threshold,
                              output_dir / f"{pdf_name}_column_profile.png")

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
            print("  WARNING: No significant vertical lines found")
            save_column_profile(col_edge_count, search_start, search_end, None, threshold,
                              output_dir / f"{pdf_name}_column_profile.png")
            return None


def save_column_profile(col_counts, search_start, search_end, detected_x, threshold, output_path):
    """Save column edge profile plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Full page profile
        axes[0].plot(col_counts)
        axes[0].axvline(x=search_start, color='orange', linestyle='--', label='Search start (60%)')
        axes[0].axvline(x=search_end, color='orange', linestyle='--', label='Search end (97%)')
        if detected_x:
            axes[0].axvline(x=detected_x, color='green', linestyle='-', linewidth=2,
                          label=f'Detected x1={detected_x}')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Total edge pixels')
        axes[0].set_title('Column Edge Profile (full page)')
        axes[0].legend()

        # Zoomed search region
        search_counts = col_counts[search_start:search_end]
        axes[1].plot(range(search_start, search_end), search_counts)
        axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold={threshold:.0f}')
        if detected_x:
            axes[1].axvline(x=detected_x, color='green', linestyle='-', linewidth=2,
                          label=f'Detected x1={detected_x}')
        axes[1].set_xlabel('Column')
        axes[1].set_ylabel('Total edge pixels')
        axes[1].set_title('Column Edge Profile (search region)')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"  Could not save plot: {e}")


def main():
    """Test vertical line detection."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
        "Janesville Nissan Full set Issued for Bids.pdf",
    ]

    print("="*70)
    print("VERTICAL LINE: Title Block Detection via Vertical Line Detection")
    print("="*70)

    results = []

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if pdf_path.exists():
            try:
                result = find_title_block_vertical_lines(pdf_path)
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
    print("SUMMARY - VERTICAL LINE DETECTION")
    print("="*70)
    print(f"{'PDF':<45} {'x1':>8} {'Width':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['pdf']:<45} {r['x1']:>7.3f} {r['width']*100:>7.1f}%")

    print("\nOutput saved to: C:\\tb\\blueprint_processor\\output\\research")


if __name__ == "__main__":
    main()
