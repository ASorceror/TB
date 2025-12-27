"""
Improved Title Block Detection via Common Edge Analysis

Key insight from the common_edges image:
- The title block STRUCTURE is clearly visible
- We need to find the LEFTMOST vertical line that defines the title block boundary
- Not just any strong vertical line

Approach:
1. Find common edges across pages (edges present on ALL pages)
2. Look for vertical continuity on the right side of page
3. Find the LEFTMOST column that has strong vertical edge continuity
4. That's the left boundary (x1) of the title block
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler


def find_title_block_improved(pdf_path: Path, sample_pages: list = None, dpi: int = 100):
    """
    Improved title block detection using common edge analysis.
    """
    output_dir = Path(r"C:\tb\blueprint_processor\output\research")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_path.stem[:30].replace(' ', '_')

    with PDFHandler(pdf_path) as handler:
        total_pages = handler.page_count

        # Sample diverse pages
        if sample_pages is None:
            start = 2 if total_pages > 5 else 1
            if total_pages > 20:
                # Sample diverse pages from different sections
                sample_pages = [2, 5, 8, 12, 15]
            else:
                sample_pages = list(range(start, min(start + 5, total_pages + 1)))

        print(f"\n{'='*60}")
        print(f"PDF: {pdf_path.name}")
        print(f"Total pages: {total_pages}")
        print(f"Sample pages: {sample_pages}")

        # Step 1: Render pages and extract edges
        edge_images = []
        target_size = None

        for page_num in sample_pages:
            img = handler.get_page_image(page_num - 1, dpi=dpi)

            # Normalize size (resize to first page's size)
            if target_size is None:
                target_size = img.size
            elif img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)

            gray = img.convert('L')
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)

            # Binary threshold
            binary = (edges_array > 25).astype(np.uint8) * 255
            edge_images.append(binary)

        print(f"  Processed {len(edge_images)} pages at {target_size}")

        # Step 2: Find common edges (edges on ALL pages)
        stack = np.stack(edge_images, axis=0)
        common_edges = np.all(stack > 128, axis=0).astype(np.uint8) * 255

        height, width = common_edges.shape

        # Save common edges
        common_img = Image.fromarray(common_edges)
        common_path = output_dir / f"{pdf_name}_common_edges_v2.png"
        common_img.save(common_path)
        print(f"  Saved: {common_path}")

        # Step 3: Find vertical line continuity for each column
        # A "strong vertical line" is a column where many consecutive pixels are edges

        # Calculate how much of each column is part of a continuous vertical edge
        col_continuity = np.zeros(width)

        for x in range(width):
            col = common_edges[:, x]
            # Count longest continuous run of edge pixels
            max_run = 0
            current_run = 0
            for y in range(height):
                if col[y] > 128:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            col_continuity[x] = max_run

        # Step 4: Find columns with strong vertical continuity in RIGHT portion
        # Title block is on right side, look from 50% to 100%
        search_start = int(width * 0.5)

        # Find columns where continuity is at least 50% of page height
        min_continuity = height * 0.5

        # Look for the LEFTMOST column with strong continuity
        leftmost_border = None
        for x in range(search_start, width):
            if col_continuity[x] >= min_continuity:
                leftmost_border = x
                break

        if leftmost_border is None:
            # Lower threshold and try again
            min_continuity = height * 0.3
            for x in range(search_start, width):
                if col_continuity[x] >= min_continuity:
                    leftmost_border = x
                    break

        if leftmost_border is not None:
            # Found the left border of title block
            x1 = leftmost_border

            # Find the right border (rightmost strong line or page edge)
            x2 = width - 1
            for x in range(width - 1, x1, -1):
                if col_continuity[x] >= min_continuity:
                    x2 = x
                    break

            # Find vertical extent (y1, y2) by analyzing the title block region
            tb_region = common_edges[:, x1:x2]
            row_has_edge = np.any(tb_region > 128, axis=1)
            rows = np.where(row_has_edge)[0]

            if len(rows) > 0:
                y1 = rows[0]
                y2 = rows[-1]
            else:
                y1, y2 = 0, height - 1

            # Calculate percentages
            x1_pct = x1 / width
            y1_pct = y1 / height
            x2_pct = x2 / width
            y2_pct = y2 / height
            width_pct = (x2 - x1) / width

            print(f"\n  DETECTED TITLE BLOCK:")
            print(f"    Left border at column: {x1} (searched from {search_start})")
            print(f"    Continuity at left border: {col_continuity[x1]:.0f}px ({col_continuity[x1]/height*100:.1f}% of height)")
            print(f"    Pixels: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"    Percent: x1={x1_pct:.3f}, y1={y1_pct:.3f}, x2={x2_pct:.3f}, y2={y2_pct:.3f}")
            print(f"    Width: {width_pct*100:.1f}% of page")

            # Draw detection on first page
            first_page = handler.get_page_image(sample_pages[0] - 1, dpi=dpi)
            if first_page.size != target_size:
                first_page = first_page.resize(target_size, Image.Resampling.LANCZOS)
            first_page_rgb = first_page.convert('RGB')
            draw = ImageDraw.Draw(first_page_rgb)

            # Draw bounding box (red)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=4)

            # Draw left border line (green)
            draw.line([(x1, 0), (x1, height)], fill='green', width=2)

            detection_path = output_dir / f"{pdf_name}_detected_v2.png"
            first_page_rgb.save(detection_path)
            print(f"  Saved: {detection_path}")

            # Also save continuity plot
            save_continuity_plot(col_continuity, search_start, leftmost_border, min_continuity,
                               output_dir / f"{pdf_name}_continuity.png")

            return {
                'bbox_pixels': (x1, y1, x2, y2),
                'bbox_percent': {
                    'x1': x1_pct,
                    'y1': y1_pct,
                    'x2': x2_pct,
                    'y2': y2_pct
                },
                'width_percent': width_pct,
            }
        else:
            print("  WARNING: Could not find title block border")
            # Save continuity plot for debugging
            save_continuity_plot(col_continuity, search_start, None, min_continuity,
                               output_dir / f"{pdf_name}_continuity.png")
            return None


def save_continuity_plot(col_continuity, search_start, detected_x, threshold, output_path):
    """Save a plot of column continuity for debugging."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 5))

        # Plot full continuity
        plt.subplot(1, 2, 1)
        plt.plot(col_continuity)
        plt.axvline(x=search_start, color='orange', linestyle='--', label='Search start (50%)')
        if detected_x:
            plt.axvline(x=detected_x, color='green', linestyle='-', label=f'Detected border (x={detected_x})')
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.0f}px)')
        plt.xlabel('Column')
        plt.ylabel('Vertical continuity (px)')
        plt.title('Column Vertical Continuity (full page)')
        plt.legend()

        # Zoom into right portion
        plt.subplot(1, 2, 2)
        right_portion = col_continuity[search_start:]
        plt.plot(range(search_start, len(col_continuity)), right_portion)
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold')
        if detected_x:
            plt.axvline(x=detected_x, color='green', linestyle='-', label=f'Detected x={detected_x}')
        plt.xlabel('Column')
        plt.ylabel('Vertical continuity (px)')
        plt.title('Column Vertical Continuity (right 50%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"  Could not save plot: {e}")


def main():
    """Test improved detection on sample PDFs."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
        "Janesville Nissan Full set Issued for Bids.pdf",
    ]

    print("="*70)
    print("IMPROVED: Title Block Detection via Common Edge Analysis")
    print("="*70)

    results = []

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if pdf_path.exists():
            try:
                result = find_title_block_improved(pdf_path)
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
        else:
            print(f"\nSKIP: {pdf_name} not found")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'PDF':<45} {'x1':>8} {'Width':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['pdf']:<45} {r['x1']:>7.3f} {r['width']*100:>7.1f}%")

    print("\n" + "="*70)
    print("Output saved to: C:\\tb\\blueprint_processor\\output\\research")
    print("="*70)


if __name__ == "__main__":
    main()
