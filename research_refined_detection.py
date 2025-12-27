"""
Refined Title Block Detection

Problem: The page BORDER is consistent across all pages, creating noise.
Solution: Exclude the outermost edges and focus on INTERNAL structure.

Approach:
1. Get common edges (structure present on ALL pages)
2. EXCLUDE the outer page border (first/last 3% of width/height)
3. In the right portion, find the LEFTMOST vertical structure
4. That's the title block boundary
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler


def find_title_block_refined(pdf_path: Path, sample_pages: list = None, dpi: int = 100):
    """
    Find title block by analyzing internal consistent structure.
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

        print(f"  Processed {len(edge_images)} pages at {target_size}")

        # Step 2: Find common edges
        stack = np.stack(edge_images, axis=0)
        common_edges = np.all(stack > 128, axis=0).astype(np.uint8) * 255

        height, width = common_edges.shape

        # Step 3: EXCLUDE outer page border
        # Mask out the outermost 3% on all sides
        margin_x = int(width * 0.03)
        margin_y = int(height * 0.03)

        internal_edges = common_edges.copy()
        internal_edges[:margin_y, :] = 0  # Top
        internal_edges[-margin_y:, :] = 0  # Bottom
        internal_edges[:, :margin_x] = 0  # Left
        internal_edges[:, -margin_x:] = 0  # Right

        print(f"  Excluded outer {margin_x}px (x) and {margin_y}px (y) borders")

        # Save internal edges
        internal_img = Image.fromarray(internal_edges)
        internal_path = output_dir / f"{pdf_name}_internal_edges.png"
        internal_img.save(internal_path)
        print(f"  Saved: {internal_path}")

        # Step 4: Focus on right portion (50% to 97%)
        search_start = int(width * 0.50)
        search_end = int(width * 0.97)

        right_portion = internal_edges[:, search_start:search_end]

        # Step 5: Find column-wise edge presence
        # For each column, count edge pixels
        col_edge_count = np.sum(right_portion > 128, axis=0)

        # Find columns with significant edge presence (>5% of height)
        min_presence = height * 0.05
        significant_cols = col_edge_count > min_presence

        if not np.any(significant_cols):
            print("  WARNING: No significant edges found in right portion")
            # Try lower threshold
            min_presence = height * 0.02
            significant_cols = col_edge_count > min_presence

        if np.any(significant_cols):
            # Find the LEFTMOST column with significant edges
            first_sig_col = np.argmax(significant_cols)
            x1 = search_start + first_sig_col

            # Find the RIGHTMOST column with significant edges
            last_sig_col = len(significant_cols) - 1 - np.argmax(significant_cols[::-1])
            x2 = search_start + last_sig_col

            # Find vertical extent
            title_region = internal_edges[:, x1:x2+1]
            row_has_edge = np.any(title_region > 128, axis=1)
            rows = np.where(row_has_edge)[0]

            if len(rows) > 0:
                y1 = rows[0]
                y2 = rows[-1]
            else:
                y1, y2 = margin_y, height - margin_y

            # Calculate percentages
            x1_pct = x1 / width
            y1_pct = y1 / height
            x2_pct = x2 / width
            y2_pct = y2 / height
            width_pct = (x2 - x1) / width

            print(f"\n  DETECTED TITLE BLOCK:")
            print(f"    Leftmost significant column: {first_sig_col} (from search_start)")
            print(f"    Edge count at that column: {col_edge_count[first_sig_col]}")
            print(f"    Pixels: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"    Percent: x1={x1_pct:.3f}, y1={y1_pct:.3f}, x2={x2_pct:.3f}, y2={y2_pct:.3f}")
            print(f"    Width: {width_pct*100:.1f}% of page")

            # Save visualization
            first_page = handler.get_page_image(sample_pages[0] - 1, dpi=dpi)
            if first_page.size != target_size:
                first_page = first_page.resize(target_size, Image.Resampling.LANCZOS)

            result_img = first_page.convert('RGB')
            draw = ImageDraw.Draw(result_img)

            # Red box for detection
            draw.rectangle([x1, y1, x2, y2], outline='red', width=4)

            # Green line at left boundary
            draw.line([(x1, 0), (x1, height)], fill='green', width=3)

            detection_path = output_dir / f"{pdf_name}_detected_refined.png"
            result_img.save(detection_path)
            print(f"  Saved: {detection_path}")

            # Save edge presence plot
            save_presence_plot(col_edge_count, first_sig_col, min_presence,
                             output_dir / f"{pdf_name}_edge_presence.png")

            return {
                'bbox_percent': {
                    'x1': x1_pct,
                    'y1': y1_pct,
                    'x2': x2_pct,
                    'y2': y2_pct
                },
                'width_percent': width_pct,
            }
        else:
            print("  WARNING: Could not detect title block")
            return None


def save_presence_plot(col_presence, detected_col, threshold, output_path):
    """Save edge presence plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        plt.plot(col_presence)
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold={threshold:.0f}')
        plt.axvline(x=detected_col, color='green', linestyle='-', linewidth=2,
                   label=f'Detected col={detected_col}')
        plt.xlabel('Column (offset from 50% of page)')
        plt.ylabel('Edge pixel count')
        plt.title('Edge Presence per Column (right 50% of page, excluding borders)')
        plt.legend()
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"  Could not save plot: {e}")


def main():
    """Test refined detection."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
        "Janesville Nissan Full set Issued for Bids.pdf",
    ]

    print("="*70)
    print("REFINED: Title Block Detection (excluding page border)")
    print("="*70)

    results = []

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if pdf_path.exists():
            try:
                result = find_title_block_refined(pdf_path)
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
    print("SUMMARY - REFINED DETECTION")
    print("="*70)
    print(f"{'PDF':<45} {'x1':>8} {'Width':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['pdf']:<45} {r['x1']:>7.3f} {r['width']*100:>7.1f}%")

    print("\nOutput saved to: C:\\tb\\blueprint_processor\\output\\research")


if __name__ == "__main__":
    main()
