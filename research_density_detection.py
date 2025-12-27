"""
Title Block Detection via Edge DENSITY Analysis

Key insight from user:
- Drawing area content CHANGES completely between pages
- Title block STRUCTURE is IDENTICAL (border lines, dividers, boxes)
- The title block has MANY MORE consistent edges than the drawing area

Approach:
1. Get common edges (edges present on ALL pages)
2. Calculate edge DENSITY per column (how many edge pixels)
3. The title block region will have MUCH HIGHER density than drawing area
4. Find the transition point = left edge of title block
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler


def find_title_block_by_density(pdf_path: Path, sample_pages: list = None, dpi: int = 100):
    """
    Find title block by analyzing edge density across columns.

    The title block has a dense, consistent STRUCTURE of lines,
    while the drawing area has sparse, changing content.
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

        # Step 2: Find common edges (structure that appears on ALL pages)
        stack = np.stack(edge_images, axis=0)
        common_edges = np.all(stack > 128, axis=0).astype(np.uint8) * 255

        height, width = common_edges.shape

        # Save common edges
        common_img = Image.fromarray(common_edges)
        common_path = output_dir / f"{pdf_name}_common_edges_density.png"
        common_img.save(common_path)

        # Step 3: Calculate edge DENSITY per column
        # This is simply the sum of edge pixels in each column
        col_density = np.sum(common_edges > 128, axis=0)

        # Smooth the density to remove noise
        from scipy.ndimage import uniform_filter1d
        col_density_smooth = uniform_filter1d(col_density.astype(float), size=20)

        # Step 4: Find the transition point
        # Look in right 50% of page
        search_start = int(width * 0.5)
        right_density = col_density_smooth[search_start:]

        # Calculate mean density of rightmost 10% (definite title block)
        rightmost_10pct = right_density[-int(len(right_density) * 0.1):]
        tb_mean_density = np.mean(rightmost_10pct)

        # Calculate mean density of drawing area (left of center)
        drawing_density = col_density_smooth[:search_start]
        drawing_mean_density = np.mean(drawing_density)

        print(f"\n  Density Analysis:")
        print(f"    Drawing area mean density: {drawing_mean_density:.1f} px/col")
        print(f"    Title block mean density: {tb_mean_density:.1f} px/col")
        print(f"    Ratio: {tb_mean_density/max(drawing_mean_density, 1):.1f}x")

        # Find the transition point where density exceeds threshold
        # Threshold = midpoint between drawing and title block density
        threshold = (drawing_mean_density + tb_mean_density) / 2

        # Or use a percentage of title block density
        threshold = tb_mean_density * 0.3  # 30% of title block density

        # Find leftmost column exceeding threshold
        x1 = None
        for i, density in enumerate(right_density):
            if density > threshold:
                x1 = search_start + i
                break

        if x1 is None:
            # Fallback: use the column where density starts increasing
            gradient = np.gradient(col_density_smooth)
            gradient_right = gradient[search_start:]

            # Find significant positive gradient (entering title block)
            threshold_gradient = np.std(gradient_right) * 1.5
            for i, grad in enumerate(gradient_right):
                if grad > threshold_gradient:
                    x1 = search_start + i
                    break

        if x1 is not None:
            # Refine x1 by looking for actual edge near this point
            # Look for the nearest strong vertical edge
            local_region = common_edges[:, max(0, x1-50):min(width, x1+50)]
            local_col_density = np.sum(local_region > 128, axis=0)

            # Find the peak in local region
            local_peak = np.argmax(local_col_density)
            x1_refined = max(0, x1 - 50) + local_peak

            x2 = width - 1
            y1, y2 = 0, height - 1

            # Calculate percentages
            x1_pct = x1_refined / width
            width_pct = (x2 - x1_refined) / width

            print(f"\n  DETECTED TITLE BLOCK:")
            print(f"    Initial x1: {x1} (from density analysis)")
            print(f"    Refined x1: {x1_refined} (local edge peak)")
            print(f"    Density threshold used: {threshold:.1f}")
            print(f"    Percent: x1={x1_pct:.3f}")
            print(f"    Width: {width_pct*100:.1f}% of page")

            # Draw detection
            first_page = handler.get_page_image(sample_pages[0] - 1, dpi=dpi)
            if first_page.size != target_size:
                first_page = first_page.resize(target_size, Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(first_page.convert('RGB'))

            result_img = first_page.convert('RGB')
            draw = ImageDraw.Draw(result_img)

            # Red box for detected region
            draw.rectangle([x1_refined, y1, x2, y2], outline='red', width=4)

            # Green line at detected boundary
            draw.line([(x1_refined, 0), (x1_refined, height)], fill='green', width=3)

            detection_path = output_dir / f"{pdf_name}_detected_density.png"
            result_img.save(detection_path)
            print(f"  Saved: {detection_path}")

            # Save density plot
            save_density_plot(col_density_smooth, search_start, x1_refined, threshold,
                            output_dir / f"{pdf_name}_density_plot.png")

            return {
                'bbox_percent': {
                    'x1': x1_pct,
                    'y1': 0.0,
                    'x2': 1.0,
                    'y2': 1.0
                },
                'width_percent': width_pct,
            }
        else:
            print("  WARNING: Could not detect title block boundary")
            save_density_plot(col_density_smooth, search_start, None, threshold,
                            output_dir / f"{pdf_name}_density_plot.png")
            return None


def save_density_plot(col_density, search_start, detected_x, threshold, output_path):
    """Save density plot for debugging."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 5))

        # Full page density
        plt.subplot(1, 2, 1)
        plt.plot(col_density, label='Edge density')
        plt.axvline(x=search_start, color='orange', linestyle='--', label='Search start')
        if detected_x:
            plt.axvline(x=detected_x, color='green', linestyle='-', linewidth=2, label=f'Detected x1={detected_x}')
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold={threshold:.0f}')
        plt.xlabel('Column')
        plt.ylabel('Edge pixel count')
        plt.title('Edge Density per Column')
        plt.legend()

        # Zoomed right portion
        plt.subplot(1, 2, 2)
        plt.plot(range(search_start, len(col_density)), col_density[search_start:])
        if detected_x:
            plt.axvline(x=detected_x, color='green', linestyle='-', linewidth=2, label=f'Detected x1={detected_x}')
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold')
        plt.xlabel('Column')
        plt.ylabel('Edge pixel count')
        plt.title('Edge Density - Right 50%')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"  Could not save plot: {e}")


def main():
    """Test density-based detection."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
        "Janesville Nissan Full set Issued for Bids.pdf",
    ]

    print("="*70)
    print("DENSITY-BASED: Title Block Detection via Edge Density Analysis")
    print("="*70)

    results = []

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if pdf_path.exists():
            try:
                result = find_title_block_by_density(pdf_path)
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
    print("SUMMARY - DENSITY-BASED DETECTION")
    print("="*70)
    print(f"{'PDF':<45} {'x1':>8} {'Width':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['pdf']:<45} {r['x1']:>7.3f} {r['width']*100:>7.1f}%")

    print("\nOutput saved to: C:\\tb\\blueprint_processor\\output\\research")


if __name__ == "__main__":
    main()
