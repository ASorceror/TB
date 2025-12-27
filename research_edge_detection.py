"""
Research: Edge-Based Title Block Detection

Better approach: Find BORDER LINES that are consistent across all pages.

The insight:
- Title block CONTENT changes (sheet numbers, titles)
- Title block BORDER LINES are IDENTICAL on every page
- Find edges → intersect across pages → consistent edges = title block border

Technique:
1. Apply edge detection to each page (find lines)
2. Binary threshold to get clean line image
3. AND operation across all pages (edges present on ALL pages)
4. The common edges = title block border structure
5. Find bounding box of consistent vertical line on right side
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler


def find_vertical_lines(binary_image: np.ndarray, min_length: int = 100) -> list:
    """
    Find strong vertical lines in binary image using column analysis.

    Returns list of (x, y_start, y_end) for each vertical line.
    """
    height, width = binary_image.shape
    lines = []

    for x in range(width):
        col = binary_image[:, x]
        # Find continuous runs of black (0) pixels
        in_line = False
        y_start = 0

        for y in range(height):
            if col[y] < 128:  # Black pixel (line)
                if not in_line:
                    in_line = True
                    y_start = y
            else:
                if in_line:
                    length = y - y_start
                    if length >= min_length:
                        lines.append((x, y_start, y))
                    in_line = False

        # Check end of column
        if in_line:
            length = height - y_start
            if length >= min_length:
                lines.append((x, y_start, height))

    return lines


def find_title_block_by_edges(pdf_path: Path, sample_pages: list = None, dpi: int = 100):
    """
    Find title block by detecting consistent border lines across pages.
    """
    output_dir = Path(r"C:\tb\blueprint_processor\output\research")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_path.stem[:30].replace(' ', '_')

    with PDFHandler(pdf_path) as handler:
        total_pages = handler.page_count

        # Sample non-consecutive pages (skip potential cover sheet)
        if sample_pages is None:
            start = 2 if total_pages > 5 else 1
            # Sample non-consecutive pages for better comparison
            if total_pages > 20:
                sample_pages = [2, 5, 10, 15, 20]
            else:
                sample_pages = list(range(start, min(start + 5, total_pages + 1)))

        print(f"\n{'='*60}")
        print(f"PDF: {pdf_path.name}")
        print(f"Total pages: {total_pages}")
        print(f"Sample pages: {sample_pages}")

        # Process each page to extract edges
        edge_images = []

        for page_num in sample_pages:
            img = handler.get_page_image(page_num - 1, dpi=dpi)
            gray = img.convert('L')

            # Apply edge detection (find contours)
            edges = gray.filter(ImageFilter.FIND_EDGES)

            # Enhance and threshold to get clean lines
            edges_array = np.array(edges)

            # Binary threshold - keep strong edges
            threshold = 30
            binary = (edges_array > threshold).astype(np.uint8) * 255

            edge_images.append(binary)
            print(f"  Page {page_num}: extracted edges")

        # Stack and find edges present on ALL pages (intersection)
        stack = np.stack(edge_images, axis=0)

        # Edge is "common" if it appears on all pages (all values > 0)
        # For binary images: all must be 255
        common_edges = np.all(stack > 128, axis=0).astype(np.uint8) * 255

        # Save common edges
        common_img = Image.fromarray(common_edges)
        common_path = output_dir / f"{pdf_name}_common_edges.png"
        common_img.save(common_path)
        print(f"  Saved common edges: {common_path}")

        # Also try OR operation (edges on ANY page) for comparison
        any_edges = np.any(stack > 128, axis=0).astype(np.uint8) * 255
        any_img = Image.fromarray(any_edges)
        any_path = output_dir / f"{pdf_name}_any_edges.png"
        any_img.save(any_path)

        # Find the rightmost consistent vertical line
        # This should be the left border of the title block
        height, width = common_edges.shape

        # Look in right 50% of page
        right_half_start = width // 2

        # Count edge pixels per column in right half
        col_density = np.sum(common_edges[:, right_half_start:] > 128, axis=0)

        # Find columns with significant vertical edge density
        # (title block border should be a strong vertical line)
        min_density = height * 0.3  # At least 30% of column height should be edge

        strong_cols = np.where(col_density > min_density)[0]

        if len(strong_cols) > 0:
            # Find the leftmost strong column (left edge of title block)
            leftmost_col = strong_cols[0] + right_half_start

            # Find the rightmost strong column (right edge, usually near page edge)
            rightmost_col = strong_cols[-1] + right_half_start

            # Calculate title block boundaries
            x1 = leftmost_col
            x2 = rightmost_col

            # Find vertical extent by looking at row density in title block region
            tb_region = common_edges[:, x1:x2]
            row_density = np.sum(tb_region > 128, axis=1)

            # Find rows with edges
            rows_with_edges = np.where(row_density > 0)[0]
            if len(rows_with_edges) > 0:
                y1 = rows_with_edges[0]
                y2 = rows_with_edges[-1]
            else:
                y1, y2 = 0, height

            print(f"\n  DETECTED TITLE BLOCK:")
            print(f"    Pixels: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"    Percent: x1={x1/width:.3f}, y1={y1/height:.3f}, x2={x2/width:.3f}, y2={y2/height:.3f}")
            print(f"    Width: {(x2-x1)/width*100:.1f}% of page")
            print(f"    Height: {(y2-y1)/height*100:.1f}% of page")

            # Draw detection on first page
            first_page = handler.get_page_image(sample_pages[0] - 1, dpi=dpi)
            first_page_rgb = first_page.convert('RGB')
            draw = ImageDraw.Draw(first_page_rgb)

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

            # Draw left border line
            draw.line([(leftmost_col, 0), (leftmost_col, height)], fill='blue', width=2)

            detection_path = output_dir / f"{pdf_name}_detected.png"
            first_page_rgb.save(detection_path)
            print(f"  Saved detection: {detection_path}")

            return {
                'bbox_pixels': (x1, y1, x2, y2),
                'bbox_percent': {
                    'x1': x1/width,
                    'y1': y1/height,
                    'x2': x2/width,
                    'y2': y2/height
                },
                'width_percent': (x2-x1)/width,
                'left_border_x': leftmost_col,
                'output_dir': str(output_dir)
            }
        else:
            print("  WARNING: Could not find strong vertical lines in right half")

            # Fallback: analyze column density more loosely
            # Show the density profile
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 4))
            plt.plot(col_density)
            plt.xlabel('Column (from middle of page)')
            plt.ylabel('Edge pixel count')
            plt.title('Column edge density in right half of page')
            plt.axhline(y=min_density, color='r', linestyle='--', label=f'Threshold ({min_density:.0f})')
            plt.legend()

            density_path = output_dir / f"{pdf_name}_col_density.png"
            plt.savefig(density_path)
            plt.close()
            print(f"  Saved column density plot: {density_path}")

            return None


def analyze_page_structure(pdf_path: Path, page_num: int = 2, dpi: int = 150):
    """
    Analyze a single page to understand its structure.
    Useful for debugging title block detection.
    """
    output_dir = Path(r"C:\tb\blueprint_processor\output\research")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_path.stem[:30].replace(' ', '_')

    with PDFHandler(pdf_path) as handler:
        img = handler.get_page_image(page_num - 1, dpi=dpi)
        gray = img.convert('L')

        # Save original
        orig_path = output_dir / f"{pdf_name}_p{page_num}_original.png"
        img.save(orig_path)

        # Edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_path = output_dir / f"{pdf_name}_p{page_num}_edges.png"
        edges.save(edges_path)

        # Contour detection
        contour = gray.filter(ImageFilter.CONTOUR)
        contour_path = output_dir / f"{pdf_name}_p{page_num}_contour.png"
        contour.save(contour_path)

        print(f"\nSingle page analysis for {pdf_path.name} page {page_num}:")
        print(f"  Original: {orig_path}")
        print(f"  Edges: {edges_path}")
        print(f"  Contour: {contour_path}")


def main():
    """Test edge-based detection on sample PDFs."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
    ]

    print("="*70)
    print("RESEARCH: Edge-Based Title Block Detection")
    print("="*70)

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if pdf_path.exists():
            try:
                result = find_title_block_by_edges(pdf_path)
                if result:
                    print(f"\n  SUCCESS: Title block at x1={result['bbox_percent']['x1']:.3f}")
                    print(f"           Width: {result['width_percent']*100:.1f}%")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\nSKIP: {pdf_name} not found")

    print("\n" + "="*70)
    print("Research complete! Check output in:")
    print(r"  C:\tb\blueprint_processor\output\research")
    print("="*70)


if __name__ == "__main__":
    main()
