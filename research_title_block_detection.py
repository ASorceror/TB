"""
Research: Multi-Page Comparison for Title Block Detection

Concept: When you overlay multiple blueprint pages:
- Drawing content CHANGES between pages (floor plans, details, etc.)
- Title block structure STAYS THE SAME (like animation flip book)

Technique:
1. Render multiple sample pages at same DPI
2. Convert to grayscale
3. Calculate pixel-wise variance across all pages
4. Low variance regions = consistent structure = title block
5. Find the bounding box of the low-variance region

This is pure computer vision - no AI required!
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import hashlib

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler


def analyze_page_variance(pdf_path: Path, sample_pages: list = None, dpi: int = 100):
    """
    Analyze variance across multiple pages to find consistent regions.

    Args:
        pdf_path: Path to PDF file
        sample_pages: List of 1-indexed page numbers to analyze (default: first 5 non-cover pages)
        dpi: Resolution to render pages (lower = faster)

    Returns:
        Dict with variance analysis results
    """
    output_dir = Path(r"C:\tb\blueprint_processor\output\research")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_path.stem[:30]

    with PDFHandler(pdf_path) as handler:
        total_pages = handler.page_count

        # Default: sample pages 2-6 (skip cover sheet)
        if sample_pages is None:
            start = 2 if total_pages > 5 else 1
            sample_pages = list(range(start, min(start + 5, total_pages + 1)))

        print(f"\nPDF: {pdf_path.name}")
        print(f"Total pages: {total_pages}")
        print(f"Sample pages: {sample_pages}")

        # Render all sample pages
        images = []
        for page_num in sample_pages:
            img = handler.get_page_image(page_num - 1, dpi=dpi)
            # Convert to grayscale
            gray = img.convert('L')
            images.append(np.array(gray, dtype=np.float32))
            print(f"  Rendered page {page_num}: {img.size}")

        # Check all images are same size
        shapes = [img.shape for img in images]
        if len(set(shapes)) > 1:
            print(f"  WARNING: Pages have different sizes: {shapes}")
            # Crop to minimum common size
            min_h = min(s[0] for s in shapes)
            min_w = min(s[1] for s in shapes)
            images = [img[:min_h, :min_w] for img in images]

        # Stack images into 3D array (num_pages, height, width)
        stack = np.stack(images, axis=0)
        print(f"  Image stack shape: {stack.shape}")

        # Calculate pixel-wise variance across pages
        # Low variance = consistent region = title block
        variance = np.var(stack, axis=0)

        # Normalize variance to 0-255 for visualization
        var_min, var_max = variance.min(), variance.max()
        if var_max > var_min:
            variance_norm = ((variance - var_min) / (var_max - var_min) * 255).astype(np.uint8)
        else:
            variance_norm = np.zeros_like(variance, dtype=np.uint8)

        # Save variance heatmap (dark = low variance = consistent)
        var_img = Image.fromarray(variance_norm)
        var_path = output_dir / f"{pdf_name}_variance_heatmap.png"
        var_img.save(var_path)
        print(f"  Saved variance heatmap: {var_path}")

        # Invert: make low variance BRIGHT (title block = white)
        inverted = 255 - variance_norm
        inv_img = Image.fromarray(inverted)
        inv_path = output_dir / f"{pdf_name}_consistency_map.png"
        inv_img.save(inv_path)
        print(f"  Saved consistency map: {inv_path}")

        # Threshold to find consistent regions
        # Pixels with variance below threshold are "consistent"
        threshold = np.percentile(variance, 10)  # Bottom 10% variance
        consistent_mask = (variance < threshold).astype(np.uint8) * 255

        mask_img = Image.fromarray(consistent_mask)
        mask_path = output_dir / f"{pdf_name}_consistent_mask.png"
        mask_img.save(mask_path)
        print(f"  Saved consistent region mask: {mask_path}")

        # Find bounding box of consistent region on right side
        # Title blocks are typically on the right edge
        height, width = consistent_mask.shape

        # Look for consistent region in right 40% of page
        right_region = consistent_mask[:, int(width * 0.6):]

        # Find rows and columns with consistent pixels
        rows_with_consistency = np.any(right_region > 0, axis=1)
        cols_with_consistency = np.any(right_region > 0, axis=0)

        if np.any(rows_with_consistency) and np.any(cols_with_consistency):
            y_indices = np.where(rows_with_consistency)[0]
            x_indices = np.where(cols_with_consistency)[0]

            y1, y2 = y_indices[0], y_indices[-1]
            # Offset x by the right region start
            x1 = int(width * 0.6) + x_indices[0]
            x2 = int(width * 0.6) + x_indices[-1]

            print(f"\n  Detected consistent region (right edge):")
            print(f"    Pixels: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"    Percent: x1={x1/width:.2f}, y1={y1/height:.2f}, x2={x2/width:.2f}, y2={y2/height:.2f}")
            print(f"    Width: {(x2-x1)/width*100:.1f}% of page")

            # Draw bounding box on first sample page
            first_page = Image.fromarray(images[0].astype(np.uint8))
            first_page = first_page.convert('RGB')
            draw = ImageDraw.Draw(first_page)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

            bbox_path = output_dir / f"{pdf_name}_detected_titleblock.png"
            first_page.save(bbox_path)
            print(f"  Saved detected title block: {bbox_path}")

            return {
                'bbox_pixels': (x1, y1, x2, y2),
                'bbox_percent': {
                    'x1': x1/width,
                    'y1': y1/height,
                    'x2': x2/width,
                    'y2': y2/height
                },
                'width_percent': (x2-x1)/width,
                'height_percent': (y2-y1)/height,
                'variance_threshold': threshold,
                'output_dir': str(output_dir)
            }
        else:
            print("  WARNING: Could not detect consistent region")
            return None


def compare_edge_structures(pdf_path: Path, sample_pages: list = None, dpi: int = 100):
    """
    Alternative approach: Compare edge/contour structures across pages.

    Uses Sobel edge detection to find lines, then compares which lines
    are consistent across all pages (= title block border lines).
    """
    try:
        from scipy import ndimage
    except ImportError:
        print("scipy not installed - skipping edge analysis")
        return None

    output_dir = Path(r"C:\tb\blueprint_processor\output\research")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_path.stem[:30]

    with PDFHandler(pdf_path) as handler:
        total_pages = handler.page_count

        if sample_pages is None:
            start = 2 if total_pages > 5 else 1
            sample_pages = list(range(start, min(start + 5, total_pages + 1)))

        print(f"\nEdge Analysis: {pdf_path.name}")

        edge_images = []
        for page_num in sample_pages:
            img = handler.get_page_image(page_num - 1, dpi=dpi)
            gray = np.array(img.convert('L'), dtype=np.float32)

            # Sobel edge detection (horizontal and vertical)
            sobel_x = ndimage.sobel(gray, axis=1)
            sobel_y = ndimage.sobel(gray, axis=0)
            edges = np.hypot(sobel_x, sobel_y)

            # Normalize
            edges = (edges / edges.max() * 255).astype(np.uint8)
            edge_images.append(edges)

        # Stack and find consistent edges
        stack = np.stack(edge_images, axis=0)

        # Find edges that appear in ALL pages (intersection)
        # Threshold each edge image
        binary_edges = [img > 50 for img in edge_images]

        # AND all binary edge images together
        common_edges = binary_edges[0].copy()
        for edges in binary_edges[1:]:
            common_edges = common_edges & edges

        # Save common edges
        common_img = Image.fromarray((common_edges * 255).astype(np.uint8))
        common_path = output_dir / f"{pdf_name}_common_edges.png"
        common_img.save(common_path)
        print(f"  Saved common edges: {common_path}")

        return {'common_edges_path': str(common_path)}


def main():
    """Test on sample PDFs."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    # Test on a few PDFs
    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
    ]

    print("="*70)
    print("RESEARCH: Multi-Page Comparison for Title Block Detection")
    print("="*70)

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if pdf_path.exists():
            try:
                result = analyze_page_variance(pdf_path)
                if result:
                    print(f"\n  Result: Title block is {result['width_percent']*100:.1f}% wide")
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
