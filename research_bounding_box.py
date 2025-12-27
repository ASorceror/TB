"""
Title Block Detection via Bounding Box of Consistent Structure

Approach:
1. Get common edges (structure present on ALL pages)
2. In the right portion of the page, find ALL consistent edge pixels
3. Calculate the bounding box that contains these pixels
4. The leftmost edge = title block left boundary

This captures BOTH sparse sections (vertical text) and dense sections (tables).
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler


def find_title_block_bbox(pdf_path: Path, sample_pages: list = None, dpi: int = 100):
    """
    Find title block by computing bounding box of all consistent edges.
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

        # Step 3: Focus on right portion (where title block should be)
        # Look in right 50% of page
        search_start = int(width * 0.5)
        right_portion = common_edges[:, search_start:]

        # Step 4: Find all edge pixels in right portion
        edge_coords = np.where(right_portion > 128)

        if len(edge_coords[0]) == 0:
            print("  WARNING: No consistent edges found in right portion")
            return None

        # Get bounding box of all edge pixels
        y_coords = edge_coords[0]
        x_coords = edge_coords[1] + search_start  # Offset back to full image coords

        bbox_y1 = np.min(y_coords)
        bbox_y2 = np.max(y_coords)
        bbox_x1 = np.min(x_coords)
        bbox_x2 = np.max(x_coords)

        print(f"\n  Raw bounding box of consistent edges:")
        print(f"    Pixels: ({bbox_x1}, {bbox_y1}) to ({bbox_x2}, {bbox_y2})")

        # Step 5: Refine - the title block is usually a clean rectangle
        # Use morphological operations to find connected components

        # Label connected regions
        labeled, num_features = ndimage.label(common_edges > 128)
        print(f"  Found {num_features} connected components")

        # Find the largest component in the right portion
        right_labeled = labeled[:, search_start:]
        unique_labels = np.unique(right_labeled)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background

        if len(unique_labels) == 0:
            print("  WARNING: No connected components in right portion")
            return None

        # Find size of each component
        component_sizes = {}
        for label in unique_labels:
            component_sizes[label] = np.sum(right_labeled == label)

        # Get the largest component
        largest_label = max(component_sizes, key=component_sizes.get)
        largest_mask = (labeled == largest_label)

        # Get bounding box of largest component
        coords = np.where(largest_mask)
        comp_y1 = np.min(coords[0])
        comp_y2 = np.max(coords[0])
        comp_x1 = np.min(coords[1])
        comp_x2 = np.max(coords[1])

        print(f"\n  Largest connected component:")
        print(f"    Size: {component_sizes[largest_label]} pixels")
        print(f"    Bounding box: ({comp_x1}, {comp_y1}) to ({comp_x2}, {comp_y2})")

        # Use the larger of: raw bbox or largest component bbox
        # (in case title block is multiple disconnected pieces)
        final_x1 = min(bbox_x1, comp_x1)
        final_y1 = min(bbox_y1, comp_y1)
        final_x2 = max(bbox_x2, comp_x2)
        final_y2 = max(bbox_y2, comp_y2)

        # Calculate percentages
        x1_pct = final_x1 / width
        y1_pct = final_y1 / height
        x2_pct = final_x2 / width
        y2_pct = final_y2 / height
        width_pct = (final_x2 - final_x1) / width

        print(f"\n  FINAL TITLE BLOCK DETECTION:")
        print(f"    Pixels: ({final_x1}, {final_y1}) to ({final_x2}, {final_y2})")
        print(f"    Percent: x1={x1_pct:.3f}, y1={y1_pct:.3f}, x2={x2_pct:.3f}, y2={y2_pct:.3f}")
        print(f"    Width: {width_pct*100:.1f}% of page")

        # Save visualization
        first_page = handler.get_page_image(sample_pages[0] - 1, dpi=dpi)
        if first_page.size != target_size:
            first_page = first_page.resize(target_size, Image.Resampling.LANCZOS)

        result_img = first_page.convert('RGB')
        draw = ImageDraw.Draw(result_img)

        # Red box for final detection
        draw.rectangle([final_x1, final_y1, final_x2, final_y2], outline='red', width=4)

        # Green line at left boundary
        draw.line([(final_x1, 0), (final_x1, height)], fill='green', width=3)

        # Blue dashed line at search start
        for y in range(0, height, 20):
            draw.line([(search_start, y), (search_start, min(y+10, height))], fill='blue', width=1)

        detection_path = output_dir / f"{pdf_name}_detected_bbox.png"
        result_img.save(detection_path)
        print(f"  Saved: {detection_path}")

        # Save common edges with bbox overlay
        common_rgb = Image.fromarray(common_edges).convert('RGB')
        draw_common = ImageDraw.Draw(common_rgb)
        draw_common.rectangle([final_x1, final_y1, final_x2, final_y2], outline='red', width=2)
        common_path = output_dir / f"{pdf_name}_common_with_bbox.png"
        common_rgb.save(common_path)
        print(f"  Saved: {common_path}")

        return {
            'bbox_pixels': (final_x1, final_y1, final_x2, final_y2),
            'bbox_percent': {
                'x1': x1_pct,
                'y1': y1_pct,
                'x2': x2_pct,
                'y2': y2_pct
            },
            'width_percent': width_pct,
        }


def main():
    """Test bounding box approach."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
        "Janesville Nissan Full set Issued for Bids.pdf",
    ]

    print("="*70)
    print("BOUNDING BOX: Title Block Detection via Edge Bounding Box")
    print("="*70)

    results = []

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if pdf_path.exists():
            try:
                result = find_title_block_bbox(pdf_path)
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
    print("SUMMARY - BOUNDING BOX DETECTION")
    print("="*70)
    print(f"{'PDF':<45} {'x1':>8} {'Width':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['pdf']:<45} {r['x1']:>7.3f} {r['width']*100:>7.1f}%")

    print("\nOutput saved to: C:\\tb\\blueprint_processor\\output\\research")


if __name__ == "__main__":
    main()
