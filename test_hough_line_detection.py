"""
Test Hough Line Detection for Title Block Borders

Test the OpenCV HoughLinesP approach on sample blueprints to find
strong vertical lines that could represent the title block left border.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.coarse_detection.line_detector import (
    find_vertical_lines,
    cluster_lines_by_x,
    detect_title_block_border,
    visualize_lines
)
from PIL import Image


def test_hough_line_detection():
    """Test Hough line detection on sample blueprints."""

    output_dir = Path(r"C:\tb\blueprint_processor\output\hough_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "DQ Matteson A Permit Drawings.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
    ]

    print("="*70)
    print("HOUGH LINE DETECTION TEST")
    print("="*70)

    results = []

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if not pdf_path.exists():
            print(f"\nSKIP: {pdf_name}")
            continue

        print(f"\n{'='*60}")
        print(f"PDF: {pdf_name}")

        try:
            with PDFHandler(pdf_path) as handler:
                total_pages = handler.page_count

                # Get sample pages (skip cover)
                if total_pages > 5:
                    sample_pages = [2, 4, 6]
                else:
                    sample_pages = [2, 3] if total_pages >= 3 else [1]

                print(f"  Total pages: {total_pages}")
                print(f"  Sample pages: {sample_pages}")

                # Get page images
                page_images = []
                for page_num in sample_pages:
                    img = handler.get_page_image(page_num - 1, dpi=150)
                    page_images.append(img)

                width, height = page_images[0].size
                print(f"  Page size: {width}x{height}")

                # Test on first page - find all lines
                print("\n  Finding vertical lines on page 1...")
                lines = find_vertical_lines(
                    page_images[0],
                    search_region=(0.60, 0.98),
                    min_line_length_ratio=0.3,
                    threshold=100
                )

                print(f"  Found {len(lines)} vertical lines")

                if lines:
                    # Show top 10 lines by length
                    lines_by_length = sorted(lines, key=lambda l: l['length'], reverse=True)[:10]
                    print("\n  Top 10 lines by length:")
                    for i, line in enumerate(lines_by_length):
                        print(f"    {i+1}. x={line['x_avg_pct']:.3f}, length={line['length_pct']*100:.1f}%")

                    # Cluster lines
                    clusters = cluster_lines_by_x(lines)
                    print(f"\n  Clustered into {len(clusters)} groups:")
                    for i, cluster in enumerate(clusters):
                        print(f"    Cluster {i+1}: x={cluster['x_avg_pct']:.3f}, "
                              f"segments={cluster['num_segments']}, "
                              f"total_length={cluster['total_length_pct']*100:.1f}%")

                    # Visualize
                    safe_name = pdf_name.replace(' ', '_').replace('.pdf', '')[:30]
                    viz_path = output_dir / f"{safe_name}_lines.png"
                    visualize_lines(page_images[0], lines[:20], viz_path)
                    print(f"\n  Visualization saved: {viz_path.name}")

                # Test multi-page detection
                print("\n  Multi-page border detection...")
                border_x1 = detect_title_block_border(
                    page_images,
                    search_region=(0.60, 0.98),
                    min_agreement=2
                )

                if border_x1:
                    print(f"  DETECTED BORDER: x1={border_x1:.3f}")
                    results.append({
                        'pdf': pdf_name[:40],
                        'x1': border_x1,
                        'num_lines': len(lines) if lines else 0
                    })

                    # Draw border on image
                    result_img = page_images[0].convert('RGB')
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(result_img)
                    border_x = int(border_x1 * width)
                    draw.line([(border_x, 0), (border_x, height)], fill='green', width=4)
                    draw.text((border_x + 5, 20), f"Hough: {border_x1:.3f}", fill='green')

                    result_path = output_dir / f"{safe_name}_detected.png"
                    result_img.save(result_path)
                    print(f"  Result saved: {result_path.name}")
                else:
                    print("  WARNING: No consistent border detected")
                    results.append({
                        'pdf': pdf_name[:40],
                        'x1': None,
                        'num_lines': len(lines) if lines else 0
                    })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - HOUGH LINE DETECTION")
    print("="*70)
    print(f"{'PDF':<45} {'x1':>8} {'Lines':>8}")
    print("-"*70)
    for r in results:
        x1_str = f"{r['x1']:.3f}" if r['x1'] else "---"
        print(f"{r['pdf']:<45} {x1_str:>8} {r['num_lines']:>8}")

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    test_hough_line_detection()
