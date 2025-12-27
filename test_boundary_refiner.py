"""
Test Constrained AI Vision Boundary Refinement

Tests the AI Vision refinement on a constrained crop to improve
boundary accuracy beyond coarse detection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.coarse_detection import CoarseDetector
from core.boundary_refiner import BoundaryRefiner
from PIL import Image, ImageDraw


def test_boundary_refiner():
    """Test boundary refinement on sample blueprints."""

    output_dir = Path(r"C:\tb\blueprint_processor\output\refiner_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "DQ Matteson A Permit Drawings.pdf",
    ]

    print("="*70)
    print("BOUNDARY REFINEMENT TEST (Coarse + AI Vision)")
    print("="*70)

    coarse_detector = CoarseDetector()
    refiner = BoundaryRefiner(margin=0.10)

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

                # Get sample pages
                if total_pages > 5:
                    sample_pages = [2, 4, 6]
                else:
                    sample_pages = [2, 3] if total_pages >= 3 else [1]

                print(f"  Total pages: {total_pages}")
                print(f"  Sample pages: {sample_pages}")

                # Get page images
                page_images = []
                for page_num in sample_pages:
                    img = handler.get_page_image(page_num - 1, dpi=100)
                    page_images.append(img)

                # Stage 1: Coarse detection
                print("\n  Stage 1: Coarse Detection...")
                coarse_result = coarse_detector.detect(page_images, strategy='median')
                coarse_x1 = coarse_result['x1']
                print(f"    Coarse x1: {coarse_x1:.3f}")
                print(f"    Spread: {coarse_result['spread']:.3f}")

                # Stage 2: AI Vision refinement
                print("\n  Stage 2: AI Vision Refinement...")

                # Use first page for refinement (higher res for AI)
                refine_img = handler.get_page_image(sample_pages[0] - 1, dpi=150)

                refined_result = refiner.refine(refine_img, coarse_x1)

                refined_x1 = refined_result['x1']
                print(f"    Refined x1: {refined_x1:.3f}")
                print(f"    Confidence: {refined_result.get('confidence', 'N/A')}")
                print(f"    Has border line: {refined_result.get('has_border_line', 'N/A')}")
                if refined_result.get('description'):
                    print(f"    Description: {refined_result['description'][:80]}...")
                if refined_result.get('error'):
                    print(f"    ERROR: {refined_result['error']}")

                # Calculate change
                delta = refined_x1 - coarse_x1
                print(f"\n    Delta: {delta:+.3f} ({delta * 100:+.1f}%)")

                # Create visualization
                width, height = refine_img.size
                viz_img = refine_img.convert('RGB')
                draw = ImageDraw.Draw(viz_img)

                # Draw coarse boundary (blue dashed)
                coarse_px = int(coarse_x1 * width)
                for y in range(0, height, 30):
                    draw.line([(coarse_px, y), (coarse_px, min(y + 15, height))], fill='blue', width=3)
                draw.text((coarse_px + 5, 20), f"Coarse: {coarse_x1:.3f}", fill='blue')

                # Draw refined boundary (green solid)
                refined_px = int(refined_x1 * width)
                draw.line([(refined_px, 0), (refined_px, height)], fill='green', width=4)
                draw.text((refined_px + 5, 50), f"Refined: {refined_x1:.3f}", fill='green')

                # Draw crop region
                crop_start = max(0, coarse_x1 - 0.10)
                crop_px = int(crop_start * width)
                draw.line([(crop_px, 0), (crop_px, 50)], fill='orange', width=2)
                draw.text((crop_px + 5, 80), f"Crop start: {crop_start:.3f}", fill='orange')

                safe_name = pdf_name.replace(' ', '_').replace('.pdf', '')[:30]
                viz_path = output_dir / f"{safe_name}_refined.png"
                viz_img.save(viz_path)
                print(f"\n  Saved: {viz_path.name}")

                results.append({
                    'pdf': pdf_name[:40],
                    'coarse': coarse_x1,
                    'refined': refined_x1,
                    'delta': delta,
                    'confidence': refined_result.get('confidence', 0)
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - BOUNDARY REFINEMENT")
    print("="*70)
    print(f"{'PDF':<40} {'Coarse':>8} {'Refined':>8} {'Delta':>8} {'Conf':>6}")
    print("-"*70)
    for r in results:
        conf_str = f"{r['confidence']:.2f}" if r['confidence'] else "---"
        print(f"{r['pdf']:<40} {r['coarse']:>8.3f} {r['refined']:>8.3f} {r['delta']:>+8.3f} {conf_str:>6}")

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    test_boundary_refiner()
