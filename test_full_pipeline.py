"""
Full Pipeline Test - Multi-Stage Title Block Detection

Tests the complete detection pipeline on all test PDFs and compares
different strategies (balanced, conservative, aggressive).
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.title_block_detector import TitleBlockDetector
from PIL import Image, ImageDraw


def test_full_pipeline():
    """Run full pipeline test on all PDFs."""

    output_dir = Path(r"C:\tb\blueprint_processor\output\full_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    # Get all PDFs
    all_pdfs = sorted(test_dir.glob("*.pdf"))

    print("="*80)
    print("FULL PIPELINE TEST - Multi-Stage Title Block Detection")
    print(f"Testing {len(all_pdfs)} PDFs")
    print("="*80)

    # Initialize detector
    detector = TitleBlockDetector(use_ai_refinement=True)

    results = []

    for pdf_idx, pdf_path in enumerate(all_pdfs, 1):
        pdf_name = pdf_path.name
        print(f"\n[{pdf_idx}/{len(all_pdfs)}] {pdf_name[:50]}")

        try:
            with PDFHandler(pdf_path) as handler:
                total_pages = handler.page_count

                # Get sample pages (skip cover)
                if total_pages > 5:
                    sample_pages = [2, 4, 6]
                else:
                    sample_pages = [2, 3] if total_pages >= 3 else [1]

                print(f"  Pages: {total_pages}, Sampling: {sample_pages}")

                # Get page images at 100 DPI for detection
                page_images = []
                for page_num in sample_pages:
                    img = handler.get_page_image(page_num - 1, dpi=100)
                    page_images.append(img)

                # Run detection with balanced strategy
                result = detector.detect(page_images, strategy='balanced')

                x1 = result['x1']
                width_pct = result['width_pct']
                method = result['method']
                confidence = result['confidence']

                print(f"  Result: x1={x1:.3f} ({width_pct*100:.1f}% width)")
                print(f"  Method: {method}, Confidence: {confidence:.2f}")

                # Get stage details
                stages = result.get('stages', {})
                if 'coarse' in stages:
                    coarse = stages['coarse']
                    coarse_x1 = coarse.get('x1', 0)
                    coarse_estimates = coarse.get('estimates', {})
                    cv_t = coarse_estimates.get('cv_transition')
                    cv_m = coarse_estimates.get('cv_majority')
                    hough = coarse_estimates.get('hough_lines')
                    cv_t_str = f"{cv_t:.3f}" if cv_t else "N/A"
                    cv_m_str = f"{cv_m:.3f}" if cv_m else "N/A"
                    hough_str = f"{hough:.3f}" if hough else "N/A"
                    print(f"  Coarse: CV_T={cv_t_str}, CV_M={cv_m_str}, Hough={hough_str}")

                if 'refined' in stages:
                    refined = stages['refined']
                    if 'error' not in refined:
                        print(f"  Refined: {refined.get('x1', 0):.3f} "
                              f"(conf={refined.get('confidence', 0):.2f})")

                # Create visualization
                viz_img = page_images[0].convert('RGB')
                viz_width, viz_height = viz_img.size
                draw = ImageDraw.Draw(viz_img)

                # Draw final boundary (green)
                x_px = int(x1 * viz_width)
                draw.line([(x_px, 0), (x_px, viz_height)], fill='green', width=4)
                draw.text((x_px + 5, 20), f"x1={x1:.3f} ({method})", fill='green')

                # Draw title block rectangle
                draw.rectangle(
                    [x_px, 0, viz_width, viz_height],
                    outline='green',
                    width=2
                )

                safe_name = pdf_name.replace(' ', '_').replace('.pdf', '')[:35]
                viz_path = output_dir / f"{safe_name}_detected.png"
                viz_img.save(viz_path)

                # Also save the cropped title block
                title_block_crop = detector.crop_title_block(page_images[0], result)
                crop_path = output_dir / f"{safe_name}_titleblock.png"
                title_block_crop.save(crop_path)

                results.append({
                    'pdf': pdf_name,
                    'pages': total_pages,
                    'x1': x1,
                    'width_pct': width_pct,
                    'method': method,
                    'confidence': confidence,
                    'cv_transition': cv_t,
                    'cv_majority': cv_m,
                    'hough_lines': hough
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'pdf': pdf_name,
                'error': str(e)
            })

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"pipeline_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*90)
    print("SUMMARY - FULL PIPELINE")
    print("="*90)
    print(f"{'PDF':<45} {'x1':>7} {'Width':>7} {'Method':>15} {'Conf':>6}")
    print("-"*90)

    successful = 0
    for r in results:
        if 'error' in r:
            print(f"{r['pdf'][:44]:<45} ERROR: {r['error'][:30]}")
        else:
            successful += 1
            print(f"{r['pdf'][:44]:<45} {r['x1']:>7.3f} {r['width_pct']*100:>6.1f}% "
                  f"{r['method']:>15} {r['confidence']:>6.2f}")

    print("-"*90)
    print(f"Successful: {successful}/{len(all_pdfs)}")

    # Statistics
    if successful > 0:
        x1_values = [r['x1'] for r in results if 'error' not in r]
        width_values = [r['width_pct'] for r in results if 'error' not in r]

        print(f"\nTitle Block Statistics:")
        print(f"  x1 range: {min(x1_values):.3f} - {max(x1_values):.3f}")
        print(f"  Width range: {min(width_values)*100:.1f}% - {max(width_values)*100:.1f}%")
        print(f"  Mean x1: {sum(x1_values)/len(x1_values):.3f}")
        print(f"  Mean width: {sum(width_values)/len(width_values)*100:.1f}%")

    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    test_full_pipeline()
