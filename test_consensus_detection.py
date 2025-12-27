"""
Test Consensus Detection System

Tests the multi-method consensus voting system for coarse title block detection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.coarse_detection import CoarseDetector


def test_consensus_detection():
    """Test consensus detection on sample blueprints."""

    output_dir = Path(r"C:\tb\blueprint_processor\output\consensus_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "DQ Matteson A Permit Drawings.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
        "Janesville Nissan Full set Issued for Bids.pdf",
    ]

    print("="*70)
    print("CONSENSUS DETECTION TEST")
    print("="*70)

    detector = CoarseDetector()
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
                    img = handler.get_page_image(page_num - 1, dpi=100)
                    page_images.append(img)

                # Run consensus detection with visualization
                safe_name = pdf_name.replace(' ', '_').replace('.pdf', '')[:30]
                viz_path = output_dir / f"{safe_name}_consensus.png"

                result, viz_img = detector.detect_with_visualization(
                    page_images,
                    output_path=viz_path,
                    strategy='median'
                )

                # Print results
                print(f"\n  Individual Estimates:")
                for method, value in result['estimates'].items():
                    if not method.endswith('_error'):
                        print(f"    {method}: {value:.3f}")
                    else:
                        print(f"    {method}: {value}")

                print(f"\n  CONSENSUS: x1={result['x1']:.3f}")
                print(f"  Spread: {result['spread']:.3f}")
                print(f"  Confidence: {result['confidence']:.2f}")
                print(f"  Method: {result['method']}")
                print(f"  Saved: {viz_path.name}")

                results.append({
                    'pdf': pdf_name[:40],
                    'x1': result['x1'],
                    'spread': result['spread'],
                    'confidence': result['confidence'],
                    'cv_trans': result['estimates'].get('cv_transition'),
                    'cv_maj': result['estimates'].get('cv_majority'),
                    'hough': result['estimates'].get('hough_lines')
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - CONSENSUS DETECTION")
    print("="*80)
    print(f"{'PDF':<40} {'CV_T':>7} {'CV_M':>7} {'Hough':>7} {'FINAL':>7} {'Spread':>7}")
    print("-"*80)
    for r in results:
        cv_t = f"{r['cv_trans']:.3f}" if r['cv_trans'] else "---"
        cv_m = f"{r['cv_maj']:.3f}" if r['cv_maj'] else "---"
        hough = f"{r['hough']:.3f}" if r['hough'] else "---"
        print(f"{r['pdf']:<40} {cv_t:>7} {cv_m:>7} {hough:>7} {r['x1']:>7.3f} {r['spread']:>7.3f}")

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    test_consensus_detection()
