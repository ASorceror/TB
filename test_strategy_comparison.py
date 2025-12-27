"""
Strategy Comparison Test

Compare different detection strategies:
- balanced: Uses consensus + AI refinement
- include_all: Uses minimum x1 to capture full title block
- coarse_only: Skip AI refinement

This helps determine which strategy works best for different PDF types.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.title_block_detector import TitleBlockDetector


def test_strategy_comparison():
    """Compare detection strategies on key PDFs."""

    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    # Test on PDFs where we know the results were problematic
    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",  # Was 0.953, should be ~0.85
        "3-7-25 Kriser's Highand Final Set.pdf",           # Was 0.912
        "DQ Matteson A Permit Drawings.pdf",               # Had large spread
        "quarry sally beauty plans 2020 05 22.pdf",        # Was 0.952, too narrow
        "Cascade East_2-17-20 permits.pdf",                # Wide: 0.728
    ]

    print("="*90)
    print("STRATEGY COMPARISON TEST")
    print("="*90)

    strategies = ['balanced', 'include_all', 'coarse_only']

    detector = TitleBlockDetector(use_ai_refinement=True)

    results = []

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if not pdf_path.exists():
            print(f"\nSKIP: {pdf_name}")
            continue

        print(f"\n{'='*80}")
        print(f"PDF: {pdf_name}")

        try:
            with PDFHandler(pdf_path) as handler:
                total_pages = handler.page_count

                # Get sample pages
                if total_pages > 5:
                    sample_pages = [2, 4, 6]
                else:
                    sample_pages = [2, 3] if total_pages >= 3 else [1]

                # Get page images
                page_images = []
                for page_num in sample_pages:
                    img = handler.get_page_image(page_num - 1, dpi=100)
                    page_images.append(img)

                print(f"  Pages: {total_pages}, Sampling: {sample_pages}")

                # Test each strategy
                strategy_results = {}
                for strategy in strategies:
                    result = detector.detect(page_images, strategy=strategy)
                    strategy_results[strategy] = {
                        'x1': result['x1'],
                        'width': result['width_pct'],
                        'method': result['method']
                    }
                    print(f"  {strategy:15}: x1={result['x1']:.3f} ({result['width_pct']*100:.1f}% width)")

                # Show CV_transition for reference
                stages = detector.detect(page_images, strategy='coarse_only')['stages']
                if 'coarse' in stages:
                    cv_t = stages['coarse'].get('estimates', {}).get('cv_transition')
                    if cv_t:
                        print(f"  {'CV_transition':15}: x1={cv_t:.3f} (reference)")

                results.append({
                    'pdf': pdf_name[:40],
                    'strategies': strategy_results
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "="*90)
    print("SUMMARY - STRATEGY COMPARISON")
    print("="*90)
    print(f"{'PDF':<40} {'Balanced':>10} {'Include':>10} {'Coarse':>10}")
    print("-"*90)

    for r in results:
        balanced = r['strategies'].get('balanced', {}).get('x1', 0)
        include_all = r['strategies'].get('include_all', {}).get('x1', 0)
        coarse = r['strategies'].get('coarse_only', {}).get('x1', 0)

        print(f"{r['pdf']:<40} {balanced:>10.3f} {include_all:>10.3f} {coarse:>10.3f}")

    # Calculate differences
    print("\n" + "-"*90)
    print("DIFFERENCES (Balanced - Include_All):")
    for r in results:
        balanced = r['strategies'].get('balanced', {}).get('x1', 0)
        include_all = r['strategies'].get('include_all', {}).get('x1', 0)
        diff = balanced - include_all
        if abs(diff) > 0.01:
            print(f"  {r['pdf']}: {diff:+.3f} ({diff * 100:+.1f}%)")


if __name__ == "__main__":
    test_strategy_comparison()
