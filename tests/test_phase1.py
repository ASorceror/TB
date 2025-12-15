"""
Blueprint Processor V4.1 - Phase 1 Checkpoint Test
Verifies: PDF loads, text extracts (or identified as scanned), image renders.

Run: python tests/test_phase1.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.pdf_handler import PDFHandler


def find_test_pdf() -> Path:
    """Find a test PDF file to use."""
    test_data_dir = project_root / 'test_data'

    # Look for any PDF in test_data
    if test_data_dir.exists():
        pdfs = list(test_data_dir.glob('*.pdf'))
        if pdfs:
            return pdfs[0]

    # Look in parent directory for any PDF
    parent_dir = project_root.parent
    pdfs = list(parent_dir.glob('*.pdf'))
    if pdfs:
        return pdfs[0]

    # Look recursively in parent (limit depth)
    pdfs = list(parent_dir.glob('**/*.pdf'))
    if pdfs:
        return pdfs[0]

    return None


def test_pdf_loading(pdf_path: Path) -> bool:
    """Test 1: PDF loads without errors."""
    print(f"\n{'='*60}")
    print("TEST 1: PDF Loading")
    print(f"{'='*60}")

    try:
        with PDFHandler(pdf_path) as handler:
            print(f"  PDF Path: {pdf_path}")
            print(f"  Page Count: {handler.page_count}")
            print(f"  RESULT: PASS")
            return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  RESULT: FAIL")
        return False


def test_text_extraction(pdf_path: Path) -> bool:
    """Test 2: Text extraction works."""
    print(f"\n{'='*60}")
    print("TEST 2: Text Extraction")
    print(f"{'='*60}")

    try:
        with PDFHandler(pdf_path) as handler:
            for page_num in range(min(handler.page_count, 3)):  # Test first 3 pages
                text = handler.get_page_text(page_num)
                print(f"\n  Page {page_num + 1}:")
                print(f"    Text length: {len(text)} characters")
                print(f"    Preview (first 200 chars): {text[:200].replace(chr(10), ' ')!r}")

            print(f"\n  RESULT: PASS")
            return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  RESULT: FAIL")
        return False


def test_text_blocks(pdf_path: Path) -> bool:
    """Test 3: Text blocks with positions work."""
    print(f"\n{'='*60}")
    print("TEST 3: Text Blocks with Positions")
    print(f"{'='*60}")

    try:
        with PDFHandler(pdf_path) as handler:
            blocks = handler.get_text_blocks(0)
            print(f"  Page 1 text blocks: {len(blocks)}")

            if blocks:
                print(f"\n  First 5 blocks:")
                for i, block in enumerate(blocks[:5]):
                    bbox = block['bbox']
                    text_preview = block['text'][:50].replace('\n', ' ')
                    print(f"    Block {i+1}: bbox={bbox}, text={text_preview!r}")

            print(f"\n  RESULT: PASS")
            return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  RESULT: FAIL")
        return False


def test_page_analysis(pdf_path: Path) -> bool:
    """Test 4: Page analysis (vector vs scanned detection)."""
    print(f"\n{'='*60}")
    print("TEST 4: Page Analysis (Vector vs Scanned)")
    print(f"{'='*60}")

    try:
        with PDFHandler(pdf_path) as handler:
            for page_num in range(min(handler.page_count, 3)):
                analysis = handler.analyze_page(page_num)
                print(f"\n  Page {page_num + 1}:")
                print(f"    Has text: {analysis['has_text']}")
                print(f"    Text length: {analysis['text_length']}")
                print(f"    Is scanned: {analysis['is_scanned']}")
                print(f"    Has large image: {analysis['has_large_image']}")
                print(f"    Recommendation: {analysis['recommendation']}")

            print(f"\n  RESULT: PASS")
            return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  RESULT: FAIL")
        return False


def test_image_rendering(pdf_path: Path) -> bool:
    """Test 5: Page renders as image."""
    print(f"\n{'='*60}")
    print("TEST 5: Image Rendering")
    print(f"{'='*60}")

    try:
        with PDFHandler(pdf_path) as handler:
            image = handler.get_page_image(0, dpi=200)
            print(f"  Image size: {image.size}")
            print(f"  Image mode: {image.mode}")

            # Optionally save for manual verification
            output_dir = project_root / 'output' / 'telemetry'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / 'phase1_test_render.png'
            image.save(output_path)
            print(f"  Saved to: {output_path}")

            print(f"\n  RESULT: PASS")
            return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  RESULT: FAIL")
        return False


def run_checkpoint():
    """Run all Phase 1 checkpoint tests."""
    print("\n" + "=" * 60)
    print("PHASE 1 CHECKPOINT: PDF Text Extraction")
    print("=" * 60)

    # Find test PDF
    pdf_path = find_test_pdf()
    if pdf_path is None:
        print("\nERROR: No test PDF found!")
        print("Please place a PDF file in test_data/sample.pdf")
        print("Or place any PDF in the project directory")
        return False

    print(f"\nUsing test PDF: {pdf_path}")

    # Run tests
    results = []
    results.append(("PDF Loading", test_pdf_loading(pdf_path)))
    results.append(("Text Extraction", test_text_extraction(pdf_path)))
    results.append(("Text Blocks", test_text_blocks(pdf_path)))
    results.append(("Page Analysis", test_page_analysis(pdf_path)))
    results.append(("Image Rendering", test_image_rendering(pdf_path)))

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("PHASE 1 CHECKPOINT: PASSED")
        print("You may proceed to Phase 2.")
    else:
        print("PHASE 1 CHECKPOINT: FAILED")
        print("Fix the failing tests before proceeding.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = run_checkpoint()
    sys.exit(0 if success else 1)
