"""
Blueprint Processor V4.1 - Phase 3 Checkpoint Test
Verifies: Title block region detection.

Run: python tests/test_phase3.py
MANUAL CHECK REQUIRED: Open cropped region image and verify it shows the title block.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.pdf_handler import PDFHandler
from core.page_normalizer import PageNormalizer
from core.region_detector import RegionDetector, get_region_visualization


def find_test_pdf() -> Path:
    """Find a test PDF file to use."""
    test_data_dir = project_root / 'test_data'

    if test_data_dir.exists():
        pdfs = list(test_data_dir.glob('*.pdf'))
        if pdfs:
            return pdfs[0]

    return None


def test_region_detection(pdf_path: Path) -> bool:
    """Test 1: Region detection finds title block."""
    print(f"\n{'='*60}")
    print("TEST 1: Title Block Detection")
    print(f"{'='*60}")

    try:
        telemetry_dir = project_root / 'output' / 'telemetry'
        normalizer = PageNormalizer(telemetry_dir=telemetry_dir)
        detector = RegionDetector(telemetry_dir=telemetry_dir)

        with PDFHandler(pdf_path) as handler:
            # Test on first few pages
            for page_num in range(min(handler.page_count, 3)):
                print(f"\n  Page {page_num + 1}:")

                # Get and normalize image
                image = handler.get_page_image(page_num, dpi=200)
                normalized, _ = normalizer.normalize(image)

                # Detect title block
                result = detector.detect_title_block(normalized)

                print(f"    Selected region: {result['region_name']}")
                print(f"    Confidence: {result['confidence']}")
                print(f"    Keywords found: {result['keywords_found']}")
                print(f"    Score: {result['score']:.2f}")
                print(f"    Bbox: {result['bbox']}")
                print(f"    Reason: {result['selection_reason']}")

                # Show all candidates
                print(f"\n    All candidates:")
                for cand in result['all_candidates']:
                    print(f"      {cand['region_name']}: {cand['keyword_count']} keywords, score={cand['score']:.2f}")

                # Crop and save title block
                cropped = detector.crop_title_block(normalized, result)
                crop_path = telemetry_dir / f'phase3_page{page_num + 1}_titleblock.png'
                cropped.save(crop_path)
                print(f"\n    Cropped title block saved to: {crop_path}")

                # Save visualization
                viz = get_region_visualization(normalized, result)
                viz_path = telemetry_dir / f'phase3_page{page_num + 1}_regions.png'
                viz.save(viz_path)
                print(f"    Region visualization saved to: {viz_path}")

                # Save telemetry
                detector.save_telemetry(f'phase3_page{page_num + 1}', result, cropped)

        print(f"\n  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_region_coordinates(pdf_path: Path) -> bool:
    """Test 2: Region coordinates are valid and relative."""
    print(f"\n{'='*60}")
    print("TEST 2: Region Coordinate Validation")
    print(f"{'='*60}")

    try:
        from constants import TITLE_BLOCK_REGIONS

        print("  Checking defined regions:")
        for name, coords in TITLE_BLOCK_REGIONS.items():
            x1, y1, x2, y2 = coords

            # Verify all coords are relative (0-1)
            valid = all(0 <= c <= 1 for c in coords)

            # Verify x1 < x2 and y1 < y2
            valid = valid and x1 < x2 and y1 < y2

            status = "OK" if valid else "INVALID"
            print(f"    {name}: ({x1}, {y1}) to ({x2}, {y2}) - {status}")

            if not valid:
                return False

        print(f"\n  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  RESULT: FAIL")
        return False


def test_keyword_matching(pdf_path: Path) -> bool:
    """Test 3: Keyword matching works correctly."""
    print(f"\n{'='*60}")
    print("TEST 3: Keyword Matching")
    print(f"{'='*60}")

    try:
        detector = RegionDetector()

        # Test cases
        test_texts = [
            ("SHEET NUMBER: A101\nPROJECT: 2024-001", ["SHEET", "PROJECT"]),
            ("DRAWING TITLE: FLOOR PLAN\nDATE: 12/15/2024", ["DRAWING", "TITLE", "DATE"]),
            ("SCALE: 1/4\" = 1'-0\"\nREVISION: 3", ["SCALE", "REVISION"]),
            ("DRAWN BY: JD\nCHECKED BY: MR\nAPPROVED BY: TL", ["DRAWN", "CHECKED", "APPROVED", "BY"]),
            ("Random text with no keywords", []),
        ]

        all_passed = True
        for text, expected_keywords in test_texts:
            found = detector._count_keywords(text)
            found_set = set(found)
            expected_set = set(expected_keywords)

            # Check that expected keywords are found
            missing = expected_set - found_set
            if missing:
                print(f"  FAIL: Missing keywords {missing} in text: {text[:50]}...")
                all_passed = False
            else:
                print(f"  OK: Found {len(found)} keywords in text: {text[:50]}...")

        if all_passed:
            print(f"\n  RESULT: PASS")
        else:
            print(f"\n  RESULT: FAIL")

        return all_passed
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  RESULT: FAIL")
        return False


def run_checkpoint():
    """Run all Phase 3 checkpoint tests."""
    print("\n" + "=" * 60)
    print("PHASE 3 CHECKPOINT: Title Block Detection")
    print("=" * 60)

    # Find test PDF
    pdf_path = find_test_pdf()
    if pdf_path is None:
        print("\nERROR: No test PDF found!")
        print("Please place a PDF file in test_data/sample.pdf")
        return False

    print(f"\nUsing test PDF: {pdf_path}")

    # Run tests
    results = []
    results.append(("Region Coordinates", test_region_coordinates(pdf_path)))
    results.append(("Keyword Matching", test_keyword_matching(pdf_path)))
    results.append(("Title Block Detection", test_region_detection(pdf_path)))

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
        print("PHASE 3 CHECKPOINT: PASSED")
        print("")
        print("MANUAL VERIFICATION REQUIRED:")
        print("  Open output/telemetry/phase3_page*_titleblock.png files")
        print("  Verify that the cropped region shows the actual title block")
        print("  Open output/telemetry/phase3_page*_regions.png for visualization")
        print("")
        print("If title block is correctly identified, you may proceed to Phase 4.")
    else:
        print("PHASE 3 CHECKPOINT: FAILED")
        print("Fix the failing tests before proceeding.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = run_checkpoint()
    sys.exit(0 if success else 1)
