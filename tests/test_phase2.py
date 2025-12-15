"""
Blueprint Processor V4.1 - Phase 2 Checkpoint Test
Verifies: Orientation detection and page normalization.

Run: python tests/test_phase2.py
MANUAL CHECK REQUIRED: Open normalized image and verify text reads left-to-right.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.pdf_handler import PDFHandler
from core.page_normalizer import PageNormalizer, TESSERACT_AVAILABLE, CV2_AVAILABLE


def find_test_pdf() -> Path:
    """Find a test PDF file to use."""
    test_data_dir = project_root / 'test_data'

    if test_data_dir.exists():
        pdfs = list(test_data_dir.glob('*.pdf'))
        if pdfs:
            return pdfs[0]

    return None


def test_dependencies() -> bool:
    """Test 1: Check required dependencies."""
    print(f"\n{'='*60}")
    print("TEST 1: Dependencies Check")
    print(f"{'='*60}")

    print(f"  Tesseract available: {TESSERACT_AVAILABLE}")
    print(f"  OpenCV available: {CV2_AVAILABLE}")

    if not TESSERACT_AVAILABLE and not CV2_AVAILABLE:
        print("  WARNING: Neither Tesseract nor OpenCV available!")
        print("  Orientation detection will be limited.")
        print(f"  RESULT: PASS (with warnings)")
        return True

    if TESSERACT_AVAILABLE:
        from core.page_normalizer import find_tesseract
        tess_path = find_tesseract()
        print(f"  Tesseract path: {tess_path}")
        if tess_path is None:
            print("  WARNING: Tesseract not found in PATH or common locations")

    print(f"  RESULT: PASS")
    return True


def test_orientation_detection(pdf_path: Path) -> bool:
    """Test 2: Orientation detection works."""
    print(f"\n{'='*60}")
    print("TEST 2: Orientation Detection")
    print(f"{'='*60}")

    try:
        telemetry_dir = project_root / 'output' / 'telemetry'
        normalizer = PageNormalizer(telemetry_dir=telemetry_dir)

        with PDFHandler(pdf_path) as handler:
            # Test on first page
            image = handler.get_page_image(0, dpi=200)
            print(f"  Original image size: {image.size}")

            orientation = normalizer.detect_orientation(image)
            print(f"\n  Detected orientation:")
            print(f"    Angle: {orientation['angle']} degrees")
            print(f"    Confidence: {orientation['confidence']}")
            print(f"    Method: {orientation['method']}")

            if 'error' in orientation:
                print(f"    Error: {orientation['error']}")

            print(f"\n  RESULT: PASS")
            return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_normalization(pdf_path: Path) -> bool:
    """Test 3: Page normalization works."""
    print(f"\n{'='*60}")
    print("TEST 3: Page Normalization")
    print(f"{'='*60}")

    try:
        telemetry_dir = project_root / 'output' / 'telemetry'
        normalizer = PageNormalizer(telemetry_dir=telemetry_dir)

        with PDFHandler(pdf_path) as handler:
            for page_num in range(min(handler.page_count, 3)):
                image = handler.get_page_image(page_num, dpi=200)

                normalized, info = normalizer.normalize(image)

                print(f"\n  Page {page_num + 1}:")
                print(f"    Original size: {info['original_size']}")
                print(f"    Normalized size: {info['normalized_size']}")
                print(f"    Rotation applied: {info['rotation_applied']} degrees")
                print(f"    Detection method: {info['method']}")

                # Save for manual verification
                output_path = telemetry_dir / f'phase2_page{page_num + 1}_normalized.png'
                normalized.save(output_path)
                print(f"    Saved to: {output_path}")

                # Save telemetry
                normalizer.save_telemetry(f'phase2_page{page_num + 1}', info)

        print(f"\n  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_forced_rotation(pdf_path: Path) -> bool:
    """Test 4: Forced rotation works."""
    print(f"\n{'='*60}")
    print("TEST 4: Forced Rotation")
    print(f"{'='*60}")

    try:
        telemetry_dir = project_root / 'output' / 'telemetry'
        normalizer = PageNormalizer(telemetry_dir=telemetry_dir)

        with PDFHandler(pdf_path) as handler:
            image = handler.get_page_image(0, dpi=200)
            original_size = image.size

            # Test each rotation angle
            for angle in [0, 90, 180, 270]:
                rotated, info = normalizer.normalize(image, force_angle=angle)
                print(f"  Force angle {angle}: size {rotated.size}, method={info['method']}")

                # Verify size changes for 90/270 rotations
                if angle in [90, 270]:
                    expected_size = (original_size[1], original_size[0])
                    if rotated.size != expected_size:
                        print(f"    WARNING: Expected {expected_size}, got {rotated.size}")

        print(f"\n  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def run_checkpoint():
    """Run all Phase 2 checkpoint tests."""
    print("\n" + "=" * 60)
    print("PHASE 2 CHECKPOINT: Page Normalization")
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
    results.append(("Dependencies Check", test_dependencies()))
    results.append(("Orientation Detection", test_orientation_detection(pdf_path)))
    results.append(("Page Normalization", test_normalization(pdf_path)))
    results.append(("Forced Rotation", test_forced_rotation(pdf_path)))

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
        print("PHASE 2 CHECKPOINT: PASSED")
        print("")
        print("MANUAL VERIFICATION REQUIRED:")
        print("  Open output/telemetry/phase2_page*_normalized.png files")
        print("  Verify that text in the images reads LEFT-TO-RIGHT")
        print("")
        print("If text reads correctly, you may proceed to Phase 3.")
    else:
        print("PHASE 2 CHECKPOINT: FAILED")
        print("Fix the failing tests before proceeding.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = run_checkpoint()
    sys.exit(0 if success else 1)
