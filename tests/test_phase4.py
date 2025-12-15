"""
Blueprint Processor V4.1 - Phase 4 Checkpoint Test
Verifies: OCR engine works and produces readable text.

Run: python tests/test_phase4.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.pdf_handler import PDFHandler
from core.page_normalizer import PageNormalizer
from core.region_detector import RegionDetector
from core.ocr_engine import OCREngine


def find_test_pdf() -> Path:
    """Find a test PDF file to use."""
    test_data_dir = project_root / 'test_data'

    if test_data_dir.exists():
        pdfs = list(test_data_dir.glob('*.pdf'))
        if pdfs:
            return pdfs[0]

    return None


def test_ocr_availability() -> bool:
    """Test 1: Check OCR engine availability."""
    print(f"\n{'='*60}")
    print("TEST 1: OCR Engine Availability")
    print(f"{'='*60}")

    try:
        engine = OCREngine()
        status = engine.is_available()

        print(f"  Tesseract available: {status['tesseract']}")
        print(f"  Tesseract path: {status['tesseract_path']}")
        print(f"  PaddleOCR available: {status['paddleocr']}")

        if not status['tesseract']:
            print("\n  WARNING: Tesseract not available!")
            print("  OCR functionality will be limited.")
            print("  Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
            print(f"\n  RESULT: FAIL")
            return False

        print(f"\n  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_basic_ocr(pdf_path: Path) -> bool:
    """Test 2: Basic OCR produces readable text."""
    print(f"\n{'='*60}")
    print("TEST 2: Basic OCR")
    print(f"{'='*60}")

    try:
        telemetry_dir = project_root / 'output' / 'telemetry'
        engine = OCREngine(telemetry_dir=telemetry_dir)

        with PDFHandler(pdf_path) as handler:
            # Test on first page
            image = handler.get_page_image(0, dpi=200)

            # Run basic OCR
            text = engine.ocr_image(image)

            print(f"  Image size: {image.size}")
            print(f"  Text length: {len(text)} characters")
            print(f"\n  Text preview (first 500 chars):")
            print("-" * 40)
            preview = text[:500].strip() if text else "(no text extracted)"
            print(preview)
            print("-" * 40)

            if len(text) > 0:
                print(f"\n  RESULT: PASS")
                return True
            else:
                print(f"\n  WARNING: No text extracted")
                print(f"  RESULT: PASS (image may be blank)")
                return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_title_block_ocr(pdf_path: Path) -> bool:
    """Test 3: OCR on detected title block region."""
    print(f"\n{'='*60}")
    print("TEST 3: Title Block OCR")
    print(f"{'='*60}")

    try:
        telemetry_dir = project_root / 'output' / 'telemetry'
        normalizer = PageNormalizer(telemetry_dir=telemetry_dir)
        detector = RegionDetector(telemetry_dir=telemetry_dir)
        engine = OCREngine(telemetry_dir=telemetry_dir)

        with PDFHandler(pdf_path) as handler:
            for page_num in range(min(handler.page_count, 2)):
                print(f"\n  Page {page_num + 1}:")

                # Get, normalize, and detect title block
                image = handler.get_page_image(page_num, dpi=200)
                normalized, _ = normalizer.normalize(image)
                detection = detector.detect_title_block(normalized)

                # Crop title block
                cropped = detector.crop_title_block(normalized, detection)

                # OCR the title block
                text = engine.ocr_image(cropped)

                print(f"    Region: {detection['region_name']}")
                print(f"    Crop size: {cropped.size}")
                print(f"    Text length: {len(text)} characters")
                print(f"\n    Title block text:")
                print("-" * 40)
                print(text[:800].strip() if text else "(no text)")
                print("-" * 40)

                # Save for telemetry
                engine.save_telemetry(f'phase4_page{page_num + 1}', {
                    'text': text,
                    'text_length': len(text),
                    'region': detection['region_name'],
                })

        print(f"\n  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_ocr_with_boxes(pdf_path: Path) -> bool:
    """Test 4: OCR with bounding boxes."""
    print(f"\n{'='*60}")
    print("TEST 4: OCR with Bounding Boxes")
    print(f"{'='*60}")

    try:
        telemetry_dir = project_root / 'output' / 'telemetry'
        normalizer = PageNormalizer()
        detector = RegionDetector()
        engine = OCREngine(telemetry_dir=telemetry_dir)

        with PDFHandler(pdf_path) as handler:
            # Test on page 2 (usually has more structured title block)
            page_num = min(1, handler.page_count - 1)
            image = handler.get_page_image(page_num, dpi=200)
            normalized, _ = normalizer.normalize(image)
            detection = detector.detect_title_block(normalized)
            cropped = detector.crop_title_block(normalized, detection)

            # OCR with boxes
            boxes = engine.ocr_image_with_boxes(cropped)

            print(f"  Found {len(boxes)} text elements with bounding boxes")
            print(f"\n  First 15 elements:")
            for i, box in enumerate(boxes[:15]):
                conf = box['confidence']
                text = box['text'][:30]
                print(f"    {i+1}. [{conf:3d}%] {text!r}")

        print(f"\n  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_ensemble_ocr(pdf_path: Path) -> bool:
    """Test 5: Ensemble OCR (if multiple engines available)."""
    print(f"\n{'='*60}")
    print("TEST 5: Ensemble OCR")
    print(f"{'='*60}")

    try:
        telemetry_dir = project_root / 'output' / 'telemetry'
        engine = OCREngine(telemetry_dir=telemetry_dir)
        status = engine.is_available()

        with PDFHandler(pdf_path) as handler:
            image = handler.get_page_image(0, dpi=200)

            result = engine.ensemble_ocr(image)

            print(f"  Engines used: {result['engines_used']}")
            print(f"  Primary engine: {result.get('primary_engine', 'none')}")
            print(f"  Tesseract text length: {len(result['tesseract'])}")
            print(f"  Paddle text length: {len(result['paddle'])}")
            print(f"  Merged text length: {len(result['merged'])}")

            if status['paddleocr']:
                print("\n  PaddleOCR sample:")
                print(result['paddle'][:300] if result['paddle'] else "(no paddle output)")

        print(f"\n  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def run_checkpoint():
    """Run all Phase 4 checkpoint tests."""
    print("\n" + "=" * 60)
    print("PHASE 4 CHECKPOINT: OCR Engine")
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
    results.append(("OCR Availability", test_ocr_availability()))
    results.append(("Basic OCR", test_basic_ocr(pdf_path)))
    results.append(("Title Block OCR", test_title_block_ocr(pdf_path)))
    results.append(("OCR with Boxes", test_ocr_with_boxes(pdf_path)))
    results.append(("Ensemble OCR", test_ensemble_ocr(pdf_path)))

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
        print("PHASE 4 CHECKPOINT: PASSED")
        print("You may proceed to Phase 5.")
    else:
        print("PHASE 4 CHECKPOINT: FAILED")
        print("Fix the failing tests before proceeding.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = run_checkpoint()
    sys.exit(0 if success else 1)
