"""
Blueprint Processor V4.2.1 - Integration Tests
Tests the full extraction flow with drawing index lookup.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extractor import Extractor
from core.pdf_handler import PDFHandler
from core.ocr_engine import OCREngine


def test_extractor_with_drawing_index():
    """Test that extractor uses drawing index when available."""
    print("=== Test: Extractor with Drawing Index ===")

    extractor = Extractor()

    # Set up a mock drawing index
    extractor.set_drawing_index({
        'A1.0': 'Code Review, Schedules and ADA Requirements',
        'A2.0': 'Floor Plan',
        'A2.1': 'Enlarged Floor Plan',
        'A2.2': 'Reflected Ceiling Plan',
        'D2.0': 'Demolition Floor Plan',
    })

    # Test extraction with a sheet number that's in the index
    text = """
    SHEET NUMBER: A2.0
    PROJECT NO: P27142
    DATE: 12/2/2019
    """

    result = extractor.extract_fields(text, page_number=2)

    print(f"  Sheet number: {result['sheet_number']}")
    print(f"  Sheet title: {result['sheet_title']}")
    print(f"  Title method: {result['title_method']}")
    print(f"  Confidence: {result['title_confidence']}")

    assert result['sheet_number'] == 'A2.0', f"Expected 'A2.0', got {result['sheet_number']}"
    assert result['sheet_title'] == 'Floor Plan', f"Expected 'Floor Plan', got {result['sheet_title']}"
    assert result['title_method'] == 'drawing_index', f"Expected 'drawing_index', got {result['title_method']}"
    assert result['title_confidence'] >= 0.95, f"Expected >= 0.95, got {result['title_confidence']}"

    print("PASSED\n")


def test_extractor_fallback_to_pattern():
    """Test that extractor falls back to pattern when not in index."""
    print("=== Test: Extractor Fallback to Pattern ===")

    extractor = Extractor()

    # Set up a drawing index that doesn't have A3.0
    extractor.set_drawing_index({
        'A1.0': 'Code Review',
        'A2.0': 'Floor Plan',
    })

    # Test extraction with a sheet number NOT in the index
    text = """
    SHEET NUMBER: A3.0
    PROJECT NO: P27142
    TITLE: ELEVATIONS
    """

    result = extractor.extract_fields(text, page_number=3)

    print(f"  Sheet number: {result['sheet_number']}")
    print(f"  Sheet title: {result['sheet_title']}")
    print(f"  Title method: {result['title_method']}")
    print(f"  Confidence: {result['title_confidence']}")

    assert result['sheet_number'] == 'A3.0', f"Expected 'A3.0', got {result['sheet_number']}"
    # Should fall back to pattern matching since A3.0 not in index
    assert result['title_method'] == 'pattern', f"Expected 'pattern', got {result['title_method']}"

    print("PASSED\n")


def test_title_validation():
    """Test that title validation rejects garbage."""
    print("=== Test: Title Validation ===")

    extractor = Extractor()

    # Garbage titles should be rejected
    garbage_texts = [
        "TITLE: PROJECT NO. PRELIMINARY",
        "TITLE: 312-555-1234",
        "TITLE: Chicago, IL 60601",
        "TITLE: ABC Construction LLC",
    ]

    for text in garbage_texts:
        text_full = f"SHEET NUMBER: A1.0\n{text}"
        result = extractor.extract_fields(text_full, page_number=2)
        print(f"  '{text}' -> title={result['sheet_title']}")
        # Garbage should be rejected (title should be None or not match the garbage)
        if result['sheet_title']:
            assert 'PROJECT NO' not in result['sheet_title'].upper()
            assert '312-555' not in result['sheet_title']
            assert 'LLC' not in result['sheet_title'].upper()

    print("PASSED\n")


def test_cover_sheet_detection():
    """Test cover sheet detection."""
    print("=== Test: Cover Sheet Detection ===")

    extractor = Extractor()

    text = """
    COVER SHEET
    PROJECT NO: P27142
    DRAWING INDEX:
    A1.0 FLOOR PLAN
    """

    result = extractor.extract_fields(text, page_number=1)

    print(f"  Is cover sheet: {result.get('is_cover_sheet', False)}")
    print(f"  Sheet title: {result['sheet_title']}")

    assert result.get('is_cover_sheet') == True, "Expected cover sheet detection"
    assert result['sheet_title'] == 'COVER SHEET', f"Expected 'COVER SHEET', got {result['sheet_title']}"

    print("PASSED\n")


def test_with_sample_pdf():
    """Integration test with actual sample.pdf."""
    sample_path = Path(__file__).parent.parent / 'test_data' / 'sample.pdf'

    print("=== Test: Full Integration with sample.pdf ===")

    if not sample_path.exists():
        print(f"  Skipping - {sample_path} not found")
        return

    ocr_engine = OCREngine()
    extractor = Extractor(ocr_engine=ocr_engine)

    with PDFHandler(sample_path) as pdf:
        print(f"  PDF has {pdf.page_count} pages")

        # Parse drawing index first (ONCE per PDF)
        index = extractor.parse_drawing_index(pdf)
        print(f"  Drawing index has {len(index)} entries")

        # Process each page
        for page_num in range(pdf.page_count):
            # Get page text (embedded or OCR)
            text = pdf.get_page_text(page_num)
            if len(text) < 50:
                # Need OCR
                img = pdf.get_page_image(page_num, dpi=200)
                text = ocr_engine.ocr_image(img)

            result = extractor.extract_fields(text, page_number=page_num + 1)

            print(f"\n  Page {page_num + 1}:")
            print(f"    Sheet: {result['sheet_number']}")
            print(f"    Title: {result['sheet_title']}")
            print(f"    Method: {result['title_method']}")
            print(f"    Confidence: {result['title_confidence']:.2f}")

    print("\nPASSED\n")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("INTEGRATION TESTS: Blueprint Processor V4.2.1")
    print("=" * 60 + "\n")

    try:
        test_extractor_with_drawing_index()
        test_extractor_fallback_to_pattern()
        test_title_validation()
        test_cover_sheet_detection()
        test_with_sample_pdf()

        print("=" * 60)
        print("=== ALL INTEGRATION TESTS PASSED ===")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n!!! TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n!!! ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
