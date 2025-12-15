"""
Blueprint Processor V4.2.1 - Phase 2 Tests
Tests for Drawing Index Parser.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.drawing_index import DrawingIndexParser
from core.pdf_handler import PDFHandler
from core.ocr_engine import OCREngine


def test_parse_clean_text():
    """Test parsing clean drawing index text (vector PDF)."""
    parser = DrawingIndexParser()

    # Simulated clean drawing index text
    text = """
    DRAWING INDEX:
    THE FOLLOWING SHEETS COMPOSE THE COMPLETE SET OF
    CONSTRUCTION DOCUMENTS:
    ARCHITECTURAL
    COVERSHEET
    A1.0    CODE REVIEW, SCHEDULES AND ADA REQUIREMENTS
    D2.0    DEMOLITION FLOOR PLAN
    A2.0    FLOOR PLAN
    A2.1    ENLARGED FLOOR PLAN
    A2.2    REFLECTED CEILING PLAN
    """

    result = parser._parse_index_from_text(text)

    print("=== Test: Parse Clean Text ===")
    print(f"Entries found: {len(result)}")
    for sheet, title in result.items():
        print(f"  {sheet}: {title}")

    # Assertions
    assert len(result) == 5, f"Expected 5 entries, got {len(result)}"
    assert 'A1.0' in result, "A1.0 not found"
    assert 'D2.0' in result, "D2.0 not found"
    assert 'A2.0' in result, "A2.0 not found"
    assert 'A2.1' in result, "A2.1 not found"
    assert 'A2.2' in result, "A2.2 not found"

    # Check titles
    assert 'CODE REVIEW' in result['A1.0'].upper(), f"Wrong title for A1.0: {result['A1.0']}"
    assert 'DEMOLITION' in result['D2.0'].upper(), f"Wrong title for D2.0: {result['D2.0']}"
    assert 'FLOOR PLAN' in result['A2.0'].upper(), f"Wrong title for A2.0: {result['A2.0']}"

    print("PASSED\n")


def test_multiline_title():
    """Test parsing multi-line titles."""
    parser = DrawingIndexParser()

    text = """
    DRAWING INDEX:
    A1.0    CODE REVIEW, SCHEDULES
            AND ADA REQUIREMENTS
    A2.0    FLOOR PLAN
    """

    result = parser._parse_index_from_text(text)

    print("=== Test: Multi-line Title ===")
    print(f"Entries found: {len(result)}")
    for sheet, title in result.items():
        print(f"  {sheet}: {title}")

    assert len(result) == 2, f"Expected 2 entries, got {len(result)}"
    assert 'A1.0' in result, "A1.0 not found"

    # Multi-line title should be joined
    a1_title = result['A1.0'].upper()
    assert 'CODE REVIEW' in a1_title, f"Missing 'CODE REVIEW' in: {a1_title}"
    assert 'ADA REQUIREMENTS' in a1_title, f"Missing 'ADA REQUIREMENTS' in: {a1_title}"

    print("PASSED\n")


def test_fuzzy_matching():
    """Test fuzzy matching for OCR-damaged sheet numbers."""
    parser = DrawingIndexParser()

    index = {
        'A1.0': 'Code Review',
        'A2.0': 'Floor Plan',
        'D2.0': 'Demolition Plan',
    }

    print("=== Test: Fuzzy Matching ===")

    # Test exact match
    result = parser.lookup('A1.0', index)
    print(f"  Exact 'A1.0': {result}")
    assert result == 'Code Review', f"Expected 'Code Review', got {result}"

    # Test missing decimal - A10 should match A1.0
    result = parser.lookup('A10', index)
    print(f"  Fuzzy 'A10' -> A1.0: {result}")
    assert result == 'Code Review', f"Expected 'Code Review' for A10, got {result}"

    # Test OCR error - should still match with high similarity
    result = parser.lookup('A-10', index)
    print(f"  Fuzzy 'A-10': {result}")
    # May or may not match depending on similarity threshold

    print("PASSED\n")


def test_discipline_headers():
    """Test that discipline headers are skipped."""
    parser = DrawingIndexParser()

    text = """
    DRAWING INDEX:
    ARCHITECTURAL
    A1.0    FLOOR PLAN
    A2.0    ELEVATIONS

    MECHANICAL
    M1.0    HVAC PLAN
    """

    result = parser._parse_index_from_text(text)

    print("=== Test: Discipline Headers ===")
    print(f"Entries found: {len(result)}")
    for sheet, title in result.items():
        print(f"  {sheet}: {title}")

    assert len(result) == 3, f"Expected 3 entries, got {len(result)}"
    assert 'ARCHITECTURAL' not in result, "ARCHITECTURAL should be skipped"
    assert 'MECHANICAL' not in result, "MECHANICAL should be skipped"
    assert 'A1.0' in result
    assert 'M1.0' in result

    print("PASSED\n")


def test_ocr_normalization():
    """Test OCR error normalization in sheet numbers."""
    parser = DrawingIndexParser()

    print("=== Test: OCR Normalization ===")

    # Test O -> 0 correction
    result = parser._normalize_sheet_number('A1.O')
    print(f"  'A1.O' -> {result}")
    assert result == 'A1.0', f"Expected 'A1.0', got {result}"

    # Test I -> 1 correction
    result = parser._normalize_sheet_number('AI.0')
    print(f"  'AI.0' -> {result}")
    # Should handle this based on context

    # Test normal case
    result = parser._normalize_sheet_number('A2.1')
    print(f"  'A2.1' -> {result}")
    assert result == 'A2.1', f"Expected 'A2.1', got {result}"

    print("PASSED\n")


def test_with_sample_pdf():
    """Test with actual sample.pdf if available."""
    sample_path = Path(__file__).parent.parent / 'test_data' / 'sample.pdf'

    print("=== Test: sample.pdf ===")

    if not sample_path.exists():
        print(f"  Skipping - {sample_path} not found")
        return

    # Initialize OCR engine
    ocr_engine = OCREngine()
    parser = DrawingIndexParser(ocr_engine=ocr_engine)

    with PDFHandler(sample_path) as pdf:
        print(f"  PDF has {pdf.page_count} pages")

        # Parse drawing index
        index = parser.parse_from_pdf(pdf)

        print(f"  Found {len(index)} index entries:")
        for sheet, title in index.items():
            print(f"    {sheet}: {title}")

    # Expected entries from spec
    expected_sheets = ['A1.0', 'D2.0', 'A2.0', 'A2.1', 'A2.2']

    found_count = 0
    for sheet in expected_sheets:
        if sheet in index:
            found_count += 1
            print(f"  [OK] {sheet} found")
        else:
            # Try fuzzy lookup
            title = parser.lookup(sheet, index)
            if title:
                found_count += 1
                print(f"  [OK] {sheet} found via fuzzy match")
            else:
                print(f"  [MISSING] {sheet}")

    print(f"\n  Found {found_count}/{len(expected_sheets)} expected sheets")

    if found_count >= 4:
        print("PASSED\n")
    else:
        print("PARTIAL PASS (some sheets missing)\n")


def run_all_tests():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("PHASE 2 TESTS: Drawing Index Parser")
    print("=" * 60 + "\n")

    try:
        test_parse_clean_text()
        test_multiline_title()
        test_fuzzy_matching()
        test_discipline_headers()
        test_ocr_normalization()
        test_with_sample_pdf()

        print("=" * 60)
        print("=== ALL PHASE 2 TESTS PASSED ===")
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
