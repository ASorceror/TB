"""
Blueprint Processor V4.7.1 - Unit Tests for extract_sheet_number()
Tests the sheet number extraction strategies with mock data.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extractor import Extractor


def create_mock_page(width=2592, height=1728):
    """Create a mock page object with rect dimensions."""
    mock_page = MagicMock()
    mock_page.rect.width = width
    mock_page.rect.height = height
    return mock_page


def create_text_block(text, x0, y0, x1, y1):
    """Create a text block dict in the expected format."""
    return {
        'text': text,
        'bbox': (x0, y0, x1, y1),
        'block_no': 0,
    }


class TestStrategy1LabeledPatterns:
    """Test Strategy 1: Labeled patterns (SHEET NO: X.X)"""

    def test_sheet_no_colon(self):
        """Test 'SHEET NO: A1.1' pattern."""
        extractor = Extractor()
        text = "Some text\nSHEET NO: A1.1\nMore text"
        result = extractor.extract_sheet_number(text)
        assert result == "A1.1", f"Expected A1.1, got {result}"
        print("  test_sheet_no_colon: PASSED")

    def test_sheet_number_colon(self):
        """Test 'SHEET NUMBER: M2.1' pattern."""
        extractor = Extractor()
        text = "Header\nSHEET NUMBER: M2.1\nFooter"
        result = extractor.extract_sheet_number(text)
        assert result == "M2.1", f"Expected M2.1, got {result}"
        print("  test_sheet_number_colon: PASSED")

    def test_dwg_no(self):
        """Test 'DWG NO: E1.1' pattern."""
        extractor = Extractor()
        text = "DWG NO: E1.1"
        result = extractor.extract_sheet_number(text)
        assert result == "E1.1", f"Expected E1.1, got {result}"
        print("  test_dwg_no: PASSED")

    def test_sheet_with_dash(self):
        """Test sheet number with dash like 'A-1.1'."""
        extractor = Extractor()
        text = "SHEET NO: A-1.1"
        result = extractor.extract_sheet_number(text)
        assert result == "A-1.1", f"Expected A-1.1, got {result}"
        print("  test_sheet_with_dash: PASSED")


class TestStrategy4SpatialProximity:
    """Test Strategy 4: Spatial proximity matching with bbox coordinates."""

    def test_label_and_value_nearby(self):
        """Test label and value that are spatially close."""
        extractor = Extractor()
        mock_page = create_mock_page(2592, 1728)

        # SHEET NO label and G.001 value nearby
        text_blocks = [
            create_text_block("SHEET NO:", 2400, 1600, 2500, 1620),
            create_text_block("G.001", 2400, 1630, 2450, 1650),
        ]

        text = "SHEET NO:\nG.001"
        result = extractor.extract_sheet_number(text, None, text_blocks, mock_page)
        assert result == "G.001", f"Expected G.001, got {result}"
        print("  test_label_and_value_nearby: PASSED")

    def test_label_and_value_too_far(self):
        """Test that values too far from label are not matched."""
        extractor = Extractor()
        mock_page = create_mock_page(2592, 1728)

        # SHEET NO label and value too far apart (more than 7% of page)
        text_blocks = [
            create_text_block("SHEET NO:", 100, 100, 200, 120),
            create_text_block("A1.1", 2400, 1600, 2450, 1620),  # Far away
        ]

        # No labeled pattern in text, so should fall through to Strategy 5
        text = "Random text\nA1.1"
        result = extractor.extract_sheet_number(text, None, text_blocks, mock_page)
        # Should find via Strategy 5 (isolated line at end)
        assert result == "A1.1", f"Expected A1.1 via Strategy 5, got {result}"
        print("  test_label_and_value_too_far: PASSED")

    def test_multiple_values_picks_closest(self):
        """Test that the closest value to label is picked."""
        extractor = Extractor()
        mock_page = create_mock_page(2592, 1728)

        text_blocks = [
            create_text_block("SHEET NUMBER:", 2400, 1600, 2520, 1620),
            create_text_block("A1.1", 2400, 1700, 2450, 1720),  # 100 pixels away
            create_text_block("B2.2", 2400, 1625, 2450, 1645),  # 25 pixels away (closer)
        ]

        text = "SHEET NUMBER:\nB2.2\nA1.1"
        result = extractor.extract_sheet_number(text, None, text_blocks, mock_page)
        assert result == "B2.2", f"Expected B2.2 (closest), got {result}"
        print("  test_multiple_values_picks_closest: PASSED")


class TestStrategy5IsolatedLine:
    """Test Strategy 5: Isolated line at end of text."""

    def test_sheet_on_own_line(self):
        """Test sheet number on its own line near end."""
        extractor = Extractor()
        text = "\n".join([
            "Lots of text here",
            "More content",
            "Address info",
            "Company name",
            "P1.1",  # Sheet number on its own line
            "Some footer",
        ])
        result = extractor.extract_sheet_number(text)
        assert result == "P1.1", f"Expected P1.1, got {result}"
        print("  test_sheet_on_own_line: PASSED")

    def test_sheet_with_prefix(self):
        """Test various sheet number formats."""
        extractor = Extractor()

        test_cases = [
            ("Line1\nA1.1\nFooter", "A1.1"),
            ("Line1\nM2.1\nFooter", "M2.1"),
            ("Line1\nE1.1\nFooter", "E1.1"),
            ("Line1\nG.001\nFooter", "G.001"),
            ("Line1\nUL.001\nFooter", "UL.001"),
        ]

        for text, expected in test_cases:
            result = extractor.extract_sheet_number(text)
            assert result == expected, f"Expected {expected}, got {result}"

        print("  test_sheet_with_prefix: PASSED")


class TestEdgeCases:
    """Test edge cases and potential false positives."""

    def test_ansi_code_not_matched(self):
        """Test that ANSI codes like Z97.1 in context are not matched."""
        extractor = Extractor()
        # Z97.1 appears as part of "ANSI Z97.1" - should not be matched
        text = "Comply with ANSI Z97.1 requirements\nA002"
        result = extractor.extract_sheet_number(text)
        assert result == "A002", f"Expected A002, got {result}"
        print("  test_ansi_code_not_matched: PASSED")

    def test_fixture_sheet_not_matched(self):
        """Test that 'FIXTURE CUT SHEET' doesn't trigger false positive."""
        extractor = Extractor()
        mock_page = create_mock_page(2592, 1728)

        # Block with "FIXTURE CUT SHEET ONSITE" - should NOT trigger Strategy 4
        text_blocks = [
            create_text_block("FIXTURE CUT SHEET ONSITE FOR FINAL INSPECTION", 100, 500, 500, 520),
            create_text_block("MODEL LFMMV-QC-M1, 1", 100, 530, 250, 550),
            create_text_block("P1.1", 2400, 1600, 2450, 1620),  # Actual sheet number
        ]

        # The text should find P1.1 via Strategy 5, not M1 via false Strategy 4 match
        text = "FIXTURE CUT SHEET ONSITE FOR FINAL INSPECTION\nMODEL LFMMV-QC-M1, 1\nP1.1"
        result = extractor.extract_sheet_number(text, None, text_blocks, mock_page)
        assert result == "P1.1", f"Expected P1.1, got {result}"
        print("  test_fixture_sheet_not_matched: PASSED")

    def test_no_page_object_uses_estimation(self):
        """Test that extraction works without page object (uses estimation)."""
        extractor = Extractor()

        text_blocks = [
            create_text_block("SHEET NO:", 2400, 1600, 2500, 1620),
            create_text_block("A1.1", 2400, 1630, 2450, 1650),
        ]

        text = "SHEET NO:\nA1.1"
        # Pass None for page - should still work via estimation or other strategies
        result = extractor.extract_sheet_number(text, None, text_blocks, None)
        assert result == "A1.1", f"Expected A1.1, got {result}"
        print("  test_no_page_object_uses_estimation: PASSED")

    def test_empty_text(self):
        """Test handling of empty text."""
        extractor = Extractor()
        result = extractor.extract_sheet_number("")
        assert result is None, f"Expected None for empty text, got {result}"
        print("  test_empty_text: PASSED")

    def test_no_sheet_number(self):
        """Test text with no valid sheet number."""
        extractor = Extractor()
        text = "This is just regular text with no sheet numbers"
        result = extractor.extract_sheet_number(text)
        assert result is None, f"Expected None, got {result}"
        print("  test_no_sheet_number: PASSED")


def run_all_tests():
    """Run all test classes."""
    print("=" * 60)
    print("Unit Tests for extract_sheet_number() - V4.7.1")
    print("=" * 60)

    test_classes = [
        ("Strategy 1: Labeled Patterns", TestStrategy1LabeledPatterns),
        ("Strategy 4: Spatial Proximity", TestStrategy4SpatialProximity),
        ("Strategy 5: Isolated Line", TestStrategy5IsolatedLine),
        ("Edge Cases", TestEdgeCases),
    ]

    total_passed = 0
    total_failed = 0

    for name, test_class in test_classes:
        print(f"\n=== {name} ===")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    total_passed += 1
                except AssertionError as e:
                    print(f"  {method_name}: FAILED - {e}")
                    total_failed += 1
                except Exception as e:
                    print(f"  {method_name}: ERROR - {e}")
                    total_failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
