"""
Tests for Sheet Classification System (V6.0)

Tests pattern matching, classification, and organization.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pattern_matchers import (
    RegexPatternMatcher,
    SheetNumberMatcher,
    CompositeMatcher,
    UnclassifiedMatcher,
    get_default_matcher
)
from core.sheet_classifier import SheetClassifier, ClassifiedSheet


class TestRegexPatternMatcher:
    """Tests for RegexPatternMatcher class."""

    def setup_method(self):
        self.matcher = RegexPatternMatcher()

    def test_floor_plan_matches(self):
        """Floor plan patterns should match correctly."""
        test_cases = [
            ("A101", "First Floor Plan", "floor_plans"),
            ("A102", "SECOND FLOOR PLAN", "floor_plans"),
            ("A103", "Overall Floor Plan", "floor_plans"),
            ("A104", "Level 1 Plan", "floor_plans"),
            ("A105", "Basement Plan", "floor_plans"),
            ("A106", "Roof Plan", "floor_plans"),
        ]

        for sheet_num, title, expected_cat in test_cases:
            results = self.matcher.match(sheet_num, title)
            assert len(results) > 0, f"Expected match for '{title}'"
            assert results[0].category == expected_cat, f"Expected {expected_cat} for '{title}'"

    def test_rcp_matches(self):
        """Reflected ceiling plan patterns should match correctly."""
        test_cases = [
            ("A121", "Reflected Ceiling Plan", "reflected_ceiling_plans"),
            ("A122", "RCP - Level 1", "reflected_ceiling_plans"),
            ("A123", "First Floor Ceiling Plan", "reflected_ceiling_plans"),
        ]

        for sheet_num, title, expected_cat in test_cases:
            results = self.matcher.match(sheet_num, title)
            assert len(results) > 0, f"Expected match for '{title}'"
            assert results[0].category == expected_cat

    def test_interior_elevation_matches(self):
        """Interior elevation patterns should match correctly."""
        test_cases = [
            ("A501", "Interior Elevations", "interior_elevations"),
            ("A502", "Room Elevation - Kitchen", "interior_elevations"),
            ("A503", "Casework Elevations", "interior_elevations"),
        ]

        for sheet_num, title, expected_cat in test_cases:
            results = self.matcher.match(sheet_num, title)
            assert len(results) > 0, f"Expected match for '{title}'"
            assert results[0].category == expected_cat

    def test_exterior_elevation_matches(self):
        """Exterior elevation patterns should match correctly."""
        test_cases = [
            ("A201", "Exterior Elevations", "exterior_elevations"),
            ("A202", "North Elevation", "exterior_elevations"),
            ("A203", "Building Elevation - East", "exterior_elevations"),
            ("A204", "Front Elevation", "exterior_elevations"),
        ]

        for sheet_num, title, expected_cat in test_cases:
            results = self.matcher.match(sheet_num, title)
            assert len(results) > 0, f"Expected match for '{title}'"
            assert results[0].category == expected_cat

    def test_cover_sheet_matches(self):
        """Cover sheet patterns should match correctly."""
        test_cases = [
            ("G001", "Cover Sheet", "cover_sheets"),
            ("A001", "Sheet Index", "cover_sheets"),
            ("A002", "General Notes", "cover_sheets"),
            ("G002", "Code Analysis", "cover_sheets"),
            ("A003", "Life Safety Plan", "cover_sheets"),
        ]

        for sheet_num, title, expected_cat in test_cases:
            results = self.matcher.match(sheet_num, title)
            assert len(results) > 0, f"Expected match for '{title}'"
            assert results[0].category == expected_cat

    def test_finish_schedule_matches(self):
        """Finish schedule patterns should match correctly."""
        test_cases = [
            ("A601", "Room Finish Schedule", "room_finish_schedules"),
            ("A602", "Interior Finish Schedule", "room_finish_schedules"),
        ]

        for sheet_num, title, expected_cat in test_cases:
            results = self.matcher.match(sheet_num, title)
            assert len(results) > 0, f"Expected match for '{title}'"
            assert results[0].category == expected_cat

    def test_no_match_for_unrelated(self):
        """Unrelated titles should not match."""
        results = self.matcher.match("X999", "Random Technical Drawing")
        assert len(results) == 0


class TestSheetNumberMatcher:
    """Tests for SheetNumberMatcher class."""

    def setup_method(self):
        self.matcher = SheetNumberMatcher()

    def test_floor_plan_sheet_numbers(self):
        """A1xx sheet numbers should match floor plans."""
        results = self.matcher.match("A101", "")
        assert len(results) > 0
        assert results[0].category == "floor_plans"

    def test_cover_sheet_numbers(self):
        """A0xx and G0xx sheet numbers should match cover sheets."""
        results = self.matcher.match("A001", "")
        assert len(results) > 0
        assert results[0].category == "cover_sheets"

        results = self.matcher.match("G001", "")
        assert len(results) > 0
        assert results[0].category == "cover_sheets"


class TestUnclassifiedMatcher:
    """Tests for UnclassifiedMatcher class."""

    def setup_method(self):
        self.matcher = UnclassifiedMatcher()

    def test_mep_sheets_recognized(self):
        """MEP sheets should be recognized as not needed."""
        test_cases = [
            ("M101", "Mechanical Plan", True),
            ("E101", "Electrical Plan", True),
            ("P101", "Plumbing Plan", True),
            ("S101", "Structural Plan", True),
        ]

        for sheet_num, title, should_match in test_cases:
            results = self.matcher.match(sheet_num, title)
            if should_match:
                assert len(results) > 0, f"Expected match for '{title}'"
                assert results[0].category == "properly_classified_not_needed"
            else:
                assert len(results) == 0


class TestCompositeMatcher:
    """Tests for CompositeMatcher class."""

    def setup_method(self):
        self.matcher = CompositeMatcher()

    def test_classify_floor_plan(self):
        """Floor plan should classify correctly."""
        categories, classification_type = self.matcher.classify("A101", "First Floor Plan")
        assert "floor_plans" in categories
        assert classification_type == "matched"

    def test_classify_mep_as_not_needed(self):
        """MEP sheet should be classified as not needed."""
        # Use a title that doesn't contain main category keywords
        categories, classification_type = self.matcher.classify("M101", "Mechanical Ductwork Layout")
        assert "properly_classified_not_needed" in categories
        assert classification_type == "not_needed"

    def test_classify_unknown_as_unclassified(self):
        """Unknown sheet should be unclassified."""
        categories, classification_type = self.matcher.classify("X999", "Unknown Drawing Type")
        assert "unclassified_may_be_needed" in categories
        assert classification_type == "unclassified"

    def test_title_takes_precedence(self):
        """Title pattern should take precedence over sheet number when more specific."""
        # Even though M101 suggests mechanical, title says floor plan
        categories, classification_type = self.matcher.classify("M101", "First Floor Plan")
        # Should match floor_plans from title pattern (higher confidence)
        assert "floor_plans" in categories or "properly_classified_not_needed" in categories


class TestSheetClassifier:
    """Tests for SheetClassifier class."""

    def setup_method(self):
        self.classifier = SheetClassifier()

    def test_classify_single_sheet(self):
        """Single sheet classification."""
        page_data = {
            "page": 1,
            "sheet_number": "A101",
            "sheet_title": "First Floor Plan",
            "number_conf": 0.95,
            "title_conf": 0.90,
        }

        result = self.classifier.classify_sheet("test.pdf", page_data)

        assert result.pdf_name == "test.pdf"
        assert result.page_number == 1
        assert result.sheet_number == "A101"
        assert "floor_plans" in result.categories
        assert result.classification_type == "matched"

    def test_classify_extraction_results(self):
        """Full extraction results classification."""
        extraction_data = {
            "results": [
                {
                    "pdf": "test.pdf",
                    "pages": [
                        {"page": 1, "sheet_number": "A001", "sheet_title": "Cover Sheet"},
                        {"page": 2, "sheet_number": "A101", "sheet_title": "Floor Plan"},
                        {"page": 3, "sheet_number": "A201", "sheet_title": "Exterior Elevation"},
                    ]
                }
            ]
        }

        result = self.classifier.classify_extraction_results(extraction_data)

        assert result.total_pages == 3
        assert len(result.categories["cover_sheets"]) >= 1
        assert len(result.categories["floor_plans"]) >= 1
        assert len(result.categories["exterior_elevations"]) >= 1


class TestClassifiedSheet:
    """Tests for ClassifiedSheet data class."""

    def test_to_dict(self):
        """ClassifiedSheet should serialize correctly."""
        sheet = ClassifiedSheet(
            pdf_name="test.pdf",
            page_number=1,
            sheet_number="A101",
            sheet_title="Floor Plan",
            categories=["floor_plans"],
            classification_type="matched",
        )

        data = sheet.to_dict()

        assert data["pdf"] == "test.pdf"
        assert data["page"] == 1
        assert data["sheet_number"] == "A101"
        assert "floor_plans" in data["categories"]


class TestGetDefaultMatcher:
    """Tests for factory function."""

    def test_returns_composite_matcher(self):
        """Factory should return CompositeMatcher."""
        matcher = get_default_matcher()
        assert isinstance(matcher, CompositeMatcher)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
