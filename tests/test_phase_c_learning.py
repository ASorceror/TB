"""
Phase C Validation Tests - Template Learning
Run with: python -m pytest tests/test_phase_c_learning.py -v

This module tests the TemplateLearner for learning templates from
quality extractions.
"""
import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.template_learner import TemplateLearner
from core.template_types import Template


class TestTemplateLearner:

    @pytest.fixture
    def learner(self):
        return TemplateLearner()

    def _make_extraction(self, title, needs_review=0, method='vector'):
        """Helper to create mock extraction dict."""
        return {
            "pdf_filename": "test.pdf",
            "page_number": 1,
            "sheet_title": title,
            "needs_review": needs_review,
            "extraction_method": method,
            "title_confidence": 0.85
        }

    def test_learns_from_sufficient_quality(self, learner):
        """Should create template when 10+ quality pages available."""
        extractions = [
            self._make_extraction("Floor Plan Level 1"),
            self._make_extraction("Floor Plan Level 2"),
            self._make_extraction("Reflected Ceiling Plan"),
            self._make_extraction("Electrical Floor Plan"),
            self._make_extraction("Plumbing Floor Plan"),
            self._make_extraction("HVAC Floor Plan"),
            self._make_extraction("North Elevation"),
            self._make_extraction("South Elevation"),
            self._make_extraction("Building Section A"),
            self._make_extraction("Building Section B"),
        ]

        template = learner.learn("testhash123", "test.pdf", extractions)

        assert template is not None
        assert isinstance(template, Template)
        assert template.pdf_hash == "testhash123"
        assert template.quality_pages_used >= 5
        assert template.confidence >= 0.70

    def test_rejects_insufficient_quality(self, learner):
        """Should return None when fewer than 5 quality pages."""
        extractions = [
            self._make_extraction("Floor Plan"),
            self._make_extraction("Section"),           # Will be rejected (single keyword)
            self._make_extraction("1425"),              # Will be rejected (number)
            self._make_extraction("Project Number"),    # Will be rejected (garbage)
        ]

        template = learner.learn("testhash123", "test.pdf", extractions)

        assert template is None

    def test_exclusion_patterns_from_failures(self, learner):
        """Should add titles from failed pages to exclusion patterns."""
        extractions = [
            # Quality pages
            self._make_extraction("Floor Plan Level 1"),
            self._make_extraction("Floor Plan Level 2"),
            self._make_extraction("Reflected Ceiling Plan"),
            self._make_extraction("North Elevation"),
            self._make_extraction("South Elevation"),
            # Failed pages with repeated bad title
            self._make_extraction("Project Number", needs_review=1),
            self._make_extraction("Project Number", needs_review=1),
            self._make_extraction("Project Number", needs_review=1),
            self._make_extraction("Bad Title", needs_review=1),
        ]

        template = learner.learn("testhash123", "test.pdf", extractions)

        assert template is not None
        # "Project Number" appeared 3x in failures, should be excluded
        exclusion_text = ' '.join(template.exclusion_patterns)
        assert "Project Number" in exclusion_text or "PROJECT" in exclusion_text.upper()

    def test_confidence_calculation_high(self, learner):
        """Confidence should be 0.90 for 60%+ quality rate."""
        # 60% quality rate should give confidence 0.90
        extractions = []
        for i in range(6):
            extractions.append(self._make_extraction(f"Floor Plan {i}"))
        for i in range(4):
            extractions.append(self._make_extraction("Bad", needs_review=1))

        template = learner.learn("testhash123", "test.pdf", extractions)

        assert template is not None
        assert template.confidence == 0.90  # 60% >= 50%

    def test_confidence_calculation_medium(self, learner):
        """Confidence should be 0.80 for 30-50% quality rate."""
        # 40% quality rate should give confidence 0.80
        extractions = []
        for i in range(5):  # 5 quality = 5/12 = ~42%
            extractions.append(self._make_extraction(f"Floor Plan {i}"))
        for i in range(7):
            extractions.append(self._make_extraction("Bad", needs_review=1))

        template = learner.learn("testhash123", "test.pdf", extractions)

        assert template is not None
        assert template.confidence == 0.80  # 42% >= 30%

    def test_length_statistics(self, learner):
        """Should calculate min/max title lengths."""
        extractions = [
            self._make_extraction("Floor Plan"),            # 10 chars
            self._make_extraction("First Floor Demolition Plan"),  # 28 chars
            self._make_extraction("Reflected Ceiling Plan Level 2"), # 31 chars
            self._make_extraction("North Elevation"),       # 15 chars
            self._make_extraction("Site Plan"),             # 9 chars
        ]

        template = learner.learn("testhash123", "test.pdf", extractions)

        assert template is not None
        assert template.typical_length_min <= 10
        assert template.typical_length_max >= 28

    def test_page_type_counts(self, learner):
        """Should count vector vs scanned pages."""
        extractions = [
            self._make_extraction("Floor Plan 1", method='vector'),
            self._make_extraction("Floor Plan 2", method='vector'),
            self._make_extraction("Floor Plan 3", method='vector'),
            self._make_extraction("Floor Plan 4", method='ocr'),
            self._make_extraction("Floor Plan 5", method='ocr'),
        ]

        template = learner.learn("testhash123", "test.pdf", extractions)

        assert template is not None
        assert template.vector_pages == 3
        assert template.scanned_pages == 2

    def test_empty_extractions(self, learner):
        """Should return None for empty extractions."""
        template = learner.learn("testhash123", "test.pdf", [])
        assert template is None

    def test_pages_analyzed_count(self, learner):
        """Should correctly count total pages analyzed."""
        extractions = [
            self._make_extraction("Floor Plan 1"),
            self._make_extraction("Floor Plan 2"),
            self._make_extraction("Floor Plan 3"),
            self._make_extraction("Floor Plan 4"),
            self._make_extraction("Floor Plan 5"),
            self._make_extraction("Bad", needs_review=1),
            self._make_extraction("Bad", needs_review=1),
        ]

        template = learner.learn("testhash123", "test.pdf", extractions)

        assert template is not None
        assert template.pages_analyzed == 7  # All pages, including bad ones

    def test_standard_exclusion_patterns_included(self, learner):
        """Standard exclusion patterns should be included."""
        extractions = [
            self._make_extraction("Floor Plan 1"),
            self._make_extraction("Floor Plan 2"),
            self._make_extraction("Floor Plan 3"),
            self._make_extraction("Floor Plan 4"),
            self._make_extraction("Floor Plan 5"),
        ]

        template = learner.learn("testhash123", "test.pdf", extractions)

        assert template is not None
        # Check for some standard patterns
        patterns_text = ' '.join(template.exclusion_patterns)
        assert 'PROJECT' in patterns_text.upper() or 'SHEET' in patterns_text.upper()

    def test_template_has_bbox_defaults(self, learner):
        """Template should have default bbox values."""
        extractions = [
            self._make_extraction("Floor Plan 1"),
            self._make_extraction("Floor Plan 2"),
            self._make_extraction("Floor Plan 3"),
            self._make_extraction("Floor Plan 4"),
            self._make_extraction("Floor Plan 5"),
        ]

        template = learner.learn("testhash123", "test.pdf", extractions)

        assert template is not None
        # Check default title block bbox
        assert template.title_block_bbox == [0.65, 0.75, 1.0, 1.0]
        assert template.title_zone_bbox == [0.0, 0.10, 1.0, 0.50]
