"""
Phase D Validation Tests - Template Application
Run with: python -m pytest tests/test_phase_d_application.py -v

This module tests the TemplateApplier for applying learned templates
to rescue failed extractions.
"""
import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.template_applier import TemplateApplier
from core.template_types import Template


class TestTemplateApplier:

    @pytest.fixture
    def applier(self):
        return TemplateApplier()

    @pytest.fixture
    def sample_template(self):
        return Template(
            template_id="test-uuid",
            pdf_hash="abc123",
            source_pdf="test.pdf",
            created_at="2024-12-14T12:00:00Z",
            pages_analyzed=100,
            quality_pages_used=50,
            confidence=0.85,
            title_block_bbox=[0.65, 0.75, 1.0, 1.0],
            title_zone_bbox=[0.0, 0.10, 1.0, 0.50],
            exclusion_patterns=[
                r"PROJECT",
                r"SHEET",
                r"SCALE",
                r"DATE",
                r"Gmp/permit Set",
            ],
            typical_length_min=10,
            typical_length_max=40,
            vector_pages=80,
            scanned_pages=20
        )

    def test_extracts_valid_title_from_text(self, applier, sample_template):
        """Should extract valid title from page text."""
        page_text = """
        PROJECT: 12345
        SHEET: A1.0
        Floor Plan - Level 1
        Scale: 1/4" = 1'-0"
        DATE: 01/15/2024
        """

        result = applier.apply(sample_template, page_text)

        assert result['title'] == "Floor Plan - Level 1"
        assert result['confidence'] >= 0.70
        assert result['method'] == "template"

    def test_rejects_text_with_only_garbage(self, applier, sample_template):
        """Should return None when only garbage text available."""
        page_text = """
        PROJECT NUMBER
        SHEET NUMBER
        DATE: 01/01/2025
        SCALE: AS NOTED
        """

        result = applier.apply(sample_template, page_text)

        assert result['title'] is None
        assert result['confidence'] == 0.0

    def test_applies_exclusion_patterns(self, applier, sample_template):
        """Should not extract text matching exclusion patterns."""
        page_text = """
        Gmp/permit Set
        Floor Plan
        """

        result = applier.apply(sample_template, page_text)

        # Should extract "Floor Plan", NOT "Gmp/permit Set"
        assert result['title'] == "Floor Plan"
        assert "Gmp" not in result['title']

    def test_handles_text_blocks(self, applier, sample_template):
        """Should use text_blocks when provided."""
        text_blocks = [
            {"text": "PROJECT NUMBER", "bbox": [500, 700, 600, 720], "block_no": 1, "font_size": 8},
            {"text": "Floor Plan Level 1", "bbox": [500, 600, 700, 650], "block_no": 2, "font_size": 14},
            {"text": "Scale: 1/4\"", "bbox": [500, 750, 600, 770], "block_no": 3, "font_size": 8},
        ]

        result = applier.apply(sample_template, "", text_blocks=text_blocks)

        assert result['title'] == "Floor Plan Level 1"
        assert result['confidence'] >= 0.80

    def test_respects_length_bounds(self, applier, sample_template):
        """Should reject titles outside length bounds."""
        # Template has typical_length_min=10, max=40
        # With 50% tolerance: 5 to 60

        page_text = """
        AB
        This is a very long title that exceeds the maximum length we would expect for a sheet title in a blueprint document
        Floor Plan
        """

        result = applier.apply(sample_template, page_text)

        assert result['title'] == "Floor Plan"  # Not the too-short or too-long ones

    def test_prefers_keyword_matches(self, applier, sample_template):
        """Should prefer lines containing quality keywords."""
        page_text = """
        Random Company Name
        Floor Plan Level 1
        Some Other Text Here
        """

        result = applier.apply(sample_template, page_text)

        assert result['title'] == "Floor Plan Level 1"

    def test_empty_page_text(self, applier, sample_template):
        """Should handle empty page text."""
        result = applier.apply(sample_template, "")

        assert result['title'] is None
        assert result['confidence'] == 0.0
        assert result['method'] == "template"

    def test_empty_text_blocks(self, applier, sample_template):
        """Should handle empty text blocks."""
        result = applier.apply(sample_template, "", text_blocks=[])

        assert result['title'] is None

    def test_blocks_preferred_over_text(self, applier, sample_template):
        """When both available, blocks should be tried first."""
        page_text = "Site Plan"
        text_blocks = [
            {"text": "Floor Plan from Block", "bbox": [100, 100, 200, 120], "block_no": 1, "font_size": 12},
        ]

        result = applier.apply(sample_template, page_text, text_blocks=text_blocks)

        # Should prefer the block result
        assert result['title'] == "Floor Plan from Block"

    def test_fallback_to_text_when_blocks_fail(self, applier, sample_template):
        """Should fall back to text when blocks don't produce result."""
        page_text = "Site Plan"
        # All blocks are excluded
        text_blocks = [
            {"text": "PROJECT", "bbox": [100, 100, 200, 120], "block_no": 1, "font_size": 12},
            {"text": "SHEET", "bbox": [100, 130, 200, 150], "block_no": 2, "font_size": 12},
        ]

        result = applier.apply(sample_template, page_text, text_blocks=text_blocks)

        assert result['title'] == "Site Plan"

    def test_rejects_sheet_numbers(self, applier, sample_template):
        """Should reject sheet number patterns."""
        page_text = """
        A1.0
        M-201
        Floor Plan
        """

        result = applier.apply(sample_template, page_text)

        assert result['title'] == "Floor Plan"
        assert "A1.0" not in result['title']
        assert "M-201" not in result['title']

    def test_rejects_pure_numbers(self, applier, sample_template):
        """Should reject purely numeric strings."""
        page_text = """
        12345
        67890
        Floor Plan
        """

        result = applier.apply(sample_template, page_text)

        assert result['title'] == "Floor Plan"

    def test_case_insensitive_exclusion(self, applier, sample_template):
        """Exclusion patterns should be case-insensitive."""
        page_text = """
        project info
        PROJECT DATA
        Floor Plan
        """

        result = applier.apply(sample_template, page_text)

        assert result['title'] == "Floor Plan"
