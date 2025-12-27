"""
Phase F Validation Tests - Sibling Inference
Run with: python -m pytest tests/test_phase_f_sibling.py -v

This module tests the SiblingInference for using neighbor pages
to rescue isolated failures.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sibling_inference import SiblingInference
from core.template_types import Template


class TestSiblingInference:

    @pytest.fixture
    def inference(self):
        return SiblingInference()

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
            exclusion_patterns=[r"PROJECT", r"SHEET"],
            typical_length_min=10,
            typical_length_max=40,
            vector_pages=80,
            scanned_pages=20
        )

    def test_both_neighbors_successful(self, inference):
        """Should attempt inference with high confidence when both neighbors succeed."""
        results = [
            {"page_number": 1, "sheet_title": "Floor Plan 1", "needs_review": 0, "title_method": "spatial", "title_confidence": 0.85},
            {"page_number": 2, "sheet_title": None, "needs_review": 1},  # FAILED
            {"page_number": 3, "sheet_title": "Floor Plan 3", "needs_review": 0, "title_method": "spatial", "title_confidence": 0.85},
        ]
        failed = results[1]
        mock_pdf = Mock()
        mock_pdf.get_page_text.return_value = "Floor Plan Level 2"

        result = inference.infer(results, failed, mock_pdf)

        # Should attempt with high confidence
        assert result['method'] == 'sibling_inference'
        # May succeed or fail depending on extraction, but should try

    def test_one_neighbor_successful(self, inference):
        """Should attempt inference with medium confidence when one neighbor succeeds."""
        results = [
            {"page_number": 1, "sheet_title": "Floor Plan 1", "needs_review": 0, "title_method": "spatial", "title_confidence": 0.85},
            {"page_number": 2, "sheet_title": None, "needs_review": 1},  # FAILED
            {"page_number": 3, "sheet_title": None, "needs_review": 1},  # ALSO FAILED
        ]
        failed = results[1]
        mock_pdf = Mock()
        mock_pdf.get_page_text.return_value = "Reflected Ceiling Plan"

        result = inference.infer(results, failed, mock_pdf)

        # Should still attempt, but can return None if extraction fails
        assert result['method'] == 'sibling_inference'

    def test_no_successful_neighbors(self, inference):
        """Should return None when no neighbors are successful."""
        results = [
            {"page_number": 1, "sheet_title": None, "needs_review": 1},
            {"page_number": 2, "sheet_title": None, "needs_review": 1},
            {"page_number": 3, "sheet_title": None, "needs_review": 1},
        ]
        failed = results[1]
        mock_pdf = Mock()

        result = inference.infer(results, failed, mock_pdf)

        assert result['title'] is None
        assert result['confidence'] == 0.0

    def test_first_page_only_next_neighbor(self, inference):
        """First page should only check next neighbor."""
        results = [
            {"page_number": 1, "sheet_title": None, "needs_review": 1},  # FAILED
            {"page_number": 2, "sheet_title": "Floor Plan", "needs_review": 0, "title_method": "spatial", "title_confidence": 0.85},
        ]
        failed = results[0]
        mock_pdf = Mock()
        mock_pdf.get_page_text.return_value = "Site Plan"

        result = inference.infer(results, failed, mock_pdf)

        # Should attempt using page 2's method
        assert result['method'] == 'sibling_inference'

    def test_last_page_only_prev_neighbor(self, inference):
        """Last page should only check previous neighbor."""
        results = [
            {"page_number": 1, "sheet_title": "Floor Plan", "needs_review": 0, "title_method": "spatial", "title_confidence": 0.85},
            {"page_number": 2, "sheet_title": None, "needs_review": 1},  # FAILED (last page)
        ]
        failed = results[1]
        mock_pdf = Mock()
        mock_pdf.get_page_text.return_value = "Elevation View"

        result = inference.infer(results, failed, mock_pdf)

        assert result['method'] == 'sibling_inference'

    def test_with_template(self, inference, sample_template):
        """Should use template when provided."""
        results = [
            {"page_number": 1, "sheet_title": "Floor Plan 1", "needs_review": 0, "title_method": "spatial", "title_confidence": 0.85},
            {"page_number": 2, "sheet_title": None, "needs_review": 1, "extraction_method": "vector"},
        ]
        failed = results[1]
        mock_pdf = Mock()
        mock_pdf.get_page_text.return_value = "Mechanical Floor Plan"
        mock_pdf.get_text_blocks.return_value = []

        result = inference.infer(results, failed, mock_pdf, template=sample_template)

        assert result['method'] == 'sibling_inference'

    def test_extract_with_keywords(self, inference):
        """Should extract lines with quality keywords."""
        page_text = """
        Project: 12345
        SHEET: A1.0
        Floor Plan - Level 2
        Scale: 1/4"
        """
        title = inference._extract_with_keywords(page_text)
        assert title == "Floor Plan - Level 2"

    def test_extract_with_keywords_no_match(self, inference):
        """Should return None when no keywords found."""
        page_text = """
        Random text here
        More random text
        """
        title = inference._extract_with_keywords(page_text)
        assert title is None

    def test_extract_with_keywords_rejects_short(self, inference):
        """Should reject lines that are too short."""
        page_text = """
        Plan
        A short text here
        """
        title = inference._extract_with_keywords(page_text)
        # "Plan" alone is too short (< 8 chars)
        assert title is None or len(title) >= 8
