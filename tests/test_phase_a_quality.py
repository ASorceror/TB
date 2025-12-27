"""
Phase A Validation Tests - Quality Filter
Run with: python -m pytest tests/test_phase_a_quality.py -v

This module tests the QualityFilter class which identifies truly correct
extractions suitable for template learning.
"""
import pytest
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.quality_filter import QualityFilter


class TestQualityFilter:

    @pytest.fixture
    def qf(self):
        return QualityFilter()

    # =========================================================================
    # TEST 1: Must REJECT these (reason may vary for short strings)
    # =========================================================================
    @pytest.mark.parametrize("title,expected_reasons", [
        ("", ["empty"]),
        (None, ["empty"]),
        ("A", ["too_short"]),
        ("AB", ["too_short"]),
        ("Section", ["single_keyword"]),           # Single keyword alone (7 chars)
        ("Plan", ["too_short", "single_keyword"]), # 4 chars - caught by too_short first
        ("Elevation", ["single_keyword"]),         # Single keyword alone (9 chars)
        ("Detail", ["single_keyword"]),            # Single keyword alone (6 chars)
        ("Project Number", ["garbage_pattern"]),   # Label
        ("Sheet Number", ["garbage_pattern"]),     # Label
        ("Gmp/permit Set", ["garbage_pattern"]),   # Project descriptor
        ("1425", ["too_short", "garbage_pattern"]),# 4 chars - caught by too_short first
        ("A101", ["too_short", "garbage_pattern"]),# 4 chars - caught by too_short first
        ("A1.0", ["too_short", "garbage_pattern"]),# 4 chars - caught by too_short first
        ("Part 1 - General", ["garbage_pattern"]), # Spec header
        ("Date", ["too_short", "garbage_pattern"]),# 4 chars - caught by too_short first
        ("Scale: 1/4", ["garbage_pattern"]),       # Label
        ("Random Text Here", ["no_keyword_or_context"]),  # No keyword
    ])
    def test_rejects_bad_titles(self, qf, title, expected_reasons):
        is_quality, reason = qf.is_quality_title(title)
        assert is_quality == False, f"'{title}' should be rejected"
        assert reason in expected_reasons, f"'{title}' rejected for unexpected reason: {reason} (expected one of {expected_reasons})"

    # =========================================================================
    # TEST 2: Must ACCEPT these
    # =========================================================================
    @pytest.mark.parametrize("title", [
        "Floor Plan",
        "First Floor Demolition Plan",
        "Reflected Ceiling Plan - Level 2",
        "Electrical Riser Diagram",
        "North Elevation",
        "Building Section A",
        "Section at Stair",
        "Wall Section Detail",
        "Mechanical Floor Plan",
        "Plumbing Fixture Schedule",
        "Door and Frame Details",
        "Site Plan",
        "Foundation Plan",
        "Roof Framing Plan",
    ])
    def test_accepts_good_titles(self, qf, title):
        is_quality, reason = qf.is_quality_title(title)
        assert is_quality == True, f"'{title}' should be accepted, got reason: {reason}"

    # =========================================================================
    # TEST 3: Run on actual production data (if available)
    # =========================================================================
    def test_actual_data_quality_rate(self, qf):
        """Test against real V4.2.1 output - must achieve ~43% quality rate."""

        # Try multiple possible locations for production data
        data_paths = [
            Path("/mnt/user-data/uploads/report_20251214_210109.json"),
            Path("C:/tb/report_20251214_210109.json"),
            Path("C:/tb/blueprint_processor/output/report_20251214_210109.json"),
        ]

        data_path = None
        for p in data_paths:
            if p.exists():
                data_path = p
                break

        if data_path is None:
            pytest.skip("Production data not available")

        with open(data_path) as f:
            data = json.load(f)

        extractions = data['sheets']
        quality = qf.filter_for_learning(extractions)

        total = len(extractions)
        quality_count = len(quality)
        quality_rate = quality_count / total

        print(f"\nQuality filter results:")
        print(f"  Total extractions: {total}")
        print(f"  Quality extractions: {quality_count}")
        print(f"  Quality rate: {quality_rate:.1%}")

        # Must be approximately 43% (between 35% and 50%)
        assert 0.35 <= quality_rate <= 0.50, \
            f"Quality rate {quality_rate:.1%} outside expected range 35-50%"

    # =========================================================================
    # TEST 4: Specific bad patterns must be rejected
    # =========================================================================
    def test_rejects_gmp_permit_set(self, qf):
        """All 248 instances of 'Gmp/permit Set' must be rejected."""
        is_quality, _ = qf.is_quality_title("Gmp/permit Set")
        assert is_quality == False

        is_quality, _ = qf.is_quality_title("GMP/Permit Set")
        assert is_quality == False

    def test_rejects_section_alone(self, qf):
        """'Section' alone must be rejected, but 'Building Section' accepted."""
        is_quality, _ = qf.is_quality_title("Section")
        assert is_quality == False, "'Section' alone should be rejected"

        is_quality, _ = qf.is_quality_title("Building Section")
        assert is_quality == True, "'Building Section' should be accepted"

    # =========================================================================
    # TEST 5: Repetition filter works
    # =========================================================================
    def test_repetition_filter(self, qf):
        """Titles appearing on >20% of pages should be filtered out."""

        # Create mock data: 100 pages, one title appears 25 times
        extractions = []
        for i in range(100):
            if i < 25:
                title = "Repeated Title Plan"  # 25% - should be filtered
            else:
                title = f"Unique Floor Plan {i}"
            extractions.append({
                "sheet_title": title,
                "needs_review": 0
            })

        quality = qf.filter_for_learning(extractions)

        # "Repeated Title Plan" should be excluded (appears 25% > 20%)
        titles = [e['sheet_title'] for e in quality]
        assert "Repeated Title Plan" not in titles

    # =========================================================================
    # TEST 6: Edge cases
    # =========================================================================
    def test_empty_list(self, qf):
        """Empty list should return empty list."""
        result = qf.filter_for_learning([])
        assert result == []

    def test_all_needs_review(self, qf):
        """All needs_review=1 should return empty list."""
        extractions = [
            {"sheet_title": "Floor Plan", "needs_review": 1},
            {"sheet_title": "Elevation", "needs_review": 1},
        ]
        result = qf.filter_for_learning(extractions)
        assert result == []

    def test_whitespace_handling(self, qf):
        """Titles with extra whitespace should be handled."""
        is_quality, _ = qf.is_quality_title("  Floor Plan  ")
        assert is_quality == True

    def test_case_insensitivity(self, qf):
        """Keywords should match case-insensitively."""
        is_quality, _ = qf.is_quality_title("floor plan")
        assert is_quality == True

        is_quality, _ = qf.is_quality_title("FLOOR PLAN")
        assert is_quality == True

    # =========================================================================
    # TEST 7: Boundary cases for length
    # =========================================================================
    def test_exactly_5_chars_no_keyword(self, qf):
        """5 chars without keyword should be rejected."""
        is_quality, reason = qf.is_quality_title("Abcde")
        assert is_quality == False
        assert reason == "no_keyword_or_context"

    def test_exactly_4_chars(self, qf):
        """4 chars should be rejected as too short."""
        is_quality, reason = qf.is_quality_title("Abcd")
        assert is_quality == False
        assert reason == "too_short"
