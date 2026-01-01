"""
Tests for Enhanced Classification System (V7.0)

Tests for:
    - Classification types and enums
    - NCS sheet number parsing
    - Title parsing and combo page detection
    - Enhanced pattern matching
    - Confidence calculation
    - Three-tier decision making
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.classification_types import (
    ClassificationDecision,
    RelevanceLevel,
    SignalSource,
    TitleComponent,
    ClassificationSignal,
    EnhancedClassificationResult,
    ConflictInfo,
)
from core.ncs_patterns import (
    NCSSheetParser,
    get_painting_relevance,
    ParsedSheetNumber,
)
from core.title_parser import (
    TitleParser,
    split_combo_title,
    detect_drawing_type,
    is_combo_page,
)
from core.enhanced_matcher import (
    EnhancedPatternMatcher,
    classify_sheet,
    get_three_tier_decision,
    is_painting_relevant,
)
from core.confidence import (
    aggregate_signals,
    detect_conflicts,
    calculate_final_confidence,
    determine_decision,
    get_relevance_for_category,
)


# =============================================================================
# CLASSIFICATION TYPES TESTS
# =============================================================================

class TestClassificationTypes:
    """Tests for classification type definitions."""

    def test_classification_decision_enum(self):
        """Test ClassificationDecision enum values."""
        assert ClassificationDecision.DEFINITELY_NEEDED is not None
        assert ClassificationDecision.DEFINITELY_NOT_NEEDED is not None
        assert ClassificationDecision.NEEDS_EVALUATION is not None

    def test_relevance_level_enum(self):
        """Test RelevanceLevel enum values."""
        assert RelevanceLevel.PRIMARY is not None
        assert RelevanceLevel.SECONDARY is not None
        assert RelevanceLevel.REFERENCE is not None
        assert RelevanceLevel.IRRELEVANT is not None

    def test_signal_source_enum(self):
        """Test SignalSource enum values."""
        assert SignalSource.DRAWING_INDEX.value == "drawing_index"
        assert SignalSource.TITLE_PATTERN.value == "title_pattern"
        assert SignalSource.SHEET_NUMBER.value == "sheet_number"

    def test_title_component_creation(self):
        """Test TitleComponent dataclass."""
        component = TitleComponent(
            text="FLOOR PLAN",
            drawing_type="floor_plan",
            confidence=0.90,
            original_position=0,
        )
        assert component.text == "FLOOR PLAN"
        assert component.drawing_type == "floor_plan"
        assert component.confidence == 0.90

    def test_classification_signal_weighted_score(self):
        """Test ClassificationSignal weighted score calculation."""
        signal = ClassificationSignal(
            source=SignalSource.TITLE_PATTERN,
            category="floor_plans",
            confidence=0.80,
            relevance=RelevanceLevel.PRIMARY,
            weight=1.0,
        )
        assert signal.weighted_score() == 0.80

        signal.weight = 0.5
        assert signal.weighted_score() == 0.40

    def test_enhanced_result_to_dict(self):
        """Test EnhancedClassificationResult serialization."""
        result = EnhancedClassificationResult(
            sheet_number="A101",
            sheet_title="First Floor Plan",
            decision=ClassificationDecision.DEFINITELY_NEEDED,
            categories=["floor_plans"],
            relevance=RelevanceLevel.PRIMARY,
            confidence=0.90,
        )
        data = result.to_dict()

        assert data['sheet_number'] == "A101"
        assert data['decision'] == "DEFINITELY_NEEDED"
        assert data['relevance'] == "PRIMARY"
        assert data['categories'] == ["floor_plans"]


# =============================================================================
# NCS PATTERNS TESTS
# =============================================================================

class TestNCSPatterns:
    """Tests for NCS sheet number parsing."""

    def test_parse_standard_sheet_number(self):
        """Test parsing standard NCS sheet numbers."""
        parser = NCSSheetParser()

        # A101 - Architectural Plans
        result = parser.parse("A101")
        assert result is not None
        assert result.discipline == "A"
        assert result.sheet_type == 1
        assert result.sequence == 1
        assert result.is_valid_ncs is True

    def test_parse_sheet_with_separator(self):
        """Test parsing sheet numbers with separators."""
        parser = NCSSheetParser()

        result = parser.parse("A-201")
        assert result is not None
        assert result.discipline == "A"
        assert result.sheet_type == 2  # Elevations
        assert result.sequence == 1

    def test_parse_interior_sheet(self):
        """Test parsing interior discipline sheets."""
        parser = NCSSheetParser()

        result = parser.parse("I-221")
        assert result is not None
        assert result.discipline == "I"
        assert result.discipline_name == "Interiors"
        assert result.relevance == RelevanceLevel.PRIMARY  # Interior elevations

    def test_parse_structural_sheet(self):
        """Test parsing structural discipline sheets."""
        parser = NCSSheetParser()

        result = parser.parse("S301")
        assert result is not None
        assert result.discipline == "S"
        assert result.relevance == RelevanceLevel.IRRELEVANT

    def test_parse_mep_sheets(self):
        """Test parsing MEP discipline sheets are IRRELEVANT."""
        parser = NCSSheetParser()

        for prefix in ["M", "E", "P"]:
            result = parser.parse(f"{prefix}101")
            assert result is not None
            assert result.relevance == RelevanceLevel.IRRELEVANT

    def test_painting_relevance_convenience(self):
        """Test get_painting_relevance convenience function."""
        assert get_painting_relevance("A101") == RelevanceLevel.PRIMARY
        assert get_painting_relevance("A201") == RelevanceLevel.SECONDARY
        assert get_painting_relevance("S101") == RelevanceLevel.IRRELEVANT
        assert get_painting_relevance("M101") == RelevanceLevel.IRRELEVANT

    def test_category_hint(self):
        """Test NCS category hints."""
        parser = NCSSheetParser()

        result = parser.parse("A101")
        hint = parser.get_category_hint(result)
        assert hint == "floor_plans"

        result = parser.parse("A601")
        hint = parser.get_category_hint(result)
        assert hint == "room_finish_schedules"


# =============================================================================
# TITLE PARSER TESTS
# =============================================================================

class TestTitleParser:
    """Tests for title parsing and combo page detection."""

    def test_simple_title_parsing(self):
        """Test parsing a simple single title."""
        parser = TitleParser()
        components = parser.parse("First Floor Plan")

        assert len(components) == 1
        assert components[0].text == "First Floor Plan"
        assert components[0].drawing_type == "floor_plan"

    def test_combo_page_newline(self):
        """Test detecting combo page with newline separator."""
        parser = TitleParser()
        components = parser.parse("FLOOR PLAN\nROOF PLAN")

        assert len(components) == 2
        assert "FLOOR PLAN" in components[0].text
        assert "ROOF PLAN" in components[1].text

    def test_combo_page_and(self):
        """Test detecting combo page with AND separator."""
        parser = TitleParser()
        components = parser.parse("FLOOR PLAN AND REFLECTED CEILING PLAN")

        assert len(components) == 2
        assert components[0].drawing_type == "floor_plan"
        assert components[1].drawing_type == "reflected_ceiling_plan"

    def test_not_combo_floor_plan_level(self):
        """Test that 'FLOOR PLAN - LEVEL 1' is NOT a combo."""
        parser = TitleParser()
        components = parser.parse("FLOOR PLAN - LEVEL 1")

        # This should NOT be split - it's one title
        assert len(components) == 1
        assert "LEVEL" in components[0].text

    def test_is_combo_page_convenience(self):
        """Test is_combo_page convenience function."""
        assert is_combo_page("FLOOR PLAN\nROOF PLAN") is True
        assert is_combo_page("First Floor Plan") is False
        assert is_combo_page("FLOOR PLAN - LEVEL 1") is False

    def test_detect_drawing_type(self):
        """Test drawing type detection."""
        assert detect_drawing_type("First Floor Plan") == "floor_plan"
        assert detect_drawing_type("REFLECTED CEILING PLAN") == "reflected_ceiling_plan"
        assert detect_drawing_type("INTERIOR ELEVATION") == "interior_elevation"
        assert detect_drawing_type("EAST ELEVATION") == "exterior_elevation"

    def test_rcp_detection(self):
        """Test RCP acronym detection."""
        parser = TitleParser()
        components = parser.parse("LEVEL 1 RCP")

        assert len(components) == 1
        assert components[0].drawing_type == "reflected_ceiling_plan"


# =============================================================================
# CONFIDENCE CALCULATION TESTS
# =============================================================================

class TestConfidenceCalculation:
    """Tests for confidence calculation and signal aggregation."""

    def test_aggregate_single_signal(self):
        """Test aggregating a single signal."""
        signals = [
            ClassificationSignal(
                source=SignalSource.TITLE_PATTERN,
                category="floor_plans",
                confidence=0.90,
                relevance=RelevanceLevel.PRIMARY,
            )
        ]
        result = aggregate_signals(signals)

        assert result.top_category == "floor_plans"
        assert result.total_signals == 1

    def test_aggregate_multiple_signals(self):
        """Test aggregating multiple signals for same category."""
        signals = [
            ClassificationSignal(
                source=SignalSource.TITLE_PATTERN,
                category="floor_plans",
                confidence=0.90,
                relevance=RelevanceLevel.PRIMARY,
            ),
            ClassificationSignal(
                source=SignalSource.SHEET_NUMBER,
                category="floor_plans",
                confidence=0.75,
                relevance=RelevanceLevel.PRIMARY,
            ),
        ]
        result = aggregate_signals(signals)

        assert result.top_category == "floor_plans"
        assert result.total_signals == 2
        # Score should be cumulative
        assert result.top_score > 0

    def test_detect_conflicts(self):
        """Test conflict detection when signals disagree."""
        signals = [
            ClassificationSignal(
                source=SignalSource.TITLE_PATTERN,
                category="floor_plans",
                confidence=0.80,
                relevance=RelevanceLevel.PRIMARY,
            ),
            ClassificationSignal(
                source=SignalSource.SHEET_NUMBER,
                category="exterior_elevations",
                confidence=0.75,
                relevance=RelevanceLevel.SECONDARY,
            ),
        ]
        aggregation = aggregate_signals(signals)
        conflict = detect_conflicts(aggregation)

        # With similar scores, should detect conflict
        assert conflict.has_conflict is True
        assert len(conflict.conflicting_categories) == 2

    def test_no_conflict_when_clear_winner(self):
        """Test no conflict when one signal is much stronger."""
        signals = [
            ClassificationSignal(
                source=SignalSource.DRAWING_INDEX,
                category="floor_plans",
                confidence=0.95,
                relevance=RelevanceLevel.PRIMARY,
                weight=1.0,
            ),
            ClassificationSignal(
                source=SignalSource.SHEET_NUMBER,
                category="exterior_elevations",
                confidence=0.50,
                relevance=RelevanceLevel.SECONDARY,
                weight=1.0,
            ),
        ]
        aggregation = aggregate_signals(signals)
        conflict = detect_conflicts(aggregation)

        # Clear winner should not trigger conflict
        assert conflict.has_conflict is False

    def test_determine_decision_definitely_needed(self):
        """Test decision is DEFINITELY_NEEDED for high confidence primary."""
        decision = determine_decision(
            confidence=0.90,
            relevance=RelevanceLevel.PRIMARY,
            has_conflict=False,
        )
        assert decision == ClassificationDecision.DEFINITELY_NEEDED

    def test_determine_decision_definitely_not_needed(self):
        """Test decision is DEFINITELY_NOT_NEEDED for high confidence irrelevant."""
        decision = determine_decision(
            confidence=0.90,
            relevance=RelevanceLevel.IRRELEVANT,
            has_conflict=False,
        )
        assert decision == ClassificationDecision.DEFINITELY_NOT_NEEDED

    def test_determine_decision_needs_evaluation_conflict(self):
        """Test decision is NEEDS_EVALUATION when conflict exists."""
        decision = determine_decision(
            confidence=0.90,
            relevance=RelevanceLevel.PRIMARY,
            has_conflict=True,
        )
        assert decision == ClassificationDecision.NEEDS_EVALUATION


# =============================================================================
# ENHANCED MATCHER TESTS
# =============================================================================

class TestEnhancedMatcher:
    """Tests for enhanced pattern matcher."""

    def test_classify_floor_plan(self):
        """Test classifying a floor plan."""
        result = classify_sheet("A101", "First Floor Plan")

        assert result.decision == ClassificationDecision.DEFINITELY_NEEDED
        assert "floor_plans" in result.categories
        assert result.relevance == RelevanceLevel.PRIMARY
        assert result.confidence >= 0.70

    def test_classify_structural(self):
        """Test classifying a structural sheet as not needed."""
        result = classify_sheet("S101", "Foundation Plan")

        # Structural sheets are IRRELEVANT for painting
        assert result.relevance == RelevanceLevel.IRRELEVANT
        # May be DEFINITELY_NOT_NEEDED or NEEDS_EVALUATION depending on confidence
        assert result.decision in [
            ClassificationDecision.DEFINITELY_NOT_NEEDED,
            ClassificationDecision.NEEDS_EVALUATION,
        ]

    def test_classify_rcp(self):
        """Test classifying a reflected ceiling plan."""
        result = classify_sheet("A120", "Reflected Ceiling Plan")

        assert "reflected_ceiling_plans" in result.categories
        assert result.relevance == RelevanceLevel.PRIMARY

    def test_classify_combo_page(self):
        """Test classifying a combo page."""
        result = classify_sheet("A101", "FLOOR PLAN\nROOF PLAN")

        assert result.is_combo_page is True
        assert len(result.title_components) == 2

    def test_drawing_index_integration(self):
        """Test that drawing index overrides other signals."""
        matcher = EnhancedPatternMatcher(
            drawing_index={"A101": "First Floor Plan"}
        )
        # Title is ambiguous but drawing index is clear
        result = matcher.classify("A101", "PLAN")

        assert "floor_plans" in result.categories

    def test_is_painting_relevant_convenience(self):
        """Test is_painting_relevant convenience function."""
        assert is_painting_relevant("A101", "First Floor Plan") is True
        assert is_painting_relevant("S101", "Foundation Plan") is False
        assert is_painting_relevant("M101", "Mechanical Plan") is False

    def test_classify_finish_schedule(self):
        """Test classifying a finish schedule."""
        result = classify_sheet("A601", "Room Finish Schedule")

        assert "room_finish_schedules" in result.categories
        assert result.relevance == RelevanceLevel.PRIMARY

    def test_classify_interior_elevation(self):
        """Test classifying an interior elevation."""
        result = classify_sheet("A301", "Interior Elevations")

        assert "interior_elevations" in result.categories
        assert result.relevance == RelevanceLevel.PRIMARY


# =============================================================================
# RELEVANCE MAPPING TESTS
# =============================================================================

class TestRelevanceMapping:
    """Tests for painting trade relevance mapping."""

    def test_primary_categories(self):
        """Test that primary categories are correctly mapped."""
        primary_categories = [
            "floor_plans",
            "reflected_ceiling_plans",
            "room_finish_schedules",
            "interior_elevations",
        ]
        for category in primary_categories:
            relevance = get_relevance_for_category(category)
            assert relevance == RelevanceLevel.PRIMARY, f"{category} should be PRIMARY"

    def test_secondary_categories(self):
        """Test that secondary categories are correctly mapped."""
        secondary_categories = [
            "exterior_elevations",
            "building_sections",
            "door_schedules",
        ]
        for category in secondary_categories:
            relevance = get_relevance_for_category(category)
            assert relevance == RelevanceLevel.SECONDARY, f"{category} should be SECONDARY"

    def test_irrelevant_categories(self):
        """Test that irrelevant categories are correctly mapped."""
        irrelevant_categories = [
            "structural",
            "mechanical",
            "electrical",
            "plumbing",
            "properly_classified_not_needed",
        ]
        for category in irrelevant_categories:
            relevance = get_relevance_for_category(category)
            assert relevance == RelevanceLevel.IRRELEVANT, f"{category} should be IRRELEVANT"

    def test_unknown_category_default(self):
        """Test that unknown categories default to REFERENCE."""
        relevance = get_relevance_for_category("unknown_category")
        assert relevance == RelevanceLevel.REFERENCE


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
