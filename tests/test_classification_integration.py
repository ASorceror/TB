"""
Integration Tests for Enhanced Classification System (V7.0)

End-to-end tests for:
    - Full classification pipeline
    - Sheet classifier with enhanced matching
    - HITL handler integration
    - Content analyzer triggers
"""

import pytest
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sheet_classifier import (
    SheetClassifier,
    ClassifiedSheet,
    ClassificationResult,
)
from core.enhanced_matcher import EnhancedPatternMatcher
from core.classification_types import ClassificationDecision, RelevanceLevel
from core.content_analyzer import (
    ContentAnalysisTrigger,
    ContentAnalyzer,
    PaintingRelevanceDetector,
    should_trigger_content_analysis,
)
from core.hitl_handler import (
    HITLDecisionHandler,
    ReviewQueue,
    ReviewItem,
)


# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_EXTRACTION_DATA = {
    "extraction_completed": "2024-01-15T10:00:00",
    "sheets": [
        {
            "pdf_filename": "test_project.pdf",
            "page_number": 1,
            "sheet_number": "G001",
            "sheet_title": "Cover Sheet",
            "number_confidence": 0.95,
            "title_confidence": 0.90,
        },
        {
            "pdf_filename": "test_project.pdf",
            "page_number": 2,
            "sheet_number": "A101",
            "sheet_title": "First Floor Plan",
            "number_confidence": 0.95,
            "title_confidence": 0.92,
        },
        {
            "pdf_filename": "test_project.pdf",
            "page_number": 3,
            "sheet_number": "A102",
            "sheet_title": "Second Floor Plan",
            "number_confidence": 0.95,
            "title_confidence": 0.88,
        },
        {
            "pdf_filename": "test_project.pdf",
            "page_number": 4,
            "sheet_number": "A120",
            "sheet_title": "Reflected Ceiling Plan",
            "number_confidence": 0.95,
            "title_confidence": 0.90,
        },
        {
            "pdf_filename": "test_project.pdf",
            "page_number": 5,
            "sheet_number": "A201",
            "sheet_title": "Exterior Elevations",
            "number_confidence": 0.95,
            "title_confidence": 0.85,
        },
        {
            "pdf_filename": "test_project.pdf",
            "page_number": 6,
            "sheet_number": "A301",
            "sheet_title": "Interior Elevations",
            "number_confidence": 0.95,
            "title_confidence": 0.87,
        },
        {
            "pdf_filename": "test_project.pdf",
            "page_number": 7,
            "sheet_number": "A601",
            "sheet_title": "Room Finish Schedule",
            "number_confidence": 0.95,
            "title_confidence": 0.91,
        },
        {
            "pdf_filename": "test_project.pdf",
            "page_number": 8,
            "sheet_number": "S101",
            "sheet_title": "Foundation Plan",
            "number_confidence": 0.95,
            "title_confidence": 0.88,
        },
        {
            "pdf_filename": "test_project.pdf",
            "page_number": 9,
            "sheet_number": "M101",
            "sheet_title": "Mechanical Plan",
            "number_confidence": 0.95,
            "title_confidence": 0.86,
        },
        {
            "pdf_filename": "test_project.pdf",
            "page_number": 10,
            "sheet_number": "E101",
            "sheet_title": "Electrical Plan",
            "number_confidence": 0.95,
            "title_confidence": 0.87,
        },
    ]
}


# =============================================================================
# SHEET CLASSIFIER INTEGRATION TESTS
# =============================================================================

class TestSheetClassifierIntegration:
    """Integration tests for SheetClassifier with enhanced matching."""

    def test_classify_extraction_results(self):
        """Test classifying full extraction results."""
        classifier = SheetClassifier(use_enhanced=True)
        result = classifier.classify_extraction_results(SAMPLE_EXTRACTION_DATA)

        assert result.total_pages == 10
        assert isinstance(result.statistics, dict)
        assert isinstance(result.categories, dict)
        assert isinstance(result.decisions, dict)

    def test_three_tier_decisions(self):
        """Test that three-tier decisions are populated."""
        classifier = SheetClassifier(use_enhanced=True)
        result = classifier.classify_extraction_results(SAMPLE_EXTRACTION_DATA)

        # Check decision buckets exist
        assert 'DEFINITELY_NEEDED' in result.decisions
        assert 'DEFINITELY_NOT_NEEDED' in result.decisions
        assert 'NEEDS_EVALUATION' in result.decisions

        # Should have some pages in each bucket
        needed = result.get_definitely_needed()
        not_needed = result.get_definitely_not_needed()
        needs_eval = result.get_needs_evaluation()

        # Floor plans, RCPs, schedules should be needed (or at least classified)
        assert len(needed) >= 4

        # Some sheets may need evaluation (depends on confidence thresholds)
        # Verify we have a reasonable distribution
        total = len(needed) + len(not_needed) + len(needs_eval)
        assert total == result.total_pages

        # Structural sheets are IRRELEVANT - they should be either
        # DEFINITELY_NOT_NEEDED or NEEDS_EVALUATION (not DEFINITELY_NEEDED)
        for sheet in [s for s in not_needed + needs_eval if s.sheet_number and s.sheet_number.startswith('S')]:
            assert sheet.relevance == 'IRRELEVANT'

    def test_statistics_include_enhanced_fields(self):
        """Test that statistics include enhanced classification fields."""
        classifier = SheetClassifier(use_enhanced=True)
        result = classifier.classify_extraction_results(SAMPLE_EXTRACTION_DATA)

        stats = result.statistics

        # Check for V7.0 statistics
        assert 'by_decision' in stats
        assert 'by_relevance' in stats
        assert 'combo_pages' in stats
        assert 'needs_review' in stats

    def test_classify_with_drawing_index(self):
        """Test classification using drawing index lookup."""
        drawing_index = {
            "A101": "First Floor Plan",
            "A102": "Second Floor Plan",
        }
        classifier = SheetClassifier(use_enhanced=True, drawing_index=drawing_index)
        result = classifier.classify_extraction_results(SAMPLE_EXTRACTION_DATA)

        # Floor plans should be classified as PRIMARY
        needed = result.get_definitely_needed()
        floor_plans = [s for s in needed if "floor_plans" in s.categories]
        assert len(floor_plans) >= 2

    def test_backward_compatibility(self):
        """Test backward compatibility with legacy classification."""
        # Use legacy mode
        classifier = SheetClassifier(use_enhanced=False)
        result = classifier.classify_extraction_results(SAMPLE_EXTRACTION_DATA)

        # Should still work with old classification_type
        stats = result.statistics
        assert 'matched' in stats['classification_types']
        assert 'not_needed' in stats['classification_types']
        assert 'unclassified' in stats['classification_types']

    def test_classify_from_file(self):
        """Test classifying from a JSON file."""
        # Create temp file with extraction data
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(SAMPLE_EXTRACTION_DATA, f)
            temp_path = Path(f.name)

        try:
            classifier = SheetClassifier(use_enhanced=True)
            result = classifier.classify_from_file(temp_path)

            assert result.total_pages == 10
        finally:
            temp_path.unlink()

    def test_relevance_classification(self):
        """Test that relevance is correctly assigned."""
        classifier = SheetClassifier(use_enhanced=True)
        result = classifier.classify_extraction_results(SAMPLE_EXTRACTION_DATA)

        # Check relevance statistics
        stats = result.statistics
        assert stats['by_relevance']['PRIMARY'] >= 4
        assert stats['by_relevance']['IRRELEVANT'] >= 3


# =============================================================================
# CONTENT ANALYZER INTEGRATION TESTS
# =============================================================================

class TestContentAnalyzerIntegration:
    """Integration tests for content analyzer."""

    def test_trigger_on_low_confidence(self):
        """Test that trigger activates on low confidence."""
        from core.enhanced_matcher import classify_sheet

        # Create a low-confidence classification
        result = classify_sheet("X999", "Unknown Drawing")

        trigger = ContentAnalysisTrigger()
        trigger_result = trigger.should_analyze(result)

        # Should trigger due to low confidence or no categories
        assert trigger_result.should_analyze is True

    def test_painting_indicator_detection(self):
        """Test detection of painting indicators from text."""
        detector = PaintingRelevanceDetector()

        text = """
        ROOM 101  PT-1
        ROOM 102  PT-2
        CEILING: ACT 2x4
        WALL TYPE: WT-3
        """

        indicators = detector.detect_from_text(text)

        # Should find room numbers, finish callouts, ceiling grid
        indicator_types = [i.indicator_type for i in indicators]
        assert 'room_numbers' in indicator_types
        assert 'finish_callouts' in indicator_types
        assert 'ceiling_grid' in indicator_types

    def test_relevance_from_indicators(self):
        """Test relevance calculation from indicators."""
        detector = PaintingRelevanceDetector()

        text = "ROOM 101 PT-1 ACT CEILING"
        indicators = detector.detect_from_text(text)
        relevance, confidence = detector.calculate_relevance(indicators)

        # Multiple primary indicators should give PRIMARY relevance
        assert relevance == RelevanceLevel.PRIMARY
        assert confidence >= 0.70


# =============================================================================
# HITL HANDLER INTEGRATION TESTS
# =============================================================================

class TestHITLHandlerIntegration:
    """Integration tests for HITL handler."""

    def test_queue_management(self):
        """Test adding and retrieving items from review queue."""
        with TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "review_queue.json"
            handler = HITLDecisionHandler(queue_storage_path=queue_path)

            # Create a sheet needing review
            sheet = ClassifiedSheet(
                pdf_name="test.pdf",
                page_number=1,
                sheet_number="X999",
                sheet_title="Unknown Drawing",
                categories=[],
                classification_type="unclassified",
                needs_review=True,
                review_reason="No categories matched",
                classification_confidence=0.40,
            )

            # Process classification
            flagged = handler.process_classification(sheet)
            assert flagged is True

            # Check queue
            stats = handler.get_review_summary()
            assert stats['statistics']['pending'] == 1

    def test_auto_flag_low_confidence(self):
        """Test auto-flagging of low confidence classifications."""
        handler = HITLDecisionHandler(auto_flag_threshold=0.70)

        sheet = ClassifiedSheet(
            pdf_name="test.pdf",
            page_number=1,
            sheet_number="A101",
            sheet_title="Plan",  # Ambiguous
            categories=["floor_plans"],
            classification_type="matched",
            classification_confidence=0.55,  # Below threshold
            needs_review=False,
        )

        flagged = handler.process_classification(sheet)
        assert flagged is True

    def test_no_flag_high_confidence(self):
        """Test that high confidence classifications are not flagged."""
        handler = HITLDecisionHandler(auto_flag_threshold=0.70)

        sheet = ClassifiedSheet(
            pdf_name="test.pdf",
            page_number=1,
            sheet_number="A101",
            sheet_title="First Floor Plan",
            categories=["floor_plans"],
            classification_type="matched",
            classification_confidence=0.90,  # Above threshold
            needs_review=False,
        )

        flagged = handler.process_classification(sheet)
        assert flagged is False

    def test_queue_persistence(self):
        """Test that queue is persisted to disk."""
        with TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "review_queue.json"

            # Create handler and add items
            handler1 = HITLDecisionHandler(queue_storage_path=queue_path)
            sheet = ClassifiedSheet(
                pdf_name="test.pdf",
                page_number=1,
                sheet_number="X999",
                sheet_title="Unknown",
                categories=[],
                classification_type="unclassified",
                needs_review=True,
                classification_confidence=0.40,
            )
            handler1.process_classification(sheet)
            handler1.queue.save()

            # Create new handler and load
            handler2 = HITLDecisionHandler(queue_storage_path=queue_path)

            # Should have the same item
            stats = handler2.get_review_summary()
            assert stats['statistics']['pending'] == 1

    def test_submit_review(self):
        """Test submitting a human review."""
        with TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "review_queue.json"
            handler = HITLDecisionHandler(queue_storage_path=queue_path)

            sheet = ClassifiedSheet(
                pdf_name="test.pdf",
                page_number=1,
                sheet_number="X999",
                sheet_title="Interior Finish Plan",
                categories=[],
                classification_type="unclassified",
                needs_review=True,
                classification_confidence=0.40,
            )
            handler.process_classification(sheet)

            # Get the item
            item = handler.queue.get_next()
            assert item is not None

            # Submit review
            handler.submit_review(
                item=item,
                decision="DEFINITELY_NEEDED",
                categories=["finish_plans"],
                notes="This is clearly a finish plan",
            )

            # Check it was reviewed
            stats = handler.get_review_summary()
            assert stats['statistics']['pending'] == 0
            assert stats['statistics']['reviewed'] == 1


# =============================================================================
# END-TO-END TESTS
# =============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test the full classification pipeline."""
        # 1. Classify extraction results
        classifier = SheetClassifier(use_enhanced=True)
        result = classifier.classify_extraction_results(SAMPLE_EXTRACTION_DATA)

        # 2. Verify three-tier decisions
        definitely_needed = result.get_definitely_needed()
        definitely_not_needed = result.get_definitely_not_needed()
        needs_evaluation = result.get_needs_evaluation()

        # 3. Verify counts make sense
        total = len(definitely_needed) + len(definitely_not_needed) + len(needs_evaluation)
        assert total == result.total_pages

        # 4. Verify painting-relevant pages are needed
        for sheet in definitely_needed:
            assert sheet.relevance in ['PRIMARY', 'SECONDARY']

        # 5. Verify MEP pages are not needed
        for sheet in definitely_not_needed:
            assert sheet.relevance == 'IRRELEVANT'

    def test_classification_to_hitl_flow(self):
        """Test flow from classification to HITL flagging."""
        with TemporaryDirectory() as tmpdir:
            # 1. Classify with some uncertain results
            classifier = SheetClassifier(use_enhanced=True)

            # Add an ambiguous sheet
            data = SAMPLE_EXTRACTION_DATA.copy()
            data['sheets'] = data['sheets'] + [{
                "pdf_filename": "test_project.pdf",
                "page_number": 11,
                "sheet_number": "X001",
                "sheet_title": "Plan",  # Very ambiguous
                "number_confidence": 0.60,
                "title_confidence": 0.50,
            }]

            result = classifier.classify_extraction_results(data)

            # 2. Process through HITL
            queue_path = Path(tmpdir) / "review_queue.json"
            handler = HITLDecisionHandler(queue_storage_path=queue_path)

            # Process needs_evaluation items
            for sheet in result.get_needs_evaluation():
                handler.process_classification(sheet)

            # 3. Verify flagging
            stats = handler.get_review_summary()
            assert stats['statistics']['pending'] >= 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
