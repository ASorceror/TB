"""
Enhanced Pattern Matcher Module (V7.0)

Integrates NCS parsing, title parsing, and signal aggregation
into a comprehensive classification system for blueprint pages.

This module provides:
    - EnhancedPatternMatcher: Main classifier with multi-signal support
    - classify_sheet: Convenience function for sheet classification
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .classification_types import (
    ClassificationSignal,
    ClassificationDecision,
    RelevanceLevel,
    SignalSource,
    TitleComponent,
    ConflictInfo,
    EnhancedClassificationResult,
)
from .ncs_patterns import NCSSheetParser, get_painting_relevance
from .title_parser import TitleParser
from .confidence import (
    calculate_final_confidence,
    determine_decision,
    get_relevance_for_category,
    needs_human_review,
    aggregate_signals,
)
from .pattern_matchers import (
    RegexPatternMatcher,
    SheetNumberMatcher,
    UnclassifiedMatcher,
    MatchResult,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    SHEET_CATEGORIES,
    EXPANDED_SHEET_CATEGORIES,
    CLASSIFICATION_THRESHOLDS,
    PAINTING_RELEVANCE,
)


# =============================================================================
# ENHANCED PATTERN MATCHER
# =============================================================================

class EnhancedPatternMatcher:
    """
    Enhanced pattern matcher with multi-signal classification.

    Combines multiple classification methods:
    1. NCS sheet number parsing (discipline + sheet type)
    2. Title pattern matching (regex patterns)
    3. Title parsing for combo page detection
    4. Signal aggregation with weighted confidence

    Returns EnhancedClassificationResult with three-tier decision.
    """

    def __init__(self, drawing_index: Optional[Dict[str, str]] = None):
        """
        Initialize the enhanced matcher.

        Args:
            drawing_index: Optional dict mapping sheet_number -> title
                          from PDF cover sheet
        """
        self.drawing_index = drawing_index or {}

        # Initialize component matchers
        self.ncs_parser = NCSSheetParser()
        self.title_parser = TitleParser()

        # Combine base and expanded categories
        all_categories = {**SHEET_CATEGORIES, **EXPANDED_SHEET_CATEGORIES}
        self.regex_matcher = RegexPatternMatcher(categories=all_categories)
        self.sheet_number_matcher = SheetNumberMatcher(categories=all_categories)
        self.unclassified_matcher = UnclassifiedMatcher()

    def set_drawing_index(self, drawing_index: Dict[str, str]) -> None:
        """
        Set the drawing index for authoritative lookups.

        Args:
            drawing_index: Dict mapping sheet_number -> title
        """
        self.drawing_index = drawing_index

    def classify(
        self,
        sheet_number: Optional[str],
        sheet_title: Optional[str],
    ) -> EnhancedClassificationResult:
        """
        Classify a sheet using all available signals.

        Args:
            sheet_number: Extracted sheet number (e.g., "A101")
            sheet_title: Extracted sheet title (e.g., "First Floor Plan")

        Returns:
            EnhancedClassificationResult with full classification details
        """
        signals: List[ClassificationSignal] = []
        metadata: Dict[str, any] = {}

        # =================================================================
        # SIGNAL 1: Drawing Index Lookup (highest authority)
        # =================================================================
        if sheet_number and self.drawing_index:
            index_title = self.drawing_index.get(sheet_number)
            if index_title:
                metadata['drawing_index_title'] = index_title
                # Use index title for classification if no title provided
                if not sheet_title:
                    sheet_title = index_title

                # Add drawing index signal
                index_matches = self.regex_matcher.match(sheet_number, index_title)
                for match in index_matches:
                    if match.category:
                        relevance = get_relevance_for_category(match.category)
                        signals.append(ClassificationSignal(
                            source=SignalSource.DRAWING_INDEX,
                            category=match.category,
                            confidence=0.95,  # Very high confidence
                            relevance=relevance,
                            evidence=f"Drawing index: {index_title}",
                            weight=1.0,
                        ))

        # =================================================================
        # SIGNAL 2: NCS Sheet Number Parsing
        # =================================================================
        ncs_parsed = self.ncs_parser.parse(sheet_number)
        if ncs_parsed:
            metadata['ncs'] = {
                'discipline': ncs_parsed.discipline,
                'discipline_name': ncs_parsed.discipline_name,
                'sheet_type': ncs_parsed.sheet_type,
                'sheet_type_name': ncs_parsed.sheet_type_name,
                'is_valid_ncs': ncs_parsed.is_valid_ncs,
            }

            # Get category hint from NCS
            category_hint = self.ncs_parser.get_category_hint(ncs_parsed)
            if category_hint:
                signals.append(ClassificationSignal(
                    source=SignalSource.SHEET_NUMBER,
                    category=category_hint,
                    confidence=0.70 if ncs_parsed.is_valid_ncs else 0.50,
                    relevance=ncs_parsed.relevance,
                    evidence=f"NCS: {ncs_parsed.discipline}{ncs_parsed.sheet_type}",
                    weight=1.0,
                ))

        # =================================================================
        # SIGNAL 3: Title Pattern Matching
        # =================================================================
        # Parse title for combo pages
        title_components = self.title_parser.parse(sheet_title)
        is_combo_page = len(title_components) > 1

        for component in title_components:
            # Match each component
            title_matches = self.regex_matcher.match(sheet_number, component.text)
            for match in title_matches:
                if match.category:
                    relevance = get_relevance_for_category(match.category)
                    component.drawing_type = match.category
                    component.confidence = match.confidence

                    signals.append(ClassificationSignal(
                        source=SignalSource.TITLE_PATTERN,
                        category=match.category,
                        confidence=match.confidence,
                        relevance=relevance,
                        evidence=f"Title pattern: {match.matched_pattern}",
                        weight=1.0,
                    ))

        # =================================================================
        # SIGNAL 4: Sheet Number Pattern Matching (from existing matchers)
        # =================================================================
        if sheet_number and not ncs_parsed:
            # Fallback to regex-based sheet number matching
            sheet_matches = self.sheet_number_matcher.match(sheet_number, sheet_title)
            for match in sheet_matches:
                if match.category:
                    relevance = get_relevance_for_category(match.category)
                    signals.append(ClassificationSignal(
                        source=SignalSource.SHEET_NUMBER,
                        category=match.category,
                        confidence=match.confidence,
                        relevance=relevance,
                        evidence=f"Sheet pattern: {match.matched_pattern}",
                        weight=1.0,
                    ))

        # =================================================================
        # SIGNAL 5: Unclassified/Not-Needed Detection
        # =================================================================
        if not signals:
            # Check if this is a recognized but not-needed type
            unclassified_matches = self.unclassified_matcher.match(sheet_number, sheet_title)
            for match in unclassified_matches:
                if match.category:
                    signals.append(ClassificationSignal(
                        source=SignalSource.TITLE_PATTERN,
                        category=match.category,
                        confidence=match.confidence,
                        relevance=RelevanceLevel.IRRELEVANT,
                        evidence=f"Not-needed pattern: {match.matched_pattern}",
                        weight=1.0,
                    ))

        # =================================================================
        # AGGREGATE SIGNALS AND DETERMINE DECISION
        # =================================================================
        final_confidence, conflict = calculate_final_confidence(
            signals,
            is_combo_page=is_combo_page,
        )

        # Get top categories from aggregation
        aggregation = aggregate_signals(signals)
        categories = self._get_categories_from_signals(signals, aggregation)

        # Determine relevance from top category
        top_category = categories[0] if categories else None
        relevance = get_relevance_for_category(top_category)

        # If NCS gives irrelevant, override
        if ncs_parsed and ncs_parsed.relevance == RelevanceLevel.IRRELEVANT:
            relevance = RelevanceLevel.IRRELEVANT

        # Determine final decision
        decision = determine_decision(
            final_confidence,
            relevance,
            has_conflict=conflict.has_conflict,
        )

        # Check if human review is needed
        review_needed, review_reason = needs_human_review(
            decision,
            final_confidence,
            has_conflict=conflict.has_conflict,
            is_combo_page=is_combo_page,
        )

        return EnhancedClassificationResult(
            sheet_number=sheet_number,
            sheet_title=sheet_title,
            decision=decision,
            categories=categories,
            relevance=relevance,
            confidence=final_confidence,
            signals=signals,
            title_components=title_components,
            is_combo_page=is_combo_page,
            conflict=conflict,
            needs_human_review=review_needed,
            review_reason=review_reason,
            metadata=metadata,
        )

    def _get_categories_from_signals(
        self,
        signals: List[ClassificationSignal],
        aggregation: any,
    ) -> List[str]:
        """
        Extract unique categories from signals, ordered by score.

        Args:
            signals: Classification signals
            aggregation: Aggregation result

        Returns:
            List of category names, highest score first
        """
        if not aggregation.category_scores:
            return []

        # Sort by score, descending
        sorted_categories = sorted(
            aggregation.category_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [cat for cat, score in sorted_categories]

    def get_classification_summary(
        self,
        result: EnhancedClassificationResult,
    ) -> Dict[str, any]:
        """
        Get a summary of the classification for reporting.

        Args:
            result: Classification result

        Returns:
            Dict with summary information
        """
        return {
            'sheet_number': result.sheet_number,
            'sheet_title': result.sheet_title,
            'decision': result.decision.name,
            'primary_category': result.primary_category,
            'relevance': result.relevance.name,
            'confidence': f"{result.confidence:.2f}",
            'is_combo_page': result.is_combo_page,
            'needs_review': result.needs_human_review,
            'review_reason': result.review_reason,
            'signal_count': len(result.signals),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def classify_sheet(
    sheet_number: Optional[str],
    sheet_title: Optional[str],
    drawing_index: Optional[Dict[str, str]] = None,
) -> EnhancedClassificationResult:
    """
    Convenience function to classify a single sheet.

    Args:
        sheet_number: Extracted sheet number
        sheet_title: Extracted sheet title
        drawing_index: Optional drawing index from PDF

    Returns:
        EnhancedClassificationResult
    """
    matcher = EnhancedPatternMatcher(drawing_index=drawing_index)
    return matcher.classify(sheet_number, sheet_title)


def get_three_tier_decision(
    sheet_number: Optional[str],
    sheet_title: Optional[str],
) -> ClassificationDecision:
    """
    Get just the three-tier decision for a sheet.

    Args:
        sheet_number: Extracted sheet number
        sheet_title: Extracted sheet title

    Returns:
        ClassificationDecision enum value
    """
    result = classify_sheet(sheet_number, sheet_title)
    return result.decision


def is_painting_relevant(
    sheet_number: Optional[str],
    sheet_title: Optional[str],
) -> bool:
    """
    Check if a sheet is relevant for painting trade.

    Args:
        sheet_number: Extracted sheet number
        sheet_title: Extracted sheet title

    Returns:
        True if relevant for painting trade
    """
    result = classify_sheet(sheet_number, sheet_title)
    return result.is_painting_relevant
