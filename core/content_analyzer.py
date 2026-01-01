"""
Content Analyzer Module (V7.0)

Triggers and performs content analysis on drawing areas when
title-based classification is ambiguous or uncertain.

This module provides:
    - ContentAnalysisTrigger: Determines when analysis is needed
    - ContentAnalyzer: Performs visual content analysis
    - PaintingRelevanceDetector: Detects painting-specific elements
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from PIL import Image

from .classification_types import (
    ClassificationSignal,
    RelevanceLevel,
    SignalSource,
    EnhancedClassificationResult,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import CONTENT_ANALYSIS_TRIGGERS, CLASSIFICATION_THRESHOLDS


# =============================================================================
# CONTENT ANALYSIS TRIGGERS
# =============================================================================

@dataclass
class TriggerResult:
    """
    Result of trigger evaluation.

    Attributes:
        should_analyze: Whether content analysis should be performed
        reasons: List of reasons for triggering
        priority: Priority level (higher = more urgent)
    """
    should_analyze: bool
    reasons: List[str]
    priority: int = 0  # 0=low, 1=medium, 2=high


class ContentAnalysisTrigger:
    """
    Determines when content analysis should be triggered.

    Triggers include:
    - Low confidence from title-based classification
    - Combo pages with multiple drawings
    - Ambiguous titles that don't clearly indicate content type
    - Conflicting signals from different sources
    """

    def __init__(self):
        """Initialize the trigger checker."""
        # Compile ambiguous title patterns
        patterns = CONTENT_ANALYSIS_TRIGGERS.get('ambiguous_title_patterns', [])
        self._ambiguous_patterns = [
            re.compile(p, re.IGNORECASE) for p in patterns
        ]
        self._low_confidence_threshold = CONTENT_ANALYSIS_TRIGGERS.get(
            'low_confidence_threshold', 0.65
        )

    def should_analyze(
        self,
        result: EnhancedClassificationResult,
    ) -> TriggerResult:
        """
        Determine if content analysis should be triggered.

        Args:
            result: Enhanced classification result from title-based classification

        Returns:
            TriggerResult indicating whether to analyze and why
        """
        reasons = []
        priority = 0

        # Check low confidence
        if result.confidence < self._low_confidence_threshold:
            reasons.append(f"Low confidence ({result.confidence:.2f})")
            priority = max(priority, 2)

        # Check combo page
        if result.is_combo_page:
            reasons.append("Combo page with multiple drawings")
            priority = max(priority, 1)

        # Check for conflicts
        if result.conflict.has_conflict:
            reasons.append(f"Conflicting signals: {result.conflict.conflicting_categories}")
            priority = max(priority, 2)

        # Check ambiguous title
        if result.sheet_title and self._is_ambiguous_title(result.sheet_title):
            reasons.append("Ambiguous title pattern")
            priority = max(priority, 1)

        # Check no categories found
        if not result.categories:
            reasons.append("No categories matched")
            priority = max(priority, 2)

        should_analyze = len(reasons) > 0

        return TriggerResult(
            should_analyze=should_analyze,
            reasons=reasons,
            priority=priority,
        )

    def _is_ambiguous_title(self, title: str) -> bool:
        """
        Check if a title is ambiguous.

        Args:
            title: Sheet title to check

        Returns:
            True if title matches ambiguous patterns
        """
        title_clean = title.strip()

        for pattern in self._ambiguous_patterns:
            if pattern.match(title_clean):
                return True

        return False


# =============================================================================
# PAINTING-SPECIFIC CONTENT DETECTION
# =============================================================================

@dataclass
class PaintingIndicator:
    """
    Indicator of painting-relevant content.

    Attributes:
        indicator_type: Type of indicator found
        description: Human-readable description
        confidence: Confidence in this indicator (0.0-1.0)
        location: Where in the image this was found (optional)
    """
    indicator_type: str
    description: str
    confidence: float
    location: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)


# Painting-relevant visual elements to look for
PAINTING_INDICATORS = {
    'room_numbers': {
        'description': 'Room numbers (e.g., "101", "ROOM 101")',
        'relevance': RelevanceLevel.PRIMARY,
        'patterns': [r'\b\d{3}\b', r'\bROOM\s*\d+\b', r'\bRM\s*\d+\b'],
    },
    'finish_callouts': {
        'description': 'Finish callouts (e.g., "PT-1", "PAINT TYPE 1")',
        'relevance': RelevanceLevel.PRIMARY,
        'patterns': [r'\bPT[-\s]?\d+\b', r'\bFIN[-\s]?\d+\b', r'\bPAINT\s+TYPE\b'],
    },
    'ceiling_grid': {
        'description': 'Ceiling grid patterns (ACT, GWB)',
        'relevance': RelevanceLevel.PRIMARY,
        'patterns': [r'\bACT\b', r'\bGWB\b', r'\bCEILING\b', r'\b2[\'x]4\b'],
    },
    'wall_tags': {
        'description': 'Wall type tags',
        'relevance': RelevanceLevel.SECONDARY,
        'patterns': [r'\bWALL\s+TYPE\b', r'\bWT[-\s]?\d+\b'],
    },
    'elevation_markers': {
        'description': 'Interior elevation markers (circles with numbers)',
        'relevance': RelevanceLevel.SECONDARY,
        'patterns': [r'\bINT\.?\s*ELEV\b', r'\bSEE\s+ELEV\b'],
    },
    'door_tags': {
        'description': 'Door tags (e.g., "D101", "DOOR 1")',
        'relevance': RelevanceLevel.SECONDARY,
        'patterns': [r'\bD\d{2,3}\b', r'\bDOOR\s*\d+\b'],
    },
}


class PaintingRelevanceDetector:
    """
    Detects painting-relevant elements in drawing content.

    Uses OCR text and visual pattern detection to identify
    elements that indicate relevance for painting trade.
    """

    def __init__(self):
        """Initialize the detector."""
        # Compile indicator patterns
        self._compiled_patterns = {}
        for indicator_type, config in PAINTING_INDICATORS.items():
            self._compiled_patterns[indicator_type] = {
                'patterns': [re.compile(p, re.IGNORECASE) for p in config['patterns']],
                'relevance': config['relevance'],
                'description': config['description'],
            }

    def detect_from_text(self, text: str) -> List[PaintingIndicator]:
        """
        Detect painting indicators from OCR text.

        Args:
            text: OCR text from the drawing area

        Returns:
            List of PaintingIndicator objects found
        """
        indicators = []

        for indicator_type, config in self._compiled_patterns.items():
            for pattern in config['patterns']:
                matches = pattern.findall(text)
                if matches:
                    # Found matches for this indicator type
                    indicators.append(PaintingIndicator(
                        indicator_type=indicator_type,
                        description=config['description'],
                        confidence=0.80,
                    ))
                    break  # One match per indicator type is enough

        return indicators

    def calculate_relevance(
        self,
        indicators: List[PaintingIndicator],
    ) -> Tuple[RelevanceLevel, float]:
        """
        Calculate overall painting relevance from indicators.

        Args:
            indicators: List of detected painting indicators

        Returns:
            Tuple of (relevance_level, confidence)
        """
        if not indicators:
            return (RelevanceLevel.REFERENCE, 0.5)

        # Count indicators by relevance level
        primary_count = sum(
            1 for i in indicators
            if PAINTING_INDICATORS.get(i.indicator_type, {}).get('relevance') == RelevanceLevel.PRIMARY
        )
        secondary_count = sum(
            1 for i in indicators
            if PAINTING_INDICATORS.get(i.indicator_type, {}).get('relevance') == RelevanceLevel.SECONDARY
        )

        # Determine relevance level
        if primary_count >= 2:
            relevance = RelevanceLevel.PRIMARY
            confidence = min(0.95, 0.70 + 0.10 * primary_count)
        elif primary_count >= 1:
            relevance = RelevanceLevel.PRIMARY
            confidence = 0.75 + 0.05 * secondary_count
        elif secondary_count >= 2:
            relevance = RelevanceLevel.SECONDARY
            confidence = 0.70 + 0.05 * secondary_count
        elif secondary_count >= 1:
            relevance = RelevanceLevel.SECONDARY
            confidence = 0.60
        else:
            relevance = RelevanceLevel.REFERENCE
            confidence = 0.50

        return (relevance, confidence)


# =============================================================================
# CONTENT ANALYZER
# =============================================================================

class ContentAnalyzer:
    """
    Analyzes drawing content to determine painting relevance.

    Uses a combination of:
    - OCR text analysis for painting indicators
    - Visual pattern detection (future)
    - AI vision analysis (optional, for complex cases)
    """

    def __init__(self, use_vision_api: bool = False):
        """
        Initialize the content analyzer.

        Args:
            use_vision_api: Whether to use AI vision API for analysis
        """
        self.use_vision_api = use_vision_api
        self.trigger = ContentAnalysisTrigger()
        self.detector = PaintingRelevanceDetector()

    def analyze(
        self,
        drawing_image: Optional[Image.Image],
        drawing_text: Optional[str],
        classification_result: EnhancedClassificationResult,
    ) -> Optional[ClassificationSignal]:
        """
        Analyze drawing content and return a classification signal.

        Args:
            drawing_image: PIL Image of the drawing area (without title block)
            drawing_text: OCR text from the drawing area
            classification_result: Current classification result

        Returns:
            ClassificationSignal from content analysis, or None if not triggered
        """
        # Check if analysis is needed
        trigger_result = self.trigger.should_analyze(classification_result)

        if not trigger_result.should_analyze:
            return None

        # Analyze text content
        indicators = []
        if drawing_text:
            indicators = self.detector.detect_from_text(drawing_text)

        if not indicators:
            # No indicators found in text
            # Could use vision API here for visual analysis
            return None

        # Calculate relevance
        relevance, confidence = self.detector.calculate_relevance(indicators)

        # Determine category from indicators
        category = self._infer_category_from_indicators(indicators)

        return ClassificationSignal(
            source=SignalSource.CONTENT_ANALYSIS,
            category=category,
            confidence=confidence,
            relevance=relevance,
            evidence=f"Indicators: {[i.indicator_type for i in indicators]}",
            weight=1.0,
        )

    def _infer_category_from_indicators(
        self,
        indicators: List[PaintingIndicator],
    ) -> str:
        """
        Infer category from detected indicators.

        Args:
            indicators: List of painting indicators

        Returns:
            Category name
        """
        indicator_types = set(i.indicator_type for i in indicators)

        # Check for specific indicator combinations
        if 'ceiling_grid' in indicator_types:
            return 'reflected_ceiling_plans'
        elif 'room_numbers' in indicator_types and 'finish_callouts' in indicator_types:
            return 'room_finish_schedules'
        elif 'elevation_markers' in indicator_types:
            return 'interior_elevations'
        elif 'room_numbers' in indicator_types:
            return 'floor_plans'
        elif 'door_tags' in indicator_types:
            return 'door_schedules'
        else:
            return 'floor_plans'  # Default to floor plans if unsure


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def should_trigger_content_analysis(
    result: EnhancedClassificationResult,
) -> bool:
    """
    Check if content analysis should be triggered.

    Args:
        result: Classification result to check

    Returns:
        True if content analysis should be performed
    """
    trigger = ContentAnalysisTrigger()
    trigger_result = trigger.should_analyze(result)
    return trigger_result.should_analyze


def analyze_drawing_content(
    drawing_text: str,
) -> Tuple[RelevanceLevel, float, List[str]]:
    """
    Analyze drawing text for painting relevance.

    Args:
        drawing_text: OCR text from drawing area

    Returns:
        Tuple of (relevance, confidence, indicator_types)
    """
    detector = PaintingRelevanceDetector()
    indicators = detector.detect_from_text(drawing_text)
    relevance, confidence = detector.calculate_relevance(indicators)
    indicator_types = [i.indicator_type for i in indicators]

    return (relevance, confidence, indicator_types)
