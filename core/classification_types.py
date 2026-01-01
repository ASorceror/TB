"""
Classification Types Module (V7.0)

Type definitions for the enhanced blueprint page classification system.
Provides enums and dataclasses for structured classification results.

This module defines:
    - ClassificationDecision: Three-tier classification output
    - RelevanceLevel: Painting trade relevance levels
    - SignalSource: Where classification signals originate
    - TitleComponent: Parsed title parts (for combo pages)
    - ClassificationSignal: Individual classification signal
    - EnhancedClassificationResult: Complete classification output
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


class ClassificationDecision(Enum):
    """
    Three-tier classification decision for painting trade relevance.

    DEFINITELY_NEEDED: Auto-include (confidence >= 0.85, relevance PRIMARY/SECONDARY)
    DEFINITELY_NOT_NEEDED: Auto-exclude (confidence >= 0.85, relevance IRRELEVANT)
    NEEDS_EVALUATION: Human review (confidence < 0.85 OR conflicts)
    """
    DEFINITELY_NEEDED = auto()
    DEFINITELY_NOT_NEEDED = auto()
    NEEDS_EVALUATION = auto()


class RelevanceLevel(Enum):
    """
    Relevance levels for painting trade scope of work.

    PRIMARY: Essential documents (floor plans, RCPs, finish schedules)
    SECONDARY: Important reference (building sections, exterior elevations)
    REFERENCE: Useful context (cover sheets, general notes)
    IRRELEVANT: Not needed (structural, MEP, civil, landscape)
    """
    PRIMARY = auto()
    SECONDARY = auto()
    REFERENCE = auto()
    IRRELEVANT = auto()


class SignalSource(Enum):
    """
    Sources of classification signals, ordered by reliability.

    DRAWING_INDEX: From PDF cover sheet - most authoritative
    TITLE_PATTERN: Explicit category keywords in title
    SHEET_NUMBER: NCS discipline/type inference
    VISION_API: Visual content analysis
    CONTENT_ANALYSIS: Inferred from drawing features
    """
    DRAWING_INDEX = "drawing_index"
    TITLE_PATTERN = "title_pattern"
    SHEET_NUMBER = "sheet_number"
    VISION_API = "vision_api"
    CONTENT_ANALYSIS = "content_analysis"


@dataclass
class TitleComponent:
    """
    A single component of a potentially multi-part title.

    Used when titles contain multiple drawing types on one sheet
    (e.g., "FLOOR PLAN\nROOF PLAN" or "FLOOR PLAN AND REFLECTED CEILING PLAN").

    Attributes:
        text: The title text for this component
        drawing_type: Detected drawing type (floor_plan, rcp, etc.)
        confidence: Confidence in this component's classification
        original_position: Position in the original title (0-indexed)
    """
    text: str
    drawing_type: Optional[str] = None
    confidence: float = 0.0
    original_position: int = 0


@dataclass
class ClassificationSignal:
    """
    An individual classification signal from any source.

    Signals are aggregated to determine final classification.
    Multiple signals can suggest the same or different categories.

    Attributes:
        source: Where this signal came from (SignalSource)
        category: Suggested category (e.g., 'floor_plans', 'reflected_ceiling_plans')
        confidence: Confidence in this signal (0.0-1.0)
        relevance: Painting trade relevance level
        evidence: Supporting evidence/matched pattern
        weight: Signal weight for aggregation (set by confidence calculator)
    """
    source: SignalSource
    category: str
    confidence: float
    relevance: RelevanceLevel = RelevanceLevel.REFERENCE
    evidence: Optional[str] = None
    weight: float = 1.0

    def weighted_score(self) -> float:
        """Calculate weighted score for aggregation."""
        return self.confidence * self.weight


@dataclass
class ConflictInfo:
    """
    Information about conflicting classification signals.

    Attributes:
        has_conflict: Whether a conflict was detected
        conflicting_categories: Categories that conflict
        conflict_severity: How severe the conflict is (0.0-1.0)
        resolution_method: How the conflict was resolved
    """
    has_conflict: bool = False
    conflicting_categories: List[str] = field(default_factory=list)
    conflict_severity: float = 0.0
    resolution_method: Optional[str] = None


@dataclass
class EnhancedClassificationResult:
    """
    Complete classification result with all metadata.

    Includes the final decision, all contributing signals,
    confidence breakdown, and conflict information.

    Attributes:
        sheet_number: Original sheet number
        sheet_title: Original sheet title
        decision: Final three-tier decision
        categories: List of matched categories (primary first)
        relevance: Overall painting trade relevance
        confidence: Final aggregated confidence (0.0-1.0)
        signals: All classification signals that contributed
        title_components: Parsed title parts (for combo pages)
        is_combo_page: Whether this is a multi-drawing page
        conflict: Conflict information if signals disagreed
        needs_human_review: Whether human review is recommended
        review_reason: Why human review is needed (if applicable)
        metadata: Additional metadata (NCS parsing, etc.)
    """
    sheet_number: Optional[str]
    sheet_title: Optional[str]
    decision: ClassificationDecision
    categories: List[str] = field(default_factory=list)
    relevance: RelevanceLevel = RelevanceLevel.REFERENCE
    confidence: float = 0.0
    signals: List[ClassificationSignal] = field(default_factory=list)
    title_components: List[TitleComponent] = field(default_factory=list)
    is_combo_page: bool = False
    conflict: ConflictInfo = field(default_factory=ConflictInfo)
    needs_human_review: bool = False
    review_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'sheet_number': self.sheet_number,
            'sheet_title': self.sheet_title,
            'decision': self.decision.name,
            'categories': self.categories,
            'relevance': self.relevance.name,
            'confidence': self.confidence,
            'is_combo_page': self.is_combo_page,
            'needs_human_review': self.needs_human_review,
            'review_reason': self.review_reason,
            'signals': [
                {
                    'source': s.source.value,
                    'category': s.category,
                    'confidence': s.confidence,
                    'relevance': s.relevance.name,
                    'evidence': s.evidence,
                }
                for s in self.signals
            ],
            'title_components': [
                {
                    'text': tc.text,
                    'drawing_type': tc.drawing_type,
                    'confidence': tc.confidence,
                }
                for tc in self.title_components
            ],
            'conflict': {
                'has_conflict': self.conflict.has_conflict,
                'conflicting_categories': self.conflict.conflicting_categories,
                'conflict_severity': self.conflict.conflict_severity,
                'resolution_method': self.conflict.resolution_method,
            },
            'metadata': self.metadata,
        }

    @property
    def primary_category(self) -> Optional[str]:
        """Get the primary (first) category if any."""
        return self.categories[0] if self.categories else None

    @property
    def is_painting_relevant(self) -> bool:
        """Check if this page is relevant for painting trade."""
        return self.relevance in (RelevanceLevel.PRIMARY, RelevanceLevel.SECONDARY)


# Type aliases for convenience
Categories = List[str]
Signals = List[ClassificationSignal]
