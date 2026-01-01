"""
Confidence Calculation Module (V7.0)

Calculates final classification confidence by aggregating signals
from multiple sources with weighted scoring and conflict detection.

This module provides:
    - calculate_final_confidence: Aggregate signals into final score
    - detect_conflicts: Identify conflicting signals
    - aggregate_signals: Combine signals by category
    - determine_decision: Map confidence to three-tier decision
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .classification_types import (
    ClassificationSignal,
    ClassificationDecision,
    RelevanceLevel,
    ConflictInfo,
    SignalSource,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    CLASSIFICATION_THRESHOLDS,
    SIGNAL_WEIGHTS,
    PAINTING_RELEVANCE,
)


# =============================================================================
# CONFIDENCE CALCULATION
# =============================================================================

@dataclass
class AggregationResult:
    """
    Result of signal aggregation.

    Attributes:
        category_scores: Weighted scores by category
        top_category: Highest-scoring category
        top_score: Score of top category
        runner_up_category: Second highest category (for conflict detection)
        runner_up_score: Score of runner-up
        total_signals: Number of signals aggregated
    """
    category_scores: Dict[str, float]
    top_category: Optional[str]
    top_score: float
    runner_up_category: Optional[str]
    runner_up_score: float
    total_signals: int


def get_signal_weight(source: SignalSource) -> float:
    """
    Get the weight for a signal source.

    Args:
        source: SignalSource enum value

    Returns:
        Weight value (0.0-1.0)
    """
    source_key = source.value
    return SIGNAL_WEIGHTS.get(source_key, 0.25)


def aggregate_signals(signals: List[ClassificationSignal]) -> AggregationResult:
    """
    Aggregate classification signals by category.

    Combines signals from different sources using weighted scoring.
    Each signal contributes: confidence * weight * source_weight.

    Args:
        signals: List of ClassificationSignal objects

    Returns:
        AggregationResult with category scores and rankings
    """
    if not signals:
        return AggregationResult(
            category_scores={},
            top_category=None,
            top_score=0.0,
            runner_up_category=None,
            runner_up_score=0.0,
            total_signals=0,
        )

    # Aggregate scores by category
    category_scores: Dict[str, float] = defaultdict(float)
    category_counts: Dict[str, int] = defaultdict(int)

    for signal in signals:
        if not signal.category:
            continue

        # Get source weight
        source_weight = get_signal_weight(signal.source)

        # Calculate weighted score
        weighted_score = signal.confidence * signal.weight * source_weight

        category_scores[signal.category] += weighted_score
        category_counts[signal.category] += 1

    # Normalize scores (optional - could divide by count for average)
    # For now, keep cumulative scores to reward multiple confirming signals

    # Sort categories by score
    sorted_categories = sorted(
        category_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Extract top and runner-up
    top_category = sorted_categories[0][0] if sorted_categories else None
    top_score = sorted_categories[0][1] if sorted_categories else 0.0

    runner_up_category = sorted_categories[1][0] if len(sorted_categories) > 1 else None
    runner_up_score = sorted_categories[1][1] if len(sorted_categories) > 1 else 0.0

    return AggregationResult(
        category_scores=dict(category_scores),
        top_category=top_category,
        top_score=top_score,
        runner_up_category=runner_up_category,
        runner_up_score=runner_up_score,
        total_signals=len(signals),
    )


def detect_conflicts(aggregation: AggregationResult) -> ConflictInfo:
    """
    Detect conflicts between classification signals.

    A conflict exists when the top two categories are within
    a threshold of each other (close scores indicate ambiguity).

    Args:
        aggregation: Result from aggregate_signals()

    Returns:
        ConflictInfo with conflict details
    """
    if not aggregation.top_category:
        return ConflictInfo(has_conflict=False)

    if not aggregation.runner_up_category:
        return ConflictInfo(has_conflict=False)

    # Calculate score difference
    if aggregation.top_score == 0:
        return ConflictInfo(has_conflict=False)

    # Conflict threshold: runner-up within 50% of top score (more sensitive)
    conflict_threshold = 0.50
    score_ratio = aggregation.runner_up_score / aggregation.top_score

    if score_ratio >= (1.0 - conflict_threshold):
        # Significant conflict detected
        severity = score_ratio  # Higher ratio = more severe conflict

        return ConflictInfo(
            has_conflict=True,
            conflicting_categories=[
                aggregation.top_category,
                aggregation.runner_up_category,
            ],
            conflict_severity=severity,
            resolution_method='weighted_preference',
        )

    return ConflictInfo(has_conflict=False)


def calculate_final_confidence(
    signals: List[ClassificationSignal],
    is_combo_page: bool = False,
) -> Tuple[float, ConflictInfo]:
    """
    Calculate final classification confidence from all signals.

    Aggregates signals, applies penalties for conflicts and combo pages,
    and returns a normalized confidence score (0.0-1.0).

    Args:
        signals: List of ClassificationSignal objects
        is_combo_page: Whether this is a multi-drawing page

    Returns:
        Tuple of (final_confidence, conflict_info)
    """
    # Aggregate signals
    aggregation = aggregate_signals(signals)

    if not aggregation.top_category:
        return (0.0, ConflictInfo(has_conflict=False))

    # Start with top score (normalize to 0.0-1.0 range)
    # Max possible score is roughly: 1.0 * 1.0 * 0.50 = 0.50 per signal
    # Multiple confirming signals increase the score
    # Use a lower denominator for better scaling when single signal
    base_confidence = min(aggregation.top_score / 0.25, 1.0)

    # Detect conflicts
    conflict = detect_conflicts(aggregation)

    # Apply penalties
    penalties = 0.0

    # Conflict penalty
    if conflict.has_conflict:
        conflict_penalty = CLASSIFICATION_THRESHOLDS.get('conflict_penalty', 0.15)
        penalties += conflict_penalty * conflict.conflict_severity

    # Combo page penalty
    if is_combo_page:
        combo_penalty = CLASSIFICATION_THRESHOLDS.get('combo_page_penalty', 0.10)
        penalties += combo_penalty

    # Apply penalties
    final_confidence = max(0.0, base_confidence - penalties)

    # Boost for multiple confirming signals
    if aggregation.total_signals >= 3:
        confirmation_bonus = 0.05 * (aggregation.total_signals - 2)
        final_confidence = min(1.0, final_confidence + confirmation_bonus)

    return (final_confidence, conflict)


def determine_decision(
    confidence: float,
    relevance: RelevanceLevel,
    has_conflict: bool = False,
) -> ClassificationDecision:
    """
    Determine the three-tier classification decision.

    Maps confidence and relevance to:
    - DEFINITELY_NEEDED
    - DEFINITELY_NOT_NEEDED
    - NEEDS_EVALUATION

    Args:
        confidence: Final confidence score (0.0-1.0)
        relevance: Painting trade relevance level
        has_conflict: Whether signals conflicted

    Returns:
        ClassificationDecision enum value
    """
    needed_threshold = CLASSIFICATION_THRESHOLDS.get('definitely_needed', 0.85)
    not_needed_threshold = CLASSIFICATION_THRESHOLDS.get('definitely_not_needed', 0.85)

    # If there's a conflict, always needs evaluation
    if has_conflict:
        return ClassificationDecision.NEEDS_EVALUATION

    # IRRELEVANT with high confidence -> DEFINITELY_NOT_NEEDED
    if relevance == RelevanceLevel.IRRELEVANT:
        if confidence >= not_needed_threshold:
            return ClassificationDecision.DEFINITELY_NOT_NEEDED
        else:
            return ClassificationDecision.NEEDS_EVALUATION

    # PRIMARY/SECONDARY with high confidence -> DEFINITELY_NEEDED
    if relevance in (RelevanceLevel.PRIMARY, RelevanceLevel.SECONDARY):
        if confidence >= needed_threshold:
            return ClassificationDecision.DEFINITELY_NEEDED
        else:
            return ClassificationDecision.NEEDS_EVALUATION

    # REFERENCE relevance
    if relevance == RelevanceLevel.REFERENCE:
        if confidence >= needed_threshold:
            # High confidence reference -> might still be needed
            return ClassificationDecision.NEEDS_EVALUATION
        else:
            return ClassificationDecision.NEEDS_EVALUATION

    # Default to needs evaluation
    return ClassificationDecision.NEEDS_EVALUATION


def get_relevance_for_category(category: Optional[str]) -> RelevanceLevel:
    """
    Get the painting trade relevance for a category.

    Args:
        category: Category name

    Returns:
        RelevanceLevel enum value
    """
    if not category:
        return RelevanceLevel.REFERENCE

    relevance_str = PAINTING_RELEVANCE.get(category, 'REFERENCE')

    if relevance_str == 'PRIMARY':
        return RelevanceLevel.PRIMARY
    elif relevance_str == 'SECONDARY':
        return RelevanceLevel.SECONDARY
    elif relevance_str == 'IRRELEVANT':
        return RelevanceLevel.IRRELEVANT
    else:
        return RelevanceLevel.REFERENCE


def needs_human_review(
    decision: ClassificationDecision,
    confidence: float,
    has_conflict: bool = False,
    is_combo_page: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Determine if human review is needed.

    Args:
        decision: The classification decision
        confidence: Final confidence score
        has_conflict: Whether signals conflicted
        is_combo_page: Whether this is a multi-drawing page

    Returns:
        Tuple of (needs_review, reason)
    """
    review_threshold = CLASSIFICATION_THRESHOLDS.get('needs_evaluation', 0.70)

    if decision == ClassificationDecision.NEEDS_EVALUATION:
        if has_conflict:
            return (True, "Conflicting classification signals")
        elif confidence < review_threshold:
            return (True, f"Low confidence ({confidence:.2f})")
        elif is_combo_page:
            return (True, "Combo page with multiple drawings")
        else:
            return (True, "Classification uncertain")

    # Even high-confidence decisions might need review in edge cases
    if confidence < review_threshold:
        return (True, f"Confidence below threshold ({confidence:.2f})")

    return (False, None)
