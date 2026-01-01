"""
Human-in-the-Loop (HITL) Handler Module (V7.0)

Manages human review workflow for uncertain classifications.
Captures corrections and learns from human feedback.

This module provides:
    - HITLDecisionHandler: Flags sheets for review and processes corrections
    - ReviewQueue: Manages the queue of sheets needing review
    - FeedbackLearner: Learns from human corrections
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from .classification_types import (
    ClassificationDecision,
    RelevanceLevel,
    EnhancedClassificationResult,
)
from .sheet_classifier import ClassifiedSheet

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import CLASSIFICATION_THRESHOLDS


# =============================================================================
# REVIEW ITEM
# =============================================================================

@dataclass
class ReviewItem:
    """
    An item flagged for human review.

    Attributes:
        sheet: The classified sheet
        classification_result: Full classification details
        review_reason: Why this was flagged for review
        priority: Review priority (0=low, 1=medium, 2=high)
        created_at: When this was flagged
        reviewed: Whether this has been reviewed
        human_decision: Human's decision (if reviewed)
        human_categories: Human-assigned categories (if reviewed)
        human_notes: Human's notes (if reviewed)
    """
    sheet: ClassifiedSheet
    classification_result: Optional[EnhancedClassificationResult] = None
    review_reason: str = ""
    priority: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    reviewed: bool = False
    human_decision: Optional[str] = None
    human_categories: List[str] = field(default_factory=list)
    human_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'sheet': self.sheet.to_dict(),
            'review_reason': self.review_reason,
            'priority': self.priority,
            'created_at': self.created_at,
            'reviewed': self.reviewed,
            'human_decision': self.human_decision,
            'human_categories': self.human_categories,
            'human_notes': self.human_notes,
            'classification': self.classification_result.to_dict() if self.classification_result else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewItem':
        """Create from dictionary."""
        # Reconstruct ClassifiedSheet
        sheet_data = data.get('sheet', {})
        sheet = ClassifiedSheet(
            pdf_name=sheet_data.get('pdf', ''),
            page_number=sheet_data.get('page', 0),
            sheet_number=sheet_data.get('sheet_number'),
            sheet_title=sheet_data.get('sheet_title'),
            categories=sheet_data.get('categories', []),
            classification_type=sheet_data.get('classification_type', 'unclassified'),
            crop_file=sheet_data.get('crop_file'),
            number_confidence=sheet_data.get('number_confidence', 0.0),
            title_confidence=sheet_data.get('title_confidence', 0.0),
            decision=sheet_data.get('decision'),
            relevance=sheet_data.get('relevance'),
            classification_confidence=sheet_data.get('classification_confidence', 0.0),
            is_combo_page=sheet_data.get('is_combo_page', False),
            needs_review=sheet_data.get('needs_review', False),
            review_reason=sheet_data.get('review_reason'),
        )

        return cls(
            sheet=sheet,
            review_reason=data.get('review_reason', ''),
            priority=data.get('priority', 0),
            created_at=data.get('created_at', datetime.now().isoformat()),
            reviewed=data.get('reviewed', False),
            human_decision=data.get('human_decision'),
            human_categories=data.get('human_categories', []),
            human_notes=data.get('human_notes', ''),
        )


# =============================================================================
# REVIEW QUEUE
# =============================================================================

class ReviewQueue:
    """
    Manages the queue of sheets needing human review.

    Supports:
    - Adding items to the queue
    - Retrieving items by priority
    - Saving/loading queue state
    - Tracking review progress
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the review queue.

        Args:
            storage_path: Path to save queue state (optional)
        """
        self.storage_path = storage_path
        self._queue: List[ReviewItem] = []
        self._reviewed: List[ReviewItem] = []

        # Load existing queue if storage exists
        if storage_path and storage_path.exists():
            self.load()

    def add(
        self,
        sheet: ClassifiedSheet,
        classification_result: Optional[EnhancedClassificationResult] = None,
        review_reason: str = "",
        priority: int = 0,
    ) -> ReviewItem:
        """
        Add a sheet to the review queue.

        Args:
            sheet: Classified sheet to review
            classification_result: Full classification result (optional)
            review_reason: Why this needs review
            priority: Priority level (0=low, 1=medium, 2=high)

        Returns:
            Created ReviewItem
        """
        item = ReviewItem(
            sheet=sheet,
            classification_result=classification_result,
            review_reason=review_reason,
            priority=priority,
        )
        self._queue.append(item)

        # Sort by priority (highest first)
        self._queue.sort(key=lambda x: x.priority, reverse=True)

        return item

    def get_next(self) -> Optional[ReviewItem]:
        """
        Get the next item to review (highest priority).

        Returns:
            Next ReviewItem or None if queue is empty
        """
        pending = [item for item in self._queue if not item.reviewed]
        if pending:
            return pending[0]
        return None

    def get_all_pending(self) -> List[ReviewItem]:
        """Get all pending review items."""
        return [item for item in self._queue if not item.reviewed]

    def get_by_priority(self, priority: int) -> List[ReviewItem]:
        """Get pending items with a specific priority."""
        return [
            item for item in self._queue
            if not item.reviewed and item.priority == priority
        ]

    def mark_reviewed(
        self,
        item: ReviewItem,
        decision: str,
        categories: List[str],
        notes: str = "",
    ) -> None:
        """
        Mark an item as reviewed with human decision.

        Args:
            item: ReviewItem to mark
            decision: Human's decision (DEFINITELY_NEEDED, DEFINITELY_NOT_NEEDED, NEEDS_EVALUATION)
            categories: Human-assigned categories
            notes: Human's notes
        """
        item.reviewed = True
        item.human_decision = decision
        item.human_categories = categories
        item.human_notes = notes
        self._reviewed.append(item)

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pending = self.get_all_pending()
        return {
            'total': len(self._queue),
            'pending': len(pending),
            'reviewed': len(self._reviewed),
            'by_priority': {
                'high': len([i for i in pending if i.priority == 2]),
                'medium': len([i for i in pending if i.priority == 1]),
                'low': len([i for i in pending if i.priority == 0]),
            },
        }

    def save(self) -> None:
        """Save queue state to storage."""
        if not self.storage_path:
            return

        data = {
            'saved_at': datetime.now().isoformat(),
            'queue': [item.to_dict() for item in self._queue],
            'reviewed': [item.to_dict() for item in self._reviewed],
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load queue state from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        with open(self.storage_path, 'r') as f:
            data = json.load(f)

        self._queue = [ReviewItem.from_dict(item) for item in data.get('queue', [])]
        self._reviewed = [ReviewItem.from_dict(item) for item in data.get('reviewed', [])]


# =============================================================================
# HITL DECISION HANDLER
# =============================================================================

class HITLDecisionHandler:
    """
    Handles human-in-the-loop decision making.

    Provides:
    - Automatic flagging of uncertain classifications
    - Review queue management
    - Feedback capture and learning
    """

    def __init__(
        self,
        queue_storage_path: Optional[Path] = None,
        auto_flag_threshold: float = 0.70,
    ):
        """
        Initialize the HITL handler.

        Args:
            queue_storage_path: Path to save review queue
            auto_flag_threshold: Confidence threshold for auto-flagging
        """
        self.queue = ReviewQueue(storage_path=queue_storage_path)
        self.auto_flag_threshold = auto_flag_threshold

    def process_classification(
        self,
        sheet: ClassifiedSheet,
        classification_result: Optional[EnhancedClassificationResult] = None,
    ) -> bool:
        """
        Process a classification result and flag for review if needed.

        Args:
            sheet: Classified sheet
            classification_result: Full classification result

        Returns:
            True if flagged for review, False otherwise
        """
        # Check if review is needed
        if sheet.needs_review:
            priority = self._calculate_priority(sheet, classification_result)
            self.queue.add(
                sheet=sheet,
                classification_result=classification_result,
                review_reason=sheet.review_reason or "Flagged by classifier",
                priority=priority,
            )
            return True

        # Check confidence threshold
        if sheet.classification_confidence < self.auto_flag_threshold:
            priority = 2 if sheet.classification_confidence < 0.50 else 1
            self.queue.add(
                sheet=sheet,
                classification_result=classification_result,
                review_reason=f"Low confidence ({sheet.classification_confidence:.2f})",
                priority=priority,
            )
            return True

        return False

    def _calculate_priority(
        self,
        sheet: ClassifiedSheet,
        classification_result: Optional[EnhancedClassificationResult],
    ) -> int:
        """
        Calculate review priority based on sheet and classification.

        Args:
            sheet: Classified sheet
            classification_result: Full classification result

        Returns:
            Priority level (0=low, 1=medium, 2=high)
        """
        priority = 0

        # High priority: combo pages, conflicts, very low confidence
        if sheet.is_combo_page:
            priority = max(priority, 1)

        if sheet.classification_confidence < 0.50:
            priority = max(priority, 2)

        if classification_result and classification_result.conflict.has_conflict:
            priority = max(priority, 2)

        # Medium priority: no categories, uncertain relevance
        if not sheet.categories:
            priority = max(priority, 1)

        return priority

    def get_review_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the review queue.

        Returns:
            Dictionary with queue statistics and sample items
        """
        stats = self.queue.get_statistics()
        pending = self.queue.get_all_pending()

        # Get sample high-priority items
        high_priority = [
            item.to_dict() for item in pending[:5]
            if item.priority == 2
        ]

        return {
            'statistics': stats,
            'high_priority_samples': high_priority,
        }

    def submit_review(
        self,
        item: ReviewItem,
        decision: str,
        categories: List[str],
        notes: str = "",
    ) -> None:
        """
        Submit a human review for an item.

        Args:
            item: ReviewItem being reviewed
            decision: Human's decision
            categories: Human-assigned categories
            notes: Human's notes
        """
        self.queue.mark_reviewed(item, decision, categories, notes)
        self.queue.save()

    def get_corrections(self) -> List[Dict[str, Any]]:
        """
        Get all corrections (where human decision differs from original).

        Returns:
            List of correction records
        """
        corrections = []

        for item in self.queue._reviewed:
            original_decision = item.sheet.decision
            if item.human_decision and original_decision != item.human_decision:
                corrections.append({
                    'sheet_number': item.sheet.sheet_number,
                    'sheet_title': item.sheet.sheet_title,
                    'original_decision': original_decision,
                    'human_decision': item.human_decision,
                    'original_categories': item.sheet.categories,
                    'human_categories': item.human_categories,
                    'notes': item.human_notes,
                })

        return corrections


# =============================================================================
# FEEDBACK LEARNER (Future Enhancement)
# =============================================================================

class FeedbackLearner:
    """
    Learns from human corrections to improve future classifications.

    This is a placeholder for future ML-based learning from corrections.
    Currently implements simple pattern extraction from corrections.
    """

    def __init__(self):
        """Initialize the feedback learner."""
        self._learned_patterns: Dict[str, List[str]] = {}

    def learn_from_corrections(
        self,
        corrections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Learn patterns from human corrections.

        Args:
            corrections: List of correction records

        Returns:
            Dictionary with learned patterns and statistics
        """
        # Count decision changes
        decision_changes: Dict[str, int] = {}
        category_corrections: Dict[str, List[str]] = {}

        for correction in corrections:
            # Track decision changes
            change_key = f"{correction['original_decision']} -> {correction['human_decision']}"
            decision_changes[change_key] = decision_changes.get(change_key, 0) + 1

            # Track category corrections
            sheet_title = correction.get('sheet_title', '')
            if sheet_title:
                for category in correction.get('human_categories', []):
                    if category not in category_corrections:
                        category_corrections[category] = []
                    category_corrections[category].append(sheet_title)

        return {
            'total_corrections': len(corrections),
            'decision_changes': decision_changes,
            'category_corrections': {
                cat: len(titles) for cat, titles in category_corrections.items()
            },
        }

    def suggest_pattern_updates(
        self,
        corrections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Suggest pattern updates based on corrections.

        Args:
            corrections: List of correction records

        Returns:
            List of suggested pattern updates
        """
        suggestions = []

        # Group corrections by human category
        by_category: Dict[str, List[str]] = {}
        for correction in corrections:
            for category in correction.get('human_categories', []):
                if category not in by_category:
                    by_category[category] = []
                title = correction.get('sheet_title', '')
                if title:
                    by_category[category].append(title)

        # Suggest patterns for categories with multiple corrections
        for category, titles in by_category.items():
            if len(titles) >= 2:
                # Find common words/patterns
                common_words = self._find_common_words(titles)
                if common_words:
                    suggestions.append({
                        'category': category,
                        'suggested_pattern': common_words,
                        'example_titles': titles[:3],
                        'count': len(titles),
                    })

        return suggestions

    def _find_common_words(self, titles: List[str]) -> Optional[str]:
        """
        Find common words across titles.

        Args:
            titles: List of title strings

        Returns:
            Common word pattern or None
        """
        if not titles:
            return None

        # Tokenize and count words
        word_counts: Dict[str, int] = {}
        for title in titles:
            words = title.upper().split()
            for word in words:
                word_clean = ''.join(c for c in word if c.isalnum())
                if len(word_clean) >= 3:
                    word_counts[word_clean] = word_counts.get(word_clean, 0) + 1

        # Find words that appear in majority of titles
        threshold = len(titles) * 0.6
        common = [word for word, count in word_counts.items() if count >= threshold]

        if common:
            return r'\b' + r'\b.*\b'.join(common) + r'\b'

        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_hitl_handler(
    output_dir: Optional[Path] = None,
) -> HITLDecisionHandler:
    """
    Create a HITL handler with default settings.

    Args:
        output_dir: Directory for storing review queue

    Returns:
        Configured HITLDecisionHandler
    """
    queue_path = None
    if output_dir:
        queue_path = output_dir / 'review_queue.json'

    return HITLDecisionHandler(
        queue_storage_path=queue_path,
        auto_flag_threshold=CLASSIFICATION_THRESHOLDS.get('needs_evaluation', 0.70),
    )
