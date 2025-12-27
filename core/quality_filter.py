"""
Blueprint Processor V4.3 - Quality Filter Module
Identifies truly correct extractions suitable for template learning.

This module implements Phase A of V4.3: Quality Filter
"""

import re
from typing import List, Dict, Tuple, Any
from collections import Counter

# Import from parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import QUALITY_KEYWORDS, QUALITY_THRESHOLDS, GARBAGE_PATTERNS


class QualityFilter:
    """Filter to identify truly correct extractions for template learning."""

    def __init__(self):
        """Initialize the quality filter with compiled patterns."""
        # Compile garbage patterns for efficiency
        self._garbage_patterns = [re.compile(p, re.IGNORECASE) for p in GARBAGE_PATTERNS]

        # Create set of keywords for fast lookup (uppercase for case-insensitive)
        self._keywords = set(kw.upper() for kw in QUALITY_KEYWORDS)

        # Thresholds
        self._min_length = QUALITY_THRESHOLDS['min_title_length']
        self._min_word_count = QUALITY_THRESHOLDS['min_word_count']
        self._min_length_if_one_word = QUALITY_THRESHOLDS['min_length_if_one_word']
        self._max_repetition_percent = QUALITY_THRESHOLDS['max_repetition_percent']
        self._min_quality_for_learning = QUALITY_THRESHOLDS['min_quality_for_learning']

    def is_quality_title(self, title: str) -> Tuple[bool, str]:
        """
        Determine if a title is a quality extraction suitable for learning.

        Args:
            title: The extracted sheet title

        Returns:
            Tuple of (is_quality: bool, reason: str)
        """
        # Rule 1: Reject if empty or None
        if not title or not isinstance(title, str):
            return (False, "empty")

        title = title.strip()
        if not title:
            return (False, "empty")

        # Rule 2: Reject if too short (< 5 characters)
        if len(title) < self._min_length:
            return (False, "too_short")

        # Rule 3: Reject if matches GARBAGE_PATTERNS
        for pattern in self._garbage_patterns:
            if pattern.search(title):
                return (False, "garbage_pattern")

        # Parse title for word analysis
        words = title.split()
        word_count = len(words)
        title_upper = title.upper()

        # Check for keyword presence
        has_keyword = any(kw in title_upper for kw in self._keywords)

        # Rule 4: Reject if is a single keyword alone
        if word_count == 1:
            # Check if it's exactly a keyword (single keyword alone)
            if title_upper in self._keywords:
                return (False, "single_keyword")

        # Rule 5: Accept if contains keyword AND has 2+ words
        if has_keyword and word_count >= self._min_word_count:
            return (True, "keyword_with_context")

        # Rule 6: Accept if contains keyword AND length >= 15
        if has_keyword and len(title) >= self._min_length_if_one_word:
            return (True, "keyword_long_enough")

        # Rule 7: Reject everything else
        return (False, "no_keyword_or_context")

    def filter_for_learning(self, extractions: List[Dict]) -> List[Dict]:
        """
        Filter a list of extractions to only quality ones suitable for learning.

        Args:
            extractions: List of extraction dicts with 'sheet_title' and 'needs_review' keys

        Returns:
            Filtered list containing only quality extractions
        """
        if not extractions:
            return []

        quality_extractions = []
        title_counts: Dict[str, int] = Counter()
        total_pages = len(extractions)

        # First pass: identify quality titles and count frequencies
        for ext in extractions:
            title = ext.get('sheet_title', '')
            needs_review = ext.get('needs_review', 0)

            # Skip if flagged for review
            if needs_review in [True, 1, '1']:
                continue

            # Check quality
            is_quality, _ = self.is_quality_title(title)
            if is_quality:
                quality_extractions.append(ext)
                if title:
                    title_counts[title] += 1

        # Second pass: exclude overly repeated titles (>20% of total pages)
        max_allowed = total_pages * self._max_repetition_percent

        filtered = []
        for ext in quality_extractions:
            title = ext.get('sheet_title', '')
            if title and title_counts.get(title, 0) > max_allowed:
                # This title appears too often - likely wrong
                continue
            filtered.append(ext)

        return filtered
