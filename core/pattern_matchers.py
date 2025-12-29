"""
Pattern Matchers Module (V6.0)

Pluggable pattern matching engine for sheet classification.
Supports regex patterns on titles and sheet numbers with future
extensibility for ML/AI-based matching.

Classes:
    PatternMatcher: Abstract base class
    RegexPatternMatcher: Regex-based matching on titles
    SheetNumberMatcher: Pattern matching on sheet numbers
    CompositeMatcher: Combines multiple matchers
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import SHEET_CATEGORIES, UNCLASSIFIED_CATEGORIES


@dataclass
class MatchResult:
    """Result of a pattern match attempt."""
    matched: bool
    category: Optional[str] = None
    confidence: float = 0.0
    match_type: str = 'none'  # 'title', 'sheet_number', 'composite'
    matched_pattern: Optional[str] = None


class PatternMatcher(ABC):
    """Abstract base class for pattern matchers."""

    @abstractmethod
    def match(self, sheet_number: Optional[str], sheet_title: Optional[str]) -> List[MatchResult]:
        """
        Attempt to match sheet data against patterns.

        Args:
            sheet_number: Extracted sheet number (e.g., "A101")
            sheet_title: Extracted sheet title (e.g., "First Floor Plan")

        Returns:
            List of MatchResult for all matching categories
        """
        pass


class RegexPatternMatcher(PatternMatcher):
    """Regex-based pattern matcher for sheet titles."""

    def __init__(self, categories: Optional[Dict] = None):
        """
        Initialize with category patterns.

        Args:
            categories: Dict of category configs. Defaults to SHEET_CATEGORIES.
        """
        self.categories = categories or SHEET_CATEGORIES
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile all regex patterns for performance."""
        for category, config in self.categories.items():
            patterns = []
            for pattern_str in config.get('title_patterns', []):
                try:
                    compiled = re.compile(pattern_str, re.IGNORECASE)
                    patterns.append((compiled, pattern_str))
                except re.error as e:
                    print(f"Warning: Invalid regex pattern '{pattern_str}': {e}")
            self._compiled_patterns[category] = patterns

    def match(self, sheet_number: Optional[str], sheet_title: Optional[str]) -> List[MatchResult]:
        """Match sheet title against category patterns."""
        results = []

        if not sheet_title:
            return results

        title_clean = sheet_title.strip()

        for category, patterns in self._compiled_patterns.items():
            for compiled_pattern, pattern_str in patterns:
                if compiled_pattern.search(title_clean):
                    results.append(MatchResult(
                        matched=True,
                        category=category,
                        confidence=0.90,
                        match_type='title',
                        matched_pattern=pattern_str
                    ))
                    break  # One match per category is enough

        return results


class SheetNumberMatcher(PatternMatcher):
    """Pattern matcher for sheet numbers (AIA/CSI conventions)."""

    def __init__(self, categories: Optional[Dict] = None):
        """
        Initialize with category patterns.

        Args:
            categories: Dict of category configs. Defaults to SHEET_CATEGORIES.
        """
        self.categories = categories or SHEET_CATEGORIES
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile all regex patterns for performance."""
        for category, config in self.categories.items():
            patterns = []
            for pattern_str in config.get('sheet_patterns', []):
                try:
                    compiled = re.compile(pattern_str, re.IGNORECASE)
                    patterns.append((compiled, pattern_str))
                except re.error as e:
                    print(f"Warning: Invalid regex pattern '{pattern_str}': {e}")
            self._compiled_patterns[category] = patterns

    def match(self, sheet_number: Optional[str], sheet_title: Optional[str]) -> List[MatchResult]:
        """Match sheet number against category patterns."""
        results = []

        if not sheet_number:
            return results

        number_clean = sheet_number.strip().upper()

        for category, patterns in self._compiled_patterns.items():
            for compiled_pattern, pattern_str in patterns:
                if compiled_pattern.match(number_clean):
                    results.append(MatchResult(
                        matched=True,
                        category=category,
                        confidence=0.75,  # Lower than title match
                        match_type='sheet_number',
                        matched_pattern=pattern_str
                    ))
                    break  # One match per category is enough

        return results


class UnclassifiedMatcher(PatternMatcher):
    """Matcher for recognized but not-needed sheet types."""

    def __init__(self):
        """Initialize with unclassified category patterns."""
        self.categories = UNCLASSIFIED_CATEGORIES
        self._title_patterns: List[Tuple[re.Pattern, str]] = []
        self._sheet_patterns: List[Tuple[re.Pattern, str]] = []
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile all regex patterns."""
        config = self.categories.get('properly_classified_not_needed', {})

        for pattern_str in config.get('title_patterns', []):
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                self._title_patterns.append((compiled, pattern_str))
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{pattern_str}': {e}")

        for pattern_str in config.get('sheet_patterns', []):
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                self._sheet_patterns.append((compiled, pattern_str))
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{pattern_str}': {e}")

    def match(self, sheet_number: Optional[str], sheet_title: Optional[str]) -> List[MatchResult]:
        """Check if sheet is a recognized but not-needed type."""
        results = []

        # Check title patterns
        if sheet_title:
            title_clean = sheet_title.strip()
            for compiled_pattern, pattern_str in self._title_patterns:
                if compiled_pattern.search(title_clean):
                    results.append(MatchResult(
                        matched=True,
                        category='properly_classified_not_needed',
                        confidence=0.85,
                        match_type='title',
                        matched_pattern=pattern_str
                    ))
                    return results  # One match is enough

        # Check sheet number patterns
        if sheet_number:
            number_clean = sheet_number.strip().upper()
            for compiled_pattern, pattern_str in self._sheet_patterns:
                if compiled_pattern.match(number_clean):
                    results.append(MatchResult(
                        matched=True,
                        category='properly_classified_not_needed',
                        confidence=0.70,
                        match_type='sheet_number',
                        matched_pattern=pattern_str
                    ))
                    return results

        return results


class CompositeMatcher(PatternMatcher):
    """Combines multiple matchers with priority ordering."""

    def __init__(self, matchers: Optional[List[PatternMatcher]] = None):
        """
        Initialize with list of matchers.

        Args:
            matchers: List of PatternMatcher instances. If None, uses defaults.
        """
        if matchers is None:
            # Default matcher chain: title patterns first, then sheet numbers
            self.matchers = [
                RegexPatternMatcher(),
                SheetNumberMatcher(),
            ]
        else:
            self.matchers = matchers

        # Unclassified matcher for fallback
        self.unclassified_matcher = UnclassifiedMatcher()

    def match(self, sheet_number: Optional[str], sheet_title: Optional[str]) -> List[MatchResult]:
        """
        Run all matchers and combine results.

        Higher-confidence matches from title patterns take precedence.
        Multiple categories can be returned if both match.
        """
        all_results: Dict[str, MatchResult] = {}

        for matcher in self.matchers:
            results = matcher.match(sheet_number, sheet_title)
            for result in results:
                if result.category:
                    # Keep highest confidence match per category
                    existing = all_results.get(result.category)
                    if not existing or result.confidence > existing.confidence:
                        all_results[result.category] = result

        return list(all_results.values())

    def classify(self, sheet_number: Optional[str], sheet_title: Optional[str]) -> Tuple[List[str], str]:
        """
        Classify a sheet and return categories.

        Args:
            sheet_number: Extracted sheet number
            sheet_title: Extracted sheet title

        Returns:
            Tuple of (list of category names, classification_type)
            classification_type is one of: 'matched', 'not_needed', 'unclassified'
        """
        # Try main category matchers first
        results = self.match(sheet_number, sheet_title)
        if results:
            categories = [r.category for r in results if r.category]
            return (categories, 'matched')

        # Try unclassified/not-needed matcher
        unclassified_results = self.unclassified_matcher.match(sheet_number, sheet_title)
        if unclassified_results:
            return (['properly_classified_not_needed'], 'not_needed')

        # Truly unclassified
        return (['unclassified_may_be_needed'], 'unclassified')


def get_default_matcher() -> CompositeMatcher:
    """Factory function to get the default composite matcher."""
    return CompositeMatcher()
