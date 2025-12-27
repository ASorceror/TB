"""
Blueprint Processor V4.3 - Template Applier Module
Apply learned template to rescue failed extractions.

This module implements Phase D of V4.3: Template Application
"""

import re
from typing import List, Dict, Optional, Any

from core.template_types import Template

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import QUALITY_KEYWORDS


class TemplateApplier:
    """Apply templates to extract titles from failed pages."""

    def __init__(self):
        """Initialize the template applier."""
        # Create set of keywords for fast lookup (uppercase for case-insensitive)
        self._keywords = set(kw.upper() for kw in QUALITY_KEYWORDS)

    def apply(
        self,
        template: Template,
        page_text: str,
        text_blocks: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Apply learned template to extract a title.

        Args:
            template: Learned template with extraction parameters
            page_text: Full text content of the page
            text_blocks: Optional list of text blocks with positions

        Returns:
            Dict with 'title', 'confidence', and 'method' keys
        """
        # Compile exclusion patterns
        exclusion_patterns = [
            re.compile(p, re.IGNORECASE) for p in template.exclusion_patterns
        ]

        # Try text blocks first if available (vector page)
        if text_blocks:
            result = self._extract_from_blocks(
                text_blocks, template, exclusion_patterns
            )
            if result['title']:
                return result

        # Fall back to page text
        result = self._extract_from_text(
            page_text, template, exclusion_patterns
        )
        return result

    def _extract_from_blocks(
        self,
        text_blocks: List[Dict],
        template: Template,
        exclusion_patterns: List[re.Pattern]
    ) -> Dict[str, Any]:
        """
        Extract title from text blocks (vector PDF).

        Args:
            text_blocks: List of text block dicts
            template: Template with parameters
            exclusion_patterns: Compiled exclusion patterns

        Returns:
            Extraction result dict
        """
        candidates = []

        for block in text_blocks:
            text = block.get('text', '').strip()
            if not text:
                continue

            # Check if valid candidate
            if not self._is_valid_candidate(text, template, exclusion_patterns):
                continue

            # Score the candidate
            score = self._score_candidate(text, block)
            candidates.append((text, score, block))

        if not candidates:
            return {"title": None, "confidence": 0.0, "method": "template"}

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        best_title = candidates[0][0]
        return {"title": best_title, "confidence": 0.85, "method": "template"}

    def _extract_from_text(
        self,
        page_text: str,
        template: Template,
        exclusion_patterns: List[re.Pattern]
    ) -> Dict[str, Any]:
        """
        Extract title from page text (scanned PDF or fallback).

        Args:
            page_text: Full page text
            template: Template with parameters
            exclusion_patterns: Compiled exclusion patterns

        Returns:
            Extraction result dict
        """
        if not page_text:
            return {"title": None, "confidence": 0.0, "method": "template"}

        # Split into lines
        lines = page_text.strip().split('\n')

        # Find first valid line with keyword preference
        keyword_lines = []
        other_valid_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if not self._is_valid_candidate(line, template, exclusion_patterns):
                continue

            # Check if contains keyword
            has_keyword = any(kw in line.upper() for kw in self._keywords)
            if has_keyword:
                keyword_lines.append(line)
            else:
                other_valid_lines.append(line)

        # Prefer lines with keywords
        if keyword_lines:
            return {"title": keyword_lines[0], "confidence": 0.75, "method": "template"}
        elif other_valid_lines:
            return {"title": other_valid_lines[0], "confidence": 0.65, "method": "template"}
        else:
            return {"title": None, "confidence": 0.0, "method": "template"}

    def _is_valid_candidate(
        self,
        text: str,
        template: Template,
        exclusion_patterns: List[re.Pattern]
    ) -> bool:
        """
        Check if text is a valid title candidate.

        Args:
            text: Text to validate
            template: Template with length bounds
            exclusion_patterns: Patterns to reject

        Returns:
            True if valid candidate, False otherwise
        """
        if not text:
            return False

        # Check exclusion patterns
        for pattern in exclusion_patterns:
            if pattern.search(text):
                return False

        # Check length bounds with tolerance
        min_len = int(template.typical_length_min * 0.5)
        max_len = int(template.typical_length_max * 1.5)

        if len(text) < min_len or len(text) > max_len:
            return False

        # Reject purely numeric
        if text.isdigit():
            return False

        # Reject sheet number patterns
        if re.match(r'^[A-Z]{1,2}[-.]?\d{1,4}(\.\d{1,2})?$', text, re.IGNORECASE):
            return False

        return True

    def _score_candidate(self, text: str, block: Dict) -> float:
        """
        Score a candidate for ranking.

        Higher score = better candidate.

        Args:
            text: Text content
            block: Block dict with font_size etc.

        Returns:
            Score value
        """
        score = 0.0

        # Prefer larger font size
        font_size = block.get('font_size', 0)
        if font_size > 0:
            score += min(font_size / 20.0, 1.0)  # Normalize to 0-1

        # Prefer text with quality keywords
        has_keyword = any(kw in text.upper() for kw in self._keywords)
        if has_keyword:
            score += 2.0

        # Prefer reasonable length (not too short or too long)
        length = len(text)
        if 15 <= length <= 40:
            score += 1.0
        elif 10 <= length <= 50:
            score += 0.5

        return score
