"""
Blueprint Processor V4.3 - Template Learner Module
Learn a template from quality extractions in a PDF.

This module implements Phase C of V4.3: Template Learning
"""

import re
from typing import List, Dict, Optional, Any
from collections import Counter

from core.quality_filter import QualityFilter
from core.template_types import Template

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    QUALITY_THRESHOLDS,
    TEMPLATE_DEFAULTS,
    STANDARD_EXCLUSION_PATTERNS
)


class TemplateLearner:
    """Learn templates from quality extractions."""

    def __init__(self):
        """Initialize the template learner."""
        self.quality_filter = QualityFilter()
        self._min_quality_for_learning = QUALITY_THRESHOLDS['min_quality_for_learning']

    def learn(
        self,
        pdf_hash: str,
        pdf_filename: str,
        extractions: List[Dict]
    ) -> Optional[Template]:
        """
        Learn a template from quality extractions in a PDF.

        Args:
            pdf_hash: First 16 chars of SHA-256 hash of PDF
            pdf_filename: Original filename of PDF
            extractions: List of extraction dicts from processing

        Returns:
            Template if enough quality pages, None otherwise
        """
        if not extractions:
            return None

        # Step 1: Filter to quality extractions
        quality_extractions = self.quality_filter.filter_for_learning(extractions)

        # If fewer than minimum required, cannot learn
        if len(quality_extractions) < self._min_quality_for_learning:
            return None

        total_count = len(extractions)
        quality_count = len(quality_extractions)

        # Step 2: Determine title block location (use defaults for now)
        title_block_bbox = TEMPLATE_DEFAULTS['title_block_bbox'].copy()

        # Step 3: Determine title zone (use defaults for now)
        title_zone_bbox = TEMPLATE_DEFAULTS['title_zone_bbox'].copy()

        # Step 4: Identify exclusion patterns
        exclusion_patterns = self._build_exclusion_patterns(extractions)

        # Step 5: Calculate length statistics
        quality_titles = [
            ext.get('sheet_title', '') for ext in quality_extractions
            if ext.get('sheet_title')
        ]
        if quality_titles:
            typical_length_min = min(len(t) for t in quality_titles)
            typical_length_max = max(len(t) for t in quality_titles)
        else:
            typical_length_min = 10
            typical_length_max = 50

        # Step 6: Count page types
        vector_pages = sum(
            1 for ext in extractions
            if ext.get('extraction_method') == 'vector'
        )
        scanned_pages = total_count - vector_pages

        # Step 7: Calculate confidence
        quality_rate = quality_count / total_count if total_count > 0 else 0
        confidence = self._calculate_confidence(quality_rate)

        # Step 8: Build and return template
        return Template.create_new(
            pdf_hash=pdf_hash,
            source_pdf=pdf_filename,
            pages_analyzed=total_count,
            quality_pages_used=quality_count,
            confidence=confidence,
            title_block_bbox=title_block_bbox,
            title_zone_bbox=title_zone_bbox,
            exclusion_patterns=exclusion_patterns,
            typical_length_min=typical_length_min,
            typical_length_max=typical_length_max,
            vector_pages=vector_pages,
            scanned_pages=scanned_pages
        )

    def _build_exclusion_patterns(self, extractions: List[Dict]) -> List[str]:
        """
        Build exclusion patterns from standard patterns plus failed extractions.

        Args:
            extractions: All extractions (including failed ones)

        Returns:
            List of regex pattern strings
        """
        # Start with standard exclusion patterns
        patterns = list(STANDARD_EXCLUSION_PATTERNS)

        # Find titles from failed pages
        failed_titles = Counter()
        for ext in extractions:
            needs_review = ext.get('needs_review', 0)
            if needs_review in [True, 1, '1']:
                title = ext.get('sheet_title', '')
                if title:
                    failed_titles[title] += 1

        # Add titles that appear 3+ times in failures
        for title, count in failed_titles.items():
            if count >= 3:
                # Escape for regex safety
                escaped = re.escape(title)
                patterns.append(escaped)

        return patterns

    def _calculate_confidence(self, quality_rate: float) -> float:
        """
        Calculate template confidence based on quality rate.

        Args:
            quality_rate: Fraction of pages that passed quality filter

        Returns:
            Confidence value between 0.0 and 1.0
        """
        if quality_rate >= 0.50:
            return 0.90
        elif quality_rate >= 0.30:
            return 0.80
        elif quality_rate >= 0.10:
            return 0.70
        else:
            return 0.60
