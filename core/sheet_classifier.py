"""
Sheet Classifier Module (V7.0)

Main orchestrator for classifying extracted sheet data.
Takes extraction results JSON and produces categorized output.

V7.0 Changes:
- Integrated EnhancedPatternMatcher for three-tier classification
- Added painting trade relevance scoring
- Added combo page detection
- Added signal aggregation with weighted confidence

Classes:
    SheetClassifier: Main classifier that processes extraction data
    ClassifiedSheet: Data class for classified sheet info
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pattern_matchers import CompositeMatcher, get_default_matcher
from core.enhanced_matcher import EnhancedPatternMatcher, EnhancedClassificationResult
from core.classification_types import ClassificationDecision, RelevanceLevel
from constants import SHEET_CATEGORIES, EXPANDED_SHEET_CATEGORIES


@dataclass
class ClassifiedSheet:
    """Represents a classified sheet with all metadata."""
    pdf_name: str
    page_number: int
    sheet_number: Optional[str]
    sheet_title: Optional[str]
    categories: List[str]
    classification_type: str  # 'matched', 'not_needed', 'unclassified'
    crop_file: Optional[str] = None
    number_confidence: float = 0.0
    title_confidence: float = 0.0
    error: Optional[str] = None
    # V7.0: Enhanced classification fields
    decision: Optional[str] = None  # DEFINITELY_NEEDED, DEFINITELY_NOT_NEEDED, NEEDS_EVALUATION
    relevance: Optional[str] = None  # PRIMARY, SECONDARY, REFERENCE, IRRELEVANT
    classification_confidence: float = 0.0
    is_combo_page: bool = False
    needs_review: bool = False
    review_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'pdf': self.pdf_name,
            'page': self.page_number,
            'sheet_number': self.sheet_number,
            'sheet_title': self.sheet_title,
            'categories': self.categories,
            'classification_type': self.classification_type,
            'crop_file': self.crop_file,
            'number_confidence': self.number_confidence,
            'title_confidence': self.title_confidence,
            'error': self.error,
            # V7.0: Enhanced classification fields
            'decision': self.decision,
            'relevance': self.relevance,
            'classification_confidence': self.classification_confidence,
            'is_combo_page': self.is_combo_page,
            'needs_review': self.needs_review,
            'review_reason': self.review_reason,
        }


@dataclass
class ClassificationResult:
    """Complete classification result for all PDFs."""
    classified_at: str
    total_pages: int
    categories: Dict[str, List[ClassifiedSheet]] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    # V7.0: Three-tier decision buckets
    decisions: Dict[str, List[ClassifiedSheet]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'classified_at': self.classified_at,
            'total_pages': self.total_pages,
            'statistics': self.statistics,
            'categories': {
                cat: [sheet.to_dict() for sheet in sheets]
                for cat, sheets in self.categories.items()
            },
            # V7.0: Three-tier decision buckets
            'decisions': {
                decision: [sheet.to_dict() for sheet in sheets]
                for decision, sheets in self.decisions.items()
            },
        }

    def get_definitely_needed(self) -> List[ClassifiedSheet]:
        """Get sheets that are definitely needed for painting."""
        return self.decisions.get('DEFINITELY_NEEDED', [])

    def get_definitely_not_needed(self) -> List[ClassifiedSheet]:
        """Get sheets that are definitely not needed for painting."""
        return self.decisions.get('DEFINITELY_NOT_NEEDED', [])

    def get_needs_evaluation(self) -> List[ClassifiedSheet]:
        """Get sheets that need human evaluation."""
        return self.decisions.get('NEEDS_EVALUATION', [])


class SheetClassifier:
    """
    Main sheet classification engine.

    Takes extraction results and classifies each page into categories
    based on sheet number and title patterns.

    V7.0: Uses EnhancedPatternMatcher for three-tier classification
    with painting trade relevance scoring.
    """

    def __init__(
        self,
        matcher: Optional[CompositeMatcher] = None,
        use_enhanced: bool = True,
        drawing_index: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize classifier.

        Args:
            matcher: Pattern matcher to use. Defaults to CompositeMatcher.
            use_enhanced: Use EnhancedPatternMatcher (V7.0). Defaults to True.
            drawing_index: Optional drawing index from PDF cover sheet.
        """
        self.matcher = matcher or get_default_matcher()
        self.use_enhanced = use_enhanced
        self.drawing_index = drawing_index or {}

        # V7.0: Initialize enhanced matcher
        if use_enhanced:
            self.enhanced_matcher = EnhancedPatternMatcher(drawing_index=drawing_index)
        else:
            self.enhanced_matcher = None

    def set_drawing_index(self, drawing_index: Dict[str, str]) -> None:
        """
        Set the drawing index for authoritative lookups.

        Args:
            drawing_index: Dict mapping sheet_number -> title
        """
        self.drawing_index = drawing_index
        if self.enhanced_matcher:
            self.enhanced_matcher.set_drawing_index(drawing_index)

    def classify_sheet(
        self,
        pdf_name: str,
        page_data: Dict[str, Any]
    ) -> ClassifiedSheet:
        """
        Classify a single sheet.

        Args:
            pdf_name: Name of the source PDF
            page_data: Page extraction data dict

        Returns:
            ClassifiedSheet with categorization
        """
        # Support both 'page' (old format) and 'page_number' (new format)
        page_number = page_data.get('page_number', page_data.get('page', 0))
        sheet_number = page_data.get('sheet_number')
        sheet_title = page_data.get('sheet_title')
        crop_file = page_data.get('crop_file')
        number_conf = page_data.get('number_conf', page_data.get('number_confidence', 0.0))
        title_conf = page_data.get('title_conf', page_data.get('title_confidence', 0.0))
        error = page_data.get('error')

        # V7.0: Use enhanced matcher if available
        if self.use_enhanced and self.enhanced_matcher:
            enhanced_result = self.enhanced_matcher.classify(sheet_number, sheet_title)

            # Map enhanced decision to classification_type for backward compatibility
            if enhanced_result.decision == ClassificationDecision.DEFINITELY_NEEDED:
                classification_type = 'matched'
            elif enhanced_result.decision == ClassificationDecision.DEFINITELY_NOT_NEEDED:
                classification_type = 'not_needed'
            else:
                classification_type = 'unclassified'

            return ClassifiedSheet(
                pdf_name=pdf_name,
                page_number=page_number,
                sheet_number=sheet_number,
                sheet_title=sheet_title,
                categories=enhanced_result.categories,
                classification_type=classification_type,
                crop_file=crop_file,
                number_confidence=number_conf,
                title_confidence=title_conf,
                error=error,
                # V7.0: Enhanced classification fields
                decision=enhanced_result.decision.name,
                relevance=enhanced_result.relevance.name,
                classification_confidence=enhanced_result.confidence,
                is_combo_page=enhanced_result.is_combo_page,
                needs_review=enhanced_result.needs_human_review,
                review_reason=enhanced_result.review_reason,
            )

        # Fallback to legacy matcher
        categories, classification_type = self.matcher.classify(sheet_number, sheet_title)

        return ClassifiedSheet(
            pdf_name=pdf_name,
            page_number=page_number,
            sheet_number=sheet_number,
            sheet_title=sheet_title,
            categories=categories,
            classification_type=classification_type,
            crop_file=crop_file,
            number_confidence=number_conf,
            title_confidence=title_conf,
            error=error,
        )

    def classify_extraction_results(
        self,
        extraction_data: Dict[str, Any],
        verbose: bool = False
    ) -> ClassificationResult:
        """
        Classify all sheets from extraction results.

        Args:
            extraction_data: Full extraction results dict
                Supports two formats:
                - Old: {"results": [{"pdf": "...", "pages": [...]}]}
                - New: {"sheets": [{...}, {...}, ...]}
            verbose: Print progress during classification

        Returns:
            ClassificationResult with all categorized sheets
        """
        # Handle both old and new report formats
        if 'results' in extraction_data:
            # Old format: results[].pages[]
            results_list = extraction_data.get('results', [])
        elif 'sheets' in extraction_data:
            # New format: sheets[] - group by pdf_filename
            sheets = extraction_data.get('sheets', [])
            # Convert to old format for processing
            pdf_pages: Dict[str, List] = {}
            for sheet in sheets:
                pdf_name = sheet.get('pdf_filename', 'unknown')
                if pdf_name not in pdf_pages:
                    pdf_pages[pdf_name] = []
                pdf_pages[pdf_name].append(sheet)
            results_list = [{'pdf': pdf, 'pages': pages} for pdf, pages in pdf_pages.items()]
        else:
            results_list = []

        # Initialize category buckets (including expanded categories)
        all_categories = {**SHEET_CATEGORIES, **EXPANDED_SHEET_CATEGORIES}
        categories: Dict[str, List[ClassifiedSheet]] = {
            cat: [] for cat in all_categories.keys()
        }
        categories['properly_classified_not_needed'] = []
        categories['unclassified_may_be_needed'] = []

        # V7.0: Three-tier decision buckets
        decision_buckets: Dict[str, List[ClassifiedSheet]] = {
            'DEFINITELY_NEEDED': [],
            'DEFINITELY_NOT_NEEDED': [],
            'NEEDS_EVALUATION': [],
        }

        total_pages = 0
        stats = {
            'by_category': {},
            'by_pdf': {},
            'classification_types': {
                'matched': 0,
                'not_needed': 0,
                'unclassified': 0,
            },
            # V7.0: Three-tier decision statistics
            'by_decision': {
                'DEFINITELY_NEEDED': 0,
                'DEFINITELY_NOT_NEEDED': 0,
                'NEEDS_EVALUATION': 0,
            },
            'by_relevance': {
                'PRIMARY': 0,
                'SECONDARY': 0,
                'REFERENCE': 0,
                'IRRELEVANT': 0,
            },
            'combo_pages': 0,
            'needs_review': 0,
        }

        for pdf_result in results_list:
            pdf_name = pdf_result.get('pdf', 'unknown')
            pages = pdf_result.get('pages', [])

            if verbose:
                print(f"  Classifying {pdf_name}: {len(pages)} pages...")

            pdf_stats = {'total': 0, 'matched': 0, 'not_needed': 0, 'unclassified': 0}

            for page_data in pages:
                total_pages += 1
                pdf_stats['total'] += 1

                classified = self.classify_sheet(pdf_name, page_data)

                # Add to appropriate category buckets
                for category in classified.categories:
                    if category in categories:
                        categories[category].append(classified)

                # Update statistics
                stats['classification_types'][classified.classification_type] += 1
                pdf_stats[classified.classification_type] += 1

                # V7.0: Update enhanced statistics
                if classified.decision:
                    stats['by_decision'][classified.decision] = stats['by_decision'].get(classified.decision, 0) + 1
                    decision_buckets[classified.decision].append(classified)
                if classified.relevance:
                    stats['by_relevance'][classified.relevance] = stats['by_relevance'].get(classified.relevance, 0) + 1
                if classified.is_combo_page:
                    stats['combo_pages'] += 1
                if classified.needs_review:
                    stats['needs_review'] += 1

            stats['by_pdf'][pdf_name] = pdf_stats

        # Calculate category statistics
        for cat, sheets in categories.items():
            stats['by_category'][cat] = len(sheets)

        return ClassificationResult(
            classified_at=datetime.now().isoformat(),
            total_pages=total_pages,
            categories=categories,
            statistics=stats,
            decisions=decision_buckets,
        )

    def classify_from_file(
        self,
        extraction_file: Path,
        verbose: bool = False
    ) -> ClassificationResult:
        """
        Load extraction results from file and classify.

        Args:
            extraction_file: Path to extraction results JSON
            verbose: Print progress during classification

        Returns:
            ClassificationResult with all categorized sheets
        """
        if verbose:
            print(f"Loading extraction results from: {extraction_file}")

        with open(extraction_file, 'r') as f:
            extraction_data = json.load(f)

        return self.classify_extraction_results(extraction_data, verbose=verbose)


def classify_sheets(extraction_file: Path, verbose: bool = False) -> ClassificationResult:
    """
    Convenience function to classify sheets from extraction file.

    Args:
        extraction_file: Path to extraction results JSON
        verbose: Print progress

    Returns:
        ClassificationResult
    """
    classifier = SheetClassifier()
    return classifier.classify_from_file(extraction_file, verbose=verbose)
