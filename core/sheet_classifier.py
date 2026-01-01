"""
Sheet Classifier Module (V6.0)

Main orchestrator for classifying extracted sheet data.
Takes extraction results JSON and produces categorized output.

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
from constants import SHEET_CATEGORIES


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
        }


@dataclass
class ClassificationResult:
    """Complete classification result for all PDFs."""
    classified_at: str
    total_pages: int
    categories: Dict[str, List[ClassifiedSheet]] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'classified_at': self.classified_at,
            'total_pages': self.total_pages,
            'statistics': self.statistics,
            'categories': {
                cat: [sheet.to_dict() for sheet in sheets]
                for cat, sheets in self.categories.items()
            }
        }


class SheetClassifier:
    """
    Main sheet classification engine.

    Takes extraction results and classifies each page into categories
    based on sheet number and title patterns.
    """

    def __init__(self, matcher: Optional[CompositeMatcher] = None):
        """
        Initialize classifier.

        Args:
            matcher: Pattern matcher to use. Defaults to CompositeMatcher.
        """
        self.matcher = matcher or get_default_matcher()

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

        # Get classification
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

        # Initialize category buckets
        categories: Dict[str, List[ClassifiedSheet]] = {
            cat: [] for cat in SHEET_CATEGORIES.keys()
        }
        categories['properly_classified_not_needed'] = []
        categories['unclassified_may_be_needed'] = []

        total_pages = 0
        stats = {
            'by_category': {},
            'by_pdf': {},
            'classification_types': {
                'matched': 0,
                'not_needed': 0,
                'unclassified': 0,
            }
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

            stats['by_pdf'][pdf_name] = pdf_stats

        # Calculate category statistics
        for cat, sheets in categories.items():
            stats['by_category'][cat] = len(sheets)

        return ClassificationResult(
            classified_at=datetime.now().isoformat(),
            total_pages=total_pages,
            categories=categories,
            statistics=stats,
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
