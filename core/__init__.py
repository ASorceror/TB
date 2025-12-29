"""
Blueprint Processor V6.0 - Core Module

V6.0: Replaced RegionDetector with TitleBlockDiscovery (Vision AI-based)
V6.1: Added multi-stage TitleBlockDetector (CV + AI hybrid approach)
V6.2: Added Sheet Classification System
"""

from .pdf_handler import PDFHandler
from .page_normalizer import PageNormalizer
from .title_block_discovery import TitleBlockDiscovery
from .ocr_engine import OCREngine
from .extractor import Extractor

# New title block detection (V6.1)
from .title_block_detector import TitleBlockDetector
from .detect_title_block import (
    detect_title_block,
    crop_title_block,
    detect_and_crop_title_block,
    quick_detect,
    accurate_detect
)

# Sheet Classification System (V6.2)
from .pattern_matchers import (
    PatternMatcher,
    RegexPatternMatcher,
    SheetNumberMatcher,
    CompositeMatcher,
    get_default_matcher
)
from .sheet_classifier import (
    SheetClassifier,
    ClassifiedSheet,
    ClassificationResult,
    classify_sheets
)
from .sheet_organizer import (
    SheetOrganizer,
    OrganizeStats,
    organize_classified_sheets
)
from .classification_manifest import (
    ManifestGenerator,
    generate_manifests
)

__all__ = [
    'PDFHandler',
    'PageNormalizer',
    'TitleBlockDiscovery',
    'OCREngine',
    'Extractor',
    # Title block detection
    'TitleBlockDetector',
    'detect_title_block',
    'crop_title_block',
    'detect_and_crop_title_block',
    'quick_detect',
    'accurate_detect',
    # Sheet Classification System
    'PatternMatcher',
    'RegexPatternMatcher',
    'SheetNumberMatcher',
    'CompositeMatcher',
    'get_default_matcher',
    'SheetClassifier',
    'ClassifiedSheet',
    'ClassificationResult',
    'classify_sheets',
    'SheetOrganizer',
    'OrganizeStats',
    'organize_classified_sheets',
    'ManifestGenerator',
    'generate_manifests',
]
