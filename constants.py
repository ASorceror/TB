"""
Blueprint Processor V4.2.1 - Constants Module
ALL patterns, thresholds, and configuration values are defined here.
Import from this module - NEVER define inline.
"""

from typing import Dict, List, Tuple
import re

# =============================================================================
# SECTION 4.1: REGEX PATTERNS
# =============================================================================
PATTERNS: Dict[str, re.Pattern] = {
    # Sheet number: A101, S-201, AD001, M1.01
    'sheet_number': re.compile(r'[A-Z]{1,2}[-]?\d{1,3}(?:\.\d{1,2})?'),

    # Project number: 2024-0156, 123456, 2024.001, P27142 (with optional letter prefix)
    'project_number': re.compile(r'[A-Z]?\d{4,}[-.]?\d*'),

    # Date: 12/15/2024, 1-5-25
    'date': re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'),

    # Scale: 1/4" = 1'-0", 1:100
    'scale': re.compile(r'\d+[\'"/]?\s*[=:]\s*\d+[\'"/-]*\d*'),
}

# =============================================================================
# SECTION 4.2: LABEL PATTERNS (for finding adjacent values)
# =============================================================================
LABELS: Dict[str, List[str]] = {
    'sheet_number': [
        'SHEET', 'SHEET NO', 'SHEET NUMBER', 'SHEET #',
        'DWG', 'DWG NO', 'DRAWING NO', 'NO.', 'NUMBER'
    ],
    'project_number': [
        'PROJECT', 'PROJECT NO', 'PROJECT NUMBER', 'PROJECT #',
        'JOB', 'JOB NO', 'JOB NUMBER', 'JOB #',
        'PROJ', 'PROJ NO', 'PROJ #'
    ],
    'sheet_title': [
        'TITLE', 'SHEET TITLE', 'DRAWING TITLE'
    ],
    'date': [
        'DATE', 'ISSUE DATE', 'ISSUED', 'PLOT DATE'
    ],
}

# =============================================================================
# SECTION 4.3: TITLE BLOCK KEYWORDS
# For validating detected regions (require >= 2 matches)
# =============================================================================
TITLE_BLOCK_KEYWORDS: List[str] = [
    'SHEET', 'PROJECT', 'DRAWING', 'SCALE', 'DATE', 'REVISION',
    'TITLE', 'DRAWN', 'CHECKED', 'APPROVED', 'DWG', 'NO.',
    'ARCHITECT', 'ENGINEER', 'CLIENT', 'ISSUE', 'REV', 'BY'
]

# =============================================================================
# SECTION 4.4: DISCIPLINE CODES
# First letter of sheet number indicates discipline
# =============================================================================
DISCIPLINE_CODES: Dict[str, str] = {
    'A': 'Architectural',
    'M': 'Mechanical',
    'S': 'Structural',
    'E': 'Electrical',
    'P': 'Plumbing',
    'C': 'Civil',
    'L': 'Landscape',
    'G': 'General',
}

# =============================================================================
# SECTION 4.5: SEARCH REGIONS (as % of image dimensions)
# Format: (x_start, y_start, x_end, y_end) as fractions
# =============================================================================
TITLE_BLOCK_REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    'bottom_right': (0.65, 0.75, 1.0, 1.0),       # PRIMARY
    'bottom_right_ext': (0.55, 0.80, 1.0, 1.0),   # Extended
    'right_strip': (0.85, 0.0, 1.0, 1.0),         # Tall narrow blocks
    'bottom_left': (0.0, 0.80, 0.35, 1.0),        # Alternate format
    'bottom_strip': (0.0, 0.85, 1.0, 1.0),        # Full bottom
}

# Search order for regions
REGION_SEARCH_ORDER: List[str] = [
    'bottom_right',
    'bottom_right_ext',
    'right_strip',
    'bottom_left',
    'bottom_strip',
]

# =============================================================================
# SECTION 4.6: TESSERACT CONFIGURATION
# =============================================================================
TESSERACT_CONFIG = {
    'osd': '--psm 0',                    # Orientation detection
    'page': '--psm 6 --oem 3',           # Page OCR
    'title_block': '--psm 6 --oem 3',    # Title block OCR
}

# =============================================================================
# SECTION 4.7: RELATIVE THRESHOLDS
# CRITICAL: Never hardcode pixel values - use these relative values
# =============================================================================
THRESHOLDS = {
    'min_line_length': 0.02,      # 2% of image width
    'max_line_gap': 0.005,        # 0.5% of image width
    'min_text_density': 0.01,     # 1%
    'min_text_for_vector': 50,    # Characters - below this, assume scanned
    'min_keywords_for_valid_region': 2,  # Minimum keywords to validate region
}

# =============================================================================
# RENDERING SETTINGS
# =============================================================================
DEFAULT_DPI: int = 200

# =============================================================================
# FILE PATHS (using forward slashes - pathlib will handle conversion)
# =============================================================================
DATABASE_NAME: str = 'blueprint_data.db'
LOG_FILENAME: str = 'blueprint_processor.log'

# =============================================================================
# EXTRACTION SETTINGS
# =============================================================================
EXTRACTION_METHODS = {
    'VECTOR_PDF': 'vector',      # Text extracted from embedded PDF text
    'SCANNED_PDF': 'ocr',        # Text extracted via OCR
}

# Confidence levels
CONFIDENCE_LEVELS = {
    'HIGH': 'high',
    'MEDIUM': 'medium',
    'LOW': 'low',
}

# =============================================================================
# SECTION 4.8: SHEET TITLE VALIDATION (V4.2.1)
# =============================================================================

# Garbage patterns - titles matching these should be REJECTED (case-insensitive)
TITLE_GARBAGE_PATTERNS: List[re.Pattern] = [
    re.compile(r'PROJECT\s*NO\.?', re.IGNORECASE),
    re.compile(r'sheet\s*number', re.IGNORECASE),
    re.compile(r'date\s*description', re.IGNORECASE),
    re.compile(r'PRELIMINARY', re.IGNORECASE),
    re.compile(r'NOT\s+FOR\s+CONSTRUCTION', re.IGNORECASE),
    re.compile(r'This\s+drawing', re.IGNORECASE),
    re.compile(r'design\s+concepts', re.IGNORECASE),
    re.compile(r'\d{3}[-.\s]\d{3}[-.\s]\d{4}'),  # Phone numbers
    re.compile(r'^\d{5}$'),  # ZIP codes (standalone 5 digits only)
    re.compile(r'\bLLC\b|\bINC\b|\bCORP\b|\bLTD\b', re.IGNORECASE),
    re.compile(r'www\.', re.IGNORECASE),
    re.compile(r'@'),  # Email addresses
]

# Valid title keywords - presence indicates legitimate sheet title (case-insensitive)
VALID_TITLE_KEYWORDS: List[str] = [
    # High frequency
    'PLAN', 'ELEVATION', 'SECTION', 'DETAIL', 'SCHEDULE', 'NOTES', 'DIAGRAM',
    'LAYOUT', 'VIEW', 'SPECIFICATIONS',
    # Building components
    'FLOOR', 'CEILING', 'ROOF', 'SITE', 'FOUNDATION', 'FRAMING', 'REFLECTED',
    'ENLARGED', 'DEMOLITION',
    # Directions
    'NORTH', 'SOUTH', 'EAST', 'WEST', 'FRONT', 'REAR', 'SIDE',
    # Building elements
    'INTERIOR', 'EXTERIOR', 'BUILDING', 'WALL', 'DOOR', 'WINDOW',
    # Disciplines
    'ELECTRICAL', 'MECHANICAL', 'PLUMBING', 'HVAC', 'FIRE', 'LIFE SAFETY',
    'STRUCTURAL', 'CIVIL', 'LANDSCAPE',
    # Administrative
    'GENERAL', 'CODE', 'ADA', 'ACCESSIBILITY', 'INDEX', 'COVER', 'PARTITION',
    'FINISH', 'FIXTURE', 'EQUIPMENT',
]

# Acronyms to preserve when converting to title case (2-4 letter all-caps)
TITLE_ACRONYMS: List[str] = [
    'ADA', 'HVAC', 'MEP', 'RCP', 'VIF', 'NIC', 'TYP', 'GWB', 'CMU', 'ACT',
    'NTS', 'EQ', 'CLR', 'OC', 'ID', 'OD', 'FF', 'TOS', 'TOC', 'BOC',
]

# Title length thresholds
TITLE_LENGTH = {
    'MIN': 3,              # < 3 chars: REJECT (too short)
    'SOFT_MAX': 45,        # 46-70 chars: ACCEPT with confidence penalty
    'MAX': 70,             # > 70 chars: REJECT (almost certainly garbage)
}

# Confidence scoring for title extraction
TITLE_CONFIDENCE = {
    'DRAWING_INDEX': 0.95,  # Found in drawing index
    'VISION_API': 0.90,     # Vision API extraction
    'SPATIAL': 0.80,        # Spatial zone detection (max)
    'PATTERN': 0.70,        # Pattern matching (existing method)
    # Adjustments
    'KEYWORD_BONUS': 0.05,  # Contains valid keyword
    'LONG_PENALTY': -0.10,  # Length 46-70 chars
    'SHORT_NO_KEYWORD_PENALTY': -0.05,  # No keyword AND length < 15 chars
}

# Review threshold - flag for HITL if below this
REVIEW_THRESHOLD: float = 0.70

# =============================================================================
# SECTION 4.9: DRAWING INDEX PARSING (V4.2.1 Phase 2)
# =============================================================================

# Keywords to search for drawing index section
DRAWING_INDEX_KEYWORDS: List[str] = [
    'DRAWING INDEX',
    'SHEET INDEX',
    'INDEX OF DRAWINGS',
    'LIST OF DRAWINGS',
    'DRAWING LIST',
    'SHEET LIST',
]

# Discipline headers to skip (not sheet entries)
DISCIPLINE_HEADERS: List[str] = [
    'ARCHITECTURAL', 'STRUCTURAL', 'MECHANICAL', 'ELECTRICAL',
    'PLUMBING', 'CIVIL', 'LANDSCAPE', 'GENERAL', 'FIRE PROTECTION',
    'INTERIOR', 'SITE', 'LIGHTING', 'HVAC', 'SPECIFICATIONS',
]

# Fuzzy matching threshold for OCR error correction
FUZZY_MATCH_THRESHOLD: float = 0.80

# Maximum pages to search for drawing index
MAX_INDEX_SEARCH_PAGES: int = 5
