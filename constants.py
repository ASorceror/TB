"""
Blueprint Processor V4.4 - Constants Module
ALL patterns, thresholds, and configuration values are defined here.
Import from this module - NEVER define inline.

V4.4 Changes:
- Removed aggressive garbage patterns (University, College, Baseball, etc.)
- Reduced QUALITY_SINGLE_KEYWORD_REJECTS list
- Added VALID_SINGLE_WORD_TITLES for acceptable single-word titles
"""

from typing import Dict, List, Tuple
import re

# =============================================================================
# SECTION 4.1: REGEX PATTERNS
# =============================================================================
PATTERNS: Dict[str, re.Pattern] = {
    # Sheet number: A101, S-201, AD001, M1.01
    'sheet_number': re.compile(r'\b[A-Z]{1,2}[-]?\d{1,3}(?:\.\d{1,2})?\b'),

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

    # === V4.2.2 QUALITY GATE PATTERNS ===

    # Project descriptors (248 occurrences in production)
    re.compile(r'Gmp[/\s]?permit', re.IGNORECASE),
    re.compile(r'permit\s*set', re.IGNORECASE),

    # Spec section headers (78 occurrences)
    re.compile(r'^Part\s+\d+', re.IGNORECASE),
    re.compile(r'End\s+of\s+Section', re.IGNORECASE),
    re.compile(r'Contract,?\s+apply', re.IGNORECASE),

    # Pure numbers or sheet-number-like patterns
    re.compile(r'^\d{3,5}$'),  # "1425", "60601"
    re.compile(r'^[A-Z]{1,2}[-.]?\d{1,3}(\.\d{1,2})?$', re.IGNORECASE),  # "A2.0", "S-101"

    # V4.4: Removed overly aggressive patterns that rejected valid titles:
    # - University, College, Ball State (these can appear in valid project titles)
    # - Northern Tool, Baseball Building (valid project names)
    # - Recreation Center (valid building type)

    # Address patterns
    re.compile(r'^\d+\s+[NSEW]\.?\s+\w+', re.IGNORECASE),  # "955 N Larrabee"
    re.compile(r'\b(Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Drive|Dr)\b', re.IGNORECASE),
    re.compile(r'Larrabee', re.IGNORECASE),
    re.compile(r'Woodward', re.IGNORECASE),
    re.compile(r'Frontage\s*Rd', re.IGNORECASE),
    re.compile(r'Loop\s+\d+', re.IGNORECASE),

    # Scale notations mistaken as titles
    re.compile(r"^\d+['\"]?\s*[-=]\s*\d+", re.IGNORECASE),  # "0'-0" = +13.00'"
    re.compile(r'^C\.?c\.?d\.?$', re.IGNORECASE),

    # Partial sentences / fragments
    re.compile(r'manufacturers?\s+specified', re.IGNORECASE),
    re.compile(r'site/platform', re.IGNORECASE),
    re.compile(r'^knowledge,?$', re.IGNORECASE),
    re.compile(r'^equipment[,.]?$', re.IGNORECASE),
    re.compile(r'^urban$', re.IGNORECASE),
    re.compile(r'^in\s+this\s+section', re.IGNORECASE),

    # Corridor/Location names (not drawing titles)
    re.compile(r'^(North|South|East|West)\s+Corridor$', re.IGNORECASE),
    re.compile(r'^Germantown$', re.IGNORECASE),
    re.compile(r'^Graceland\s+Avenue$', re.IGNORECASE),
]

# V4.4: Single keywords that are NOT valid titles on their own
# Reduced list - many single words ARE valid drawing titles
QUALITY_SINGLE_KEYWORD_REJECTS: List[str] = [
    'PLAN',       # Too generic alone
    'SECTION',    # Too generic alone
    'DETAIL',     # Too generic alone
    'DIAGRAM',    # Too generic alone
    'NOTES',      # Too generic alone
    'LEGEND',     # Too generic alone
    'INDEX',      # Too generic alone
    'FLOOR',      # Too generic alone
    'GENERAL',    # Too generic alone
]

# V4.4: Single words that ARE valid titles (construction/architecture terms)
VALID_SINGLE_WORD_TITLES: List[str] = [
    'APARTMENTS', 'ELEVATIONS', 'ELEVATION', 'DETAILS', 'SECTIONS',
    'PLANS', 'SCHEDULES', 'SCHEDULE', 'SPECIFICATIONS', 'SPECIFICATION',
    'DEMOLITION', 'MECHANICAL', 'ELECTRICAL', 'PLUMBING', 'STRUCTURAL',
    'CIVIL', 'LANDSCAPE', 'INTERIOR', 'EXTERIOR', 'FOUNDATION',
    'HVAC', 'FRAMING', 'PARTITIONS', 'CEILINGS', 'FINISHES',
]

# V4.2.2: Quality Gate thresholds
QUALITY_GATE_THRESHOLDS = {
    'min_length': 3,                    # Minimum characters
    'min_words_without_keyword': 2,     # If no keyword, need at least 2 words
    'min_length_single_word': 15,       # Single word must be 15+ chars if no keyword
    'max_length': 70,                   # Maximum characters (same as existing)
}

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
    'SOFT_MAX': 60,        # 61-120 chars: ACCEPT with confidence penalty
    'MAX': 120,            # > 120 chars: TRUNCATE (V4.5: multi-part titles can be long)
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
# SECTION 4.8.1: TITLE LABEL BLACKLIST (V4.2.4)
# =============================================================================

# Labels that appear in title blocks but are NOT titles
# These get incorrectly extracted when text ordering places them after "TITLE:"
TITLE_LABEL_BLACKLIST = {
    # Field labels
    "PROJECT NUMBER", "PROJECT NO", "PROJECT #", "PROJECT",
    "PLOT DATE", "DATE", "ISSUE DATE", "REVISION DATE",
    "SCALE", "DRAWING SCALE", "SHEET SCALE",
    "SHEET NUMBER", "SHEET NO", "SHEET #", "SHEET",
    "DRAWING NUMBER", "DRAWING NO", "DWG NO", "DWG",
    "REVISION", "REV", "REVISION NO",

    # Signature/approval labels
    "DRAWN BY", "DRAWN", "DRAFTED BY",
    "CHECKED BY", "CHECKED", "CHECKER",
    "APPROVED BY", "APPROVED", "APPROVAL",
    "DESIGNED BY", "DESIGNER",
    "NAME AND TITLE", "NAME SIGNATURE", "SIGNATURE",

    # Other common labels
    "ADDRESS", "LOCATION", "SITE",
    "CLIENT", "OWNER", "ARCHITECT", "ENGINEER",
    "PHONE", "FAX", "EMAIL",
    "COPYRIGHT", "CONFIDENTIAL",
    "NOT FOR CONSTRUCTION", "PRELIMINARY", "DRAFT",
    "TITLE BLOCK", "TITLE:",
}

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

# =============================================================================
# SECTION 4.10: SHEET NUMBER VALIDATION (V4.2.3)
# =============================================================================

# Strict regex pattern for sheet numbers - REQUIRES at least one digit
# Pattern: 1-2 letters + optional separator + 1-4 digits + optional decimal
SHEET_NUMBER_STRICT_PATTERN = re.compile(r'^[A-Z]{1,2}[-.]?\d{1,4}(?:\.\d{1,2})?$', re.IGNORECASE)

# Blacklist of known garbage values that slip through pattern matching
SHEET_NUMBER_BLACKLIST = {
    # Two-letter garbage (no digits) - from OCR pattern [\dO] matching "O" as digit
    "TO", "NO", "SO", "RO", "CO", "FO", "DO", "GO", "PO", "MO",
    "LO", "WO", "BO", "HO", "OO", "YO", "IO", "UO", "EO", "AO",
    "AT", "AS", "BY", "IF", "IN", "IS", "IT", "OF", "ON", "OR",
    "UP", "AN", "BE", "GO", "HE", "ME", "MY", "OK", "OX",
    "US", "WE",

    # Abbreviations with periods (Top Of, Not Otherwise, etc.)
    "T.O", "N.O", "F.O", "R.O", "P.O", "B.O", "C.O", "D.O", "S.O",
    "T.O.", "N.O.", "F.O.", "R.O.", "P.O.", "B.O.", "C.O.", "D.O.", "S.O.",

    # Common OCR/extraction errors
    "TA4", "NO.", "SH", "DW", "DWG", "REV", "REF",

    # Common words that might slip through
    "ADD", "ALL", "AND", "ARE", "BUT", "CAN", "FOR", "HAD", "HAS",
    "HER", "HIM", "HIS", "HOW", "ITS", "LET", "MAY", "NEW", "NOT",
    "NOW", "OLD", "OUR", "OUT", "OWN", "SAY", "SHE", "THE", "TOO",
    "TRY", "TWO", "USE", "WAY", "WHO", "WHY", "YET", "YOU",
}
