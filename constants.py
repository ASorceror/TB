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
# SECTION 4.4.1: SHEET CATEGORIES (V6.0 Classification System)
# Used by sheet_classifier.py to organize extracted pages by type.
# Each category has title patterns (regex) and optional sheet number patterns.
# =============================================================================
SHEET_CATEGORIES: Dict[str, Dict] = {
    'floor_plans': {
        'folder': 'floor_plans',
        'description': 'Floor plans and overall plans',
        'title_patterns': [
            r'\bfloor\s*plan',
            r'\boverall\s*plan',
            r'\bbuilding\s*plan',
            r'\barea\s*plan',
            r'\blevel\s*\d+\s*plan',
            r'\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\s*(floor|level)\s*plan',
            r'\bbasement\s*plan',
            r'\bmezzanine\s*plan',
            r'\bground\s*(floor|level)\s*plan',
            r'\bpenthouse\s*plan',
            r'\broof\s*plan',
            r'\bunit\s*plans?\b',  # V6.1: "Unit Plan", "Fourth Floor Unit Plan"
        ],
        'sheet_patterns': [
            r'^A[1-9]0[1-9]',   # A101, A102, etc. - Floor plans
            r'^A[1-9]1\d',      # A110-A119 - Additional floor plans
        ],
    },
    'reflected_ceiling_plans': {
        'folder': 'reflected_ceiling_plans',
        'description': 'Reflected ceiling plans (RCP)',
        'title_patterns': [
            r'\breflected\s*ceiling',
            r'\bRCP\b',
            r'\bceiling\s*plan',
            r'\bceiling\s*layout',
        ],
        'sheet_patterns': [
            r'^A[1-9]2\d',      # A120-A129 - RCP series
        ],
    },
    'room_finish_schedules': {
        'folder': 'room_finish_schedules',
        'description': 'Room finish schedules and finish plans',
        'title_patterns': [
            r'\broom\s*finish',
            r'\bfinish\s*schedule',
            r'\bfinish\s*plan',
            r'\binterior\s*finish',
            r'\bmaterial\s*schedule',
            r'\bcolor\s*schedule',
        ],
        'sheet_patterns': [],
    },
    'interior_elevations': {
        'folder': 'interior_elevations',
        'description': 'Interior elevations',
        'title_patterns': [
            r'\binterior\s*elevation',
            r'\bint\.?\s*elevation',
            r'\broom\s*elevation',
            r'\bwall\s*elevation',
            r'\bcasework\s*elevation',
            r'\bcabinet\s*elevation',
            r'\bmillwork\s*elevation',
            r'\belevation.*\binterior',
        ],
        'sheet_patterns': [
            r'^A[1-9]3\d',      # A130-A139 - Interior elevations
        ],
    },
    'exterior_elevations': {
        'folder': 'exterior_elevations',
        'description': 'Exterior elevations',
        'title_patterns': [
            r'\bexterior\s*elevation',
            r'\bext\.?\s*elevation',
            r'\bbuilding\s*elevation',
            r'\bfront\s*elevation',
            r'\brear\s*elevation',
            r'\bside\s*elevation',
            r'\b(north|south|east|west)\s*elevation',
            r'\belevation.*\bexterior',
            r'^elevations?$',  # V6.1: Standalone "ELEVATIONS" defaults to exterior
        ],
        'sheet_patterns': [
            r'^A[2-3]0\d',      # A201-A209, A301-A309 - Exterior elevations/sections
        ],
    },
    'cover_sheets': {
        'folder': 'cover_sheets',
        'description': 'Cover sheets, title sheets, and indexes',
        'title_patterns': [
            r'\bcover\s*sheet',
            r'\btitle\s*sheet',
            r'\bsheet\s*index',
            r'\bdrawing\s*index',
            r'\bgeneral\s*notes',
            r'\bcode\s*analysis',
            r'\bcode\s*compliance',  # V6.1: "Code Compliance"
            r'\bproject\s*data',
            r'\blife\s*safety',
            r'\bada\s*',
            r'\baccessibility',
            r'\babbreviations',
            r'\bsymbols?\s*legend',
        ],
        'sheet_patterns': [
            r'^[AG]0',          # A001, G001, etc. - Cover/general sheets
            r'^A-0',            # A-001, A-002 format
        ],
    },
}

# Categories for unclassified sheets (not in main categories)
UNCLASSIFIED_CATEGORIES: Dict[str, Dict] = {
    'properly_classified_not_needed': {
        'folder': None,  # No folder - manifest only
        'description': 'Recognized sheet types not needed for current processing',
        'title_patterns': [
            # Structural
            r'\bstructural',
            r'\bframing\s*plan',
            r'\bfoundation\s*plan',
            # MEP
            r'\bmechanical',
            r'\belectrical',
            r'\bplumbing',
            r'\bhvac',
            r'\bfire\s*(protection|alarm|sprinkler)',
            r'\bpower\s*plan',
            r'\blighting\s*plan',
            # Civil/Site
            r'\bcivil',
            r'\bsite\s*plan',
            r'\bgrading\s*plan',
            r'\butility\s*plan',
            r'\blandscape',
            r'\birrigation',
            # Sections/Details
            r'\bwall\s*section',
            r'\bbuilding\s*section',
            r'\bdetail',
            r'\benlarged',
            r'\bstair',
            r'\belevator',
            # Schedules
            r'\bdoor\s*schedule',
            r'\bwindow\s*schedule',
            r'\bhardware\s*schedule',
            r'^schedules?$',  # V6.1: Generic standalone "Schedule(s)"
            # V6.1: Additional not-needed patterns
            r'\bloading\s*plan',
            r'\bdemolition\s*elevation',
            r'\bsignage',
            r'\bsign\s*plan',
        ],
        'sheet_patterns': [
            r'^S\d',           # Structural
            r'^M\d',           # Mechanical
            r'^E\d',           # Electrical
            r'^P\d',           # Plumbing
            r'^C\d',           # Civil
            r'^L\d',           # Landscape
        ],
    },
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

# Search order for regions (legacy - used as fallback)
REGION_SEARCH_ORDER: List[str] = [
    'bottom_right',
    'bottom_right_ext',
    'right_strip',
    'bottom_left',
    'bottom_strip',
]

# =============================================================================
# SECTION 4.5.1: TITLE BLOCK PHYSICAL DIMENSIONS (V4.5)
# Standard architectural title blocks have fixed physical sizes regardless
# of sheet size. We use these to calculate dynamic region percentages.
# =============================================================================
TITLE_BLOCK_PHYSICAL = {
    'width_inches': 6.5,      # Standard title block width (slightly generous)
    'height_inches': 4.0,     # Standard title block height
    'max_width_pct': 0.35,    # Never exceed 35% of page width (for small sheets)
    'max_height_pct': 0.30,   # Never exceed 30% of page height
}

# =============================================================================
# SECTION 4.6: TESSERACT CONFIGURATION
# V4.9: Expanded PSM modes for different use cases
# PSM Modes:
#   0 = OSD only (orientation detection)
#   3 = Fully automatic page segmentation (default)
#   4 = Single column of text (good for title blocks)
#   6 = Single uniform block of text
#   7 = Single text line
#   8 = Single word
#   11 = Sparse text (find as much text as possible)
#   12 = Sparse text with OSD
# OEM Modes:
#   0 = Legacy engine only
#   1 = LSTM neural net only
#   3 = Default (best available)
# =============================================================================
TESSERACT_CONFIG = {
    'osd': '--psm 0',                    # Orientation detection only
    'page': '--psm 6 --oem 3',           # Single uniform block (standard pages)
    'title_block': '--psm 4 --oem 3',    # Single column (better for title blocks)
    'single_line': '--psm 7 --oem 3',    # Single text line
    'single_word': '--psm 8 --oem 3',    # Single word
    'sparse': '--psm 11 --oem 3',        # Sparse text (find all text)
    'auto': '--psm 3 --oem 3',           # Fully automatic segmentation
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
# V4.9: Increased DPI from 200 to 300 for better OCR accuracy
# Tesseract works best at 300+ DPI, accuracy drops below 10pt at 200 DPI
# =============================================================================
DEFAULT_DPI: int = 300

# OCR-specific settings
OCR_MIN_DPI: int = 300          # Minimum DPI for OCR operations
OCR_TARGET_DPI: int = 300       # Target DPI when upscaling for OCR
OCR_CONFIDENCE_THRESHOLD: int = 60  # Minimum confidence score (0-100)

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

# =============================================================================
# SECTION 4.11: QUALITY FILTER (V4.3)
# =============================================================================

QUALITY_KEYWORDS: List[str] = [
    'PLAN', 'ELEVATION', 'SECTION', 'DETAIL', 'SCHEDULE', 'DIAGRAM',
    'LAYOUT', 'VIEW', 'FLOOR', 'CEILING', 'ROOF', 'SITE', 'FOUNDATION',
    'FRAMING', 'REFLECTED', 'ENLARGED', 'DEMOLITION', 'INTERIOR',
    'EXTERIOR', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'FRONT', 'REAR',
    'WALL', 'DOOR', 'WINDOW', 'STAIR', 'PARTITION', 'ELECTRICAL',
    'MECHANICAL', 'PLUMBING', 'HVAC', 'FIRE', 'STRUCTURAL', 'CIVIL',
    'LANDSCAPE', 'NOTES', 'SPECIFICATIONS', 'CODE', 'ADA', 'FINISH',
    'FIXTURE', 'EQUIPMENT', 'RISER', 'ISOMETRIC', 'SCHEMATIC',
]

# Patterns that indicate garbage (always reject even if contains keywords)
GARBAGE_PATTERNS: List[str] = [
    r'^Part\s+\d+',              # Spec section headers: "Part 1 - General"
    r'^\d+$',                    # Pure numbers: "1425"
    r'^[A-Z]\d+(\.\d+)?$',       # Sheet numbers: "A101", "A1.0"
    r'^[A-Z]{1,2}-\d+$',         # Sheet numbers: "A-101", "M-201"
    r'Permit\s*Set',             # Project descriptors
    r'Gmp[/\s]',                 # GMP references
    r'^(Project|Sheet)\s*(Number|No|#)',  # Labels
    r'^(Date|Scale|Drawn|Checked|Approved)\b',  # Field labels
]

QUALITY_THRESHOLDS: Dict[str, any] = {
    'min_title_length': 5,
    'min_word_count': 2,
    'min_length_if_one_word': 15,
    'max_repetition_percent': 0.20,
    'min_quality_for_learning': 5,
}

# =============================================================================
# SECTION 4.12: TEMPLATE LEARNING (V4.3)
# =============================================================================

TEMPLATE_DEFAULTS: Dict[str, any] = {
    'title_block_bbox': [0.65, 0.75, 1.0, 1.0],  # Bottom-right 35% x 25%
    'title_zone_bbox': [0.0, 0.10, 1.0, 0.50],   # Top half of title block
}

STANDARD_EXCLUSION_PATTERNS: List[str] = [
    r'PROJECT\s*(NUMBER|NO\.?|#)',
    r'SHEET\s*(NUMBER|NO\.?|#)',
    r'DATE\s*:?',
    r'SCALE\s*:?',
    r'DRAWN\s*BY',
    r'CHECKED\s*BY',
    r'APPROVED\s*BY',
    r'REVISION',
    r'REV\s*:?',
]

# =============================================================================
# SECTION 4.13: ENHANCED CLASSIFICATION SYSTEM (V7.0)
# Three-tier classification with painting trade focus
# =============================================================================

# Classification decision thresholds
CLASSIFICATION_THRESHOLDS = {
    'definitely_needed': 0.85,      # Auto-include threshold
    'definitely_not_needed': 0.85,  # Auto-exclude threshold
    'needs_evaluation': 0.70,       # Below this -> human review
    'conflict_penalty': 0.15,       # Penalty when signals disagree
    'combo_page_penalty': 0.10,     # Penalty for multi-drawing pages
    'high_confidence': 0.90,        # Very confident
    'medium_confidence': 0.70,      # Moderately confident
    'low_confidence': 0.50,         # Low confidence
}

# Signal weights for confidence aggregation
# Higher weight = more influence on final decision
SIGNAL_WEIGHTS = {
    'drawing_index': 0.50,      # From PDF cover sheet - most authoritative
    'title_pattern': 0.40,      # Explicit category keywords in title
    'vision_api': 0.35,         # Visual content analysis
    'content_analysis': 0.30,   # Inferred from drawing features
    'sheet_number': 0.25,       # NCS discipline/type inference
}

# Painting trade relevance by category
# Maps category names to relevance levels for painting scope of work
PAINTING_RELEVANCE = {
    # PRIMARY - Essential for painting trade
    'floor_plans': 'PRIMARY',
    'reflected_ceiling_plans': 'PRIMARY',
    'room_finish_schedules': 'PRIMARY',
    'interior_elevations': 'PRIMARY',
    'finish_plans': 'PRIMARY',
    'paint_schedules': 'PRIMARY',
    'color_schedules': 'PRIMARY',
    'partition_plans': 'PRIMARY',

    # SECONDARY - Important reference
    'exterior_elevations': 'SECONDARY',
    'building_sections': 'SECONDARY',
    'enlarged_plans': 'SECONDARY',
    'door_schedules': 'SECONDARY',
    'window_schedules': 'SECONDARY',
    'wall_sections': 'SECONDARY',
    'details': 'SECONDARY',

    # REFERENCE - Useful context
    'cover_sheets': 'REFERENCE',
    'general_notes': 'REFERENCE',
    'code_analysis': 'REFERENCE',
    'life_safety': 'REFERENCE',
    'accessibility': 'REFERENCE',

    # IRRELEVANT - Not needed for painting
    'structural': 'IRRELEVANT',
    'mechanical': 'IRRELEVANT',
    'electrical': 'IRRELEVANT',
    'plumbing': 'IRRELEVANT',
    'fire_protection': 'IRRELEVANT',
    'civil': 'IRRELEVANT',
    'landscape': 'IRRELEVANT',
    'site_plans': 'IRRELEVANT',
    'properly_classified_not_needed': 'IRRELEVANT',
}

# =============================================================================
# SECTION 4.14: EXPANDED SHEET CATEGORIES (V7.0)
# Additional categories for comprehensive classification
# =============================================================================

# New categories to add to SHEET_CATEGORIES
EXPANDED_SHEET_CATEGORIES: Dict[str, Dict] = {
    'finish_plans': {
        'folder': 'finish_plans',
        'description': 'Finish plans showing materials and finishes',
        'title_patterns': [
            r'\bfinish\s*plan',
            r'\blower\s*level\s*finish',
            r'\bupper\s*level\s*finish',
            r'\bfinish\s*floor\s*plan',
            r'\bfinish\s*layout',
        ],
        'sheet_patterns': [],
    },
    'partition_plans': {
        'folder': 'partition_plans',
        'description': 'Partition plans showing wall layouts',
        'title_patterns': [
            r'\bpartition\s*plan',
            r'\bpartition\s*layout',
            r'\bwall\s*layout\s*plan',
            r'\bwall\s*type\s*plan',
        ],
        'sheet_patterns': [],
    },
    'door_schedules': {
        'folder': 'door_schedules',
        'description': 'Door schedules with door types and hardware',
        'title_patterns': [
            r'\bdoor\s*schedule',
            r'\bdoor\s*&\s*frame\s*schedule',
            r'\bdoor\s*type',
            r'\bhardware\s*schedule',
            r'\bhardware\s*set',
        ],
        'sheet_patterns': [],
    },
    'window_schedules': {
        'folder': 'window_schedules',
        'description': 'Window schedules with window types',
        'title_patterns': [
            r'\bwindow\s*schedule',
            r'\bwindow\s*type',
            r'\bglazing\s*schedule',
            r'\bstorefront\s*schedule',
        ],
        'sheet_patterns': [],
    },
    'building_sections': {
        'folder': 'building_sections',
        'description': 'Building cross-sections',
        'title_patterns': [
            r'\bbuilding\s*section',
            r'\bcross\s*section',
            r'\blongitudinal\s*section',
            r'\btransverse\s*section',
            r'\btypical\s*section',
        ],
        'sheet_patterns': [
            r'^A[3-4]0\d',  # A301-A309, A401-A409
        ],
    },
    'wall_sections': {
        'folder': 'wall_sections',
        'description': 'Wall section details',
        'title_patterns': [
            r'\bwall\s*section',
            r'\bexterior\s*wall\s*section',
            r'\binterior\s*wall\s*section',
            r'\btypical\s*wall',
        ],
        'sheet_patterns': [],
    },
    'enlarged_plans': {
        'folder': 'enlarged_plans',
        'description': 'Enlarged floor plans of specific areas',
        'title_patterns': [
            r'\benlarged\s*plan',
            r'\benlarged\s*floor\s*plan',
            r'\btoilet\s*room\s*plan',
            r'\brestroom\s*plan',
            r'\bbathroom\s*plan',
            r'\blobby\s*plan',
            r'\bstair\s*plan',
            r'\belevator\s*lobby',
            r'\bentry\s*plan',
            r'\bvestibule\s*plan',
        ],
        'sheet_patterns': [
            r'^A[4-5]0\d',  # A401-A409, A501-A509 - Enlarged views
        ],
    },
    'demolition_plans': {
        'folder': 'demolition_plans',
        'description': 'Demolition plans showing existing conditions',
        'title_patterns': [
            r'\bdemolition\s*plan',
            r'\bdemo\s*plan',
            r'\bexisting\s*plan',
            r'\bexisting\s*conditions',
            r'\bexisting\s*to\s*remain',
        ],
        'sheet_patterns': [],
    },
}

# Categories for content analysis triggers (when title is ambiguous)
CONTENT_ANALYSIS_TRIGGERS = {
    'ambiguous_title_patterns': [
        r'^plan$',           # Just "Plan" - which type?
        r'^elevation$',      # Just "Elevation" - interior or exterior?
        r'^schedule$',       # Just "Schedule" - which type?
        r'^section$',        # Just "Section" - which type?
        r'^details?$',       # Just "Detail(s)" - which type?
        r'^(level|floor)\s*\d+$',  # Just a level number
    ],
    'combo_page_detected': True,     # Trigger when combo page found
    'low_confidence_threshold': 0.65, # Trigger when confidence below this
    'conflict_detected': True,        # Trigger when signals conflict
}
