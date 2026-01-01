"""
Title Parser Module (V7.0)

Parses sheet titles to detect combo pages (multiple drawings on one sheet)
and extract individual title components for classification.

Combo pages are common in construction documents:
    - "FLOOR PLAN\nROOF PLAN" - Two plans on one sheet
    - "FLOOR PLAN AND REFLECTED CEILING PLAN" - Combined drawing
    - "FIRST FLOOR PLAN - SECOND FLOOR PLAN" - Multiple levels

This module provides:
    - TitleParser: Parse titles and detect combo pages
    - split_combo_title: Split multi-part titles into components
    - detect_drawing_type: Identify drawing type from title text
"""

import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .classification_types import TitleComponent


# =============================================================================
# COMBO PAGE DETECTION PATTERNS
# =============================================================================

# Separators that indicate multiple drawings on one sheet
COMBO_SEPARATORS = [
    r'\n',           # Newline - very common for multi-part titles
    r'\s+AND\s+',    # "FLOOR PLAN AND ROOF PLAN"
    r'\s+&\s+',      # "FLOOR PLAN & ROOF PLAN"
    r'\s+/\s+',      # "FLOOR PLAN / ROOF PLAN"
    r'\s+-\s+-\s+',  # "FLOOR PLAN -- ROOF PLAN" (double dash)
]

# Patterns that look like separators but are NOT combos
# These are valid single-title patterns that should not be split
NOT_COMBO_PATTERNS = [
    # Level/floor designations
    r'FLOOR\s+PLAN\s*[-/]\s*LEVEL\s+\d+',
    r'LEVEL\s+\d+\s*[-/]\s*FLOOR\s+PLAN',
    r'\d+(ST|ND|RD|TH)\s+FLOOR\s*[-/]\s*',
    r'[-/]\s*\d+(ST|ND|RD|TH)\s+FLOOR',
    r'BASEMENT\s*[-/]\s*LEVEL',

    # Option/alternative designations
    r'OPTION\s+[A-Z]\s*[-/]',
    r'[-/]\s*OPTION\s+[A-Z]',
    r'ALTERNATE\s*[-/]',
    r'ALT\s*[-/]',

    # Phase designations
    r'PHASE\s+\d+\s*[-/]',
    r'[-/]\s*PHASE\s+\d+',

    # Building/wing designations
    r'BUILDING\s+[A-Z0-9]+\s*[-/]',
    r'[-/]\s*BUILDING\s+[A-Z0-9]+',
    r'WING\s+[A-Z0-9]+\s*[-/]',

    # Direction designations (not combos)
    r'NORTH\s*[-/]\s*SOUTH',
    r'EAST\s*[-/]\s*WEST',

    # Scale notations
    r'\d+[\'\"]\s*[-=]\s*\d+',

    # Dates
    r'\d+\s*[-/]\s*\d+\s*[-/]\s*\d+',
]

# Drawing types we recognize (for component classification)
DRAWING_TYPES: Dict[str, List[str]] = {
    'floor_plan': [
        r'\bfloor\s*plan\b',
        r'\bfloor\s+\d+\s*plan\b',
        r'\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\s+(floor|level)\s*plan\b',
        r'\bbasement\s*plan\b',
        r'\bmezzanine\s*plan\b',
        r'\bpenthouse\s*plan\b',
        r'\bground\s*(floor|level)\s*plan\b',
        r'\blevel\s+\d+\s*plan\b',
        r'\boverall\s*plan\b',
        r'\bfinish\s*plan\b',
        r'\bdemolition\s*plan\b',
        r'\bpartition\s*plan\b',
        r'\bfurniture\s*plan\b',
        r'\bpower\s*\&?\s*comm(unications?)?\s*plan\b',
    ],
    'reflected_ceiling_plan': [
        r'\breflected\s*ceiling\s*plan\b',
        r'\bRCP\b',
        r'\bceiling\s*plan\b',
        r'\bceiling\s*layout\b',
    ],
    'roof_plan': [
        r'\broof\s*plan\b',
        r'\broofing\s*plan\b',
    ],
    'interior_elevation': [
        r'\binterior\s*elevation\b',
        r'\bint\.?\s*elevation\b',
        r'\broom\s*elevation\b',
        r'\bwall\s*elevation\b',
        r'\bcasework\s*elevation\b',
        r'\bcabinet\s*elevation\b',
        r'\bmillwork\s*elevation\b',
    ],
    'exterior_elevation': [
        r'\bexterior\s*elevation\b',
        r'\bext\.?\s*elevation\b',
        r'\bbuilding\s*elevation\b',
        r'\b(north|south|east|west)\s*elevation\b',
        r'\bfront\s*elevation\b',
        r'\brear\s*elevation\b',
        r'\bside\s*elevation\b',
    ],
    'section': [
        r'\bbuilding\s*section\b',
        r'\bwall\s*section\b',
        r'\bcross\s*section\b',
        r'\blongitudinal\s*section\b',
        r'\btransverse\s*section\b',
    ],
    'schedule': [
        r'\bfinish\s*schedule\b',
        r'\broom\s*finish\b',
        r'\bdoor\s*schedule\b',
        r'\bwindow\s*schedule\b',
        r'\bhardware\s*schedule\b',
        r'\bcolor\s*schedule\b',
        r'\bpaint\s*schedule\b',
    ],
    'detail': [
        r'\bdetail\b',
        r'\bdetails\b',
        r'\benlarged\b',
    ],
    'cover_sheet': [
        r'\bcover\s*sheet\b',
        r'\btitle\s*sheet\b',
        r'\bsheet\s*index\b',
        r'\bdrawing\s*index\b',
        r'\bgeneral\s*notes\b',
    ],
}

# Compile patterns for performance
_COMPILED_NOT_COMBO = [re.compile(p, re.IGNORECASE) for p in NOT_COMBO_PATTERNS]
_COMPILED_DRAWING_TYPES: Dict[str, List[re.Pattern]] = {
    dtype: [re.compile(p, re.IGNORECASE) for p in patterns]
    for dtype, patterns in DRAWING_TYPES.items()
}


# =============================================================================
# TITLE PARSER
# =============================================================================

class TitleParser:
    """
    Parser for sheet titles with combo page detection.

    Detects when a sheet contains multiple drawings and extracts
    each component for individual classification.
    """

    def __init__(self):
        """Initialize the parser."""
        # Pre-compile separator pattern
        self._separator_pattern = re.compile(
            '|'.join(f'({sep})' for sep in COMBO_SEPARATORS),
            re.IGNORECASE
        )

    def parse(self, title: Optional[str]) -> List[TitleComponent]:
        """
        Parse a sheet title into components.

        Args:
            title: Raw sheet title string

        Returns:
            List of TitleComponent objects (one for simple titles,
            multiple for combo pages)
        """
        if not title:
            return []

        # Clean the title
        clean_title = self._clean_title(title)

        # Check if this looks like a combo page
        if self._is_combo_title(clean_title):
            components = self._split_combo(clean_title)
        else:
            components = [TitleComponent(text=clean_title, original_position=0)]

        # Detect drawing type for each component
        for i, component in enumerate(components):
            component.drawing_type = self._detect_drawing_type(component.text)
            component.original_position = i

        return components

    def is_combo_page(self, title: Optional[str]) -> bool:
        """
        Check if a title represents a combo page.

        Args:
            title: Raw sheet title string

        Returns:
            True if this is a combo page
        """
        if not title:
            return False

        clean_title = self._clean_title(title)
        return self._is_combo_title(clean_title)

    def get_drawing_types(self, title: Optional[str]) -> List[str]:
        """
        Get all drawing types found in a title.

        Args:
            title: Raw sheet title string

        Returns:
            List of detected drawing type names
        """
        components = self.parse(title)
        return [c.drawing_type for c in components if c.drawing_type]

    def _clean_title(self, title: str) -> str:
        """Clean and normalize a title string."""
        # Preserve newlines for combo detection, but clean other whitespace
        # First, normalize line endings
        clean = title.strip()
        # Replace multiple spaces (but not newlines) with single space
        clean = re.sub(r'[ \t]+', ' ', clean)
        return clean

    def _is_combo_title(self, title: str) -> bool:
        """
        Determine if a title represents multiple drawings.

        Args:
            title: Cleaned title string

        Returns:
            True if this appears to be a combo page
        """
        # First, check if any NOT_COMBO patterns match (these are false positives)
        for pattern in _COMPILED_NOT_COMBO:
            if pattern.search(title):
                # Contains a pattern that looks like a separator but isn't
                # Need to check if there are OTHER separators
                pass  # Continue checking

        # Look for actual separators
        if self._separator_pattern.search(title):
            # Found a separator - but is it a real combo?
            parts = self._separator_pattern.split(title)
            # Filter out None and empty parts
            parts = [p for p in parts if p and p.strip() and not self._is_separator_only(p)]

            if len(parts) >= 2:
                # Check if at least 2 parts have distinct drawing types
                types_found = set()
                for part in parts:
                    dtype = self._detect_drawing_type(part)
                    if dtype:
                        types_found.add(dtype)

                # It's a combo if we found 2+ distinct drawing types
                # OR if we have 2+ substantial parts (even without detected types)
                if len(types_found) >= 2:
                    return True

                # Also check if both parts are substantial (5+ chars each)
                substantial_parts = [p for p in parts if len(p.strip()) >= 5]
                if len(substantial_parts) >= 2:
                    # Check that it's not a false positive
                    full_text = ' '.join(parts)
                    for pattern in _COMPILED_NOT_COMBO:
                        if pattern.search(full_text):
                            return False
                    return True

        return False

    def _split_combo(self, title: str) -> List[TitleComponent]:
        """
        Split a combo title into components.

        Args:
            title: Cleaned title string

        Returns:
            List of TitleComponent objects
        """
        # Split by separators
        parts = self._separator_pattern.split(title)

        # Filter and clean parts
        components = []
        for i, part in enumerate(parts):
            if part and part.strip() and not self._is_separator_only(part):
                clean_part = part.strip()
                if len(clean_part) >= 3:  # Minimum meaningful length
                    components.append(TitleComponent(
                        text=clean_part,
                        original_position=i,
                    ))

        # If we end up with just one part, return it as single title
        if len(components) <= 1:
            return [TitleComponent(text=title, original_position=0)]

        return components

    def _is_separator_only(self, text: str) -> bool:
        """Check if text is just a separator (AND, &, /, etc.)."""
        clean = text.strip().upper()
        return clean in ('AND', '&', '/', '-', '--', '\n')

    def _detect_drawing_type(self, text: str) -> Optional[str]:
        """
        Detect the drawing type from title text.

        Args:
            text: Title text to analyze

        Returns:
            Drawing type name or None if not detected
        """
        text_upper = text.upper()

        for dtype, patterns in _COMPILED_DRAWING_TYPES.items():
            for pattern in patterns:
                if pattern.search(text_upper):
                    return dtype

        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def split_combo_title(title: Optional[str]) -> List[str]:
    """
    Convenience function to split a combo title into text parts.

    Args:
        title: Raw sheet title string

    Returns:
        List of title text strings
    """
    parser = TitleParser()
    components = parser.parse(title)
    return [c.text for c in components]


def detect_drawing_type(title: Optional[str]) -> Optional[str]:
    """
    Convenience function to detect primary drawing type.

    Args:
        title: Raw sheet title string

    Returns:
        Drawing type name or None
    """
    if not title:
        return None

    parser = TitleParser()
    components = parser.parse(title)

    # Return first detected type
    for component in components:
        if component.drawing_type:
            return component.drawing_type

    return None


def is_combo_page(title: Optional[str]) -> bool:
    """
    Convenience function to check if title represents combo page.

    Args:
        title: Raw sheet title string

    Returns:
        True if combo page
    """
    parser = TitleParser()
    return parser.is_combo_page(title)
