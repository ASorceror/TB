"""
NCS Sheet Number Parser Module (V7.0)

Parses sheet numbers according to NCS (National CAD Standard) conventions
and determines painting trade relevance based on discipline and sheet type.

NCS Sheet Number Format: Discipline-SheetType-Sequence
Examples:
    A101  -> Architectural, Plans, Sheet 01
    A-201 -> Architectural, Elevations, Sheet 01
    I-221 -> Interiors, Elevations, Sheet 21
    S301  -> Structural, Sections, Sheet 01
    M101  -> Mechanical, Plans, Sheet 01

Discipline Designators (per NCS):
    A = Architectural
    I = Interiors
    S = Structural
    M = Mechanical
    E = Electrical
    P = Plumbing
    C = Civil
    L = Landscape
    G = General
    Q = Equipment
    F = Fire Protection
    T = Telecommunications

Sheet Type Designators (per NCS):
    0 = General (cover sheets, notes, symbols)
    1 = Plans (floor plans, site plans)
    2 = Elevations
    3 = Sections
    4 = Large-Scale Views (enlarged plans)
    5 = Details
    6 = Schedules and Diagrams
    7 = User-Defined
    8 = User-Defined
    9 = 3D Representations

This module provides:
    - NCSSheetParser: Parse sheet numbers into components
    - get_painting_relevance: Determine relevance for painting trade
"""

import re
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from .classification_types import RelevanceLevel


# =============================================================================
# NCS DISCIPLINE DEFINITIONS
# =============================================================================

class Discipline(Enum):
    """NCS Discipline Designators."""
    ARCHITECTURAL = "A"
    INTERIORS = "I"
    STRUCTURAL = "S"
    MECHANICAL = "M"
    ELECTRICAL = "E"
    PLUMBING = "P"
    CIVIL = "C"
    LANDSCAPE = "L"
    GENERAL = "G"
    EQUIPMENT = "Q"
    FIRE_PROTECTION = "F"
    TELECOMMUNICATIONS = "T"
    UNKNOWN = "?"


# Human-readable names for disciplines
DISCIPLINE_NAMES: Dict[str, str] = {
    "A": "Architectural",
    "I": "Interiors",
    "S": "Structural",
    "M": "Mechanical",
    "E": "Electrical",
    "P": "Plumbing",
    "C": "Civil",
    "L": "Landscape",
    "G": "General",
    "Q": "Equipment",
    "F": "Fire Protection",
    "T": "Telecommunications",
}


# =============================================================================
# NCS SHEET TYPE DEFINITIONS
# =============================================================================

class SheetType(Enum):
    """NCS Sheet Type Designators."""
    GENERAL = 0      # Cover sheets, notes, legends, symbols
    PLANS = 1        # Floor plans, site plans, roof plans
    ELEVATIONS = 2   # Interior and exterior elevations
    SECTIONS = 3     # Building sections, wall sections
    LARGE_SCALE = 4  # Enlarged plans, toilet rooms, stairs
    DETAILS = 5      # Construction details
    SCHEDULES = 6    # Schedules and diagrams
    USER_DEFINED_7 = 7
    USER_DEFINED_8 = 8
    THREE_D = 9      # 3D representations, renderings
    UNKNOWN = -1


# Human-readable names for sheet types
SHEET_TYPE_NAMES: Dict[int, str] = {
    0: "General",
    1: "Plans",
    2: "Elevations",
    3: "Sections",
    4: "Large-Scale Views",
    5: "Details",
    6: "Schedules",
    7: "User-Defined",
    8: "User-Defined",
    9: "3D Representations",
}


# =============================================================================
# PAINTING TRADE RELEVANCE MAPPING
# =============================================================================

# Relevance by discipline for painting trade
# Format: (discipline, sheet_type) -> RelevanceLevel
# None for sheet_type means "any sheet type for this discipline"
PAINTING_RELEVANCE_MAP: Dict[Tuple[str, Optional[int]], RelevanceLevel] = {
    # IRRELEVANT - Never needed for painting
    ("S", None): RelevanceLevel.IRRELEVANT,  # All Structural
    ("M", None): RelevanceLevel.IRRELEVANT,  # All Mechanical
    ("E", None): RelevanceLevel.IRRELEVANT,  # All Electrical
    ("P", None): RelevanceLevel.IRRELEVANT,  # All Plumbing
    ("C", None): RelevanceLevel.IRRELEVANT,  # All Civil
    ("L", None): RelevanceLevel.IRRELEVANT,  # All Landscape
    ("F", None): RelevanceLevel.IRRELEVANT,  # All Fire Protection
    ("T", None): RelevanceLevel.IRRELEVANT,  # All Telecommunications
    ("Q", None): RelevanceLevel.IRRELEVANT,  # All Equipment

    # PRIMARY - Essential for painting
    ("A", 1): RelevanceLevel.PRIMARY,   # Architectural Plans (floor plans)
    ("A", 6): RelevanceLevel.PRIMARY,   # Architectural Schedules (finish schedules)
    ("I", 1): RelevanceLevel.PRIMARY,   # Interior Plans
    ("I", 2): RelevanceLevel.PRIMARY,   # Interior Elevations
    ("I", 6): RelevanceLevel.PRIMARY,   # Interior Schedules (finish schedules)

    # SECONDARY - Important reference
    ("A", 2): RelevanceLevel.SECONDARY,  # Architectural Elevations
    ("A", 3): RelevanceLevel.SECONDARY,  # Architectural Sections
    ("A", 4): RelevanceLevel.SECONDARY,  # Enlarged Plans
    ("I", 3): RelevanceLevel.SECONDARY,  # Interior Sections
    ("I", 4): RelevanceLevel.SECONDARY,  # Interior Large-Scale
    ("I", 5): RelevanceLevel.SECONDARY,  # Interior Details

    # REFERENCE - Useful context
    ("A", 0): RelevanceLevel.REFERENCE,  # Architectural General
    ("A", 5): RelevanceLevel.REFERENCE,  # Architectural Details
    ("G", 0): RelevanceLevel.REFERENCE,  # General Cover/Notes
    ("G", None): RelevanceLevel.REFERENCE,  # All General
    ("I", 0): RelevanceLevel.REFERENCE,  # Interior General
}


# =============================================================================
# SHEET NUMBER PARSING
# =============================================================================

@dataclass
class ParsedSheetNumber:
    """
    Parsed components of an NCS sheet number.

    Attributes:
        original: Original sheet number string
        discipline: Discipline letter (A, I, S, M, E, P, C, L, G, Q, F, T)
        discipline_name: Human-readable discipline name
        sheet_type: Sheet type number (0-9)
        sheet_type_name: Human-readable sheet type name
        sequence: Sheet sequence number
        sub_sequence: Sub-sequence (after decimal, e.g., A101.1 -> 1)
        is_valid_ncs: Whether this follows NCS conventions
        relevance: Painting trade relevance level
    """
    original: str
    discipline: str
    discipline_name: str
    sheet_type: int
    sheet_type_name: str
    sequence: int
    sub_sequence: Optional[int] = None
    is_valid_ncs: bool = True
    relevance: RelevanceLevel = RelevanceLevel.REFERENCE


class NCSSheetParser:
    """
    Parser for NCS (National CAD Standard) sheet numbers.

    Supports various common formats:
        - A101, A-101, A.101 (with separators)
        - A1.01, A101.1 (with decimals)
        - AD101 (two-letter discipline codes)
        - I-221 (interior discipline)
    """

    # Regex patterns for sheet number parsing
    # Pattern: 1-2 letters + optional separator + 1-4 digits + optional decimal
    PATTERNS = [
        # Standard: A101, A-101, A.101, AD101
        re.compile(
            r'^(?P<discipline>[A-Z]{1,2})[-.]?(?P<type>\d)(?P<seq>\d{1,3})(?:\.(?P<sub>\d{1,2}))?$',
            re.IGNORECASE
        ),
        # Alternative: A1.01 (type and sequence separated by decimal)
        re.compile(
            r'^(?P<discipline>[A-Z]{1,2})[-.]?(?P<type>\d)\.(?P<seq>\d{1,2})$',
            re.IGNORECASE
        ),
    ]

    def __init__(self):
        """Initialize the parser."""
        pass

    def parse(self, sheet_number: Optional[str]) -> Optional[ParsedSheetNumber]:
        """
        Parse a sheet number into its NCS components.

        Args:
            sheet_number: Raw sheet number string (e.g., "A101", "I-221")

        Returns:
            ParsedSheetNumber if valid, None if cannot parse
        """
        if not sheet_number:
            return None

        # Clean the input
        clean = sheet_number.strip().upper()

        # Try each pattern
        for pattern in self.PATTERNS:
            match = pattern.match(clean)
            if match:
                groups = match.groupdict()

                # Extract discipline (use first letter for relevance lookup)
                discipline_raw = groups['discipline']
                discipline = discipline_raw[0]  # First letter is the main discipline
                discipline_name = DISCIPLINE_NAMES.get(discipline, "Unknown")

                # Extract sheet type
                sheet_type = int(groups['type'])
                sheet_type_name = SHEET_TYPE_NAMES.get(sheet_type, "Unknown")

                # Extract sequence
                seq_str = groups.get('seq', '0')
                sequence = int(seq_str) if seq_str else 0

                # Extract sub-sequence
                sub_str = groups.get('sub')
                sub_sequence = int(sub_str) if sub_str else None

                # Determine relevance
                relevance = self._get_relevance(discipline, sheet_type)

                return ParsedSheetNumber(
                    original=sheet_number,
                    discipline=discipline,
                    discipline_name=discipline_name,
                    sheet_type=sheet_type,
                    sheet_type_name=sheet_type_name,
                    sequence=sequence,
                    sub_sequence=sub_sequence,
                    is_valid_ncs=True,
                    relevance=relevance,
                )

        # Could not parse - return minimal info
        return self._fallback_parse(sheet_number)

    def _get_relevance(self, discipline: str, sheet_type: int) -> RelevanceLevel:
        """
        Determine painting trade relevance from discipline and sheet type.

        Args:
            discipline: Single-letter discipline code
            sheet_type: Sheet type number (0-9)

        Returns:
            RelevanceLevel for painting trade
        """
        # Check specific (discipline, sheet_type) mapping first
        key = (discipline, sheet_type)
        if key in PAINTING_RELEVANCE_MAP:
            return PAINTING_RELEVANCE_MAP[key]

        # Check discipline-only mapping (any sheet type)
        key_any = (discipline, None)
        if key_any in PAINTING_RELEVANCE_MAP:
            return PAINTING_RELEVANCE_MAP[key_any]

        # Default to REFERENCE for unknown combinations
        return RelevanceLevel.REFERENCE

    def _fallback_parse(self, sheet_number: str) -> Optional[ParsedSheetNumber]:
        """
        Fallback parsing for non-standard sheet numbers.

        Tries to extract at least the discipline letter.

        Args:
            sheet_number: Raw sheet number string

        Returns:
            ParsedSheetNumber with partial info, or None
        """
        clean = sheet_number.strip().upper()

        # Try to get first letter as discipline
        if clean and clean[0].isalpha():
            discipline = clean[0]
            discipline_name = DISCIPLINE_NAMES.get(discipline, "Unknown")

            # Check if it's a known irrelevant discipline
            if discipline in ('S', 'M', 'E', 'P', 'C', 'L', 'F', 'T', 'Q'):
                relevance = RelevanceLevel.IRRELEVANT
            elif discipline in ('A', 'I'):
                relevance = RelevanceLevel.SECONDARY  # Can't determine sheet type
            else:
                relevance = RelevanceLevel.REFERENCE

            return ParsedSheetNumber(
                original=sheet_number,
                discipline=discipline,
                discipline_name=discipline_name,
                sheet_type=-1,  # Unknown
                sheet_type_name="Unknown",
                sequence=0,
                is_valid_ncs=False,
                relevance=relevance,
            )

        return None

    def get_category_hint(self, parsed: ParsedSheetNumber) -> Optional[str]:
        """
        Get a category hint based on NCS parsing.

        Args:
            parsed: ParsedSheetNumber from parse()

        Returns:
            Category name hint or None
        """
        if not parsed or not parsed.is_valid_ncs:
            return None

        # Map (discipline, sheet_type) to category hints
        if parsed.discipline in ('A', 'I') and parsed.sheet_type == 1:
            return 'floor_plans'
        elif parsed.discipline in ('A', 'I') and parsed.sheet_type == 2:
            # Could be interior or exterior - need title to disambiguate
            return 'elevations'  # Generic
        elif parsed.discipline in ('A', 'I') and parsed.sheet_type == 6:
            return 'room_finish_schedules'
        elif parsed.discipline in ('A', 'I') and parsed.sheet_type == 0:
            return 'cover_sheets'
        elif parsed.discipline in ('S', 'M', 'E', 'P', 'C', 'L'):
            return 'properly_classified_not_needed'

        return None


def get_painting_relevance(sheet_number: Optional[str]) -> RelevanceLevel:
    """
    Convenience function to get painting trade relevance from sheet number.

    Args:
        sheet_number: Raw sheet number string

    Returns:
        RelevanceLevel for painting trade
    """
    parser = NCSSheetParser()
    parsed = parser.parse(sheet_number)

    if parsed:
        return parsed.relevance

    return RelevanceLevel.REFERENCE  # Default for unparseable
