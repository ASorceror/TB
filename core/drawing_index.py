"""
Blueprint Processor V4.2.1 - Drawing Index Parser
Parses drawing index tables from cover sheets to map sheet numbers to titles.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from difflib import SequenceMatcher

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    DISCIPLINE_CODES,
    DRAWING_INDEX_KEYWORDS,
    DISCIPLINE_HEADERS,
    FUZZY_MATCH_THRESHOLD,
    MAX_INDEX_SEARCH_PAGES,
)

logger = logging.getLogger(__name__)

# Sheet number pattern: A1.0, A-101, S201, M1.01, AD001, etc.
SHEET_NUMBER_PATTERN = re.compile(
    r'^([A-Z]{1,2})[-.]?(\d{1,3})(?:[-.](\d{1,2}))?$',
    re.IGNORECASE
)

# More flexible pattern for OCR-damaged text
# Handles: Az2 -> A2.2, A20 -> A2.0, Az} -> A2.1
SHEET_NUMBER_FLEXIBLE = re.compile(
    r'^([A-Za-z]{1,2})[-.]?([0-9OoIlzZ\}\]\|]{1,3})(?:[-.]([0-9OoIlzZ\}\]\|]{1,2}))?$'
)

# Additional OCR character corrections for numeric parts
OCR_CHAR_FIXES = {
    'O': '0', 'o': '0',
    'I': '1', 'l': '1', '|': '1',
    'z': '2', 'Z': '2',
    '}': '1', ']': '1',
    '{': '1', '[': '1',
    'S': '5', 's': '5',
    'B': '8',
}

# Additional transformations for letter+digit combinations
# Handles cases like "AZ1" -> "A2.1" where Z was misread as the second letter
OCR_PREFIX_FIXES = {
    'AZ': 'A2.',  # AZ1 -> A2.1, AZ2 -> A2.2
    'DZ': 'D2.',  # DZ0 -> D2.0
    'SZ': 'S2.',
    'MZ': 'M2.',
}


class DrawingIndexParser:
    """
    Parses drawing index tables from blueprint cover sheets.
    Extracts mappings from sheet numbers to sheet titles.
    """

    def __init__(self, ocr_engine=None):
        """
        Initialize the DrawingIndexParser.

        Args:
            ocr_engine: Optional OCREngine instance for scanned pages
        """
        self._ocr_engine = ocr_engine
        self._cache: Dict[str, Dict[str, str]] = {}

    def parse_from_pdf(self, pdf_handler, max_pages: int = MAX_INDEX_SEARCH_PAGES) -> Dict[str, str]:
        """
        Parse drawing index from a PDF.
        Searches first max_pages pages for index section.

        Args:
            pdf_handler: PDFHandler instance
            max_pages: Maximum number of pages to search (default 5)

        Returns:
            Dict mapping sheet_number -> title
            Example: {'A1.0': 'Floor Plan', 'A2.0': 'Elevations'}
        """
        pdf_path = str(pdf_handler.pdf_path)

        # Check cache
        if pdf_path in self._cache:
            logger.debug(f"Using cached drawing index for {pdf_path}")
            return self._cache[pdf_path]

        result = {}
        pages_to_search = min(max_pages, pdf_handler.page_count)

        for page_num in range(pages_to_search):
            # Try vector-first approach
            text = pdf_handler.get_page_text(page_num)

            if len(text) >= 50:
                # Vector page - use embedded text
                logger.debug(f"Page {page_num + 1}: Using embedded text ({len(text)} chars)")
                page_result = self._parse_index_from_text(text)
            else:
                # Scanned page - try OCR if available
                if self._ocr_engine:
                    logger.debug(f"Page {page_num + 1}: Using OCR (embedded text < 50 chars)")
                    try:
                        page_image = pdf_handler.get_page_image(page_num, dpi=200)
                        text = self._ocr_engine.ocr_image(page_image)
                        page_result = self._parse_index_from_text(text)
                    except Exception as e:
                        logger.warning(f"OCR failed on page {page_num + 1}: {e}")
                        page_result = {}
                else:
                    logger.debug(f"Page {page_num + 1}: Skipping (scanned, no OCR engine)")
                    page_result = {}

            # Merge results
            if page_result:
                logger.info(f"Found {len(page_result)} index entries on page {page_num + 1}")
                result.update(page_result)

        # Cache the result
        self._cache[pdf_path] = result

        if result:
            logger.info(f"Drawing index complete: {len(result)} total entries")
        else:
            logger.info("No drawing index found in PDF")

        return result

    def _parse_index_from_text(self, text: str) -> Dict[str, str]:
        """
        Parse drawing index entries from text.

        Args:
            text: Text content (from vector extraction or OCR)

        Returns:
            Dict mapping sheet_number -> title
        """
        if not text:
            return {}

        # Check if text contains index keywords
        text_upper = text.upper()
        has_index_keyword = any(kw in text_upper for kw in DRAWING_INDEX_KEYWORDS)

        if not has_index_keyword:
            return {}

        logger.debug("Found drawing index keyword in text")

        result = {}

        # First, try line-by-line parsing (works best for clean vector text)
        lines = text.split('\n')
        current_discipline = None
        previous_sheet = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip index header keywords
            line_upper = line.upper()
            if any(kw in line_upper for kw in DRAWING_INDEX_KEYWORDS):
                continue

            # Skip common preamble text
            if self._is_preamble_line(line_upper):
                continue

            # Check for discipline header
            discipline = self._extract_discipline_header(line_upper)
            if discipline:
                current_discipline = discipline
                logger.debug(f"Discipline header: {discipline}")
                continue

            # Try to parse as sheet entry
            sheet_entry = self._parse_sheet_line(line)
            if sheet_entry:
                sheet_number, title = sheet_entry
                result[sheet_number] = title
                previous_sheet = sheet_number
                logger.debug(f"Parsed entry: {sheet_number} -> {title}")
            elif previous_sheet and self._is_continuation_line(line):
                # Multi-line title continuation
                result[previous_sheet] = result[previous_sheet] + ' ' + line.strip()
                logger.debug(f"Appended to {previous_sheet}: {line.strip()}")

        # If line-by-line didn't find much, try pattern-based extraction
        # (for OCR text where line breaks aren't preserved)
        if len(result) < 3:
            pattern_results = self._extract_embedded_entries(text)
            for sheet, title in pattern_results.items():
                if sheet not in result:
                    result[sheet] = title
                    logger.debug(f"Pattern-extracted: {sheet} -> {title}")

        return result

    def _extract_embedded_entries(self, text: str) -> Dict[str, str]:
        """
        Extract sheet entries that are embedded in text (OCR without line breaks).

        Looks for patterns like "A20 FLOOR PLAN" or "Az2 REFLECTED CEILING PLAN"
        embedded within longer text.

        Args:
            text: Full OCR text

        Returns:
            Dict mapping sheet_number -> title
        """
        result = {}

        # Full title patterns to search for (longer/more specific first)
        title_patterns = [
            # Multi-word titles
            r'CODE\s+REVIEW[,\s]+SCHEDULES?\s+AND\s+ADA\s+REQUIREMENTS?',
            r'REFLECTED\s+CEILING\s+PLAN',
            r'DEMOLITION\s+FLOOR\s+PLAN',
            r'ENLARGED\s+FLOOR\s+PLAN',
            r'LIFE\s+SAFETY\s+PLAN',
            r'FLOOR\s+PLAN',
            r'CEILING\s+PLAN',
            r'ROOF\s+PLAN',
            r'SITE\s+PLAN',
            r'FOUNDATION\s+PLAN',
            # Single words
            r'ELEVATION[S]?',
            r'SECTION[S]?',
            r'DETAIL[S]?',
            r'SCHEDULE[S]?',
            r'NOTES?',
            r'SPECIFICATIONS?',
            r'PLAN',
        ]

        for title_pattern in title_patterns:
            # Pattern: sheet-like prefix + title
            # Sheet prefix: 1-2 letters + optional separator + digits with OCR chars
            pattern = (
                r'([A-Za-z]{1,2}[-.]?[0-9OoIlzZ\}\]\|]{1,3}(?:[-.]?[0-9OoIlzZ\}\]\|]{1,2})?)'
                r'\s+'
                r'(' + title_pattern + r')'
            )

            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                raw_sheet = match.group(1)
                raw_title = match.group(2).strip()

                normalized = self._normalize_sheet_number(raw_sheet)
                if normalized and len(raw_title) >= 4:
                    # Clean the title
                    clean = self._clean_title(raw_title)
                    if clean and normalized not in result:
                        result[normalized] = clean

        return result

    def _is_preamble_line(self, line_upper: str) -> bool:
        """Check if line is preamble text to skip."""
        preamble_patterns = [
            'THE FOLLOWING SHEETS',
            'COMPLETE SET',
            'CONSTRUCTION DOCUMENTS',
            'COVERSHEET',
            'COVER SHEET',
            'THIS SET',
            'THIS DOCUMENT',
            'ALL DRAWINGS',
        ]
        return any(p in line_upper for p in preamble_patterns)

    def _extract_discipline_header(self, line_upper: str) -> Optional[str]:
        """Check if line is a discipline header."""
        line_clean = line_upper.strip()
        for discipline in DISCIPLINE_HEADERS:
            if line_clean == discipline or line_clean.startswith(discipline + ' '):
                return discipline
        return None

    def _parse_sheet_line(self, line: str) -> Optional[Tuple[str, str]]:
        """
        Parse a line as a sheet number + title entry.

        Handles formats:
        - A1.0    FLOOR PLAN (spaces)
        - A1.0 - FLOOR PLAN (dash)
        - A1.0\tFLOOR PLAN (tab)
        - A-101   FLOOR PLAN (with hyphen in number)
        - Az2 REFLECTED CEILING PLAN (OCR errors)
        - A20 FLOOR PLAN (missing decimal)

        Returns:
            Tuple of (sheet_number, title) or None
        """
        if not line or len(line) < 3:
            return None

        # Try various separators
        separators = ['\t', '    ', '   ', '  ', ' - ', ' – ', ' — ']

        for sep in separators:
            if sep in line:
                parts = line.split(sep, 1)
                if len(parts) == 2:
                    potential_sheet = parts[0].strip()
                    potential_title = parts[1].strip()

                    # Normalize the sheet number
                    normalized = self._normalize_sheet_number(potential_sheet)
                    if normalized and potential_title:
                        # Clean the title
                        clean_title = self._clean_title(potential_title)
                        if clean_title:
                            return (normalized, clean_title)

        # Try flexible regex match at start of line (handles OCR errors like Az2, A20)
        # Pattern allows: A-Za-z prefix + optional separator + alphanumeric with OCR chars
        match = re.match(
            r'^([A-Za-z]{1,2}[-.]?[0-9OoIlzZ\}\]\|]{1,3}(?:[-.]?[0-9OoIlzZ\}\]\|]{1,2})?)\s+(.+)$',
            line
        )
        if match:
            potential_sheet = match.group(1).strip()
            potential_title = match.group(2).strip()

            normalized = self._normalize_sheet_number(potential_sheet)
            if normalized and potential_title:
                clean_title = self._clean_title(potential_title)
                if clean_title:
                    return (normalized, clean_title)

        # Also try standard pattern
        match = re.match(r'^([A-Z]{1,2}[-.]?\d{1,3}(?:[-.]?\d{1,2})?)\s+(.+)$', line, re.IGNORECASE)
        if match:
            potential_sheet = match.group(1).strip()
            potential_title = match.group(2).strip()

            normalized = self._normalize_sheet_number(potential_sheet)
            if normalized and potential_title:
                clean_title = self._clean_title(potential_title)
                if clean_title:
                    return (normalized, clean_title)

        return None

    def _normalize_sheet_number(self, raw: str) -> Optional[str]:
        """
        Normalize a sheet number, fixing common OCR errors.

        Args:
            raw: Raw sheet number string

        Returns:
            Normalized sheet number or None if invalid
        """
        if not raw or len(raw) < 2:
            return None

        # Clean up whitespace
        s = raw.strip().upper()

        # Check for prefix fixes first (e.g., AZ1 -> A2.1)
        for ocr_prefix, fixed_prefix in OCR_PREFIX_FIXES.items():
            if s.startswith(ocr_prefix):
                remainder = s[len(ocr_prefix):]
                # Fix any remaining OCR errors in the remainder
                fixed_remainder = ''
                for char in remainder:
                    if char in OCR_CHAR_FIXES:
                        fixed_remainder += OCR_CHAR_FIXES[char]
                    elif char.isdigit() or char in '.-':
                        fixed_remainder += char

                if fixed_remainder:
                    result = fixed_prefix + fixed_remainder
                    if SHEET_NUMBER_PATTERN.match(result):
                        return result

        # Standard parsing
        match = re.match(r'^([A-Za-z]{1,2})([-.]?)(.+)$', s)
        if not match:
            return None

        prefix = match.group(1).upper()
        separator = match.group(2)
        numeric_part = match.group(3)

        # Fix OCR errors in numeric part using extended fixes
        fixed_numeric = ''
        for char in numeric_part:
            if char in OCR_CHAR_FIXES:
                fixed_numeric += OCR_CHAR_FIXES[char]
            elif char.isdigit() or char in '.-':
                fixed_numeric += char
            # Skip other characters

        if not fixed_numeric:
            return None

        # Try to add decimal if missing (e.g., A20 -> A2.0, Az2 -> A2.2)
        if '.' not in fixed_numeric and len(fixed_numeric) >= 2:
            # Common pattern: last digit is after decimal
            fixed_with_decimal = fixed_numeric[:-1] + '.' + fixed_numeric[-1]
        else:
            fixed_with_decimal = fixed_numeric

        # Try various formats
        candidates = [
            prefix + separator + fixed_numeric,
            prefix + fixed_numeric,
            prefix + separator + fixed_with_decimal,
            prefix + fixed_with_decimal,
        ]

        for candidate in candidates:
            if SHEET_NUMBER_PATTERN.match(candidate):
                return candidate

        # If still no match, return the best guess
        best = prefix + fixed_with_decimal
        if len(best) >= 3 and best[0].isalpha():
            return best

        return None

    def _clean_title(self, title: str) -> Optional[str]:
        """
        Clean a title string.

        Args:
            title: Raw title string

        Returns:
            Cleaned title or None if invalid
        """
        if not title:
            return None

        # Remove leading/trailing punctuation and whitespace
        clean = re.sub(r'^[\s\-:]+', '', title)
        clean = re.sub(r'[\s\-:]+$', '', clean)
        clean = re.sub(r'\s+', ' ', clean)

        if len(clean) < 3:
            return None

        return clean.strip()

    def _is_continuation_line(self, line: str) -> bool:
        """
        Check if a line is a continuation of the previous title.

        A continuation line:
        - Does NOT start with a sheet number pattern
        - Has meaningful text content
        - Is not a discipline header
        """
        if not line or len(line.strip()) < 3:
            return False

        line_stripped = line.strip()
        line_upper = line_stripped.upper()

        # Check if it's a discipline header
        if self._extract_discipline_header(line_upper):
            return False

        # Check if it starts with a sheet number pattern
        if re.match(r'^[A-Z]{1,2}[-.]?\d', line_stripped, re.IGNORECASE):
            return False

        # Check if it's preamble
        if self._is_preamble_line(line_upper):
            return False

        # Has meaningful content
        return True

    def lookup(self, sheet_number: str, index: Dict[str, str]) -> Optional[str]:
        """
        Look up a sheet number in the index, with fuzzy matching.

        Args:
            sheet_number: The sheet number to look up
            index: The drawing index mapping

        Returns:
            The title if found, None otherwise
        """
        if not sheet_number or not index:
            return None

        # Normalize the input
        normalized = self._normalize_sheet_number(sheet_number)
        if not normalized:
            normalized = sheet_number.upper()

        # Exact match
        if normalized in index:
            return index[normalized]

        # Try without separator variations
        variants = self._generate_variants(normalized)
        for variant in variants:
            if variant in index:
                logger.debug(f"Fuzzy match: {sheet_number} -> {variant}")
                return index[variant]

        # Fuzzy match against all keys
        best_match = None
        best_ratio = 0.0

        for key in index.keys():
            ratio = SequenceMatcher(None, normalized, key).ratio()
            if ratio > best_ratio and ratio >= FUZZY_MATCH_THRESHOLD:
                best_ratio = ratio
                best_match = key

        if best_match:
            logger.debug(f"Fuzzy match ({best_ratio:.0%}): {sheet_number} -> {best_match}")
            return index[best_match]

        return None

    def _generate_variants(self, sheet_number: str) -> List[str]:
        """
        Generate variant forms of a sheet number.

        Args:
            sheet_number: Normalized sheet number

        Returns:
            List of variant forms
        """
        variants = []

        # Remove separator
        no_sep = re.sub(r'[-.]', '', sheet_number)
        variants.append(no_sep)

        # Add decimal if missing
        match = re.match(r'^([A-Z]{1,2})(\d+)$', sheet_number)
        if match:
            prefix = match.group(1)
            numbers = match.group(2)
            if len(numbers) >= 2:
                # Try inserting decimal
                variants.append(f"{prefix}{numbers[0]}.{numbers[1:]}")
                if len(numbers) >= 3:
                    variants.append(f"{prefix}{numbers[:2]}.{numbers[2:]}")

        # Add hyphen variant
        match = re.match(r'^([A-Z]{1,2})(\d.*)$', sheet_number)
        if match:
            variants.append(f"{match.group(1)}-{match.group(2)}")

        return variants

    def clear_cache(self):
        """Clear the drawing index cache."""
        self._cache.clear()
        logger.debug("Drawing index cache cleared")
