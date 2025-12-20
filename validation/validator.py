"""
Blueprint Processor V4.4 - Validator
Validates extracted field data including sheet title validation.

V4.4 Changes:
- Accept single-word titles for construction terms (VALID_SINGLE_WORD_TITLES)
- Truncate long titles instead of rejecting
- Removed aggressive proper noun pattern rejection
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    PATTERNS, TITLE_GARBAGE_PATTERNS, VALID_TITLE_KEYWORDS,
    TITLE_ACRONYMS, TITLE_LENGTH, TITLE_CONFIDENCE, REVIEW_THRESHOLD,
    QUALITY_SINGLE_KEYWORD_REJECTS, QUALITY_GATE_THRESHOLDS,
    SHEET_NUMBER_BLACKLIST, SHEET_NUMBER_STRICT_PATTERN,
    VALID_SINGLE_WORD_TITLES  # V4.4
)

logger = logging.getLogger(__name__)


class Validator:
    """
    Validates extracted blueprint data.
    Checks format compliance and detects contamination.
    """

    def __init__(self):
        """Initialize the Validator."""
        pass

    def is_valid_sheet_number(self, value: str, log_rejection: bool = False) -> Tuple[bool, str]:
        """
        Validate a sheet number candidate (V4.2.3).

        A valid sheet number:
        1. Is not None or empty
        2. Is 2-10 characters long
        3. Starts with a letter A-Z
        4. Contains at least one digit (0-9) - CRITICAL
        5. Is NOT in the blacklist
        6. Matches the strict regex pattern

        Args:
            value: The sheet number candidate to validate
            log_rejection: If True, log rejected candidates for debugging

        Returns:
            Tuple of (is_valid: bool, reason: str)
        """
        # Check None/empty
        if not value:
            return (False, "empty_or_none")

        value_clean = value.strip().upper()

        # Check length (2-10 characters)
        if len(value_clean) < 2:
            return (False, f"too_short:{len(value_clean)}")
        if len(value_clean) > 10:
            return (False, f"too_long:{len(value_clean)}")

        # Must start with a letter A-Z
        if not value_clean[0].isalpha():
            return (False, "no_letter_prefix")

        # CRITICAL: Must contain at least one digit (0-9)
        if not any(c.isdigit() for c in value_clean):
            return (False, "no_digits")

        # Check blacklist
        if value_clean in SHEET_NUMBER_BLACKLIST:
            return (False, f"blacklisted:{value_clean}")

        # Also check with periods stripped for blacklist
        value_no_periods = value_clean.replace(".", "")
        if value_no_periods in SHEET_NUMBER_BLACKLIST:
            return (False, f"blacklisted:{value_no_periods}")

        # Check strict pattern match
        if not SHEET_NUMBER_STRICT_PATTERN.match(value_clean):
            return (False, "pattern_mismatch")

        return (True, "valid")

    def is_quality_title(self, title: str, project_number: str = None) -> Tuple[bool, str]:
        """
        Quality Gate: Validation for extracted titles (V4.4).

        V4.4 Changes:
        - Accept single-word titles in VALID_SINGLE_WORD_TITLES
        - Removed aggressive proper noun pattern rejection
        - More lenient for multi-word titles

        Args:
            title: The extracted title to evaluate
            project_number: Optional project number to check for contamination

        Returns:
            Tuple of (is_quality: bool, reason: str)
        """
        # Handle empty/None
        if not title:
            return (False, "empty_title")

        title_clean = title.strip()

        if len(title_clean) < QUALITY_GATE_THRESHOLDS['min_length']:
            return (False, f"too_short:{len(title_clean)}")

        # V4.4: Don't reject long titles - they'll be truncated later
        # Just log a warning
        if len(title_clean) > QUALITY_GATE_THRESHOLDS['max_length']:
            logger.debug(f"Long title will be truncated: {len(title_clean)} chars")

        title_upper = title_clean.upper()

        # Check for contamination with project number
        if project_number and title_clean == project_number:
            return (False, "matches_project_number")

        # Check garbage patterns
        for pattern in TITLE_GARBAGE_PATTERNS:
            if pattern.search(title_clean):
                pattern_str = pattern.pattern[:30].replace('\\', '')
                return (False, f"garbage_pattern:{pattern_str}")

        # Check single keyword rejects (reduced list in V4.4)
        if title_upper in QUALITY_SINGLE_KEYWORD_REJECTS:
            return (False, f"single_keyword:{title_upper}")

        # V4.4: Accept single-word titles that are valid construction terms
        if title_upper in VALID_SINGLE_WORD_TITLES:
            return (True, f"valid_single_word:{title_upper}")

        # Check for valid keywords
        has_keyword = self._has_valid_keyword(title_clean)
        word_count = len(title_clean.split())

        # PASS: Has keyword AND has context (2+ words)
        if has_keyword and word_count >= 2:
            return (True, f"keyword_with_context:{word_count}_words")

        # PASS: Has keyword AND is long enough (single word but 15+ chars)
        if has_keyword and len(title_clean) >= QUALITY_GATE_THRESHOLDS['min_length_single_word']:
            return (True, f"keyword_long:{len(title_clean)}_chars")

        # PASS: Has keyword even if single word (V4.4 - more lenient)
        if has_keyword and word_count == 1:
            return (True, f"single_keyword_valid:{title_upper}")

        # V4.4: Accept multi-word titles without keyword check
        # Removed aggressive proper noun pattern rejection
        if word_count >= 2 and len(title_clean) >= 8:
            return (True, f"multi_word:{word_count}_words")

        # Default: reject only very short/generic titles
        return (False, "insufficient_context")

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data.

        Args:
            data: Dict with extracted fields

        Returns:
            Dict with keys: is_valid, errors, warnings, validated_data
        """
        errors = []
        warnings = []
        validated_data = data.copy()

        # Remove extraction_details from validated data
        if 'extraction_details' in validated_data:
            del validated_data['extraction_details']

        # Validate sheet_number
        sheet_number = data.get('sheet_number')
        if sheet_number:
            sheet_valid, sheet_errors, sheet_warnings = self._validate_sheet_number(sheet_number)
            errors.extend(sheet_errors)
            warnings.extend(sheet_warnings)
            if not sheet_valid:
                validated_data['sheet_number'] = None
        else:
            warnings.append("sheet_number is missing")

        # Validate project_number
        project_number = data.get('project_number')
        if project_number:
            proj_valid, proj_errors, proj_warnings = self._validate_project_number(project_number)
            errors.extend(proj_errors)
            warnings.extend(proj_warnings)
            if not proj_valid:
                validated_data['project_number'] = None
        else:
            warnings.append("project_number is missing")

        # Check for contamination (sheet_number == project_number)
        if sheet_number and project_number:
            if sheet_number == project_number:
                errors.append(f"Contamination detected: sheet_number and project_number are identical ({sheet_number})")
                # Clear the project_number as it's likely wrong
                validated_data['project_number'] = None

        # Validate date format
        date = data.get('date')
        if date:
            date_valid, date_errors, date_warnings = self._validate_date(date)
            errors.extend(date_errors)
            warnings.extend(date_warnings)

        if not (data.get('sheet_title') or '').strip():
            errors.append('sheet_title is missing')

        # Overall validity
        is_valid = len(errors) == 0

        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'validated_data': validated_data,
        }

    def _validate_sheet_number(self, sheet_number: str) -> tuple:
        """
        Validate sheet number format.
        Must start with a letter [A-Z].

        Args:
            sheet_number: Value to validate

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        if not sheet_number:
            return False, ["sheet_number is empty"], []

        # Must start with a letter
        if not sheet_number[0].isalpha():
            errors.append(f"sheet_number '{sheet_number}' must start with a letter [A-Z]")
            return False, errors, warnings

        # Should match pattern
        if not PATTERNS['sheet_number'].match(sheet_number.upper()):
            warnings.append(f"sheet_number '{sheet_number}' doesn't match expected pattern")

        return True, errors, warnings

    def _validate_project_number(self, project_number: str) -> tuple:
        """
        Validate project number format.
        Must be primarily numeric.

        Args:
            project_number: Value to validate

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        if not project_number:
            return False, ["project_number is empty"], []

        # Count digits
        digit_count = sum(1 for c in project_number if c.isdigit())
        total_chars = len(project_number.replace('-', '').replace('.', '').replace(' ', ''))

        if total_chars == 0:
            errors.append(f"project_number '{project_number}' has no valid characters")
            return False, errors, warnings

        digit_ratio = digit_count / total_chars

        # Should be primarily numeric (at least 70% digits)
        if digit_ratio < 0.7:
            errors.append(f"project_number '{project_number}' must be primarily numeric (currently {digit_ratio:.0%} digits)")
            return False, errors, warnings

        # Check for minimum length
        if digit_count < 4:
            warnings.append(f"project_number '{project_number}' is shorter than typical (< 4 digits)")

        return True, errors, warnings

    def _validate_date(self, date: str) -> tuple:
        """
        Validate date format.

        Args:
            date: Value to validate

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        if not date:
            return True, [], []  # Date is optional

        # Check pattern match
        if not PATTERNS['date'].match(date):
            warnings.append(f"date '{date}' doesn't match expected format (MM/DD/YYYY)")

        return True, errors, warnings

    def check_contamination(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for field contamination (values that got mixed up).

        Args:
            data: Dict with extracted fields

        Returns:
            Dict with contamination analysis
        """
        issues = []

        sheet = data.get('sheet_number', '')
        project = data.get('project_number', '')

        # Check if sheet_number looks like a project number
        if sheet and sheet[0].isdigit():
            issues.append({
                'field': 'sheet_number',
                'issue': 'starts_with_digit',
                'value': sheet,
                'suggestion': 'May be a project number instead',
            })

        # Check if project_number looks like a sheet number
        if project and project[0].isalpha():
            issues.append({
                'field': 'project_number',
                'issue': 'starts_with_letter',
                'value': project,
                'suggestion': 'May be a sheet number instead',
            })

        # Check if values are identical
        if sheet and project and sheet == project:
            issues.append({
                'field': 'both',
                'issue': 'identical_values',
                'value': sheet,
                'suggestion': 'Values are identical - likely extraction error',
            })

        return {
            'has_contamination': len(issues) > 0,
            'issues': issues,
        }

    def validate_sheet_title(
        self,
        title: Optional[str],
        method: str = 'pattern',
        project_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate and score a sheet title (V4.2.1).

        Args:
            title: The extracted title to validate
            method: Extraction method ('drawing_index', 'spatial', 'vision_api', 'pattern')
            project_number: Project number to check for contamination

        Returns:
            Dict with keys:
                - is_valid: bool
                - title: cleaned title or None
                - confidence: float 0.0-1.0
                - needs_review: bool
                - rejection_reason: str or None
        """
        # Handle empty/None titles
        if not title or not title.strip():
            return {
                'is_valid': False,
                'title': None,
                'confidence': 0.0,
                'needs_review': True,
                'rejection_reason': 'Empty or missing title'
            }

        title = title.strip()

        # Check for contamination with project number
        if project_number and title == project_number:
            return {
                'is_valid': False,
                'title': None,
                'confidence': 0.0,
                'needs_review': True,
                'rejection_reason': 'Title matches project number (contamination)'
            }

        # Check length constraints
        if len(title) < TITLE_LENGTH['MIN']:
            return {
                'is_valid': False,
                'title': None,
                'confidence': 0.0,
                'needs_review': True,
                'rejection_reason': f'Title too short ({len(title)} chars, min {TITLE_LENGTH["MIN"]})'
            }

        # V4.4: Truncate long titles instead of rejecting
        if len(title) > TITLE_LENGTH['MAX']:
            original_len = len(title)
            title = title[:TITLE_LENGTH['MAX'] - 3] + "..."
            logger.debug(f"Truncated title from {original_len} to {len(title)} chars")

        # Check for garbage patterns
        for pattern in TITLE_GARBAGE_PATTERNS:
            if pattern.search(title):
                return {
                    'is_valid': False,
                    'title': None,
                    'confidence': 0.0,
                    'needs_review': True,
                    'rejection_reason': f'Matches garbage pattern: {pattern.pattern}'
                }

        # Title passed validation - now normalize and score
        normalized_title = self.normalize_title_case(title)

        # Calculate confidence score
        base_confidence = TITLE_CONFIDENCE.get(method.upper(), TITLE_CONFIDENCE['PATTERN'])
        confidence = base_confidence

        # Check for valid keywords (confidence boost)
        has_keyword = self._has_valid_keyword(normalized_title)
        if has_keyword:
            confidence += TITLE_CONFIDENCE['KEYWORD_BONUS']

        # Length penalty for 46-70 chars
        if len(title) > TITLE_LENGTH['SOFT_MAX']:
            confidence += TITLE_CONFIDENCE['LONG_PENALTY']

        # Penalty for short titles without keywords
        if len(title) < 15 and not has_keyword:
            confidence += TITLE_CONFIDENCE['SHORT_NO_KEYWORD_PENALTY']

        # Clamp confidence to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))

        # Determine if needs review
        needs_review = confidence < REVIEW_THRESHOLD

        return {
            'is_valid': True,
            'title': normalized_title,
            'confidence': confidence,
            'needs_review': needs_review,
            'rejection_reason': None
        }

    def _has_valid_keyword(self, title: str) -> bool:
        """
        Check if title contains a valid keyword.
        V4.2.4: Handle plural forms (keyword + 'S')
        """
        title_upper = title.upper()
        for keyword in VALID_TITLE_KEYWORDS:
            # Match keyword with optional 'S' for plurals
            # e.g., "ELEVATION" matches "ELEVATION" and "ELEVATIONS"
            pattern = r'\b' + re.escape(keyword) + r'S?\b'
            if re.search(pattern, title_upper):
                return True
        return False

    def normalize_title_case(self, title: str) -> str:
        """
        Convert title to proper title case, preserving acronyms.

        Acronyms (2-4 letter all-caps words) are preserved.
        Examples:
            "FLOOR PLAN" -> "Floor Plan"
            "ADA COMPLIANCE NOTES" -> "ADA Compliance Notes"
            "HVAC SCHEDULE" -> "HVAC Schedule"
        """
        if not title:
            return title

        # If not all caps, return as-is (already mixed case)
        if not title.isupper():
            return title

        # Split into words and process
        words = title.split()
        result = []
        small_words = {'and', 'or', 'the', 'a', 'an', 'of', 'for', 'to', 'in', 'on', 'at', 'by', 'with'}

        for i, word in enumerate(words):
            # Check if word is an acronym to preserve
            word_upper = word.upper()
            if word_upper in TITLE_ACRONYMS:
                # Preserve as uppercase
                result.append(word_upper)
            elif i == 0:
                # First word always capitalized
                result.append(word.capitalize())
            elif word.lower() in small_words:
                # Small words lowercase (unless first)
                result.append(word.lower())
            else:
                # Regular word - capitalize
                result.append(word.capitalize())

        return ' '.join(result)