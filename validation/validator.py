"""
Blueprint Processor V4.2.1 - Validator
Validates extracted field data including sheet title validation.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    PATTERNS, TITLE_GARBAGE_PATTERNS, VALID_TITLE_KEYWORDS,
    TITLE_ACRONYMS, TITLE_LENGTH, TITLE_CONFIDENCE, REVIEW_THRESHOLD
)


class Validator:
    """
    Validates extracted blueprint data.
    Checks format compliance and detects contamination.
    """

    def __init__(self):
        """Initialize the Validator."""
        pass

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

        if len(title) > TITLE_LENGTH['MAX']:
            return {
                'is_valid': False,
                'title': None,
                'confidence': 0.0,
                'needs_review': True,
                'rejection_reason': f'Title too long ({len(title)} chars, max {TITLE_LENGTH["MAX"]})'
            }

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
        """Check if title contains a valid keyword."""
        title_upper = title.upper()
        for keyword in VALID_TITLE_KEYWORDS:
            # Check for whole word match
            if re.search(r'\b' + re.escape(keyword) + r'\b', title_upper):
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