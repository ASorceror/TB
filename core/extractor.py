"""
Blueprint Processor V4.2.1 - Field Extractor
Extracts structured fields from text using multiple strategies.
Integrates with Validator for sheet title validation.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import PATTERNS, LABELS, DISCIPLINE_CODES, TITLE_CONFIDENCE, REVIEW_THRESHOLD
from validation.validator import Validator
from core.drawing_index import DrawingIndexParser
from core.spatial_extractor import SpatialExtractor
from core.vision_extractor import VisionExtractor


class Extractor:
    """
    Extracts structured fields from blueprint text.
    Uses multiple strategies: label detection, pattern matching, spatial weighting.
    """

    def __init__(self, ocr_engine=None):
        """Initialize the Extractor with all extraction layers.

        Args:
            ocr_engine: Optional OCREngine instance for drawing index OCR
        """
        self._validator = Validator()
        self._index_parser = DrawingIndexParser(ocr_engine=ocr_engine)
        self._spatial_extractor = SpatialExtractor()
        self._vision_extractor = VisionExtractor()
        self._drawing_index: Dict[str, str] = {}

    def reset_for_new_pdf(self):
        """Reset state for processing a new PDF."""
        self._vision_extractor.reset_pdf_counter()
        self._drawing_index = {}

    def set_drawing_index(self, index: Dict[str, str]):
        """Set the drawing index for title lookups.

        Args:
            index: Dict mapping sheet_number -> title
        """
        self._drawing_index = index

    def parse_drawing_index(self, pdf_handler) -> Dict[str, str]:
        """Parse drawing index from a PDF and cache it.

        Args:
            pdf_handler: PDFHandler instance

        Returns:
            Dict mapping sheet_number -> title
        """
        self._drawing_index = self._index_parser.parse_from_pdf(pdf_handler)
        return self._drawing_index

    def lookup_in_index(self, sheet_number: str) -> Optional[str]:
        """Look up a sheet number in the cached drawing index.

        Args:
            sheet_number: Sheet number to look up

        Returns:
            Title if found, None otherwise
        """
        if not self._drawing_index or not sheet_number:
            return None
        return self._index_parser.lookup(sheet_number, self._drawing_index)

    def identify_cover_sheet(self, text: str, page_number: int = 1, is_cropped_region: bool = False) -> bool:
        text_upper = text.upper()
        for label in ["SHEET NO", "SHEET:", "SHEET NUMBER", "SHEET #", "DWG NO", "DWG:", "DRAWING NO", "DRAWING:"]:
            if label in text_upper:
                return False
        for indicator in ["COVER SHEET", "COVERSHEET", "TITLE SHEET", "TITLE PAGE"]:
            if indicator in text_upper:
                return True
        if page_number == 1 and not is_cropped_region:
            if "DRAWING INDEX" in text_upper or "SHEET INDEX" in text_upper:
                return True
        return False

    def remove_index_sections(self, text: str) -> str:
        result = text
        for pattern in [r'DRAWING INDEX:.*?(?=\n\n|\Z)', r'SHEET INDEX:.*?(?=\n\n|\Z)']:
            result = re.sub(pattern, '', result, flags=re.DOTALL | re.IGNORECASE)
        return result

    def _extract_project_from_cover(self, text: str) -> Optional[str]:
        return self._extract_project_number(text)

    def _normalize_ocr_digits(self, text: str) -> str:
        result = re.sub(r'(\d)\.O\b', r'\1.0', text)
        result = re.sub(r'(\d)\.o\b', r'\1.0', result)
        return result

    def extract_sheet_number(self, text: str, project_number: str = None) -> Optional[str]:
        text_upper = text.upper()
        text_for_extraction = self.remove_index_sections(text_upper)
        sheet_labels = ["SHEET NUMBER", "SHEET NO", "SHEET #", "SHEET", "DRAWING NO", "DWG NO", "DWG"]
        for label in sheet_labels:
            label_pattern = rf'{re.escape(label)}[:\s.#\n]*([A-Z]{{1,2}}[-.]?[\dO]{{1,3}}(?:\.[\dO]{{1,2}})?)'
            match = re.search(label_pattern, text_for_extraction, re.IGNORECASE)
            if match:
                candidate = self._normalize_ocr_digits(match.group(1).upper())
                if project_number and candidate in project_number:
                    continue
                return candidate
        sheet_pattern = r'\b([A-Z]{1,2}[-.]?[\dO]{1,3}(?:\.[\dO]{1,2})?)\b'
        candidates = re.findall(sheet_pattern, text_for_extraction)
        valid_candidates = []
        for candidate in candidates:
            normalized = self._normalize_ocr_digits(candidate)
            if project_number and normalized in project_number:
                continue
            if len(normalized) > 6:
                continue
            letter_part = re.match(r'^([A-Z]+)', normalized)
            if letter_part:
                digit_part = normalized[len(letter_part.group(1)):]
                if re.match(r'^-?\d{4,}$', digit_part.replace(".", "")):
                    continue
            valid_candidates.append(normalized)
        if valid_candidates:
            with_decimal = [c for c in valid_candidates if "." in c]
            return with_decimal[0].upper() if with_decimal else valid_candidates[0].upper()
        return None

    def extract_fields(self, text: str, text_blocks: Optional[List[Dict[str, Any]]] = None,
                       image_size: Optional[Tuple[int, int]] = None, page_number: int = 1,
                       is_cropped_region: bool = False,
                       page=None,
                       title_block_bbox_pixels: Optional[Tuple[float, float, float, float]] = None,
                       title_block_image=None) -> Dict[str, Any]:
        if self.identify_cover_sheet(text, page_number, is_cropped_region):
            return {
                "sheet_number": None,
                "project_number": self._extract_project_from_cover(text),
                "sheet_title": "COVER SHEET",
                "date": self._extract_date(text),
                "scale": None,
                "discipline": None,
                "is_cover_sheet": True,
                "title_confidence": 0.95,
                "title_method": "pattern",
                "needs_review": False,
                "extraction_details": {"cover_sheet": True},
            }
        result = {
            "sheet_number": None,
            "project_number": None,
            "sheet_title": None,
            "date": None,
            "scale": None,
            "discipline": None,
            "title_confidence": 0.0,
            "title_method": None,
            "needs_review": True,
            "extraction_details": {},
        }
        project_number = self._extract_project_number(text)
        result["project_number"] = project_number
        if project_number:
            result["extraction_details"]["project_number"] = "extracted"
        sheet_number = self.extract_sheet_number(text, project_number)
        result["sheet_number"] = sheet_number
        if sheet_number:
            result["extraction_details"]["sheet_number"] = "extracted"
        result["date"] = self._extract_date(text)
        if result["date"]:
            result["extraction_details"]["date"] = "pattern"
        scale_matches = PATTERNS["scale"].findall(text)
        if scale_matches:
            result["scale"] = scale_matches[0]
            result["extraction_details"]["scale"] = "pattern"

        # Extract sheet title using layered approach (V4.2.1)
        title_found = False

        # Layer 1: Drawing Index lookup (95% confidence)
        index_title = self.lookup_in_index(sheet_number) if sheet_number else None
        if index_title:
            validation_result = self._validator.validate_sheet_title(
                title=index_title, method="drawing_index", project_number=project_number
            )
            if validation_result["is_valid"]:
                result["sheet_title"] = validation_result["title"]
                result["title_confidence"] = validation_result["confidence"]
                result["title_method"] = "drawing_index"
                result["needs_review"] = validation_result["needs_review"]
                result["extraction_details"]["sheet_title"] = "drawing_index"
                title_found = True

        # Layer 2: Spatial Zone Detection (80% confidence) - Vector pages only
        if not title_found and page is not None:
            # Check if page is vector (has embedded text)
            if self._spatial_extractor.is_vector_page(page):
                spatial_result = self._spatial_extractor.extract_title(
                    page=page,
                    title_block_bbox_pixels=title_block_bbox_pixels
                )
                if spatial_result["title"]:
                    validation_result = self._validator.validate_sheet_title(
                        title=spatial_result["title"], method="spatial", project_number=project_number
                    )
                    if validation_result["is_valid"]:
                        result["sheet_title"] = validation_result["title"]
                        result["title_confidence"] = validation_result["confidence"]
                        result["title_method"] = "spatial"
                        result["needs_review"] = validation_result["needs_review"]
                        result["extraction_details"]["sheet_title"] = "spatial"
                        result["extraction_details"]["spatial_candidates"] = len(spatial_result.get("candidates", []))
                        title_found = True

        # Layer 3: Vision API (90% confidence) - When Layers 1-2 fail
        if not title_found and title_block_image is not None:
            if self._vision_extractor.is_available():
                vision_result = self._vision_extractor.extract_title(title_block_image)
                if vision_result["title"]:
                    validation_result = self._validator.validate_sheet_title(
                        title=vision_result["title"], method="vision_api", project_number=project_number
                    )
                    if validation_result["is_valid"]:
                        result["sheet_title"] = validation_result["title"]
                        result["title_confidence"] = validation_result["confidence"]
                        result["title_method"] = "vision_api"
                        result["needs_review"] = validation_result["needs_review"]
                        result["extraction_details"]["sheet_title"] = "vision_api"
                        result["extraction_details"]["vision_cached"] = vision_result.get("cached", False)
                        title_found = True
                elif vision_result.get("error"):
                    result["extraction_details"]["vision_error"] = vision_result["error"]

        # Layer 4: Pattern matching fallback (70% confidence)
        if not title_found:
            raw_title = self._extract_sheet_title(text)
            validation_result = self._validator.validate_sheet_title(
                title=raw_title, method="pattern", project_number=project_number
            )
            result["sheet_title"] = validation_result["title"]
            result["title_confidence"] = validation_result["confidence"]
            result["title_method"] = "pattern" if validation_result["is_valid"] else None
            result["needs_review"] = validation_result["needs_review"]
            if validation_result["is_valid"]:
                result["extraction_details"]["sheet_title"] = "pattern_validated"
            elif validation_result["rejection_reason"]:
                result["extraction_details"]["sheet_title_rejected"] = validation_result["rejection_reason"]
        if result["sheet_number"]:
            result["discipline"] = DISCIPLINE_CODES.get(result["sheet_number"][0].upper(), "Unknown")
        return result

    def _is_false_positive_project(self, value: str, text: str) -> bool:
        value_clean = value.strip()
        text_upper = text.upper()
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]' + re.escape(value_clean), text):
            return True
        if value_clean in ["2018","2019","2020","2021","2022","2023","2024","2025","2026","2027","2028","2029","2030"]:
            return True
        street_indicators = [r'S\.?\s+WOLF', r'N\.?\s+\w+', r'ARMOU?R', r'LINCOLN',
            r'\s+(?:ROAD|RD|STREET|ST|AVENUE|AVE|BLVD|BOULEVARD|DRIVE|DR|WAY|LANE|LN|HIGHWAY|HWY)']
        for street in street_indicators:
            if re.search(re.escape(value_clean) + r'\s+' + street, text_upper):
                return True
        if re.search(re.escape(value_clean) + r'\s+[NSEW]?\.?\s*\w+\s+(ROAD|RD|STREET|ST|AVE|AVENUE|BLVD|DR|DRIVE|WAY|LN|LANE|HWY|HIGHWAY)', text_upper):
            return True
        if re.search(r'\d{3}[-.\s]\d{3}[-.\s]' + re.escape(value_clean), text):
            return True
        if re.search(re.escape(value_clean) + r'[-.\s]\d{3}[-.\s]\d{4}', text):
            return True
        if re.search(r'\d{3}[-.\s]' + re.escape(value_clean) + r'[-.\s]\d{4}', text):
            return True
        if value_clean.isdigit() and len(value_clean) == 5:
            if re.search(r'[A-Z]{2}\s+' + re.escape(value_clean), text_upper):
                return True
        return False

    def _extract_project_number(self, text: str) -> Optional[str]:
        text_upper = text.upper()
        for label_pattern in [r'PROJECT\s*(?:NO|NUMBER|#)?[:\s]*', r'PROJ\s*(?:NO|#)?[:\s]*', r'JOB\s*(?:NO|NUMBER|#)?[:\s]*']:
            match = re.search(label_pattern + r'([A-Z]?\d{4,}[-.]?\d*)', text_upper)
            if match and not self._is_false_positive_project(match.group(1), text):
                return match.group(1)
        p_match = re.search(r'P\d{4,}', text_upper)
        if p_match and not self._is_false_positive_project(p_match.group(), text):
            return p_match.group()
        candidates = re.findall(r'\b([A-Z]?\d{4,}[-.]?\d*)\b', text_upper)
        valid = []
        for c in candidates:
            if self._is_false_positive_project(c, text):
                continue
            clean = c.replace("-","").replace(".","")
            if len(clean) in [5,7,10,11] or (len(clean)==4 and not c[0].isalpha()):
                continue
            valid.append(c)
        if valid:
            with_prefix = [c for c in valid if c[0].isalpha()]
            return with_prefix[0] if with_prefix else valid[0]
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        matches = PATTERNS["date"].findall(text)
        return matches[0] if matches else None

    def _clean_sheet_title(self, title: str) -> Optional[str]:
        if not title:
            return None
        title = re.sub(r'^[\s\-:]+', '', title)
        title = re.sub(r'[\s\-:]+$', '', title)
        title = re.sub(r'\s+', ' ', title)
        return None if title.upper() in ["TITLE","SHEET TITLE","DRAWING TITLE"] else title.strip()

    def _extract_sheet_title(self, text: str) -> Optional[str]:
        if not text:
            return None
        text_clean = text.strip()
        text_upper = text_clean.upper()
        for label_pattern in [r'SHEET\s*TITLE[:\s]+', r'DRAWING\s*TITLE[:\s]+', r'TITLE[:\s]+']:
            match = re.search(label_pattern + r'([A-Za-z][A-Za-z0-9\s,&\-\.]+)', text_clean, re.IGNORECASE)
            if match:
                title = self._clean_sheet_title(match.group(1).strip())
                if title and len(title) >= 3:
                    return title
        title_patterns = [
            r'\b(CODE\s+REVIEW[,\s]+SCHEDULES?\s+AND\s+ADA\s+REQUIREMENTS?)\b',
            r'\b(REFLECTED\s+CEILING\s+PLAN)\b', r'\b(DEMOLITION\s+FLOOR\s+PLAN)\b',
            r'\b(ENLARGED\s+FLOOR\s+PLAN)\b', r'\b(LIFE\s+SAFETY\s+PLAN)\b', r'\b(DEMOLITION\s+PLAN)\b',
            r'\b((?:FIRST|SECOND|THIRD|FOURTH|FIFTH|GROUND|BASEMENT|MAIN|UPPER|LOWER|MEZZANINE|PARTIAL)\s+FLOOR\s+PLAN)\b',
            r'\b(FLOOR\s+PLAN)\b', r'\b(CEILING\s+PLAN)\b', r'\b(ROOF\s+PLAN)\b',
            r'\b(SITE\s+PLAN)\b', r'\b(FOUNDATION\s+PLAN)\b',
            r'\b((?:NORTH|SOUTH|EAST|WEST|FRONT|REAR|SIDE|INTERIOR|EXTERIOR|BUILDING)\s+ELEVATIONS?)\b',
            r'\b(ELEVATIONS?)\b', r'\b((?:BUILDING|WALL|TYPICAL|DETAIL)\s+SECTIONS?)\b', r'\b(SECTIONS?)\b',
            r'\b((?:WALL|DOOR|WINDOW|STAIR|MILLWORK|CABINET|CEILING|FLOOR|ROOF)\s+DETAILS?)\b', r'\b(DETAILS?)\b',
            r'\b((?:DOOR|WINDOW|ROOM|FINISH|HARDWARE|FIXTURE|EQUIPMENT)\s+SCHEDULES?)\b', r'\b(SCHEDULES?)\b',
            r'\b(GENERAL\s+NOTES)\b', r'\b(SPECIFICATIONS?)\b', r'\b(COVER\s+SHEET)\b', r'\b(PLAN)\b']
        for pattern in title_patterns:
            match = re.search(pattern, text_upper)
            if match and len(match.group(1).strip()) >= 3:
                return match.group(1).strip()
        split_patterns = [r'(CODE\s+REVIEW,?)\s*\n\s*(SCHEDULES?\s+AND)\s*\n?\s*(ADA\s+REQUIREMENTS?)',
            r'(REFLECTED\s+CEILING)\s*\n\s*(PLAN)', r'(DEMOLITION\s+FLOOR)\s*\n\s*(PLAN)',
            r'(ENLARGED\s+FLOOR)\s*\n\s*(PLAN)', r'(FLOOR)\s*\n\s*(PLAN)', r'(CEILING)\s*\n\s*(PLAN)']
        for pattern in split_patterns:
            match = re.search(pattern, text_upper)
            if match:
                title = ' '.join(g for g in match.groups() if g)
                if len(title) >= 3:
                    return title
        addr_match = re.search(r'(?:IL|ILLINOIS)\s+\d{5}\s*\n+(.*?)(?=PROJECT|SHEET\s*NO|$)', text_clean, re.IGNORECASE|re.DOTALL)
        if addr_match:
            lines = [l.strip() for l in addr_match.group(1).strip().split('\n') if l.strip() and len(l.strip())>=3
                     and not re.match(r'^(PROJECT|SHEET|SCALE|DATE|DRAWN|CHECKED|APPROVED)[:\s]*$', l.strip(), re.I)
                     and not re.match(r'^[\W\d]+$', l.strip())]
            if lines:
                potential = re.sub(r'\s+', ' ', ' '.join(lines[:3])).strip()
                if len(potential) >= 3:
                    return potential
        return None

    def extract_all_matches(self, text: str) -> Dict[str, List[str]]:
        return {field: (pattern.findall(text.upper()) if field=="sheet_number" else pattern.findall(text)) for field, pattern in PATTERNS.items()}