"""
Blueprint Processor V4.4 - Field Extractor
Extracts structured fields from text using multiple strategies.
Integrates with Validator for sheet title validation.

V4.4 Changes:
- Project number extraction DISABLED (returns None) - not needed
- Sheet number extraction rewritten with LABEL-FIRST strategy
- Only extracts sheet numbers near SHEET/DWG labels
- Never guesses from random text patterns
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import pytesseract for title block OCR fallback
try:
    import pytesseract
    from PIL import Image, ImageEnhance
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)

from constants import PATTERNS, LABELS, DISCIPLINE_CODES, TITLE_CONFIDENCE, REVIEW_THRESHOLD, SHEET_NUMBER_BLACKLIST, TITLE_LABEL_BLACKLIST
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

    def _ocr_title_block_for_sheet_number(self, title_block_image) -> Optional[str]:
        """
        OCR the title block image at high resolution to extract sheet number.

        This is a fallback for when the full-page text extraction misses small
        title block text (common with small fonts or scanned PDFs).

        Sheet numbers are typically in a small box on the edges of the title block.
        We check BOTH left and right edges because page rotation can flip the layout.

        Args:
            title_block_image: PIL Image of the title block region

        Returns:
            OCR text from the title block, or None if OCR fails
        """
        if not TESSERACT_AVAILABLE or title_block_image is None:
            return None

        try:
            width, height = title_block_image.size
            logger.debug(f"Title block image size: {width}x{height}")

            # Define edge regions to check - sheet number could be on either side
            # depending on page orientation/rotation
            edge_regions = [
                ('right', (int(width * 0.70), 0, width, height)),       # Right 30%
                ('left', (0, 0, int(width * 0.30), height)),            # Left 30%
            ]

            all_text = []
            for edge_name, (x1, y1, x2, y2) in edge_regions:
                edge_strip = title_block_image.crop((x1, y1, x2, y2))
                strip_width, strip_height = edge_strip.size

                if strip_width < 10 or strip_height < 10:
                    continue

                # Upscale for better OCR quality
                min_dimension = 800
                scale = max(min_dimension / strip_width, min_dimension / strip_height, 2.0)
                new_size = (int(strip_width * scale), int(strip_height * scale))
                upscaled = edge_strip.resize(new_size, Image.Resampling.LANCZOS)

                # Enhance contrast for better OCR
                enhancer = ImageEnhance.Contrast(upscaled)
                enhanced = enhancer.enhance(1.5)

                # Use PSM 4 (single column) which works better for title blocks
                custom_config = r'--oem 3 --psm 4'
                text = pytesseract.image_to_string(enhanced, config=custom_config)

                if text.strip():
                    all_text.append(text.upper())
                    logger.debug(f"Title block {edge_name} edge OCR: {len(text)} chars")

            combined_text = '\n'.join(all_text)
            logger.debug(f"Title block OCR total: {len(combined_text)} chars")
            return combined_text if combined_text else None

        except Exception as e:
            logger.debug(f"Title block OCR failed: {e}")
            return None

    def extract_sheet_number(self, text: str, project_number: str = None, text_blocks: list = None) -> Optional[str]:
        """
        Extract sheet number using LABEL-FIRST strategy (V4.4).

        ONLY extract sheet numbers that appear next to labels like:
        - SHEET / SHEET NO / SHEET NUMBER / SHEET #
        - DWG / DWG NO / DRAWING NO
        - Handles truncated OCR like "eet No.:" or "No.:" in title blocks

        Valid formats: A1.0, A-1, T1.1, D1.1, AD1.01, M1.1
        - 1-2 letters + optional separator + 1-3 digits + optional decimal

        NEVER extract:
        - Pure numbers (1020, 2430)
        - Long codes (U465, D6226)
        - Anything not near a SHEET/DWG label
        """
        text_upper = text.upper()
        text_for_extraction = self.remove_index_sections(text_upper)

        # Sheet number pattern: 1-2 letters, optional dash/dot, 1-3 digits, optional .decimal
        sheet_pattern = r'([A-Z]{1,2}[-.]?\d{1,3}(?:\.\d{1,2})?)'

        def validate_sheet_candidate(candidate: str) -> bool:
            """Validate a sheet number candidate."""
            if not candidate or len(candidate) < 2 or len(candidate) > 8:
                return False
            if not re.match(r'^[A-Z]+', candidate):
                return False
            digits_only = re.sub(r'[^0-9]', '', candidate)
            if len(digits_only) < 1 or len(digits_only) > 3:
                return False
            return True

        # Strategy 1: Labels that precede sheet numbers on same line
        labels = [
            r'SHEET\s*NUMBER\s*[:\s]*',
            r'SHEET\s*NO\.?\s*[:\s]*',
            r'SHEET\s*#\s*[:\s]*',
            r'DWG\s*NO\.?\s*[:\s]*',
            r'DRAWING\s*NO\.?\s*[:\s]*',
            r'SHEET\s*[:\s]+',  # "SHEET:" or "SHEET " followed by value
        ]

        for label in labels:
            pattern = label + sheet_pattern
            match = re.search(pattern, text_for_extraction)
            if match:
                candidate = self._normalize_ocr_digits(match.group(1))
                if validate_sheet_candidate(candidate):
                    logger.debug(f"Sheet number found via label: '{candidate}'")
                    return candidate

        # Strategy 2: Handle truncated OCR labels (common in small title block text)
        # OCR often misses "Sh" in "Sheet No.:" producing "eet No.:" or just "No.:"
        truncated_labels = [
            r'EET\s*NO\.?\s*[:\s]*',    # "eet No.:" (truncated "Sheet No.:")
            r'(?<![A-Z])NO\.?\s*[:\s]+', # "No.:" or "NO:" not preceded by letter (standalone)
        ]

        for label in truncated_labels:
            pattern = label + sheet_pattern
            match = re.search(pattern, text_for_extraction)
            if match:
                candidate = self._normalize_ocr_digits(match.group(1))
                if validate_sheet_candidate(candidate):
                    logger.debug(f"Sheet number found via truncated label: '{candidate}'")
                    return candidate

        # Strategy 3: Multi-line pattern - label on one line, value on next
        # Handle cases where OCR splits "Sheet No.:\nA1.1" across lines
        # Also handles blank lines between label and value (common in OCR)
        lines = text_for_extraction.split('\n')
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # Check if this line ends with a label pattern
            if re.search(r'(SHEET|DWG|DRAWING)\s*(NO\.?|NUMBER|#)?\s*:?\s*$', line_stripped):
                # Check next few lines for sheet number (skip blanks)
                for j in range(i + 1, min(len(lines), i + 4)):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue  # Skip blank lines
                    match = re.match(sheet_pattern, next_line)
                    if match:
                        candidate = self._normalize_ocr_digits(match.group(1))
                        if validate_sheet_candidate(candidate):
                            logger.debug(f"Sheet number found via multi-line: '{candidate}'")
                            return candidate
                    break  # Stop at first non-blank, non-matching line
            # Also check for truncated labels
            if re.search(r'(EET|NO\.?)\s*:?\s*$', line_stripped):
                for j in range(i + 1, min(len(lines), i + 4)):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue  # Skip blank lines
                    match = re.match(sheet_pattern, next_line)
                    if match:
                        candidate = self._normalize_ocr_digits(match.group(1))
                        if validate_sheet_candidate(candidate):
                            logger.debug(f"Sheet number found via truncated multi-line: '{candidate}'")
                            return candidate
                    break  # Stop at first non-blank, non-matching line

        # Strategy 4: Spatial approach with text_blocks (if available)
        if text_blocks:
            sheet_label_keywords = ['SHEET', 'DWG NO', 'DRAWING NO', 'SHEET NO', 'SHEET NUMBER', 'EET NO', 'NO.:', 'NO:']
            for i, block in enumerate(text_blocks):
                block_text = str(block.get('text', '')).upper()
                if any(lbl in block_text for lbl in sheet_label_keywords):
                    # Check this block and adjacent blocks (within 3 positions for multi-line)
                    for j in range(max(0, i-1), min(len(text_blocks), i+4)):
                        nearby_text = str(text_blocks[j].get('text', '')).upper()
                        match = re.search(r'\b' + sheet_pattern + r'\b', nearby_text)
                        if match:
                            candidate = self._normalize_ocr_digits(match.group(1))
                            if validate_sheet_candidate(candidate):
                                logger.debug(f"Sheet number found via spatial: '{candidate}'")
                                return candidate

        # Strategy 5: Check last lines of text for isolated sheet numbers
        # Title blocks often have sheet numbers on their own line at the end of vector text
        # Pattern: A1.1, M2.1, E1.1 etc. on a line by itself (typical title block layout)
        last_lines = lines[-10:] if len(lines) > 10 else lines
        for line in reversed(last_lines):
            line_stripped = line.strip()
            # Only match lines that are JUST a sheet number (possibly with whitespace)
            if re.match(r'^' + sheet_pattern + r'$', line_stripped):
                candidate = self._normalize_ocr_digits(line_stripped)
                if validate_sheet_candidate(candidate):
                    logger.debug(f"Sheet number found via isolated line at end of text: '{candidate}'")
                    return candidate

        # No labeled sheet number found - return None, don't guess
        logger.debug("No labeled sheet number found - returning None")
        return None

    def _ocr_wide_title_block(self, page_image) -> Optional[str]:
        """
        OCR a wide title block region (rightmost 15%) of the page.

        This is used as a fallback for rotated pages where the standard
        title block detection may have found the wrong region.

        Args:
            page_image: PIL Image of the page

        Returns:
            OCR text from the title block region, or None if OCR fails
        """
        if not TESSERACT_AVAILABLE or page_image is None:
            return None

        try:
            width, height = page_image.size
            logger.debug(f"Wide title block OCR on image: {width}x{height}")

            # Crop rightmost 15% of page (typical title block location)
            right_strip = page_image.crop((int(width * 0.85), 0, width, height))
            strip_width, strip_height = right_strip.size

            # Upscale for better OCR
            min_dimension = 1000
            scale = max(min_dimension / strip_width, 2.0)
            new_size = (int(strip_width * scale), int(strip_height * scale))
            upscaled = right_strip.resize(new_size, Image.Resampling.LANCZOS)

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(upscaled)
            enhanced = enhancer.enhance(1.5)

            # OCR with PSM 4 (single column)
            custom_config = r'--oem 3 --psm 4'
            text = pytesseract.image_to_string(enhanced, config=custom_config)

            logger.debug(f"Wide title block OCR extracted {len(text)} chars")
            return text.upper() if text.strip() else None

        except Exception as e:
            logger.debug(f"Wide title block OCR failed: {e}")
            return None

    def _ocr_page_edges_for_sheet_number(self, page_image, original_page_image=None, rotation_applied: int = 0) -> Optional[str]:
        """
        OCR the edges of the full page image to find sheet numbers.

        This is a last-resort fallback when title block detection fails.
        Searches right edge, left edge, and bottom-right corner.

        V4.4: Also tries the original (non-rotated) image if normalized image fails.
        For 180° rotation, searches opposite edges on original image.

        Args:
            page_image: PIL Image of the (normalized) full page
            original_page_image: PIL Image of the original (non-rotated) page
            rotation_applied: Rotation angle applied during normalization (0, 90, 180, 270)

        Returns:
            OCR text from page edges, or None if OCR fails
        """
        if not TESSERACT_AVAILABLE:
            return None

        def ocr_edges(img, edge_regions, high_quality: bool = False):
            """OCR specified edge regions of an image.

            Args:
                img: PIL Image to OCR
                edge_regions: List of (name, coord_func) tuples
                high_quality: If True, use higher upscaling for small text
            """
            all_text = []
            width, height = img.size

            for edge_name, get_coords in edge_regions:
                x1, y1, x2, y2 = get_coords(width, height)
                edge_strip = img.crop((x1, y1, x2, y2))
                strip_width, strip_height = edge_strip.size

                if strip_width < 50 or strip_height < 50:
                    continue

                # Upscale for better OCR quality
                # Use higher scale for high_quality mode (small text in title blocks)
                if high_quality:
                    scale = 4.0  # Higher scale for small text
                else:
                    min_dimension = 800
                    scale = max(min_dimension / strip_width, 2.0)

                new_size = (int(strip_width * scale), int(strip_height * scale))
                upscaled = edge_strip.resize(new_size, Image.Resampling.LANCZOS)

                # Enhance contrast
                enhancer = ImageEnhance.Contrast(upscaled)
                enhanced = enhancer.enhance(1.5)

                # OCR with PSM 4
                custom_config = r'--oem 3 --psm 4'
                text = pytesseract.image_to_string(enhanced, config=custom_config)

                if text.strip():
                    all_text.append(text.upper())
                    logger.debug(f"Page edge {edge_name} OCR: {len(text)} chars")

            return '\n'.join(all_text) if all_text else None

        # Standard edge regions - use 5% strips for better precision
        standard_edges = [
            ('far_right_5pct', lambda w, h: (int(w * 0.95), 0, w, h)),          # Far right 5%
            ('far_left_5pct', lambda w, h: (0, 0, int(w * 0.05), h)),           # Far left 5%
            ('bottom_right', lambda w, h: (int(w * 0.85), int(h * 0.85), w, h)), # Bottom-right corner
        ]

        # For original image, use same regions but with high quality OCR
        # Also add slightly wider regions as backup
        original_edges = [
            ('orig_far_right_5pct', lambda w, h: (int(w * 0.95), 0, w, h)),     # Far right 5%
            ('orig_far_right_10pct', lambda w, h: (int(w * 0.90), 0, w, h)),    # Far right 10%
            ('orig_far_left_5pct', lambda w, h: (0, 0, int(w * 0.05), h)),      # Far left 5%
        ]

        try:
            # First try normalized image with standard edges
            if page_image is not None:
                logger.debug(f"Searching normalized page edges: {page_image.size}")
                result = ocr_edges(page_image, standard_edges)
                if result:
                    # Check if we found sheet number patterns in the text
                    if re.search(r'(SHEET|EET|NO\.?:?)\s*\n*\s*[A-Z]\d', result) or \
                       re.search(r'\b[A-Z]\d+\.\d+\b', result):
                        return result

            # If no sheet number found and we have original image, try with high quality OCR
            if original_page_image is not None:
                logger.debug(f"Trying original page edges with high quality OCR (rotation={rotation_applied}): {original_page_image.size}")

                # For original image, always try both sides with high quality OCR
                # The sheet number could be on either edge depending on how the PDF was scanned
                result = ocr_edges(original_page_image, original_edges, high_quality=True)
                if result:
                    # Check if we found sheet number patterns
                    if re.search(r'(SHEET|EET|NO\.?:?)\s*\n*\s*[A-Z]\d', result) or \
                       re.search(r'\b[A-Z]\d+\.\d+\b', result):
                        return result

            return None

        except Exception as e:
            logger.debug(f"Page edge OCR failed: {e}")
            return None

    def extract_fields(self, text: str, text_blocks: Optional[List[Dict[str, Any]]] = None,
                       image_size: Optional[Tuple[int, int]] = None, page_number: int = 1,
                       is_cropped_region: bool = False,
                       page=None,
                       title_block_bbox_pixels: Optional[Tuple[float, float, float, float]] = None,
                       title_block_image=None,
                       page_image=None,
                       original_page_image=None,
                       rotation_applied: int = 0) -> Dict[str, Any]:
        # V4.4.1: Check if this is a cover/title sheet - but DON'T skip extraction!
        # Title sheets (T1.1, etc.) have sheet numbers even if they contain "TITLE SHEET" text
        is_cover_sheet = self.identify_cover_sheet(text, page_number, is_cropped_region)

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

        # Try sheet number extraction from full-page text first
        sheet_number = self.extract_sheet_number(text, project_number, text_blocks)

        # Fallback 1: If sheet number not found and we have a title block image,
        # OCR the title block at high resolution (catches small fonts)
        if not sheet_number and title_block_image is not None:
            logger.debug("Sheet number not found in full-page text, trying title block OCR")
            title_block_text = self._ocr_title_block_for_sheet_number(title_block_image)
            if title_block_text:
                sheet_number = self.extract_sheet_number(title_block_text, project_number)
                if sheet_number:
                    result["extraction_details"]["sheet_number"] = "title_block_ocr"
                    logger.debug(f"Sheet number found via title block OCR: {sheet_number}")

        # Fallback 2: If still not found and we have full page image, search page edges
        # This handles cases where title block detection selected the wrong region
        # V4.4: Also tries original (non-rotated) image to handle 180° rotation issues
        if not sheet_number and (page_image is not None or original_page_image is not None):
            logger.debug(f"Sheet number not found via title block, trying page edge OCR (rotation={rotation_applied})")
            page_edge_text = self._ocr_page_edges_for_sheet_number(
                page_image, original_page_image, rotation_applied
            )
            if page_edge_text:
                sheet_number = self.extract_sheet_number(page_edge_text, project_number)
                if sheet_number:
                    result["extraction_details"]["sheet_number"] = "page_edge_ocr"
                    logger.debug(f"Sheet number found via page edge OCR: {sheet_number}")

        # Fallback 3: For rotated pages, try wider title block region on ORIGINAL image
        # Blueprints typically have title blocks on the right 15% of the page
        if not sheet_number and original_page_image is not None and rotation_applied != 0:
            logger.debug(f"Trying wide title block OCR on original image (rotation={rotation_applied})")
            original_tb_text = self._ocr_wide_title_block(original_page_image)
            if original_tb_text:
                sheet_number = self.extract_sheet_number(original_tb_text, project_number)
                if sheet_number:
                    result["extraction_details"]["sheet_number"] = "original_wide_tb_ocr"
                    logger.debug(f"Sheet number found via original wide title block OCR: {sheet_number}")

        # Fallback 4: For cover/title sheets, look for T-prefix sheet numbers anywhere in text
        # Title sheets often have sheet numbers like T1.1, T-1, T1 without explicit "SHEET NO:" labels
        if not sheet_number and is_cover_sheet:
            logger.debug("Title/cover sheet detected - searching for T-prefix sheet numbers")
            # Look for T followed by number pattern (T1.1, T-1, T1, etc.)
            t_pattern = r'(T[-.]?\d{1,2}(?:\.\d{1,2})?)'
            t_matches = re.findall(t_pattern, text.upper())
            if t_matches:
                sheet_number = t_matches[0]
                result["extraction_details"]["sheet_number"] = "title_sheet_pattern"
                logger.debug(f"Sheet number found via title sheet pattern: {sheet_number}")

        result["sheet_number"] = sheet_number
        if sheet_number and "sheet_number" not in result["extraction_details"]:
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
        spatial_rejected_by_qg = False  # Track if we rejected spatial for Vision fallback
        if not title_found and page is not None:
            # Check if page is vector (has embedded text)
            is_vector = self._spatial_extractor.is_vector_page(page)
            logger.debug(f"Layer 2: page={page_number}, is_vector={is_vector}")

            if is_vector:
                spatial_result = self._spatial_extractor.extract_title(
                    page=page,
                    title_block_bbox_pixels=title_block_bbox_pixels
                )
                logger.debug(f"Layer 2: spatial_result title='{spatial_result.get('title', '')[:50]}'")

                if spatial_result["title"]:
                    # === V4.2.2: QUALITY GATE ===
                    # Check if spatial result passes stricter quality criteria
                    is_quality, quality_reason = self._validator.is_quality_title(
                        spatial_result["title"],
                        project_number
                    )
                    logger.debug(f"Layer 2: quality_gate is_quality={is_quality}, reason={quality_reason}")

                    if is_quality:
                        # Passed quality gate - proceed with normal validation
                        validation_result = self._validator.validate_sheet_title(
                            title=spatial_result["title"],
                            method="spatial",
                            project_number=project_number
                        )
                        if validation_result["is_valid"]:
                            result["sheet_title"] = validation_result["title"]
                            result["title_confidence"] = validation_result["confidence"]
                            result["title_method"] = "spatial"
                            result["needs_review"] = validation_result["needs_review"]
                            result["extraction_details"]["sheet_title"] = "spatial"
                            result["extraction_details"]["spatial_candidates"] = len(spatial_result.get("candidates", []))
                            title_found = True
                    else:
                        # Failed quality gate - log and fall through to Vision API
                        spatial_rejected_by_qg = True
                        result["extraction_details"]["spatial_rejected"] = quality_reason
                        result["extraction_details"]["spatial_attempted"] = spatial_result["title"][:50]
                        logger.info(
                            f"Quality gate REJECTED spatial: '{spatial_result['title'][:40]}' "
                            f"reason={quality_reason} - will try Vision API"
                        )
                        # title_found stays False - will try Vision API next

        # Layer 3: Vision API (90% confidence) - When Layers 1-2 fail
        logger.debug(f"Layer 3: title_found={title_found}, title_block_image={'exists' if title_block_image else 'None'}, is_available={self._vision_extractor.is_available()}")
        if not title_found and title_block_image is not None:
            if self._vision_extractor.is_available():
                # Log if this is a fallback from rejected spatial
                if spatial_rejected_by_qg:
                    logger.info(f"Vision API fallback triggered after spatial rejection")
                vision_result = self._vision_extractor.extract_title(title_block_image)
                logger.debug(f"Layer 3: vision_result title='{vision_result.get('title', '')}'")
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
                else:
                    if vision_result.get("error"):
                        result["extraction_details"]["vision_error"] = vision_result["error"]
                    else:
                        result["extraction_details"]["vision_status"] = "no_title"
            else:
                result["extraction_details"]["vision_status"] = "unavailable"
                logger.warning(f"Layer 3: Vision API not available, skipping")

        # Layer 4: Pattern matching fallback (70% confidence)
        # V4.2.4: Apply quality gate to pattern results (same as spatial)
        if not title_found:
            raw_title = self._extract_sheet_title(text)

            # V4.2.4: Apply quality gate to pattern results
            if raw_title:
                is_quality, quality_reason = self._validator.is_quality_title(
                    raw_title, project_number
                )
                if not is_quality:
                    logger.debug(f"Pattern title rejected by quality gate: '{raw_title}' - {quality_reason}")
                    result["extraction_details"]["pattern_rejected_qg"] = quality_reason
                    result["extraction_details"]["pattern_attempted"] = raw_title[:50] if len(raw_title) > 50 else raw_title
                    raw_title = None  # Reject garbage, will result in null title

            if raw_title:
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
            else:
                # No valid title after quality gate
                result["sheet_title"] = None
                result["title_confidence"] = 0.0
                result["title_method"] = None
                result["needs_review"] = True

        # V4.4.1: Handle cover/title sheets - still extract sheet number but use default title if needed
        if is_cover_sheet:
            result["is_cover_sheet"] = True
            result["extraction_details"]["cover_sheet"] = True
            # If no title was found via other methods, use "TITLE SHEET" as default
            if not result.get("sheet_title"):
                result["sheet_title"] = "TITLE SHEET"
                result["title_confidence"] = 0.95
                result["title_method"] = "pattern"
                result["needs_review"] = False

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
        """Project number extraction disabled in V4.4 - not needed for business use case."""
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
        """
        Extract sheet title from text using pattern matching.
        V4.5: Multi-part title extraction - captures full titles like
        "FLOOR PLAN, CODE ANALYSIS, KEYNOTES AND PARTITION SCHEDULE"
        instead of just "FLOOR PLAN".
        """
        if not text:
            return None
        text_clean = text.strip()
        # Preserve newlines but collapse horizontal whitespace for pattern matching
        text_normalized = re.sub(r'[ \t]+', ' ', text_clean)
        text_upper = text_normalized.upper()
        extracted_title = None

        # Terminator patterns that mark the END of a title block
        # These are field labels that appear AFTER the title in title blocks
        terminator_pattern = (
            r'(?:'
            r'(?:\n|\s{2,})PROJECT\s*(?:NO|#|NUMBER)?[:\s=]|'
            r'(?:\n|\s{2,})SHEET\s*(?:NO|#|NUMBER)?[:\s=]|'
            r'(?:\n|\s{2,})DWG\s*(?:NO|#|NUMBER)?[:\s=]|'
            r'(?:\n|\s{2,})SCALE[:\s=]|'
            r'(?:\n|\s{2,})DATE[:\s=]|'
            r'(?:\n|\s{2,})(?:DRAWN|CHECKED|APPROVED|REVISED)[:\s]|'
            r'(?:\n|\s{2,})(?:IL|ILLINOIS)\s+\d{5}|'  # Address/zip code
            r'(?:\n|\s{2,})U\.?L\.?\s+(?:PARTITION|DESIGN)|'  # UL references
            r'(?:\n|\s{2,})(?:PHONE|FAX|TEL)[:\s]|'
            r'\n{2,}|'  # Double newline (paragraph break)
            r'$'
            r')'
        )

        # Strategy 1: Label-adjacent extraction with multi-part capture
        for label_pattern in [r'SHEET\s*TITLE[:\s]+', r'DRAWING\s*TITLE[:\s]+', r'TITLE[:\s]+']:
            match = re.search(label_pattern + r'(.+?)' + terminator_pattern, text_upper, re.DOTALL)
            if match:
                title = self._normalize_multipart_title(match.group(1))
                if title and len(title) >= 3:
                    extracted_title = title
                    break

        # Strategy 2: Find title keywords and capture ALL text until terminator
        if not extracted_title:
            # Title keywords in priority order (more specific first)
            title_keywords = [
                r'REFLECTED\s+CEILING\s+PLAN',
                r'DEMOLITION\s+FLOOR\s+PLAN',
                r'ENLARGED\s+FLOOR\s+PLAN',
                r'LIFE\s+SAFETY\s+PLAN',
                r'DEMOLITION\s+PLAN',
                r'(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|GROUND|BASEMENT|MAIN|UPPER|LOWER|MEZZANINE|PARTIAL)\s+FLOOR\s+PLAN',
                r'FLOOR\s+PLAN',
                r'MECHANICAL\s+PLAN',
                r'ELECTRICAL\s+PLAN',
                r'PLUMBING\s+PLAN',
                r'CEILING\s+PLAN',
                r'ROOF\s+PLAN',
                r'SITE\s+PLAN',
                r'FOUNDATION\s+PLAN',
                r'(?:NORTH|SOUTH|EAST|WEST|FRONT|REAR|SIDE|INTERIOR|EXTERIOR|BUILDING)\s+ELEVATIONS?',
                r'ELEVATIONS?',
                r'(?:BUILDING|WALL|TYPICAL|DETAIL)\s+SECTIONS?',
                r'SECTIONS?',
                r'(?:WALL|DOOR|WINDOW|STAIR|MILLWORK|CABINET|CEILING|FLOOR|ROOF)\s+DETAILS?',
                r'DETAILS?',
                r'(?:DOOR|WINDOW|ROOM|FINISH|HARDWARE|FIXTURE|EQUIPMENT)\s+SCHEDULES?',
                r'MECHANICAL\s+SCHEDULES?',
                r'SCHEDULES?',
                r'GENERAL\s+NOTES',
                r'SPECIFICATIONS?',
                r'COVER\s+SHEET',
                r'TITLE\s+SHEET',
            ]

            for keyword in title_keywords:
                # Find the keyword and capture everything until terminator
                pattern = r'\b(' + keyword + r')(.*)' + terminator_pattern
                match = re.search(pattern, text_upper, re.DOTALL)
                if match:
                    keyword_text = match.group(1).strip()
                    additional_text = match.group(2).strip() if match.group(2) else ''

                    # Combine keyword with additional text using newline separator
                    # so normalization treats them as separate parts
                    if additional_text:
                        full_title = keyword_text + '\n' + additional_text
                    else:
                        full_title = keyword_text

                    title = self._normalize_multipart_title(full_title)
                    if title and len(title) >= 3:
                        extracted_title = title
                        break

        # Strategy 3: Address-adjacent extraction (fallback)
        if not extracted_title:
            addr_match = re.search(r'(?:IL|ILLINOIS)\s+\d{5}\s*\n+(.*?)' + terminator_pattern, text_upper, re.DOTALL)
            if addr_match:
                title = self._normalize_multipart_title(addr_match.group(1))
                if title and len(title) >= 3:
                    extracted_title = title

        # Reject if extracted text is a label, not a title
        if extracted_title:
            title_upper = extracted_title.upper().strip()
            if title_upper in TITLE_LABEL_BLACKLIST:
                logger.debug(f"Title rejected - is label text: '{extracted_title}'")
                return None

        return extracted_title

    def _normalize_multipart_title(self, raw_title: str) -> Optional[str]:
        """
        Normalize a multi-part title extracted from text.
        Handles newlines, commas, and "AND" connectors.
        V4.5: Properly joins multi-line titles with commas.
        """
        if not raw_title:
            return None

        # Split by newlines to get individual parts
        parts = [p.strip() for p in raw_title.split('\n') if p.strip()]

        # Filter out parts that look like field labels or metadata
        label_patterns = [
            r'^(?:PROJECT|SHEET|SCALE|DATE|DRAWN|CHECKED|APPROVED|REVISED)[:\s]*$',
            r'^(?:NO|#|NUMBER)[:\s]*$',
            r'^\d+[:\s]*$',  # Pure numbers
            r'^[A-Z]\d+\.\d+$',  # Sheet numbers like A1.1
            r'^(?:PHONE|FAX|TEL)',
            r'^U\.?L\.?\s+',  # UL references
            r'IL\s+DESIGN\s+FIRM',  # IL Design Firm metadata
            r'^#?\d{6,}',  # Long numbers (firm IDs like #184001114-0002)
            r'^\d+-\d+-\d+$',  # Date-like patterns
            r'^(?:NORTH|SOUTH|EAST|WEST)$',  # Cardinal directions alone
        ]

        filtered_parts = []
        for part in parts:
            part_upper = part.upper()
            is_label = any(re.match(pat, part_upper) for pat in label_patterns)
            if not is_label and len(part) >= 2:
                filtered_parts.append(part)

        if not filtered_parts:
            return None

        # Join ALL parts with commas (each line becomes a comma-separated component)
        title = ', '.join(filtered_parts)

        # Clean up the title
        title = re.sub(r'\s+', ' ', title)  # Collapse whitespace
        title = re.sub(r',\s*,', ',', title)  # Remove double commas
        title = re.sub(r'\s*,\s*', ', ', title)  # Normalize comma spacing
        title = title.strip(' ,')  # Remove leading/trailing commas

        # Title case conversion for better readability
        title = title.title()

        # Limit length to reasonable title size
        if len(title) > 100:
            title = title[:100].rsplit(',', 1)[0].strip()

        return title if title else None

    def extract_all_matches(self, text: str) -> Dict[str, List[str]]:
        return {field: (pattern.findall(text.upper()) if field=="sheet_number" else pattern.findall(text)) for field, pattern in PATTERNS.items()}