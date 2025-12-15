"""
Blueprint Processor V4.2.1 - Spatial Zone Extractor
Extracts sheet titles from vector PDFs using text position and font analysis.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import VALID_TITLE_KEYWORDS, TITLE_CONFIDENCE
from core.coordinates import (
    pixels_to_points,
    points_to_pixels,
    get_vertical_zone,
    get_title_zone_bbox,
    bbox_overlap,
    bbox_center,
)

logger = logging.getLogger(__name__)

# Labels to skip - these are field labels, not titles
SKIP_LABELS = [
    # Field labels
    'PROJECT', 'SHEET', 'SCALE', 'DATE', 'DRAWN', 'CHECKED', 'APPROVED',
    'REVISION', 'REV', 'ISSUE', 'BY', 'DESCRIPTION', 'NO', 'NUMBER',
    # Common values
    'AS NOTED', 'VARIES', 'SEE PLANS', 'N.T.S.', 'NTS', 'NOT TO SCALE',
    # Parties
    'CLIENT', 'ARCHITECT', 'ENGINEER', 'CONTRACTOR', 'OWNER',
    # Other
    'DRAWING', 'DWG', 'TITLE', 'NAME', 'ADDRESS', 'PHONE', 'FAX', 'EMAIL',
]

# Minimum font size to consider (skip tiny text)
MIN_FONT_SIZE = 8.0

# Base font size for scoring (title text is usually larger)
BASE_FONT_SIZE = 12.0


class SpatialExtractor:
    """
    Extracts sheet titles from vector PDFs by analyzing text positions and fonts.

    Only works for pages with embedded text (vector PDFs).
    Scanned PDFs have no text positions and require OCR.
    """

    def __init__(self):
        """Initialize the SpatialExtractor."""
        pass

    def extract_title(
        self,
        page,  # fitz.Page object
        title_block_bbox_pixels: Optional[Tuple[float, float, float, float]] = None,
        dpi: int = 200
    ) -> Dict[str, Any]:
        """
        Extract sheet title from a vector PDF page using spatial analysis.

        Args:
            page: PyMuPDF page object (fitz.Page)
            title_block_bbox_pixels: Optional title block region in pixels.
                                     If None, uses full page.
            dpi: DPI used for pixel coordinates (default 200)

        Returns:
            Dict with keys:
                - title: extracted title string or None
                - confidence: float 0.0-1.0 (max 0.80 for spatial)
                - candidates: list of scored candidates for debugging
                - method: 'spatial' if found
        """
        # Get text with positions
        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])

        # Convert title block bbox to points if provided
        if title_block_bbox_pixels:
            title_block_bbox = tuple(pixels_to_points(title_block_bbox_pixels, dpi))
        else:
            # Use page rect as fallback
            rect = page.rect
            title_block_bbox = (rect.x0, rect.y0, rect.x1, rect.y1)

        # Get title zone within title block
        title_zone_bbox = get_title_zone_bbox(title_block_bbox)

        # Extract and score text spans
        candidates = []

        for block in blocks:
            if block.get("type") != 0:  # 0 = text block
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text or len(text) < 3:
                        continue

                    span_bbox = span.get("bbox", (0, 0, 0, 0))
                    font_size = span.get("size", 12.0)

                    # Score this candidate
                    score_result = self._score_candidate(
                        text=text,
                        bbox=span_bbox,
                        font_size=font_size,
                        title_block_bbox=title_block_bbox,
                        title_zone_bbox=title_zone_bbox,
                    )

                    if score_result["score"] > 0:  # Only keep non-disqualified
                        candidates.append({
                            "text": text,
                            "bbox": span_bbox,
                            "font_size": font_size,
                            "score": score_result["score"],
                            "zone": score_result["zone"],
                            "reasons": score_result["reasons"],
                        })

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Select best candidate
        if candidates:
            best = candidates[0]

            # Calculate confidence (max 0.80 for spatial)
            # Higher scores get higher confidence, but capped at 0.80
            base_confidence = TITLE_CONFIDENCE.get('SPATIAL', 0.80)

            # Adjust based on score (scores typically range 0-100)
            if best["score"] >= 50:
                confidence = base_confidence
            elif best["score"] >= 30:
                confidence = base_confidence - 0.10
            else:
                confidence = base_confidence - 0.20

            # Bonus for valid keywords
            if self._has_valid_keyword(best["text"]):
                confidence += TITLE_CONFIDENCE.get('KEYWORD_BONUS', 0.05)

            # Clamp to [0.0, 0.80]
            confidence = max(0.0, min(0.80, confidence))

            return {
                "title": best["text"],
                "confidence": confidence,
                "candidates": candidates[:5],  # Top 5 for debugging
                "method": "spatial",
            }

        return {
            "title": None,
            "confidence": 0.0,
            "candidates": [],
            "method": None,
        }

    def _score_candidate(
        self,
        text: str,
        bbox: Tuple[float, float, float, float],
        font_size: float,
        title_block_bbox: Tuple[float, float, float, float],
        title_zone_bbox: Tuple[float, float, float, float],
    ) -> Dict[str, Any]:
        """
        Score a text candidate for being the sheet title.

        Scoring criteria from spec:
        - Larger font size = higher score (+10 per 2pt above 12)
        - Contains valid keyword = +15
        - Centered in title zone = +10
        - Length 10-30 chars = +5
        - Length > 35 chars = -10
        - Is a label = DISQUALIFY (score = 0)

        Args:
            text: Text content
            bbox: Text bounding box in points
            font_size: Font size in points
            title_block_bbox: Title block bounding box in points
            title_zone_bbox: Title zone bounding box in points

        Returns:
            Dict with score, zone, and reasons
        """
        score = 0
        reasons = []
        text_upper = text.upper().strip()

        # Check if it's a label to skip
        if self._is_label(text_upper):
            return {"score": 0, "zone": "skip", "reasons": ["Is a label"]}

        # Skip very small fonts
        if font_size < MIN_FONT_SIZE:
            return {"score": 0, "zone": "skip", "reasons": ["Font too small"]}

        # Determine which zone the text is in
        text_center_y = (bbox[1] + bbox[3]) / 2
        zone = get_vertical_zone(text_center_y, title_block_bbox)

        # Text should be in title zone for best score
        if zone == "title":
            score += 20
            reasons.append("In title zone (+20)")
        elif zone == "header":
            score += 5
            reasons.append("In header zone (+5)")
        else:  # info zone
            score -= 10
            reasons.append("In info zone (-10)")

        # Font size scoring: +10 per 2pt above base
        if font_size > BASE_FONT_SIZE:
            font_bonus = int((font_size - BASE_FONT_SIZE) / 2) * 10
            font_bonus = min(font_bonus, 50)  # Cap at +50
            score += font_bonus
            reasons.append(f"Font size {font_size:.1f}pt (+{font_bonus})")

        # Keyword bonus
        if self._has_valid_keyword(text):
            score += 15
            reasons.append("Contains valid keyword (+15)")

        # Check if centered in title zone
        if zone == "title":
            tb_center_x = (title_block_bbox[0] + title_block_bbox[2]) / 2
            text_center_x = (bbox[0] + bbox[2]) / 2
            tb_width = title_block_bbox[2] - title_block_bbox[0]

            # Within 20% of center
            if abs(text_center_x - tb_center_x) < tb_width * 0.2:
                score += 10
                reasons.append("Centered in title zone (+10)")

        # Length scoring
        text_len = len(text)
        if 10 <= text_len <= 30:
            score += 5
            reasons.append(f"Good length {text_len} (+5)")
        elif text_len > 35:
            score -= 10
            reasons.append(f"Too long {text_len} (-10)")
        elif text_len < 5:
            score -= 5
            reasons.append(f"Too short {text_len} (-5)")

        return {"score": score, "zone": zone, "reasons": reasons}

    def _is_label(self, text_upper: str) -> bool:
        """
        Check if text is a field label that should be skipped.

        Args:
            text_upper: Uppercase text

        Returns:
            True if text is a label
        """
        # Exact match
        if text_upper in SKIP_LABELS:
            return True

        # Starts with label
        for label in SKIP_LABELS:
            if text_upper.startswith(label + ':') or text_upper.startswith(label + ' '):
                return True
            if text_upper == label + '.':
                return True

        # Contains colon at end (likely a label)
        if text_upper.endswith(':'):
            return True

        # Common label patterns
        if re.match(r'^(NO|#|REV|SHEET|PROJECT|SCALE|DATE|DWG)[:\.\s]', text_upper):
            return True

        return False

    def _has_valid_keyword(self, text: str) -> bool:
        """Check if text contains a valid title keyword."""
        text_upper = text.upper()
        for keyword in VALID_TITLE_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_upper):
                return True
        return False

    def is_vector_page(self, page) -> bool:
        """
        Check if a page has enough embedded text to be considered vector.

        Args:
            page: PyMuPDF page object

        Returns:
            True if page has >= 50 characters of embedded text
        """
        text = page.get_text()
        return len(text.strip()) >= 50
