"""
Blueprint Processor V5.0 - Region Detector
Locates title block region in normalized blueprint images.

V5.0: Use LSTM-friendly preprocessing (no binarization) for OCR.
V4.5: Dynamic region calculation based on page size.
V4.9: Centralized find_tesseract(), added preprocessing for OCR.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import re

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    TITLE_BLOCK_REGIONS,
    REGION_SEARCH_ORDER,
    TITLE_BLOCK_KEYWORDS,
    TESSERACT_CONFIG,
    THRESHOLDS,
    CONFIDENCE_LEVELS,
    DEFAULT_DPI,
    TITLE_BLOCK_PHYSICAL,
)
from core.ocr_utils import find_tesseract, preprocess_for_ocr

# Try to import pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class RegionDetector:
    """
    Detects and locates title block region in blueprint images.
    Uses keyword matching to validate detected regions.
    """

    def __init__(self, telemetry_dir: Optional[Path] = None):
        """
        Initialize RegionDetector.

        Args:
            telemetry_dir: Directory to save telemetry JSON files
        """
        self.telemetry_dir = telemetry_dir

        if TESSERACT_AVAILABLE:
            tesseract_path = find_tesseract()
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def _calculate_dynamic_regions(
        self,
        width: int,
        height: int,
        dpi: int = DEFAULT_DPI
    ) -> Tuple[Dict[str, Tuple[float, float, float, float]], List[str]]:
        """
        Calculate title block regions based on physical page size.

        Standard architectural title blocks are ~6" wide x 4" tall,
        regardless of sheet size. This calculates appropriate percentages
        for the actual page dimensions.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            dpi: Rendering DPI (default from constants)

        Returns:
            Tuple of (regions_dict, search_order_list)
        """
        # Calculate physical page dimensions
        page_width_inches = width / dpi
        page_height_inches = height / dpi

        # Get standard title block dimensions from constants
        tb_width = TITLE_BLOCK_PHYSICAL['width_inches']
        tb_height = TITLE_BLOCK_PHYSICAL['height_inches']
        max_width_pct = TITLE_BLOCK_PHYSICAL['max_width_pct']
        max_height_pct = TITLE_BLOCK_PHYSICAL['max_height_pct']

        # Calculate as percentage of page (clamped to max)
        tb_width_pct = min(max_width_pct, tb_width / page_width_inches)
        tb_height_pct = min(max_height_pct, tb_height / page_height_inches)

        # Log the calculation for debugging
        logger.debug(
            f"Dynamic regions: page={page_width_inches:.1f}x{page_height_inches:.1f} inches, "
            f"title_block={tb_width_pct*100:.1f}%x{tb_height_pct*100:.1f}%"
        )

        # Build dynamic regions - tightest first
        regions = {
            # Primary: tight title block region (most likely correct)
            'bottom_right_tight': (
                1.0 - tb_width_pct,
                1.0 - tb_height_pct,
                1.0,
                1.0
            ),
            # Extended: 30% wider for edge cases
            'bottom_right': (
                1.0 - min(max_width_pct, tb_width_pct * 1.3),
                1.0 - tb_height_pct,
                1.0,
                1.0
            ),
            # Right strip: full height for tall narrow title blocks
            'right_strip': (
                1.0 - tb_width_pct,
                0.0,
                1.0,
                1.0
            ),
            # Right strip bottom: bottom half only (alternative)
            'right_strip_bottom': (
                1.0 - tb_width_pct,
                0.5,
                1.0,
                1.0
            ),
            # Bottom left: for alternate formats
            'bottom_left': (
                0.0,
                1.0 - tb_height_pct,
                tb_width_pct,
                1.0
            ),
            # Bottom strip: full width, reduced height
            'bottom_strip': (
                0.0,
                1.0 - tb_height_pct * 0.75,
                1.0,
                1.0
            ),
        }

        # Search order: tight first, then progressively wider
        # V4.5: Added right_strip_bottom before full right_strip
        search_order = [
            'bottom_right_tight',
            'bottom_right',
            'right_strip_bottom',  # Try bottom portion first
            'right_strip',         # Then full height
            'bottom_left',
            'bottom_strip',
        ]

        return regions, search_order

    def detect_title_block(self, image: Image.Image, dpi: int = DEFAULT_DPI) -> Dict[str, Any]:
        """
        Detect title block region in the image.

        Searches candidate regions in order, scores each by keyword matches,
        and selects the highest-scoring region with >= 2 keywords.

        V4.5: Uses dynamic region calculation based on page physical size.

        Args:
            image: Normalized PIL Image
            dpi: DPI used for rendering (default 200)

        Returns:
            Dict with keys: bbox, confidence, keywords_found, region_name,
                           all_candidates, selection_reason
        """
        width, height = image.size

        # V4.5: Calculate dynamic regions based on page size
        regions, search_order = self._calculate_dynamic_regions(width, height, dpi)

        candidates = []
        best_candidate = None
        best_score = -1

        # Search regions in calculated order (tightest first)
        for region_name in search_order:
            region_coords = regions[region_name]

            # Convert relative coords to absolute pixels
            x1 = int(region_coords[0] * width)
            y1 = int(region_coords[1] * height)
            x2 = int(region_coords[2] * width)
            y2 = int(region_coords[3] * height)

            bbox = (x1, y1, x2, y2)

            # Crop region
            cropped = image.crop(bbox)

            # OCR the cropped region
            text = self._ocr_region(cropped)

            # Count keyword matches
            keywords_found = self._count_keywords(text)

            # Calculate text density
            text_density = len(text.strip()) / max(1, (x2 - x1) * (y2 - y1) / 1000)

            # Score: keyword_count + density_bonus
            density_bonus = 0.5 if text_density > THRESHOLDS['min_text_density'] else 0
            score = len(keywords_found) + density_bonus

            candidate = {
                'region_name': region_name,
                'bbox': bbox,
                'relative_coords': region_coords,
                'keywords_found': keywords_found,
                'keyword_count': len(keywords_found),
                'text_length': len(text),
                'text_density': text_density,
                'score': score,
                'text_preview': text[:200] if text else '',
            }
            candidates.append(candidate)

            # Track best candidate with >= 2 keywords
            if len(keywords_found) >= THRESHOLDS['min_keywords_for_valid_region']:
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

        # Determine result
        if best_candidate is not None:
            confidence = CONFIDENCE_LEVELS['HIGH'] if best_candidate['keyword_count'] >= 3 else CONFIDENCE_LEVELS['MEDIUM']
            selection_reason = f"Region '{best_candidate['region_name']}' selected with {best_candidate['keyword_count']} keywords (score: {best_candidate['score']:.2f})"
        else:
            # Default to bottom_right_tight (dynamic) with LOW confidence
            default_coords = regions.get('bottom_right_tight', (0.82, 0.75, 1.0, 1.0))
            x1 = int(default_coords[0] * width)
            y1 = int(default_coords[1] * height)
            x2 = int(default_coords[2] * width)
            y2 = int(default_coords[3] * height)

            best_candidate = {
                'region_name': 'bottom_right_tight',
                'bbox': (x1, y1, x2, y2),
                'relative_coords': default_coords,
                'keywords_found': [],
                'keyword_count': 0,
                'score': 0,
            }
            confidence = CONFIDENCE_LEVELS['LOW']
            selection_reason = "No region with >= 2 keywords found. Using default bottom_right_tight."

        result = {
            'bbox': best_candidate['bbox'],
            'confidence': confidence,
            'keywords_found': best_candidate['keywords_found'],
            'region_name': best_candidate['region_name'],
            'score': best_candidate['score'],
            'all_candidates': candidates,
            'selection_reason': selection_reason,
            'image_size': (width, height),
        }

        return result

    def _ocr_region(self, image: Image.Image, preprocess: bool = True) -> str:
        """
        OCR a cropped region with optional preprocessing.

        V4.9: Added preprocessing for improved OCR accuracy.

        Args:
            image: Cropped PIL Image
            preprocess: Whether to apply preprocessing (default: True)

        Returns:
            Extracted text
        """
        if not TESSERACT_AVAILABLE:
            return ''

        try:
            # V5.0: Use LSTM-friendly preprocessing (no binarization)
            # Research: Tesseract LSTM does internal preprocessing, external binarization hurts accuracy
            if preprocess:
                processed_image = preprocess_for_ocr(
                    image,
                    apply_grayscale=True,
                    apply_denoise=True,
                    apply_border=True,
                    border_size=10,
                    invert_if_light_text=True,
                    preprocessing_mode='lstm',  # Skip binarization for LSTM
                )
            else:
                processed_image = image

            config = TESSERACT_CONFIG['title_block']
            text = pytesseract.image_to_string(processed_image, config=config)
            return text
        except Exception as e:
            logger.debug(f"OCR region failed: {e}")
            return ''

    def _count_keywords(self, text: str) -> List[str]:
        """
        Count title block keywords found in text.

        Args:
            text: Text to search

        Returns:
            List of keywords found
        """
        if not text:
            return []

        text_upper = text.upper()
        found = []

        for keyword in TITLE_BLOCK_KEYWORDS:
            # Use word boundary matching to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_upper):
                found.append(keyword)

        return found

    def crop_title_block(self, image: Image.Image,
                         detection_result: Dict[str, Any]) -> Image.Image:
        """
        Crop the detected title block region from the image.

        Args:
            image: Full page image
            detection_result: Result from detect_title_block()

        Returns:
            Cropped PIL Image of title block
        """
        bbox = detection_result['bbox']
        return image.crop(bbox)

    def save_telemetry(self, filename: str, detection_result: Dict[str, Any],
                       cropped_image: Optional[Image.Image] = None):
        """
        Save telemetry JSON file with detection details.

        Args:
            filename: Base filename for telemetry
            detection_result: Detection results
            cropped_image: Optional cropped image to save
        """
        if self.telemetry_dir is None:
            return

        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON telemetry
        json_path = self.telemetry_dir / f"{filename}_titleblock.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detection_result, f, indent=2, default=str)

        # Save cropped image if provided
        if cropped_image is not None:
            img_path = self.telemetry_dir / f"{filename}_titleblock.png"
            cropped_image.save(img_path)


def get_region_visualization(image: Image.Image,
                            detection_result: Dict[str, Any]) -> Image.Image:
    """
    Create a visualization of detected regions on the image.
    Draws bounding boxes for all candidates and highlights the selected one.

    Args:
        image: Original image
        detection_result: Detection results

    Returns:
        PIL Image with region visualizations
    """
    try:
        from PIL import ImageDraw
    except ImportError:
        return image

    # Create a copy to draw on
    viz = image.copy()
    draw = ImageDraw.Draw(viz)

    # Draw all candidates in light gray
    for candidate in detection_result.get('all_candidates', []):
        bbox = candidate['bbox']
        color = 'gray'
        width = 2

        # Highlight selected region in green
        if candidate['region_name'] == detection_result['region_name']:
            color = 'green'
            width = 5

        draw.rectangle(bbox, outline=color, width=width)

        # Add label
        label = f"{candidate['region_name']}: {candidate['keyword_count']} keywords"
        draw.text((bbox[0] + 5, bbox[1] + 5), label, fill=color)

    return viz
