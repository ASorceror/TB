"""
Blueprint Processor V6.2.1 - Page Normalizer
Detects orientation and rotates pages to 0 degrees.
Implements ROTATION-FIRST principle: correct orientation BEFORE title block detection.

V6.2.1 Changes:
- Added MIN_ROTATION_CONFIDENCE threshold (2.0) to prevent false rotations
- Low-confidence OSD detections now skip rotation instead of applying wrong angle

V4.9 Changes:
- Centralized find_tesseract() moved to ocr_utils.py
- Added deskew_image() for small angle correction
- Added apply_deskew parameter to normalize() method
"""

# Minimum confidence required to apply rotation
# Tesseract OSD with confidence < 2.0 is unreliable and often wrong
MIN_ROTATION_CONFIDENCE = 2.0

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

from constants import TESSERACT_CONFIG, THRESHOLDS
from core.ocr_utils import find_tesseract, deskew_image

# Try to import pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Try to import cv2 for line-based fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class PageNormalizer:
    """
    Detects and corrects page orientation.
    Implements ROTATION-FIRST principle.
    """

    def __init__(self, telemetry_dir: Optional[Path] = None):
        """
        Initialize PageNormalizer.

        Args:
            telemetry_dir: Directory to save telemetry JSON files
        """
        self.telemetry_dir = telemetry_dir

        # Configure Tesseract path if available
        if TESSERACT_AVAILABLE:
            tesseract_path = find_tesseract()
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def detect_orientation(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect page orientation using Tesseract OSD.

        Args:
            image: PIL Image to analyze

        Returns:
            Dict with keys: angle, confidence, method
            angle is 0, 90, 180, or 270 degrees
        """
        result = {
            'angle': 0,
            'confidence': 0.0,
            'method': 'none',
            'raw_output': None,
        }

        # Try Tesseract OSD first
        if TESSERACT_AVAILABLE:
            try:
                osd_result = self._detect_with_tesseract_osd(image)
                if osd_result['confidence'] > 0:
                    return osd_result
            except Exception as e:
                result['tesseract_error'] = str(e)

        # Fallback to line-angle detection
        if CV2_AVAILABLE:
            try:
                line_result = self._detect_with_line_angles(image)
                if line_result['confidence'] > 0:
                    return line_result
            except Exception as e:
                result['line_detection_error'] = str(e)

        return result

    def _detect_with_tesseract_osd(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect orientation using Tesseract OSD (--psm 0).

        Args:
            image: PIL Image

        Returns:
            Dict with angle, confidence, method
        """
        config = TESSERACT_CONFIG['osd']

        try:
            osd_output = pytesseract.image_to_osd(image, config=config)

            # Parse output for orientation
            angle = 0
            confidence = 0.0

            for line in osd_output.split('\n'):
                if 'Orientation in degrees:' in line:
                    angle = int(line.split(':')[1].strip())
                elif 'Orientation confidence:' in line:
                    confidence = float(line.split(':')[1].strip())

            return {
                'angle': angle,
                'confidence': confidence,
                'method': 'tesseract_osd',
                'raw_output': osd_output,
            }
        except pytesseract.TesseractError as e:
            # OSD may fail on images with too little text
            return {
                'angle': 0,
                'confidence': 0.0,
                'method': 'tesseract_osd_failed',
                'error': str(e),
            }

    def _detect_with_line_angles(self, image: Image.Image) -> Dict[str, Any]:
        """
        Fallback: Detect orientation using line angle analysis.
        Most blueprint lines should be horizontal/vertical when correctly oriented.

        Args:
            image: PIL Image

        Returns:
            Dict with angle, confidence, method
        """
        # Convert to grayscale numpy array
        img_array = np.array(image.convert('L'))

        # Edge detection
        edges = cv2.Canny(img_array, 50, 150, apertureSize=3)

        # Calculate relative thresholds
        height, width = img_array.shape
        min_line_length = int(width * THRESHOLDS['min_line_length'])
        max_line_gap = int(width * THRESHOLDS['max_line_gap'])

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        if lines is None or len(lines) < 10:
            return {
                'angle': 0,
                'confidence': 0.0,
                'method': 'line_detection_insufficient',
            }

        # Calculate angles of all detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

        angles = np.array(angles)

        # Normalize angles to 0-90 range (since lines are bidirectional)
        normalized = np.abs(angles) % 90

        # Count lines near 0 degrees (horizontal) and near 45 degrees
        horizontal_count = np.sum((normalized < 5) | (normalized > 85))
        diagonal_count = np.sum((normalized > 40) & (normalized < 50))

        total_lines = len(angles)
        horizontal_ratio = horizontal_count / total_lines

        # If most lines are diagonal (40-50 degrees), image is likely rotated 45 degrees
        # But blueprints are typically rotated 90 degrees, not 45

        # Analyze which quadrant dominates
        # 0 degrees: lines horizontal
        # 90 degrees: lines vertical

        # For 90-degree rotation detection, check if vertical lines dominate
        near_vertical = np.sum((np.abs(angles - 90) < 10) | (np.abs(angles + 90) < 10))
        near_horizontal = np.sum(np.abs(angles) < 10)

        detected_angle = 0
        confidence = horizontal_ratio

        if near_vertical > near_horizontal * 1.5:
            # More vertical lines than horizontal - might be rotated 90 degrees
            detected_angle = 90
            confidence = near_vertical / total_lines

        return {
            'angle': detected_angle,
            'confidence': confidence,
            'method': 'line_angle_detection',
            'total_lines': total_lines,
            'horizontal_ratio': horizontal_ratio,
        }

    def normalize(self, image: Image.Image,
                  force_angle: Optional[int] = None,
                  apply_deskew: bool = True,
                  max_deskew_angle: float = 5.0) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Detect orientation and rotate image to 0 degrees.

        V4.9: Added deskew capability for small angle correction.
        Tesseract accuracy drops significantly with skew > 2 degrees.

        Args:
            image: PIL Image to normalize
            force_angle: Optional angle to force (overrides detection)
            apply_deskew: Whether to apply small-angle deskew correction (default: True)
            max_deskew_angle: Maximum skew angle to correct (default: 5.0 degrees)

        Returns:
            Tuple of (rotated_image, orientation_info)
        """
        if force_angle is not None:
            orientation_info = {
                'angle': force_angle,
                'confidence': 1.0,
                'method': 'forced',
            }
        else:
            orientation_info = self.detect_orientation(image)

        angle = orientation_info['angle']
        confidence = orientation_info.get('confidence', 0.0)

        # V6.2.1: Only apply rotation if confidence is above threshold
        # Low-confidence detections are often wrong and cause more harm than good
        if confidence < MIN_ROTATION_CONFIDENCE and angle != 0:
            logger.debug(f"Skipping rotation (angle={angle}°, confidence={confidence:.2f} < {MIN_ROTATION_CONFIDENCE})")
            orientation_info['rotation_skipped'] = True
            orientation_info['skip_reason'] = f'confidence {confidence:.2f} < threshold {MIN_ROTATION_CONFIDENCE}'
            angle = 0  # Don't rotate

        # V6.2.2: Skip 180° rotation for blueprints - OSD is unreliable
        # Blueprints have text in multiple orientations which confuses OSD
        # 180° scans are very rare, but false positives are common
        if angle == 180:
            logger.debug(f"Skipping 180° rotation (OSD unreliable for blueprints, conf={confidence:.2f})")
            orientation_info['rotation_skipped'] = True
            orientation_info['skip_reason'] = '180° rotation disabled for blueprints'
            angle = 0

        # Rotate image to correct orientation (90-degree increments)
        if angle == 90:
            rotated = image.transpose(Image.ROTATE_270)
        elif angle == 180:
            rotated = image.transpose(Image.ROTATE_180)  # Kept for completeness but won't trigger
        elif angle == 270:
            rotated = image.transpose(Image.ROTATE_90)
        else:
            rotated = image

        # V4.9: Apply small-angle deskew correction
        deskew_angle = 0.0
        if apply_deskew:
            try:
                rotated, deskew_angle = deskew_image(rotated, max_angle=max_deskew_angle)
                if abs(deskew_angle) > 0.1:
                    logger.debug(f"Applied deskew correction of {deskew_angle:.2f}°")
            except Exception as e:
                logger.warning(f"Deskew failed: {e}")
                deskew_angle = 0.0

        orientation_info['original_size'] = image.size
        orientation_info['normalized_size'] = rotated.size
        orientation_info['rotation_applied'] = angle
        orientation_info['deskew_angle'] = deskew_angle
        orientation_info['deskew_applied'] = abs(deskew_angle) > 0.1

        return rotated, orientation_info

    def save_telemetry(self, filename: str, orientation_info: Dict[str, Any],
                       image: Optional[Image.Image] = None):
        """
        Save telemetry JSON file with orientation detection details.

        Args:
            filename: Base filename for telemetry
            orientation_info: Orientation detection results
            image: Optional image to save for visual verification
        """
        if self.telemetry_dir is None:
            return

        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON telemetry
        json_path = self.telemetry_dir / f"{filename}_orientation.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Filter out non-serializable items
            serializable = {k: v for k, v in orientation_info.items()
                          if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            json.dump(serializable, f, indent=2)

        # Save image if provided
        if image is not None:
            img_path = self.telemetry_dir / f"{filename}_normalized.png"
            image.save(img_path)
