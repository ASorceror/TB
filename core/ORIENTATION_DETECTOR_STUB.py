"""
Blueprint Processor - Orientation Detector Stub

This is a stub/template for implementing the crop orientation detection system.

The orientation detector identifies and corrects 90° rotations in title block crops
before they're passed to OCR. This improves OCR accuracy on rotated text.

IMPLEMENTATION STEPS:
1. Choose detection algorithm (see recommendations below)
2. Implement detect_orientation() function
3. Implement detect_and_correct_orientation() wrapper (if needed)
4. Run tests/test_orientation_template.py to validate
5. Integrate into sheet_title_extractor.py

RECOMMENDED ALGORITHM: Hough Line Detection
- Fast (~25ms per crop)
- Robust to noise and sparse text
- Works well on architectural drawings with clear lines
- Handles edge cases gracefully

See ORIENTATION_VALIDATION_STRATEGY.md for complete test plan and metrics.

Author: Implementation Template
Date: 2025-12-30
Version: 0.1 (Stub)
"""

import logging
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


def detect_orientation(crop: Image.Image) -> Tuple[int, float]:
    """
    Detect the orientation of a title block crop.

    This function analyzes a crop image and determines if it's rotated 0°, 90°,
    180°, or 270° from the normal orientation.

    Args:
        crop: PIL Image of a title block crop

    Returns:
        Tuple of (angle, confidence):
            - angle: Detected rotation in degrees {0, 90, 180, 270}
            - confidence: Detection confidence score [0.0, 1.0]
                  0.95+ = Very confident (normal text orientation)
                  0.85-0.94 = Confident
                  0.70-0.84 = Moderate confidence
                  0.50-0.69 = Low confidence (flag as uncertain)
                  < 0.50 = Very uncertain (recommend default to 0°)

    Example:
        >>> crop = Image.open('p001_titleblock.png')
        >>> angle, confidence = detect_orientation(crop)
        >>> print(f"Detected {angle}° rotation with {confidence:.2f} confidence")
        Detected 0 degrees rotation with 0.95 confidence
    """
    # TODO: Implement your detection algorithm here
    # This is a stub that always returns 0° with 0.95 confidence

    # STEP 1: Input validation
    if crop is None:
        logger.warning("Received None crop image")
        return 0, 0.0

    if crop.mode not in ['RGB', 'L', 'RGBA']:
        logger.debug(f"Converting image mode from {crop.mode} to RGB")
        crop = crop.convert('RGB')

    # STEP 2: Check for blank/empty crops
    if _is_blank_crop(crop):
        logger.debug("Detected blank/empty crop - returning default 0° with low confidence")
        return 0, 0.30  # Low confidence for blank crops

    # STEP 3: Convert to numpy array for processing
    img_array = np.array(crop)

    # TODO: IMPLEMENT YOUR ALGORITHM HERE
    # Example approaches:
    # 1. Hough Line Detection (RECOMMENDED)
    # 2. Text Line Angle Analysis
    # 3. Gradient Analysis
    # 4. Consensus (combine multiple methods)
    #
    # For now, return safe defaults:

    angle = 0
    confidence = 0.95

    # STEP 4: Validate results
    if angle not in [0, 90, 180, 270]:
        logger.warning(f"Invalid angle detected: {angle}°, defaulting to 0°")
        angle = 0
        confidence = 0.50

    if not (0.0 <= confidence <= 1.0):
        logger.warning(f"Invalid confidence: {confidence}, clamping to [0.0, 1.0]")
        confidence = max(0.0, min(1.0, confidence))

    logger.debug(f"Detected orientation: {angle}° (confidence: {confidence:.2f})")
    return angle, confidence


def detect_and_correct_orientation(crop: Image.Image) -> Dict[str, Any]:
    """
    Detect orientation and apply correction if needed.

    This is a convenience wrapper that calls detect_orientation() and then
    rotates the image back to normal orientation if needed.

    Args:
        crop: PIL Image of a title block crop

    Returns:
        Dict with keys:
            - 'image': Corrected PIL Image (same orientation as input if 0°)
            - 'detected_angle': Detected rotation angle {0, 90, 180, 270}
            - 'confidence': Detection confidence [0.0, 1.0]
            - 'rotation_applied': Degrees rotated to correct (-angle if rotated, 0 if normal)
            - 'was_rotated': Boolean - was correction applied?

    Example:
        >>> crop = Image.open('p001_titleblock.png')
        >>> result = detect_and_correct_orientation(crop)
        >>> if result['was_rotated']:
        ...     print(f"Corrected {result['detected_angle']}° rotation")
        ...     corrected_crop = result['image']
    """
    # Detect orientation
    angle, confidence = detect_orientation(crop)

    # Determine if rotation is needed
    # Use confidence threshold to avoid rotating uncertain cases
    CONFIDENCE_THRESHOLD = 0.70  # Only rotate if confidence > 70%
    should_rotate = (angle != 0) and (confidence >= CONFIDENCE_THRESHOLD)

    if should_rotate:
        # Apply correction (negative angle to rotate back)
        correction_angle = -angle
        corrected_image = crop.rotate(
            correction_angle,
            expand=False,  # Keep original size
            fillcolor='white'  # Fill empty corners with white
        )
        logger.debug(f"Applied {correction_angle}° rotation (detected: {angle}°, conf: {confidence:.2f})")
    else:
        # No rotation needed
        corrected_image = crop
        logger.debug(f"No rotation needed (angle: {angle}°, conf: {confidence:.2f})")

    return {
        'image': corrected_image,
        'detected_angle': angle,
        'confidence': confidence,
        'rotation_applied': -angle if should_rotate else 0,
        'was_rotated': should_rotate,
    }


# ============================================================================
# HELPER FUNCTIONS - Implement based on your chosen algorithm
# ============================================================================

def _is_blank_crop(img: Image.Image, threshold: float = 0.95) -> bool:
    """
    Check if a crop is mostly blank/white.

    Args:
        img: PIL Image
        threshold: Percentage of white pixels to consider blank (0.0-1.0)

    Returns:
        True if crop is mostly blank, False otherwise
    """
    # Convert to grayscale
    if img.mode != 'L':
        gray = img.convert('L')
    else:
        gray = img

    # Convert to numpy array
    arr = np.array(gray)

    # Count white pixels (value >= 240)
    white_pixels = np.sum(arr >= 240)
    total_pixels = arr.size

    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0

    return white_ratio >= threshold


# ============================================================================
# ALGORITHM IMPLEMENTATIONS - Choose one and implement
# ============================================================================

def _detect_orientation_hough_lines(img_array: np.ndarray) -> Tuple[int, float]:
    """
    Detect orientation using Hough line detection.

    RECOMMENDED APPROACH for this use case:
    - Fast: ~15-25ms per crop
    - Robust: Works well with sparse text
    - Simple: Clear implementation

    Algorithm:
    1. Convert to grayscale
    2. Apply edge detection (Canny)
    3. Apply Hough line transform
    4. Cluster detected lines by angle
    5. Find dominant angle
    6. Map to nearest 90° rotation

    Args:
        img_array: Numpy array of image

    Returns:
        Tuple of (angle, confidence)

    Reference Implementation:
        https://docs.opencv.org/3.4/d3/de5/tutorial_js_houghlines.html
    """
    # TODO: Implement Hough line detection
    # This requires OpenCV (cv2)
    #
    # Pseudocode:
    # 1. gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # 2. edges = cv2.Canny(gray, 50, 150)
    # 3. lines = cv2.HoughLinesP(edges, ...)
    # 4. Extract angles from lines
    # 5. Cluster by angle
    # 6. Return dominant angle

    return 0, 0.95


def _detect_orientation_text_angle(img_array: np.ndarray) -> Tuple[int, float]:
    """
    Detect orientation using text line analysis.

    ALTERNATIVE APPROACH:
    - Uses Tesseract to get text bounding boxes
    - Analyzes angle of bounding box clusters
    - More accurate for dense text

    Requires: pytesseract, Tesseract OCR

    Algorithm:
    1. Run Tesseract with angle detection (use_angle_cls=True)
    2. Get text bounding boxes
    3. Calculate angle of box cluster
    4. Map to nearest 90° rotation

    Args:
        img_array: Numpy array of image

    Returns:
        Tuple of (angle, confidence)
    """
    # TODO: Implement text-based angle detection
    # This requires Tesseract OCR setup
    #
    # Pseudocode:
    # 1. Convert array back to PIL Image
    # 2. Run pytesseract.image_to_data(..., output_type=Output.DICT)
    # 3. Get 'angle' field from results
    # 4. Cluster angles
    # 5. Return dominant angle

    return 0, 0.95


def _detect_orientation_gradient(img_array: np.ndarray) -> Tuple[int, float]:
    """
    Detect orientation using image gradient analysis.

    SIMPLE APPROACH:
    - Fast, no dependencies
    - Works on any image (even sparse text)

    Algorithm:
    1. Compute Sobel gradients (dx, dy)
    2. Calculate angle: atan2(dy, dx)
    3. Histogram of angles
    4. Find peaks
    5. Map to nearest 90° rotation

    Args:
        img_array: Numpy array of image

    Returns:
        Tuple of (angle, confidence)
    """
    # TODO: Implement gradient-based detection
    # Can be done with numpy only (no OpenCV needed)
    #
    # Pseudocode:
    # 1. Gray = RGB to grayscale
    # 2. gx = sobel(gray, x)
    # 3. gy = sobel(gray, y)
    # 4. angles = atan2(gy, gx)
    # 5. hist = histogram(angles, bins=180)
    # 6. Find peaks (multiple, for 0°, 90°, etc.)
    # 7. Return dominant peak

    return 0, 0.95


def _detect_orientation_consensus(
    img_array: np.ndarray,
    methods: list = None
) -> Tuple[int, float]:
    """
    Detect orientation using consensus of multiple methods.

    ROBUST APPROACH:
    - Combines strengths of multiple algorithms
    - Fallback if one method fails
    - Higher confidence

    Methods (if available):
    - Hough lines
    - Text angle
    - Gradient analysis

    Args:
        img_array: Numpy array of image
        methods: List of detection functions to use (default: all available)

    Returns:
        Tuple of (angle, confidence)

    Algorithm:
    1. Run each available method
    2. Collect results
    3. If all agree: high confidence
    4. If 2/3 agree: medium confidence
    5. If no consensus: return safe default (0°) with low confidence
    """
    # TODO: Implement consensus approach
    #
    # Pseudocode:
    # 1. results = []
    # 2. For each method:
    #      results.append(method(img_array))
    # 3. Count votes for each angle
    # 4. If unanimous: confidence = 0.95
    # 5. If 2/3: confidence = 0.80
    # 6. Else: confidence = 0.50, angle = 0

    return 0, 0.95


# ============================================================================
# VALIDATION HELPERS - Use in testing
# ============================================================================

def validate_output(angle: int, confidence: float) -> bool:
    """
    Validate detection output is well-formed.

    Args:
        angle: Detected angle
        confidence: Confidence score

    Returns:
        True if valid, False otherwise
    """
    # Check angle
    if angle not in [0, 90, 180, 270]:
        logger.warning(f"Invalid angle: {angle}")
        return False

    # Check confidence
    if not (0.0 <= confidence <= 1.0):
        logger.warning(f"Invalid confidence: {confidence}")
        return False

    return True


def get_algorithm_info() -> Dict[str, Any]:
    """
    Return information about the implemented algorithm.

    Returns:
        Dict with algorithm metadata
    """
    return {
        'name': 'Stub Implementation',
        'version': '0.1',
        'status': 'Not Implemented',
        'description': 'Placeholder for orientation detection algorithm',
        'supported_angles': [0, 90, 180, 270],
        'typical_speed_ms': None,
        'dependencies': [],
        'notes': 'Replace stub with actual implementation',
    }


# ============================================================================
# INTEGRATION POINTS
# ============================================================================

# For integration into sheet_title_extractor.py:
#
# In sheet_title_extractor.py, before OCR:
#
#     # Before: title_block_image (may be rotated)
#
#     from core.orientation_detector import detect_and_correct_orientation
#
#     result = detect_and_correct_orientation(title_block_image)
#     title_block_image = result['image']  # Use corrected image
#     orientation_info = {
#         'detected_angle': result['detected_angle'],
#         'confidence': result['confidence'],
#         'was_corrected': result['was_rotated'],
#     }
#
#     # After: title_block_image is normalized to 0° orientation
#     # Pass to OCR as normal

if __name__ == '__main__':
    """Test the orientation detector with sample data."""

    # Test 1: Normal crop
    print("Test 1: Normal orientation")
    from PIL import Image
    try:
        # Try to load a real crop if available
        crop_path = Path(__file__).parent.parent / 'output' / 'crops' / '2f36353b' / 'p001_titleblock.png'
        if crop_path.exists():
            crop = Image.open(crop_path)
            angle, conf = detect_orientation(crop)
            print(f"  Result: {angle}° (confidence: {conf:.2f})")
        else:
            print(f"  Crop not found at {crop_path}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 2: Algorithm info
    print("\nAlgorithm Information:")
    info = get_algorithm_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nIMPLEMENTATION STATUS: Stub - awaiting algorithm implementation")
    print("See ORIENTATION_VALIDATION_STRATEGY.md for details")
