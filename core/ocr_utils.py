"""
Blueprint Processor V5.0 - OCR Utilities
Centralized OCR utility functions for Tesseract configuration and image preprocessing.

V5.0 Changes:
- Added LSTM-friendly preprocessing mode (skips binarization for Tesseract 4.0+ LSTM)
- Added morphological dilation option to preserve thin characters
- New preprocessing_mode parameter: 'lstm' (minimal), 'standard' (balanced), 'legacy' (aggressive)
- Research-backed: LSTM engine does internal preprocessing, external binarization often hurts

V4.9 Changes:
- Centralized find_tesseract() function (was duplicated in 3 files)
- Added preprocess_for_ocr() with grayscale, thresholding, noise removal, border
- Added invert_if_needed() to ensure dark text on light background
- Added deskew_image() for small angle correction
- Added confidence filtering utilities
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Try to import cv2 for advanced preprocessing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not available. Advanced preprocessing will be limited.")


def find_tesseract() -> Optional[str]:
    """
    Find Tesseract executable path.
    Checks PATH, common Windows locations, and Windows Registry.

    Returns:
        Path to tesseract executable or None if not found
    """
    # Check PATH first
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        return tesseract_path

    # Common Windows locations
    common_paths = [
        Path('C:/Program Files/Tesseract-OCR/tesseract.exe'),
        Path('C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'),
        Path.home() / 'AppData/Local/Programs/Tesseract-OCR/tesseract.exe',
        Path.home() / 'AppData/Local/Tesseract-OCR/tesseract.exe',
    ]

    for path in common_paths:
        if path.exists():
            return str(path)

    # Try Windows Registry
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Tesseract-OCR')
        install_path = winreg.QueryValueEx(key, 'InstallDir')[0]
        winreg.CloseKey(key)
        tesseract_exe = Path(install_path) / 'tesseract.exe'
        if tesseract_exe.exists():
            return str(tesseract_exe)
    except Exception:
        pass

    return None


def preprocess_for_ocr(
    image: Image.Image,
    apply_grayscale: bool = True,
    apply_denoise: bool = True,
    apply_threshold: bool = True,
    threshold_method: str = 'otsu',
    apply_border: bool = True,
    border_size: int = 10,
    invert_if_light_text: bool = True,
    target_dpi: Optional[int] = None,
    current_dpi: int = 200,
    preprocessing_mode: str = 'lstm',
    apply_dilation: bool = False,
    dilation_kernel_size: int = 2,
) -> Image.Image:
    """
    Preprocess an image for optimal Tesseract OCR accuracy.

    V5.0: Added preprocessing_mode to support LSTM-friendly processing.
    Research shows Tesseract 4.0+ LSTM does its own internal preprocessing,
    so external binarization often hurts accuracy (especially for thin characters).

    Preprocessing Modes:
    - 'lstm': Minimal preprocessing for Tesseract LSTM (OEM 1 or 3).
              Grayscale + light denoise only. NO binarization.
              Best for most modern use cases.
    - 'standard': Balanced preprocessing with optional binarization.
                  Respects apply_threshold parameter.
    - 'legacy': Full aggressive preprocessing for legacy Tesseract (OEM 0).
                Always applies Otsu binarization.

    Args:
        image: PIL Image to preprocess
        apply_grayscale: Convert to grayscale (default: True)
        apply_denoise: Apply Gaussian blur for noise reduction (default: True)
        apply_threshold: Apply binarization - IGNORED in 'lstm' mode (default: True)
        threshold_method: 'otsu' or 'adaptive' (default: 'otsu')
        apply_border: Add white border around image (default: True)
        border_size: Border size in pixels (default: 10)
        invert_if_light_text: Invert if image has light text on dark bg (default: True)
        target_dpi: Target DPI to upscale to (default: None, no upscaling)
        current_dpi: Current image DPI for upscaling calculation (default: 200)
        preprocessing_mode: 'lstm' (default), 'standard', or 'legacy'
        apply_dilation: Apply morphological dilation to thicken thin strokes (default: False)
        dilation_kernel_size: Kernel size for dilation (default: 2)

    Returns:
        Preprocessed PIL Image ready for OCR
    """
    if not CV2_AVAILABLE:
        # Fallback: basic preprocessing without OpenCV
        return _preprocess_basic(image, apply_grayscale, apply_border, border_size)

    # Convert PIL to numpy array
    if image.mode == 'RGBA':
        # Convert RGBA to RGB first (remove alpha channel)
        image = image.convert('RGB')

    img_array = np.array(image)

    # Step 1: Convert to grayscale
    if apply_grayscale and len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif len(img_array.shape) == 2:
        gray = img_array
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Step 2: Upscale if needed (before other processing for better quality)
    if target_dpi and current_dpi and target_dpi > current_dpi:
        scale_factor = target_dpi / current_dpi
        new_width = int(gray.shape[1] * scale_factor)
        new_height = int(gray.shape[0] * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        logger.debug(f"Upscaled image from {current_dpi} to {target_dpi} DPI (scale: {scale_factor:.2f})")

    # Step 3: Denoise with Gaussian blur (light for LSTM, standard for others)
    if apply_denoise:
        if preprocessing_mode == 'lstm':
            # Lighter denoise for LSTM - preserve more detail
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        else:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 4: Check if we need to invert (light text on dark background)
    if invert_if_light_text:
        gray = _ensure_dark_text_on_light_bg(gray)

    # Step 5: Apply thresholding/binarization based on mode
    # LSTM mode: Skip binarization - LSTM does its own internal preprocessing
    # This is the key fix for thin character cutoff (ST-01 → T-01)
    if preprocessing_mode == 'lstm':
        # LSTM mode: No binarization, just use grayscale
        result = gray
        logger.debug("LSTM mode: skipping binarization, using grayscale")
    elif preprocessing_mode == 'legacy' or (preprocessing_mode == 'standard' and apply_threshold):
        # Legacy/standard mode with binarization
        if threshold_method == 'adaptive':
            # Adaptive threshold - better for uneven lighting
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        else:
            # Otsu's method - automatic threshold selection
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = binary
    else:
        result = gray

    # Step 5b: Apply morphological dilation to preserve thin strokes
    # This helps when first characters are being cut off (S in ST-01, G in GA-01)
    # Only apply AFTER binarization (not needed for LSTM grayscale mode)
    if apply_dilation and preprocessing_mode != 'lstm':
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        # For dark text on light bg, we need erosion (which dilates the black text)
        # Actually, if text is black (0) on white (255), erosion makes black areas bigger
        result = cv2.erode(result, kernel, iterations=1)
        logger.debug(f"Applied dilation (erosion) with kernel size {dilation_kernel_size}")

    # Step 6: Add white border
    if apply_border and border_size > 0:
        border_value = 255 if len(result.shape) == 2 else (255, 255, 255)
        result = cv2.copyMakeBorder(
            result,
            border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT,
            value=border_value
        )

    # Convert back to PIL Image
    return Image.fromarray(result)


def _preprocess_basic(
    image: Image.Image,
    apply_grayscale: bool = True,
    apply_border: bool = True,
    border_size: int = 10,
) -> Image.Image:
    """
    Basic preprocessing fallback when OpenCV is not available.

    Args:
        image: PIL Image to preprocess
        apply_grayscale: Convert to grayscale
        apply_border: Add border around image
        border_size: Border size in pixels

    Returns:
        Preprocessed PIL Image
    """
    from PIL import ImageOps

    # Convert to grayscale
    if apply_grayscale and image.mode != 'L':
        image = image.convert('L')

    # Add border
    if apply_border and border_size > 0:
        image = ImageOps.expand(image, border=border_size, fill=255)

    return image


def _ensure_dark_text_on_light_bg(gray_image: np.ndarray) -> np.ndarray:
    """
    Ensure image has dark text on light background (required by Tesseract 4.x).

    Analyzes the image histogram to determine if it needs inversion.

    Args:
        gray_image: Grayscale numpy array

    Returns:
        Numpy array with correct polarity for OCR
    """
    # Calculate the mean intensity
    mean_intensity = np.mean(gray_image)

    # If the mean is dark (< 127), assume light text on dark background
    # and invert the image
    if mean_intensity < 127:
        logger.debug(f"Image appears to have light text on dark bg (mean={mean_intensity:.1f}), inverting")
        return cv2.bitwise_not(gray_image)

    return gray_image


def deskew_image(image: Image.Image, max_angle: float = 5.0) -> Tuple[Image.Image, float]:
    """
    Detect and correct small skew angles in an image.

    Tesseract accuracy drops significantly with skew > 2 degrees.
    This function detects skew using line detection and corrects it.

    Args:
        image: PIL Image to deskew
        max_angle: Maximum skew angle to correct (default: 5 degrees)

    Returns:
        Tuple of (deskewed_image, detected_angle)
    """
    if not CV2_AVAILABLE:
        return image, 0.0

    # Convert to grayscale numpy array
    if image.mode != 'L':
        gray = np.array(image.convert('L'))
    else:
        gray = np.array(image)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 10,  # 10% of width
        maxLineGap=20
    )

    if lines is None or len(lines) < 5:
        return image, 0.0

    # Calculate angles of all detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:  # Avoid division by zero
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines (within 45 degrees of horizontal)
            if abs(angle) < 45:
                angles.append(angle)

    if not angles:
        return image, 0.0

    # Use median angle to be robust to outliers
    median_angle = np.median(angles)

    # Only correct if within the max_angle threshold
    if abs(median_angle) > max_angle:
        logger.debug(f"Skew angle {median_angle:.2f}° exceeds max {max_angle}°, not correcting")
        return image, median_angle

    if abs(median_angle) < 0.1:
        # Negligible skew, don't bother rotating
        return image, 0.0

    logger.debug(f"Correcting skew angle of {median_angle:.2f}°")

    # Rotate the image to correct skew
    # PIL rotation is counterclockwise, so we negate the angle
    rotated = image.rotate(-median_angle, resample=Image.BICUBIC, expand=True, fillcolor=255)

    return rotated, median_angle


def filter_by_confidence(
    ocr_results: list,
    min_confidence: int = 60,
    confidence_key: str = 'confidence'
) -> list:
    """
    Filter OCR results by confidence threshold.

    Args:
        ocr_results: List of OCR result dictionaries
        min_confidence: Minimum confidence score (0-100) to keep
        confidence_key: Key name for confidence value in results

    Returns:
        Filtered list with only high-confidence results
    """
    filtered = []
    for result in ocr_results:
        conf = result.get(confidence_key, 0)
        # Handle string confidence values
        if isinstance(conf, str):
            try:
                conf = int(conf)
            except ValueError:
                conf = 0

        if conf >= min_confidence:
            filtered.append(result)
        else:
            logger.debug(f"Filtered out low-confidence result: '{result.get('text', '')}' (conf={conf})")

    return filtered


def upscale_for_ocr(
    image: Image.Image,
    min_height: int = 50,
    target_height: int = 100,
    max_scale: float = 4.0
) -> Image.Image:
    """
    Upscale small images for better OCR accuracy.

    Tesseract works best when text is at least 10-12 pixels tall.
    This function upscales small images to improve recognition.

    Args:
        image: PIL Image to upscale
        min_height: Minimum height threshold to trigger upscaling
        target_height: Target height after upscaling
        max_scale: Maximum scale factor to apply

    Returns:
        Upscaled PIL Image (or original if no upscaling needed)
    """
    width, height = image.size

    if height >= min_height:
        return image

    # Calculate scale factor
    scale = min(target_height / height, max_scale)

    new_width = int(width * scale)
    new_height = int(height * scale)

    logger.debug(f"Upscaling image from {width}x{height} to {new_width}x{new_height} (scale: {scale:.2f})")

    # Use LANCZOS for high-quality upscaling
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# Confidence threshold constants
DEFAULT_CONFIDENCE_THRESHOLD = 60
HIGH_CONFIDENCE_THRESHOLD = 80
LOW_CONFIDENCE_THRESHOLD = 40

# Orientation detection constants
# V6.2.2: Lowered from 2.0 to 1.0 for crops - smaller images have less text
# so OSD confidence is naturally lower. Consistency across pages is a stronger signal.
MIN_ORIENTATION_CONFIDENCE = 1.0


def detect_and_correct_crop_orientation(
    image: Image.Image,
    min_confidence: float = MIN_ORIENTATION_CONFIDENCE
) -> Tuple[Image.Image, dict]:
    """
    Detect and correct text orientation in a title block crop.

    V6.2.2: Added to fix crops where text is rotated (e.g., CrunchFitness).
    Uses Tesseract OSD to detect orientation, then rotates if needed.

    Args:
        image: PIL Image of the crop
        min_confidence: Minimum OSD confidence to apply rotation (default: 2.0)

    Returns:
        Tuple of (corrected_image, info_dict)
        info_dict contains: detected_angle, confidence, rotation_applied, method
    """
    info = {
        'detected_angle': 0,
        'confidence': 0.0,
        'rotation_applied': 0,
        'method': 'none',
        'original_size': image.size,
    }

    # Try Tesseract OSD
    try:
        import pytesseract
        osd_output = pytesseract.image_to_osd(image)

        # Parse OSD output
        angle = 0
        confidence = 0.0
        for line in osd_output.split('\n'):
            if 'Orientation in degrees:' in line:
                angle = int(line.split(':')[1].strip())
            elif 'Orientation confidence:' in line:
                confidence = float(line.split(':')[1].strip())

        info['detected_angle'] = angle
        info['confidence'] = confidence
        info['method'] = 'tesseract_osd'

        # Apply rotation if confidence is sufficient and rotation is needed
        if confidence >= min_confidence and angle != 0:
            # PIL rotate is counterclockwise, OSD angle is how much to rotate to correct
            # OSD angle 90 means rotate 90° counterclockwise to fix
            # OSD angle 270 means rotate 270° counterclockwise (or 90° clockwise) to fix
            if angle == 90:
                corrected = image.transpose(Image.ROTATE_270)
            elif angle == 180:
                corrected = image.transpose(Image.ROTATE_180)
            elif angle == 270:
                corrected = image.transpose(Image.ROTATE_90)
            else:
                corrected = image

            info['rotation_applied'] = angle
            info['corrected_size'] = corrected.size
            logger.debug(f"Crop orientation corrected: {angle}° (confidence: {confidence:.2f})")
            return corrected, info

        elif angle != 0:
            logger.debug(f"Crop rotation skipped: {angle}° (confidence: {confidence:.2f} < {min_confidence})")

    except Exception as e:
        info['method'] = 'osd_failed'
        info['error'] = str(e)
        logger.debug(f"OSD failed for crop orientation: {e}")

    info['corrected_size'] = image.size
    return image, info
