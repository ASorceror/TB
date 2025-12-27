"""
Blueprint Processor V5.0 - OCR Engine
Handles OCR for scanned documents using Tesseract (and optionally PaddleOCR).

V5.0 Changes:
- Uses LSTM-friendly preprocessing (no binarization) by default
- Research: Tesseract LSTM does internal preprocessing, external binarization hurts accuracy

V4.9 Changes:
- Centralized find_tesseract() moved to ocr_utils.py
- Added image preprocessing before OCR for improved accuracy
- Added confidence filtering to ocr_image_with_boxes()
- Added preprocess parameter to control preprocessing
- Converts image mode to ensure compatibility
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logger for OCR engine
logger = logging.getLogger('ocr_engine')

from constants import TESSERACT_CONFIG, OCR_CONFIDENCE_THRESHOLD
from core.ocr_utils import (
    find_tesseract,
    preprocess_for_ocr,
    filter_by_confidence,
    DEFAULT_CONFIDENCE_THRESHOLD,
)

# Try to import pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Try to import PaddleOCR (optional)
PADDLE_AVAILABLE = False
PADDLE_INSTALL_ATTEMPTS = 0
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    pass


class OCREngine:
    """
    OCR engine for extracting text from images.
    Uses Tesseract as primary engine, PaddleOCR as optional secondary.
    """

    def __init__(self, telemetry_dir: Optional[Path] = None):
        """
        Initialize OCR Engine.

        Args:
            telemetry_dir: Directory to save telemetry JSON files
        """
        self.telemetry_dir = telemetry_dir
        self.tesseract_path = None
        self.paddle_ocr = None

        # Configure Tesseract
        if TESSERACT_AVAILABLE:
            self.tesseract_path = find_tesseract()
            if self.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        # Initialize PaddleOCR if available
        if PADDLE_AVAILABLE:
            try:
                # Initialize with English, use_gpu=False for CPU
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False,
                                            show_log=False)
            except Exception:
                self.paddle_ocr = None

    def is_available(self) -> Dict[str, bool]:
        """
        Check which OCR engines are available.

        Returns:
            Dict with engine availability status
        """
        return {
            'tesseract': TESSERACT_AVAILABLE and self.tesseract_path is not None,
            'tesseract_path': self.tesseract_path,
            'paddleocr': self.paddle_ocr is not None,
        }

    def ocr_image(self, image: Image.Image,
                  config: Optional[str] = None,
                  preprocess: bool = True) -> str:
        """
        OCR an image using Tesseract with optional preprocessing.

        V4.9: Added preprocessing pipeline for improved accuracy:
        - Grayscale conversion
        - Noise reduction
        - Binarization (Otsu thresholding)
        - Border addition
        - Dark text on light background enforcement

        Args:
            image: PIL Image to OCR
            config: Tesseract config string (default: --psm 6 --oem 3)
            preprocess: Whether to apply preprocessing (default: True)

        Returns:
            Extracted text
        """
        if not TESSERACT_AVAILABLE:
            logger.warning("pytesseract module not available")
            return ''

        if self.tesseract_path is None:
            logger.warning("Tesseract executable not found")
            return ''

        if config is None:
            config = TESSERACT_CONFIG['page']

        try:
            logger.debug(f"Running Tesseract OCR with config: {config}")
            logger.debug(f"Tesseract path: {self.tesseract_path}")
            logger.debug(f"Image size: {image.size}, mode: {image.mode}")

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
                logger.debug(f"Preprocessed image size: {processed_image.size}, mode: {processed_image.mode}")
            else:
                # At minimum, ensure compatible image mode
                if image.mode == 'RGBA':
                    processed_image = image.convert('RGB')
                else:
                    processed_image = image

            text = pytesseract.image_to_string(processed_image, config=config)

            logger.debug(f"OCR successful, extracted {len(text)} characters")
            return text

        except pytesseract.TesseractError as e:
            logger.error(f"Tesseract error: {e}")
            logger.error(f"Tesseract path: {self.tesseract_path}")
            logger.error(traceback.format_exc())
            return ''

        except PermissionError as e:
            logger.error(f"Permission error during OCR: {e}")
            logger.error("This may be caused by antivirus blocking temp file execution")
            logger.error(f"Tesseract path: {self.tesseract_path}")
            logger.error(traceback.format_exc())
            return ''

        except OSError as e:
            logger.error(f"OS error during OCR: {e}")
            logger.error("Check if Tesseract is installed correctly and temp directory is accessible")
            logger.error(f"Tesseract path: {self.tesseract_path}")
            logger.error(traceback.format_exc())
            return ''

        except Exception as e:
            logger.error(f"Unexpected error during OCR: {type(e).__name__}: {e}")
            logger.error(f"Tesseract path: {self.tesseract_path}")
            logger.error(traceback.format_exc())
            return ''

    def ocr_image_with_boxes(self, image: Image.Image,
                             config: Optional[str] = None,
                             preprocess: bool = True,
                             min_confidence: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        OCR an image and return text with bounding boxes.

        V4.9: Added preprocessing and confidence filtering.

        Args:
            image: PIL Image to OCR
            config: Tesseract config string
            preprocess: Whether to apply preprocessing (default: True)
            min_confidence: Minimum confidence threshold (0-100).
                           Default: OCR_CONFIDENCE_THRESHOLD from constants.
                           Set to 0 or None to disable filtering.

        Returns:
            List of dicts with text, bbox, confidence (filtered by confidence)
        """
        if not TESSERACT_AVAILABLE or self.tesseract_path is None:
            logger.warning("Tesseract not available for ocr_image_with_boxes")
            return []

        if config is None:
            config = TESSERACT_CONFIG['page']

        # Set default confidence threshold
        if min_confidence is None:
            min_confidence = OCR_CONFIDENCE_THRESHOLD

        try:
            # V5.0: Use LSTM-friendly preprocessing (no binarization)
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
                if image.mode == 'RGBA':
                    processed_image = image.convert('RGB')
                else:
                    processed_image = image

            # Get detailed data
            data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)

            results = []
            n_boxes = len(data['text'])
            filtered_count = 0

            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = data['conf'][i]

                # Convert confidence to int if it's a string
                if isinstance(conf, str):
                    try:
                        conf = int(conf)
                    except ValueError:
                        conf = 0

                if text:  # Only include non-empty text
                    # V4.9: Apply confidence filtering
                    if min_confidence > 0 and conf < min_confidence:
                        filtered_count += 1
                        logger.debug(f"Filtered low-confidence text: '{text}' (conf={conf})")
                        continue

                    results.append({
                        'text': text,
                        'bbox': (
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ),
                        'confidence': conf,
                        'level': data['level'][i],
                        'block_num': data['block_num'][i],
                        'line_num': data['line_num'][i],
                        'word_num': data['word_num'][i],
                    })

            if filtered_count > 0:
                logger.debug(f"Filtered {filtered_count} low-confidence results (threshold={min_confidence})")

            return results
        except Exception as e:
            logger.error(f"Error in ocr_image_with_boxes: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            return []

    def ocr_with_paddle(self, image: Image.Image) -> str:
        """
        OCR an image using PaddleOCR (if available).

        Args:
            image: PIL Image to OCR

        Returns:
            Extracted text or empty string if not available
        """
        if self.paddle_ocr is None:
            logger.debug("PaddleOCR not available")
            return ''

        try:
            import numpy as np
            # Convert PIL to numpy array (RGB)
            img_array = np.array(image)

            # Run PaddleOCR
            result = self.paddle_ocr.ocr(img_array, cls=True)

            if result is None or len(result) == 0:
                return ''

            # Extract text from result
            # PaddleOCR returns: [[[box, (text, confidence)], ...]]
            lines = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]  # (text, confidence) tuple
                    lines.append(text)

            return '\n'.join(lines)
        except Exception as e:
            logger.error(f"Error in PaddleOCR: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            return ''

    def ensemble_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run OCR with all available engines and merge results.

        Args:
            image: PIL Image to OCR

        Returns:
            Dict with results from each engine and merged result
        """
        results = {
            'tesseract': '',
            'paddle': '',
            'merged': '',
            'engines_used': [],
        }

        # Tesseract OCR
        if TESSERACT_AVAILABLE and self.tesseract_path:
            results['tesseract'] = self.ocr_image(image)
            results['engines_used'].append('tesseract')

        # PaddleOCR
        if self.paddle_ocr is not None:
            results['paddle'] = self.ocr_with_paddle(image)
            results['engines_used'].append('paddle')

        # Merge strategy: prefer Tesseract, use Paddle if Tesseract fails
        if results['tesseract'].strip():
            results['merged'] = results['tesseract']
            results['primary_engine'] = 'tesseract'
        elif results['paddle'].strip():
            results['merged'] = results['paddle']
            results['primary_engine'] = 'paddle'
        else:
            results['merged'] = ''
            results['primary_engine'] = 'none'

        return results

    def save_telemetry(self, filename: str, ocr_result: Dict[str, Any]):
        """
        Save telemetry JSON file with OCR details.

        Args:
            filename: Base filename for telemetry
            ocr_result: OCR results
        """
        if self.telemetry_dir is None:
            return

        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.telemetry_dir / f"{filename}_ocr.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Make result serializable
            serializable = {k: v for k, v in ocr_result.items()
                          if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            json.dump(serializable, f, indent=2)
