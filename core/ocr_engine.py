"""
Blueprint Processor V4.1 - OCR Engine
Handles OCR for scanned documents using Tesseract (and optionally PaddleOCR).
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logger for OCR engine
logger = logging.getLogger('ocr_engine')

from constants import TESSERACT_CONFIG

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


def find_tesseract() -> Optional[str]:
    """
    Find Tesseract executable path.
    Checks PATH, common locations, and Windows Registry.

    Returns:
        Path to tesseract executable or None
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
                  config: Optional[str] = None) -> str:
        """
        OCR an image using Tesseract.

        Args:
            image: PIL Image to OCR
            config: Tesseract config string (default: --psm 6 --oem 3)

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

            text = pytesseract.image_to_string(image, config=config)

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
                             config: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        OCR an image and return text with bounding boxes.

        Args:
            image: PIL Image to OCR
            config: Tesseract config string

        Returns:
            List of dicts with text, bbox, confidence
        """
        if not TESSERACT_AVAILABLE or self.tesseract_path is None:
            logger.warning("Tesseract not available for ocr_image_with_boxes")
            return []

        if config is None:
            config = TESSERACT_CONFIG['page']

        try:
            # Get detailed data
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)

            results = []
            n_boxes = len(data['text'])

            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text:  # Only include non-empty text
                    results.append({
                        'text': text,
                        'bbox': (
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ),
                        'confidence': data['conf'][i],
                        'level': data['level'][i],
                        'block_num': data['block_num'][i],
                        'line_num': data['line_num'][i],
                        'word_num': data['word_num'][i],
                    })

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
