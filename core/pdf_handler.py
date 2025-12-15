"""
Blueprint Processor V4.1 - PDF Handler
Handles PDF loading and text extraction (Vector-first approach).
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import fitz  # PyMuPDF
from PIL import Image
import io

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    THRESHOLDS,
    DEFAULT_DPI,
    EXTRACTION_METHODS,
)


class PDFHandler:
    """
    Handles PDF loading and text extraction.
    Implements VECTOR-FIRST principle: extract embedded text before OCR.
    """

    def __init__(self, pdf_path: Union[str, Path]):
        """
        Initialize PDFHandler with a PDF file.

        Args:
            pdf_path: Path to the PDF file (str or Path object)
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        self.doc = fitz.open(str(self.pdf_path))
        self.page_count = len(self.doc)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()

    def get_page_text(self, page_num: int) -> str:
        """
        Extract embedded text from a specific page.
        This is the VECTOR-FIRST approach - try this before OCR.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Extracted text as string
        """
        if page_num < 0 or page_num >= self.page_count:
            raise ValueError(f"Page number {page_num} out of range (0-{self.page_count - 1})")

        page = self.doc[page_num]
        return page.get_text()

    def get_text_blocks(self, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract text blocks with position information.
        Useful for spatial analysis of text placement.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            List of dicts with keys: text, bbox, block_no
            bbox is (x0, y0, x1, y1) coordinates
        """
        if page_num < 0 or page_num >= self.page_count:
            raise ValueError(f"Page number {page_num} out of range (0-{self.page_count - 1})")

        page = self.doc[page_num]
        blocks = page.get_text("blocks")

        result = []
        for i, block in enumerate(blocks):
            # block format: (x0, y0, x1, y1, "text", block_no, block_type)
            # block_type 0 = text, 1 = image
            if block[6] == 0:  # Text block
                result.append({
                    'text': block[4].strip(),
                    'bbox': (block[0], block[1], block[2], block[3]),
                    'block_no': block[5],
                })

        return result

    def analyze_page(self, page_num: int) -> Dict[str, Any]:
        """
        Analyze a page to determine if it's vector (has embedded text) or scanned.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Dict with keys: has_text, is_scanned, text_length, recommendation
        """
        if page_num < 0 or page_num >= self.page_count:
            raise ValueError(f"Page number {page_num} out of range (0-{self.page_count - 1})")

        page = self.doc[page_num]
        text = page.get_text()
        text_length = len(text.strip())

        # Check for images on the page
        image_list = page.get_images()
        has_large_image = False

        for img_info in image_list:
            xref = img_info[0]
            try:
                img_dict = self.doc.extract_image(xref)
                if img_dict:
                    # Consider "large" if image covers significant portion
                    # This is a heuristic - large scanned images are usually full-page
                    img_width = img_dict.get('width', 0)
                    img_height = img_dict.get('height', 0)
                    if img_width > 1000 and img_height > 1000:
                        has_large_image = True
                        break
            except Exception:
                pass

        min_text_threshold = THRESHOLDS['min_text_for_vector']
        is_scanned = text_length < min_text_threshold and has_large_image

        if text_length >= min_text_threshold:
            recommendation = EXTRACTION_METHODS['VECTOR_PDF']
        else:
            recommendation = EXTRACTION_METHODS['SCANNED_PDF']

        return {
            'has_text': text_length > 0,
            'is_scanned': is_scanned,
            'text_length': text_length,
            'has_large_image': has_large_image,
            'recommendation': recommendation,
        }

    def get_page_image(self, page_num: int, dpi: int = DEFAULT_DPI) -> Image.Image:
        """
        Render a page as a PIL Image.
        Used when OCR is needed for scanned documents.

        Args:
            page_num: Page number (0-indexed)
            dpi: Resolution for rendering (default 200)

        Returns:
            PIL Image object
        """
        if page_num < 0 or page_num >= self.page_count:
            raise ValueError(f"Page number {page_num} out of range (0-{self.page_count - 1})")

        page = self.doc[page_num]

        # Calculate zoom factor from DPI (72 is PDF default DPI)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pixmap = page.get_pixmap(matrix=matrix)

        # Convert to PIL Image
        img_data = pixmap.tobytes("png")
        image = Image.open(io.BytesIO(img_data))

        return image

    def get_page_dimensions(self, page_num: int) -> Dict[str, float]:
        """
        Get the dimensions of a page in points.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Dict with keys: width, height (in points)
        """
        if page_num < 0 or page_num >= self.page_count:
            raise ValueError(f"Page number {page_num} out of range (0-{self.page_count - 1})")

        page = self.doc[page_num]
        rect = page.rect

        return {
            'width': rect.width,
            'height': rect.height,
        }
