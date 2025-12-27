"""
Title Block Detection - Simple API

This module provides a simple interface for detecting title blocks in blueprints.
It uses the multi-stage pipeline:
1. CV Transition Detection (finds where consistent edge structure begins)
2. AI Vision Refinement (optional, for fine-tuning boundary)

Usage:
    from core.detect_title_block import detect_and_crop_title_block

    with PDFHandler(pdf_path) as handler:
        # Get sample pages
        pages = [handler.get_page_image(i, dpi=100) for i in [1, 3, 5]]

        # Detect and crop title block
        result = detect_and_crop_title_block(pages)

        # Result contains:
        # - x1: Left boundary (0.0-1.0)
        # - width_pct: Title block width as fraction
        # - confidence: Detection confidence
        # - method: Detection method used
"""

from PIL import Image
from typing import List, Dict, Optional, Tuple


def detect_title_block(
    page_images: List[Image.Image],
    use_ai: bool = True,
    strategy: str = 'balanced'
) -> Dict:
    """
    Detect title block boundary in blueprint pages.

    Args:
        page_images: List of PIL Images (sample pages from PDF, typically 3-5)
        use_ai: Whether to use AI Vision refinement (slower but sometimes more accurate)
        strategy: Detection strategy
            - 'balanced': Use CV_transition, refined by AI if needed
            - 'conservative': Use tighter boundary (won't cut drawings)
            - 'coarse_only': Skip AI refinement (fastest)

    Returns:
        Dict with:
            - x1: Left boundary as fraction (0.0 = left edge, 1.0 = right edge)
            - width_pct: Title block width as fraction of page
            - confidence: Detection confidence (0.0-1.0)
            - method: Detection method used
    """
    from .title_block_detector import TitleBlockDetector

    detector = TitleBlockDetector(use_ai_refinement=use_ai)
    return detector.detect(page_images, strategy=strategy)


def crop_title_block(
    page_image: Image.Image,
    x1: float
) -> Image.Image:
    """
    Crop the title block from a page image.

    Args:
        page_image: Full page PIL Image
        x1: Left boundary as fraction (0.0-1.0)

    Returns:
        Cropped title block image
    """
    width, height = page_image.size
    x1_px = int(x1 * width)
    return page_image.crop((x1_px, 0, width, height))


def detect_and_crop_title_block(
    page_images: List[Image.Image],
    page_to_crop: Optional[Image.Image] = None,
    use_ai: bool = True,
    strategy: str = 'balanced'
) -> Dict:
    """
    Detect title block and return both detection result and cropped image.

    Args:
        page_images: List of sample page images for detection
        page_to_crop: Specific page to crop (defaults to first in page_images)
        use_ai: Whether to use AI refinement
        strategy: Detection strategy

    Returns:
        Dict with detection result plus 'crop' key containing the cropped image
    """
    result = detect_title_block(page_images, use_ai=use_ai, strategy=strategy)

    target_page = page_to_crop if page_to_crop is not None else page_images[0]
    result['crop'] = crop_title_block(target_page, result['x1'])

    return result


# Convenience aliases
def quick_detect(page_images: List[Image.Image]) -> float:
    """Quick detection without AI - returns just x1."""
    result = detect_title_block(page_images, use_ai=False, strategy='coarse_only')
    return result['x1']


def accurate_detect(page_images: List[Image.Image]) -> float:
    """Accurate detection with AI - returns just x1."""
    result = detect_title_block(page_images, use_ai=True, strategy='balanced')
    return result['x1']
