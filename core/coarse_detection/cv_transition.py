"""
CV Transition Detection for Title Block Borders

Uses edge detection across multiple pages to find the transition
from low edge density (drawing area) to high edge density (title block).

This is the most reliable pure CV approach from research phase.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from scipy.ndimage import uniform_filter1d


def get_common_edges(
    page_images: List[Image.Image],
    edge_threshold: int = 25
) -> np.ndarray:
    """
    Find edges that appear on ALL sample pages (common structure).

    Args:
        page_images: List of PIL Images
        edge_threshold: Threshold for edge detection

    Returns:
        2D numpy array of common edges (255 where edge on all pages)
    """
    edge_images = []
    target_size = None

    for img in page_images:
        if target_size is None:
            target_size = img.size
        elif img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_array = np.array(edges)
        binary = (edges_array > edge_threshold).astype(np.uint8) * 255
        edge_images.append(binary)

    # Find edges present on ALL pages
    stack = np.stack(edge_images, axis=0)
    common_edges = np.all(stack > 128, axis=0).astype(np.uint8) * 255

    return common_edges


def get_majority_edges(
    page_images: List[Image.Image],
    edge_threshold: int = 25,
    agreement_ratio: float = 0.6
) -> np.ndarray:
    """
    Find edges that appear on MOST sample pages (majority vote).

    Args:
        page_images: List of PIL Images
        edge_threshold: Threshold for edge detection
        agreement_ratio: Fraction of pages that must agree

    Returns:
        2D numpy array of majority edges
    """
    edge_images = []
    target_size = None

    for img in page_images:
        if target_size is None:
            target_size = img.size
        elif img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_array = np.array(edges)
        binary = (edges_array > edge_threshold).astype(np.uint8)
        edge_images.append(binary)

    # Count agreement
    stack = np.stack(edge_images, axis=0)
    agreement = np.sum(stack, axis=0)

    min_pages = int(np.ceil(len(page_images) * agreement_ratio))
    majority_edges = (agreement >= min_pages).astype(np.uint8) * 255

    return majority_edges


def find_transition_x1(
    common_edges: np.ndarray,
    search_start: float = 0.60,
    search_end: float = 0.97,
    window_size: int = 20,
    smoothing_size: int = 10
) -> Optional[float]:
    """
    Find the x position where edge density transitions from low to high.

    This finds the LEFT EDGE of the title block by detecting where
    consistent structure begins.

    Args:
        common_edges: 2D array of common edges
        search_start: Start of search region (fraction of width)
        search_end: End of search region (fraction of width)
        window_size: Window for transition detection
        smoothing_size: Size for column density smoothing

    Returns:
        x1 as fraction (0.0-1.0) or None if not found
    """
    height, width = common_edges.shape

    # Calculate column edge counts
    col_edge_count = np.sum(common_edges > 128, axis=0)

    # Smooth
    col_smoothed = uniform_filter1d(col_edge_count.astype(float), size=smoothing_size)

    # Determine threshold from the left portion (should be noise/drawing)
    left_portion = col_smoothed[:int(width * 0.5)]
    noise_level = np.percentile(left_portion, 95)
    threshold = max(noise_level * 2, 20)

    # Find transition point
    search_start_px = int(width * search_start)
    search_end_px = int(width * search_end)

    x1_px = None
    for i in range(search_start_px, search_end_px - window_size):
        window = col_smoothed[i:i + window_size]
        if np.all(window > threshold):
            x1_px = i
            break

    if x1_px is not None:
        return x1_px / width
    return None


def detect_title_block_cv(
    page_images: List[Image.Image],
    use_majority: bool = False,
    agreement_ratio: float = 0.6
) -> Optional[Dict]:
    """
    Detect title block left boundary using CV transition detection.

    Args:
        page_images: List of PIL Images (sample pages)
        use_majority: If True, use majority vote; otherwise use strict AND
        agreement_ratio: For majority vote, fraction of pages that must agree

    Returns:
        Dict with 'x1', 'method', 'width_pct' or None if not found
    """
    if not page_images:
        return None

    if use_majority:
        edges = get_majority_edges(page_images, agreement_ratio=agreement_ratio)
        method = 'cv_majority'
    else:
        edges = get_common_edges(page_images)
        method = 'cv_transition'

    x1 = find_transition_x1(edges)

    if x1 is not None:
        return {
            'x1': x1,
            'method': method,
            'width_pct': 1.0 - x1
        }
    return None


def visualize_edges(
    common_edges: np.ndarray,
    page_image: Image.Image,
    x1: Optional[float] = None,
    output_path: Optional[Path] = None
) -> Image.Image:
    """
    Create visualization showing common edges and detected boundary.
    """
    height, width = common_edges.shape

    # Create side-by-side image
    result = Image.new('RGB', (width * 2, height))

    # Left: original page
    result.paste(page_image.convert('RGB'), (0, 0))

    # Right: common edges
    edges_img = Image.fromarray(common_edges).convert('RGB')
    result.paste(edges_img, (width, 0))

    # Draw detected boundary on both
    if x1 is not None:
        draw = ImageDraw.Draw(result)
        x_px = int(x1 * width)

        # Green line on original
        draw.line([(x_px, 0), (x_px, height)], fill='green', width=3)

        # Green line on edges
        draw.line([(width + x_px, 0), (width + x_px, height)], fill='green', width=3)

        # Labels
        draw.text((x_px + 5, 20), f"CV: {x1:.3f}", fill='green')

    if output_path:
        result.save(output_path)

    return result
