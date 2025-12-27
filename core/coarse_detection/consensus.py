"""
Consensus Voting System for Coarse Title Block Detection

Combines multiple detection methods to get a robust estimate of the
title block left boundary. Uses median voting to be robust to outliers.
"""

import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from .cv_transition import detect_title_block_cv
from .line_detector import detect_title_block_border


class CoarseDetector:
    """
    Combines multiple coarse detection methods using consensus voting.

    Methods included:
    1. CV Transition Detection (strict AND across pages)
    2. CV Majority Detection (60% agreement)
    3. Hough Line Detection (vertical lines)

    The consensus uses median to be robust to single-method failures.
    """

    def __init__(
        self,
        search_region: Tuple[float, float] = (0.60, 0.98),
        default_x1: float = 0.85
    ):
        """
        Initialize the coarse detector.

        Args:
            search_region: Region to search for title block (start, end)
            default_x1: Fallback value if all methods fail
        """
        self.search_region = search_region
        self.default_x1 = default_x1

    def detect(
        self,
        page_images: List[Image.Image],
        strategy: str = 'median'
    ) -> Dict:
        """
        Run all detection methods and return consensus result.

        Args:
            page_images: List of PIL Images (sample pages)
            strategy: 'median' (robust), 'conservative' (max), or 'aggressive' (min)

        Returns:
            Dict with 'x1', 'method', 'estimates', 'spread'
        """
        estimates = {}
        estimates_list = []

        # Method 1: CV Transition (strict AND)
        try:
            cv_result = detect_title_block_cv(page_images, use_majority=False)
            if cv_result and cv_result['x1']:
                estimates['cv_transition'] = cv_result['x1']
                estimates_list.append(cv_result['x1'])
        except Exception as e:
            estimates['cv_transition_error'] = str(e)

        # Method 2: CV Majority (60% agreement)
        try:
            cv_majority = detect_title_block_cv(page_images, use_majority=True, agreement_ratio=0.6)
            if cv_majority and cv_majority['x1']:
                estimates['cv_majority'] = cv_majority['x1']
                estimates_list.append(cv_majority['x1'])
        except Exception as e:
            estimates['cv_majority_error'] = str(e)

        # Method 3: Hough Line Detection
        try:
            hough_x1 = detect_title_block_border(
                page_images,
                search_region=self.search_region,
                min_agreement=2
            )
            if hough_x1:
                estimates['hough_lines'] = hough_x1
                estimates_list.append(hough_x1)
        except Exception as e:
            estimates['hough_lines_error'] = str(e)

        # Calculate consensus
        if not estimates_list:
            # All methods failed - use default
            return {
                'x1': self.default_x1,
                'method': 'default_fallback',
                'estimates': estimates,
                'spread': 0.0,
                'confidence': 0.0
            }

        if len(estimates_list) == 1:
            # Only one method succeeded
            x1 = estimates_list[0]
            method_name = [k for k, v in estimates.items() if v == x1 and not k.endswith('_error')][0]
            return {
                'x1': x1,
                'method': method_name,
                'estimates': estimates,
                'spread': 0.0,
                'confidence': 0.5
            }

        # Multiple methods - filter outliers first
        estimates_array = np.array(estimates_list)

        # Remove outliers: values that are more than 0.10 from the median
        initial_median = np.median(estimates_array)
        filtered = [e for e in estimates_list if abs(e - initial_median) < 0.10]

        if not filtered:
            # All were outliers - use the one closest to expected range (0.80-0.95)
            filtered = estimates_list

        estimates_array = np.array(filtered)

        if strategy == 'median':
            x1 = float(np.median(estimates_array))
            method = 'consensus_median'
        elif strategy == 'conservative':
            # Maximum = smallest title block = won't cut into drawings
            x1 = float(np.max(estimates_array))
            method = 'consensus_conservative'
        elif strategy == 'aggressive':
            # Minimum = largest title block = might include some drawing
            x1 = float(np.min(estimates_array))
            method = 'consensus_aggressive'
        else:
            x1 = float(np.median(estimates_array))
            method = 'consensus_median'

        spread = float(np.max(estimates_array) - np.min(estimates_array))

        # Confidence based on agreement
        if spread < 0.02:
            confidence = 0.95
        elif spread < 0.05:
            confidence = 0.80
        elif spread < 0.10:
            confidence = 0.60
        else:
            confidence = 0.40

        return {
            'x1': x1,
            'method': method,
            'estimates': estimates,
            'spread': spread,
            'confidence': confidence,
            'num_methods': len(estimates_list)
        }

    def detect_with_visualization(
        self,
        page_images: List[Image.Image],
        output_path: Optional[Path] = None,
        strategy: str = 'median'
    ) -> Tuple[Dict, Image.Image]:
        """
        Run detection and create visualization.

        Returns:
            Tuple of (result dict, visualization image)
        """
        result = self.detect(page_images, strategy=strategy)

        # Create visualization
        if page_images:
            viz_img = page_images[0].convert('RGB')
            width, height = viz_img.size

            from PIL import ImageDraw
            draw = ImageDraw.Draw(viz_img)

            # Draw each method's estimate with different colors
            colors = {
                'cv_transition': 'blue',
                'cv_majority': 'cyan',
                'hough_lines': 'orange'
            }

            y_offset = 20
            for method_name, color in colors.items():
                if method_name in result['estimates']:
                    x1 = result['estimates'][method_name]
                    x_px = int(x1 * width)

                    # Dashed line for individual estimates
                    for y in range(0, height, 20):
                        draw.line([(x_px, y), (x_px, min(y + 10, height))], fill=color, width=2)

                    draw.text((x_px + 5, y_offset), f"{method_name}: {x1:.3f}", fill=color)
                    y_offset += 25

            # Solid green line for consensus
            consensus_x = int(result['x1'] * width)
            draw.line([(consensus_x, 0), (consensus_x, height)], fill='green', width=4)
            draw.text((consensus_x + 5, y_offset + 10),
                     f"CONSENSUS: {result['x1']:.3f} (spread={result['spread']:.3f})",
                     fill='green')

            if output_path:
                viz_img.save(output_path)

            return result, viz_img

        return result, None


def get_consensus_x1(
    page_images: List[Image.Image],
    strategy: str = 'median'
) -> float:
    """
    Convenience function to get just the x1 value.

    Args:
        page_images: List of PIL Images
        strategy: 'median', 'conservative', or 'aggressive'

    Returns:
        x1 as fraction (0.0-1.0)
    """
    detector = CoarseDetector()
    result = detector.detect(page_images, strategy=strategy)
    return result['x1']
