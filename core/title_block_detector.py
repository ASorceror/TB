"""
Multi-Stage Title Block Detector

Combines multiple detection methods into a robust pipeline:
1. Coarse Detection: CV + Hough line consensus
2. Boundary Refinement: Constrained AI Vision
3. Structure Learning: Cache template for future pages

This pipeline prevents wild errors while achieving high accuracy.
"""

import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import json


class TitleBlockDetector:
    """
    Main detector class for finding title blocks in blueprint PDFs.

    Usage:
        detector = TitleBlockDetector()
        result = detector.detect(page_images)
        # result['x1'] is the left boundary as fraction (0.0-1.0)
    """

    def __init__(
        self,
        use_ai_refinement: bool = True,
        ai_margin: float = 0.10,
        search_region: Tuple[float, float] = (0.60, 0.98),
        default_x1: float = 0.85
    ):
        """
        Initialize the detector.

        Args:
            use_ai_refinement: Whether to use AI Vision for refinement
            ai_margin: Margin for AI Vision crop (fraction of page width)
            search_region: Region to search for title block
            default_x1: Default fallback value
        """
        self.use_ai_refinement = use_ai_refinement
        self.ai_margin = ai_margin
        self.search_region = search_region
        self.default_x1 = default_x1

        # Lazy load components
        self._coarse_detector = None
        self._refiner = None
        self._learned_template = None

    @property
    def coarse_detector(self):
        """Lazy-load coarse detector."""
        if self._coarse_detector is None:
            from .coarse_detection import CoarseDetector
            self._coarse_detector = CoarseDetector(
                search_region=self.search_region,
                default_x1=self.default_x1
            )
        return self._coarse_detector

    @property
    def refiner(self):
        """Lazy-load boundary refiner."""
        if self._refiner is None:
            from .boundary_refiner import BoundaryRefiner
            self._refiner = BoundaryRefiner(margin=self.ai_margin)
        return self._refiner

    def detect(
        self,
        page_images: List[Image.Image],
        strategy: str = 'balanced'
    ) -> Dict:
        """
        Detect title block using multi-stage pipeline.

        Args:
            page_images: List of PIL Images (sample pages from PDF)
            strategy: Detection strategy
                - 'balanced': Use median consensus + AI refinement
                - 'conservative': Use maximum (tightest boundary, won't cut drawings)
                - 'aggressive': Use minimum (widest boundary, may include some drawing)
                - 'include_all': Use CV_transition as base (captures full structure)
                - 'coarse_only': Skip AI refinement

        Returns:
            Dict with detection results:
                - x1: Left boundary as fraction (0.0-1.0)
                - width_pct: Title block width as fraction
                - method: Detection method used
                - confidence: Confidence score (0.0-1.0)
                - stages: Details from each stage
        """
        if not page_images:
            return self._default_result("no_images")

        stages = {}

        # Stage 1: Coarse Detection
        if strategy in ('balanced', 'coarse_only', 'include_all'):
            coarse_strategy = 'median'
        else:
            coarse_strategy = strategy

        coarse_result = self.coarse_detector.detect(page_images, strategy=coarse_strategy)
        stages['coarse'] = coarse_result

        # V6.2.3: Use consensus x1, not cv_transition override
        # The consensus detector already handles method disagreement properly
        # (prefers cv_majority when spread is high)
        coarse_x1 = coarse_result['x1']
        stages['consensus_method'] = coarse_result.get('method', 'unknown')

        # Decide if we need AI refinement
        need_refinement = (
            self.use_ai_refinement and
            strategy != 'coarse_only' and
            coarse_result.get('confidence', 0) < 0.95  # Already high confidence
        )

        if need_refinement:
            # Stage 2: AI Vision Refinement
            # Use the page with highest DPI/quality for refinement
            refine_image = page_images[0]  # Could select best quality

            try:
                refined_result = self.refiner.refine(refine_image, coarse_x1)
                stages['refined'] = refined_result

                if 'error' not in refined_result:
                    refined_x1 = refined_result['x1']

                    # Decide which to use based on strategy
                    if strategy == 'balanced':
                        # Use the smaller of coarse (CV_transition) and refined
                        # This ensures we don't miss title block content
                        final_x1 = min(coarse_x1, refined_x1)
                        method = 'balanced'
                    elif strategy == 'conservative':
                        # Use the larger x1 (tighter boundary)
                        final_x1 = max(coarse_x1, refined_x1)
                        method = 'conservative'
                    elif strategy == 'aggressive':
                        # Use the smaller x1 (wider boundary)
                        final_x1 = min(coarse_x1, refined_x1)
                        method = 'aggressive'
                    elif strategy == 'include_all':
                        # Use CV_transition (coarse_x1 already set to this)
                        # as it captures full structure
                        final_x1 = coarse_x1
                        method = 'include_all'
                    else:
                        final_x1 = refined_x1
                        method = 'refined'
                else:
                    final_x1 = coarse_x1
                    method = 'coarse_fallback'
            except Exception as e:
                stages['refined_error'] = str(e)
                final_x1 = coarse_x1
                method = 'coarse_fallback'
        else:
            # No AI refinement - use consensus result
            final_x1 = coarse_x1
            method = stages.get('consensus_method', 'coarse')

        # Calculate confidence
        confidence = self._calculate_confidence(stages, method)

        return {
            'x1': final_x1,
            'width_pct': 1.0 - final_x1,
            'method': method,
            'confidence': confidence,
            'stages': stages
        }

    def detect_with_learning(
        self,
        page_images: List[Image.Image],
        strategy: str = 'balanced'
    ) -> Dict:
        """
        Detect and learn the title block template for this PDF.

        The learned template can be applied to remaining pages without
        re-running detection.
        """
        result = self.detect(page_images, strategy=strategy)

        # Learn the template
        self._learned_template = {
            'x1': result['x1'],
            'method': result['method'],
            'confidence': result['confidence']
        }

        return result

    def apply_learned(
        self,
        page_image: Image.Image
    ) -> Dict:
        """
        Apply the learned template to a new page.

        This is fast since no detection is needed.
        """
        if self._learned_template is None:
            # No template learned, run full detection
            return self.detect([page_image])

        return {
            'x1': self._learned_template['x1'],
            'width_pct': 1.0 - self._learned_template['x1'],
            'method': 'learned',
            'confidence': self._learned_template['confidence']
        }

    def crop_title_block(
        self,
        page_image: Image.Image,
        detection_result: Optional[Dict] = None
    ) -> Image.Image:
        """
        Crop the title block from a page image.

        Args:
            page_image: Full page PIL Image
            detection_result: Optional pre-computed detection result

        Returns:
            Cropped title block image
        """
        if detection_result is None:
            detection_result = self.detect([page_image])

        x1 = detection_result['x1']
        width, height = page_image.size

        x1_px = int(x1 * width)
        return page_image.crop((x1_px, 0, width, height))

    def _default_result(self, reason: str) -> Dict:
        """Return a default result when detection fails."""
        return {
            'x1': self.default_x1,
            'width_pct': 1.0 - self.default_x1,
            'method': f'default_{reason}',
            'confidence': 0.0,
            'stages': {}
        }

    def _calculate_confidence(self, stages: Dict, method: str) -> float:
        """Calculate overall confidence from stage results."""
        if method == 'refined':
            refined = stages.get('refined', {})
            return refined.get('confidence', 0.8)
        elif method == 'coarse':
            coarse = stages.get('coarse', {})
            return coarse.get('confidence', 0.7)
        elif method in ('conservative', 'aggressive'):
            # Average of coarse and refined confidence
            coarse_conf = stages.get('coarse', {}).get('confidence', 0.5)
            refined_conf = stages.get('refined', {}).get('confidence', 0.5)
            return (coarse_conf + refined_conf) / 2
        else:
            return 0.5


def detect_title_block(
    page_images: List[Image.Image],
    use_ai: bool = True
) -> float:
    """
    Convenience function to detect title block left boundary.

    Args:
        page_images: List of page images
        use_ai: Whether to use AI refinement

    Returns:
        x1 as fraction (0.0-1.0)
    """
    detector = TitleBlockDetector(use_ai_refinement=use_ai)
    result = detector.detect(page_images)
    return result['x1']
