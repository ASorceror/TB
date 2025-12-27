"""
Boundary Refinement using Constrained AI Vision

Key insight: AI Vision fails when given the whole page because it sometimes
finds drawing content. By only showing AI a CROPPED region around the coarse
estimate, it can only adjust within that constrained area.

The AI sees the region from (coarse_x1 - margin) to (page_right), and must
find the exact title block left boundary within that crop.
"""

import os
import io
import re
import base64
from PIL import Image
from typing import Optional, Dict, List, Tuple
from pathlib import Path


class BoundaryRefiner:
    """
    Refines coarse title block boundary using AI Vision on a constrained crop.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        margin: float = 0.10
    ):
        """
        Initialize the boundary refiner.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use for vision
            margin: Margin to add left of coarse estimate for cropping
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.model = model
        self.margin = margin
        self._client = None

    @property
    def client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("No API key available for AI Vision")
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def refine(
        self,
        page_image: Image.Image,
        coarse_x1: float,
        debug: bool = False
    ) -> Dict:
        """
        Refine the boundary using AI Vision on a constrained crop.

        Args:
            page_image: Full page PIL Image
            coarse_x1: Coarse estimate of title block left boundary (0.0-1.0)
            debug: If True, return additional debug info

        Returns:
            Dict with 'x1' (refined), 'local_x1', 'confidence', 'description'
        """
        width, height = page_image.size

        # Create constrained crop
        crop_x1 = int(max(0, (coarse_x1 - self.margin)) * width)
        crop_x2 = width  # Always go to right edge

        cropped = page_image.crop((crop_x1, 0, crop_x2, height))

        # Resize if too large for API
        max_size = 1500
        crop_w, crop_h = cropped.size
        if max(crop_w, crop_h) > max_size:
            scale = max_size / max(crop_w, crop_h)
            cropped = cropped.resize(
                (int(crop_w * scale), int(crop_h * scale)),
                Image.Resampling.LANCZOS
            )

        # Convert to base64
        img_bytes = io.BytesIO()
        cropped.save(img_bytes, format='PNG')
        img_data = base64.standard_b64encode(img_bytes.getvalue()).decode('utf-8')

        # Constrained prompt
        prompt = """This image shows the RIGHT PORTION of a blueprint page.
The title block LEFT BOUNDARY is somewhere in this image.

The title block is the structured information panel containing:
- Company/firm name and logo
- Project name and address
- Sheet number (A-1, S-1, etc.)
- Date, scale, drawn by info

I need you to find the EXACT LEFT EDGE of the title block within this cropped image.

Look for:
1. A visible vertical border line separating drawing from title block
2. OR the natural boundary where structured boxes/text begin
3. The boundary should have whitespace - NO text should be cut off

Return ONLY this JSON:
{
  "left_boundary_x": 0.XX,
  "confidence": 0.XX,
  "has_visible_border_line": true/false,
  "description": "brief description"
}

Where left_boundary_x is the position within THIS CROP (0.0 = left edge of crop, 1.0 = right edge).
"""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_data}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )

            response_text = message.content[0].text.strip()

            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                import json
                result = json.loads(json_match.group())
                local_x1 = result.get('left_boundary_x', 0.5)

                # Convert local coordinate back to full page coordinate
                # local_x1 is within the crop, need to convert to full page
                crop_width_ratio = (crop_x2 - crop_x1) / width
                crop_start_ratio = crop_x1 / width

                global_x1 = crop_start_ratio + (local_x1 * crop_width_ratio)

                return {
                    'x1': global_x1,
                    'local_x1': local_x1,
                    'confidence': result.get('confidence', 0.5),
                    'has_border_line': result.get('has_visible_border_line', False),
                    'description': result.get('description', ''),
                    'method': 'ai_vision_constrained',
                    'crop_region': (crop_x1 / width, crop_x2 / width)
                }

        except Exception as e:
            return {
                'x1': coarse_x1,  # Fall back to coarse estimate
                'error': str(e),
                'method': 'ai_vision_failed'
            }

        return {
            'x1': coarse_x1,
            'error': 'Could not parse AI response',
            'method': 'ai_vision_failed'
        }

    def refine_with_multi_page(
        self,
        page_images: List[Image.Image],
        coarse_x1: float,
        num_pages: int = 2
    ) -> Dict:
        """
        Refine using multiple pages for more reliable results.

        Args:
            page_images: List of page images
            coarse_x1: Coarse estimate
            num_pages: Number of pages to use for refinement

        Returns:
            Dict with refined x1 and confidence
        """
        if not page_images:
            return {'x1': coarse_x1, 'error': 'No images provided', 'method': 'fallback'}

        pages_to_use = page_images[:num_pages]
        refinements = []

        for img in pages_to_use:
            result = self.refine(img, coarse_x1)
            if 'error' not in result:
                refinements.append(result['x1'])

        if not refinements:
            return {'x1': coarse_x1, 'method': 'fallback_no_refinements'}

        if len(refinements) == 1:
            return {
                'x1': refinements[0],
                'method': 'ai_vision_single',
                'confidence': 0.7
            }

        # Use median of refinements
        import numpy as np
        refined_x1 = float(np.median(refinements))
        spread = float(max(refinements) - min(refinements))

        # Confidence based on agreement
        if spread < 0.02:
            confidence = 0.95
        elif spread < 0.05:
            confidence = 0.80
        else:
            confidence = 0.60

        return {
            'x1': refined_x1,
            'method': 'ai_vision_multi',
            'spread': spread,
            'confidence': confidence,
            'individual_estimates': refinements
        }


def refine_boundary(
    page_image: Image.Image,
    coarse_x1: float,
    margin: float = 0.10
) -> float:
    """
    Convenience function to refine a boundary.

    Args:
        page_image: Full page PIL Image
        coarse_x1: Coarse estimate (0.0-1.0)
        margin: Margin to add left of estimate

    Returns:
        Refined x1 as fraction
    """
    refiner = BoundaryRefiner(margin=margin)
    result = refiner.refine(page_image, coarse_x1)
    return result['x1']
