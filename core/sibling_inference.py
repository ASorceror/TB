"""
Blueprint Processor V4.3 - Sibling Inference Module
Use successful neighboring pages to rescue isolated failures.

This module implements Phase F of V4.3: Sibling Inference
"""

from typing import List, Dict, Optional, Any

from core.template_applier import TemplateApplier
from core.template_types import Template

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import QUALITY_KEYWORDS


class SiblingInference:
    """Use successful neighbor pages to rescue isolated failures."""

    def __init__(self):
        """Initialize the sibling inference engine."""
        self.template_applier = TemplateApplier()
        self._keywords = set(kw.upper() for kw in QUALITY_KEYWORDS)

    def infer(
        self,
        results: List[Dict],
        failed_result: Dict,
        pdf_handler,
        template: Optional[Template] = None
    ) -> Dict[str, Any]:
        """
        Use neighboring pages to infer a title for a failed page.

        Args:
            results: Full list of extraction results for the PDF
            failed_result: The specific result that failed
            pdf_handler: Open PDFHandler instance to re-extract text
            template: Optional template for extraction parameters

        Returns:
            Dict with 'title', 'confidence', and 'method' keys
        """
        page_num = failed_result.get('page_number', 0)

        # Find neighbors
        prev_result = self._find_neighbor(results, page_num - 1)
        next_result = self._find_neighbor(results, page_num + 1)

        # Check neighbor success
        prev_success = self._is_successful(prev_result)
        next_success = self._is_successful(next_result)

        # Determine confidence base
        if prev_success and next_success:
            confidence_base = 0.80
            # Use method from neighbor with higher confidence
            if (prev_result.get('title_confidence', 0) >=
                    next_result.get('title_confidence', 0)):
                reference_method = prev_result.get('title_method')
            else:
                reference_method = next_result.get('title_method')
        elif prev_success:
            confidence_base = 0.70
            reference_method = prev_result.get('title_method')
        elif next_success:
            confidence_base = 0.70
            reference_method = next_result.get('title_method')
        else:
            # No successful neighbors
            return {"title": None, "confidence": 0.0, "method": "sibling_inference"}

        # Re-extract from failed page using neighbor's approach
        try:
            page_idx = page_num - 1  # Convert to 0-indexed
            page_text = pdf_handler.get_page_text(page_idx)

            # If we have a template, use it
            if template:
                text_blocks = None
                if failed_result.get('extraction_method') == 'vector':
                    text_blocks = pdf_handler.get_text_blocks(page_idx)

                applied = self.template_applier.apply(
                    template=template,
                    page_text=page_text,
                    text_blocks=text_blocks
                )

                if applied['title']:
                    return {
                        "title": applied['title'],
                        "confidence": min(confidence_base, applied['confidence']),
                        "method": "sibling_inference"
                    }

            # Fallback: look for lines with keywords
            title = self._extract_with_keywords(page_text)
            if title:
                return {
                    "title": title,
                    "confidence": confidence_base * 0.8,  # Lower confidence for keyword-only
                    "method": "sibling_inference"
                }

        except Exception:
            pass

        return {"title": None, "confidence": 0.0, "method": "sibling_inference"}

    def _find_neighbor(self, results: List[Dict], page_num: int) -> Optional[Dict]:
        """Find result for a specific page number."""
        for r in results:
            if r.get('page_number') == page_num:
                return r
        return None

    def _is_successful(self, result: Optional[Dict]) -> bool:
        """Check if a result is a successful extraction."""
        if not result:
            return False
        if result.get('needs_review') in [True, 1, '1']:
            return False
        if not result.get('sheet_title'):
            return False
        return True

    def _extract_with_keywords(self, page_text: str) -> Optional[str]:
        """
        Extract a title line that contains quality keywords.

        Args:
            page_text: Full page text

        Returns:
            First valid title line found, or None
        """
        if not page_text:
            return None

        for line in page_text.split('\n'):
            line = line.strip()
            if not line or len(line) < 8:
                continue

            # Check for keywords
            line_upper = line.upper()
            has_keyword = any(kw in line_upper for kw in self._keywords)

            if has_keyword and len(line) >= 10 and len(line) <= 60:
                # Basic validation
                if not line.isdigit() and not line.startswith(('DATE', 'SCALE', 'PROJECT')):
                    return line

        return None
