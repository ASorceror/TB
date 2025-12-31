"""
Blueprint Processor V6.2.2 - Sheet Title Extractor
Master coordinator for the 4-layer title extraction system.

V6.2.2 Changes:
- Added crop orientation detection and correction using Tesseract OSD
- Fixes crops with rotated text (e.g., CrunchFitness title blocks)
- Crops are automatically rotated so text is readable

V6.2 Changes:
- Replaced slow TitleBlockDiscovery (Vision AI) with fast TitleBlockDetector (CV-based)
- Detection now uses Hough lines + density transition (same as run_complete_extraction.py)
- Processing is now ~360x faster (1 sec vs 6 min per PDF)

V6.1 Changes:
- Added automatic saving of title block crops to output/crops/<pdf_hash>/
- Each page's crop saved as p###_titleblock.png

V4.7 Changes:
- Added text_blocks retrieval and passing to extract_fields
- Enables Strategy 4b spatial proximity matching
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import TITLE_CONFIDENCE, REVIEW_THRESHOLD
from core.extractor import Extractor
from core.drawing_index import DrawingIndexParser
from core.spatial_extractor import SpatialExtractor
from core.vision_extractor import VisionExtractor
from core.pdf_handler import PDFHandler
from core.title_block_detector import TitleBlockDetector
from core.ocr_engine import OCREngine
from core.ocr_utils import detect_and_correct_crop_orientation
from validation.validator import Validator

logger = logging.getLogger(__name__)


class SheetTitleExtractor:
    """
    Master coordinator for sheet title extraction using 4-layer approach.

    Layers (in priority order):
    1. Drawing Index lookup (95% confidence) - from cover sheet
    2. Spatial Zone Detection (80% confidence) - vector PDFs only
    3. Vision API (90% confidence) - when Layers 1-2 fail
    4. Pattern Matching (70% confidence) - fallback

    Also provides:
    - Cross-reference validation (use index to correct OCR errors)
    - PDF hash for stable database keys
    - HITL flagging for low-confidence results
    """

    def __init__(self, telemetry_dir: Optional[Path] = None):
        """Initialize the SheetTitleExtractor.

        Args:
            telemetry_dir: Optional directory for debug images
        """
        self._extractor = Extractor()
        self._validator = Validator()
        # V6.2: Use fast CV-based detector (same as run_complete_extraction.py)
        self._title_block_detector = TitleBlockDetector(use_ai_refinement=False)
        self._ocr_engine = OCREngine(telemetry_dir=telemetry_dir)

        # V6.1: Store telemetry_dir and create crops subdirectory
        self._telemetry_dir = telemetry_dir
        self._crops_dir: Optional[Path] = None
        if telemetry_dir:
            self._crops_dir = telemetry_dir.parent / 'crops'
            self._crops_dir.mkdir(parents=True, exist_ok=True)

        # Current PDF state
        self._current_pdf_hash: Optional[str] = None
        self._drawing_index: Dict[str, str] = {}
        self._detection_result: Optional[Dict] = None  # V6.2: Cached detection result
        self._extraction_stats: Dict[str, int] = {
            'drawing_index': 0,
            'spatial': 0,
            'vision_api': 0,
            'pattern': 0,
            'failed': 0,
        }

    def reset_for_new_pdf(self):
        """Reset state for processing a new PDF."""
        self._current_pdf_hash = None
        self._drawing_index = {}
        self._detection_result = None  # V6.2: Reset detection cache
        self._extractor.reset_for_new_pdf()
        self._extraction_stats = {
            'drawing_index': 0,
            'spatial': 0,
            'vision_api': 0,
            'pattern': 0,
            'failed': 0,
        }

    def run_discovery(self, pdf_handler: PDFHandler, pdf_hash: str) -> None:
        """
        Run title block detection for the PDF using fast CV-based detection.

        V6.2: Uses TitleBlockDetector (CV-based) instead of slow Vision AI.
        This should be called ONCE per PDF before processing pages.

        Args:
            pdf_handler: PDFHandler instance
            pdf_hash: SHA-256 hash of PDF
        """
        if self._detection_result is not None and self._current_pdf_hash == pdf_hash:
            logger.debug("Detection already complete for this PDF")
            return

        self._current_pdf_hash = pdf_hash

        # Sample pages for detection (same strategy as run_complete_extraction.py)
        total_pages = pdf_handler.page_count
        if total_pages > 5:
            sample_indices = [1, 3, 5]  # 0-indexed: pages 2, 4, 6
        else:
            sample_indices = [1, 2] if total_pages >= 3 else [0]

        # Render sample pages at low DPI for fast detection
        sample_images = []
        for idx in sample_indices:
            if idx < total_pages:
                img = pdf_handler.get_page_image(idx, dpi=100)
                sample_images.append(img)

        if not sample_images:
            logger.warning("No sample pages available for detection")
            self._detection_result = {'x1': 0.85, 'width_pct': 0.15, 'method': 'default'}
            return

        # Run CV-based detection
        logger.info(f"Running title block detection for PDF hash {pdf_hash}")
        self._detection_result = self._title_block_detector.detect(
            sample_images, strategy='balanced'
        )

        # V6.2.2: Sanity check - title blocks are typically 10-20% of page width
        # If detected width > 25%, detection is likely wrong - use default
        MAX_TITLE_BLOCK_WIDTH = 0.25
        if self._detection_result['width_pct'] > MAX_TITLE_BLOCK_WIDTH:
            logger.warning(f"Detection width {self._detection_result['width_pct']*100:.1f}% exceeds max {MAX_TITLE_BLOCK_WIDTH*100:.0f}% - using default")
            self._detection_result = {'x1': 0.85, 'width_pct': 0.15, 'method': 'default_width_exceeded'}

        logger.info(f"Detection complete: x1={self._detection_result['x1']:.3f}, "
                   f"width={self._detection_result['width_pct']*100:.1f}%, "
                   f"method={self._detection_result['method']}")

    def compute_pdf_hash(self, pdf_path: Path) -> str:
        """
        Compute SHA-256 hash of PDF file for stable identification.

        Args:
            pdf_path: Path to PDF file

        Returns:
            First 16 characters of SHA-256 hash
        """
        hasher = hashlib.sha256()
        with open(pdf_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        self._current_pdf_hash = hasher.hexdigest()[:16]
        return self._current_pdf_hash

    def parse_drawing_index(self, pdf_handler: PDFHandler) -> Dict[str, str]:
        """
        Parse drawing index from PDF cover sheet.

        Args:
            pdf_handler: PDFHandler instance

        Returns:
            Dict mapping sheet_number -> title
        """
        self._drawing_index = self._extractor.parse_drawing_index(pdf_handler)
        logger.info(f"Drawing index parsed: {len(self._drawing_index)} entries")
        return self._drawing_index

    def extract_title(
        self,
        pdf_handler: PDFHandler,
        page_num: int,
        page_image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        original_page_image: Optional[Image.Image] = None,
        orientation_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract sheet title using the 4-layer approach.

        Args:
            pdf_handler: PDFHandler instance
            page_num: 0-indexed page number
            page_image: Optional pre-rendered (normalized) page image
            text: Optional pre-extracted text
            original_page_image: Optional original (non-rotated) page image (V4.4)
            orientation_info: Optional dict with rotation info from normalizer (V4.4)

        Returns:
            Dict with extraction results including:
                - sheet_number: str or None
                - sheet_title: str or None
                - title_confidence: float 0.0-1.0
                - title_method: str (drawing_index, spatial, vision_api, pattern)
                - needs_review: bool
                - pdf_hash: str
        """
        page = pdf_handler.doc[page_num]
        page_number_1idx = page_num + 1

        # Get text if not provided
        if text is None:
            text = page.get_text()

        # Get page image if not provided (needed for Vision API)
        title_block_image = None
        title_block_bbox = None

        # V4.4: Get rotation info from orientation_info passed from main.py
        rotation_applied = 0
        if orientation_info:
            rotation_applied = orientation_info.get('rotation_applied', 0)

        if page_image is not None:
            # V6.2: Use TitleBlockDetector (CV-based) for cropping
            # Detection should already be complete (run via run_discovery)
            if self._detection_result is None:
                # Fallback: use default if detection wasn't run
                self._detection_result = {'x1': 0.85, 'width_pct': 0.15, 'method': 'default'}
                logger.warning("Detection not run - using default x1=0.85")

            # Crop title block using detected x1 (full height strip)
            title_block_image = self._title_block_detector.crop_title_block(
                page_image, self._detection_result
            )
            width, height = page_image.size
            x1_px = int(self._detection_result['x1'] * width)
            title_block_bbox = (x1_px, 0, width, height)

            # V6.2.2: DISABLED - Orientation correction was making crops upside down
            # The OSD detection was incorrect for CrunchFitness crops
            # TODO: Investigate why OSD detected 180° when text was already correct
            # title_block_image, orientation_correction = detect_and_correct_crop_orientation(
            #     title_block_image
            # )
            # if orientation_correction.get('rotation_applied', 0) != 0:
            #     logger.info(f"Page {page_number_1idx}: Crop rotated {orientation_correction['rotation_applied']}° "
            #                f"(confidence: {orientation_correction['confidence']:.2f})")

            # V6.1: Save title block crop to disk
            if self._crops_dir and title_block_image and self._current_pdf_hash:
                try:
                    # Create PDF-specific subfolder using first 8 chars of hash
                    pdf_crop_dir = self._crops_dir / self._current_pdf_hash[:8]
                    pdf_crop_dir.mkdir(parents=True, exist_ok=True)

                    crop_filename = f"p{page_number_1idx:03d}_titleblock.png"
                    crop_path = pdf_crop_dir / crop_filename
                    title_block_image.save(crop_path, 'PNG')
                    logger.debug(f"Saved crop: {crop_path}")
                except Exception as e:
                    logger.warning(f"Failed to save crop for page {page_number_1idx}: {e}")

        # V4.7: Get text blocks with coordinates for spatial proximity matching
        text_blocks = pdf_handler.get_text_blocks(page_num)

        # Extract using layered approach
        # V4.4: Pass both normalized and original images for rotation fallback
        # V4.7: Pass text_blocks for Strategy 4b spatial proximity matching
        result = self._extractor.extract_fields(
            text=text,
            text_blocks=text_blocks,  # V4.7: Text blocks with bbox coordinates
            page_number=page_number_1idx,
            page=page,
            title_block_bbox_pixels=title_block_bbox,
            title_block_image=title_block_image,
            page_image=page_image,  # V4.4: Normalized page for edge OCR
            original_page_image=original_page_image,  # V4.4: Original for rotation fallback
            rotation_applied=rotation_applied,  # V4.4: Rotation info
        )

        # Add PDF hash
        result['pdf_hash'] = self._current_pdf_hash

        # Track extraction method
        method = result.get('title_method')
        if method:
            self._extraction_stats[method] = self._extraction_stats.get(method, 0) + 1
        else:
            self._extraction_stats['failed'] += 1

        # Apply cross-reference validation
        result = self._cross_reference_validate(result)

        return result

    def _cross_reference_validate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use drawing index to validate and correct extraction results.

        If sheet_number is found but title doesn't match index:
        - If index title is more specific, use it
        - Flag for review if conflict

        Args:
            result: Extraction result dict

        Returns:
            Updated result dict
        """
        if not self._drawing_index:
            return result

        sheet_number = result.get('sheet_number')
        extracted_title = result.get('sheet_title')
        current_method = result.get('title_method')

        if not sheet_number:
            return result

        # Look up in index
        index_title = self._extractor.lookup_in_index(sheet_number)

        if not index_title:
            return result

        # If we got a title from a lower-confidence method, prefer index
        if current_method in ['pattern', 'spatial'] and index_title:
            # Index is more reliable
            if extracted_title != index_title:
                result['sheet_title'] = index_title
                result['title_confidence'] = TITLE_CONFIDENCE['DRAWING_INDEX']
                result['title_method'] = 'drawing_index_xref'
                result['needs_review'] = False
                result['extraction_details'] = result.get('extraction_details', {})
                result['extraction_details']['xref_override'] = True
                result['extraction_details']['original_title'] = extracted_title
                result['extraction_details']['original_method'] = current_method
                logger.debug(
                    f"Cross-ref override: '{extracted_title}' -> '{index_title}' "
                    f"(sheet {sheet_number})"
                )

        # If Vision API result conflicts with index, flag for review
        elif current_method == 'vision_api' and extracted_title and index_title:
            # Both sources have a title - check for conflict
            if extracted_title.upper() != index_title.upper():
                # Significant conflict - flag for review
                result['needs_review'] = True
                result['extraction_details'] = result.get('extraction_details', {})
                result['extraction_details']['xref_conflict'] = True
                result['extraction_details']['index_title'] = index_title

        return result

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics for current PDF."""
        total = sum(self._extraction_stats.values())
        return {
            'total_pages': total,
            'by_method': self._extraction_stats.copy(),
            'success_rate': (total - self._extraction_stats['failed']) / total if total > 0 else 0,
            'pdf_hash': self._current_pdf_hash,
        }

    def get_sheets_needing_review(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter results to those needing human review.

        Args:
            results: List of extraction results

        Returns:
            List of results where needs_review is True
        """
        return [r for r in results if r.get('needs_review', False)]

    def process_pdf(
        self,
        pdf_path: Path,
        render_dpi: int = 200
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process an entire PDF through the extraction pipeline.

        V6.2: Uses fast CV-based title block detection before processing pages.

        Args:
            pdf_path: Path to PDF file
            render_dpi: DPI for page rendering

        Returns:
            Tuple of (results_list, stats_dict)
        """
        self.reset_for_new_pdf()

        # Compute PDF hash
        pdf_hash = self.compute_pdf_hash(pdf_path)
        logger.info(f"Processing PDF: {pdf_path.name} (hash: {pdf_hash})")

        results = []

        with PDFHandler(pdf_path) as handler:
            # V6.2: Run fast CV-based title block detection FIRST
            self.run_discovery(handler, pdf_hash)

            # Parse drawing index
            self.parse_drawing_index(handler)

            # Process each page
            for page_num in range(handler.page_count):
                # Render page image for region detection
                page_image = handler.get_page_image(page_num, dpi=render_dpi)

                # Extract title
                result = self.extract_title(
                    pdf_handler=handler,
                    page_num=page_num,
                    page_image=page_image,
                )

                # Add source info
                result['pdf_filename'] = pdf_path.name
                result['page_number'] = page_num + 1

                results.append(result)

        stats = self.get_extraction_stats()
        return results, stats
