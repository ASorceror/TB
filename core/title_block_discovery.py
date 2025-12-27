"""
Blueprint Processor V6.0 - Title Block Discovery

Uses Vision AI to analyze sample pages from a PDF and discover:
1. Whether page 1 is a cover sheet (no title block)
2. Title block location and dimensions
3. Internal zones within the title block (sheet number, title, etc.)
4. Text orientation (horizontal or rotated 90°)

This replaces the unreliable RegionDetector that used OCR-based keyword matching.

The discovery runs ONCE per PDF, samples non-consecutive pages, and caches
results to disk for reuse.
"""

import os
import io
import json
import base64
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================================
# DISCOVERY PROMPT - CRITICAL FOR ACCURACY
# ============================================================================

TITLE_BLOCK_DISCOVERY_PROMPT = """You are analyzing architectural/engineering blueprint pages to discover the title block structure.

I am showing you {num_pages} pages from the same PDF blueprint set. Analyze ALL pages together to identify consistent patterns.

## CRITICAL: What is a Title Block?

A title block is the FORMATTED INFORMATION PANEL on a blueprint - NOT the drawing area.

## HOW TO FIND THE EXACT BOUNDARIES - CRITICAL INSTRUCTIONS

**STEP 1: Identify the title block visually**
- The title block is a rectangular panel, usually on the RIGHT EDGE or BOTTOM EDGE of the page
- It contains structured information: firm name, project name, sheet number, dates, scales, etc.
- It is separated from the main drawing area by visible border lines

**STEP 2: Find the LEFT BOUNDARY (x1) - THIS IS THE MOST IMPORTANT**
- Look for the thick VERTICAL BORDER LINE that separates the title block from the drawing area
- Your x1 value should be positioned so that:
  * ALL text inside the title block is FULLY included (no partial letters or cut-off words)
  * There is a small WHITESPACE MARGIN between x1 and the leftmost text
  * The vertical border line itself is NOT included in the crop
- VALIDATION: If cropping at x1 would cut through ANY text (even 1 pixel of a letter), move x1 FURTHER LEFT
- The goal is: COMPLETE TEXT + NATURAL WHITESPACE MARGIN on all sides

**STEP 3: Find the OTHER BOUNDARIES (y1, x2, y2)**
- Apply the same principle: include ALL text with whitespace margins
- For right edge (x2): include all content plus margin before the page edge
- For top (y1) and bottom (y2): include all content with margins

**STEP 4: Width Reality Check**
- Title block widths VARY SIGNIFICANTLY between different architectural firms
- Some are narrow (8-10% of page width), others are wide (15-25% of page width)
- DO NOT assume a specific width - measure what you actually see
- If the title block contains the firm logo, revision table, AND project info stacked vertically, it may be wider

**Common title block layouts:**
- VERTICAL STRIP on right edge (varies from 8% to 25% of page width)
- HORIZONTAL STRIP on bottom edge (full width, varies in height)
- CORNER BLOCK in bottom-right (varies in size)

**Title block characteristics:**
- Contains: firm name/logo, project name, sheet number, date, scale, drawn by, revision history, etc.
- Has CLEARLY VISIBLE BORDER LINES forming a closed rectangle
- Often divided into smaller boxes/cells inside
- Text may be horizontal OR rotated 90 degrees (especially firm names on narrow vertical strips)

## Your Task

1. **Page 1 Analysis**: Determine if Page 1 is a COVER SHEET (no title block, contains drawing index, project info only) or a DRAWING SHEET (has title block with sheet number).

2. **Title Block Location - FIND THE CONTENT BOUNDARIES**:
   - Find where the title block CONTENT actually is (text, logos, tables)
   - Set boundaries so ALL text is COMPLETELY inside with whitespace margins
   - The bbox should EXCLUDE the border lines but INCLUDE all content with margins
   - NEVER cut through any text - if in doubt, make the bbox slightly larger
   - Measure the actual width you observe - don't assume a percentage

3. **Zone Detection**: Within the title block, identify these zones with PRECISE boundaries that DO NOT OVERLAP:

   **ZONE A - Sheet Identification** (CRITICAL - extraction target)
   - Contains: Sheet Number (e.g., "A-101", "M-2.1") and Sheet Title (e.g., "FLOOR PLAN")
   - Location: Usually bottom portion of title block
   - Text orientation: Usually horizontal

   **ZONE B - Revision Block**
   - Contains: Revision table with dates, descriptions, revision numbers
   - Location: Often above or beside Zone A
   - Text orientation: Usually horizontal

   **ZONE C - Project Information**
   - Contains: Project name, client name, site address
   - Location: Middle portion of title block
   - Text orientation: Usually horizontal

   **ZONE D - Firm/Designer Block**
   - Contains: Architecture firm logo, name, address, phone, professional seals
   - Location: Top of title block OR left side
   - Text orientation: Often VERTICAL (rotated 90° clockwise) when on left/right edges

4. **Consistency Check**: Verify if the title block layout is CONSISTENT across all drawing pages shown.

## Output Format

Return ONLY valid JSON in this exact structure:

```json
{
  "page_1_is_cover_sheet": true,
  "page_1_analysis": "Brief explanation of why page 1 is/isn't a cover sheet",

  "title_block": {
    "location": "right_edge_vertical",
    "bbox_percent": {
      "x1": 0.82,
      "y1": 0.01,
      "x2": 0.99,
      "y2": 0.99
    },
    "confidence": 0.98,
    "notes": "x1 set to include all text with whitespace margin, actual width ~17% of page"
  },

  "zones": {
    "sheet_identification": {
      "bbox_percent": {"x1": 0.0, "y1": 0.75, "x2": 0.5, "y2": 1.0},
      "text_orientation": "horizontal",
      "contains": ["sheet_number", "sheet_title"],
      "confidence": 0.95
    },
    "revision_block": {
      "bbox_percent": {"x1": 0.5, "y1": 0.0, "x2": 1.0, "y2": 0.40},
      "text_orientation": "horizontal",
      "contains": ["revision_table", "dates"],
      "confidence": 0.90
    },
    "project_info": {
      "bbox_percent": {"x1": 0.0, "y1": 0.40, "x2": 1.0, "y2": 0.75},
      "text_orientation": "horizontal",
      "contains": ["project_name", "client", "address"],
      "confidence": 0.85
    },
    "firm_block": {
      "bbox_percent": {"x1": 0.0, "y1": 0.0, "x2": 0.3, "y2": 0.40},
      "text_orientation": "vertical_90cw",
      "contains": ["firm_name", "logo", "seal"],
      "confidence": 0.80
    }
  },

  "layout_consistency": {
    "is_consistent": true,
    "consistency_score": 0.95,
    "notes": "All drawing pages have identical title block layout"
  },

  "warnings": []
}
```

## Critical Rules

1. **bbox_percent values**: All coordinates are percentages (0.0 to 1.0):
   - For title_block: relative to the FULL PAGE
   - For zones: relative to the TITLE BLOCK (not the page)

2. **NO ZONE OVERLAP**: Zone boundaries must NOT overlap. Each pixel belongs to exactly ONE zone.

3. **text_orientation values**:
   - "horizontal" - normal left-to-right text
   - "vertical_90cw" - text rotated 90° clockwise (read bottom-to-top)
   - "vertical_90ccw" - text rotated 90° counter-clockwise (read top-to-bottom)

4. **If a zone is not present**, omit it from the zones object (don't include with null values).

5. **Confidence scores**: 0.0 to 1.0 based on how certain you are about the detection.

Analyze the images now and return ONLY the JSON response."""


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BBox:
    """Bounding box as percentages (0.0 to 1.0)."""
    x1: float
    y1: float
    x2: float
    y2: float

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert percentage bbox to pixel coordinates."""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height)
        )

    def validate(self) -> bool:
        """Check if bbox is valid (no negative area, within bounds)."""
        return (
            0.0 <= self.x1 < self.x2 <= 1.0 and
            0.0 <= self.y1 < self.y2 <= 1.0
        )


@dataclass
class Zone:
    """A zone within the title block."""
    name: str
    bbox_percent: BBox
    text_orientation: str  # horizontal, vertical_90cw, vertical_90ccw
    contains: List[str]
    confidence: float


@dataclass
class TitleBlockTelemetry:
    """Complete title block discovery result for a PDF."""
    pdf_hash: str
    discovery_timestamp: str
    pages_analyzed: List[int]

    page_1_is_cover_sheet: bool
    page_1_analysis: str

    title_block_location: str  # bottom_right, right_strip, bottom_strip
    title_block_bbox: BBox
    title_block_confidence: float

    zones: Dict[str, Zone]

    layout_is_consistent: bool
    consistency_score: float

    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            'pdf_hash': self.pdf_hash,
            'discovery_timestamp': self.discovery_timestamp,
            'pages_analyzed': self.pages_analyzed,
            'page_1_is_cover_sheet': self.page_1_is_cover_sheet,
            'page_1_analysis': self.page_1_analysis,
            'title_block': {
                'location': self.title_block_location,
                'bbox_percent': asdict(self.title_block_bbox),
                'confidence': self.title_block_confidence
            },
            'zones': {},
            'layout_consistency': {
                'is_consistent': self.layout_is_consistent,
                'consistency_score': self.consistency_score
            },
            'warnings': self.warnings
        }
        for zone_name, zone in self.zones.items():
            result['zones'][zone_name] = {
                'bbox_percent': asdict(zone.bbox_percent),
                'text_orientation': zone.text_orientation,
                'contains': zone.contains,
                'confidence': zone.confidence
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TitleBlockTelemetry':
        """Create from JSON dict."""
        tb_bbox = data['title_block']['bbox_percent']
        zones = {}
        for zone_name, zone_data in data.get('zones', {}).items():
            zb = zone_data['bbox_percent']
            zones[zone_name] = Zone(
                name=zone_name,
                bbox_percent=BBox(zb['x1'], zb['y1'], zb['x2'], zb['y2']),
                text_orientation=zone_data.get('text_orientation', 'horizontal'),
                contains=zone_data.get('contains', []),
                confidence=zone_data.get('confidence', 0.0)
            )
        return cls(
            pdf_hash=data['pdf_hash'],
            discovery_timestamp=data['discovery_timestamp'],
            pages_analyzed=data['pages_analyzed'],
            page_1_is_cover_sheet=data['page_1_is_cover_sheet'],
            page_1_analysis=data.get('page_1_analysis', ''),
            title_block_location=data['title_block']['location'],
            title_block_bbox=BBox(tb_bbox['x1'], tb_bbox['y1'], tb_bbox['x2'], tb_bbox['y2']),
            title_block_confidence=data['title_block']['confidence'],
            zones=zones,
            layout_is_consistent=data['layout_consistency']['is_consistent'],
            consistency_score=data['layout_consistency']['consistency_score'],
            warnings=data.get('warnings', [])
        )


# ============================================================================
# SAMPLING STRATEGY
# ============================================================================

def calculate_sample_pages(total_pages: int) -> List[int]:
    """
    Calculate which pages to sample for title block discovery.

    Strategy:
    - Always include page 1 (to check if cover sheet)
    - Sample non-consecutive pages evenly distributed
    - More pages for larger PDFs

    Args:
        total_pages: Total number of pages in PDF

    Returns:
        List of 1-indexed page numbers to sample
    """
    if total_pages <= 3:
        # Very small PDF - sample all pages
        return list(range(1, total_pages + 1))

    # Determine sample size based on PDF length
    if total_pages <= 10:
        sample_size = 3
    elif total_pages <= 30:
        sample_size = 4
    elif total_pages <= 100:
        sample_size = 5
    else:
        sample_size = 6

    # Always start with page 1
    pages = [1]

    # Distribute remaining samples evenly (excluding page 1)
    remaining = sample_size - 1
    if remaining > 0 and total_pages > 1:
        # Calculate step size to distribute evenly
        available_pages = total_pages - 1  # Exclude page 1
        step = max(1, available_pages // (remaining + 1))

        for i in range(1, remaining + 1):
            page = 1 + (step * i)
            if page <= total_pages and page not in pages:
                pages.append(page)

    # Ensure we have at least one non-page-1 sample for consistency check
    if len(pages) == 1 and total_pages > 1:
        pages.append(min(2, total_pages))

    return sorted(pages)


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

class DiscoveryCache:
    """Manages disk caching of discovery results."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache with directory path."""
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / 'output' / 'telemetry'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, pdf_hash: str) -> Path:
        """Get cache file path for a PDF hash."""
        return self.cache_dir / f"{pdf_hash}_discovery.json"

    def get(self, pdf_hash: str) -> Optional[TitleBlockTelemetry]:
        """Load cached discovery result if available."""
        cache_path = self._get_cache_path(pdf_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded cached discovery for {pdf_hash}")
                return TitleBlockTelemetry.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load cache for {pdf_hash}: {e}")
        return None

    def save(self, telemetry: TitleBlockTelemetry) -> Path:
        """Save discovery result to cache."""
        cache_path = self._get_cache_path(telemetry.pdf_hash)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(telemetry.to_dict(), f, indent=2)
            logger.info(f"Saved discovery cache to {cache_path}")
            return cache_path
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            raise

    def invalidate(self, pdf_hash: str) -> bool:
        """Remove cached result for a PDF."""
        cache_path = self._get_cache_path(pdf_hash)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False


# ============================================================================
# MAIN DISCOVERY CLASS
# ============================================================================

class TitleBlockDiscovery:
    """
    Discovers title block structure using Vision AI.

    Replaces RegionDetector with AI-based detection that:
    1. Samples pages from the PDF
    2. Uses Vision API to analyze title block structure
    3. Caches results for reuse
    4. Provides same interface as RegionDetector for compatibility
    """

    def __init__(self, cache_dir: Optional[Path] = None, telemetry_dir: Optional[Path] = None):
        """
        Initialize TitleBlockDiscovery.

        Args:
            cache_dir: Directory for caching discovery results
            telemetry_dir: Directory for saving debug images (compatibility with RegionDetector)
        """
        self._client = None
        self._api_available = False
        self._cache = DiscoveryCache(cache_dir)
        self.telemetry_dir = telemetry_dir

        # Current PDF state
        self._current_telemetry: Optional[TitleBlockTelemetry] = None
        self._current_pdf_hash: Optional[str] = None

        # Initialize API client
        self._init_client()

    def _init_client(self):
        """Initialize the Anthropic API client."""
        api_key = os.environ.get('ANTHROPIC_API_KEY')

        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set - Title Block Discovery disabled")
            self._api_available = False
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            self._api_available = True
            logger.info("Title Block Discovery API initialized successfully")
        except ImportError:
            logger.warning("anthropic package not installed - Title Block Discovery disabled")
            self._api_available = False
        except Exception as e:
            logger.error(f"Failed to initialize Discovery API: {e}")
            self._api_available = False

    def is_available(self) -> bool:
        """Check if Vision API is available for discovery."""
        return self._api_available

    def discover(
        self,
        pdf_handler,
        pdf_hash: str,
        force_refresh: bool = False,
        dpi: int = 150
    ) -> TitleBlockTelemetry:
        """
        Discover title block structure for a PDF.

        This should be called ONCE per PDF, typically during initialization.
        Results are cached for subsequent page processing.

        Args:
            pdf_handler: PDFHandler instance with open PDF
            pdf_hash: SHA-256 hash of PDF for cache key
            force_refresh: If True, ignore cache and re-discover
            dpi: DPI for rendering sample pages (default 150 for speed)

        Returns:
            TitleBlockTelemetry with discovery results

        Raises:
            RuntimeError: If discovery fails and no fallback available
        """
        self._current_pdf_hash = pdf_hash

        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = self._cache.get(pdf_hash)
            if cached:
                self._current_telemetry = cached
                return cached

        # Check API availability
        if not self._api_available:
            raise RuntimeError(
                "Title Block Discovery requires Vision API but ANTHROPIC_API_KEY is not set. "
                "Please set the environment variable and try again."
            )

        # Calculate which pages to sample
        total_pages = pdf_handler.page_count
        sample_pages = calculate_sample_pages(total_pages)
        logger.info(f"Sampling pages {sample_pages} from {total_pages} total pages")

        # Render sample pages
        page_images = []
        for page_num in sample_pages:
            try:
                image = pdf_handler.get_page_image(page_num - 1, dpi=dpi)  # 0-indexed
                page_images.append((page_num, image))
            except Exception as e:
                logger.warning(f"Failed to render page {page_num}: {e}")

        if not page_images:
            raise RuntimeError("Failed to render any sample pages")

        # Call Vision API with all sample images
        try:
            telemetry = self._call_vision_api(page_images, pdf_hash, sample_pages)
        except Exception as e:
            logger.error(f"Vision API discovery failed: {e}")
            raise RuntimeError(f"Title block discovery failed: {e}")

        # Cache the result
        self._cache.save(telemetry)
        self._current_telemetry = telemetry

        return telemetry

    def _call_vision_api(
        self,
        page_images: List[Tuple[int, Image.Image]],
        pdf_hash: str,
        sample_pages: List[int]
    ) -> TitleBlockTelemetry:
        """
        Call Vision API with multiple page images.

        Args:
            page_images: List of (page_number, PIL Image) tuples
            pdf_hash: PDF hash for telemetry
            sample_pages: List of sampled page numbers

        Returns:
            TitleBlockTelemetry parsed from API response
        """
        # Build message content with all images
        content = []

        for page_num, image in page_images:
            # Resize if needed (max 1568 pixels on long edge for efficiency)
            image = self._resize_for_api(image, max_size=1568)

            # Convert to base64
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_data = base64.standard_b64encode(img_bytes.getvalue()).decode('utf-8')

            # Add image with label
            content.append({
                "type": "text",
                "text": f"--- PAGE {page_num} ---"
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_data
                }
            })

        # Add the prompt (replace placeholder manually to avoid format() issues with JSON braces)
        prompt = TITLE_BLOCK_DISCOVERY_PROMPT.replace("{num_pages}", str(len(page_images)))
        content.append({
            "type": "text",
            "text": prompt
        })

        logger.info(f"Calling Vision API with {len(page_images)} page images")

        # Call API
        message = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": content
            }]
        )

        # Parse response
        response_text = message.content[0].text.strip()
        logger.debug(f"Vision API response length: {len(response_text)} chars")
        logger.debug(f"Vision API response preview: {response_text[:500]}...")

        # Extract JSON from response (handle markdown code blocks)
        # Use regex for more robust extraction
        import re

        json_text = None

        # Strategy 1: Find JSON in ```json ... ``` code block
        code_block_match = re.search(r'```json\s*([\s\S]*?)```', response_text)
        if code_block_match:
            json_text = code_block_match.group(1).strip()
            logger.debug(f"Extracted JSON from ```json block: {len(json_text)} chars")

        # Strategy 2: Find JSON in ``` ... ``` code block (no language marker)
        if not json_text:
            code_block_match = re.search(r'```\s*([\s\S]*?)```', response_text)
            if code_block_match:
                candidate = code_block_match.group(1).strip()
                if candidate.startswith('{'):
                    json_text = candidate
                    logger.debug(f"Extracted JSON from ``` block: {len(json_text)} chars")

        # Strategy 3: Find the outermost { ... } in the response
        if not json_text:
            # Find first { and count brackets to find matching }
            start_idx = response_text.find('{')
            if start_idx >= 0:
                depth = 0
                end_idx = start_idx
                for i in range(start_idx, len(response_text)):
                    if response_text[i] == '{':
                        depth += 1
                    elif response_text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
                if end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx]
                    logger.debug(f"Extracted raw JSON via bracket matching: {len(json_text)} chars")

        if not json_text:
            json_text = response_text  # Last resort, try to parse the whole thing

        try:
            result = json.loads(json_text)
            logger.debug(f"JSON parsed successfully, keys: {list(result.keys())}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Vision API response as JSON: {e}")
            logger.error(f"Response was:\n{response_text}")
            raise RuntimeError(f"Vision API returned invalid JSON: {e}")

        # Convert to TitleBlockTelemetry
        return self._parse_vision_response(result, pdf_hash, sample_pages)

    def _parse_vision_response(
        self,
        response: Dict[str, Any],
        pdf_hash: str,
        sample_pages: List[int]
    ) -> TitleBlockTelemetry:
        """Parse Vision API response into TitleBlockTelemetry."""
        # Extract title block bbox
        tb = response.get('title_block', {})
        tb_bbox = tb.get('bbox_percent', {'x1': 0.65, 'y1': 0.70, 'x2': 1.0, 'y2': 1.0})

        # Extract zones
        zones = {}
        for zone_name, zone_data in response.get('zones', {}).items():
            zb = zone_data.get('bbox_percent', {})
            if zb:
                zones[zone_name] = Zone(
                    name=zone_name,
                    bbox_percent=BBox(
                        zb.get('x1', 0.0),
                        zb.get('y1', 0.0),
                        zb.get('x2', 1.0),
                        zb.get('y2', 1.0)
                    ),
                    text_orientation=zone_data.get('text_orientation', 'horizontal'),
                    contains=zone_data.get('contains', []),
                    confidence=zone_data.get('confidence', 0.0)
                )

        # Extract consistency info
        consistency = response.get('layout_consistency', {})

        return TitleBlockTelemetry(
            pdf_hash=pdf_hash,
            discovery_timestamp=datetime.now().isoformat(),
            pages_analyzed=sample_pages,
            page_1_is_cover_sheet=response.get('page_1_is_cover_sheet', False),
            page_1_analysis=response.get('page_1_analysis', ''),
            title_block_location=tb.get('location', 'bottom_right'),
            title_block_bbox=BBox(
                tb_bbox.get('x1', 0.65),
                tb_bbox.get('y1', 0.70),
                tb_bbox.get('x2', 1.0),
                tb_bbox.get('y2', 1.0)
            ),
            title_block_confidence=tb.get('confidence', 0.0),
            zones=zones,
            layout_is_consistent=consistency.get('is_consistent', True),
            consistency_score=consistency.get('consistency_score', 0.0),
            warnings=response.get('warnings', [])
        )

    def _resize_for_api(self, image: Image.Image, max_size: int = 1568) -> Image.Image:
        """Resize image if any dimension exceeds max_size."""
        width, height = image.size
        if width <= max_size and height <= max_size:
            return image

        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # ========================================================================
    # COMPATIBILITY INTERFACE (matches RegionDetector)
    # ========================================================================

    def detect_title_block(self, image: Image.Image, dpi: int = 300) -> Dict[str, Any]:
        """
        Detect title block region in the image.

        This provides the same interface as RegionDetector.detect_title_block()
        but uses cached telemetry from the discovery phase.

        Args:
            image: Page image (PIL Image)
            dpi: DPI of the image (for reference, not used in detection)

        Returns:
            Dict with keys: bbox, confidence, method, region_name
        """
        if self._current_telemetry is None:
            # No discovery has been run - return default region
            logger.warning("detect_title_block called without discovery - using defaults")
            width, height = image.size
            return {
                'bbox': (int(width * 0.65), int(height * 0.70), width, height),
                'confidence': 0.0,
                'method': 'default_fallback',
                'region_name': 'bottom_right_default',
            }

        # Use telemetry to calculate bbox - AI should have found correct boundaries
        # with whitespace margins, so no additional padding needed
        width, height = image.size
        tb_bbox = self._current_telemetry.title_block_bbox

        # Convert to pixels directly (no padding - AI finds correct boundaries)
        bbox_pixels = (
            int(tb_bbox.x1 * width),
            int(tb_bbox.y1 * height),
            int(tb_bbox.x2 * width),
            int(tb_bbox.y2 * height)
        )

        return {
            'bbox': bbox_pixels,
            'confidence': self._current_telemetry.title_block_confidence,
            'method': 'vision_discovery',
            'region_name': self._current_telemetry.title_block_location,
            'zones': {
                name: {
                    'bbox_relative': asdict(zone.bbox_percent),
                    'text_orientation': zone.text_orientation,
                    'confidence': zone.confidence
                }
                for name, zone in self._current_telemetry.zones.items()
            }
        }

    def crop_title_block(self, image: Image.Image, detection_result: Dict[str, Any]) -> Image.Image:
        """
        Crop the detected title block region from the image.

        This provides the same interface as RegionDetector.crop_title_block().

        Args:
            image: Full page image
            detection_result: Result from detect_title_block()

        Returns:
            Cropped PIL Image of title block
        """
        bbox = detection_result.get('bbox')
        if bbox:
            return image.crop(bbox)

        # Fallback: crop bottom-right 35% x 30%
        width, height = image.size
        return image.crop((int(width * 0.65), int(height * 0.70), width, height))

    def get_zone_crop(
        self,
        title_block_image: Image.Image,
        zone_name: str
    ) -> Optional[Tuple[Image.Image, str]]:
        """
        Get a cropped image of a specific zone within the title block.

        Args:
            title_block_image: Cropped title block image
            zone_name: Name of zone (e.g., 'sheet_identification')

        Returns:
            Tuple of (cropped image, text_orientation) or None if zone not found
        """
        if self._current_telemetry is None:
            return None

        zone = self._current_telemetry.zones.get(zone_name)
        if zone is None:
            return None

        width, height = title_block_image.size
        bbox = zone.bbox_percent.to_pixels(width, height)
        cropped = title_block_image.crop(bbox)

        # Rotate if needed
        if zone.text_orientation == 'vertical_90cw':
            cropped = cropped.rotate(-90, expand=True)
        elif zone.text_orientation == 'vertical_90ccw':
            cropped = cropped.rotate(90, expand=True)

        return (cropped, zone.text_orientation)

    def get_current_telemetry(self) -> Optional[TitleBlockTelemetry]:
        """Get the current telemetry (for debugging/testing)."""
        return self._current_telemetry

    def is_page_1_cover_sheet(self) -> bool:
        """Check if page 1 was identified as a cover sheet."""
        if self._current_telemetry is None:
            return False
        return self._current_telemetry.page_1_is_cover_sheet


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_region_visualization(
    image: Image.Image,
    detection_result: Dict[str, Any]
) -> Image.Image:
    """
    Create a visualization of detected regions on the image.

    Compatibility function matching region_detector.get_region_visualization().

    Args:
        image: Original image
        detection_result: Detection results from detect_title_block()

    Returns:
        PIL Image with region visualizations
    """
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        return image

    viz = image.copy()
    draw = ImageDraw.Draw(viz)

    # Draw main title block bbox
    bbox = detection_result.get('bbox')
    if bbox:
        draw.rectangle(bbox, outline='green', width=3)
        draw.text((bbox[0] + 5, bbox[1] + 5), 'TITLE_BLOCK', fill='green')

    # Draw zones if available
    zones = detection_result.get('zones', {})
    colors = ['red', 'blue', 'orange', 'purple', 'cyan']

    for i, (zone_name, zone_info) in enumerate(zones.items()):
        # Zone bbox is relative to title block, need to convert
        if bbox and 'bbox_relative' in zone_info:
            tb_width = bbox[2] - bbox[0]
            tb_height = bbox[3] - bbox[1]
            zb = zone_info['bbox_relative']
            zone_bbox = (
                bbox[0] + int(zb['x1'] * tb_width),
                bbox[1] + int(zb['y1'] * tb_height),
                bbox[0] + int(zb['x2'] * tb_width),
                bbox[1] + int(zb['y2'] * tb_height)
            )
            color = colors[i % len(colors)]
            draw.rectangle(zone_bbox, outline=color, width=2)
            draw.text((zone_bbox[0] + 3, zone_bbox[1] + 3), zone_name, fill=color)

    return viz
