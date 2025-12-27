"""
Blueprint Processor V4.2.1 - Vision API Extractor
Uses Claude Vision to extract sheet titles from title block images.
"""

import os
import io
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import TITLE_CONFIDENCE

logger = logging.getLogger(__name__)

# Rate limiting constants
MAX_CALLS_PER_PDF = 2000  # Effectively unlimited per PDF
MAX_CALLS_PER_MINUTE = 10
RATE_LIMIT_BACKOFF_SECONDS = 60
RATE_LIMIT_EXTENDED_BACKOFF_SECONDS = 120

# Vision API prompt for title extraction
EXTRACTION_PROMPT = """Look at this title block image from a construction/architectural blueprint.

Extract ONLY the sheet title - this is the text that describes what the drawing shows (like "Floor Plan", "Reflected Ceiling Plan", "Elevations", "Demolition Plan", etc.).

The sheet title is usually:
- Larger text in the middle portion of the title block
- Describes the content of the drawing
- NOT the project number, sheet number, date, scale, company name, or address

DO NOT return:
- Project numbers (like P27142, 2024-001)
- Sheet numbers (like A1.0, S-201)
- Company names or addresses
- Dates or scales
- Legal text or disclaimers

If you cannot clearly identify a sheet title, respond with exactly: UNKNOWN

Respond with ONLY the sheet title text, nothing else. No explanations, no prefixes like "Title:" - just the title itself."""

# Vision API prompt for sheet number extraction
SHEET_NUMBER_PROMPT = """Look at this title block image from a construction/architectural blueprint.

Extract ONLY the sheet number - this is a code that identifies the drawing (like "A-1", "A1.0", "S-201", "M-1.1", "E-2", "T-1", "D-1", etc.).

The sheet number is usually:
- Located at the bottom of the title block in a box labeled "Sheet" or "Sheet No."
- A short alphanumeric code with letters and numbers
- Often includes a prefix (A for Architectural, S for Structural, M for Mechanical, E for Electrical, P for Plumbing, etc.)
- May include dots, dashes, or decimals (like A-2.1, M1.1, E-4)

DO NOT return:
- The sheet title (like "Floor Plan", "Elevations")
- Project numbers or job numbers
- Company names or addresses
- Dates or scales
- Revision numbers

If you cannot clearly identify a sheet number, respond with exactly: UNKNOWN

Respond with ONLY the sheet number, nothing else. No explanations, no prefixes - just the sheet number itself (e.g., "A-1" or "M-2.1")."""


class VisionExtractor:
    """
    Extracts sheet titles from title block images using Claude Vision API.

    Features:
    - Rate limiting: max 10 calls per minute (per-PDF limit removed)
    - Backoff on 429 errors: wait 60s, retry once
    - Caching by image hash to avoid duplicate API calls
    """

    def __init__(self):
        """Initialize the VisionExtractor."""
        self._client = None
        self._api_available = False
        self._cache: Dict[str, str] = {}  # image_hash -> title

        # Rate limiting state
        self._calls_this_pdf = 0
        self._call_timestamps: list = []  # timestamps of recent calls
        self._rate_limited = False  # True if we've hit repeated 429s
        limit_override = os.environ.get('VISION_MAX_CALLS_PER_PDF')
        if limit_override is not None:
            try:
                parsed_limit = int(limit_override)
            except ValueError:
                parsed_limit = MAX_CALLS_PER_PDF
        else:
            parsed_limit = MAX_CALLS_PER_PDF
        self._per_pdf_limit = None if parsed_limit <= 0 else parsed_limit

        # Try to initialize the API client
        self._init_client()

    def _init_client(self):
        """Initialize the Anthropic API client."""
        api_key = os.environ.get('ANTHROPIC_API_KEY')

        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set - Vision API disabled")
            self._api_available = False
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            self._api_available = True
            logger.info("Vision API initialized successfully")
        except ImportError:
            logger.warning("anthropic package not installed - Vision API disabled")
            logger.warning("Install with: pip install anthropic")
            self._api_available = False
        except Exception as e:
            logger.error(f"Failed to initialize Vision API: {e}")
            self._api_available = False

    def is_available(self) -> bool:
        """Check if Vision API is available."""
        return self._api_available and not self._rate_limited

    def reset_pdf_counter(self):
        """Reset the per-PDF call counter. Call this when starting a new PDF."""
        self._calls_this_pdf = 0
        logger.debug("Vision API per-PDF counter reset")

    def _get_image_hash(self, image: Image.Image) -> str:
        """
        Calculate a hash of the image for caching.

        Args:
            image: PIL Image

        Returns:
            MD5 hash string (first 16 chars)
        """
        # Convert to bytes and hash
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        return hashlib.md5(img_data).hexdigest()[:16]

    def _check_rate_limit(self) -> bool:
        """
        Check if we can make another API call.

        Returns:
            True if we can proceed, False if rate limited
        """
        # Check if we've hit too many 429s
        if self._rate_limited:
            logger.warning("Vision API disabled due to repeated rate limits")
            return False

        # Per-PDF limit removed - now effectively unlimited (MAX_CALLS_PER_PDF = 2000)

        # Check per-minute limit
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._call_timestamps = [ts for ts in self._call_timestamps if now - ts < 60]

        if len(self._call_timestamps) >= MAX_CALLS_PER_MINUTE:
            # Need to wait
            oldest = min(self._call_timestamps)
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s before next Vision API call")
                time.sleep(wait_time)
                # Clean up old timestamps after waiting
                now = time.time()
                self._call_timestamps = [ts for ts in self._call_timestamps if now - ts < 60]

        return True

    def _record_call(self):
        """Record that we made an API call."""
        self._calls_this_pdf += 1
        self._call_timestamps.append(time.time())

    def extract_title(
        self,
        image: Image.Image,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Extract sheet title from a title block image using Vision API.

        Args:
            image: PIL Image of the title block
            use_cache: Whether to use cached results (default True)

        Returns:
            Dict with keys:
                - title: extracted title string or None
                - confidence: float 0.0-1.0 (0.90 for vision)
                - method: 'vision_api' if successful
                - cached: True if result was from cache
                - error: error message if failed
        """
        # Check cache first (works even if API is unavailable)
        image_hash = self._get_image_hash(image)
        if use_cache and image_hash in self._cache:
            cached_title = self._cache[image_hash]
            logger.debug(f"Cache hit for image {image_hash}: {cached_title}")
            return {
                "title": cached_title if cached_title != "UNKNOWN" else None,
                "confidence": TITLE_CONFIDENCE.get('VISION_API', 0.90) if cached_title != "UNKNOWN" else 0.0,
                "method": "vision_api" if cached_title != "UNKNOWN" else None,
                "cached": True,
                "error": None
            }

        # Check API availability (after cache check)
        if not self._api_available:
            return {
                "title": None,
                "confidence": 0.0,
                "method": None,
                "cached": False,
                "error": "Vision API not available"
            }

        # Check rate limits
        if not self._check_rate_limit():
            return {
                "title": None,
                "confidence": 0.0,
                "method": None,
                "cached": False,
                "error": "Rate limited"
            }

        # Make API call
        try:
            result = self._call_vision_api(image)
            self._record_call()

            # Cache the result
            self._cache[image_hash] = result

            if result == "UNKNOWN" or not result:
                return {
                    "title": None,
                    "confidence": 0.0,
                    "method": None,
                    "cached": False,
                    "error": None
                }

            return {
                "title": result,
                "confidence": TITLE_CONFIDENCE.get('VISION_API', 0.90),
                "method": "vision_api",
                "cached": False,
                "error": None
            }

        except Exception as e:
            error_msg = str(e)

            # Check for rate limit error
            if "429" in error_msg or "rate" in error_msg.lower():
                logger.warning(f"Vision API rate limited: {e}")
                return self._handle_rate_limit_error(image, image_hash)

            logger.error(f"Vision API error: {e}")
            return {
                "title": None,
                "confidence": 0.0,
                "method": None,
                "cached": False,
                "error": error_msg
            }

    def extract_sheet_number(
        self,
        image: Image.Image,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Extract sheet number from a title block image using Vision API.

        This is particularly useful for handwritten/sketch fonts that Tesseract cannot read.

        Args:
            image: PIL Image of the title block
            use_cache: Whether to use cached results (default True)

        Returns:
            Dict with keys:
                - sheet_number: extracted sheet number string or None
                - confidence: float 0.0-1.0 (0.90 for vision)
                - method: 'vision_api' if successful
                - cached: True if result was from cache
                - error: error message if failed
        """
        # Use a different cache prefix for sheet numbers
        image_hash = "sn_" + self._get_image_hash(image)
        if use_cache and image_hash in self._cache:
            cached_sn = self._cache[image_hash]
            logger.debug(f"Cache hit for sheet number {image_hash}: {cached_sn}")
            return {
                "sheet_number": cached_sn if cached_sn != "UNKNOWN" else None,
                "confidence": 0.90 if cached_sn != "UNKNOWN" else 0.0,
                "method": "vision_api" if cached_sn != "UNKNOWN" else None,
                "cached": True,
                "error": None
            }

        # Check API availability (after cache check)
        if not self._api_available:
            return {
                "sheet_number": None,
                "confidence": 0.0,
                "method": None,
                "cached": False,
                "error": "Vision API not available"
            }

        # Check rate limits
        if not self._check_rate_limit():
            return {
                "sheet_number": None,
                "confidence": 0.0,
                "method": None,
                "cached": False,
                "error": "Rate limited"
            }

        # Make API call
        try:
            result = self._call_vision_api_for_sheet_number(image)
            self._record_call()

            # Cache the result
            self._cache[image_hash] = result

            if result == "UNKNOWN" or not result:
                return {
                    "sheet_number": None,
                    "confidence": 0.0,
                    "method": None,
                    "cached": False,
                    "error": None
                }

            return {
                "sheet_number": result,
                "confidence": 0.90,
                "method": "vision_api",
                "cached": False,
                "error": None
            }

        except Exception as e:
            error_msg = str(e)

            # Check for rate limit error
            if "429" in error_msg or "rate" in error_msg.lower():
                logger.warning(f"Vision API rate limited: {e}")
                # For sheet numbers, don't retry on rate limit - just return error
                return {
                    "sheet_number": None,
                    "confidence": 0.0,
                    "method": None,
                    "cached": False,
                    "error": f"Rate limited: {error_msg}"
                }

            logger.error(f"Vision API error extracting sheet number: {e}")
            return {
                "sheet_number": None,
                "confidence": 0.0,
                "method": None,
                "cached": False,
                "error": error_msg
            }

    def _call_vision_api_for_sheet_number(self, image: Image.Image) -> str:
        """
        Make the Vision API call to extract sheet number.

        Args:
            image: PIL Image

        Returns:
            Extracted sheet number string or "UNKNOWN"
        """
        import base64

        # Resize image if needed to stay under API limits (8000 pixels max)
        image = self._resize_image_if_needed(image)

        # Convert PIL Image to base64
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_data = base64.standard_b64encode(img_bytes.getvalue()).decode('utf-8')

        logger.debug(f"Calling Vision API for sheet number (call #{self._calls_this_pdf + 1} this PDF)")

        message = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_data
                        }
                    },
                    {
                        "type": "text",
                        "text": SHEET_NUMBER_PROMPT
                    }
                ]
            }]
        )

        # Extract response text
        response_text = message.content[0].text.strip()
        logger.debug(f"Vision API sheet number response: {response_text}")

        return response_text

    def _resize_image_if_needed(self, image: Image.Image, max_size: int = 7500) -> Image.Image:
        """
        Resize image if any dimension exceeds max_size.

        Args:
            image: PIL Image
            max_size: Maximum dimension in pixels (default 7500 for safety margin under 8000)

        Returns:
            Resized PIL Image or original if within limits
        """
        width, height = image.size
        if width <= max_size and height <= max_size:
            return image

        # Calculate scale factor
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        logger.debug(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _call_vision_api(self, image: Image.Image) -> str:
        """
        Make the actual Vision API call.

        Args:
            image: PIL Image

        Returns:
            Extracted title string or "UNKNOWN"
        """
        import base64

        # Resize image if needed to stay under API limits (8000 pixels max)
        image = self._resize_image_if_needed(image)

        # Convert PIL Image to base64
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_data = base64.standard_b64encode(img_bytes.getvalue()).decode('utf-8')

        logger.debug(f"Calling Vision API (call #{self._calls_this_pdf + 1} this PDF)")

        message = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_data
                        }
                    },
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT
                    }
                ]
            }]
        )

        # Extract response text
        response_text = message.content[0].text.strip()
        logger.debug(f"Vision API response: {response_text}")

        return response_text

    def _handle_rate_limit_error(
        self,
        image: Image.Image,
        image_hash: str
    ) -> Dict[str, Any]:
        """
        Handle a 429 rate limit error with backoff and retry.

        Args:
            image: PIL Image to retry
            image_hash: Hash of the image for caching

        Returns:
            Result dict
        """
        logger.info(f"Rate limited - waiting {RATE_LIMIT_BACKOFF_SECONDS}s before retry")
        time.sleep(RATE_LIMIT_BACKOFF_SECONDS)

        try:
            result = self._call_vision_api(image)
            self._record_call()

            # Cache the result
            self._cache[image_hash] = result

            if result == "UNKNOWN" or not result:
                return {
                    "title": None,
                    "confidence": 0.0,
                    "method": None,
                    "cached": False,
                    "error": None
                }

            return {
                "title": result,
                "confidence": TITLE_CONFIDENCE.get('VISION_API', 0.90),
                "method": "vision_api",
                "cached": False,
                "error": None
            }

        except Exception as e:
            error_msg = str(e)

            # Second rate limit - disable vision for this session
            if "429" in error_msg or "rate" in error_msg.lower():
                logger.warning(f"Repeated rate limit - disabling Vision API for this session")
                logger.info(f"Waiting {RATE_LIMIT_EXTENDED_BACKOFF_SECONDS}s before continuing")
                time.sleep(RATE_LIMIT_EXTENDED_BACKOFF_SECONDS)
                self._rate_limited = True

            return {
                "title": None,
                "confidence": 0.0,
                "method": None,
                "cached": False,
                "error": f"Rate limited after retry: {error_msg}"
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about Vision API usage."""
        return {
            "available": self._api_available,
            "rate_limited": self._rate_limited,
            "calls_this_pdf": self._calls_this_pdf,
            "max_calls_per_pdf": MAX_CALLS_PER_PDF,
            "cache_size": len(self._cache),
            "recent_calls": len(self._call_timestamps),
        }

    def clear_cache(self):
        """Clear the image cache."""
        self._cache.clear()
        logger.debug("Vision API cache cleared")
