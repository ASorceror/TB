"""
Blueprint Processor V4.2.1 - Phase 4 Tests
Tests for Vision API Integration.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.vision_extractor import VisionExtractor, MAX_CALLS_PER_PDF, MAX_CALLS_PER_MINUTE


def test_vision_extractor_init_no_key():
    """Test VisionExtractor initializes without API key."""
    print("=== Test: VisionExtractor Init (No API Key) ===")

    # Clear any existing key
    original_key = os.environ.get('ANTHROPIC_API_KEY')
    if 'ANTHROPIC_API_KEY' in os.environ:
        del os.environ['ANTHROPIC_API_KEY']

    try:
        extractor = VisionExtractor()
        print(f"  API available: {extractor.is_available()}")
        assert not extractor.is_available(), "Should not be available without API key"
        print("PASSED\n")
    finally:
        # Restore key if it existed
        if original_key:
            os.environ['ANTHROPIC_API_KEY'] = original_key


def test_rate_limit_constants():
    """Test rate limit constants are defined correctly."""
    print("=== Test: Rate Limit Constants ===")

    print(f"  MAX_CALLS_PER_PDF: {MAX_CALLS_PER_PDF}")
    print(f"  MAX_CALLS_PER_MINUTE: {MAX_CALLS_PER_MINUTE}")

    assert MAX_CALLS_PER_PDF == 50, f"Expected 50, got {MAX_CALLS_PER_PDF}"
    assert MAX_CALLS_PER_MINUTE == 10, f"Expected 10, got {MAX_CALLS_PER_MINUTE}"

    print("PASSED\n")


def test_per_pdf_counter():
    """Test per-PDF call counter."""
    print("=== Test: Per-PDF Counter ===")

    extractor = VisionExtractor()
    extractor._api_available = True  # Simulate available API

    print(f"  Initial calls: {extractor._calls_this_pdf}")
    assert extractor._calls_this_pdf == 0, "Should start at 0"

    # Simulate some calls
    extractor._record_call()
    extractor._record_call()
    extractor._record_call()
    print(f"  After 3 calls: {extractor._calls_this_pdf}")
    assert extractor._calls_this_pdf == 3, "Should be 3"

    # Reset
    extractor.reset_pdf_counter()
    print(f"  After reset: {extractor._calls_this_pdf}")
    assert extractor._calls_this_pdf == 0, "Should be 0 after reset"

    print("PASSED\n")


def test_per_pdf_limit():
    """Test per-PDF limit enforcement."""
    print("=== Test: Per-PDF Limit ===")

    extractor = VisionExtractor()
    extractor._api_available = True

    # Set calls near limit
    extractor._calls_this_pdf = MAX_CALLS_PER_PDF - 1
    print(f"  Calls at {extractor._calls_this_pdf} (limit is {MAX_CALLS_PER_PDF})")

    assert extractor._check_rate_limit(), "Should allow one more call"

    extractor._calls_this_pdf = MAX_CALLS_PER_PDF
    print(f"  Calls at {extractor._calls_this_pdf} (at limit)")

    assert not extractor._check_rate_limit(), "Should not allow calls at limit"

    print("PASSED\n")


def test_image_hash_caching():
    """Test image hash caching."""
    print("=== Test: Image Hash Caching ===")

    try:
        from PIL import Image
    except ImportError:
        print("  Skipping - PIL not installed")
        return

    extractor = VisionExtractor()

    # Create a test image
    img = Image.new('RGB', (100, 100), color='red')

    # Get hash
    hash1 = extractor._get_image_hash(img)
    hash2 = extractor._get_image_hash(img)

    print(f"  Hash 1: {hash1}")
    print(f"  Hash 2: {hash2}")

    assert hash1 == hash2, "Same image should produce same hash"
    assert len(hash1) == 16, f"Hash should be 16 chars, got {len(hash1)}"

    # Different image should have different hash
    img2 = Image.new('RGB', (100, 100), color='blue')
    hash3 = extractor._get_image_hash(img2)
    print(f"  Hash 3 (different image): {hash3}")

    assert hash3 != hash1, "Different images should have different hashes"

    print("PASSED\n")


def test_cache_hit():
    """Test cache hit returns cached result."""
    print("=== Test: Cache Hit ===")

    try:
        from PIL import Image
    except ImportError:
        print("  Skipping - PIL not installed")
        return

    extractor = VisionExtractor()
    extractor._api_available = True

    # Create test image and add to cache
    img = Image.new('RGB', (100, 100), color='red')
    image_hash = extractor._get_image_hash(img)
    extractor._cache[image_hash] = "FLOOR PLAN"

    # Extract should hit cache
    result = extractor.extract_title(img, use_cache=True)

    print(f"  Result: {result}")
    assert result['title'] == "FLOOR PLAN", f"Expected 'FLOOR PLAN', got {result['title']}"
    assert result['cached'] is True, "Should be marked as cached"
    assert result['method'] == "vision_api", f"Expected 'vision_api', got {result['method']}"

    print("PASSED\n")


def test_cache_bypass():
    """Test cache can be bypassed."""
    print("=== Test: Cache Bypass ===")

    try:
        from PIL import Image
    except ImportError:
        print("  Skipping - PIL not installed")
        return

    extractor = VisionExtractor()
    extractor._api_available = False  # No API so extract will fail

    # Create test image and add to cache
    img = Image.new('RGB', (100, 100), color='red')
    image_hash = extractor._get_image_hash(img)
    extractor._cache[image_hash] = "CACHED TITLE"

    # With cache=True, should return cached value
    result1 = extractor.extract_title(img, use_cache=True)
    print(f"  With cache: {result1['title']}, cached={result1['cached']}")
    assert result1['title'] == "CACHED TITLE", "Should return cached value"

    # With cache=False and no API, should fail (API not available)
    extractor._api_available = False
    result2 = extractor.extract_title(img, use_cache=False)
    print(f"  Without cache (no API): {result2['title']}, error={result2.get('error')}")
    assert result2['title'] is None, "Should fail without cache and no API"

    print("PASSED\n")


def test_unknown_response():
    """Test UNKNOWN response handling."""
    print("=== Test: UNKNOWN Response ===")

    try:
        from PIL import Image
    except ImportError:
        print("  Skipping - PIL not installed")
        return

    extractor = VisionExtractor()
    extractor._api_available = True

    # Add UNKNOWN to cache
    img = Image.new('RGB', (100, 100), color='red')
    image_hash = extractor._get_image_hash(img)
    extractor._cache[image_hash] = "UNKNOWN"

    result = extractor.extract_title(img, use_cache=True)

    print(f"  Result: {result}")
    assert result['title'] is None, "UNKNOWN should return None title"
    assert result['confidence'] == 0.0, "UNKNOWN should have 0 confidence"

    print("PASSED\n")


def test_get_stats():
    """Test stats reporting."""
    print("=== Test: Get Stats ===")

    extractor = VisionExtractor()
    extractor._calls_this_pdf = 5
    extractor._cache['hash1'] = 'title1'
    extractor._cache['hash2'] = 'title2'

    stats = extractor.get_stats()

    print(f"  Stats: {stats}")
    assert stats['calls_this_pdf'] == 5
    assert stats['cache_size'] == 2
    assert stats['max_calls_per_pdf'] == MAX_CALLS_PER_PDF

    print("PASSED\n")


def test_clear_cache():
    """Test cache clearing."""
    print("=== Test: Clear Cache ===")

    extractor = VisionExtractor()
    extractor._cache['hash1'] = 'title1'
    extractor._cache['hash2'] = 'title2'

    print(f"  Cache size before: {len(extractor._cache)}")
    assert len(extractor._cache) == 2

    extractor.clear_cache()

    print(f"  Cache size after: {len(extractor._cache)}")
    assert len(extractor._cache) == 0

    print("PASSED\n")


def test_rate_limited_flag():
    """Test rate limited flag disables API."""
    print("=== Test: Rate Limited Flag ===")

    extractor = VisionExtractor()
    extractor._api_available = True
    extractor._rate_limited = False

    print(f"  Initial is_available: {extractor.is_available()}")
    assert extractor.is_available(), "Should be available initially"

    extractor._rate_limited = True
    print(f"  After rate limit: {extractor.is_available()}")
    assert not extractor.is_available(), "Should not be available after rate limit"

    print("PASSED\n")


@patch('core.vision_extractor.VisionExtractor._call_vision_api')
def test_api_call_with_mock(mock_api):
    """Test API call with mocked response."""
    print("=== Test: API Call (Mocked) ===")

    try:
        from PIL import Image
    except ImportError:
        print("  Skipping - PIL not installed")
        return

    # Setup mock
    mock_api.return_value = "REFLECTED CEILING PLAN"

    extractor = VisionExtractor()
    extractor._api_available = True
    extractor._client = Mock()  # Mock client

    img = Image.new('RGB', (100, 100), color='white')

    result = extractor.extract_title(img, use_cache=False)

    print(f"  Result: {result}")
    assert result['title'] == "REFLECTED CEILING PLAN"
    assert result['method'] == "vision_api"
    assert result['confidence'] == 0.90
    assert result['cached'] is False

    print("PASSED\n")


def test_extractor_integration():
    """Test VisionExtractor integration with main Extractor."""
    print("=== Test: Extractor Integration ===")

    from core.extractor import Extractor

    extractor = Extractor()

    # Check vision extractor is initialized
    assert hasattr(extractor, '_vision_extractor'), "Should have _vision_extractor"
    assert isinstance(extractor._vision_extractor, VisionExtractor), "Should be VisionExtractor instance"

    # Test reset
    extractor._vision_extractor._calls_this_pdf = 10
    extractor.reset_for_new_pdf()
    assert extractor._vision_extractor._calls_this_pdf == 0, "Should reset counter"

    print("  VisionExtractor integrated correctly")
    print("PASSED\n")


def run_all_tests():
    """Run all Phase 4 tests."""
    print("=" * 60)
    print("PHASE 4 TESTS: Vision API Integration")
    print("=" * 60 + "\n")

    try:
        test_vision_extractor_init_no_key()
        test_rate_limit_constants()
        test_per_pdf_counter()
        test_per_pdf_limit()
        test_image_hash_caching()
        test_cache_hit()
        test_cache_bypass()
        test_unknown_response()
        test_get_stats()
        test_clear_cache()
        test_rate_limited_flag()
        test_api_call_with_mock()
        test_extractor_integration()

        print("=" * 60)
        print("=== ALL PHASE 4 TESTS PASSED ===")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n!!! TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n!!! ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
