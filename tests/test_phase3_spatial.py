"""
Blueprint Processor V4.2.1 - Phase 3 Tests
Tests for Spatial Zone Detection (vector PDFs only).
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.spatial_extractor import SpatialExtractor
from core.coordinates import (
    pixels_to_points,
    points_to_pixels,
    get_vertical_zone,
    get_title_zone_bbox,
)
from core.pdf_handler import PDFHandler
from core.extractor import Extractor


def test_coordinate_conversion():
    """Test pixel/point coordinate conversion."""
    print("=== Test: Coordinate Conversion ===")

    # Test pixels to points at 200 DPI
    # 200 pixels = 72 points (1 inch)
    pixels = [200, 200, 400, 400]
    points = pixels_to_points(pixels, dpi=200)
    print(f"  Pixels {pixels} -> Points {points}")

    expected = [72, 72, 144, 144]
    for p, e in zip(points, expected):
        assert abs(p - e) < 0.01, f"Expected {e}, got {p}"

    # Test points to pixels
    back_to_pixels = points_to_pixels(points, dpi=200)
    print(f"  Points {points} -> Pixels {back_to_pixels}")

    for p, o in zip(back_to_pixels, pixels):
        assert abs(p - o) < 0.01, f"Expected {o}, got {p}"

    print("PASSED\n")


def test_vertical_zones():
    """Test vertical zone detection."""
    print("=== Test: Vertical Zones ===")

    # Title block bbox: y ranges from 0 to 100
    bbox = (0, 0, 100, 100)

    # Top 15% = header
    zone = get_vertical_zone(7, bbox)
    print(f"  y=7 (7%) -> zone={zone}")
    assert zone == 'header', f"Expected 'header', got {zone}"

    # 15% to 65% = title
    zone = get_vertical_zone(40, bbox)
    print(f"  y=40 (40%) -> zone={zone}")
    assert zone == 'title', f"Expected 'title', got {zone}"

    # Bottom 35% = info
    zone = get_vertical_zone(80, bbox)
    print(f"  y=80 (80%) -> zone={zone}")
    assert zone == 'info', f"Expected 'info', got {zone}"

    print("PASSED\n")


def test_title_zone_bbox():
    """Test title zone bounding box calculation."""
    print("=== Test: Title Zone Bbox ===")

    # Full title block
    title_block = (0, 0, 200, 100)

    title_zone = get_title_zone_bbox(title_block)
    print(f"  Title block: {title_block}")
    print(f"  Title zone: {title_zone}")

    # Title zone should be 15% to 65% of vertical extent
    # y0 = 0 + (100 * 0.15) = 15
    # y1 = 0 + (100 * 0.65) = 65
    expected = (0, 15, 200, 65)
    for a, b in zip(title_zone, expected):
        assert abs(a - b) < 0.01, f"Expected {expected}, got {title_zone}"

    print("PASSED\n")


def test_label_detection():
    """Test that labels are correctly identified and skipped."""
    print("=== Test: Label Detection ===")

    extractor = SpatialExtractor()

    labels = [
        "PROJECT", "SHEET", "SCALE", "DATE", "DRAWN BY",
        "PROJECT:", "SHEET NO:", "SCALE:", "AS NOTED",
        "REV", "REVISION", "ISSUE", "CLIENT", "ARCHITECT",
    ]

    non_labels = [
        "FLOOR PLAN", "REFLECTED CEILING PLAN", "ELEVATIONS",
        "FIRST FLOOR", "DEMOLITION PLAN", "HVAC SCHEDULE",
    ]

    print("  Labels (should skip):")
    for label in labels:
        is_label = extractor._is_label(label.upper())
        print(f"    '{label}' -> is_label={is_label}")
        assert is_label, f"Expected '{label}' to be detected as label"

    print("  Non-labels (should keep):")
    for text in non_labels:
        is_label = extractor._is_label(text.upper())
        print(f"    '{text}' -> is_label={is_label}")
        assert not is_label, f"Expected '{text}' to NOT be detected as label"

    print("PASSED\n")


def test_with_vector_sample_pdf():
    """Integration test with actual vector_sample.pdf."""
    sample_path = Path(__file__).parent.parent / 'test_data' / 'vector_sample.pdf'

    print("=== Test: vector_sample.pdf Spatial Extraction ===")

    if not sample_path.exists():
        print(f"  Skipping - {sample_path} not found")
        return

    extractor = Extractor()
    spatial = SpatialExtractor()

    with PDFHandler(sample_path) as pdf:
        print(f"  PDF has {pdf.page_count} pages")

        # Check which pages are vector
        vector_pages = []
        for page_num in range(min(pdf.page_count, 10)):  # Check first 10 pages
            page = pdf.doc[page_num]
            is_vector = spatial.is_vector_page(page)
            text_len = len(page.get_text().strip())
            if is_vector:
                vector_pages.append(page_num + 1)
            print(f"    Page {page_num + 1}: {'VECTOR' if is_vector else 'SCANNED'} ({text_len} chars)")

        print(f"\n  Vector pages: {vector_pages}")

        if not vector_pages:
            print("  No vector pages found - skipping spatial tests")
            return

        # Test spatial extraction on first few vector pages
        print("\n  Spatial extraction results:")
        for page_num in vector_pages[:5]:
            page = pdf.doc[page_num - 1]
            text = page.get_text()

            # Extract with spatial analysis
            result = extractor.extract_fields(
                text=text,
                page_number=page_num,
                page=page,
            )

            print(f"\n    Page {page_num}:")
            print(f"      Sheet: {result['sheet_number']}")
            print(f"      Title: {result['sheet_title']}")
            print(f"      Method: {result['title_method']}")
            print(f"      Confidence: {result['title_confidence']:.2f}")

    print("\nPASSED\n")


def test_spatial_scoring():
    """Test candidate scoring logic."""
    print("=== Test: Spatial Scoring ===")

    extractor = SpatialExtractor()

    # Mock title block bbox
    title_block = (0, 0, 612, 792)  # Standard letter page in points
    title_zone = get_title_zone_bbox(title_block)

    # Test scoring a good candidate (in title zone, large font, keyword)
    result = extractor._score_candidate(
        text="FLOOR PLAN",
        bbox=(200, 300, 400, 330),  # In title zone (15%-65% = 118-515)
        font_size=24.0,
        title_block_bbox=title_block,
        title_zone_bbox=title_zone,
    )
    print(f"  'FLOOR PLAN' (24pt, title zone):")
    print(f"    Score: {result['score']}")
    print(f"    Reasons: {result['reasons']}")
    assert result['score'] > 50, f"Expected high score, got {result['score']}"

    # Test scoring a label (should be disqualified)
    result = extractor._score_candidate(
        text="PROJECT:",
        bbox=(200, 700, 300, 720),
        font_size=12.0,
        title_block_bbox=title_block,
        title_zone_bbox=title_zone,
    )
    print(f"\n  'PROJECT:' (label):")
    print(f"    Score: {result['score']}")
    print(f"    Reasons: {result['reasons']}")
    assert result['score'] == 0, f"Expected 0 for label, got {result['score']}"

    # Test scoring text in info zone (lower score)
    result = extractor._score_candidate(
        text="Some text",
        bbox=(200, 700, 300, 720),  # In info zone (bottom 35%)
        font_size=10.0,
        title_block_bbox=title_block,
        title_zone_bbox=title_zone,
    )
    print(f"\n  'Some text' (10pt, info zone):")
    print(f"    Score: {result['score']}")
    print(f"    Reasons: {result['reasons']}")
    assert result['score'] < 20, f"Expected low score for info zone, got {result['score']}"

    print("PASSED\n")


def run_all_tests():
    """Run all Phase 3 tests."""
    print("=" * 60)
    print("PHASE 3 TESTS: Spatial Zone Detection")
    print("=" * 60 + "\n")

    try:
        test_coordinate_conversion()
        test_vertical_zones()
        test_title_zone_bbox()
        test_label_detection()
        test_spatial_scoring()
        test_with_vector_sample_pdf()

        print("=" * 60)
        print("=== ALL PHASE 3 TESTS PASSED ===")
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
