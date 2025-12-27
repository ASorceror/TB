"""
Test PaddleOCR Layout Detection for Title Block Detection

PaddleOCR has PP-Structure which can detect:
- Text regions
- Tables
- Figures
- Title regions

Let's see if it can identify title blocks in blueprints.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from PIL import Image, ImageDraw
import numpy as np


def test_paddleocr_layout():
    """Test PaddleOCR layout detection on sample blueprints."""

    output_dir = Path(r"C:\tb\blueprint_processor\output\paddleocr_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    # Test PDFs
    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "DQ Matteson A Permit Drawings.pdf",
    ]

    print("="*70)
    print("PaddleOCR Layout Detection Test")
    print("="*70)

    # Import PaddleOCR
    try:
        from paddleocr import PPStructure
        print("PaddleOCR imported successfully")
    except ImportError as e:
        print(f"ERROR: Could not import PaddleOCR: {e}")
        return

    # Initialize PP-Structure for layout detection only (no OCR)
    print("\nInitializing PP-Structure (layout detection only)...")
    try:
        # layout=True enables layout detection
        # table=False disables table structure recognition (faster)
        # ocr=False disables text recognition (we just want regions)
        engine = PPStructure(layout=True, table=False, ocr=False, show_log=False)
        print("PP-Structure initialized")
    except Exception as e:
        print(f"ERROR initializing PP-Structure: {e}")
        # Try alternative initialization
        try:
            from paddleocr import PaddleOCR
            engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            print("Fallback to PaddleOCR (basic) initialized")
        except Exception as e2:
            print(f"ERROR with fallback: {e2}")
            return

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if not pdf_path.exists():
            print(f"\nSKIP: {pdf_name}")
            continue

        print(f"\n{'='*60}")
        print(f"PDF: {pdf_name}")

        try:
            with PDFHandler(pdf_path) as handler:
                # Get page 2 (skip cover)
                page_num = 2 if handler.page_count > 1 else 1
                page_img = handler.get_page_image(page_num - 1, dpi=150)

                # Save original
                safe_name = pdf_name.replace(' ', '_').replace('.pdf', '')[:30]
                orig_path = output_dir / f"{safe_name}_original.png"
                page_img.save(orig_path)

                # Convert to numpy array for PaddleOCR
                img_array = np.array(page_img)

                print(f"  Page {page_num}: {page_img.size}")
                print("  Running layout detection...")

                # Run layout detection
                result = engine(img_array)

                print(f"  Found {len(result)} regions")

                # Analyze results
                width, height = page_img.size
                result_img = page_img.convert('RGB')
                draw = ImageDraw.Draw(result_img)

                # Colors for different region types
                colors = {
                    'text': 'blue',
                    'title': 'green',
                    'table': 'orange',
                    'figure': 'purple',
                    'list': 'cyan',
                    'header': 'magenta',
                    'footer': 'brown',
                }

                rightmost_regions = []

                for i, region in enumerate(result):
                    # PP-Structure returns dict with 'type' and 'bbox' keys
                    if isinstance(region, dict):
                        region_type = region.get('type', 'unknown')
                        bbox = region.get('bbox', [])

                        if bbox and len(bbox) >= 4:
                            x1, y1, x2, y2 = bbox[:4]

                            # Calculate position as percentage
                            x1_pct = x1 / width
                            x2_pct = x2 / width

                            color = colors.get(region_type, 'gray')
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw.text((x1, y1 - 15), f"{region_type}", fill=color)

                            print(f"    Region {i}: {region_type} at x={x1_pct:.2f}-{x2_pct:.2f}")

                            # Track rightmost regions (potential title block)
                            if x2_pct > 0.85:
                                rightmost_regions.append({
                                    'type': region_type,
                                    'bbox': bbox,
                                    'x1_pct': x1_pct,
                                    'x2_pct': x2_pct
                                })
                    else:
                        # Might be a list format
                        print(f"    Region {i}: {type(region)} - {str(region)[:100]}")

                # Save annotated image
                result_path = output_dir / f"{safe_name}_layout.png"
                result_img.save(result_path)
                print(f"  Saved: {result_path}")

                # Report rightmost regions
                if rightmost_regions:
                    print(f"\n  Rightmost regions (potential title block):")
                    for r in rightmost_regions:
                        print(f"    {r['type']}: x1={r['x1_pct']:.3f}, x2={r['x2_pct']:.3f}")

                    # Find leftmost x1 among rightmost regions
                    leftmost_x1 = min(r['x1_pct'] for r in rightmost_regions)
                    print(f"  -> Suggested title block x1: {leftmost_x1:.3f}")
                else:
                    print("  No regions found in rightmost 15% of page")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Output saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    test_paddleocr_layout()
