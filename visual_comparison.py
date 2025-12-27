"""
Visual Comparison - Show detected boundaries on actual pages

Create side-by-side images showing:
- Original page with all three detection lines drawn
- Color-coded: CV (blue), Hybrid (green), AI (red)
"""

import sys
import os
import json
import base64
import io
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler


def cv_transition_detection(pdf_handler, sample_pages, dpi=100):
    """Pure CV approach."""
    edge_images = []
    target_size = None

    for page_num in sample_pages:
        img = pdf_handler.get_page_image(page_num - 1, dpi=dpi)
        if target_size is None:
            target_size = img.size
        elif img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        binary = (np.array(edges) > 25).astype(np.uint8) * 255
        edge_images.append(binary)

    stack = np.stack(edge_images, axis=0)
    common_edges = np.all(stack > 128, axis=0).astype(np.uint8) * 255

    height, width = common_edges.shape
    col_edge_count = np.sum(common_edges > 128, axis=0)

    from scipy.ndimage import uniform_filter1d
    col_smoothed = uniform_filter1d(col_edge_count.astype(float), size=10)

    left_portion = col_smoothed[:int(width * 0.5)]
    noise_level = np.percentile(left_portion, 95)
    threshold = max(noise_level * 2, 20)

    search_start = int(width * 0.60)
    search_end = int(width * 0.97)
    window_size = 20

    for i in range(search_start, search_end - window_size):
        window = col_smoothed[i:i + window_size]
        if np.all(window > threshold):
            return i / width
    return None


def ai_vision_detection(pdf_handler, sample_pages, dpi=100):
    """Improved AI Vision with explicit boundary finding."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Use just 3 pages for speed
        pages_to_use = sample_pages[:3]

        content = []
        for page_num in pages_to_use:
            img = pdf_handler.get_page_image(page_num - 1, dpi=dpi)

            # Resize for API
            max_size = 1000
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_data = base64.standard_b64encode(img_bytes.getvalue()).decode('utf-8')

            content.append({"type": "text", "text": f"PAGE {page_num}:"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_data}
            })

        prompt = """Look at these blueprint pages.

Find the LEFT EDGE of the TITLE BLOCK.

The title block is the rectangular information panel on the RIGHT SIDE containing:
- Firm/company name and logo
- Project name and address
- Sheet number (like A-1, S-1, etc.)
- Date, scale, drawn by info

I need the X position of the LEFT BOUNDARY of this title block as a decimal:
- 0.0 = left edge of page
- 1.0 = right edge of page

Typical values are between 0.80 and 0.92.

IMPORTANT: Look at the ACTUAL boundary line separating drawings from the title block.
Do NOT just guess 0.85 - measure it from what you see.

Reply with ONLY a number like: 0.87"""

        content.append({"type": "text", "text": prompt})

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": content}]
        )

        response_text = message.content[0].text.strip()

        # Extract number
        import re
        match = re.search(r'0\.\d+', response_text)
        if match:
            return float(match.group())

    except Exception as e:
        print(f"    AI error: {e}")

    return None


def create_visual_comparison():
    """Create visual comparison for select PDFs."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")
    output_dir = Path(r"C:\tb\blueprint_processor\output\visual_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test on a few representative PDFs
    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
        "DQ Matteson A Permit Drawings.pdf",
    ]

    print("="*70)
    print("VISUAL COMPARISON")
    print("="*70)

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if not pdf_path.exists():
            print(f"\nSKIP: {pdf_name}")
            continue

        print(f"\n{pdf_name}")

        try:
            with PDFHandler(pdf_path) as handler:
                total_pages = handler.page_count
                sample_pages = [2, 4, 6] if total_pages >= 6 else [2, 3, 4][:total_pages]

                # Get detections
                cv_x1 = cv_transition_detection(handler, sample_pages)
                ai_x1 = ai_vision_detection(handler, sample_pages)

                print(f"  CV: {cv_x1:.3f}" if cv_x1 else "  CV: FAILED")
                print(f"  AI: {ai_x1:.3f}" if ai_x1 else "  AI: FAILED")

                # Render page 2 for visualization
                page_img = handler.get_page_image(1, dpi=150)
                width, height = page_img.size

                # Draw on image
                result_img = page_img.convert('RGB')
                draw = ImageDraw.Draw(result_img)

                # Draw CV line (blue)
                if cv_x1:
                    x = int(cv_x1 * width)
                    draw.line([(x, 0), (x, height)], fill='blue', width=4)
                    draw.text((x + 5, 20), f"CV: {cv_x1:.3f}", fill='blue')

                # Draw AI line (red)
                if ai_x1:
                    x = int(ai_x1 * width)
                    draw.line([(x, 0), (x, height)], fill='red', width=4)
                    draw.text((x + 5, 60), f"AI: {ai_x1:.3f}", fill='red')

                # Save
                safe_name = pdf_name.replace(' ', '_').replace('.pdf', '')[:40]
                out_path = output_dir / f"{safe_name}_comparison.png"
                result_img.save(out_path)
                print(f"  Saved: {out_path.name}")

                # Also save just the detected title block crops for comparison
                if cv_x1:
                    cv_crop = page_img.crop((int(cv_x1 * width), 0, width, height))
                    cv_crop.save(output_dir / f"{safe_name}_cv_crop.png")

                if ai_x1:
                    ai_crop = page_img.crop((int(ai_x1 * width), 0, width, height))
                    ai_crop.save(output_dir / f"{safe_name}_ai_crop.png")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    create_visual_comparison()
