"""
Compare ALL Title Block Detection Approaches

1. CV Transition Detection (pure computer vision)
2. Hybrid (CV initial estimate + AI Vision refinement)
3. Improved AI Vision (informed by CV research)

Run on all 17 PDFs and compare results.
"""

import sys
import os
import json
import base64
import io
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler

# ============================================================================
# APPROACH 1: CV TRANSITION DETECTION
# ============================================================================

def cv_transition_detection(pdf_handler, sample_pages, dpi=100):
    """
    Pure CV approach: Find transition from zero to non-zero consistent edges.
    """
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
        edges_array = np.array(edges)
        binary = (edges_array > 25).astype(np.uint8) * 255
        edge_images.append(binary)

    # Find common edges
    stack = np.stack(edge_images, axis=0)
    common_edges = np.all(stack > 128, axis=0).astype(np.uint8) * 255

    height, width = common_edges.shape

    # Calculate column edge counts
    col_edge_count = np.sum(common_edges > 128, axis=0)

    # Smooth
    from scipy.ndimage import uniform_filter1d
    col_smoothed = uniform_filter1d(col_edge_count.astype(float), size=10)

    # Find transition
    left_portion = col_smoothed[:int(width * 0.5)]
    noise_level = np.percentile(left_portion, 95)
    threshold = max(noise_level * 2, 20)

    search_start = int(width * 0.60)
    search_end = int(width * 0.97)
    window_size = 20

    x1 = None
    for i in range(search_start, search_end - window_size):
        window = col_smoothed[i:i + window_size]
        if np.all(window > threshold):
            x1 = i
            break

    if x1 is not None:
        x1_pct = x1 / width
        return {'x1': x1_pct, 'method': 'cv_transition', 'width_pct': 1.0 - x1_pct}
    return None


# ============================================================================
# APPROACH 2: HYBRID (CV + AI Vision Refinement)
# ============================================================================

def hybrid_detection(pdf_handler, sample_pages, dpi=100):
    """
    Hybrid approach:
    1. CV gives approximate range
    2. AI Vision refines by looking at the approximate region
    """
    # First get CV estimate
    cv_result = cv_transition_detection(pdf_handler, sample_pages, dpi)

    if cv_result is None:
        cv_x1 = 0.85  # Default fallback
    else:
        cv_x1 = cv_result['x1']

    # Expand the search range for AI Vision
    search_x1 = max(0.70, cv_x1 - 0.10)  # 10% left of CV estimate
    search_x2 = min(1.0, cv_x1 + 0.10)   # 10% right of CV estimate

    # Render a sample page and crop to search region
    img = pdf_handler.get_page_image(sample_pages[0] - 1, dpi=150)
    width, height = img.size

    crop_x1 = int(search_x1 * width)
    crop_x2 = int(search_x2 * width)
    cropped = img.crop((crop_x1, 0, crop_x2, height))

    # Call AI Vision to find exact boundary within cropped region
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Convert image to base64
        img_bytes = io.BytesIO()
        cropped.save(img_bytes, format='PNG')
        img_data = base64.standard_b64encode(img_bytes.getvalue()).decode('utf-8')

        prompt = """This image shows a CROPPED portion of a blueprint page.
The title block LEFT BORDER is somewhere in this image.

Find the vertical line that separates:
- LEFT: Drawing area (may have some content or be mostly blank)
- RIGHT: Title block (has structured boxes, text, firm name, sheet number)

Return ONLY a JSON object:
{
  "left_border_x_percent": 0.XX,
  "confidence": 0.XX,
  "description": "brief description of what you see"
}

Where left_border_x_percent is the position within THIS CROPPED IMAGE (0.0 = left edge, 1.0 = right edge).
"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_data}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        response_text = message.content[0].text.strip()

        # Parse JSON
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            local_x = result.get('left_border_x_percent', 0.5)

            # Convert local coordinate back to full page coordinate
            global_x1 = search_x1 + local_x * (search_x2 - search_x1)

            return {
                'x1': global_x1,
                'method': 'hybrid',
                'width_pct': 1.0 - global_x1,
                'cv_estimate': cv_x1,
                'confidence': result.get('confidence', 0)
            }
    except Exception as e:
        print(f"    Hybrid AI error: {e}")

    return None


# ============================================================================
# APPROACH 3: IMPROVED AI VISION (Full pages, informed prompt)
# ============================================================================

IMPROVED_AI_PROMPT = """You are analyzing blueprint pages to find the EXACT title block boundary.

I'm showing you {num_pages} sample pages from the same PDF.

## KEY INSIGHT FROM COMPUTER VISION ANALYSIS:
- The title block is on the RIGHT EDGE of each page
- It's a vertical strip containing: firm name, project info, sheet number, etc.
- The LEFT BORDER of the title block is a vertical line separating it from drawings
- Title block widths vary: some are narrow (8-10%), others wide (15-20%)

## YOUR TASK:
Look at ALL pages and find where the title block LEFT BORDER is.

The left border is where:
- Drawing content (floor plans, details, schedules) ENDS
- Title block structure (boxes, firm info, sheet ID) BEGINS

## CRITICAL RULES:
1. Find the LEFTMOST vertical line that is part of the title block structure
2. This line should be consistent across ALL sample pages
3. Include ALL title block content (firm name, even if sparse/vertical text)
4. The boundary should have natural whitespace - NO text cutoff

Return ONLY this JSON:
{{
  "title_block_x1": 0.XX,
  "confidence": 0.XX,
  "width_percent": XX.X,
  "notes": "what you observed"
}}

Where title_block_x1 is the LEFT edge as a fraction (0.0 = page left, 1.0 = page right).
Typical values: 0.80 to 0.92 depending on the firm's title block design.
"""


def improved_ai_vision(pdf_handler, sample_pages, dpi=100):
    """
    Improved AI Vision approach with informed prompt.
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Build content with all sample pages
        content = []
        target_size = None

        for page_num in sample_pages:
            img = pdf_handler.get_page_image(page_num - 1, dpi=dpi)

            if target_size is None:
                target_size = img.size
            elif img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Resize for API efficiency
            max_size = 1200
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_data = base64.standard_b64encode(img_bytes.getvalue()).decode('utf-8')

            content.append({"type": "text", "text": f"--- PAGE {page_num} ---"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_data}
            })

        prompt = IMPROVED_AI_PROMPT.replace("{num_pages}", str(len(sample_pages)))
        content.append({"type": "text", "text": prompt})

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": content}]
        )

        response_text = message.content[0].text.strip()

        # Parse JSON
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            x1 = result.get('title_block_x1', 0.85)

            return {
                'x1': x1,
                'method': 'improved_ai',
                'width_pct': result.get('width_percent', (1.0 - x1) * 100) / 100,
                'confidence': result.get('confidence', 0),
                'notes': result.get('notes', '')
            }
    except Exception as e:
        print(f"    AI Vision error: {e}")

    return None


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_comparison():
    """Run all approaches on all PDFs and compare."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")
    output_dir = Path(r"C:\tb\blueprint_processor\output\comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get all PDFs
    all_pdfs = sorted(test_dir.glob("*.pdf"))

    print("="*80)
    print("COMPREHENSIVE COMPARISON: All Title Block Detection Approaches")
    print(f"Testing {len(all_pdfs)} PDFs")
    print("="*80)

    results = []

    for pdf_idx, pdf_path in enumerate(all_pdfs, 1):
        pdf_name = pdf_path.name
        print(f"\n[{pdf_idx}/{len(all_pdfs)}] {pdf_name[:50]}")

        try:
            with PDFHandler(pdf_path) as handler:
                total_pages = handler.page_count

                # Sample pages (skip potential cover sheet)
                if total_pages > 5:
                    sample_pages = [2, 4, 6, 8, 10][:min(5, total_pages - 1)]
                else:
                    sample_pages = list(range(2, min(6, total_pages + 1)))

                if not sample_pages:
                    sample_pages = [1]

                print(f"  Pages: {total_pages}, Sampling: {sample_pages}")

                # Run all approaches
                cv_result = cv_transition_detection(handler, sample_pages)
                hybrid_result = hybrid_detection(handler, sample_pages)
                ai_result = improved_ai_vision(handler, sample_pages)

                # Store results
                pdf_results = {
                    'pdf': pdf_name,
                    'total_pages': total_pages,
                    'cv_transition': cv_result,
                    'hybrid': hybrid_result,
                    'improved_ai': ai_result
                }
                results.append(pdf_results)

                # Print comparison
                cv_x1 = cv_result['x1'] if cv_result else None
                hybrid_x1 = hybrid_result['x1'] if hybrid_result else None
                ai_x1 = ai_result['x1'] if ai_result else None

                print(f"  CV Transition:  x1={cv_x1:.3f}" if cv_x1 else "  CV Transition:  FAILED")
                print(f"  Hybrid:         x1={hybrid_x1:.3f}" if hybrid_x1 else "  Hybrid:         FAILED")
                print(f"  Improved AI:    x1={ai_x1:.3f}" if ai_x1 else "  Improved AI:    FAILED")

                # Check agreement
                values = [v for v in [cv_x1, hybrid_x1, ai_x1] if v is not None]
                if len(values) >= 2:
                    spread = max(values) - min(values)
                    if spread < 0.03:
                        print(f"  ✓ AGREEMENT (spread: {spread:.3f})")
                    else:
                        print(f"  ✗ DISAGREEMENT (spread: {spread:.3f})")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'pdf': pdf_name, 'error': str(e)})

    # Save results
    results_path = output_dir / f"comparison_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'PDF':<45} {'CV':>8} {'Hybrid':>8} {'AI':>8} {'Agree':>8}")
    print("-"*80)

    for r in results:
        if 'error' in r:
            print(f"{r['pdf'][:44]:<45} ERROR")
            continue

        cv_x1 = r['cv_transition']['x1'] if r['cv_transition'] else None
        hybrid_x1 = r['hybrid']['x1'] if r['hybrid'] else None
        ai_x1 = r['improved_ai']['x1'] if r['improved_ai'] else None

        cv_str = f"{cv_x1:.3f}" if cv_x1 else "---"
        hybrid_str = f"{hybrid_x1:.3f}" if hybrid_x1 else "---"
        ai_str = f"{ai_x1:.3f}" if ai_x1 else "---"

        values = [v for v in [cv_x1, hybrid_x1, ai_x1] if v is not None]
        if len(values) >= 2:
            spread = max(values) - min(values)
            agree_str = "✓" if spread < 0.03 else f"±{spread:.2f}"
        else:
            agree_str = "?"

        print(f"{r['pdf'][:44]:<45} {cv_str:>8} {hybrid_str:>8} {ai_str:>8} {agree_str:>8}")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    run_comparison()
