"""
Fix missing sheet titles in ground truth CSV by re-processing those pages.
"""

import base64
import csv
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
import anthropic

from core.pdf_handler import PDFHandler
from config import get_anthropic_key

TITLE_PROMPT = """Look at this blueprint/construction drawing page and find the SHEET TITLE.

The sheet title is the descriptive name of what this drawing shows, such as:
- "FLOOR PLAN"
- "ROOF PLAN"
- "ELEVATIONS"
- "ELECTRICAL PLAN"
- "MECHANICAL SCHEDULE"
- "WALL SECTIONS"
- "DETAILS"
- "FOUNDATION PLAN"
- "SITE PLAN"

The title is usually located in the title block (bottom right corner or right edge of the page), often directly above or below the sheet number.

Return ONLY a JSON object:
{"sheet_title": "THE TITLE HERE"}

If you truly cannot find any title, return:
{"sheet_title": null}

Return ONLY valid JSON, no other text."""


def fix_missing_titles(csv_path: Path, pdf_folder: Path, output_path: Path = None):
    """Re-process pages with missing titles."""

    api_key = get_anthropic_key()
    client = anthropic.Anthropic(api_key=api_key)

    # Load existing data
    rows = []
    missing_indices = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader):
            rows.append(row)
            title = (row.get('sheet_title') or '').strip()
            if not title and row.get('api_success') == 'True':
                missing_indices.append(i)

    print(f"Total rows: {len(rows)}")
    print(f"Missing titles to fix: {len(missing_indices)}")

    if not missing_indices:
        print("No missing titles to fix!")
        return

    # Group by PDF for efficient processing
    pdf_pages = {}
    for idx in missing_indices:
        row = rows[idx]
        pdf_name = row['pdf_filename']
        page_num = int(row['page_number'])
        if pdf_name not in pdf_pages:
            pdf_pages[pdf_name] = []
        pdf_pages[pdf_name].append((idx, page_num))

    print(f"PDFs to process: {len(pdf_pages)}")

    fixed_count = 0

    for pdf_name, pages in pdf_pages.items():
        pdf_path = pdf_folder / pdf_name
        if not pdf_path.exists():
            print(f"  WARNING: PDF not found: {pdf_path}")
            continue

        print(f"\nProcessing: {pdf_name} ({len(pages)} pages)")

        try:
            with PDFHandler(pdf_path) as handler:
                for idx, page_num in tqdm(pages, desc="  Fixing"):
                    try:
                        # Render page
                        image = handler.get_page_image(page_num - 1, dpi=100)

                        # Resize if needed
                        max_dim = 1568
                        if max(image.size) > max_dim:
                            ratio = max_dim / max(image.size)
                            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                            image = image.resize(new_size)

                        # Convert to JPEG
                        buffer = io.BytesIO()
                        if image.mode == 'RGBA':
                            image = image.convert('RGB')

                        for quality in [75, 60, 45, 30]:
                            buffer = io.BytesIO()
                            image.save(buffer, format="JPEG", quality=quality)
                            if len(buffer.getvalue()) < 4_500_000:
                                break

                        image_bytes = buffer.getvalue()
                        base64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

                        # Call Vision API with focused prompt
                        response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=200,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": base64_image,
                                        },
                                    },
                                    {"type": "text", "text": TITLE_PROMPT},
                                ],
                            }],
                        )

                        response_text = response.content[0].text.strip()
                        if response_text.startswith("```"):
                            lines = response_text.split("\n")
                            response_text = "\n".join(lines[1:-1])

                        result = json.loads(response_text)
                        new_title = result.get('sheet_title')

                        if new_title:
                            rows[idx]['sheet_title'] = new_title
                            fixed_count += 1

                        time.sleep(0.3)

                    except Exception as e:
                        print(f"    Error on page {page_num}: {e}")

        except Exception as e:
            print(f"  Error opening PDF: {e}")

    # Save updated CSV
    output_path = output_path or csv_path
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} of {len(missing_indices)} missing titles")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    csv_path = Path("output/ground_truth_full.csv")
    pdf_folder = Path("C:/Hybrid-Extraction-Test/Test Blueprints")

    fix_missing_titles(csv_path, pdf_folder)
