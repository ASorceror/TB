"""
Ground Truth Generator for Blueprint Processor
Uses Claude Vision API to extract correct values from blueprint pages.

Usage:
    python ground_truth_generator.py <pdf_path_or_folder> [--sample N] [--output ground_truth.csv]
"""

import argparse
import base64
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
import anthropic

from core.pdf_handler import PDFHandler
from config import get_anthropic_key


EXTRACTION_PROMPT = """Analyze this blueprint/construction drawing page and extract the following information.

Return a JSON object with these fields:
{
    "sheet_number": "The sheet/drawing number (e.g., A101, S-201, M1.01, C401, G0.10)",
    "sheet_title": "What the drawing SHOWS - describes the content (see examples below)",
    "date": "Any date shown (e.g., '12/15/2024', '2024-01-15')",
    "project_name": "The project/building name (e.g., 'NORTHERN TOOL', 'ACME OFFICES')",
    "confidence": "high/medium/low - your confidence in the extraction"
}

CRITICAL: sheet_title vs project_name are DIFFERENT things:
- sheet_title = WHAT THE DRAWING SHOWS (the type of drawing)
  Examples: "Floor Plan", "Reflected Ceiling Plan", "Elevations", "Site Plan",
  "Demolition Plan", "Electrical Plan", "Plumbing Plan", "Wall Sections",
  "Details", "Door Schedule", "Finish Schedule", "General Notes"

- project_name = The building/client/project being built
  Examples: "Northern Tool & Equipment", "Acme Corporate HQ", "123 Main Street"

The sheet_title is usually:
- Found in the title block OR as a large label on the drawing
- Describes the TYPE of drawing, not WHO it's for
- If you only see a project name and sheet number but no drawing description,
  set sheet_title to null

Rules:
1. Sheet numbers typically start with a letter prefix (A=Architectural, S=Structural, M=Mechanical, E=Electrical, P=Plumbing, C=Civil, G=General)
2. If a field is not visible or unclear, use null
3. NEVER put the project name in the sheet_title field
4. Look at the actual drawing content to help identify the sheet title

Return ONLY valid JSON, no other text."""


class GroundTruthGenerator:
    """Generate ground truth data using Claude Vision API."""

    def __init__(self, output_path: Path = None):
        """Initialize the generator."""
        self.api_key = get_anthropic_key()
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.output_path = output_path or Path("ground_truth.csv")
        self.results = []
        self.errors = []

    def extract_from_image(self, image_data: bytes, page_info: str) -> Dict[str, Any]:
        """
        Extract ground truth from a single page image using Vision API.

        Args:
            image_data: PNG image bytes
            page_info: String describing the page (for logging)

        Returns:
            Dict with extracted fields
        """
        base64_image = base64.standard_b64encode(image_data).decode("utf-8")

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[
                    {
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
                            {
                                "type": "text",
                                "text": EXTRACTION_PROMPT,
                            },
                        ],
                    }
                ],
            )

            # Parse JSON response
            response_text = response.content[0].text.strip()

            # Handle markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            result = json.loads(response_text)
            result["api_success"] = True
            result["raw_response"] = response_text
            return result

        except json.JSONDecodeError as e:
            return {
                "sheet_number": None,
                "sheet_title": None,
                "date": None,
                "project_name": None,
                "confidence": "error",
                "api_success": False,
                "error": f"JSON parse error: {e}",
                "raw_response": response_text if 'response_text' in dir() else None,
            }
        except Exception as e:
            return {
                "sheet_number": None,
                "sheet_title": None,
                "date": None,
                "project_name": None,
                "confidence": "error",
                "api_success": False,
                "error": str(e),
            }

    def process_pdf(
        self, pdf_path: Path, sample_rate: int = 1, max_pages: int = None
    ) -> List[Dict[str, Any]]:
        """
        Process a PDF and extract ground truth for each page.

        Args:
            pdf_path: Path to PDF file
            sample_rate: Process every Nth page (1 = all pages)
            max_pages: Maximum pages to process (None = all)

        Returns:
            List of extraction results
        """
        results = []
        pdf_name = pdf_path.name

        print(f"\nProcessing: {pdf_name}")

        with PDFHandler(pdf_path) as handler:
            total_pages = handler.page_count
            pages_to_process = list(range(0, total_pages, sample_rate))

            if max_pages:
                pages_to_process = pages_to_process[:max_pages]

            print(f"  Total pages: {total_pages}, Processing: {len(pages_to_process)}")

            for page_num in tqdm(pages_to_process, desc="  Extracting"):
                # Render page as image
                try:
                    image = handler.get_page_image(page_num, dpi=100)

                    # Resize to stay under 5MB API limit
                    import io
                    max_dim = 1568  # Claude Vision recommended max
                    if max(image.size) > max_dim:
                        ratio = max_dim / max(image.size)
                        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                        image = image.resize(new_size)

                    # Convert to JPEG for smaller file size
                    buffer = io.BytesIO()
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')

                    # Try quality levels until under 5MB
                    for quality in [75, 60, 45, 30]:
                        buffer = io.BytesIO()
                        image.save(buffer, format="JPEG", quality=quality)
                        if len(buffer.getvalue()) < 4_500_000:  # Leave margin
                            break

                    image_bytes = buffer.getvalue()

                    # Extract using Vision API
                    page_info = f"{pdf_name} page {page_num + 1}"
                    extraction = self.extract_from_image(image_bytes, page_info)

                    # Add metadata
                    extraction["pdf_filename"] = pdf_name
                    extraction["page_number"] = page_num + 1
                    extraction["timestamp"] = datetime.now().isoformat()

                    results.append(extraction)

                    # Rate limiting - be nice to the API
                    time.sleep(0.5)

                except Exception as e:
                    error_result = {
                        "pdf_filename": pdf_name,
                        "page_number": page_num + 1,
                        "sheet_number": None,
                        "sheet_title": None,
                        "date": None,
                        "confidence": "error",
                        "api_success": False,
                        "error": str(e),
                    }
                    results.append(error_result)
                    self.errors.append(f"{pdf_name} p{page_num + 1}: {e}")

        return results

    def process_folder(
        self, folder_path: Path, sample_rate: int = 1, max_pages_per_pdf: int = None
    ) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a folder.

        Args:
            folder_path: Path to folder containing PDFs
            sample_rate: Process every Nth page
            max_pages_per_pdf: Max pages per PDF

        Returns:
            List of all extraction results
        """
        pdf_files = sorted(folder_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")

        all_results = []
        for pdf_path in pdf_files:
            results = self.process_pdf(pdf_path, sample_rate, max_pages_per_pdf)
            all_results.extend(results)
            self.save_results(all_results)  # Save after each PDF

        return all_results

    def save_results(self, results: List[Dict[str, Any]]):
        """Save results to CSV file."""
        if not results:
            return

        fieldnames = [
            "pdf_filename",
            "page_number",
            "sheet_number",
            "sheet_title",
            "date",
            "project_name",
            "confidence",
            "api_success",
            "error",
        ]

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved {len(results)} results to: {self.output_path}")

    def compare_with_extracted(
        self, ground_truth_path: Path, extracted_db_path: Path = None
    ) -> Dict[str, Any]:
        """
        Compare ground truth with processor extractions.

        Args:
            ground_truth_path: Path to ground truth CSV
            extracted_db_path: Path to processor database

        Returns:
            Accuracy metrics
        """
        import sqlite3

        if extracted_db_path is None:
            extracted_db_path = Path(__file__).parent / "data" / "blueprint_data.db"

        # Load ground truth
        ground_truth = {}
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["pdf_filename"], int(row["page_number"]))
                ground_truth[key] = row

        # Load extracted data
        conn = sqlite3.connect(extracted_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM extracted_sheets")

        extracted = {}
        for row in cursor.fetchall():
            key = (row["pdf_filename"], row["page_number"])
            extracted[key] = dict(row)

        conn.close()

        # Compare
        metrics = {
            "total_compared": 0,
            "sheet_number_correct": 0,
            "sheet_number_both_null": 0,
            "sheet_number_wrong": 0,
            "sheet_title_correct": 0,
            "sheet_title_wrong": 0,
            "date_correct": 0,
            "date_wrong": 0,
            "mismatches": [],
        }

        for key, gt in ground_truth.items():
            if key not in extracted:
                continue

            ext = extracted[key]
            metrics["total_compared"] += 1

            # Compare sheet number
            gt_sheet = (gt.get("sheet_number") or "").strip().upper()
            ext_sheet = (ext.get("sheet_number") or "").strip().upper()

            if gt_sheet == ext_sheet:
                if gt_sheet:
                    metrics["sheet_number_correct"] += 1
                else:
                    metrics["sheet_number_both_null"] += 1
            else:
                metrics["sheet_number_wrong"] += 1
                if gt_sheet:  # Only log if ground truth has a value
                    metrics["mismatches"].append({
                        "pdf": key[0],
                        "page": key[1],
                        "field": "sheet_number",
                        "expected": gt_sheet,
                        "got": ext_sheet,
                    })

            # Compare sheet title (fuzzy - check if similar)
            gt_title = (gt.get("sheet_title") or "").strip().upper()
            ext_title = (ext.get("sheet_title") or "").strip().upper()

            if gt_title and ext_title:
                # Consider correct if one contains the other or >80% overlap
                if gt_title in ext_title or ext_title in gt_title:
                    metrics["sheet_title_correct"] += 1
                else:
                    metrics["sheet_title_wrong"] += 1
            elif not gt_title and not ext_title:
                metrics["sheet_title_correct"] += 1
            else:
                metrics["sheet_title_wrong"] += 1

        # Calculate percentages
        total = metrics["total_compared"]
        if total > 0:
            metrics["sheet_number_accuracy"] = (
                (metrics["sheet_number_correct"] + metrics["sheet_number_both_null"])
                / total
                * 100
            )
            metrics["sheet_title_accuracy"] = (
                metrics["sheet_title_correct"] / total * 100
            )

        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth for blueprint PDFs using Vision API"
    )
    parser.add_argument("path", help="Path to PDF file or folder")
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="Process every Nth page (default: 1 = all)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Max pages per PDF (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ground_truth.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Compare ground truth CSV with extracted data",
    )

    args = parser.parse_args()

    if args.compare:
        # Compare mode
        generator = GroundTruthGenerator()
        metrics = generator.compare_with_extracted(Path(args.compare))
        print("\n" + "=" * 60)
        print("ACCURACY COMPARISON")
        print("=" * 60)
        print(f"Total pages compared: {metrics['total_compared']}")
        print(f"Sheet Number Accuracy: {metrics.get('sheet_number_accuracy', 0):.1f}%")
        print(f"  - Correct: {metrics['sheet_number_correct']}")
        print(f"  - Both null: {metrics['sheet_number_both_null']}")
        print(f"  - Wrong: {metrics['sheet_number_wrong']}")
        print(f"Sheet Title Accuracy: {metrics.get('sheet_title_accuracy', 0):.1f}%")

        if metrics["mismatches"]:
            print(f"\nSample mismatches (first 10):")
            for m in metrics["mismatches"][:10]:
                print(f"  {m['pdf'][:30]} p{m['page']}: expected '{m['expected']}', got '{m['got']}'")
        return

    # Generation mode
    path = Path(args.path)
    output_path = Path(args.output)
    generator = GroundTruthGenerator(output_path)

    print("=" * 60)
    print("GROUND TRUTH GENERATOR")
    print("=" * 60)
    print(f"Source: {path}")
    print(f"Output: {output_path}")
    print(f"Sample rate: every {args.sample} page(s)")
    if args.max_pages:
        print(f"Max pages per PDF: {args.max_pages}")
    print("=" * 60)

    if path.is_file():
        results = generator.process_pdf(path, args.sample, args.max_pages)
    elif path.is_dir():
        results = generator.process_folder(path, args.sample, args.max_pages)
    else:
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    generator.save_results(results)

    # Summary
    success = sum(1 for r in results if r.get("api_success", False))
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total pages processed: {len(results)}")
    print(f"Successful extractions: {success}")
    print(f"Errors: {len(results) - success}")
    print(f"Output saved to: {output_path}")

    if generator.errors:
        print(f"\nErrors encountered:")
        for err in generator.errors[:5]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
