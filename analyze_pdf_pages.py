"""
Extract PDF pages as images for vision analysis.
"""

import fitz  # PyMuPDF
from pathlib import Path
import sys

def extract_pages(pdf_path: Path, output_dir: Path, pages: list = None, dpi: int = 150):
    """Extract specified pages from a PDF as PNG images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pdf_name = pdf_path.stem[:30]  # Truncate long names

    if pages is None:
        pages = range(1, min(doc.page_count + 1, 6))  # First 5 pages by default

    for page_num in pages:
        if page_num < 1 or page_num > doc.page_count:
            print(f"  Page {page_num} out of range (1-{doc.page_count})")
            continue

        page = doc[page_num - 1]  # 0-indexed

        # Render at specified DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        output_path = output_dir / f"{pdf_name}_p{page_num}.png"
        pix.save(str(output_path))
        print(f"  Saved: {output_path.name}")

    doc.close()

def main():
    test_data_dir = Path(r"C:\BPA-CC\tests\e2e\test_data")
    output_dir = Path(r"C:\tb\output\pdf_analysis")

    # PDFs with most extraction errors based on accuracy test
    problem_pdfs = [
        ("Kriser's Highand Final Set.pdf", [2, 3, 4, 5, 6]),  # Got '' for many
        ("Janesville Nissan Full set Issued for Bids.pdf", [1, 2, 3, 4, 5]),  # Got '' for many
        ("Senju Office TI Arch Permit Set 2.4.20.pdf", [1, 2, 3, 4, 5]),  # A2.0 vs A15
        ("0_full_permit_set_chiro_one_evergreen_park.pdf", [1, 2, 3, 4]),  # Control - likely correct format
    ]

    # Find PDFs with glob pattern matching
    for pdf_pattern, pages in problem_pdfs:
        matches = list(test_data_dir.glob(f"*{pdf_pattern.split()[0]}*"))
        if not matches:
            # Try with first word
            first_word = pdf_pattern.split()[0].replace("'", "*")
            matches = list(test_data_dir.glob(f"*{first_word}*"))

        if matches:
            pdf_path = matches[0]
            print(f"\nExtracting from: {pdf_path.name}")
            extract_pages(pdf_path, output_dir, pages)
        else:
            print(f"\nNot found: {pdf_pattern}")

if __name__ == "__main__":
    main()
