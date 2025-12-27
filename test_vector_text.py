"""
Test if sheet numbers are in the PDF text layer (vector PDFs).
"""

import fitz
from pathlib import Path

def analyze_text_layer(pdf_path: Path, pages: list = None):
    """Check the PDF text layer for sheet number patterns."""
    print(f"\nAnalyzing: {pdf_path.name}")
    print("=" * 60)

    doc = fitz.open(pdf_path)

    if pages is None:
        pages = list(range(1, min(doc.page_count + 1, 6)))

    for page_num in pages:
        page = doc[page_num - 1]
        print(f"\n--- Page {page_num} ---")

        # Get all text
        full_text = page.get_text()

        # Search for sheet number patterns
        import re
        sheet_patterns = [
            r'SHEET\s*NUMBER[:\s]*([A-Z0-9.\-]+)',
            r'SHEET\s*NO\.?[:\s]*([A-Z0-9.\-]+)',
            r'SHEET[:\s]+([A-Z]\d+(?:\.\d+)?)',
            r'\bA0\b',
            r'\bA0\.1\b',
            r'\bT2\b',
            r'\bA-1\b',
            r'\bD-1\b',
        ]

        print("Pattern matches:")
        for pattern in sheet_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                print(f"  '{pattern}': {matches}")

        # Get text with positions (text blocks)
        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])

        # Look for text blocks containing potential sheet numbers
        potential_sheet_nums = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        # Check if it looks like a sheet number
                        if re.match(r'^[A-Z]{1,2}[-.]?\d{1,3}(?:\.\d{1,2})?$', text):
                            bbox = span["bbox"]
                            potential_sheet_nums.append({
                                "text": text,
                                "bbox": bbox,
                                "font": span.get("font", ""),
                                "size": span.get("size", 0),
                            })

        if potential_sheet_nums:
            print("\nPotential sheet numbers found in text layer:")
            for item in potential_sheet_nums:
                print(f"  '{item['text']}' at {item['bbox'][:2]} (font: {item['font']}, size: {item['size']:.1f})")
        else:
            print("\nNo obvious sheet numbers in text layer")

        # Show last 500 chars of page text (likely title block area)
        print(f"\nLast 500 chars of text:\n{full_text[-500:]}")

    doc.close()

if __name__ == "__main__":
    test_data = Path(r"C:\BPA-CC\tests\e2e\test_data")

    # Test Kriser's
    krisers = list(test_data.glob("*Kriser*"))[0]
    analyze_text_layer(krisers, [2, 3])

    # Test Janesville
    janesville = list(test_data.glob("*Janesville*"))[0]
    analyze_text_layer(janesville, [2, 3])
