"""Debug extraction for a specific page with detailed tracing."""
import sys
import logging
from pathlib import Path

# Enable DEBUG logging for extractor
logging.basicConfig(level=logging.DEBUG, format='%(name)s %(levelname)s: %(message)s')

sys.path.insert(0, str(Path(__file__).parent))

from main import BlueprintProcessor
from core.pdf_handler import PDFHandler

def debug_page(pdf_name: str, page_num: int):
    """Debug extraction for a specific page."""
    test_data_dir = Path(r"C:\BPA-CC\tests\e2e\test_data")
    pdf_path = test_data_dir / pdf_name

    processor = BlueprintProcessor()

    with PDFHandler(pdf_path) as handler:
        result = processor.process_page(handler, page_num - 1, pdf_name)

        print("\n" + "="*60)
        print(f"RESULT FOR {pdf_name} PAGE {page_num}")
        print("="*60)
        print(f"Sheet Number: {result.get('sheet_number')}")
        print(f"Sheet Title: {result.get('sheet_title')}")
        print(f"Extraction Method: {result.get('extraction_method')}")
        print(f"Extraction Details: {result.get('extraction_details')}")

if __name__ == "__main__":
    # Debug Janesville cover sheet (T-1)
    debug_page("Janesville Nissan Full set Issued for Bids.pdf", 1)
