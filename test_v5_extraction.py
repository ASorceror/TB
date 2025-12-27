"""
Test V5.0 extraction improvements on problematic PDFs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.region_detector import RegionDetector
from core.extractor import Extractor
from core.ocr_engine import OCREngine

def test_pdf(pdf_path: Path, pages: list, prefix: str):
    """Test full extraction pipeline on specific pages."""
    print(f"\nTesting: {pdf_path.name}")
    print("=" * 60)

    handler = PDFHandler(pdf_path)
    ocr_engine = OCREngine()
    region_detector = RegionDetector()
    extractor = Extractor(ocr_engine=ocr_engine)

    for page_num in pages:
        print(f"\n--- Page {page_num} ---")

        # Get page components
        image = handler.get_page_image(page_num)
        page = handler.doc[page_num - 1]
        text = handler.get_page_text(page_num)
        text_blocks = handler.get_text_blocks(page_num)

        # Detect title block
        detection = region_detector.detect_title_block(image)
        title_block_image = region_detector.crop_title_block(image, detection)

        # Run full extraction
        result = extractor.extract_fields(
            text=text,
            text_blocks=text_blocks,
            page=page,
            title_block_image=title_block_image,
            page_image=image,
        )

        # Print results
        sheet_number = result.get('sheet_number', '')
        sheet_title = result.get('sheet_title', '')
        extraction_method = result.get('extraction_details', {}).get('sheet_number', 'unknown')

        print(f"Sheet Number: {sheet_number or '(empty)'}")
        print(f"Sheet Title: {sheet_title or '(empty)'}")
        print(f"Method: {extraction_method}")

    handler.close()


if __name__ == "__main__":
    test_data = Path(r"C:\BPA-CC\tests\e2e\test_data")

    # Test Kriser's (had empty extractions)
    # Ground truth: p2=T2, p3=A0, p4=A0.1
    krisers = list(test_data.glob("*Kriser*"))[0]
    test_pdf(krisers, [2, 3, 4], "krisers")

    # Test Janesville (had empty extractions)
    # Ground truth: p1=A-100, p2=A-1, p3=D-1
    janesville = list(test_data.glob("*Janesville*"))[0]
    test_pdf(janesville, [1, 2, 3], "janesville")
