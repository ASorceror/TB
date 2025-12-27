"""
Quick test to verify V5.0 fix on a single problematic PDF.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.region_detector import RegionDetector
from core.extractor import Extractor
from core.ocr_engine import OCREngine

def test_pdf(pdf_path: Path, pages_to_test: list = None, prefix: str = ""):
    """Test extraction on specific pages of a PDF."""
    print(f"\nTesting: {pdf_path.name}")
    print("=" * 60)

    if not prefix:
        prefix = pdf_path.stem[:10]

    handler = PDFHandler(pdf_path)
    ocr_engine = OCREngine()
    region_detector = RegionDetector()
    extractor = Extractor(ocr_engine=ocr_engine)

    if pages_to_test is None:
        pages_to_test = list(range(1, min(handler.page_count + 1, 6)))

    for page_num in pages_to_test:
        print(f"\n--- Page {page_num} ---")

        # Get page image
        image = handler.get_page_image(page_num)
        page = handler.doc[page_num - 1]  # fitz pages are 0-indexed

        # Detect title block
        detection = region_detector.detect_title_block(image)
        print(f"Title block region: {detection['region_name']} (score: {detection['score']:.2f})")
        print(f"Keywords found: {detection['keywords_found']}")

        # Get text from page
        text = handler.get_page_text(page_num)
        text_blocks = handler.get_text_blocks(page_num)

        # Extract sheet number
        sheet_num = extractor.extract_sheet_number(
            text,
            text_blocks=text_blocks,
            page=page
        )
        print(f"Sheet Number: {sheet_num or '(empty)'}")

        # Try title block OCR if main extraction failed
        if not sheet_num:
            title_block_image = region_detector.crop_title_block(image, detection)

            # Save full title block for inspection
            tb_width, tb_height = title_block_image.size
            title_block_image.save(f"C:/tb/output/pdf_analysis/{prefix}_titleblock_p{page_num}.png")
            print(f"Title block size: {tb_width}x{tb_height}")

            # For vertical title blocks (right_strip), sheet number is often at BOTTOM
            # but in a specific box. Let's look at the bottom 15% of the title block
            bottom_region = title_block_image.crop((
                0,
                int(tb_height * 0.85),
                tb_width,
                tb_height
            ))
            bottom_region.save(f"C:/tb/output/pdf_analysis/{prefix}_bottom_p{page_num}.png")

            # OCR the bottom region
            bottom_ocr = ocr_engine.ocr_image(bottom_region)
            print(f"Bottom region OCR: '{bottom_ocr.strip()[:200]}'")

            # Try to isolate the sheet number box (right half of bottom region)
            br_width, br_height = bottom_region.size
            sheet_box = bottom_region.crop((
                int(br_width * 0.5),  # Right half
                0,
                br_width,
                br_height
            ))
            sheet_box.save(f"C:/tb/output/pdf_analysis/{prefix}_sheetbox_p{page_num}.png")

            # Try different Tesseract PSM modes on the sheet box
            import pytesseract
            from PIL import ImageOps, ImageFilter
            import cv2
            import numpy as np

            # Enhance for thin line fonts
            enhanced = sheet_box.convert('L')
            enhanced = ImageOps.autocontrast(enhanced)

            # Convert to numpy for OpenCV processing
            img_array = np.array(enhanced)

            # Apply morphological dilation to thicken thin lines
            # First, ensure dark text on light background
            if np.mean(img_array) < 127:
                img_array = cv2.bitwise_not(img_array)

            # Threshold to binary
            _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Dilate the text (erode since text is black on white)
            # Use larger kernel for thin outline fonts
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.erode(binary, kernel, iterations=2)

            # Save dilated version for inspection
            from PIL import Image
            dilated_pil = Image.fromarray(dilated)
            dilated_pil.save(f"C:/tb/output/pdf_analysis/{prefix}_dilated_p{page_num}.png")

            # Try OCR on dilated image
            print("  Dilated OCR tests:")
            for psm in [6, 7, 8, 10]:
                config = f'--psm {psm} --oem 3'
                try:
                    result = pytesseract.image_to_string(dilated_pil, config=config).strip()
                    if result:
                        print(f"    PSM {psm}: '{result}'")
                except Exception as e:
                    print(f"    PSM {psm}: error - {e}")

    handler.close()

if __name__ == "__main__":
    test_data = Path(r"C:\BPA-CC\tests\e2e\test_data")

    # Test Kriser's (had empty extractions)
    krisers = list(test_data.glob("*Kriser*"))[0]
    test_pdf(krisers, [2, 3, 4], prefix="krisers")

    # Test Janesville (had empty extractions)
    janesville = list(test_data.glob("*Janesville*"))[0]
    test_pdf(janesville, [1, 2, 3], prefix="janesville")
