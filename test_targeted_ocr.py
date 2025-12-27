"""
Targeted OCR for sheet numbers in thin outline fonts.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.region_detector import RegionDetector

def find_sheet_number_box(image):
    """
    Find the sheet number value box by locating the 'SHEET NUMBER' label.
    Returns a crop of just the value area.
    """
    # Convert to grayscale
    if image.mode != 'L':
        gray = image.convert('L')
    else:
        gray = image

    # OCR with boxes to find "SHEET NUMBER" label position
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    # Find "SHEET" or "NUMBER" in the OCR results
    sheet_idx = None
    number_idx = None

    for i, text in enumerate(data['text']):
        if 'SHEET' in text.upper():
            sheet_idx = i
        if 'NUMBER' in text.upper():
            number_idx = i

    # If we found the label, crop below it
    if sheet_idx is not None or number_idx is not None:
        idx = number_idx if number_idx else sheet_idx
        label_bottom = data['top'][idx] + data['height'][idx]
        label_left = data['left'][idx]

        # Crop from below the label to end of image
        # Extend LEFT significantly to capture full sheet number (e.g., "A0.1")
        width, height = image.size
        crop_box = (
            0,  # Start from left edge to capture full value
            label_bottom + 5,  # Start below label
            width,  # Full width
            min(height, label_bottom + 400)  # Capture value area
        )
        return image.crop(crop_box)

    return None


def ocr_sheet_number_with_dilation(image, save_prefix=""):
    """
    Apply heavy dilation and OCR to read thin outline font sheet numbers.
    """
    # Upscale the image first for better OCR
    width, height = image.size
    scale = 2  # 2x upscale
    image = image.resize((width * scale, height * scale), Image.Resampling.LANCZOS)

    # Convert to numpy
    if image.mode != 'L':
        img_array = np.array(image.convert('L'))
    else:
        img_array = np.array(image)

    # Ensure dark text on light background
    if np.mean(img_array) < 127:
        img_array = cv2.bitwise_not(img_array)

    # Apply strong contrast
    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Heavy dilation to thicken thin outline font
    # Kernel size adjusted for 2x upscaled image
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.erode(binary, kernel, iterations=2)

    # Convert back to PIL
    result_img = Image.fromarray(dilated)

    if save_prefix:
        result_img.save(f"C:/tb/output/pdf_analysis/{save_prefix}_value_dilated.png")

    # OCR with character whitelist for sheet numbers
    # Sheet numbers are typically: A-Z, 0-9, -, .
    configs = [
        '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-',
        '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-',
        '--psm 6 --oem 3',
        '--psm 7 --oem 0',  # Legacy mode
        '--psm 8 --oem 0',  # Legacy mode
    ]

    results = []
    for config in configs:
        try:
            result = pytesseract.image_to_string(result_img, config=config).strip()
            if result:
                results.append((config.split()[0] + ' ' + config.split()[1], result))
        except Exception as e:
            pass

    return results


def test_pdf(pdf_path: Path, pages: list, prefix: str):
    """Test targeted sheet number extraction on a PDF."""
    print(f"\nTesting: {pdf_path.name}")
    print("=" * 60)

    handler = PDFHandler(pdf_path)
    region_detector = RegionDetector()

    for page_num in pages:
        print(f"\n--- Page {page_num} ---")

        # Get page image
        image = handler.get_page_image(page_num)

        # Detect title block
        detection = region_detector.detect_title_block(image)
        title_block = region_detector.crop_title_block(image, detection)

        # Get bottom portion of title block (where sheet number is)
        tb_width, tb_height = title_block.size
        bottom = title_block.crop((
            int(tb_width * 0.5),  # Right half
            int(tb_height * 0.80),  # Bottom 20%
            tb_width,
            tb_height
        ))
        bottom.save(f"C:/tb/output/pdf_analysis/{prefix}_p{page_num}_bottom.png")

        # Find and crop the sheet number value box
        value_box = find_sheet_number_box(bottom)
        if value_box:
            value_box.save(f"C:/tb/output/pdf_analysis/{prefix}_p{page_num}_value.png")
            print("Found sheet number value box")

            # OCR with dilation
            results = ocr_sheet_number_with_dilation(value_box, f"{prefix}_p{page_num}")
            print("OCR Results:")
            for config, result in results:
                print(f"  {config}: '{result}'")
        else:
            print("Could not locate sheet number value box")

            # Fall back to OCRing the full bottom area
            results = ocr_sheet_number_with_dilation(bottom, f"{prefix}_p{page_num}")
            print("Fallback OCR Results:")
            for config, result in results:
                print(f"  {config}: '{result}'")

    handler.close()


if __name__ == "__main__":
    test_data = Path(r"C:\BPA-CC\tests\e2e\test_data")

    # Test Kriser's
    krisers = list(test_data.glob("*Kriser*"))[0]
    test_pdf(krisers, [2, 3, 4], "krisers")
