"""
Diagnostic Extraction Tool - Shows EXACTLY what the code is doing.

For each page, this saves:
1. Raw page image (before normalization)
2. Normalized page image (after rotation correction)
3. Detected title block region (what we're searching for sheet number)
4. Edge regions checked during fallback OCR
5. Step-by-step log of extraction decisions

Output: CSV + folders of images for human verification.
"""
import csv
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))

import fitz  # PyMuPDF

from core.pdf_handler import PDFHandler
from core.page_normalizer import PageNormalizer
from core.title_block_discovery import TitleBlockDiscovery
from core.extractor import Extractor
from core.ocr_engine import OCREngine
from core.vision_extractor import VisionExtractor
from constants import DEFAULT_DPI, TESSERACT_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger('diagnostic')


def load_ground_truth(csv_path: Path):
    """Load ground truth from CSV file."""
    ground_truth = {}
    if not csv_path.exists():
        return ground_truth
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_name = row.get('pdf_filename', '')
            page_num = row.get('page_number', '')
            if pdf_name and page_num:
                key = (pdf_name, int(page_num))
                ground_truth[key] = {
                    'sheet_number': row.get('sheet_number', ''),
                    'sheet_title': row.get('sheet_title', ''),
                }
    return ground_truth


def normalize(value):
    if value is None:
        return ''
    return str(value).strip().upper()


def draw_bbox_on_image(image: Image.Image, bbox: tuple, color='red', label='') -> Image.Image:
    """Draw a bounding box on an image copy."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        if label:
            draw.text((x0 + 5, y0 + 5), label, fill=color)
    return img_copy


def run_diagnostic():
    """Run diagnostic extraction with full telemetry."""

    # Setup output
    output_dir = Path(r"C:\tb\blueprint_processor\output\diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC EXTRACTION")
    print(f"Output: {run_dir}")
    print(f"{'='*70}\n")

    # Load ground truth
    gt_path = Path(r"C:\tb\blueprint_processor\output\ground_truth_full.csv")
    ground_truth = load_ground_truth(gt_path)
    print(f"Ground truth: {len(ground_truth)} entries loaded\n")

    test_data_dir = Path(r"C:\BPA-CC\tests\e2e\test_data")

    # Test PDFs - the ones we claim work well
    test_pdfs = [
        ("Janesville Nissan Full set Issued for Bids.pdf", [1, 2]),
        ("0_full_permit_set_chiro_one_evergreen_park.pdf", [1, 4]),
        ("18222 midland tx - final 2-19-19 rev 1.pdf", [1, 2]),
    ]

    # Initialize components
    normalizer = PageNormalizer()
    title_block_discovery = TitleBlockDiscovery(cache_dir=run_dir / 'telemetry')
    extractor = Extractor()
    ocr_engine = OCREngine()
    vision_extractor = VisionExtractor()

    # CSV output
    csv_path = run_dir / "diagnostic_results.csv"
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'pdf_name', 'page_num',
        'expected_sn', 'extracted_sn', 'match',
        'extraction_method',
        'rotation_detected', 'rotation_applied',
        'title_block_found', 'title_block_bbox',
        'raw_page_image', 'normalized_image', 'title_block_image',
        'step_by_step_log'
    ])

    # Detailed log
    log_path = run_dir / "extraction_steps.txt"
    log_file = open(log_path, 'w', encoding='utf-8')

    stats = {'correct': 0, 'wrong': 0, 'empty': 0}

    for pdf_name, pages in test_pdfs:
        pdf_path = test_data_dir / pdf_name
        if not pdf_path.exists():
            print(f"SKIP: {pdf_name} not found")
            continue

        # Create PDF output folder
        pdf_safe = pdf_name.replace(' ', '_').replace('.pdf', '')[:40]
        pdf_dir = run_dir / pdf_safe
        pdf_dir.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print(f"PDF: {pdf_name}")
        log_file.write(f"\n{'='*70}\n")
        log_file.write(f"PDF: {pdf_name}\n")
        log_file.write(f"{'='*70}\n")

        try:
            with PDFHandler(pdf_path) as handler:
                extractor.reset_for_new_pdf()

                # V6.0: Run title block discovery ONCE per PDF
                import hashlib
                with open(pdf_path, 'rb') as f:
                    pdf_hash = hashlib.sha256(f.read()).hexdigest()[:16]

                print(f"  Running title block discovery (hash: {pdf_hash})...")
                log_file.write(f"  PDF hash: {pdf_hash}\n")
                steps_discovery = [f"Running discovery for {pdf_name}"]

                try:
                    title_block_discovery.discover(handler, pdf_hash)
                    telemetry = title_block_discovery.get_current_telemetry()
                    if telemetry:
                        steps_discovery.append(f"  Location: {telemetry.title_block_location}")
                        steps_discovery.append(f"  Confidence: {telemetry.title_block_confidence:.2f}")
                        steps_discovery.append(f"  Zones: {list(telemetry.zones.keys())}")
                        steps_discovery.append(f"  Page 1 is cover: {telemetry.page_1_is_cover_sheet}")
                        log_file.write(f"  Discovery: location={telemetry.title_block_location}, "
                                      f"confidence={telemetry.title_block_confidence:.2f}\n")
                        log_file.write(f"  Zones: {list(telemetry.zones.keys())}\n")
                        print(f"  Discovery complete: {telemetry.title_block_location}, "
                              f"conf={telemetry.title_block_confidence:.2f}")
                except Exception as e:
                    steps_discovery.append(f"  Discovery FAILED: {e}")
                    log_file.write(f"  Discovery FAILED: {e}\n")
                    print(f"  Discovery FAILED: {e}")

                for page_num in pages:
                    steps = []
                    print(f"\n  Page {page_num}:")
                    log_file.write(f"\n--- Page {page_num} ---\n")
                    steps.append(f"Processing page {page_num}")

                    # Get ground truth
                    key = (pdf_name, page_num)
                    expected = ground_truth.get(key, {})
                    expected_sn = normalize(expected.get('sheet_number', ''))

                    # === STEP 1: Render raw page ===
                    steps.append("Step 1: Render raw page at 150 DPI")
                    page = handler.doc[page_num - 1]
                    matrix = fitz.Matrix(150/72, 150/72)
                    pix = page.get_pixmap(matrix=matrix)
                    raw_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    raw_path = pdf_dir / f"p{page_num:02d}_1_raw.png"
                    raw_image.save(raw_path)
                    steps.append(f"  Saved: {raw_path.name} ({raw_image.size})")
                    log_file.write(f"  Raw image: {raw_image.size}\n")

                    # === STEP 2: Orientation detection and normalization ===
                    steps.append("Step 2: Orientation detection")
                    normalized_image, orient_info = normalizer.normalize(raw_image)
                    rotation_detected = orient_info.get('angle', 0)
                    rotation_applied = orient_info.get('rotation_applied', 0)
                    rotation_method = orient_info.get('method', 'unknown')

                    steps.append(f"  Detected: {rotation_detected}° via {rotation_method}")
                    steps.append(f"  Applied: {rotation_applied}° rotation")
                    log_file.write(f"  Orientation: detected={rotation_detected}°, applied={rotation_applied}°, method={rotation_method}\n")

                    norm_path = pdf_dir / f"p{page_num:02d}_2_normalized.png"
                    normalized_image.save(norm_path)
                    steps.append(f"  Saved: {norm_path.name} ({normalized_image.size})")

                    # === STEP 3: Title block detection ===
                    steps.append("Step 3: Title block detection (V6.0 - uses discovery)")
                    detection = title_block_discovery.detect_title_block(normalized_image)
                    tb_bbox = detection.get('bbox')
                    tb_confidence = detection.get('confidence', 0)
                    tb_method = detection.get('method', 'unknown')

                    try:
                        tb_conf_val = float(tb_confidence) if tb_confidence else 0.0
                    except:
                        tb_conf_val = 0.0
                    steps.append(f"  Method: {tb_method}, confidence: {tb_conf_val:.2f}")
                    log_file.write(f"  Title block: method={tb_method}, confidence={tb_conf_val:.2f}\n")

                    title_block_image = None
                    tb_path_str = ""
                    if tb_bbox:
                        steps.append(f"  BBox: {tb_bbox}")
                        log_file.write(f"  Title block bbox: {tb_bbox}\n")

                        # Crop and save title block
                        title_block_image = title_block_discovery.crop_title_block(normalized_image, detection)
                        if title_block_image:
                            tb_path = pdf_dir / f"p{page_num:02d}_3_titleblock.png"
                            title_block_image.save(tb_path)
                            tb_path_str = tb_path.name
                            steps.append(f"  Saved: {tb_path.name} ({title_block_image.size})")

                        # Save normalized image with bbox drawn
                        annotated = draw_bbox_on_image(normalized_image, tb_bbox, 'red', 'TITLE_BLOCK')
                        annotated_path = pdf_dir / f"p{page_num:02d}_2b_annotated.png"
                        annotated.save(annotated_path)
                        steps.append(f"  Saved annotated: {annotated_path.name}")
                    else:
                        steps.append("  WARNING: No title block detected!")
                        log_file.write("  WARNING: Title block detection FAILED\n")

                    # === STEP 4: Text extraction ===
                    steps.append("Step 4: Text extraction")
                    text = page.get_text()
                    text_len = len(text.strip())
                    steps.append(f"  Vector text: {text_len} chars")
                    log_file.write(f"  Vector text: {text_len} chars\n")

                    # Get text blocks for spatial matching
                    text_blocks = handler.get_text_blocks(page_num - 1)
                    steps.append(f"  Text blocks: {len(text_blocks)} blocks")

                    # === STEP 5: Run extraction ===
                    steps.append("Step 5: Sheet number extraction")

                    # Call the extractor directly to see what it does
                    result = extractor.extract_fields(
                        text=text,
                        text_blocks=text_blocks,
                        page_number=page_num,
                        page=page,
                        title_block_bbox_pixels=tb_bbox,
                        title_block_image=title_block_image,
                        page_image=normalized_image,
                        original_page_image=raw_image,
                        rotation_applied=rotation_applied,
                    )

                    extracted_sn = normalize(result.get('sheet_number', ''))
                    method = result.get('extraction_details', {}).get('sheet_number', 'unknown')

                    steps.append(f"  Extracted: '{extracted_sn}' via {method}")
                    log_file.write(f"  Extracted sheet number: '{extracted_sn}' via {method}\n")

                    # Show all extraction details
                    details = result.get('extraction_details', {})
                    if details:
                        steps.append(f"  Details: {details}")
                        log_file.write(f"  Extraction details: {details}\n")

                    # === COMPARE ===
                    if extracted_sn == expected_sn:
                        status = 'OK'
                        stats['correct'] += 1
                    elif not extracted_sn:
                        status = 'EMPTY'
                        stats['empty'] += 1
                    else:
                        status = 'WRONG'
                        stats['wrong'] += 1

                    steps.append(f"Result: [{status}] Expected='{expected_sn}', Got='{extracted_sn}'")
                    log_file.write(f"  RESULT: [{status}] expected='{expected_sn}', got='{extracted_sn}'\n")

                    print(f"    [{status}] '{expected_sn}' vs '{extracted_sn}' via {method}")

                    # Write to CSV
                    csv_writer.writerow([
                        pdf_name, page_num,
                        expected_sn, extracted_sn, 'Y' if status == 'OK' else 'N',
                        method,
                        rotation_detected, rotation_applied,
                        'Y' if tb_bbox else 'N', str(tb_bbox) if tb_bbox else '',
                        raw_path.name, norm_path.name, tb_path_str,
                        ' | '.join(steps)
                    ])

                    # Write step log
                    for step in steps:
                        log_file.write(f"  {step}\n")

                    time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"  ERROR: {e}")
            log_file.write(f"ERROR: {e}\n")
            import traceback
            log_file.write(traceback.format_exc())

    csv_file.close()
    log_file.close()

    # Summary
    total = stats['correct'] + stats['wrong'] + stats['empty']
    print(f"\n{'='*70}")
    pct = 100*stats['correct']/total if total > 0 else 0
    print(f"SUMMARY: {stats['correct']}/{total} correct ({pct:.1f}%)")
    print(f"  Correct: {stats['correct']}")
    print(f"  Wrong:   {stats['wrong']}")
    print(f"  Empty:   {stats['empty']}")
    print(f"\nOutput:")
    print(f"  CSV:    {csv_path}")
    print(f"  Log:    {log_path}")
    print(f"  Images: {run_dir}")
    print(f"\nOpen the images to verify what the code is actually seeing!")


if __name__ == "__main__":
    run_diagnostic()
