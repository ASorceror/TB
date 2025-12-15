"""
Blueprint Processor V4.1 - Phase 5 Checkpoint Test
Verifies: Field extraction and validation work correctly.

Run: python tests/test_phase5.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.pdf_handler import PDFHandler
from core.page_normalizer import PageNormalizer
from core.region_detector import RegionDetector
from core.ocr_engine import OCREngine
from core.extractor import Extractor
from validation.validator import Validator

# Telemetry output directory
telemetry_dir = project_root / 'output' / 'telemetry'
telemetry_dir.mkdir(parents=True, exist_ok=True)


def find_test_pdf() -> Path:
    """Find a test PDF file to use."""
    test_data_dir = project_root / 'test_data'

    if test_data_dir.exists():
        pdfs = list(test_data_dir.glob('*.pdf'))
        if pdfs:
            return pdfs[0]

    return None


def test_pattern_extraction() -> bool:
    """Test 1: Pattern extraction works on sample text."""
    print(f"\n{'='*60}")
    print("TEST 1: Pattern Extraction")
    print(f"{'='*60}")

    try:
        extractor = Extractor()

        # Test cases with expected results
        test_cases = [
            (
                "SHEET NO: A101\nPROJECT: 2024-0156\nDATE: 12/15/2024\nSCALE: 1/4\" = 1'-0\"",
                {'sheet_number': 'A101', 'project_number': '2024-0156', 'date': '12/15/2024'}
            ),
            (
                "DRAWING NO. S-201\nJOB #: 123456\nISSUE DATE: 1-5-25",
                {'sheet_number': 'S-201', 'project_number': '123456'}
            ),
            (
                "DWG: M1.01\nPROJECT NUMBER: 2024.001",
                {'sheet_number': 'M1.01', 'project_number': '2024.001'}
            ),
        ]

        all_passed = True
        for text, expected in test_cases:
            result = extractor.extract_fields(text)
            print(f"\n  Input: {text[:50]}...")
            print(f"  Extracted: sheet={result['sheet_number']}, project={result['project_number']}")

            for field, expected_value in expected.items():
                if result.get(field) != expected_value:
                    print(f"    FAIL: Expected {field}={expected_value}, got {result.get(field)}")
                    all_passed = False
                else:
                    print(f"    OK: {field}={expected_value}")

        if all_passed:
            print(f"\n  RESULT: PASS")
        else:
            print(f"\n  RESULT: FAIL")

        return all_passed
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_validation() -> bool:
    """Test 2: Validation works correctly."""
    print(f"\n{'='*60}")
    print("TEST 2: Validation")
    print(f"{'='*60}")

    try:
        validator = Validator()

        # Test cases
        test_cases = [
            # Valid data
            (
                {'sheet_number': 'A101', 'project_number': '2024-001'},
                True, 0
            ),
            # Invalid: sheet_number starts with digit
            (
                {'sheet_number': '101A', 'project_number': '2024-001'},
                False, 1
            ),
            # Invalid: project_number not numeric
            (
                {'sheet_number': 'A101', 'project_number': 'ABC-DEF'},
                False, 1
            ),
            # Invalid: contamination (identical values)
            (
                {'sheet_number': 'A101', 'project_number': 'A101'},
                False, 1
            ),
        ]

        all_passed = True
        for data, expected_valid, expected_errors in test_cases:
            result = validator.validate(data)
            print(f"\n  Input: {data}")
            print(f"  Valid: {result['is_valid']} (expected: {expected_valid})")
            print(f"  Errors: {len(result['errors'])} (expected: {expected_errors})")

            if result['is_valid'] != expected_valid:
                print(f"    FAIL: Validity mismatch")
                all_passed = False
            elif len(result['errors']) != expected_errors:
                print(f"    FAIL: Error count mismatch")
                print(f"    Errors: {result['errors']}")
                all_passed = False
            else:
                print(f"    OK")

        if all_passed:
            print(f"\n  RESULT: PASS")
        else:
            print(f"\n  RESULT: FAIL")

        return all_passed
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_real_extraction(pdf_path: Path) -> bool:
    """Test 3: Full extraction pipeline on real PDF."""
    print(f"\n{'='*60}")
    print("TEST 3: Real PDF Extraction")
    print(f"{'='*60}")

    try:
        normalizer = PageNormalizer()
        detector = RegionDetector()
        ocr_engine = OCREngine()
        extractor = Extractor()
        validator = Validator()

        all_extractions = []

        with PDFHandler(pdf_path) as handler:
            for page_num in range(min(handler.page_count, 3)):
                print(f"\n  Page {page_num + 1}:")

                # Check if vector or scanned
                analysis = handler.analyze_page(page_num)

                if analysis['recommendation'] == 'vector':
                    # Use embedded text
                    text = handler.get_page_text(page_num)
                    text_blocks = handler.get_text_blocks(page_num)
                    source = 'vector'
                    print(f"    Source: Vector PDF (embedded text)")
                else:
                    # Use OCR
                    image = handler.get_page_image(page_num, dpi=200)
                    normalized, _ = normalizer.normalize(image)
                    detection = detector.detect_title_block(normalized)
                    cropped = detector.crop_title_block(normalized, detection)
                    text = ocr_engine.ocr_image(cropped)
                    text_blocks = None
                    source = f"ocr_{detection['region_name']}"
                    print(f"    Source: Scanned PDF (OCR from {detection['region_name']})")

                # Extract fields
                image_size = handler.get_page_image(page_num, dpi=200).size
                fields = extractor.extract_fields(text, text_blocks, image_size)

                print(f"    Sheet Number: {fields['sheet_number']}")
                print(f"    Project Number: {fields['project_number']}")
                print(f"    Date: {fields['date']}")
                print(f"    Discipline: {fields['discipline']}")

                # Validate
                validation = validator.validate(fields)
                print(f"    Valid: {validation['is_valid']}")
                if validation['errors']:
                    print(f"    Errors: {validation['errors']}")
                if validation['warnings']:
                    print(f"    Warnings: {validation['warnings'][:3]}")

                # Store for telemetry
                all_extractions.append({
                    'page': page_num + 1,
                    'source': source,
                    'fields': {k: v for k, v in fields.items() if k != 'extraction_details'},
                    'extraction_details': fields.get('extraction_details', {}),
                    'validation': validation,
                })

        # Save telemetry JSON
        telemetry_path = telemetry_dir / 'phase5_extraction_results.json'
        with open(telemetry_path, 'w', encoding='utf-8') as f:
            json.dump({
                'pdf': pdf_path.name,
                'extractions': all_extractions,
            }, f, indent=2, default=str)
        print(f"\n  Telemetry saved to: {telemetry_path}")

        print(f"\n  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_all_matches(pdf_path: Path) -> bool:
    """Test 4: Find all pattern matches (for debugging)."""
    print(f"\n{'='*60}")
    print("TEST 4: All Pattern Matches")
    print(f"{'='*60}")

    try:
        normalizer = PageNormalizer()
        detector = RegionDetector()
        ocr_engine = OCREngine()
        extractor = Extractor()

        with PDFHandler(pdf_path) as handler:
            # Test on one page
            page_num = min(1, handler.page_count - 1)
            print(f"\n  Analyzing page {page_num + 1}...")

            analysis = handler.analyze_page(page_num)

            if analysis['recommendation'] == 'vector':
                text = handler.get_page_text(page_num)
            else:
                image = handler.get_page_image(page_num, dpi=200)
                normalized, _ = normalizer.normalize(image)
                detection = detector.detect_title_block(normalized)
                cropped = detector.crop_title_block(normalized, detection)
                text = ocr_engine.ocr_image(cropped)

            # Get all matches
            all_matches = extractor.extract_all_matches(text)

            print(f"\n  All matches found:")
            for field, matches in all_matches.items():
                print(f"    {field}: {matches[:5]}{'...' if len(matches) > 5 else ''}")

        print(f"\n  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_contamination_check() -> bool:
    """Test 5: Contamination detection."""
    print(f"\n{'='*60}")
    print("TEST 5: Contamination Detection")
    print(f"{'='*60}")

    try:
        validator = Validator()

        test_cases = [
            # No contamination
            (
                {'sheet_number': 'A101', 'project_number': '2024-001'},
                False
            ),
            # Sheet starts with digit (contaminated)
            (
                {'sheet_number': '2024-001', 'project_number': '2024-001'},
                True
            ),
            # Project starts with letter (contaminated)
            (
                {'sheet_number': 'A101', 'project_number': 'A101'},
                True
            ),
        ]

        all_passed = True
        for data, expected_contaminated in test_cases:
            result = validator.check_contamination(data)
            print(f"\n  Input: {data}")
            print(f"  Has contamination: {result['has_contamination']} (expected: {expected_contaminated})")

            if result['has_contamination'] != expected_contaminated:
                print(f"    FAIL: Contamination detection mismatch")
                all_passed = False
            else:
                print(f"    OK")

            if result['issues']:
                for issue in result['issues']:
                    print(f"    Issue: {issue['field']} - {issue['issue']}")

        if all_passed:
            print(f"\n  RESULT: PASS")
        else:
            print(f"\n  RESULT: FAIL")

        return all_passed
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def run_checkpoint():
    """Run all Phase 5 checkpoint tests."""
    print("\n" + "=" * 60)
    print("PHASE 5 CHECKPOINT: Field Extraction & Validation")
    print("=" * 60)

    # Find test PDF
    pdf_path = find_test_pdf()
    if pdf_path is None:
        print("\nERROR: No test PDF found!")
        print("Please place a PDF file in test_data/sample.pdf")
        return False

    print(f"\nUsing test PDF: {pdf_path}")

    # Run tests
    results = []
    results.append(("Pattern Extraction", test_pattern_extraction()))
    results.append(("Validation", test_validation()))
    results.append(("Contamination Check", test_contamination_check()))
    results.append(("Real PDF Extraction", test_real_extraction(pdf_path)))
    results.append(("All Matches", test_all_matches(pdf_path)))

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("PHASE 5 CHECKPOINT: PASSED")
        print("You may proceed to Phase 6.")
    else:
        print("PHASE 5 CHECKPOINT: FAILED")
        print("Fix the failing tests before proceeding.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = run_checkpoint()
    sys.exit(0 if success else 1)
