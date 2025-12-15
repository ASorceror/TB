"""
Blueprint Processor V4.2.1 - Phase 5 Tests
Tests for Integration & HITL Reports.
"""

import sys
import os
import tempfile
import csv
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sheet_title_extractor import SheetTitleExtractor
from reports.hitl_report import HITLReportGenerator
from database.models import ExtractedSheet, Base
from database.operations import DatabaseOperations


def test_pdf_hash():
    """Test PDF hash computation."""
    print("=== Test: PDF Hash Computation ===")

    extractor = SheetTitleExtractor()
    sample_path = Path(__file__).parent.parent / 'test_data' / 'sample.pdf'

    if not sample_path.exists():
        print(f"  Skipping - {sample_path} not found")
        return

    # Compute hash
    hash1 = extractor.compute_pdf_hash(sample_path)
    print(f"  Hash: {hash1}")
    assert len(hash1) == 16, f"Hash should be 16 chars, got {len(hash1)}"

    # Same file should produce same hash
    hash2 = extractor.compute_pdf_hash(sample_path)
    assert hash1 == hash2, "Same file should produce same hash"

    print("PASSED\n")


def test_cross_reference_validation():
    """Test cross-reference validation using drawing index."""
    print("=== Test: Cross-Reference Validation ===")

    extractor = SheetTitleExtractor()

    # Set up a mock drawing index (on both extractor levels)
    mock_index = {
        'A2.0': 'FLOOR PLAN',
        'A2.1': 'ENLARGED FLOOR PLAN',
        'A2.2': 'REFLECTED CEILING PLAN',
    }
    extractor._drawing_index = mock_index
    extractor._extractor._drawing_index = mock_index

    # Mock extraction result with lower-confidence method
    result = {
        'sheet_number': 'A2.0',
        'sheet_title': 'Floor',  # Incomplete title
        'title_method': 'pattern',
        'title_confidence': 0.70,
        'needs_review': True,
    }

    # Apply cross-reference
    updated = extractor._cross_reference_validate(result)

    print(f"  Original title: 'Floor'")
    print(f"  Cross-ref title: '{updated['sheet_title']}'")
    print(f"  Method: {updated['title_method']}")

    assert updated['sheet_title'] == 'FLOOR PLAN', "Should use index title"
    assert updated['title_method'] == 'drawing_index_xref', "Method should be xref"
    assert updated['title_confidence'] == 0.95, "Confidence should be 0.95"

    print("PASSED\n")


def test_hitl_csv_generation():
    """Test HITL CSV report generation."""
    print("=== Test: HITL CSV Generation ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = HITLReportGenerator(output_dir=Path(tmpdir))

        # Mock results
        results = [
            {
                'pdf_filename': 'test.pdf',
                'page_number': 1,
                'pdf_hash': 'abc123',
                'sheet_number': 'A1.0',
                'sheet_title': 'FLOOR PLAN',
                'title_confidence': 0.95,
                'title_method': 'drawing_index',
                'needs_review': False,
                'extraction_details': {},
            },
            {
                'pdf_filename': 'test.pdf',
                'page_number': 2,
                'pdf_hash': 'abc123',
                'sheet_number': 'A2.0',
                'sheet_title': 'Unknown',
                'title_confidence': 0.60,
                'title_method': 'pattern',
                'needs_review': True,
                'extraction_details': {'index_title': 'ELEVATIONS'},
            },
        ]

        stats = {
            'total_pages': 2,
            'by_method': {'drawing_index': 1, 'pattern': 1},
            'success_rate': 1.0,
            'pdf_hash': 'abc123',
        }

        reports = reporter.generate_reports(results, stats, 'test')

        # Check CSV exists
        assert reports['csv'].exists(), "CSV file should exist"
        print(f"  CSV: {reports['csv'].name}")

        # Check CSV content
        with open(reports['csv'], 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2, f"Should have 2 rows, got {len(rows)}"
        print(f"  Rows: {len(rows)}")

        # First row should be the one needing review
        assert rows[0]['needs_review'] == 'YES', "First row should need review"
        assert rows[0]['suggested_title'] == 'ELEVATIONS', "Should have suggested title"

        print("PASSED\n")


def test_hitl_html_generation():
    """Test HITL HTML report generation."""
    print("=== Test: HITL HTML Generation ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = HITLReportGenerator(output_dir=Path(tmpdir))

        results = [
            {
                'pdf_filename': 'test.pdf',
                'page_number': 1,
                'pdf_hash': 'abc123',
                'sheet_number': 'A1.0',
                'sheet_title': 'FLOOR PLAN',
                'title_confidence': 0.95,
                'title_method': 'drawing_index',
                'needs_review': False,
            },
        ]

        stats = {
            'total_pages': 1,
            'by_method': {'drawing_index': 1},
            'success_rate': 1.0,
            'pdf_hash': 'abc123',
        }

        reports = reporter.generate_reports(results, stats, 'test')

        # Check HTML exists
        assert reports['html'].exists(), "HTML file should exist"
        print(f"  HTML: {reports['html'].name}")

        # Check HTML content
        with open(reports['html'], 'r', encoding='utf-8') as f:
            html = f.read()

        assert 'HITL Review Report' in html, "Should have title"
        assert 'FLOOR PLAN' in html, "Should contain sheet title"
        assert 'drawing_index' in html, "Should show method"

        print("PASSED\n")


def test_hitl_import_corrections():
    """Test importing corrections from CSV."""
    print("=== Test: HITL Import Corrections ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a corrections CSV
        csv_path = Path(tmpdir) / 'corrections.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'pdf_filename', 'page_number', 'pdf_hash', 'sheet_number',
                'extracted_title', 'confidence', 'method', 'needs_review',
                'review_reason', 'suggested_title', 'corrected_title'
            ])
            writer.writerow([
                'test.pdf', '1', 'abc123', 'A1.0',
                'Wrong Title', '0.60', 'pattern', 'YES',
                'Low confidence', 'FLOOR PLAN', 'CORRECTED FLOOR PLAN'
            ])
            writer.writerow([
                'test.pdf', '2', 'abc123', 'A2.0',
                'OK Title', '0.95', 'drawing_index', 'NO',
                '', '', ''  # No correction
            ])

        reporter = HITLReportGenerator(output_dir=Path(tmpdir))
        corrections = reporter.import_corrections(csv_path)

        print(f"  Corrections imported: {len(corrections)}")
        assert len(corrections) == 1, "Should have 1 correction"

        correction = corrections[0]
        assert correction['corrected_title'] == 'CORRECTED FLOOR PLAN'
        assert correction['original_title'] == 'Wrong Title'
        assert correction['page_number'] == 1

        print("PASSED\n")


def test_database_new_fields():
    """Test database model has new V4.2.1 fields."""
    print("=== Test: Database New Fields ===")

    # Use a fixed temp path to avoid Windows file locking issues
    import uuid
    db_path = Path(tempfile.gettempdir()) / f'test_bp_{uuid.uuid4().hex[:8]}.db'

    try:
        db = DatabaseOperations(db_path)

        # Create a sheet with new fields
        sheet_data = {
            'pdf_filename': 'test.pdf',
            'page_number': 1,
            'pdf_hash': 'abc123def456',
            'sheet_number': 'A1.0',
            'sheet_title': 'Floor Plan',
            'title_confidence': 0.95,
            'title_method': 'drawing_index',
            'needs_review': 0,
        }

        sheet = db.upsert_sheet(sheet_data)

        print(f"  Created sheet ID: {sheet.id}")
        print(f"  PDF hash: {sheet.pdf_hash}")
        print(f"  Title confidence: {sheet.title_confidence}")
        print(f"  Title method: {sheet.title_method}")
        print(f"  Needs review: {sheet.needs_review}")

        assert sheet.pdf_hash == 'abc123def456', "Should store pdf_hash"
        assert sheet.title_confidence == 0.95, "Should store title_confidence"
        assert sheet.title_method == 'drawing_index', "Should store title_method"
        assert sheet.needs_review == 0, "Should store needs_review"

        # Test to_dict includes new fields
        data = sheet.to_dict()
        assert 'pdf_hash' in data, "to_dict should include pdf_hash"
        assert 'title_confidence' in data, "to_dict should include title_confidence"
        assert 'title_method' in data, "to_dict should include title_method"
        assert 'needs_review' in data, "to_dict should include needs_review"

        # Close engine to release file lock
        db.engine.dispose()

        print("PASSED\n")
    finally:
        # Clean up
        try:
            if db_path.exists():
                db_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors on Windows


def test_extraction_stats():
    """Test extraction statistics tracking."""
    print("=== Test: Extraction Stats ===")

    extractor = SheetTitleExtractor()

    # Simulate some extractions
    extractor._extraction_stats['drawing_index'] = 5
    extractor._extraction_stats['spatial'] = 2
    extractor._extraction_stats['pattern'] = 3
    extractor._extraction_stats['failed'] = 1

    stats = extractor.get_extraction_stats()

    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Success rate: {stats['success_rate']:.0%}")
    print(f"  By method: {stats['by_method']}")

    assert stats['total_pages'] == 11, "Total should be 11"
    assert stats['success_rate'] == 10/11, "Success rate should be 10/11"

    print("PASSED\n")


def test_full_integration_with_sample():
    """Integration test with actual sample.pdf."""
    print("=== Test: Full Integration with sample.pdf ===")

    sample_path = Path(__file__).parent.parent / 'test_data' / 'sample.pdf'

    if not sample_path.exists():
        print(f"  Skipping - {sample_path} not found")
        return

    extractor = SheetTitleExtractor()

    # Process the PDF
    results, stats = extractor.process_pdf(sample_path)

    print(f"  Pages processed: {len(results)}")
    print(f"  Success rate: {stats['success_rate']:.0%}")
    print(f"  PDF hash: {stats['pdf_hash']}")
    print(f"\n  By method:")
    for method, count in stats['by_method'].items():
        if count > 0:
            print(f"    {method}: {count}")

    print(f"\n  Results:")
    for result in results:
        print(f"    Page {result['page_number']}: {result.get('sheet_number', '-')} - {result.get('sheet_title', '-')} ({result.get('title_method', 'none')})")

    assert len(results) > 0, "Should have results"
    assert stats['pdf_hash'] is not None, "Should have PDF hash"

    print("PASSED\n")


def test_needs_review_filtering():
    """Test filtering results that need review."""
    print("=== Test: Needs Review Filtering ===")

    extractor = SheetTitleExtractor()

    results = [
        {'sheet_number': 'A1.0', 'needs_review': False, 'title_confidence': 0.95},
        {'sheet_number': 'A2.0', 'needs_review': True, 'title_confidence': 0.60},
        {'sheet_number': 'A3.0', 'needs_review': False, 'title_confidence': 0.85},
        {'sheet_number': 'A4.0', 'needs_review': True, 'title_confidence': 0.55},
    ]

    needs_review = extractor.get_sheets_needing_review(results)

    print(f"  Total sheets: {len(results)}")
    print(f"  Needs review: {len(needs_review)}")

    assert len(needs_review) == 2, "Should have 2 sheets needing review"
    assert all(r['needs_review'] for r in needs_review), "All should need review"

    print("PASSED\n")


def run_all_tests():
    """Run all Phase 5 tests."""
    print("=" * 60)
    print("PHASE 5 TESTS: Integration & HITL Reports")
    print("=" * 60 + "\n")

    try:
        test_pdf_hash()
        test_cross_reference_validation()
        test_hitl_csv_generation()
        test_hitl_html_generation()
        test_hitl_import_corrections()
        test_database_new_fields()
        test_extraction_stats()
        test_needs_review_filtering()
        test_full_integration_with_sample()

        print("=" * 60)
        print("=== ALL PHASE 5 TESTS PASSED ===")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n!!! TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n!!! ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
