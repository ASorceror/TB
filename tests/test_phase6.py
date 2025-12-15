"""
Blueprint Processor V4.1 - Phase 6 Checkpoint Test
Verifies: Database operations and batch processing.

Run: python tests/test_phase6.py

CRITICAL: Verify no duplicates when running twice!
"""

import json
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.models import ExtractedSheet, ProcessingRun, Base
from database.operations import DatabaseOperations

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


def get_test_db_path() -> Path:
    """Get path for test database."""
    return project_root / 'data' / 'test_blueprint_data.db'


def cleanup_test_db():
    """Remove test database if it exists."""
    db_path = get_test_db_path()
    if db_path.exists():
        try:
            db_path.unlink()
        except PermissionError:
            pass  # Windows may lock the file; ignore cleanup errors


def test_database_creation() -> bool:
    """Test 1: Database and tables are created correctly."""
    print(f"\n{'='*60}")
    print("TEST 1: Database Creation")
    print(f"{'='*60}")

    try:
        cleanup_test_db()
        db_path = get_test_db_path()

        db = DatabaseOperations(db_path)

        # Check database file exists
        if not db_path.exists():
            print(f"  FAIL: Database file not created at {db_path}")
            return False

        print(f"  Database created at: {db_path}")
        print(f"  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_upsert_insert() -> bool:
    """Test 2: Upsert creates new records."""
    print(f"\n{'='*60}")
    print("TEST 2: Upsert - Insert New Records")
    print(f"{'='*60}")

    try:
        db = DatabaseOperations(get_test_db_path())

        # Insert test data
        test_data = {
            'pdf_filename': 'test_file.pdf',
            'page_number': 1,
            'sheet_number': 'A101',
            'project_number': '2024-001',
            'discipline': 'Architectural',
            'confidence': 'high',
            'extraction_method': 'vector',
            'is_valid': 1,
        }

        sheet = db.upsert_sheet(test_data)

        print(f"  Inserted: {sheet}")
        print(f"  ID: {sheet.id}")
        print(f"  Sheet Number: {sheet.sheet_number}")

        # Verify record exists
        retrieved = db.get_sheet('test_file.pdf', 1)
        if retrieved is None:
            print(f"  FAIL: Could not retrieve inserted record")
            return False

        print(f"  Retrieved: {retrieved.sheet_number}")
        print(f"  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_upsert_update() -> bool:
    """Test 3: Upsert updates existing records."""
    print(f"\n{'='*60}")
    print("TEST 3: Upsert - Update Existing Records")
    print(f"{'='*60}")

    try:
        db = DatabaseOperations(get_test_db_path())

        # Get current count
        initial_count = db.count_sheets()
        print(f"  Initial count: {initial_count}")

        # Update existing record (same filename + page)
        updated_data = {
            'pdf_filename': 'test_file.pdf',
            'page_number': 1,
            'sheet_number': 'A101-UPDATED',
            'project_number': '2024-002',
            'discipline': 'Architectural',
            'confidence': 'medium',
            'extraction_method': 'ocr',
            'is_valid': 1,
        }

        sheet = db.upsert_sheet(updated_data)

        # Verify count hasn't increased (no duplicate created)
        final_count = db.count_sheets()
        print(f"  Final count: {final_count}")

        if final_count != initial_count:
            print(f"  FAIL: Record count changed from {initial_count} to {final_count}")
            print(f"  This indicates a duplicate was created!")
            return False

        # Verify data was updated
        retrieved = db.get_sheet('test_file.pdf', 1)
        if retrieved.sheet_number != 'A101-UPDATED':
            print(f"  FAIL: Sheet number not updated")
            return False

        print(f"  Updated sheet_number: {retrieved.sheet_number}")
        print(f"  Updated project_number: {retrieved.project_number}")
        print(f"  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_no_duplicates_on_rerun() -> bool:
    """Test 4: CRITICAL - No duplicates when processing twice."""
    print(f"\n{'='*60}")
    print("TEST 4: No Duplicates on Re-run (CRITICAL)")
    print(f"{'='*60}")

    try:
        db = DatabaseOperations(get_test_db_path())

        # Insert multiple records
        test_records = [
            {'pdf_filename': 'dup_test.pdf', 'page_number': 1, 'sheet_number': 'A101'},
            {'pdf_filename': 'dup_test.pdf', 'page_number': 2, 'sheet_number': 'A102'},
            {'pdf_filename': 'dup_test.pdf', 'page_number': 3, 'sheet_number': 'A103'},
        ]

        print(f"  First run: inserting {len(test_records)} records...")
        for data in test_records:
            db.upsert_sheet(data)

        count_after_first = db.count_sheets()
        print(f"  Count after first run: {count_after_first}")

        # Run again (simulating re-processing)
        print(f"  Second run: upserting same {len(test_records)} records...")
        for data in test_records:
            db.upsert_sheet(data)

        count_after_second = db.count_sheets()
        print(f"  Count after second run: {count_after_second}")

        if count_after_second != count_after_first:
            print(f"  FAIL: Count changed from {count_after_first} to {count_after_second}")
            print(f"  DUPLICATES WERE CREATED!")
            return False

        print(f"  No duplicates created!")
        print(f"  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_processing_run() -> bool:
    """Test 5: Processing run tracking."""
    print(f"\n{'='*60}")
    print("TEST 5: Processing Run Tracking")
    print(f"{'='*60}")

    try:
        db = DatabaseOperations(get_test_db_path())

        # Start a run
        run = db.start_processing_run('/test/folder')
        print(f"  Started run: {run.id}")
        print(f"  Status: {run.status}")

        # Complete the run
        completed = db.complete_processing_run(
            run.id,
            pdf_count=5,
            page_count=25,
            success_count=23,
            error_count=2
        )

        print(f"  Completed run: {completed.status}")
        print(f"  Stats: {completed.pdf_count} PDFs, {completed.page_count} pages")

        # Get recent runs
        runs = db.get_processing_runs(5)
        print(f"  Recent runs: {len(runs)}")

        print(f"  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def test_full_processing() -> bool:
    """Test 6: Full processing with real PDF."""
    print(f"\n{'='*60}")
    print("TEST 6: Full Processing Pipeline")
    print(f"{'='*60}")

    pdf_path = find_test_pdf()
    if pdf_path is None:
        print("  SKIP: No test PDF available")
        return True

    try:
        # Import processor
        sys.path.insert(0, str(project_root))
        from main import BlueprintProcessor

        # Use test database
        processor = BlueprintProcessor(db_path=get_test_db_path())

        # Get initial count
        initial_count = processor.db.count_sheets()
        print(f"  Initial sheet count: {initial_count}")

        # Process once
        print(f"  First processing of {pdf_path.name}...")
        results1 = processor.process(pdf_path)
        count_after_first = processor.db.count_sheets()
        print(f"  Count after first run: {count_after_first}")
        print(f"  Pages processed: {results1['summary']['total_pages']}")

        # Process again
        print(f"  Second processing of {pdf_path.name}...")
        results2 = processor.process(pdf_path)
        count_after_second = processor.db.count_sheets()
        print(f"  Count after second run: {count_after_second}")

        # Verify no duplicates
        if count_after_second != count_after_first:
            print(f"  FAIL: Duplicate records created!")
            return False

        print(f"  No duplicates - upsert working correctly!")

        # Save telemetry JSON
        telemetry_data = {
            'pdf': pdf_path.name,
            'initial_count': initial_count,
            'count_after_first_run': count_after_first,
            'count_after_second_run': count_after_second,
            'duplicates_created': count_after_second != count_after_first,
            'summary': results1['summary'],
            'sheets': results1['sheets'],
        }
        telemetry_path = telemetry_dir / 'phase6_database_results.json'
        with open(telemetry_path, 'w', encoding='utf-8') as f:
            json.dump(telemetry_data, f, indent=2, default=str)
        print(f"\n  Telemetry saved to: {telemetry_path}")

        print(f"  RESULT: PASS")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"  RESULT: FAIL")
        return False


def run_checkpoint():
    """Run all Phase 6 checkpoint tests."""
    print("\n" + "=" * 60)
    print("PHASE 6 CHECKPOINT: Database & Batch Processing")
    print("=" * 60)

    # Clean up before tests
    cleanup_test_db()

    # Run tests
    results = []
    results.append(("Database Creation", test_database_creation()))
    results.append(("Upsert Insert", test_upsert_insert()))
    results.append(("Upsert Update", test_upsert_update()))
    results.append(("No Duplicates on Re-run", test_no_duplicates_on_rerun()))
    results.append(("Processing Run Tracking", test_processing_run()))
    results.append(("Full Processing Pipeline", test_full_processing()))

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
        print("PHASE 6 CHECKPOINT: PASSED")
        print("You may proceed to Phase 7.")
    else:
        print("PHASE 6 CHECKPOINT: FAILED")
        print("Fix the failing tests before proceeding.")
    print("=" * 60)

    # Cleanup
    cleanup_test_db()

    return all_passed


if __name__ == '__main__':
    success = run_checkpoint()
    sys.exit(0 if success else 1)
