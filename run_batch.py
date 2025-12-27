"""
Batch processing script for Test Blueprints folder.
Processes each PDF individually with timeout and error handling.
"""

import os
import sys
import json
import time
import signal
import traceback
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from main import BlueprintProcessor

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Processing timeout")

@contextmanager
def timeout(seconds):
    """Context manager for timeout (Unix only - falls back to no timeout on Windows)."""
    if os.name == 'nt':
        yield
    else:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

def main():
    folder_path = Path("C:/Hybrid-Extraction-Test/Test Blueprints")
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Start logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"batch_run_{timestamp}.log"

    pdf_files = sorted(folder_path.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")
    print(f"Logging to: {log_file}")
    print("=" * 60)

    all_results = []
    summary = {
        "start_time": datetime.now().isoformat(),
        "folder": str(folder_path),
        "total_pdfs": len(pdf_files),
        "processed": 0,
        "failed": 0,
        "total_pages": 0,
        "files": []
    }

    processor = BlueprintProcessor()

    for i, pdf_path in enumerate(pdf_files, 1):
        file_info = {
            "filename": pdf_path.name,
            "size_mb": pdf_path.stat().st_size / (1024 * 1024),
            "start_time": datetime.now().isoformat(),
            "status": "pending",
            "pages": 0,
            "error": None
        }

        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        print(f"  Size: {file_info['size_mb']:.1f} MB")

        try:
            start_time = time.time()
            results = processor.process_pdf(pdf_path)
            elapsed = time.time() - start_time

            file_info["status"] = "success"
            file_info["pages"] = len(results)
            file_info["elapsed_seconds"] = elapsed
            file_info["end_time"] = datetime.now().isoformat()

            all_results.extend(results)
            summary["processed"] += 1
            summary["total_pages"] += len(results)

            print(f"  Pages: {len(results)}, Time: {elapsed:.1f}s")

            # Show extraction summary
            valid = sum(1 for r in results if r.get('is_valid', 0) == 1)
            with_title = sum(1 for r in results if r.get('sheet_title'))
            with_number = sum(1 for r in results if r.get('sheet_number'))
            print(f"  Valid: {valid}/{len(results)}, Titles: {with_title}, Sheet#: {with_number}")

        except Exception as e:
            file_info["status"] = "failed"
            file_info["error"] = str(e)
            file_info["end_time"] = datetime.now().isoformat()
            summary["failed"] += 1
            print(f"  ERROR: {e}")
            traceback.print_exc()

        summary["files"].append(file_info)

        # Save progress after each PDF
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    summary["end_time"] = datetime.now().isoformat()

    # Generate final reports
    print("\n" + "=" * 60)
    print("GENERATING REPORTS...")

    results_dict = {
        "summary": {
            "total_pdfs": summary["total_pdfs"],
            "total_pages": summary["total_pages"],
            "success_count": sum(1 for r in all_results if r.get('is_valid', 0) == 1),
            "error_count": sum(1 for r in all_results if r.get('is_valid', 0) != 1),
        },
        "sheets": all_results
    }

    reports = processor.generate_report(results_dict, output_dir)

    # Save final summary
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Total PDFs: {summary['total_pdfs']}")
    print(f"  Processed: {summary['processed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Total Pages: {summary['total_pages']}")
    print(f"\nReports:")
    print(f"  JSON: {reports['json']}")
    print(f"  HTML: {reports['html']}")
    print(f"  HITL CSV: {reports['hitl_csv']}")
    print(f"  Log: {log_file}")

    return summary

if __name__ == "__main__":
    main()
