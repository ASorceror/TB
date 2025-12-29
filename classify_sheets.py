"""
Sheet Classification Script (V6.1)

Classifies extracted blueprint sheets into categories and organizes files.
Takes extraction results JSON and extracts single-page PDFs to category folders.

Usage:
    python classify_sheets.py --extraction-file results.json --pdfs-dir "path/to/pdfs" --output-dir classified/
    python classify_sheets.py --extraction-file results.json --pdfs-dir "path/to/pdfs" --output-dir classified/ --dry-run

Categories:
    - floor_plans: Floor plans and overall plans
    - reflected_ceiling_plans: RCP sheets
    - room_finish_schedules: Finish schedules
    - interior_elevations: Interior elevations
    - exterior_elevations: Exterior elevations
    - cover_sheets: Cover sheets, indexes, general notes
    - properly_classified_not_needed: Recognized MEP/structural sheets (manifest only)
    - unclassified_may_be_needed: Unknown sheet types (files + manifest)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.sheet_classifier import SheetClassifier, classify_sheets
from core.sheet_organizer import SheetOrganizer, organize_classified_sheets
from core.classification_manifest import ManifestGenerator, generate_manifests
from constants import SHEET_CATEGORIES


def print_summary(classification, organize_stats=None):
    """Print classification summary to console."""
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)

    stats = classification.statistics

    # Category breakdown
    print(f"\n{'Category':<35} {'Count':>8}")
    print("-" * 45)

    total_matched = 0
    for category in SHEET_CATEGORIES.keys():
        count = stats['by_category'].get(category, 0)
        total_matched += count
        if count > 0:
            print(f"{category:<35} {count:>8}")

    # Unclassified categories
    not_needed = stats['by_category'].get('properly_classified_not_needed', 0)
    unclassified = stats['by_category'].get('unclassified_may_be_needed', 0)

    print("-" * 45)
    print(f"{'Total in target categories':<35} {total_matched:>8}")
    print(f"{'Recognized (not needed)':<35} {not_needed:>8}")
    print(f"{'Unclassified (may need)':<35} {unclassified:>8}")
    print("-" * 45)
    print(f"{'TOTAL PAGES':<35} {classification.total_pages:>8}")

    # Classification type breakdown
    print(f"\nClassification Types:")
    print(f"  Matched to categories: {stats['classification_types']['matched']}")
    print(f"  Recognized not needed: {stats['classification_types']['not_needed']}")
    print(f"  Unclassified:          {stats['classification_types']['unclassified']}")

    # Organization stats if available
    if organize_stats:
        print(f"\nPDF Extraction:")
        print(f"  Folders created:  {organize_stats.folders_created}")
        print(f"  Pages extracted:  {organize_stats.files_extracted}")
        print(f"  Pages skipped:    {organize_stats.files_skipped}")
        if organize_stats.errors > 0:
            print(f"  Errors:           {organize_stats.errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify extracted blueprint sheets and extract single-page PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Classify and extract single-page PDFs
    python classify_sheets.py --extraction-file final_extraction_results.json \\
                              --pdfs-dir "C:/Hybrid-Extraction-Test/Test Blueprints" \\
                              --output-dir output/classified_sheets

    # Preview without extracting files
    python classify_sheets.py --extraction-file results.json --pdfs-dir pdfs/ \\
                              --output-dir output/ --dry-run

    # Classify only (no file extraction)
    python classify_sheets.py --extraction-file results.json --classify-only
        """
    )

    parser.add_argument(
        '--extraction-file', type=str, required=True,
        help='Path to extraction results JSON (e.g., final_extraction_results.json)'
    )
    parser.add_argument(
        '--pdfs-dir', type=str, default=None,
        help='Directory containing source PDF files (required unless --classify-only)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for extracted PDFs (default: auto-generated)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview what would be done without extracting files'
    )
    parser.add_argument(
        '--classify-only', action='store_true',
        help='Only classify sheets, do not extract files'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print detailed progress during processing'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress all output except errors'
    )

    args = parser.parse_args()

    # Validate arguments
    extraction_file = Path(args.extraction_file)
    if not extraction_file.exists():
        print(f"ERROR: Extraction file not found: {extraction_file}")
        sys.exit(1)

    if not args.classify_only and not args.pdfs_dir:
        print("ERROR: --pdfs-dir is required unless using --classify-only")
        sys.exit(1)

    pdfs_dir = Path(args.pdfs_dir) if args.pdfs_dir else None
    if pdfs_dir and not pdfs_dir.exists():
        print(f"ERROR: PDFs directory not found: {pdfs_dir}")
        sys.exit(1)

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"classified_output_{timestamp}")

    verbose = args.verbose and not args.quiet

    # Header
    if not args.quiet:
        print("=" * 80)
        print("SHEET CLASSIFICATION SYSTEM V6.1")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if args.dry_run:
            print("MODE: DRY RUN (no files will be extracted)")
        print("=" * 80)
        print(f"\nExtraction file: {extraction_file}")
        if pdfs_dir:
            print(f"Source PDFs: {pdfs_dir}")
        print(f"Output directory: {output_dir}")

    # Step 1: Classification
    if not args.quiet:
        print("\n" + "-" * 40)
        print("STEP 1: Classifying sheets...")
        print("-" * 40)

    classifier = SheetClassifier()
    classification = classifier.classify_from_file(extraction_file, verbose=verbose)

    if not args.quiet:
        print(f"Classified {classification.total_pages} pages into categories")

    # Step 2: PDF Extraction (unless classify-only)
    organize_stats = None
    if not args.classify_only and pdfs_dir:
        if not args.quiet:
            print("\n" + "-" * 40)
            action = "STEP 2: Previewing PDF extraction..." if args.dry_run else "STEP 2: Extracting PDF pages..."
            print(action)
            print("-" * 40)

        organizer = SheetOrganizer(pdfs_dir, output_dir)
        organize_stats = organizer.organize(
            classification,
            dry_run=args.dry_run,
            verbose=verbose
        )

        if not args.quiet:
            action = "Would extract" if args.dry_run else "Extracted"
            print(f"\n{action} {organize_stats.files_extracted} pages to {organize_stats.folders_created} folders")
            if organize_stats.files_skipped > 0:
                print(f"Skipped {organize_stats.files_skipped} pages (PDF not found or extraction failed)")
            if organize_stats.errors > 0:
                print(f"Encountered {organize_stats.errors} errors")

    # Step 3: Generate Manifests
    if not args.quiet:
        print("\n" + "-" * 40)
        action = "STEP 3: Generating manifests..." if not args.dry_run else "STEP 3: Would generate manifests..."
        print(action)
        print("-" * 40)

    if not args.dry_run:
        manifest_generator = ManifestGenerator(output_dir)
        manifest_files = manifest_generator.save_manifests(
            classification,
            organize_stats,
            verbose=verbose
        )
        if not args.quiet:
            print(f"\nGenerated {len(manifest_files)} manifest files")
    else:
        if not args.quiet:
            print("(Skipped in dry-run mode)")

    # Summary
    if not args.quiet:
        print_summary(classification, organize_stats)

        print("\n" + "=" * 80)
        if args.dry_run:
            print("DRY RUN COMPLETE - No files were modified")
        else:
            print(f"CLASSIFICATION COMPLETE")
            print(f"Output saved to: {output_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
