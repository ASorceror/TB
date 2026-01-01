"""
Sheet Classification Script (V7.0)

Classifies extracted blueprint sheets into categories and organizes files.
Uses enhanced three-tier classification system for painting trade relevance.

Usage:
    python classify_sheets.py --extraction-file results.json --pdfs-dir "path/to/pdfs" --output-dir classified/
    python classify_sheets.py --extraction-file results.json --pdfs-dir "path/to/pdfs" --output-dir classified/ --dry-run

Three-Tier Decisions:
    - DEFINITELY_NEEDED: Pages required for painting trade (floor plans, RCPs, schedules, elevations)
    - DEFINITELY_NOT_NEEDED: Pages not needed (MEP, structural, civil, landscape)
    - NEEDS_EVALUATION: Ambiguous pages requiring human review

Output includes:
    - JSON manifests (per-category and master)
    - CSV export (all classification data)
    - Crop images (title block extractions)
"""

import sys
import csv
import hashlib
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from core.sheet_classifier import SheetClassifier, ClassificationResult, ClassifiedSheet
from core.sheet_organizer import SheetOrganizer, organize_classified_sheets
from core.classification_manifest import ManifestGenerator, generate_manifests
from constants import SHEET_CATEGORIES


def print_summary(classification, organize_stats=None, csv_path=None, crops_copied=0):
    """Print classification summary to console."""
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY (V7.0 Enhanced)")
    print("=" * 80)

    stats = classification.statistics

    # Three-tier decision breakdown (V7.0)
    if 'by_decision' in stats:
        print(f"\n{'THREE-TIER DECISIONS':<35}")
        print("-" * 45)
        print(f"{'DEFINITELY_NEEDED':<35} {stats['by_decision'].get('DEFINITELY_NEEDED', 0):>8}")
        print(f"{'DEFINITELY_NOT_NEEDED':<35} {stats['by_decision'].get('DEFINITELY_NOT_NEEDED', 0):>8}")
        print(f"{'NEEDS_EVALUATION':<35} {stats['by_decision'].get('NEEDS_EVALUATION', 0):>8}")

    # Relevance breakdown (V7.0)
    if 'by_relevance' in stats:
        print(f"\n{'PAINTING TRADE RELEVANCE':<35}")
        print("-" * 45)
        print(f"{'PRIMARY (floor plans, RCPs, etc)':<35} {stats['by_relevance'].get('PRIMARY', 0):>8}")
        print(f"{'SECONDARY (elevations, sections)':<35} {stats['by_relevance'].get('SECONDARY', 0):>8}")
        print(f"{'REFERENCE (cover sheets, notes)':<35} {stats['by_relevance'].get('REFERENCE', 0):>8}")
        print(f"{'IRRELEVANT (MEP, structural)':<35} {stats['by_relevance'].get('IRRELEVANT', 0):>8}")

    # Category breakdown
    print(f"\n{'CATEGORY BREAKDOWN':<35}")
    print("-" * 45)
    print(f"{'Category':<35} {'Count':>8}")
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

    # Output files
    print(f"\nOutput Files:")
    if csv_path:
        print(f"  CSV:    {csv_path}")
    if crops_copied > 0:
        print(f"  Crops:  {crops_copied} title block images copied")


def compute_pdf_hash(pdf_path: Path) -> str:
    """Compute SHA-256 hash of PDF file (first 16 chars)."""
    hasher = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def export_csv(
    classification: ClassificationResult,
    output_path: Path,
    verbose: bool = False
) -> Path:
    """
    Export classification results to CSV.

    Args:
        classification: ClassificationResult from classifier
        output_path: Path to save CSV file
        verbose: Print progress

    Returns:
        Path to saved CSV file
    """
    csv_path = output_path / 'classification_results.csv'

    # Define CSV columns
    columns = [
        'pdf_name',
        'page_number',
        'sheet_number',
        'sheet_title',
        'decision',
        'relevance',
        'categories',
        'classification_type',
        'classification_confidence',
        'number_confidence',
        'title_confidence',
        'is_combo_page',
        'needs_review',
        'review_reason',
    ]

    # Collect all sheets
    all_sheets: List[ClassifiedSheet] = []
    for category, sheets in classification.categories.items():
        all_sheets.extend(sheets)

    # Sort by PDF name, then page number
    all_sheets.sort(key=lambda s: (s.pdf_name or '', s.page_number or 0))

    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for sheet in all_sheets:
            row = {
                'pdf_name': sheet.pdf_name,
                'page_number': sheet.page_number,
                'sheet_number': sheet.sheet_number,
                'sheet_title': sheet.sheet_title,
                'decision': sheet.decision,
                'relevance': sheet.relevance,
                'categories': ';'.join(sheet.categories) if sheet.categories else '',
                'classification_type': sheet.classification_type,
                'classification_confidence': f"{sheet.classification_confidence:.3f}" if sheet.classification_confidence else '',
                'number_confidence': f"{sheet.number_confidence:.3f}" if sheet.number_confidence else '',
                'title_confidence': f"{sheet.title_confidence:.3f}" if sheet.title_confidence else '',
                'is_combo_page': sheet.is_combo_page,
                'needs_review': sheet.needs_review,
                'review_reason': sheet.review_reason or '',
            }
            writer.writerow(row)

    if verbose:
        print(f"Exported {len(all_sheets)} rows to CSV: {csv_path}")

    return csv_path


def copy_crops(
    classification: ClassificationResult,
    pdfs_dir: Path,
    output_dir: Path,
    crops_base_dir: Optional[Path] = None,
    verbose: bool = False
) -> int:
    """
    Copy crop images to output directory.

    Args:
        classification: ClassificationResult from classifier
        pdfs_dir: Directory containing source PDFs
        output_dir: Output directory for crops
        crops_base_dir: Base directory for crops (default: output/crops)
        verbose: Print progress

    Returns:
        Number of crops copied
    """
    # Default crops directory
    if crops_base_dir is None:
        crops_base_dir = Path(__file__).parent / 'output' / 'crops'

    if not crops_base_dir.exists():
        if verbose:
            print(f"Crops directory not found: {crops_base_dir}")
        return 0

    # Create crops output directory
    crops_output = output_dir / 'crops'
    crops_output.mkdir(parents=True, exist_ok=True)

    # Build PDF hash lookup
    pdf_hashes: Dict[str, str] = {}

    # Get unique PDF names from classification
    pdf_names = set()
    for category, sheets in classification.categories.items():
        for sheet in sheets:
            if sheet.pdf_name:
                pdf_names.add(sheet.pdf_name)

    # Compute hashes for each PDF
    for pdf_name in pdf_names:
        # Try various extensions
        for ext in ['.pdf', '.PDF', '']:
            pdf_path = pdfs_dir / f"{pdf_name}{ext}"
            if pdf_path.exists():
                try:
                    pdf_hashes[pdf_name] = compute_pdf_hash(pdf_path)
                    if verbose:
                        print(f"  Hash for {pdf_name}: {pdf_hashes[pdf_name]}")
                except Exception as e:
                    if verbose:
                        print(f"  Error computing hash for {pdf_name}: {e}")
                break

    # Copy crops
    copied = 0
    for category, sheets in classification.categories.items():
        for sheet in sheets:
            if not sheet.pdf_name or sheet.pdf_name not in pdf_hashes:
                continue

            pdf_hash = pdf_hashes[sheet.pdf_name]
            page_num = sheet.page_number or 0

            # Source crop path
            crop_filename = f"p{page_num:03d}_titleblock.png"
            src_crop = crops_base_dir / pdf_hash / crop_filename

            if src_crop.exists():
                # Destination crop path (organized by PDF and page)
                dst_filename = f"{sheet.pdf_name}_p{page_num:03d}.png"
                dst_crop = crops_output / dst_filename

                try:
                    shutil.copy2(src_crop, dst_crop)
                    copied += 1
                except Exception as e:
                    if verbose:
                        print(f"  Error copying crop {src_crop}: {e}")

    if verbose:
        print(f"Copied {copied} crop images to {crops_output}")

    return copied


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
        print("SHEET CLASSIFICATION SYSTEM V7.0 (Enhanced)")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if args.dry_run:
            print("MODE: DRY RUN (no files will be extracted)")
        print("=" * 80)
        print(f"\nExtraction file: {extraction_file}")
        if pdfs_dir:
            print(f"Source PDFs: {pdfs_dir}")
        print(f"Output directory: {output_dir}")

    # Step 1: Classification (using V7.0 enhanced classification)
    if not args.quiet:
        print("\n" + "-" * 40)
        print("STEP 1: Classifying sheets (V7.0 Enhanced)...")
        print("-" * 40)

    # Use enhanced classification by default
    classifier = SheetClassifier(use_enhanced=True)
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

    # Step 3: Generate Manifests (JSON)
    if not args.quiet:
        print("\n" + "-" * 40)
        action = "STEP 3: Generating JSON manifests..." if not args.dry_run else "STEP 3: Would generate manifests..."
        print(action)
        print("-" * 40)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_generator = ManifestGenerator(output_dir)
        manifest_files = manifest_generator.save_manifests(
            classification,
            organize_stats,
            verbose=verbose
        )
        if not args.quiet:
            print(f"\nGenerated {len(manifest_files)} manifest files")
    else:
        manifest_files = []
        if not args.quiet:
            print("(Skipped in dry-run mode)")

    # Step 4: Export CSV (always)
    csv_path = None
    if not args.dry_run:
        if not args.quiet:
            print("\n" + "-" * 40)
            print("STEP 4: Exporting CSV...")
            print("-" * 40)

        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = export_csv(classification, output_dir, verbose=verbose)

        if not args.quiet:
            print(f"Exported classification data to: {csv_path}")

    # Step 5: Copy Crops (when PDFs directory available)
    crops_copied = 0
    if not args.dry_run and pdfs_dir:
        if not args.quiet:
            print("\n" + "-" * 40)
            print("STEP 5: Copying crop images...")
            print("-" * 40)

        crops_copied = copy_crops(
            classification,
            pdfs_dir,
            output_dir,
            verbose=verbose
        )

        if not args.quiet:
            if crops_copied > 0:
                print(f"Copied {crops_copied} crop images")
            else:
                print("No crop images found to copy")

    # Summary
    if not args.quiet:
        print_summary(classification, organize_stats, csv_path, crops_copied)

        print("\n" + "=" * 80)
        if args.dry_run:
            print("DRY RUN COMPLETE - No files were modified")
        else:
            print(f"CLASSIFICATION COMPLETE")
            print(f"Output saved to: {output_dir}")
            print(f"\nOutput contents:")
            print(f"  - classification_manifest.json (master manifest)")
            print(f"  - classification_results.csv (all pages)")
            if crops_copied > 0:
                print(f"  - crops/ ({crops_copied} title block images)")
            print(f"  - Per-category folders with manifests")
        print("=" * 80)


if __name__ == "__main__":
    main()
