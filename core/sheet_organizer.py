"""
Sheet Organizer Module (V6.1)

Handles file organization for classified sheets.
Extracts individual PDF pages from source PDFs and saves as single-page PDFs.

Classes:
    SheetOrganizer: PDF page extraction and folder management
"""

import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sheet_classifier import ClassificationResult, ClassifiedSheet
from constants import SHEET_CATEGORIES


@dataclass
class OrganizeStats:
    """Statistics from organization operation."""
    folders_created: int = 0
    files_extracted: int = 0
    files_skipped: int = 0
    errors: int = 0
    error_details: List[str] = None

    def __post_init__(self):
        if self.error_details is None:
            self.error_details = []


class SheetOrganizer:
    """
    Organizes classified sheets into folder structure.

    Extracts individual PDF pages from source PDFs and saves as single-page PDFs
    in category folders. Supports multi-category sheets (extracts to all matching folders).
    """

    def __init__(self, pdfs_dir: Path, output_dir: Path):
        """
        Initialize organizer.

        Args:
            pdfs_dir: Directory containing source PDF files
            output_dir: Base directory for organized output
        """
        self.pdfs_dir = Path(pdfs_dir)
        self.output_dir = Path(output_dir)
        self._pdf_cache: Dict[str, fitz.Document] = {}

    def _sanitize_filename(self, text: Optional[str], max_length: int = 50) -> str:
        """
        Sanitize text for use in filename.

        Args:
            text: Text to sanitize
            max_length: Maximum length for output

        Returns:
            Safe filename string
        """
        if not text:
            return "untitled"

        # Replace unsafe characters with underscores
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', text)
        # Replace multiple underscores/spaces with single underscore
        safe = re.sub(r'[_\s]+', '_', safe)
        # Remove leading/trailing underscores
        safe = safe.strip('_')
        # Truncate
        if len(safe) > max_length:
            safe = safe[:max_length].rstrip('_')

        return safe or "untitled"

    def _build_filename(self, sheet: ClassifiedSheet) -> str:
        """
        Build output filename for a sheet.

        Format: {source_pdf_name}_{sheet_number}_{sheet_title}.pdf

        Args:
            sheet: ClassifiedSheet instance

        Returns:
            Filename string
        """
        # Get PDF name without extension
        pdf_base = sheet.pdf_name.replace('.pdf', '').replace('.PDF', '')
        pdf_part = self._sanitize_filename(pdf_base, max_length=40)
        number_part = self._sanitize_filename(sheet.sheet_number, max_length=20)
        title_part = self._sanitize_filename(sheet.sheet_title, max_length=50)

        return f"{pdf_part}_{number_part}_{title_part}.pdf"

    def _normalize_name(self, name: str) -> str:
        """Normalize a filename for comparison."""
        # Remove extension, lowercase, replace underscores with spaces
        normalized = name.lower()
        if normalized.endswith('.pdf'):
            normalized = normalized[:-4]
        normalized = normalized.replace('_', ' ').replace('-', ' ')
        # Collapse multiple spaces
        normalized = ' '.join(normalized.split())
        return normalized

    def _find_source_pdf(self, pdf_name: str) -> Optional[Path]:
        """
        Find the source PDF file.

        Args:
            pdf_name: PDF filename from extraction results (may be truncated/sanitized)

        Returns:
            Path to PDF file, or None if not found
        """
        # Try exact match first
        pdf_path = self.pdfs_dir / pdf_name
        if pdf_path.exists():
            return pdf_path

        # Try with .pdf extension
        if not pdf_name.lower().endswith('.pdf'):
            pdf_path = self.pdfs_dir / f"{pdf_name}.pdf"
            if pdf_path.exists():
                return pdf_path

        # Normalize the search name
        search_normalized = self._normalize_name(pdf_name)

        # Search for matching PDFs
        for f in self.pdfs_dir.iterdir():
            if not f.is_file() or f.suffix.lower() != '.pdf':
                continue

            file_normalized = self._normalize_name(f.name)

            # Exact normalized match
            if file_normalized == search_normalized:
                return f

            # Check if file starts with search name (extraction truncates at 30 chars)
            # The search name is typically truncated, so check if file starts with it
            if file_normalized.startswith(search_normalized[:25]):
                return f

            # Check if search name starts with file name (reverse check)
            if search_normalized.startswith(file_normalized[:25]):
                return f

        return None

    def _get_pdf_document(self, pdf_path: Path) -> Optional[fitz.Document]:
        """
        Get PDF document, using cache for efficiency.

        Args:
            pdf_path: Path to PDF file

        Returns:
            fitz.Document or None
        """
        path_str = str(pdf_path)
        if path_str not in self._pdf_cache:
            try:
                self._pdf_cache[path_str] = fitz.open(pdf_path)
            except Exception:
                return None
        return self._pdf_cache[path_str]

    def _close_pdf_cache(self):
        """Close all cached PDF documents."""
        for doc in self._pdf_cache.values():
            try:
                doc.close()
            except Exception:
                pass
        self._pdf_cache.clear()

    def _extract_page_to_pdf(
        self,
        source_pdf: fitz.Document,
        page_number: int,
        output_path: Path
    ) -> bool:
        """
        Extract a single page from PDF and save as new PDF.

        Args:
            source_pdf: Source PDF document
            page_number: 1-indexed page number
            output_path: Path for output PDF

        Returns:
            True if successful, False otherwise
        """
        try:
            # Page numbers in fitz are 0-indexed
            page_idx = page_number - 1

            if page_idx < 0 or page_idx >= source_pdf.page_count:
                return False

            # Create new PDF with single page
            new_pdf = fitz.open()
            new_pdf.insert_pdf(source_pdf, from_page=page_idx, to_page=page_idx)
            new_pdf.save(output_path)
            new_pdf.close()

            return True
        except Exception:
            return False

    def create_folder_structure(self, dry_run: bool = False) -> List[str]:
        """
        Create category folder structure.

        Args:
            dry_run: If True, only report what would be created

        Returns:
            List of folder paths created (or would be created)
        """
        folders = []

        for category, config in SHEET_CATEGORIES.items():
            folder_name = config.get('folder', category)
            if folder_name:
                folder_path = self.output_dir / folder_name
                folders.append(str(folder_path))
                if not dry_run:
                    folder_path.mkdir(parents=True, exist_ok=True)

        # Create unclassified folder
        unclassified_path = self.output_dir / 'unclassified_may_be_needed'
        folders.append(str(unclassified_path))
        if not dry_run:
            unclassified_path.mkdir(parents=True, exist_ok=True)

        return folders

    def organize(
        self,
        classification: ClassificationResult,
        dry_run: bool = False,
        verbose: bool = False
    ) -> OrganizeStats:
        """
        Organize sheets by extracting PDF pages to category folders.

        Args:
            classification: ClassificationResult from SheetClassifier
            dry_run: If True, only report what would be done
            verbose: Print progress during organization

        Returns:
            OrganizeStats with operation results
        """
        stats = OrganizeStats()

        # Create folder structure
        if verbose:
            print("Creating folder structure...")

        folders = self.create_folder_structure(dry_run=dry_run)
        stats.folders_created = len(folders)

        if verbose:
            action = "Would create" if dry_run else "Created"
            print(f"  {action} {len(folders)} category folders")

        # Track which pages we've already extracted to avoid duplicates
        extracted_pages: Dict[str, set] = {}  # pdf_name -> set of (page, category)

        try:
            # Process each category
            for category, sheets in classification.categories.items():
                if not sheets:
                    continue

                # Get folder name for category
                folder_name = None
                if category in SHEET_CATEGORIES:
                    folder_name = SHEET_CATEGORIES[category].get('folder', category)
                elif category == 'unclassified_may_be_needed':
                    folder_name = 'unclassified_may_be_needed'
                # 'properly_classified_not_needed' has no folder - manifest only

                if not folder_name:
                    # No folder for this category - skip file operations
                    continue

                target_folder = self.output_dir / folder_name

                if verbose:
                    print(f"Processing {category}: {len(sheets)} sheets...")

                for sheet in sheets:
                    # Find source PDF
                    source_pdf_path = self._find_source_pdf(sheet.pdf_name)

                    if not source_pdf_path:
                        stats.files_skipped += 1
                        if verbose:
                            print(f"    Skip: {sheet.pdf_name} p{sheet.page_number} - PDF not found")
                        continue

                    # Build target filename
                    filename = self._build_filename(sheet)
                    target = target_folder / filename

                    # Check for duplicate extraction
                    pdf_key = str(source_pdf_path)
                    page_cat_key = (sheet.page_number, category)
                    if pdf_key not in extracted_pages:
                        extracted_pages[pdf_key] = set()
                    if page_cat_key in extracted_pages[pdf_key]:
                        # Already extracted this page to this category
                        continue
                    extracted_pages[pdf_key].add(page_cat_key)

                    try:
                        if dry_run:
                            if verbose:
                                print(f"    Would extract: {source_pdf_path.name} p{sheet.page_number} -> {filename}")
                            stats.files_extracted += 1
                        else:
                            # Get PDF document
                            source_doc = self._get_pdf_document(source_pdf_path)
                            if not source_doc:
                                stats.files_skipped += 1
                                if verbose:
                                    print(f"    Skip: {sheet.pdf_name} p{sheet.page_number} - cannot open PDF")
                                continue

                            # Extract page
                            success = self._extract_page_to_pdf(source_doc, sheet.page_number, target)
                            if success:
                                stats.files_extracted += 1
                                if verbose:
                                    print(f"    Extracted: p{sheet.page_number} -> {filename}")
                            else:
                                stats.files_skipped += 1
                                if verbose:
                                    print(f"    Skip: {sheet.pdf_name} p{sheet.page_number} - extraction failed")

                    except Exception as e:
                        stats.errors += 1
                        error_msg = f"{sheet.pdf_name} p{sheet.page_number}: {e}"
                        stats.error_details.append(error_msg)
                        if verbose:
                            print(f"    Error: {error_msg}")

        finally:
            # Close all cached PDFs
            self._close_pdf_cache()

        return stats


def organize_classified_sheets(
    classification: ClassificationResult,
    pdfs_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = False
) -> OrganizeStats:
    """
    Convenience function to organize classified sheets.

    Args:
        classification: Classification results
        pdfs_dir: Directory containing source PDF files
        output_dir: Output directory for organized files
        dry_run: If True, only report what would be done
        verbose: Print progress

    Returns:
        OrganizeStats
    """
    organizer = SheetOrganizer(pdfs_dir, output_dir)
    return organizer.organize(classification, dry_run=dry_run, verbose=verbose)
