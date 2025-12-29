"""
Classification Manifest Module (V6.0)

Generates JSON manifest files for classified sheets.
Creates both per-category manifests and a master manifest.

Classes:
    ManifestGenerator: Creates JSON manifests for classification results
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sheet_classifier import ClassificationResult, ClassifiedSheet
from core.sheet_organizer import OrganizeStats
from constants import SHEET_CATEGORIES


class ManifestGenerator:
    """
    Generates JSON manifests for classification results.

    Creates:
    - Master manifest with all categories and statistics
    - Per-category manifests in each folder
    """

    def __init__(self, output_dir: Path):
        """
        Initialize manifest generator.

        Args:
            output_dir: Base directory for manifest output
        """
        self.output_dir = Path(output_dir)

    def _format_sheet_entry(self, sheet: ClassifiedSheet) -> Dict[str, Any]:
        """Format a single sheet entry for manifest."""
        return {
            'pdf': sheet.pdf_name,
            'page': sheet.page_number,
            'sheet_number': sheet.sheet_number,
            'sheet_title': sheet.sheet_title,
            'number_confidence': sheet.number_confidence,
            'title_confidence': sheet.title_confidence,
        }

    def generate_category_manifest(
        self,
        category: str,
        sheets: List[ClassifiedSheet],
        folder_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate manifest for a single category.

        Args:
            category: Category name
            sheets: List of ClassifiedSheet instances
            folder_path: Optional folder path for the category

        Returns:
            Category manifest dict
        """
        manifest = {
            'category': category,
            'description': SHEET_CATEGORIES.get(category, {}).get('description', ''),
            'generated_at': datetime.now().isoformat(),
            'total_sheets': len(sheets),
            'sheets': [self._format_sheet_entry(s) for s in sheets]
        }

        # Group by PDF for easier navigation
        by_pdf: Dict[str, List[Dict]] = {}
        for sheet in sheets:
            if sheet.pdf_name not in by_pdf:
                by_pdf[sheet.pdf_name] = []
            by_pdf[sheet.pdf_name].append(self._format_sheet_entry(sheet))

        manifest['by_pdf'] = by_pdf

        return manifest

    def generate_master_manifest(
        self,
        classification: ClassificationResult,
        organize_stats: Optional[OrganizeStats] = None
    ) -> Dict[str, Any]:
        """
        Generate master manifest with all categories.

        Args:
            classification: ClassificationResult from classifier
            organize_stats: Optional organization statistics

        Returns:
            Master manifest dict
        """
        manifest = {
            'generated_at': datetime.now().isoformat(),
            'classification_timestamp': classification.classified_at,
            'total_pages': classification.total_pages,
            'statistics': classification.statistics,
            'categories': {}
        }

        # Add each category's summary
        for category, sheets in classification.categories.items():
            manifest['categories'][category] = {
                'count': len(sheets),
                'description': SHEET_CATEGORIES.get(category, {}).get('description', ''),
                'folder': SHEET_CATEGORIES.get(category, {}).get('folder'),
            }

        # Add organization stats if available
        if organize_stats:
            manifest['organization'] = {
                'folders_created': organize_stats.folders_created,
                'pages_extracted': organize_stats.files_extracted,
                'pages_skipped': organize_stats.files_skipped,
                'errors': organize_stats.errors,
            }

        return manifest

    def save_manifests(
        self,
        classification: ClassificationResult,
        organize_stats: Optional[OrganizeStats] = None,
        verbose: bool = False
    ) -> List[Path]:
        """
        Save all manifests to disk.

        Args:
            classification: ClassificationResult
            organize_stats: Optional organization stats
            verbose: Print progress

        Returns:
            List of paths to saved manifest files
        """
        saved_files = []

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save master manifest
        master_manifest = self.generate_master_manifest(classification, organize_stats)
        master_path = self.output_dir / 'classification_manifest.json'
        with open(master_path, 'w') as f:
            json.dump(master_manifest, f, indent=2)
        saved_files.append(master_path)
        if verbose:
            print(f"Saved master manifest: {master_path}")

        # Save per-category manifests
        for category, sheets in classification.categories.items():
            if not sheets:
                continue

            # Get folder for this category
            folder_name = None
            if category in SHEET_CATEGORIES:
                folder_name = SHEET_CATEGORIES[category].get('folder', category)
            elif category == 'unclassified_may_be_needed':
                folder_name = 'unclassified_may_be_needed'
            elif category == 'properly_classified_not_needed':
                # No folder, but still create manifest in output root
                folder_name = None

            # Determine manifest location
            if folder_name:
                manifest_dir = self.output_dir / folder_name
            else:
                manifest_dir = self.output_dir

            manifest_dir.mkdir(parents=True, exist_ok=True)

            # Generate and save category manifest
            category_manifest = self.generate_category_manifest(category, sheets)
            manifest_filename = f"{category}_manifest.json"
            manifest_path = manifest_dir / manifest_filename

            with open(manifest_path, 'w') as f:
                json.dump(category_manifest, f, indent=2)
            saved_files.append(manifest_path)

            if verbose:
                print(f"Saved {category} manifest: {manifest_path}")

        return saved_files


def generate_manifests(
    classification: ClassificationResult,
    output_dir: Path,
    organize_stats: Optional[OrganizeStats] = None,
    verbose: bool = False
) -> List[Path]:
    """
    Convenience function to generate all manifests.

    Args:
        classification: Classification results
        output_dir: Output directory
        organize_stats: Optional organization statistics
        verbose: Print progress

    Returns:
        List of manifest file paths
    """
    generator = ManifestGenerator(output_dir)
    return generator.save_manifests(classification, organize_stats, verbose=verbose)
