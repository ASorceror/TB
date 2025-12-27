"""
Blueprint Processor V4.3 - Template Store Module
Store and retrieve learned templates.

This module implements Phase B of V4.3: Template Data Structure
"""

import json
from pathlib import Path
from typing import Optional, List

from core.template_types import Template


class TemplateStore:
    """Store and retrieve learned templates."""

    def __init__(self, templates_dir: str = "templates"):
        """
        Initialize the template store.

        Args:
            templates_dir: Directory to store template JSON files
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)

    def _get_path(self, pdf_hash: str) -> Path:
        """Get file path for a template."""
        return self.templates_dir / f"{pdf_hash}.json"

    def save(self, template: Template) -> None:
        """
        Save template to JSON file.

        Args:
            template: Template to save
        """
        path = self._get_path(template.pdf_hash)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(template.to_dict(), f, indent=2)

    def load(self, pdf_hash: str) -> Optional[Template]:
        """
        Load template from JSON file.

        Args:
            pdf_hash: Hash of the PDF to load template for

        Returns:
            Template if found, None otherwise
        """
        path = self._get_path(pdf_hash)
        if not path.exists():
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Template.from_dict(data)

    def exists(self, pdf_hash: str) -> bool:
        """
        Check if template exists.

        Args:
            pdf_hash: Hash of the PDF to check

        Returns:
            True if template exists, False otherwise
        """
        return self._get_path(pdf_hash).exists()

    def delete(self, pdf_hash: str) -> bool:
        """
        Delete template.

        Args:
            pdf_hash: Hash of the PDF template to delete

        Returns:
            True if deleted, False if not found
        """
        path = self._get_path(pdf_hash)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_all(self) -> List[str]:
        """
        List all stored template hashes.

        Returns:
            List of PDF hashes for stored templates
        """
        return [p.stem for p in self.templates_dir.glob("*.json")]
