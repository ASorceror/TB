"""
Blueprint Processor V4.3 - Template Types Module
Defines the data structure for storing learned templates.

This module implements Phase B of V4.3: Template Data Structure
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


@dataclass
class Template:
    """Learned template for a PDF's title block structure."""

    # Identification
    template_id: str                    # UUID string
    pdf_hash: str                       # First 16 chars of SHA-256
    source_pdf: str                     # Original filename
    created_at: str                     # ISO datetime string

    # Learning stats
    pages_analyzed: int                 # Total pages in PDF
    quality_pages_used: int             # Pages that passed quality filter
    confidence: float                   # 0.0-1.0

    # Title block location (as fraction of page, 0.0-1.0)
    # [x1, y1, x2, y2] where (0,0) is top-left
    title_block_bbox: List[float]       # e.g., [0.65, 0.75, 1.0, 1.0]

    # Title zone within block (relative to block, 0.0-1.0)
    title_zone_bbox: List[float]        # e.g., [0.0, 0.10, 1.0, 0.50]

    # Exclusion patterns (regex patterns to ignore)
    exclusion_patterns: List[str]       # e.g., [r"Project\s*Number", ...]

    # Observed title characteristics
    typical_length_min: int             # Shortest quality title seen
    typical_length_max: int             # Longest quality title seen

    # Page type distribution
    vector_pages: int                   # Count of vector pages
    scanned_pages: int                  # Count of scanned/OCR pages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Template':
        """Create Template from dictionary."""
        return cls(**data)

    @classmethod
    def create_new(
        cls,
        pdf_hash: str,
        source_pdf: str,
        pages_analyzed: int,
        quality_pages_used: int,
        confidence: float,
        title_block_bbox: List[float],
        title_zone_bbox: List[float],
        exclusion_patterns: List[str],
        typical_length_min: int,
        typical_length_max: int,
        vector_pages: int,
        scanned_pages: int
    ) -> 'Template':
        """Create a new template with auto-generated ID and timestamp."""
        return cls(
            template_id=str(uuid.uuid4()),
            pdf_hash=pdf_hash,
            source_pdf=source_pdf,
            created_at=datetime.utcnow().isoformat() + 'Z',
            pages_analyzed=pages_analyzed,
            quality_pages_used=quality_pages_used,
            confidence=confidence,
            title_block_bbox=title_block_bbox,
            title_zone_bbox=title_zone_bbox,
            exclusion_patterns=exclusion_patterns,
            typical_length_min=typical_length_min,
            typical_length_max=typical_length_max,
            vector_pages=vector_pages,
            scanned_pages=scanned_pages
        )
