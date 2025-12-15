"""
Blueprint Processor V4.2.1 - Database Models
SQLAlchemy models for storing extracted data.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Text,
    UniqueConstraint, create_engine
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ExtractedSheet(Base):
    """
    Model for storing extracted sheet information.
    Each page of each PDF gets its own record.
    """
    __tablename__ = 'extracted_sheets'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Source identification
    pdf_filename = Column(String(500), nullable=False, index=True)
    pdf_hash = Column(String(16), index=True)  # SHA-256 hash (first 16 chars) for stable ID
    page_number = Column(Integer, nullable=False)

    # Extracted fields
    sheet_number = Column(String(50), index=True)
    project_number = Column(String(100), index=True)
    sheet_title = Column(String(500))
    date = Column(String(50))
    scale = Column(String(100))
    discipline = Column(String(50))

    # Metadata
    confidence = Column(String(20))  # high, medium, low (legacy)
    extraction_method = Column(String(50))  # vector, ocr
    is_valid = Column(Integer, default=1)  # 1=valid, 0=invalid

    # V4.2.1 title extraction metadata
    title_confidence = Column(Float)  # 0.0-1.0 confidence score
    title_method = Column(String(50))  # drawing_index, spatial, vision_api, pattern
    needs_review = Column(Integer, default=0)  # 1=needs HITL review

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # CRITICAL: Unique constraint to prevent duplicates
    __table_args__ = (
        UniqueConstraint('pdf_filename', 'page_number', name='uq_pdf_page'),
    )

    def __repr__(self):
        return f"<ExtractedSheet(pdf='{self.pdf_filename}', page={self.page_number}, sheet='{self.sheet_number}')>"

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'pdf_filename': self.pdf_filename,
            'pdf_hash': self.pdf_hash,
            'page_number': self.page_number,
            'sheet_number': self.sheet_number,
            'project_number': self.project_number,
            'sheet_title': self.sheet_title,
            'date': self.date,
            'scale': self.scale,
            'discipline': self.discipline,
            'confidence': self.confidence,
            'extraction_method': self.extraction_method,
            'is_valid': self.is_valid,
            'title_confidence': self.title_confidence,
            'title_method': self.title_method,
            'needs_review': self.needs_review,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class ProcessingRun(Base):
    """
    Model for tracking processing runs (batch operations).
    """
    __tablename__ = 'processing_runs'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Run information
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    # Statistics
    pdf_count = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)

    # Source path
    source_path = Column(String(1000))

    # Status
    status = Column(String(50), default='running')  # running, completed, failed

    def __repr__(self):
        return f"<ProcessingRun(id={self.id}, status='{self.status}', pdfs={self.pdf_count})>"

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'pdf_count': self.pdf_count,
            'page_count': self.page_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'source_path': self.source_path,
            'status': self.status,
        }
