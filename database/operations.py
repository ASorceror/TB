"""
Blueprint Processor V4.1 - Database Operations
CRUD operations for the database.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import Base, ExtractedSheet, ProcessingRun
from constants import DATABASE_NAME


class DatabaseOperations:
    """
    Database operations for blueprint data.
    Handles CRUD operations and upsert logic.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file (default: data/blueprint_data.db)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'data' / DATABASE_NAME

        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine and session
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.Session = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.Session()

    def upsert_sheet(self, data: Dict[str, Any]) -> ExtractedSheet:
        """
        Insert or update an extracted sheet record.
        Uses (pdf_filename, page_number) as unique key.

        Args:
            data: Dict with sheet data

        Returns:
            ExtractedSheet instance
        """
        session = self.get_session()
        try:
            # Check for existing record
            existing = session.query(ExtractedSheet).filter_by(
                pdf_filename=data['pdf_filename'],
                page_number=data['page_number']
            ).first()

            if existing:
                # Update existing record
                for key, value in data.items():
                    if hasattr(existing, key) and key not in ['id', 'created_at']:
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
                sheet = existing
            else:
                # Insert new record
                sheet = ExtractedSheet(**data)
                session.add(sheet)

            session.commit()
            session.refresh(sheet)
            return sheet
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_sheet(self, pdf_filename: str, page_number: int) -> Optional[ExtractedSheet]:
        """
        Get a sheet by filename and page number.

        Args:
            pdf_filename: PDF filename
            page_number: Page number

        Returns:
            ExtractedSheet or None
        """
        session = self.get_session()
        try:
            return session.query(ExtractedSheet).filter_by(
                pdf_filename=pdf_filename,
                page_number=page_number
            ).first()
        finally:
            session.close()

    def get_all_sheets(self) -> List[ExtractedSheet]:
        """Get all extracted sheets."""
        session = self.get_session()
        try:
            return session.query(ExtractedSheet).all()
        finally:
            session.close()

    def get_sheets_by_pdf(self, pdf_filename: str) -> List[ExtractedSheet]:
        """Get all sheets from a specific PDF."""
        session = self.get_session()
        try:
            return session.query(ExtractedSheet).filter_by(
                pdf_filename=pdf_filename
            ).order_by(ExtractedSheet.page_number).all()
        finally:
            session.close()

    def count_sheets(self) -> int:
        """Get total count of extracted sheets."""
        session = self.get_session()
        try:
            return session.query(ExtractedSheet).count()
        finally:
            session.close()

    def delete_sheet(self, pdf_filename: str, page_number: int) -> bool:
        """Delete a specific sheet."""
        session = self.get_session()
        try:
            result = session.query(ExtractedSheet).filter_by(
                pdf_filename=pdf_filename,
                page_number=page_number
            ).delete()
            session.commit()
            return result > 0
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def start_processing_run(self, source_path: str) -> ProcessingRun:
        """Start a new processing run."""
        session = self.get_session()
        try:
            run = ProcessingRun(
                source_path=source_path,
                status='running'
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return run
        finally:
            session.close()

    def update_processing_run(self, run_id: int, **kwargs) -> Optional[ProcessingRun]:
        """Update a processing run."""
        session = self.get_session()
        try:
            run = session.query(ProcessingRun).filter_by(id=run_id).first()
            if run:
                for key, value in kwargs.items():
                    if hasattr(run, key):
                        setattr(run, key, value)
                session.commit()
                session.refresh(run)
            return run
        finally:
            session.close()

    def complete_processing_run(self, run_id: int,
                               pdf_count: int, page_count: int,
                               success_count: int, error_count: int) -> Optional[ProcessingRun]:
        """Complete a processing run with statistics."""
        return self.update_processing_run(
            run_id,
            completed_at=datetime.utcnow(),
            pdf_count=pdf_count,
            page_count=page_count,
            success_count=success_count,
            error_count=error_count,
            status='completed' if error_count == 0 else 'completed_with_errors'
        )

    def get_processing_runs(self, limit: int = 10) -> List[ProcessingRun]:
        """Get recent processing runs."""
        session = self.get_session()
        try:
            return session.query(ProcessingRun).order_by(
                ProcessingRun.started_at.desc()
            ).limit(limit).all()
        finally:
            session.close()
