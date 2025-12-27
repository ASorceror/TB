"""
Phase B Validation Tests - Template Data Structure
Run with: python -m pytest tests/test_phase_b_template.py -v

This module tests the Template dataclass and TemplateStore for
storing and retrieving learned templates.
"""
import pytest
from pathlib import Path
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.template_types import Template
from core.template_store import TemplateStore


class TestTemplateStructure:

    @pytest.fixture
    def sample_template(self):
        """Create a sample template for testing."""
        return Template(
            template_id="test-uuid-1234",
            pdf_hash="abc123def456gh78",
            source_pdf="test_document.pdf",
            created_at="2024-12-14T12:00:00Z",
            pages_analyzed=100,
            quality_pages_used=45,
            confidence=0.85,
            title_block_bbox=[0.65, 0.75, 1.0, 1.0],
            title_zone_bbox=[0.0, 0.10, 1.0, 0.50],
            exclusion_patterns=[r"Project\s*Number", r"Sheet\s*Number"],
            typical_length_min=10,
            typical_length_max=50,
            vector_pages=80,
            scanned_pages=20
        )

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a temporary template store."""
        return TemplateStore(templates_dir=str(tmp_path / "templates"))

    def test_template_to_dict(self, sample_template):
        """Template can be converted to dict."""
        d = sample_template.to_dict()
        assert d['template_id'] == "test-uuid-1234"
        assert d['pdf_hash'] == "abc123def456gh78"
        assert d['confidence'] == 0.85
        assert d['title_block_bbox'] == [0.65, 0.75, 1.0, 1.0]

    def test_template_from_dict(self, sample_template):
        """Template can be created from dict."""
        d = sample_template.to_dict()
        t2 = Template.from_dict(d)
        assert t2.template_id == sample_template.template_id
        assert t2.confidence == sample_template.confidence

    def test_save_and_load(self, sample_template, temp_store):
        """Template can be saved and loaded."""
        temp_store.save(sample_template)

        loaded = temp_store.load(sample_template.pdf_hash)

        assert loaded is not None
        assert loaded.template_id == sample_template.template_id
        assert loaded.pdf_hash == sample_template.pdf_hash
        assert loaded.confidence == sample_template.confidence
        assert loaded.exclusion_patterns == sample_template.exclusion_patterns

    def test_load_nonexistent(self, temp_store):
        """Loading non-existent template returns None."""
        result = temp_store.load("nonexistent_hash")
        assert result is None

    def test_exists(self, sample_template, temp_store):
        """Exists check works correctly."""
        assert temp_store.exists(sample_template.pdf_hash) == False

        temp_store.save(sample_template)

        assert temp_store.exists(sample_template.pdf_hash) == True

    def test_delete(self, sample_template, temp_store):
        """Delete removes template."""
        temp_store.save(sample_template)
        assert temp_store.exists(sample_template.pdf_hash) == True

        result = temp_store.delete(sample_template.pdf_hash)

        assert result == True
        assert temp_store.exists(sample_template.pdf_hash) == False

    def test_delete_nonexistent(self, temp_store):
        """Deleting non-existent template returns False."""
        result = temp_store.delete("nonexistent_hash")
        assert result == False

    def test_create_new_template(self):
        """Test the create_new factory method."""
        template = Template.create_new(
            pdf_hash="abc123",
            source_pdf="test.pdf",
            pages_analyzed=50,
            quality_pages_used=25,
            confidence=0.80,
            title_block_bbox=[0.65, 0.75, 1.0, 1.0],
            title_zone_bbox=[0.0, 0.10, 1.0, 0.50],
            exclusion_patterns=["pattern1"],
            typical_length_min=5,
            typical_length_max=40,
            vector_pages=40,
            scanned_pages=10
        )

        # Should have auto-generated ID and timestamp
        assert template.template_id is not None
        assert len(template.template_id) > 0
        assert template.created_at is not None
        assert 'T' in template.created_at  # ISO format

    def test_list_all(self, temp_store):
        """Test listing all templates."""
        # Initially empty
        assert temp_store.list_all() == []

        # Add some templates
        t1 = Template(
            template_id="1",
            pdf_hash="hash1",
            source_pdf="doc1.pdf",
            created_at="2024-12-14T12:00:00Z",
            pages_analyzed=10,
            quality_pages_used=5,
            confidence=0.80,
            title_block_bbox=[0.65, 0.75, 1.0, 1.0],
            title_zone_bbox=[0.0, 0.10, 1.0, 0.50],
            exclusion_patterns=[],
            typical_length_min=10,
            typical_length_max=40,
            vector_pages=8,
            scanned_pages=2
        )
        t2 = Template(
            template_id="2",
            pdf_hash="hash2",
            source_pdf="doc2.pdf",
            created_at="2024-12-14T12:00:00Z",
            pages_analyzed=20,
            quality_pages_used=10,
            confidence=0.85,
            title_block_bbox=[0.65, 0.75, 1.0, 1.0],
            title_zone_bbox=[0.0, 0.10, 1.0, 0.50],
            exclusion_patterns=[],
            typical_length_min=10,
            typical_length_max=40,
            vector_pages=15,
            scanned_pages=5
        )

        temp_store.save(t1)
        temp_store.save(t2)

        all_hashes = temp_store.list_all()
        assert len(all_hashes) == 2
        assert "hash1" in all_hashes
        assert "hash2" in all_hashes

    def test_template_all_fields_preserved(self, sample_template, temp_store):
        """All template fields should be preserved through save/load."""
        temp_store.save(sample_template)
        loaded = temp_store.load(sample_template.pdf_hash)

        assert loaded.template_id == sample_template.template_id
        assert loaded.pdf_hash == sample_template.pdf_hash
        assert loaded.source_pdf == sample_template.source_pdf
        assert loaded.created_at == sample_template.created_at
        assert loaded.pages_analyzed == sample_template.pages_analyzed
        assert loaded.quality_pages_used == sample_template.quality_pages_used
        assert loaded.confidence == sample_template.confidence
        assert loaded.title_block_bbox == sample_template.title_block_bbox
        assert loaded.title_zone_bbox == sample_template.title_zone_bbox
        assert loaded.exclusion_patterns == sample_template.exclusion_patterns
        assert loaded.typical_length_min == sample_template.typical_length_min
        assert loaded.typical_length_max == sample_template.typical_length_max
        assert loaded.vector_pages == sample_template.vector_pages
        assert loaded.scanned_pages == sample_template.scanned_pages

    def test_templates_directory_created(self, tmp_path):
        """Templates directory should be created if it doesn't exist."""
        non_existent_dir = tmp_path / "new_templates_dir"
        assert not non_existent_dir.exists()

        store = TemplateStore(templates_dir=str(non_existent_dir))

        assert non_existent_dir.exists()
