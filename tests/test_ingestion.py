"""Tests for PDF ingestion service."""

import pytest
from core.decomposition.ingestion import PDFIngestionService, DocumentContent


@pytest.fixture
def pdf_service() -> PDFIngestionService:
    """Create PDF ingestion service instance."""
    return PDFIngestionService()


def test_clean_text(pdf_service: PDFIngestionService) -> None:
    """Test text cleaning functionality."""
    dirty_text = "Hello   world\n\n\n\nNew paragraph"
    clean = pdf_service._clean_text(dirty_text)
    
    assert "   " not in clean  # No multiple spaces
    assert "\n\n\n" not in clean  # No excessive newlines
    assert "Hello world" in clean
    assert "New paragraph" in clean


def test_document_content_is_empty() -> None:
    """Test empty document detection."""
    empty_doc = DocumentContent(
        filename="empty.pdf",
        page_count=1,
        pages=[],
        total_word_count=0,
        raw_text="",
    )
    assert empty_doc.is_empty is True
    
    non_empty_doc = DocumentContent(
        filename="full.pdf",
        page_count=1,
        pages=[],
        total_word_count=100,
        raw_text="Some content here",
    )
    assert non_empty_doc.is_empty is False
