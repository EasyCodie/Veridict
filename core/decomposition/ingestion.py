"""PDF Ingestion Service - Component 1 of the MAKER Framework.

This module handles the extraction of text and metadata from PDF documents,
preparing them for clause detection and analysis.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import fitz  # PyMuPDF


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_number: int
    text: str
    word_count: int


@dataclass
class DocumentContent:
    """Represents extracted content from an entire PDF document."""
    filename: str
    page_count: int
    pages: list[PageContent]
    total_word_count: int
    raw_text: str

    @property
    def is_empty(self) -> bool:
        """Check if the document has no extractable text."""
        return self.total_word_count == 0


class PDFIngestionService:
    """Service for extracting text and metadata from PDF documents."""

    def __init__(self) -> None:
        """Initialize the PDF ingestion service."""
        pass

    def extract_from_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> DocumentContent:
        """Extract text content from PDF bytes.
        
        Args:
            pdf_bytes: Raw PDF file bytes.
            filename: Name of the source file.
            
        Returns:
            DocumentContent with extracted text and metadata.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return self._process_document(doc, filename)

    def extract_from_file(self, file_path: str | Path) -> DocumentContent:
        """Extract text content from a PDF file path.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            DocumentContent with extracted text and metadata.
        """
        path = Path(file_path)
        doc = fitz.open(str(path))
        return self._process_document(doc, path.name)

    def extract_from_stream(self, stream: BinaryIO, filename: str = "document.pdf") -> DocumentContent:
        """Extract text content from a file stream.
        
        Args:
            stream: File-like object containing PDF data.
            filename: Name of the source file.
            
        Returns:
            DocumentContent with extracted text and metadata.
        """
        pdf_bytes = stream.read()
        return self.extract_from_bytes(pdf_bytes, filename)

    def _process_document(self, doc: fitz.Document, filename: str) -> DocumentContent:
        """Process a PyMuPDF document and extract content.
        
        Args:
            doc: PyMuPDF Document object.
            filename: Name of the source file.
            
        Returns:
            DocumentContent with extracted text and metadata.
        """
        pages: list[PageContent] = []
        all_text_parts: list[str] = []
        total_words = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Clean up the text
            text = self._clean_text(text)
            word_count = len(text.split())
            
            pages.append(PageContent(
                page_number=page_num + 1,  # 1-indexed
                text=text,
                word_count=word_count,
            ))
            
            all_text_parts.append(text)
            total_words += word_count

        doc.close()

        return DocumentContent(
            filename=filename,
            page_count=len(pages),
            pages=pages,
            total_word_count=total_words,
            raw_text="\n\n".join(all_text_parts),
        )

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by normalizing whitespace.
        
        Args:
            text: Raw extracted text.
            
        Returns:
            Cleaned text with normalized whitespace.
        """
        # Replace multiple newlines with double newline (paragraph break)
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


# Global instance for dependency injection
pdf_service = PDFIngestionService()
