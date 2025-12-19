"""Document data models."""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Status of document processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentCreate(BaseModel):
    """Schema for creating a new document."""
    filename: str
    content_type: str = "application/pdf"


class Document(BaseModel):
    """Document model representing an uploaded legal document."""
    id: UUID = Field(default_factory=uuid4)
    filename: str
    content_type: str
    status: DocumentStatus = DocumentStatus.PENDING
    page_count: int = 0
    clause_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    raw_text: str | None = None


class DocumentResponse(BaseModel):
    """Response schema for document operations."""
    id: UUID
    filename: str
    status: DocumentStatus
    page_count: int
    clause_count: int
    created_at: datetime
