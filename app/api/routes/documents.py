"""Document upload and management routes."""

from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.document import Document, DocumentResponse, DocumentStatus

router = APIRouter()

# In-memory storage for now (will be replaced with database)
documents_db: dict[UUID, Document] = {}


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentResponse:
    """Upload a PDF document for analysis."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Create document record
    document = Document(
        filename=file.filename,
        content_type=file.content_type or "application/pdf",
        status=DocumentStatus.PENDING,
    )

    # Read file content (for now just store metadata)
    content = await file.read()
    document.raw_text = f"[PDF content: {len(content)} bytes]"  # Placeholder

    # Store in memory
    documents_db[document.id] = document

    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        status=document.status,
        page_count=document.page_count,
        clause_count=document.clause_count,
        created_at=document.created_at,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: UUID) -> DocumentResponse:
    """Get document by ID."""
    document = documents_db.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        status=document.status,
        page_count=document.page_count,
        clause_count=document.clause_count,
        created_at=document.created_at,
    )


@router.get("/", response_model=list[DocumentResponse])
async def list_documents() -> list[DocumentResponse]:
    """List all documents."""
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            status=doc.status,
            page_count=doc.page_count,
            clause_count=doc.clause_count,
            created_at=doc.created_at,
        )
        for doc in documents_db.values()
    ]
