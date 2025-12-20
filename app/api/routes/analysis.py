"""Analysis API routes for clause extraction and document analysis."""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.api.routes.documents import documents_db
from core.decomposition.clause_detector import clause_detector

router = APIRouter()


@router.get("/{document_id}/clauses")
async def extract_clauses(document_id: UUID) -> dict:
    """Extract clauses from a document.
    
    Returns the document sliced into atomic clause units for micro-agent processing.
    """
    document = documents_db.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not document.raw_text:
        raise HTTPException(status_code=422, detail="Document has no extracted text")
    
    # Run clause detection
    result = clause_detector.extract_to_json(document.raw_text)
    
    return {
        "document_id": str(document_id),
        "filename": document.filename,
        **result,
    }


@router.get("/{document_id}/summary")
async def get_document_summary(document_id: UUID) -> dict:
    """Get a summary of document analysis.
    
    Returns clause counts by type and risk indicators.
    """
    document = documents_db.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not document.raw_text:
        raise HTTPException(status_code=422, detail="Document has no extracted text")
    
    # Run clause detection
    clauses = clause_detector.detect_clauses(document.raw_text)
    
    # Count by type
    type_counts: dict[str, int] = {}
    for clause in clauses:
        ctype = clause.clause_type.value
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
    
    # Identify high-risk clauses
    high_risk_types = ["indemnification", "limitation_of_liability", "termination"]
    high_risk_clauses = [c for c in clauses if c.clause_type.value in high_risk_types]
    
    return {
        "document_id": str(document_id),
        "filename": document.filename,
        "total_clauses": len(clauses),
        "clause_type_counts": type_counts,
        "high_risk_count": len(high_risk_clauses),
        "high_risk_types": [c.clause_type.value for c in high_risk_clauses],
    }
