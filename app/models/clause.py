"""Clause data models."""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ClauseType(str, Enum):
    """Standard legal clause types."""
    INDEMNIFICATION = "indemnification"
    TERMINATION = "termination"
    CONFIDENTIALITY = "confidentiality"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    GOVERNING_LAW = "governing_law"
    DISPUTE_RESOLUTION = "dispute_resolution"
    FORCE_MAJEURE = "force_majeure"
    ASSIGNMENT = "assignment"
    AMENDMENT = "amendment"
    WAIVER = "waiver"
    SEVERABILITY = "severability"
    ENTIRE_AGREEMENT = "entire_agreement"
    NOTICE = "notice"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    WARRANTY = "warranty"
    REPRESENTATIONS = "representations"
    INSURANCE = "insurance"
    COMPLIANCE = "compliance"
    DATA_PROTECTION = "data_protection"
    AUDIT_RIGHTS = "audit_rights"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk level assessment for a clause."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Clause(BaseModel):
    """Clause model representing an extracted contract clause."""
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    clause_type: ClauseType
    text: str
    page_number: int
    start_position: int
    end_position: int
    risk_level: RiskLevel = RiskLevel.LOW
    confidence_score: float = 0.0
    vote_count: int = 0
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    embedding: list[float] | None = None


class ClauseAnalysis(BaseModel):
    """Analysis result for a clause."""
    clause_id: UUID
    clause_type: ClauseType
    risk_level: RiskLevel
    confidence_score: float
    key_terms: list[str] = []
    obligations: list[str] = []
    red_flags: list[str] = []
    vote_breakdown: dict[str, int] = {}
