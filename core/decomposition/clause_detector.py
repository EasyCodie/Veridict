"""Clause Boundary Detection - Core of the Decomposition Engine.

This module identifies and extracts individual clauses from legal documents,
slicing them into atomic units for independent analysis by micro-agents.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ClauseType(str, Enum):
    """Standard legal clause types recognized by Veridict."""
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
    PAYMENT = "payment"
    DUTIES = "duties"
    SCOPE = "scope"
    TERM = "term"
    DEFINITIONS = "definitions"
    RECITALS = "recitals"
    UNKNOWN = "unknown"


# Mapping of keywords to clause types
CLAUSE_KEYWORDS: dict[str, ClauseType] = {
    "indemnif": ClauseType.INDEMNIFICATION,
    "hold harmless": ClauseType.INDEMNIFICATION,
    "terminat": ClauseType.TERMINATION,
    "cancel": ClauseType.TERMINATION,
    "confidential": ClauseType.CONFIDENTIALITY,
    "non-disclosure": ClauseType.CONFIDENTIALITY,
    "nda": ClauseType.CONFIDENTIALITY,
    "proprietary": ClauseType.CONFIDENTIALITY,
    "limitation of liability": ClauseType.LIMITATION_OF_LIABILITY,
    "limit of liability": ClauseType.LIMITATION_OF_LIABILITY,
    "liability cap": ClauseType.LIMITATION_OF_LIABILITY,
    "consequential damage": ClauseType.LIMITATION_OF_LIABILITY,
    "governing law": ClauseType.GOVERNING_LAW,
    "choice of law": ClauseType.GOVERNING_LAW,
    "applicable law": ClauseType.GOVERNING_LAW,
    "jurisdiction": ClauseType.GOVERNING_LAW,
    "dispute": ClauseType.DISPUTE_RESOLUTION,
    "arbitrat": ClauseType.DISPUTE_RESOLUTION,
    "mediat": ClauseType.DISPUTE_RESOLUTION,
    "force majeure": ClauseType.FORCE_MAJEURE,
    "act of god": ClauseType.FORCE_MAJEURE,
    "assign": ClauseType.ASSIGNMENT,
    "transfer": ClauseType.ASSIGNMENT,
    "amend": ClauseType.AMENDMENT,
    "modif": ClauseType.AMENDMENT,
    "waiv": ClauseType.WAIVER,
    "severab": ClauseType.SEVERABILITY,
    "entire agreement": ClauseType.ENTIRE_AGREEMENT,
    "whole agreement": ClauseType.ENTIRE_AGREEMENT,
    "integration": ClauseType.ENTIRE_AGREEMENT,
    "notice": ClauseType.NOTICE,
    "notification": ClauseType.NOTICE,
    "intellectual property": ClauseType.INTELLECTUAL_PROPERTY,
    "copyright": ClauseType.INTELLECTUAL_PROPERTY,
    "patent": ClauseType.INTELLECTUAL_PROPERTY,
    "trademark": ClauseType.INTELLECTUAL_PROPERTY,
    "warrant": ClauseType.WARRANTY,
    "represent": ClauseType.REPRESENTATIONS,
    "insurance": ClauseType.INSURANCE,
    "complian": ClauseType.COMPLIANCE,
    "data protection": ClauseType.DATA_PROTECTION,
    "privacy": ClauseType.DATA_PROTECTION,
    "gdpr": ClauseType.DATA_PROTECTION,
    "personal data": ClauseType.DATA_PROTECTION,
    "audit": ClauseType.AUDIT_RIGHTS,
    "inspect": ClauseType.AUDIT_RIGHTS,
    "payment": ClauseType.PAYMENT,
    "compensat": ClauseType.PAYMENT,
    "fee": ClauseType.PAYMENT,
    "dut": ClauseType.DUTIES,
    "obligation": ClauseType.DUTIES,
    "responsibilit": ClauseType.DUTIES,
    "scope": ClauseType.SCOPE,
    "services": ClauseType.SCOPE,
    "term": ClauseType.TERM,
    "duration": ClauseType.TERM,
    "period": ClauseType.TERM,
    "defin": ClauseType.DEFINITIONS,
    "meaning": ClauseType.DEFINITIONS,
    "recital": ClauseType.RECITALS,
    "whereas": ClauseType.RECITALS,
    "background": ClauseType.RECITALS,
}


@dataclass
class ExtractedClause:
    """A single clause extracted from a legal document."""
    section_number: str
    title: str
    text: str
    clause_type: ClauseType
    start_position: int
    end_position: int
    confidence: float = 1.0
    sub_clauses: List["ExtractedClause"] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "section_number": self.section_number,
            "title": self.title,
            "text": self.text,
            "clause_type": self.clause_type.value,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "confidence": self.confidence,
            "sub_clauses": [sc.to_dict() for sc in self.sub_clauses],
        }


class ClauseDetector:
    """Detects and extracts clause boundaries from legal document text."""

    # Pattern to match section headers like "1.", "2.1", "A.", "ARTICLE I", etc.
    SECTION_PATTERNS = [
        # Numbered sections: "1.", "2.", "10."
        r'^(\d+)\.\s+([A-Z][A-Za-z\s\-]+?)\.?\s*$',
        # Lettered sections: "A.", "B."
        r'^([A-Z])\.\s+(.+?)\.?\s*$',
        # Numbered with letters: "1.1", "2.1.1"
        r'^(\d+(?:\.\d+)+)\s+(.+?)\.?\s*$',
        # ARTICLE format: "ARTICLE I", "ARTICLE 1"
        r'^(ARTICLE\s+[IVX\d]+)[:\.\s]+(.+?)\.?\s*$',
        # SECTION format: "SECTION 1", "Section 2"
        r'^(SECTION\s+\d+)[:\.\s]+(.+?)\.?\s*$',
        # All caps headers
        r'^(\d+)\.\s+([A-Z][A-Z\s\-]+)\.?\s*$',
    ]

    def __init__(self) -> None:
        """Initialize the clause detector."""
        self._compiled_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE) 
                                   for p in self.SECTION_PATTERNS]

    def detect_clauses(self, text: str) -> List[ExtractedClause]:
        """Detect and extract all clauses from document text.
        
        Args:
            text: Full document text.
            
        Returns:
            List of extracted clauses.
        """
        # Find all section headers
        sections = self._find_section_headers(text)
        
        if not sections:
            # No clear structure found, treat entire text as one clause
            return [ExtractedClause(
                section_number="1",
                title="Document Content",
                text=text.strip(),
                clause_type=ClauseType.UNKNOWN,
                start_position=0,
                end_position=len(text),
            )]
        
        # Extract text between section headers
        clauses = []
        for i, (match, section_num, title) in enumerate(sections):
            start_pos = match.start()
            
            # End position is start of next section or end of document
            if i + 1 < len(sections):
                end_pos = sections[i + 1][0].start()
            else:
                end_pos = len(text)
            
            clause_text = text[start_pos:end_pos].strip()
            clause_type = self._classify_clause(title, clause_text)
            
            clauses.append(ExtractedClause(
                section_number=section_num,
                title=title.strip(),
                text=clause_text,
                clause_type=clause_type,
                start_position=start_pos,
                end_position=end_pos,
            ))
        
        return clauses

    def _find_section_headers(self, text: str) -> List[tuple]:
        """Find all section headers in the text.
        
        Returns:
            List of (match, section_number, title) tuples.
        """
        all_matches = []
        
        for pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                section_num = match.group(1)
                title = match.group(2) if match.lastindex >= 2 else ""
                all_matches.append((match, section_num, title))
        
        # Sort by position and remove duplicates
        all_matches.sort(key=lambda x: x[0].start())
        
        # Remove overlapping matches (keep earlier one)
        filtered = []
        last_end = -1
        for match_tuple in all_matches:
            if match_tuple[0].start() >= last_end:
                filtered.append(match_tuple)
                last_end = match_tuple[0].end()
        
        return filtered

    def _classify_clause(self, title: str, text: str) -> ClauseType:
        """Classify a clause based on its title and content.
        
        Args:
            title: Section title.
            text: Full clause text.
            
        Returns:
            Detected clause type.
        """
        # First check title only (higher priority)
        title_lower = title.lower()
        for keyword, clause_type in CLAUSE_KEYWORDS.items():
            if keyword in title_lower:
                return clause_type
        
        # Then check content
        content_lower = text[:500].lower()
        for keyword, clause_type in CLAUSE_KEYWORDS.items():
            if keyword in content_lower:
                return clause_type
        
        return ClauseType.UNKNOWN

    def extract_to_json(self, text: str) -> dict:
        """Extract clauses and return as JSON-serializable structure.
        
        Args:
            text: Full document text.
            
        Returns:
            Dictionary with document structure.
        """
        clauses = self.detect_clauses(text)
        
        return {
            "total_clauses": len(clauses),
            "clause_types": list(set(c.clause_type.value for c in clauses)),
            "clauses": [c.to_dict() for c in clauses],
        }


# Global instance for dependency injection
clause_detector = ClauseDetector()
