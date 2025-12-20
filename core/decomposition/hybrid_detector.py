"""Enhanced Clause Detector with Hybrid AI+Regex Approach.

Structure detection: Regex (reliable for section boundaries)
Classification: AI (semantic understanding via GPT-4o-mini)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from core.workers.classifier import ai_classifier, ClassificationResult


@dataclass
class ExtractedClause:
    """A single clause extracted from a legal document."""
    section_number: str
    title: str
    text: str
    clause_type: str
    start_position: int
    end_position: int
    confidence: float = 1.0
    risk_level: str = "medium"
    key_obligations: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    reasoning: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    sub_clauses: List["ExtractedClause"] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "section_number": self.section_number,
            "title": self.title,
            "text": self.text,
            "clause_type": self.clause_type,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "key_obligations": self.key_obligations,
            "red_flags": self.red_flags,
            "reasoning": self.reasoning,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "sub_clauses": [sc.to_dict() for sc in self.sub_clauses],
        }


class HybridClauseDetector:
    """Hybrid clause detector using regex for structure and AI for classification."""

    # Pattern to match section headers
    SECTION_PATTERNS = [
        r'^(\d+)\.\s+([A-Z][A-Za-z\s\-]+?)\.?\s*$',
        r'^([A-Z])\.\s+(.+?)\.?\s*$',
        r'^(\d+(?:\.\d+)+)\s+(.+?)\.?\s*$',
        r'^(ARTICLE\s+[IVX\d]+)[:\.\s]+(.+?)\.?\s*$',
        r'^(SECTION\s+\d+)[:\.\s]+(.+?)\.?\s*$',
        r'^(\d+)\.\s+([A-Z][A-Z\s\-]+)\.?\s*$',
    ]

    def __init__(self) -> None:
        """Initialize the hybrid detector."""
        self._compiled_patterns = [
            re.compile(p, re.MULTILINE | re.IGNORECASE) 
            for p in self.SECTION_PATTERNS
        ]
        self._use_ai = True  # Toggle for AI classification

    def detect_clauses(self, text: str, use_ai: bool = True) -> List[ExtractedClause]:
        """Detect and extract all clauses from document text.
        
        Args:
            text: Full document text.
            use_ai: Whether to use AI for classification (default True).
            
        Returns:
            List of extracted clauses with AI classification.
        """
        # Step 1: Find structural boundaries (regex)
        sections = self._find_section_headers(text)
        
        if not sections:
            # No clear structure found
            return [ExtractedClause(
                section_number="1",
                title="Document Content",
                text=text.strip(),
                clause_type="unstructured",
                start_position=0,
                end_position=len(text),
            )]
        
        # Step 2: Extract text between sections
        raw_clauses = []
        for i, (match, section_num, title) in enumerate(sections):
            start_pos = match.start()
            end_pos = sections[i + 1][0].start() if i + 1 < len(sections) else len(text)
            clause_text = text[start_pos:end_pos].strip()
            
            raw_clauses.append({
                "section_number": section_num,
                "title": title.strip(),
                "text": clause_text,
                "start_position": start_pos,
                "end_position": end_pos,
            })
        
        # Step 3: Classify using AI (if enabled)
        clauses = []
        for raw in raw_clauses:
            if use_ai:
                try:
                    ai_result = ai_classifier.classify(
                        clause_text=raw["text"],
                        title=raw["title"]
                    )
                    clause = ExtractedClause(
                        section_number=raw["section_number"],
                        title=raw["title"],
                        text=raw["text"],
                        clause_type=ai_result.clause_type,
                        start_position=raw["start_position"],
                        end_position=raw["end_position"],
                        confidence=ai_result.confidence,
                        risk_level=ai_result.risk_level,
                        key_obligations=ai_result.key_obligations,
                        red_flags=ai_result.red_flags,
                        reasoning=ai_result.reasoning,
                        input_tokens=ai_result.input_tokens,
                        output_tokens=ai_result.output_tokens,
                        cost_usd=ai_result.cost_usd,
                    )
                except Exception as e:
                    # Fallback to basic detection
                    clause = ExtractedClause(
                        section_number=raw["section_number"],
                        title=raw["title"],
                        text=raw["text"],
                        clause_type="unknown",
                        start_position=raw["start_position"],
                        end_position=raw["end_position"],
                        reasoning=f"AI unavailable: {str(e)}",
                    )
            else:
                clause = ExtractedClause(
                    section_number=raw["section_number"],
                    title=raw["title"],
                    text=raw["text"],
                    clause_type="pending_classification",
                    start_position=raw["start_position"],
                    end_position=raw["end_position"],
                )
            clauses.append(clause)
        
        return clauses

    def _find_section_headers(self, text: str) -> List[tuple]:
        """Find all section headers using regex patterns."""
        all_matches = []
        
        for pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                section_num = match.group(1)
                title = match.group(2) if match.lastindex >= 2 else ""
                all_matches.append((match, section_num, title))
        
        # Sort by position and remove overlaps
        all_matches.sort(key=lambda x: x[0].start())
        
        filtered = []
        last_end = -1
        for match_tuple in all_matches:
            if match_tuple[0].start() >= last_end:
                filtered.append(match_tuple)
                last_end = match_tuple[0].end()
        
        return filtered

    def extract_to_json(self, text: str, use_ai: bool = True) -> dict:
        """Extract clauses and return as JSON structure."""
        clauses = self.detect_clauses(text, use_ai=use_ai)
        
        # Count risk levels
        risk_summary = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for c in clauses:
            if c.risk_level in risk_summary:
                risk_summary[c.risk_level] += 1
        
        # Collect all red flags
        all_red_flags = []
        for c in clauses:
            all_red_flags.extend(c.red_flags)
        
        # Calculate total token usage
        total_input_tokens = sum(c.input_tokens for c in clauses)
        total_output_tokens = sum(c.output_tokens for c in clauses)
        total_cost = sum(c.cost_usd for c in clauses)
        
        return {
            "total_clauses": len(clauses),
            "clause_types": list(set(c.clause_type for c in clauses)),
            "risk_summary": risk_summary,
            "total_red_flags": len(all_red_flags),
            "red_flags": all_red_flags,
            "usage": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "total_cost_usd": round(total_cost, 6),
            },
            "clauses": [c.to_dict() for c in clauses],
        }


# Global instance
hybrid_detector = HybridClauseDetector()
