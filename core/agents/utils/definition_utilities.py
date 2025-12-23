"""Definition Utilities for Definition Validator Agent.

This module provides utilities for parsing definitions sections, extracting
defined terms, and cross-referencing term usage across contract clauses.

Usage:
    from core.agents.utils.definition_utilities import extract_defined_terms
    from core.agents.utils.definition_utilities import find_capitalized_terms
"""

import re
from enum import Enum
from typing import Any


class DefinitionSeverity(str, Enum):
    """Severity levels for undefined terms."""
    
    CRITICAL = "critical"
    """Key term used throughout contract, missing definition is fatal risk."""
    
    HIGH = "high"
    """Important operational term, undefined creates ambiguity."""
    
    MEDIUM = "medium"
    """Secondary term, missing definition complicates interpretation."""


# Common legal terms that don't require definitions (standard meanings)
COMMON_LEGAL_TERMS = {
    "agreement", "party", "parties", "section", "clause", "article",
    "paragraph", "herein", "hereof", "hereby", "hereto", "hereunder",
    "therein", "thereof", "thereby", "thereto", "thereunder",
    "shall", "will", "may", "must", "including", "includes",
    "without limitation", "provided that", "notwithstanding",
    "in connection with", "pursuant to", "subject to", "with respect to",
    "effective date", "date hereof", "term", "termination",
    "notice", "written notice", "business day", "calendar day",
    "representation", "warranty", "covenant", "indemnification",
    "liability", "damages", "breach", "default", "force majeure",
    "governing law", "jurisdiction", "arbitration", "dispute",
    "confidential", "proprietary", "intellectual property",
    "amendment", "waiver", "assignment", "successor", "assign",
    "severability", "entire agreement", "counterparts",
}

# Standard locations where definitions are typically found
DEFINITION_LOCATIONS = [
    "Section 1 - Definitions",
    "Article 1 - Definitions",
    "Section 1.1 Definitions",
    "Definitions section",
    "Exhibit A - Definitions",
    "Schedule of Definitions",
    "Recitals section",
]


def extract_defined_terms(definitions_text: str) -> set[str]:
    """Extract defined terms from a definitions section.
    
    Looks for patterns like:
    - "Term" means...
    - 'Term' means...
    - "Term" shall mean...
    - Term: definition
    
    Args:
        definitions_text: Text of the definitions section.
        
    Returns:
        Set of defined term strings.
    """
    defined_terms: set[str] = set()
    
    if not definitions_text:
        return defined_terms
    
    # Pattern 1: "Term" means/shall mean
    pattern1 = r'"([A-Z][A-Za-z\s]+?)"\s+(?:shall\s+)?(?:mean|refers? to|has the meaning)'
    matches = re.findall(pattern1, definitions_text, re.IGNORECASE)
    defined_terms.update(m.strip() for m in matches)
    
    # Pattern 2: 'Term' means/shall mean
    pattern2 = r"'([A-Z][A-Za-z\s]+?)'\s+(?:shall\s+)?(?:mean|refers? to|has the meaning)"
    matches = re.findall(pattern2, definitions_text, re.IGNORECASE)
    defined_terms.update(m.strip() for m in matches)
    
    # Pattern 3: (a) "Term" definition style (numbered definitions)
    pattern3 = r'[(\d+)]\s*"([A-Z][A-Za-z\s]+?)"'
    matches = re.findall(pattern3, definitions_text)
    defined_terms.update(m.strip() for m in matches)
    
    # Pattern 4: Term: definition (colon-based)
    pattern4 = r'^([A-Z][A-Za-z\s]{2,30}):\s+[A-Z]'
    matches = re.findall(pattern4, definitions_text, re.MULTILINE)
    defined_terms.update(m.strip() for m in matches)
    
    return defined_terms


def find_capitalized_terms(clause_text: str) -> list[str]:
    """Find capitalized terms in a clause that may need definitions.
    
    Identifies terms that:
    - Start with capital letter
    - Are not at start of sentence
    - Are not common legal terms
    - May be multi-word terms (e.g., "Material Adverse Effect")
    
    Args:
        clause_text: Text of the clause to analyze.
        
    Returns:
        List of capitalized terms found.
    """
    terms: list[str] = []
    
    if not clause_text:
        return terms
    
    # Find single capitalized words (not at sentence start)
    # Look for words preceded by lowercase letter, comma, or mid-sentence position
    pattern = r'(?<=[a-z,;]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    matches = re.findall(pattern, clause_text)
    
    # Also find quoted terms which are likely defined terms
    quoted_pattern = r'"([A-Z][A-Za-z\s]+?)"'
    quoted_matches = re.findall(quoted_pattern, clause_text)
    
    all_matches = matches + quoted_matches
    
    for term in all_matches:
        term = term.strip()
        # Filter out common terms that don't need definitions
        if term.lower() not in COMMON_LEGAL_TERMS and len(term) > 1:
            if term not in terms:
                terms.append(term)
    
    return terms


def is_term_defined(term: str, defined_terms: set[str]) -> bool:
    """Check if a term is in the set of defined terms.
    
    Performs case-insensitive matching and handles variations.
    
    Args:
        term: The term to check.
        defined_terms: Set of known defined terms.
        
    Returns:
        True if term is defined, False otherwise.
    """
    if not term or not defined_terms:
        return False
    
    term_lower = term.lower().strip()
    defined_lower = {t.lower() for t in defined_terms}
    
    # Direct match
    if term_lower in defined_lower:
        return True
    
    # Check if term is part of a longer defined term
    for defined in defined_lower:
        if term_lower in defined or defined in term_lower:
            return True
    
    return False


def suggest_definition_location(term: str) -> list[str]:
    """Suggest where a term's definition might be found.
    
    Args:
        term: The undefined term.
        
    Returns:
        List of suggested locations to check.
    """
    suggestions = ["Section 1 - Definitions"]
    
    term_lower = term.lower()
    
    # Add specific suggestions based on term type
    if any(word in term_lower for word in ["material", "adverse", "change"]):
        suggestions.append("MAC/MAE definition clause")
    
    if any(word in term_lower for word in ["purchase", "price", "consideration"]):
        suggestions.append("Purchase Price section")
        suggestions.append("Consideration section")
    
    if any(word in term_lower for word in ["closing", "completion"]):
        suggestions.append("Closing/Completion section")
    
    if any(word in term_lower for word in ["intellectual", "property", "ip"]):
        suggestions.append("IP Schedule/Exhibit")
    
    if any(word in term_lower for word in ["employee", "benefit", "compensation"]):
        suggestions.append("Employee Benefits Schedule")
    
    if any(word in term_lower for word in ["permit", "license", "regulatory"]):
        suggestions.append("Regulatory/Permits Schedule")
    
    if any(word in term_lower for word in ["subsidiary", "affiliate"]):
        suggestions.append("Corporate Structure Schedule")
    
    # Generic fallback
    if len(suggestions) == 1:
        suggestions.append("Recitals section")
        suggestions.append("Schedules and Exhibits")
    
    return suggestions


def classify_term_severity(
    term: str,
    usage_count: int = 1,
    is_quoted: bool = False,
) -> DefinitionSeverity:
    """Classify the severity of an undefined term.
    
    Args:
        term: The undefined term.
        usage_count: How many times term appears in contract.
        is_quoted: Whether term appears in quotes (suggesting importance).
        
    Returns:
        Severity classification.
    """
    term_lower = term.lower()
    
    # Critical: Core deal terms
    critical_indicators = [
        "purchase", "price", "consideration", "material", "adverse",
        "closing", "completion", "target", "buyer", "seller",
        "shares", "assets", "business", "transaction", "agreement",
    ]
    
    if any(indicator in term_lower for indicator in critical_indicators):
        return DefinitionSeverity.CRITICAL
    
    if is_quoted:
        return DefinitionSeverity.CRITICAL
    
    if usage_count >= 5:
        return DefinitionSeverity.CRITICAL
    
    # High: Important operational terms
    high_indicators = [
        "intellectual", "property", "confidential", "employee",
        "subsidiary", "affiliate", "representative", "indemnif",
        "liability", "breach", "default", "termination",
    ]
    
    if any(indicator in term_lower for indicator in high_indicators):
        return DefinitionSeverity.HIGH
    
    if usage_count >= 3:
        return DefinitionSeverity.HIGH
    
    # Medium: Everything else
    return DefinitionSeverity.MEDIUM


def estimate_definition_elsewhere(term: str) -> bool:
    """Estimate if a term is likely defined elsewhere in the contract.
    
    Based on common contract structure patterns.
    
    Args:
        term: The undefined term.
        
    Returns:
        True if term is likely defined elsewhere, False otherwise.
    """
    term_lower = term.lower()
    
    # Terms that are commonly defined in standard contracts
    commonly_defined = [
        "material adverse", "purchase price", "closing date",
        "business day", "confidential information", "intellectual property",
        "affiliate", "subsidiary", "representative", "knowledge",
        "permitted", "excluded", "scheduled", "disclosed",
    ]
    
    for pattern in commonly_defined:
        if pattern in term_lower or term_lower in pattern:
            return True
    
    # Multi-word capitalized terms are usually defined
    if " " in term and term[0].isupper():
        return True
    
    return False


def aggregate_undefined_terms(terms: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate statistics about undefined terms.
    
    Args:
        terms: List of undefined term dictionaries.
        
    Returns:
        Dictionary with aggregated statistics.
    """
    result = {
        "total_undefined": len(terms),
        "by_severity": {s.value: 0 for s in DefinitionSeverity},
        "likely_elsewhere_count": 0,
        "needs_definition_count": 0,
    }
    
    if not terms:
        return result
    
    for term in terms:
        severity = term.get("severity", "medium")
        if severity in result["by_severity"]:
            result["by_severity"][severity] += 1
        
        if term.get("likelihood_defined_elsewhere", False):
            result["likely_elsewhere_count"] += 1
        else:
            result["needs_definition_count"] += 1
    
    return result


def normalize_severity(value: str) -> str:
    """Normalize a severity string to valid enum value.
    
    Args:
        value: Input severity string.
        
    Returns:
        Normalized severity string.
    """
    value = value.lower().strip()
    
    valid_severities = {s.value for s in DefinitionSeverity}
    
    if value in valid_severities:
        return value
    
    if "critical" in value or "fatal" in value or "major" in value:
        return DefinitionSeverity.CRITICAL.value
    if "high" in value or "important" in value:
        return DefinitionSeverity.HIGH.value
    
    return DefinitionSeverity.MEDIUM.value
