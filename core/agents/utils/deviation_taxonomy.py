"""Deviation Taxonomy for Comparison Agent.

This module defines the deviation types, severity levels, and alignment scoring
for the Comparison Agent in the Micro-Agent Factory.

Usage:
    from core.agents.utils.deviation_taxonomy import DeviationType, Severity
"""

from enum import Enum
from typing import Any


class DeviationType(str, Enum):
    """Types of deviations from market standard language."""
    
    MORE_RESTRICTIVE = "more_restrictive"
    """Clause is stricter than market standard (protects other party)."""
    
    LESS_PROTECTIVE = "less_protective"
    """Clause lacks protections found in market standard (exposes our party)."""
    
    NON_STANDARD = "non_standard"
    """Deviation from typical market language (unusual provision)."""
    
    MISSING_SAFEGUARD = "missing_safeguard"
    """Expected protective provision is absent."""
    
    ADDITIONAL_OBLIGATION = "additional_obligation"
    """Extra obligations not found in market standard."""


class Severity(str, Enum):
    """Severity levels for deviations."""
    
    CRITICAL = "critical"
    """Major deviation requiring immediate attention."""
    
    HIGH = "high"
    """Significant deviation that should be addressed."""
    
    MEDIUM = "medium"
    """Notable deviation worth reviewing."""
    
    LOW = "low"
    """Minor deviation with limited impact."""


class OverallAssessment(str, Enum):
    """Overall assessment of market alignment."""
    
    MARKET_STANDARD = "market_standard"
    """Clause is fully compliant with market standards (0.9-1.0)."""
    
    MINOR_DEVIATIONS = "minor_deviations"
    """Small variations from market standard (0.7-0.9)."""
    
    SIGNIFICANT_DEVIATIONS = "significant_deviations"
    """Material changes from market standard (0.5-0.7)."""
    
    HIGHLY_UNUSUAL = "highly_unusual"
    """Major deviations from market practice (0.0-0.5)."""


# Deviation type descriptions for prompts and display
DEVIATION_TYPE_DESCRIPTIONS: dict[DeviationType, str] = {
    DeviationType.MORE_RESTRICTIVE: (
        "Clause imposes stricter requirements than market standard, "
        "typically favoring the counterparty."
    ),
    DeviationType.LESS_PROTECTIVE: (
        "Clause provides fewer protections than market standard, "
        "exposing our party to additional risk."
    ),
    DeviationType.NON_STANDARD: (
        "Language deviates from typical market practice without clear "
        "benefit to either party."
    ),
    DeviationType.MISSING_SAFEGUARD: (
        "Expected protective provision commonly found in market standard "
        "is absent from this clause."
    ),
    DeviationType.ADDITIONAL_OBLIGATION: (
        "Clause contains extra obligations beyond what is typically "
        "found in market standard language."
    ),
}


# Severity weights for alignment score calculation
SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.CRITICAL: 0.40,
    Severity.HIGH: 0.25,
    Severity.MEDIUM: 0.10,
    Severity.LOW: 0.05,
}


def calculate_alignment_score(deviations: list[dict[str, Any]]) -> float:
    """Calculate alignment score from deviations.
    
    Score is 1.0 (fully aligned) minus penalties for each deviation.
    
    Args:
        deviations: List of deviation dictionaries with 'severity' field.
        
    Returns:
        Alignment score between 0 and 1.
    """
    if not deviations:
        return 1.0
    
    total_penalty = 0.0
    for deviation in deviations:
        severity_str = deviation.get("severity", "low").lower()
        try:
            severity = Severity(severity_str)
            total_penalty += SEVERITY_WEIGHTS.get(severity, 0.05)
        except ValueError:
            total_penalty += 0.05
    
    # Cap penalty at 1.0 (score can't go below 0)
    return max(0.0, 1.0 - total_penalty)


def determine_overall_assessment(alignment_score: float) -> OverallAssessment:
    """Determine overall assessment from alignment score.
    
    Args:
        alignment_score: Score between 0 and 1.
        
    Returns:
        OverallAssessment enum value.
    """
    if alignment_score >= 0.9:
        return OverallAssessment.MARKET_STANDARD
    if alignment_score >= 0.7:
        return OverallAssessment.MINOR_DEVIATIONS
    if alignment_score >= 0.5:
        return OverallAssessment.SIGNIFICANT_DEVIATIONS
    return OverallAssessment.HIGHLY_UNUSUAL


def aggregate_deviations(deviations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate deviation statistics.
    
    Args:
        deviations: List of deviation dictionaries.
        
    Returns:
        Dictionary with aggregated statistics.
    """
    result = {
        "total_deviations": len(deviations),
        "by_severity": {s.value: 0 for s in Severity},
        "by_type": {t.value: 0 for t in DeviationType},
        "highest_severity": None,
        "alignment_score": 1.0,
    }
    
    if not deviations:
        return result
    
    highest_weight = 0.0
    highest_severity = None
    
    for deviation in deviations:
        # Count by severity
        severity_str = deviation.get("severity", "low").lower()
        if severity_str in result["by_severity"]:
            result["by_severity"][severity_str] += 1
        
        try:
            severity = Severity(severity_str)
            weight = SEVERITY_WEIGHTS.get(severity, 0.0)
            if weight > highest_weight:
                highest_weight = weight
                highest_severity = severity
        except ValueError:
            pass
        
        # Count by type
        type_str = deviation.get("deviation_type", "").lower()
        if type_str in result["by_type"]:
            result["by_type"][type_str] += 1
    
    result["highest_severity"] = highest_severity.value if highest_severity else None
    result["alignment_score"] = calculate_alignment_score(deviations)
    
    return result


def normalize_deviation_type(value: str) -> str:
    """Normalize a deviation type string to valid enum value.
    
    Args:
        value: Input deviation type string.
        
    Returns:
        Normalized deviation type string.
    """
    value = value.lower().strip().replace(" ", "_").replace("-", "_")
    
    valid_types = {t.value for t in DeviationType}
    
    if value in valid_types:
        return value
    
    # Best-effort mapping
    if "restrictive" in value or "strict" in value:
        return DeviationType.MORE_RESTRICTIVE.value
    if "protective" in value or "exposed" in value or "risk" in value:
        return DeviationType.LESS_PROTECTIVE.value
    if "missing" in value or "absent" in value:
        return DeviationType.MISSING_SAFEGUARD.value
    if "additional" in value or "extra" in value or "obligation" in value:
        return DeviationType.ADDITIONAL_OBLIGATION.value
    
    return DeviationType.NON_STANDARD.value


def normalize_severity(value: str) -> str:
    """Normalize a severity string to valid enum value.
    
    Args:
        value: Input severity string.
        
    Returns:
        Normalized severity string.
    """
    value = value.lower().strip()
    
    valid_severities = {s.value for s in Severity}
    
    if value in valid_severities:
        return value
    
    if "critical" in value or "major" in value:
        return Severity.CRITICAL.value
    if "high" in value or "significant" in value:
        return Severity.HIGH.value
    if "medium" in value or "moderate" in value:
        return Severity.MEDIUM.value
    
    return Severity.LOW.value
