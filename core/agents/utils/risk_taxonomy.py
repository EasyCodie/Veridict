"""Risk Taxonomy for Risk Identifier Agent.

This module defines the risk types, severity levels, and aggregation utilities
for the Risk Identifier Agent in the Micro-Agent Factory.

Usage:
    from core.agents.utils.risk_taxonomy import RiskType, Severity, OverallRiskRating
    from core.agents.utils.risk_taxonomy import calculate_overall_rating, aggregate_risks
"""

from enum import Enum
from typing import Any


class RiskType(str, Enum):
    """Types of risks that can be identified in contract clauses."""
    
    UNUSUAL_LANGUAGE = "unusual_language"
    """Non-standard or archaic legal terminology that deviates from common usage."""
    
    AMBIGUOUS_DEFINITION = "ambiguous_definition"
    """Unclear terms or vague provisions that could lead to multiple interpretations."""
    
    CONTRADICTS_STANDARD = "contradicts_standard"
    """Language that deviates from market-standard provisions in similar contracts."""
    
    MISSING_SAFEGUARD = "missing_safeguard"
    """Expected protective provisions are absent (e.g., notice period, cure rights)."""
    
    UNLIMITED_EXPOSURE = "unlimited_exposure"
    """Unlimited liability or obligations without caps, baskets, or time limits."""
    
    UNDEFINED_TERM = "undefined_term"
    """Key terms used without definition in the contract or clause."""


class Severity(str, Enum):
    """Severity levels for identified risks."""
    
    CRITICAL = "critical"
    """Immediate attention required. Could result in significant financial or legal exposure."""
    
    HIGH = "high"
    """Significant risk that should be addressed before signing."""
    
    MEDIUM = "medium"
    """Notable concern that warrants review but may be acceptable."""
    
    LOW = "low"
    """Minor issue with limited impact."""


class OverallRiskRating(str, Enum):
    """Overall risk rating for a clause or contract."""
    
    CLEAN = "clean"
    """No risks identified."""
    
    MINOR_ISSUES = "minor_issues"
    """Only low-severity risks present."""
    
    SIGNIFICANT_ISSUES = "significant_issues"
    """Medium or high-severity risks present."""
    
    CRITICAL_ISSUES = "critical_issues"
    """At least one critical-severity risk present."""


# Risk type descriptions for prompting and display
RISK_TYPE_DESCRIPTIONS: dict[RiskType, str] = {
    RiskType.UNUSUAL_LANGUAGE: (
        "Non-standard or archaic legal terminology that deviates from common "
        "usage in similar contracts. May indicate intentional obfuscation."
    ),
    RiskType.AMBIGUOUS_DEFINITION: (
        "Unclear terms or vague provisions that could lead to multiple "
        "interpretations and potential disputes."
    ),
    RiskType.CONTRADICTS_STANDARD: (
        "Language that deviates from market-standard provisions, potentially "
        "shifting risk unfavorably or creating unexpected obligations."
    ),
    RiskType.MISSING_SAFEGUARD: (
        "Expected protective provisions are absent, such as notice periods, "
        "cure rights, or limitation of liability clauses."
    ),
    RiskType.UNLIMITED_EXPOSURE: (
        "Unlimited liability, indemnification, or obligations without caps, "
        "baskets, deductibles, or time limitations."
    ),
    RiskType.UNDEFINED_TERM: (
        "Key terms used in the clause without definition elsewhere in the "
        "contract, leading to interpretation uncertainty."
    ),
}


# Severity weights for calculating overall rating
SEVERITY_WEIGHTS: dict[Severity, int] = {
    Severity.CRITICAL: 100,
    Severity.HIGH: 50,
    Severity.MEDIUM: 20,
    Severity.LOW: 5,
}


def get_risk_description(risk_type: RiskType | str) -> str:
    """Get detailed description for a risk type.
    
    Args:
        risk_type: The risk type to describe.
        
    Returns:
        Description string for the risk type.
    """
    if isinstance(risk_type, str):
        try:
            risk_type = RiskType(risk_type)
        except ValueError:
            return "Unknown risk type."
    
    return RISK_TYPE_DESCRIPTIONS.get(risk_type, "Unknown risk type.")


def calculate_overall_rating(risks: list[dict[str, Any]]) -> OverallRiskRating:
    """Calculate overall risk rating from a list of identified risks.
    
    The rating is determined by the highest severity risk present:
    - Any CRITICAL → CRITICAL_ISSUES
    - Any HIGH or MEDIUM → SIGNIFICANT_ISSUES
    - Only LOW → MINOR_ISSUES
    - No risks → CLEAN
    
    Args:
        risks: List of risk dictionaries with 'severity' field.
        
    Returns:
        OverallRiskRating enum value.
    """
    if not risks:
        return OverallRiskRating.CLEAN
    
    severities = set()
    for risk in risks:
        severity_str = risk.get("severity", "low")
        try:
            severities.add(Severity(severity_str.lower()))
        except ValueError:
            severities.add(Severity.LOW)
    
    if Severity.CRITICAL in severities:
        return OverallRiskRating.CRITICAL_ISSUES
    
    if Severity.HIGH in severities or Severity.MEDIUM in severities:
        return OverallRiskRating.SIGNIFICANT_ISSUES
    
    if Severity.LOW in severities:
        return OverallRiskRating.MINOR_ISSUES
    
    return OverallRiskRating.CLEAN


def aggregate_risks(risks: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate risk statistics for a collection of risks.
    
    Args:
        risks: List of risk dictionaries.
        
    Returns:
        Dictionary with aggregated statistics:
        - total_risks: Total count
        - by_severity: Count per severity level
        - by_type: Count per risk type
        - risk_score: Weighted severity score
        - highest_severity: The most severe risk level present
    """
    result = {
        "total_risks": len(risks),
        "by_severity": {s.value: 0 for s in Severity},
        "by_type": {t.value: 0 for t in RiskType},
        "risk_score": 0,
        "highest_severity": None,
    }
    
    if not risks:
        return result
    
    highest_weight = 0
    highest_severity = None
    
    for risk in risks:
        # Count by severity
        severity_str = risk.get("severity", "low").lower()
        try:
            severity = Severity(severity_str)
            result["by_severity"][severity.value] += 1
            
            weight = SEVERITY_WEIGHTS.get(severity, 0)
            result["risk_score"] += weight
            
            if weight > highest_weight:
                highest_weight = weight
                highest_severity = severity
        except ValueError:
            result["by_severity"]["low"] += 1
        
        # Count by type
        risk_type_str = risk.get("risk_type", "").lower()
        if risk_type_str in result["by_type"]:
            result["by_type"][risk_type_str] += 1
    
    result["highest_severity"] = highest_severity.value if highest_severity else None
    
    return result


def normalize_risk_type(value: str) -> str:
    """Normalize a risk type string to valid enum value.
    
    Args:
        value: Input risk type string.
        
    Returns:
        Normalized risk type string.
    """
    value = value.lower().strip().replace(" ", "_").replace("-", "_")
    
    valid_types = {t.value for t in RiskType}
    
    if value in valid_types:
        return value
    
    # Best-effort mapping for common variations
    if "unusual" in value or "archaic" in value or "non-standard" in value:
        return RiskType.UNUSUAL_LANGUAGE.value
    if "ambiguous" in value or "vague" in value or "unclear" in value:
        return RiskType.AMBIGUOUS_DEFINITION.value
    if "contradict" in value or "deviate" in value or "non-market" in value:
        return RiskType.CONTRADICTS_STANDARD.value
    if "missing" in value or "absent" in value or "lack" in value:
        return RiskType.MISSING_SAFEGUARD.value
    if "unlimited" in value or "uncapped" in value or "no limit" in value:
        return RiskType.UNLIMITED_EXPOSURE.value
    if "undefined" in value or "not defined" in value:
        return RiskType.UNDEFINED_TERM.value
    
    # Default fallback
    return RiskType.UNUSUAL_LANGUAGE.value


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
    
    # Best-effort mapping
    if "critical" in value or "severe" in value or "extreme" in value:
        return Severity.CRITICAL.value
    if "high" in value or "major" in value:
        return Severity.HIGH.value
    if "medium" in value or "moderate" in value:
        return Severity.MEDIUM.value
    
    return Severity.LOW.value
