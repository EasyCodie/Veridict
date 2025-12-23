"""Finding Taxonomy for Clause Analysis Agent.

This module defines the finding types, importance levels, and aggregation utilities
for the Clause Analysis Agent in the Micro-Agent Factory.

Usage:
    from core.agents.utils.finding_taxonomy import FindingType, Importance, RiskLevel
"""

from enum import Enum
from typing import Any


class FindingType(str, Enum):
    """Types of findings that can be extracted from contract clauses."""
    
    OBLIGATION = "obligation"
    """Who must do what and under what conditions."""
    
    CAP = "cap"
    """Financial limits or liability caps."""
    
    EXCEPTION = "exception"
    """Carve-outs or exceptions from main clause provisions."""
    
    DEFINITION = "definition"
    """Key terms defined in or referenced by the clause."""
    
    TRIGGER = "trigger"
    """Events or conditions that activate clause provisions."""
    
    DURATION = "duration"
    """Time periods or effective dates for clause provisions."""
    
    PROCESS = "process"
    """Procedural steps or requirements specified."""


class Importance(str, Enum):
    """Importance levels for extracted findings."""
    
    CRITICAL = "critical"
    """Must be addressed before signing. Material impact on deal."""
    
    HIGH = "high"
    """Significant finding that warrants attention."""
    
    MEDIUM = "medium"
    """Notable finding worth reviewing."""
    
    LOW = "low"
    """Minor finding with limited impact."""


class RiskLevel(str, Enum):
    """Risk levels for clause analysis."""
    
    CRITICAL = "critical"
    """Immediate attention required. Significant exposure."""
    
    HIGH = "high"
    """Significant risk that should be addressed."""
    
    MEDIUM = "medium"
    """Notable concern that warrants review."""
    
    LOW = "low"
    """Minor issues with limited impact."""
    
    INFORMATIONAL = "informational"
    """No risks identified. Purely informational."""


class ClauseType(str, Enum):
    """Types of clauses commonly found in contracts."""
    
    INDEMNITY = "Indemnity"
    TERMINATION = "Termination"
    WARRANTY = "Warranty"
    LIABILITY = "Liability"
    CONFIDENTIALITY = "Confidentiality"
    IP = "IP"
    PAYMENT = "Payment"
    DEFINITIONS = "Definitions"
    OTHER = "Other"


# Finding type descriptions for prompts and display
FINDING_TYPE_DESCRIPTIONS: dict[FindingType, str] = {
    FindingType.OBLIGATION: (
        "Who must do what and under what conditions. Includes duties, "
        "responsibilities, and required actions by parties."
    ),
    FindingType.CAP: (
        "Financial limits or liability caps. Includes maximum amounts, "
        "deductibles, and monetary thresholds."
    ),
    FindingType.EXCEPTION: (
        "Carve-outs or exceptions from main clause provisions. Includes "
        "exclusions, limitations on scope, and special cases."
    ),
    FindingType.DEFINITION: (
        "Key terms defined in or referenced by the clause. Includes "
        "capitalized terms and their meanings."
    ),
    FindingType.TRIGGER: (
        "Events or conditions that activate clause provisions. Includes "
        "triggering events, prerequisites, and conditions precedent."
    ),
    FindingType.DURATION: (
        "Time periods or effective dates for clause provisions. Includes "
        "survival periods, notice periods, and deadlines."
    ),
    FindingType.PROCESS: (
        "Procedural steps or requirements specified. Includes notice "
        "requirements, approval processes, and dispute procedures."
    ),
}


# Importance weights for calculating overall risk
IMPORTANCE_WEIGHTS: dict[Importance, int] = {
    Importance.CRITICAL: 100,
    Importance.HIGH: 50,
    Importance.MEDIUM: 20,
    Importance.LOW: 5,
}


def get_finding_description(finding_type: FindingType | str) -> str:
    """Get detailed description for a finding type.
    
    Args:
        finding_type: The finding type to describe.
        
    Returns:
        Description string for the finding type.
    """
    if isinstance(finding_type, str):
        try:
            finding_type = FindingType(finding_type)
        except ValueError:
            return "Unknown finding type."
    
    return FINDING_TYPE_DESCRIPTIONS.get(finding_type, "Unknown finding type.")


def calculate_risk_level(findings: list[dict[str, Any]]) -> RiskLevel:
    """Calculate risk level from a list of findings.
    
    The risk level is determined by the highest importance finding:
    - Any CRITICAL → CRITICAL
    - Any HIGH → HIGH
    - Any MEDIUM → MEDIUM
    - Any LOW → LOW
    - No findings → INFORMATIONAL
    
    Args:
        findings: List of finding dictionaries with 'importance' field.
        
    Returns:
        RiskLevel enum value.
    """
    if not findings:
        return RiskLevel.INFORMATIONAL
    
    importances = set()
    for finding in findings:
        importance_str = finding.get("importance", "low")
        try:
            importances.add(Importance(importance_str.lower()))
        except ValueError:
            importances.add(Importance.LOW)
    
    if Importance.CRITICAL in importances:
        return RiskLevel.CRITICAL
    
    if Importance.HIGH in importances:
        return RiskLevel.HIGH
    
    if Importance.MEDIUM in importances:
        return RiskLevel.MEDIUM
    
    return RiskLevel.LOW


def aggregate_findings(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate finding statistics for a collection of findings.
    
    Args:
        findings: List of finding dictionaries.
        
    Returns:
        Dictionary with aggregated statistics.
    """
    result = {
        "total_findings": len(findings),
        "by_importance": {i.value: 0 for i in Importance},
        "by_type": {t.value: 0 for t in FindingType},
        "importance_score": 0,
        "highest_importance": None,
        "clarification_count": 0,
    }
    
    if not findings:
        return result
    
    highest_weight = 0
    highest_importance = None
    
    for finding in findings:
        # Count by importance
        importance_str = finding.get("importance", "low").lower()
        try:
            importance = Importance(importance_str)
            result["by_importance"][importance.value] += 1
            
            weight = IMPORTANCE_WEIGHTS.get(importance, 0)
            result["importance_score"] += weight
            
            if weight > highest_weight:
                highest_weight = weight
                highest_importance = importance
        except ValueError:
            result["by_importance"]["low"] += 1
        
        # Count by type
        finding_type_str = finding.get("finding_type", "").lower()
        if finding_type_str in result["by_type"]:
            result["by_type"][finding_type_str] += 1
    
    result["highest_importance"] = highest_importance.value if highest_importance else None
    
    return result


def normalize_finding_type(value: str) -> str:
    """Normalize a finding type string to valid enum value.
    
    Args:
        value: Input finding type string.
        
    Returns:
        Normalized finding type string.
    """
    value = value.lower().strip().replace(" ", "_").replace("-", "_")
    
    valid_types = {t.value for t in FindingType}
    
    if value in valid_types:
        return value
    
    # Best-effort mapping for common variations
    if "obligat" in value or "duty" in value or "must" in value:
        return FindingType.OBLIGATION.value
    if "cap" in value or "limit" in value or "maximum" in value:
        return FindingType.CAP.value
    if "except" in value or "carve" in value or "exclud" in value:
        return FindingType.EXCEPTION.value
    if "defin" in value or "mean" in value or "term" in value:
        return FindingType.DEFINITION.value
    if "trigger" in value or "event" in value or "condition" in value:
        return FindingType.TRIGGER.value
    if "duration" in value or "period" in value or "time" in value:
        return FindingType.DURATION.value
    if "process" in value or "procedure" in value or "step" in value:
        return FindingType.PROCESS.value
    
    return FindingType.OBLIGATION.value


def normalize_importance(value: str) -> str:
    """Normalize an importance string to valid enum value.
    
    Args:
        value: Input importance string.
        
    Returns:
        Normalized importance string.
    """
    value = value.lower().strip()
    
    valid_importances = {i.value for i in Importance}
    
    if value in valid_importances:
        return value
    
    # Best-effort mapping
    if "critical" in value or "must" in value or "essential" in value:
        return Importance.CRITICAL.value
    if "high" in value or "major" in value or "significant" in value:
        return Importance.HIGH.value
    if "medium" in value or "moderate" in value:
        return Importance.MEDIUM.value
    
    return Importance.LOW.value


def normalize_risk_level(value: str) -> str:
    """Normalize a risk level string to valid enum value.
    
    Args:
        value: Input risk level string.
        
    Returns:
        Normalized risk level string.
    """
    value = value.lower().strip().replace(" ", "_").replace("-", "_")
    
    valid_levels = {r.value for r in RiskLevel}
    
    if value in valid_levels:
        return value
    
    if "critical" in value:
        return RiskLevel.CRITICAL.value
    if "high" in value:
        return RiskLevel.HIGH.value
    if "medium" in value or "moderate" in value:
        return RiskLevel.MEDIUM.value
    if "low" in value or "minor" in value:
        return RiskLevel.LOW.value
    
    return RiskLevel.INFORMATIONAL.value
