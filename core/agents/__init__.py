"""Micro-Agent Factory - Agent implementations for the MAKER Framework.

This package contains specialized agent implementations for analyzing
contract clauses. Each agent is stateless and hyper-focused on a single
task type.
"""

from core.agents.obligation_extractor import (
    ObligationExtractorAgent,
    ObligationExtractorInput,
    ObligationExtractorOutput,
    Obligation,
)

__all__ = [
    "ObligationExtractorAgent",
    "ObligationExtractorInput",
    "ObligationExtractorOutput",
    "Obligation",
]
