"""Micro-Agent Factory - Component 2 of the MAKER Framework.

This module provides worker agents for contract analysis including:
- AIClauseClassifier: Classifies clauses by type using GPT-5 Mini
- ObligationExtractorAgent: Extracts obligations from clauses using GPT-5 Nano

See core.agents for the full agent implementations.
"""

from core.workers.classifier import AIClauseClassifier, ClassificationResult

__all__ = [
    "AIClauseClassifier",
    "ClassificationResult",
]
