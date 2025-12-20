"""MAKER Framework Reliability Filter.

Detects process unreliability in LLM responses per the MAKER framework.
When an agent exhibits "confused" behavior, the response should be discarded
and resampled rather than used for final decisions.

Key insight: "Bad behaviors are correlated - if a model produces structural 
issues, its internal reasoning is likely compromised."
"""

import re
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReliabilityCheck:
    """Result of a reliability check."""
    is_reliable: bool
    reason: str = ""
    pattern_matched: Optional[str] = None


class ReliabilityFilter:
    """Filters out unreliable LLM responses based on MAKER framework principles.
    
    Detects:
    1. Logical loop markers (agent "talking in circles")
    2. Uncertainty/confusion markers
    3. Token overflow (reasoning cliff at ~750 tokens)
    4. Format violations (invalid JSON structure)
    """
    
    # Reasoning cliff threshold from MAKER research
    MAX_OUTPUT_TOKENS = 750
    
    # Patterns indicating agent confusion/unreliability
    CONFUSION_PATTERNS = [
        # Logical loops - agent is doing too much work
        r"wait,?\s*maybe",
        r"let'?s\s*(check|try)\s*again",
        r"is\s*there\s*a\s*mistake",
        r"on\s*second\s*thought",
        r"actually,?\s*no",
        r"I\s*need\s*to\s*reconsider",
        r"let\s*me\s*recalculate",
        r"I\s*(made|think\s*I\s*made)\s*an?\s*error",
        
        # Uncertainty markers - agent is unsure
        r"is\s*ambiguous\s*here",
        r"I'?m\s*not\s*(entirely\s*)?sure",
        r"this\s*could\s*mean",
        r"it'?s\s*(hard|difficult)\s*to\s*(say|tell|determine)",
        r"I\s*cannot\s*determine",
        r"unclear\s*from\s*the\s*text",
        r"there\s*is\s*no\s*clear\s*answer",
        r"multiple\s*interpretations",
    ]
    
    # Required fields for valid classification output
    REQUIRED_FIELDS = ["clause_type", "confidence", "risk_level"]
    
    def __init__(self, max_tokens: int = MAX_OUTPUT_TOKENS):
        """Initialize the filter.
        
        Args:
            max_tokens: Maximum allowed output tokens before triggering cutoff.
        """
        self.max_tokens = max_tokens
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CONFUSION_PATTERNS
        ]
    
    def check(self, response_text: str, output_tokens: int) -> ReliabilityCheck:
        """Check if a response is reliable.
        
        Args:
            response_text: The raw LLM response content.
            output_tokens: Number of tokens in the response.
            
        Returns:
            ReliabilityCheck with is_reliable=True if acceptable.
        """
        # Check 1: Token overflow (reasoning cliff)
        if output_tokens > self.max_tokens:
            return ReliabilityCheck(
                is_reliable=False,
                reason=f"Response exceeded token limit ({output_tokens} > {self.max_tokens})",
            )
        
        # Check 2: Confusion patterns in response
        for pattern in self._compiled_patterns:
            match = pattern.search(response_text)
            if match:
                return ReliabilityCheck(
                    is_reliable=False,
                    reason=f"Confusion marker detected: '{match.group()}'",
                    pattern_matched=pattern.pattern,
                )
        
        # Check 3: JSON format validity (if applicable)
        format_check = self._check_json_format(response_text)
        if not format_check.is_reliable:
            return format_check
        
        return ReliabilityCheck(is_reliable=True)
    
    def _check_json_format(self, response_text: str) -> ReliabilityCheck:
        """Verify the response contains valid JSON with required fields."""
        # Extract JSON from potential markdown code blocks
        text = response_text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        text = text.strip()
        
        # Try to parse as JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Don't fail here - the classifier has its own JSON repair
            # Only fail on obvious non-JSON responses
            if not text.startswith("{"):
                return ReliabilityCheck(
                    is_reliable=False,
                    reason="Response is not JSON format",
                )
            return ReliabilityCheck(is_reliable=True)
        
        # Check for required fields
        missing = [f for f in self.REQUIRED_FIELDS if f not in data]
        if missing:
            return ReliabilityCheck(
                is_reliable=False,
                reason=f"Missing required fields: {missing}",
            )
        
        return ReliabilityCheck(is_reliable=True)


# Singleton instance for convenience
reliability_filter = ReliabilityFilter()
