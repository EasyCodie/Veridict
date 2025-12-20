"""AI-Powered Clause Classifier - Worker Agent using GPT-5 Nano.

This module uses LLM intelligence to semantically classify clauses,
replacing brittle keyword matching with true understanding.
"""

import json
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from app.config import get_settings


@dataclass
class ClassificationResult:
    """Result of AI clause classification."""
    clause_type: str
    confidence: float
    reasoning: str
    risk_level: str  # low, medium, high, critical
    key_obligations: list[str]
    red_flags: list[str]
    # Token usage tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0  # Estimated cost


# System prompt for the clause classifier worker
CLASSIFIER_SYSTEM_PROMPT = """System: You are a legal document analysis expert tasked with contract clause classification and risk assessment.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

Given a clause from a legal contract, perform the following:

1. Identify the clause type from standard legal categories.
2. Assess the risk level based on the language used.
3. Extract any key obligations mentioned.
4. Flag potentially problematic terms.

If unsure, clarify assumptions and choose the closest applicable category or risk level (default to 'other' and 'medium' respectively if classification is unclear).

After classifying, validate your output to ensure all required fields are present and explanations are concise.

Standard clause categories (you may also identify and label others not listed):
- indemnification: Hold harmless, indemnify, defend against claims
- termination: End of agreement, cancellation rights
- confidentiality: NDA, proprietary information, trade secrets
- limitation_of_liability: Liability caps, exclusion of damages
- governing_law: Jurisdiction, choice of law, venue
- dispute_resolution: Arbitration, mediation, litigation procedures
- force_majeure: Acts of God, unforeseeable circumstances
- assignment: Transfer of rights, delegation
- amendment: Modification procedures
- waiver: Waiver of rights, no waiver clauses
- severability: Invalidity of provisions
- entire_agreement: Integration clause
- notice: Notification requirements
- intellectual_property: IP ownership, licensing
- warranty: Representations, guarantees
- representations: Statements of fact
- insurance: Coverage requirements
- compliance: Regulatory requirements
- data_protection: Privacy, GDPR, personal data
- audit_rights: Inspection, records access
- payment: Compensation, fees, invoicing
- scope_of_work: Services, deliverables
- term: Duration, renewal
- definitions: Defined terms
- other: For clauses that do not fit standard categories

Risk levels:
- low: Standard language and balanced terms
- medium: Some one-sided terms but reasonable
- high: Significantly one-sided or unusual provisions
- critical: Extremely one-sided or potentially harmful terms

## Output Format
Respond ONLY with valid JSON in the following format:
{
  "clause_type": "string (required; must be one of the standard categories or 'other')",
  "confidence": "float [0.0 - 1.0] (required; likelihood the classification is correct)",
  "reasoning": "string (required; brief explanation of classification)",
  "risk_level": "string (required; one of: low, medium, high, critical)",
  "key_obligations": ["string", ...],
  "red_flags": ["string", ...]
}
Note: The arrays key_obligations and red_flags must be included in output. They may be empty arrays if none found, but must always be present.

Example Output:
{
  "clause_type": "indemnification",
  "confidence": 0.95,
  "reasoning": "The clause explicitly requires one party to indemnify and hold harmless the other party against third-party claims.",
  "risk_level": "high",
  "key_obligations": ["Party A must indemnify Party B for any third-party claims arising from breaches."],
  "red_flags": ["Obligation is one-sided favoring Party B", "No carve-outs for gross negligence"]
}"""


class AIClauseClassifier:
    """AI-powered clause classifier using GPT-5 Nano."""
    
    def __init__(self) -> None:
        """Initialize the AI classifier."""
        self._client: Optional[OpenAI] = None
    
    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            settings = get_settings()
            if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
                raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env")
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client
    
    def classify(self, clause_text: str, title: str = "") -> ClassificationResult:
        """Classify a single clause using AI.
        
        Args:
            clause_text: The full text of the clause.
            title: Optional section title/header.
            
        Returns:
            ClassificationResult with type, confidence, and analysis.
        """
        # Prepare the prompt
        user_prompt = f"""Classify this contract clause:

Title: {title if title else "Not specified"}

Clause Text:
{clause_text[:2000]}  # Limit to avoid token overflow
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=500,
                # GPT-5 Nano specific parameters
                extra_body={
                    "reasoning_effort": "minimal",
                    "verbosity": "low"
                }
            )
            
            result_text = response.choices[0].message.content or "{}"
            
            # Parse JSON response
            # Handle markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text.strip())
            
            # Extract token usage from response
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            
            # Calculate cost (GPT-5 Nano pricing: $0.05/M input, $0.40/M output)
            cost_usd = (input_tokens * 0.05 / 1_000_000) + (output_tokens * 0.40 / 1_000_000)
            
            return ClassificationResult(
                clause_type=result.get("clause_type", "unknown"),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", ""),
                risk_level=result.get("risk_level", "medium"),
                key_obligations=result.get("key_obligations", []),
                red_flags=result.get("red_flags", []),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
            )
            
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            return ClassificationResult(
                clause_type="unknown",
                confidence=0.0,
                reasoning=f"Failed to parse AI response: {str(e)}",
                risk_level="medium",
                key_obligations=[],
                red_flags=["AI classification failed"],
            )
        except Exception as e:
            return ClassificationResult(
                clause_type="unknown",
                confidence=0.0,
                reasoning=f"AI classification error: {str(e)}",
                risk_level="medium",
                key_obligations=[],
                red_flags=["AI classification error"],
            )
    
    def classify_batch(self, clauses: list[dict]) -> list[ClassificationResult]:
        """Classify multiple clauses.
        
        Args:
            clauses: List of dicts with 'text' and optional 'title' keys.
            
        Returns:
            List of ClassificationResults in same order.
        """
        results = []
        for clause in clauses:
            result = self.classify(
                clause_text=clause.get("text", ""),
                title=clause.get("title", "")
            )
            results.append(result)
        return results


# Global instance
ai_classifier = AIClauseClassifier()
