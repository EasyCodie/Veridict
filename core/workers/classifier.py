"""AI-Powered Clause Classifier - Worker Agent using GPT-5 Mini.

This module uses LLM intelligence to semantically classify clauses,
replacing brittle keyword matching with true understanding.

Implements MAKER framework reliability filtering with discard-and-retry.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from app.config import get_settings
from core.filtering.reliability_filter import reliability_filter


@dataclass
class ClassificationResult:
    """Result of AI clause classification."""
    clause_type: str
    confidence: float
    confidence_label: str  # "Very High", "High", "Medium", "Low", "Very Low"
    reasoning: str
    risk_level: str  # low, medium, high, critical
    key_obligations: list[str]
    red_flags: list[str]
    # Token usage tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0  # Estimated cost


def bucket_confidence(raw: float) -> tuple[str, float]:
    """Convert raw confidence to a consistent bucketed value.
    
    Returns:
        Tuple of (label, normalized_value) for consistent display.
    """
    if raw >= 0.90:
        return ("Very High", 0.95)
    elif raw >= 0.75:
        return ("High", 0.80)
    elif raw >= 0.60:
        return ("Medium", 0.70)
    elif raw >= 0.40:
        return ("Low", 0.50)
    else:
        return ("Very Low", 0.30)


# System prompt for the clause classifier worker
CLASSIFIER_SYSTEM_PROMPT = """You are a legal document analysis expert. Classify contract clauses accurately and confidently.

## Your Task
Given a clause from a legal contract:
1. Identify the clause type from the categories below
2. Assess risk level based on language
3. Extract key obligations
4. Flag problematic terms

## Clause Categories
- indemnification: Hold harmless, indemnify, defend against claims
- termination: End of agreement, cancellation rights, early termination
- confidentiality: NDA, proprietary information, trade secrets
- limitation_of_liability: Liability caps, exclusion of damages
- governing_law: Jurisdiction, choice of law, venue
- dispute_resolution: Arbitration, mediation, litigation procedures
- force_majeure: Acts of God, unforeseeable circumstances
- assignment: Transfer of rights, delegation, subcontracting restrictions
- amendment: Modification procedures, changes require written consent
- waiver: Waiver of rights, no waiver clauses
- severability: Invalidity of provisions
- entire_agreement: Integration clause, supersedes prior agreements
- notice: Notification requirements, addresses, delivery methods
- intellectual_property: IP ownership, licensing, work product
- warranty: Representations, guarantees, disclaimers
- insurance: Coverage requirements, policy limits
- compliance: Regulatory requirements, legal obligations
- audit_rights: Inspection, records access, financial review
- payment: Compensation, fees, invoicing, reimbursement
- scope_of_work: Services, deliverables, specifications
- term: Duration, renewal, extension
- definitions: Defined terms
- other: Only if no category fits

## Confidence Scoring Guide
- 0.90-1.00: Clear match with explicit keywords (e.g., "shall indemnify" → indemnification)
- 0.80-0.89: Strong match with typical clause language
- 0.70-0.79: Good match but some ambiguity or mixed content
- 0.60-0.69: Reasonable classification, could fit multiple categories
- Below 0.60: Unclear content or just a header/title

## Risk Levels
- low: Standard, balanced terms
- medium: Some one-sided terms but industry-standard
- high: Significantly one-sided, unusual provisions
- critical: Extremely one-sided, potentially harmful

## Output Format
Respond with valid JSON only:
{"clause_type": "category", "confidence": 0.85, "reasoning": "Brief explanation", "risk_level": "low|medium|high|critical", "key_obligations": ["obligation1"], "red_flags": ["flag1"]}

## Examples

### Example 1: High Confidence Classification
Input: "INDEMNIFICATION. Contractor shall defend, indemnify, and hold harmless the Company from any claims arising from Contractor's negligence."
Output:
{"clause_type": "indemnification", "confidence": 0.95, "reasoning": "Contains explicit 'defend, indemnify, and hold harmless' language - classic indemnification clause.", "risk_level": "medium", "key_obligations": ["Contractor must defend Company from claims", "Contractor must indemnify Company for negligence-related claims"], "red_flags": []}

### Example 2: High Confidence with Risk
Input: "TERMINATION FOR CONVENIENCE. Company may terminate this Agreement at any time for any reason upon 30 days written notice."
Output:
{"clause_type": "termination", "confidence": 0.92, "reasoning": "Explicit termination for convenience provision with notice period.", "risk_level": "high", "key_obligations": ["Company may terminate at any time", "30 days written notice required"], "red_flags": ["One-sided termination right favoring Company", "No termination right for Contractor"]}

### Example 3: Title Only (Lower Confidence)
Input: "COMPENSATION"
Output:
{"clause_type": "payment", "confidence": 0.70, "reasoning": "Title indicates payment/compensation section but no substantive text to analyze.", "risk_level": "low", "key_obligations": [], "red_flags": ["Incomplete clause - only title provided"]}"""


class AIClauseClassifier:
    """AI-powered clause classifier using GPT-5 Mini."""
    
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
    
    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response with multiple repair strategies.
        
        Handles common LLM JSON generation issues:
        1. Markdown code blocks
        2. Unescaped quotes in strings
        3. Missing/extra commas
        4. Truncated responses
        """
        import re
        
        # Strategy 1: Extract from markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        text = text.strip()
        
        # Strategy 2: Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Extract JSON object with regex (handles surrounding text)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                text = json_match.group()  # Continue with extracted JSON
        
        # Strategy 4: Repair common issues
        repaired = text
        
        # Fix: Remove newlines within string values (common LLM issue)
        repaired = re.sub(r'(?<=["\'])\s*\n\s*(?=[^"\']*["\'])', ' ', repaired)
        
        # Fix: Escape unescaped quotes inside strings
        # Find string values and escape internal quotes
        def escape_inner_quotes(match):
            content = match.group(1)
            # Escape quotes that aren't already escaped
            escaped = re.sub(r'(?<!\\)"', '\\"', content)
            return f'"{escaped}"'
        
        # Match string values (content between quotes)
        repaired = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_inner_quotes, repaired)
        
        # Fix: Remove trailing commas before closing brackets
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        
        # Fix: Add missing commas between array elements
        repaired = re.sub(r'"\s*"', '", "', repaired)
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Build minimal valid response from regex extraction
        result = {
            "clause_type": "unknown",
            "confidence": 0.5,
            "reasoning": "JSON parsing required fallback extraction",
            "risk_level": "medium",
            "key_obligations": [],
            "red_flags": []
        }
        
        # Try to extract individual fields
        type_match = re.search(r'"clause_type"\s*:\s*"([^"]+)"', text)
        if type_match:
            result["clause_type"] = type_match.group(1)
        
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
        if conf_match:
            result["confidence"] = float(conf_match.group(1))
        
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        if reason_match:
            result["reasoning"] = reason_match.group(1)
        
        risk_match = re.search(r'"risk_level"\s*:\s*"([^"]+)"', text)
        if risk_match:
            result["risk_level"] = risk_match.group(1)
        
        # Extract obligations array
        oblig_match = re.search(r'"key_obligations"\s*:\s*\[([^\]]*)\]', text)
        if oblig_match:
            items = re.findall(r'"([^"]+)"', oblig_match.group(1))
            result["key_obligations"] = items
        
        # Extract red_flags array
        flags_match = re.search(r'"red_flags"\s*:\s*\[([^\]]*)\]', text)
        if flags_match:
            items = re.findall(r'"([^"]+)"', flags_match.group(1))
            result["red_flags"] = items
        
        return result
    
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
        
        # MAKER framework: retry loop with reliability filtering
        MAX_RETRIES = 3
        logger = logging.getLogger(__name__)
        
        last_discard_reason = ""
        total_input_tokens = 0
        total_output_tokens = 0
        
        for attempt in range(MAX_RETRIES):
            try:
                # Vary seed on retries to get different responses
                response = self.client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=500,
                    seed=42 + attempt,  # Vary seed on retries
                    # GPT-5 Mini specific parameters
                    extra_body={
                        "reasoning_effort": "low",
                        "verbosity": "low"
                    }
                )
                
                result_text = response.choices[0].message.content or "{}"
                
                # Extract token usage
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                
                # Track cumulative tokens across retries
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # MAKER reliability check - discard unreliable responses
                reliability_check = reliability_filter.check(result_text, output_tokens)
                
                if not reliability_check.is_reliable:
                    last_discard_reason = reliability_check.reason
                    logger.warning(
                        f"🔴 Attempt {attempt + 1}/{MAX_RETRIES} DISCARDED: {reliability_check.reason}"
                    )
                    continue  # Retry with different seed
                
                # Log successful reliability check
                logger.info(
                    f"🟢 Reliability check PASSED (attempt {attempt + 1}, {output_tokens} tokens)"
                )
                
                # Response is reliable - parse and return
                result = self._parse_json_response(result_text)
                
                # Calculate cost (GPT-5 Mini pricing: $0.25/M input, $2/M output)
                cost_usd = (total_input_tokens * 0.25 / 1_000_000) + (total_output_tokens * 2 / 1_000_000)
                
                # Bucket confidence for consistent display
                raw_confidence = float(result.get("confidence", 0.5))
                confidence_label, bucketed_confidence = bucket_confidence(raw_confidence)
                
                return ClassificationResult(
                    clause_type=result.get("clause_type", "unknown"),
                    confidence=bucketed_confidence,
                    confidence_label=confidence_label,
                    reasoning=result.get("reasoning", ""),
                    risk_level=result.get("risk_level", "medium"),
                    key_obligations=result.get("key_obligations", []),
                    red_flags=result.get("red_flags", []),
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    total_tokens=total_input_tokens + total_output_tokens,
                    cost_usd=cost_usd,
                )
                
            except Exception as e:
                last_discard_reason = f"API error: {str(e)}"
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                continue
        
        # All retries exhausted - return low-confidence fallback
        logger.error(f"All {MAX_RETRIES} attempts failed. Last reason: {last_discard_reason}")
        return ClassificationResult(
            clause_type="unknown",
            confidence=0.30,
            confidence_label="Very Low",
            reasoning=f"Classification unreliable after {MAX_RETRIES} attempts: {last_discard_reason}",
            risk_level="medium",
            key_obligations=[],
            red_flags=["Classification failed reliability checks"],
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            cost_usd=(total_input_tokens * 0.25 / 1_000_000) + (total_output_tokens * 2 / 1_000_000),
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


# -----------------------------------------------------------------------------
# BaseWorker-compatible wrapper for the voting framework
# -----------------------------------------------------------------------------

from .base_worker import BaseWorker, WorkerContext, WorkerResult
from .worker_registry import WorkerRegistry


@WorkerRegistry.register
class ClauseClassifierWorker(BaseWorker):
    """Stateless worker for clause classification.
    
    Wraps AIClauseClassifier in the BaseWorker interface
    for use with the voting engine.
    """
    
    TASK_TYPE = "classify_clause"
    DESCRIPTION = "Classify contract clauses into types with risk analysis"
    
    def get_system_prompt(self) -> str:
        """Return the system prompt for classification."""
        return CLASSIFIER_SYSTEM_PROMPT
    
    def execute(self, context: WorkerContext) -> WorkerResult:
        """Execute clause classification.
        
        Args:
            context: WorkerContext with clause text and optional title.
            
        Returns:
            WorkerResult with classification data.
        """
        # Use the existing classifier logic
        result = ai_classifier.classify(
            clause_text=context.text,
            title=context.title
        )
        
        return WorkerResult(
            success=result.clause_type != "unknown",
            data={
                "clause_type": result.clause_type,
                "risk_level": result.risk_level,
                "key_obligations": result.key_obligations,
                "red_flags": result.red_flags,
                "reasoning": result.reasoning,
            },
            confidence=result.confidence,
            confidence_label=result.confidence_label,
            reasoning=result.reasoning,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        )
