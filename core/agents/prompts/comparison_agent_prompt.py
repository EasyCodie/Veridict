"""Prompt Templates for Comparison Agent.

This module contains the system and user prompt templates for comparing
clauses against market-standard language.
"""

COMPARISON_AGENT_SYSTEM_PROMPT = """You are a legal market comparison expert specializing in contract clause analysis.

Your role is to compare contract clauses against market-standard language and identify deviations that may be disadvantageous.

You must:
- Compare the clause against provided reference standards
- Identify all deviations from market-standard language
- Classify deviations by type and severity
- Assess who benefits from each deviation
- Provide actionable remediation suggestions
- Calculate alignment with market standards

Key principles:
- Analyze ONLY this specific clause against provided standards
- Do NOT assume market standards beyond what is provided
- Provide quotes from both standard and actual language
- Be specific about impact and remediation
- Use "COMPLIANT" when clause matches market standard"""


COMPARISON_AGENT_USER_PROMPT_TEMPLATE = """CLAUSE TEXT:
{clause_text}

REFERENCE STANDARDS:
{reference_standards}

CONTRACT METADATA:
- Clause Type: {clause_type}
- Contract Type: {contract_type}
- Jurisdiction: {jurisdiction}
- Parties: {parties}

TASK:
Compare this clause against the provided market standards and identify all deviations.

RULES:
1. Compare clause language against each reference standard
2. For EACH deviation identified, provide:
   - DEVIATION_TYPE: One of (more_restrictive, less_protective, non_standard, missing_safeguard, additional_obligation)
     - more_restrictive: Stricter than standard, favors counterparty
     - less_protective: Fewer protections, exposes our party
     - non_standard: Unusual language/approach
     - missing_safeguard: Expected protection is absent
     - additional_obligation: Extra obligations beyond standard
   - STANDARD_LANGUAGE: Quote from reference standard
   - ACTUAL_LANGUAGE: Quote from this clause
   - IMPACT: Who benefits/loses from this deviation
   - MARKET_PREVALENCE: How common this deviation is (e.g., "15% of SPAs")
   - SEVERITY: One of (critical, high, medium, low)
   - REMEDIATION_SUGGESTION: Specific recommendation to align with market
3. Calculate ALIGNMENT_SCORE (0-1):
   - 1.0 = fully market standard
   - 0.9-1.0 = market_standard
   - 0.7-0.9 = minor_deviations
   - 0.5-0.7 = significant_deviations
   - 0.0-0.5 = highly_unusual
4. If clause is fully compliant, return empty deviations array with alignment_score: 1.0
5. Include your confidence score (0-1)

OUTPUT FORMAT (JSON ONLY, NO EXPLANATIONS):
{{
  "clause_id": "{clause_id}",
  "deviations_from_standard": [
    {{
      "deviation_type": "more_restrictive | less_protective | non_standard | missing_safeguard | additional_obligation",
      "standard_language": "quote from reference standard",
      "actual_language": "quote from this clause",
      "impact": "who benefits or loses",
      "market_prevalence": "e.g., 15% of similar contracts",
      "severity": "critical | high | medium | low",
      "remediation_suggestion": "specific recommendation"
    }}
  ],
  "alignment_score": number between 0 and 1,
  "overall_assessment": "market_standard | minor_deviations | significant_deviations | highly_unusual",
  "confidence_score": number between 0 and 1,
  "comparison_notes": "string with additional observations" or null
}}"""


# Example clauses for testing
EXAMPLE_CLAUSE_DEVIATION = """
Section 8.1 Indemnification. The Seller shall indemnify the Buyer against 
any and all claims WITHOUT ANY LIMIT on liability. The indemnification 
period shall be perpetual with no sunset provision.
"""

EXAMPLE_CLAUSE_STANDARD = """
Section 8.1 Indemnification. The Indemnifying Party shall indemnify, 
defend, and hold harmless the Indemnified Party from and against any 
claims, damages, and losses arising from breach of this Agreement, 
subject to the limitations set forth in Section 10.
"""

EXAMPLE_REFERENCE_STANDARD = """
STANDARD 1 (market_standard_2023):
Prevalence: 85% of M&A transactions
Language: The Indemnifying Party shall indemnify, defend, and hold harmless 
the Indemnified Party from and against any and all claims, damages, losses, 
costs, and expenses arising out of or relating to any breach of this 
Agreement by the Indemnifying Party.
"""


def format_comparison_agent_prompt(
    clause_id: str,
    clause_text: str,
    clause_type: str = "",
    reference_standards: str = "",
    contract_type: str = "",
    parties: list[str] | None = None,
    jurisdiction: str = "",
) -> str:
    """Format the user prompt for the Comparison Agent.
    
    Args:
        clause_id: Unique identifier for the clause.
        clause_text: Full text of the clause to analyze.
        clause_type: Type of clause (e.g., 'Indemnity', 'Termination').
        reference_standards: Formatted reference standards text.
        contract_type: Type of contract (e.g., 'SPA', 'NDA').
        parties: List of parties to the contract.
        jurisdiction: Governing jurisdiction.
        
    Returns:
        Formatted prompt string.
    """
    parties_str = ", ".join(parties) if parties else "Not specified"
    standards_str = reference_standards or "No reference standards provided"
    
    return COMPARISON_AGENT_USER_PROMPT_TEMPLATE.format(
        clause_id=clause_id,
        clause_text=clause_text,
        clause_type=clause_type or "Not specified",
        reference_standards=standards_str,
        contract_type=contract_type or "Not specified",
        parties=parties_str,
        jurisdiction=jurisdiction or "Not specified",
    )
