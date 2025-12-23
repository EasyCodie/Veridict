"""Prompt Templates for Risk Identifier Agent.

This module contains the system and user prompt templates for identifying
risks, red flags, and unusual provisions in contract clauses.
"""

RISK_IDENTIFIER_SYSTEM_PROMPT = """You are a legal risk identification expert specializing in contract analysis.

Your role is to identify potential risks, red flags, and unusual provisions in contract clauses.
You must flag:
- UNUSUAL LANGUAGE: Non-standard or archaic legal terminology
- AMBIGUOUS DEFINITIONS: Unclear terms or vague provisions
- CONTRADICTS STANDARD: Deviates from market-standard language
- MISSING SAFEGUARDS: Expected protective provisions are absent
- UNLIMITED EXPOSURE: Unlimited liability or obligations without caps
- UNDEFINED TERMS: Key terms used without definition

Key principles:
- Identify risks ONLY from this specific clause text
- Do NOT assume risks from context outside the clause
- Provide a supporting quote for EVERY risk identified
- Include actionable remediation suggestions
- If no risks exist, return empty risks array with "clean" rating
- Be conservative: flag potential issues, even if uncertain (mark with lower confidence)"""


RISK_IDENTIFIER_USER_PROMPT_TEMPLATE = """CLAUSE TEXT:
{clause_text}

CONTRACT METADATA:
- Type: {contract_type}
- Clause Type: {clause_type}
- Parties: {parties}
- Jurisdiction: {jurisdiction}

TASK:
Identify ALL potential risks in this clause ONLY.

RULES:
1. Analyze only this clause. Do NOT assume risks from other sections.
2. For EACH risk, identify:
   - RISK TYPE: One of (unusual_language, ambiguous_definition, contradicts_standard, missing_safeguard, unlimited_exposure, undefined_term)
   - RISK DESCRIPTION: Clear explanation of the risk
   - SEVERITY: One of (critical, high, medium, low)
   - REMEDIATION SUGGESTION: Specific action to address the risk
   - CONFIDENCE: Number between 0 and 1
   - SUPPORTING QUOTE: Exact text from the clause that evidences the risk
3. Severity guidelines:
   - CRITICAL: Immediate attention required. Significant financial/legal exposure.
   - HIGH: Significant risk that should be addressed before signing.
   - MEDIUM: Notable concern that warrants review.
   - LOW: Minor issue with limited impact.
4. If NO risks exist in this clause, return empty risks array with overall_risk_rating: "clean".
5. Be thorough but avoid false positives - only flag genuine concerns.

OUTPUT FORMAT (JSON ONLY, NO EXPLANATIONS):
{{
  "clause_id": "{clause_id}",
  "risks_identified": [
    {{
      "risk_type": "unusual_language | ambiguous_definition | contradicts_standard | missing_safeguard | unlimited_exposure | undefined_term",
      "risk_description": "string explaining the risk",
      "severity": "critical | high | medium | low",
      "remediation_suggestion": "specific action to address risk",
      "confidence": number between 0 and 1,
      "supporting_quote": "exact text from clause"
    }}
  ],
  "overall_risk_rating": "clean | minor_issues | significant_issues | critical_issues"
}}"""


# Example clauses for testing and few-shot prompting
EXAMPLE_CLAUSE_CLEAN = """
Section 12.1 Notices. All notices under this Agreement shall be in writing 
and shall be deemed delivered when: (a) delivered personally; (b) sent by 
confirmed facsimile; (c) sent by commercial overnight courier with tracking 
capability; or (d) three (3) business days after mailing by certified mail, 
return receipt requested.
"""

EXAMPLE_CLAUSE_HIGH_RISK = """
Section 8.1 Indemnification. The Seller shall indemnify, defend, and hold 
harmless the Buyer and its affiliates, officers, directors, employees, 
agents, successors, and assigns from and against any and all claims, damages, 
losses, costs, and expenses (including reasonable attorneys' fees) arising 
out of or relating to this Agreement, without limitation as to amount or time, 
including claims arising from the Seller's own negligence.
"""

EXAMPLE_CLAUSE_AMBIGUOUS = """
Section 5.3 Material Adverse Effect. The Seller represents that there has 
been no Material Adverse Effect since the Balance Sheet Date. For purposes 
of this Agreement, "Material Adverse Effect" shall mean any change, effect, 
event, or circumstance that is, or would reasonably be expected to be, 
material to the Business.
"""

EXAMPLE_CLAUSE_MULTI_RISK = """
Section 9.2 Termination Rights. Either Party may terminate this Agreement 
immediately upon written notice if the other Party fails to perform its 
obligations, without any cure period or opportunity to remedy. Upon 
termination, the defaulting Party shall be liable for all consequential, 
punitive, and exemplary damages, and shall forfeit any rights to intellectual 
property developed during the term.
"""


def format_risk_identifier_prompt(
    clause_id: str,
    clause_text: str,
    clause_type: str = "",
    contract_type: str = "",
    parties: list[str] | None = None,
    jurisdiction: str = "",
) -> str:
    """Format the user prompt for the Risk Identifier Agent.
    
    Args:
        clause_id: Unique identifier for the clause.
        clause_text: Full text of the clause to analyze.
        clause_type: Type of clause (e.g., 'Indemnity', 'Termination').
        contract_type: Type of contract (e.g., 'SPA', 'NDA').
        parties: List of parties to the contract.
        jurisdiction: Governing jurisdiction.
        
    Returns:
        Formatted prompt string.
    """
    parties_str = ", ".join(parties) if parties else "Not specified"
    
    return RISK_IDENTIFIER_USER_PROMPT_TEMPLATE.format(
        clause_id=clause_id,
        clause_text=clause_text,
        clause_type=clause_type or "Not specified",
        contract_type=contract_type or "Not specified",
        parties=parties_str,
        jurisdiction=jurisdiction or "Not specified",
    )
