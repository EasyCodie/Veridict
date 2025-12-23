"""Prompt templates for the Obligation Extractor Agent.

This module contains the hyper-specific prompts for extracting obligations
from contract clauses. The prompts follow the MAKER Framework guidelines
for stateless, focused micro-task execution.
"""

OBLIGATION_EXTRACTOR_SYSTEM_PROMPT = """You are a legal obligation extractor specializing in contract analysis.

Your role is to extract all obligations from a single contract clause.
You must identify WHO must do WHAT, WHEN, and what happens if they don't.

Key principles:
- Extract only what is explicitly stated in the clause
- Do not infer obligations from context outside the clause
- Mark unclear obligations with confidence < 0.7
- Always provide structured JSON output
- Back findings with exact text from the clause"""


OBLIGATION_EXTRACTOR_USER_PROMPT_TEMPLATE = """You are a legal obligation extractor analyzing a single contract clause.

CLAUSE TEXT:
{clause_text}

CONTRACT METADATA:
- Type: {contract_type}
- Parties: {parties}
- Jurisdiction: {jurisdiction}

TASK:
Extract all obligations (who must do what, under what conditions) from this clause ONLY.

RULES:
1. Extract only what is explicitly stated in this clause.
2. Do NOT infer obligations from other sections.
3. For each obligation, identify:
   - WHO must act (the obligor)
   - WHAT they must do (obligation description)
   - WHEN/IF it applies (trigger condition and deadline)
   - WHAT happens if they don't (consequences)
4. If an obligation is unclear, mark confidence < 0.7.
5. If no obligations exist, return empty obligations array with "no_obligations_found": true.

OUTPUT FORMAT (JSON ONLY, NO EXPLANATIONS):
{{
  "clause_id": "{clause_id}",
  "obligations": [
    {{
      "obligor": "string",
      "obligation_description": "string",
      "trigger_condition": "string",
      "deadline": "string or null",
      "consequences_of_breach": "string or null",
      "confidence": number between 0 and 1
    }}
  ],
  "no_obligations_found": boolean
}}"""


def format_obligation_extractor_prompt(
    clause_id: str,
    clause_text: str,
    contract_type: str,
    parties: list[str],
    jurisdiction: str,
) -> str:
    """Format the obligation extractor user prompt with clause data.

    Args:
        clause_id: Unique identifier for the clause.
        clause_text: Full text of the clause to analyze.
        contract_type: Type of contract (e.g., "SPA", "NDA").
        parties: List of party names in the contract.
        jurisdiction: Governing jurisdiction.

    Returns:
        Formatted prompt string ready for LLM.
    """
    parties_str = ", ".join(parties) if parties else "Not specified"

    return OBLIGATION_EXTRACTOR_USER_PROMPT_TEMPLATE.format(
        clause_id=clause_id,
        clause_text=clause_text,
        contract_type=contract_type or "Not specified",
        parties=parties_str,
        jurisdiction=jurisdiction or "Not specified",
    )


# Example prompts for testing and documentation
EXAMPLE_CLAUSE_INDEMNIFICATION = """
5.1 Indemnification by Seller. The Seller shall indemnify, defend, and hold harmless
the Buyer and its affiliates, directors, officers, and employees from and against any
and all losses, damages, claims, and expenses (including reasonable attorney's fees)
arising out of or relating to any breach of this Agreement by the Seller or any
misrepresentation made by the Seller in connection herewith. Such indemnification
shall be made within thirty (30) days of receipt of a written claim from Buyer.
Failure to provide timely indemnification shall result in the accrual of interest
at the rate of 1.5% per month on the outstanding amount.
"""

EXAMPLE_CLAUSE_PAYMENT = """
3.2 Payment Terms. The Buyer shall pay the Purchase Price to the Seller in three
installments: (a) 30% upon execution of this Agreement; (b) 40% upon completion
of the due diligence review; and (c) the remaining 30% at Closing. All payments
shall be made by wire transfer to the account designated by the Seller.
Late payments shall bear interest at the prime rate plus 2% per annum.
"""

EXAMPLE_CLAUSE_NO_OBLIGATIONS = """
1.1 Definitions. For purposes of this Agreement, the following terms shall have
the meanings set forth below:
"Affiliate" means any entity that directly or indirectly controls, is controlled by,
or is under common control with such party.
"Business Day" means any day that is not a Saturday, Sunday, or legal holiday
in the State of Delaware.
"""
