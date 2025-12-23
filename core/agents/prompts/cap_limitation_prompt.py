"""Prompt Templates for Cap & Limitation Agent.

This module contains the system and user prompt templates for extracting
caps and limitations from contract clauses.
"""

CAP_LIMITATION_SYSTEM_PROMPT = """You are a legal cap & limitation extractor specializing in contract analysis.

Your role is to extract all financial caps, liability limitations, and exceptions from a single contract clause.
You must identify:
- CAP TYPE (liability_cap, indemnity_cap, damages_limitation, deductible)
- CAP AMOUNT (e.g., "$5M" or "X% of Purchase Price")
- APPLIES TO (what losses/claims does this cap apply to)
- EXCEPTIONS (carve-outs from the cap)

Key principles:
- Extract only what is explicitly stated in the clause
- Do not infer caps from context outside the clause
- Mark unclear caps with confidence < 0.7
- Preserve relative amounts as strings (e.g., "15% of Purchase Price")
- Always identify exceptions/carve-outs when present
- If no caps found, set no_cap_found: true"""


CAP_LIMITATION_USER_PROMPT_TEMPLATE = """CLAUSE TEXT:
{clause_text}

CONTRACT METADATA:
- Type: {contract_type}
- Parties: {parties}
- Jurisdiction: {jurisdiction}

TASK:
Extract all caps and limitations from this clause ONLY.

RULES:
1. Extract only what is explicitly stated in this clause.
2. Do NOT infer caps from other sections.
3. For each cap, identify:
   - CAP TYPE: One of (liability_cap, indemnity_cap, damages_limitation, deductible)
   - CAP AMOUNT: The monetary limit (e.g., "$5M") or relative amount (e.g., "X% of Purchase Price")
   - CAP AMOUNT NUMERIC: If amount is a specific number in USD, provide it. Otherwise null.
   - APPLIES TO: What losses, damages, or claims does this cap apply to
   - EXCEPTIONS: List of carve-outs that are explicitly excluded from the cap
4. If cap amount is relative (e.g., "15% of Purchase Price"), preserve as string in cap_amount.
5. If cap language is ambiguous or unclear, mark confidence < 0.7.
6. If no caps exist in this clause, return empty caps array with "no_cap_found": true.

OUTPUT FORMAT (JSON ONLY, NO EXPLANATIONS):
{{
  "clause_id": "{clause_id}",
  "caps": [
    {{
      "cap_type": "liability_cap | indemnity_cap | damages_limitation | deductible",
      "cap_amount": "string (e.g., '$5M' or '15% of Purchase Price')",
      "cap_amount_numeric": number or null,
      "applies_to": "string describing what the cap applies to",
      "exceptions": ["string list of carve-outs from cap"],
      "confidence": number between 0 and 1
    }}
  ],
  "no_cap_found": boolean
}}"""


# Example clauses for testing and few-shot prompting
EXAMPLE_CLAUSE_LIABILITY_CAP = """
Section 8.1 Limitation of Liability. Notwithstanding anything to the contrary 
in this Agreement, the aggregate liability of the Seller to the Buyer under 
this Agreement shall not exceed Five Million Dollars ($5,000,000), except 
for claims arising from fraud, willful misconduct, or breach of the 
Seller's fundamental representations in Section 3.1.
"""

EXAMPLE_CLAUSE_INDEMNITY_CAP = """
Section 7.3 Indemnification Cap. The Indemnifying Party's maximum liability 
under this Section 7 shall be limited to an amount equal to fifteen percent 
(15%) of the Purchase Price, provided that such cap shall not apply to 
claims for third-party intellectual property infringement or gross negligence.
"""

EXAMPLE_CLAUSE_NO_CAP = """
Section 5.2 Definitions. "Business Day" means any day other than a Saturday, 
Sunday, or a day on which banking institutions in New York are authorized 
or required by law to close.
"""

EXAMPLE_CLAUSE_DEDUCTIBLE = """
Section 9.2 Basket and Deductible. The Buyer shall not be entitled to 
indemnification under Section 9.1 unless and until the aggregate amount 
of Losses exceeds Two Hundred Fifty Thousand Dollars ($250,000) (the "Basket"), 
in which case the Buyer shall be entitled to recover all Losses from the 
first dollar (not just amounts in excess of the Basket).
"""


def format_cap_limitation_prompt(
    clause_id: str,
    clause_text: str,
    contract_type: str = "",
    parties: list[str] | None = None,
    jurisdiction: str = "",
) -> str:
    """Format the user prompt for the Cap & Limitation Agent.
    
    Args:
        clause_id: Unique identifier for the clause.
        clause_text: Full text of the clause to analyze.
        contract_type: Type of contract (e.g., 'SPA', 'NDA').
        parties: List of parties to the contract.
        jurisdiction: Governing jurisdiction.
        
    Returns:
        Formatted prompt string.
    """
    parties_str = ", ".join(parties) if parties else "Not specified"
    
    return CAP_LIMITATION_USER_PROMPT_TEMPLATE.format(
        clause_id=clause_id,
        clause_text=clause_text,
        contract_type=contract_type or "Not specified",
        parties=parties_str,
        jurisdiction=jurisdiction or "Not specified",
    )
