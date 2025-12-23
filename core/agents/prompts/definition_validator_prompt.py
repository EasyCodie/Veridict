"""Prompt Templates for Definition Validator Agent.

This module contains the system and user prompt templates for identifying
undefined or ambiguously defined terms in contract clauses.
"""

DEFINITION_VALIDATOR_SYSTEM_PROMPT = """You are a legal definition validation expert specializing in contract term analysis.

Your role is to identify undefined or ambiguously defined terms in contract clauses.
You must:
- Identify capitalized terms that appear to need definitions
- Check against provided definitions section (if available)
- Flag terms that are undefined or ambiguously defined
- Rate severity based on importance to contract interpretation
- Suggest where definitions might be found

Key principles:
- Analyze ONLY this specific clause text
- Compare against provided definitions section (if available)
- Do NOT assume definitions from context outside what's provided
- Common legal terms (shall, may, herein, etc.) don't need definitions
- Capitalized multi-word terms usually need definitions
- Provide context for how each undefined term is used"""


DEFINITION_VALIDATOR_USER_PROMPT_TEMPLATE = """CLAUSE TEXT:
{clause_text}

DEFINITIONS SECTION (if available):
{definitions_section}

CONTRACT METADATA:
- Clause Type: {clause_type}
- Contract Type: {contract_type}
- Parties: {parties}
- Jurisdiction: {jurisdiction}

TASK:
Identify undefined or ambiguously defined terms in this clause ONLY.

RULES:
1. Identify terms that:
   - Are capitalized (suggesting defined term status)
   - Are NOT in the provided definitions section
   - Are NOT common legal terms (shall, may, herein, etc.)
2. For EACH undefined term, provide:
   - TERM: The exact term as used
   - CONTEXT_USAGE: How it's used in the clause (quote the relevant text)
   - SEVERITY: One of (critical, high, medium)
     - CRITICAL: Key term, missing definition is fatal risk
     - HIGH: Important operational term, creates ambiguity
     - MEDIUM: Secondary term, complicates interpretation
   - LIKELIHOOD_DEFINED_ELSEWHERE: Boolean - is it likely defined elsewhere?
   - SEARCH_SUGGESTIONS: Where to look for the definition
3. If ALL terms are properly defined, set all_terms_defined: true
4. Provide a brief summary of definition gaps (if any)
5. Include your confidence score (0-1)

OUTPUT FORMAT (JSON ONLY, NO EXPLANATIONS):
{{
  "clause_id": "{clause_id}",
  "undefined_terms": [
    {{
      "term": "string",
      "context_usage": "string (exact quote showing usage)",
      "severity": "critical | high | medium",
      "likelihood_defined_elsewhere": true | false,
      "search_suggestions": ["string", "string"]
    }}
  ],
  "all_terms_defined": true | false,
  "definition_gaps_summary": "string describing gaps" or null,
  "confidence_score": number between 0 and 1
}}"""


# Example clauses for testing
EXAMPLE_CLAUSE_WITH_UNDEFINED = """
Section 5.1 Material Adverse Effect. The Seller represents that since the 
Balance Sheet Date, there has been no Material Adverse Effect on the Business 
or the Target Company. For the avoidance of doubt, any Permitted Change shall 
not constitute a Material Adverse Effect.
"""

EXAMPLE_CLAUSE_ALL_DEFINED = """
Section 3.1 Definitions. For purposes of this Agreement:
(a) "Business Day" means any day other than a Saturday, Sunday, or legal 
holiday on which banks in New York are authorized to close.
(b) "Closing Date" means the date on which the Closing occurs.
"""

EXAMPLE_DEFINITIONS_SECTION = """
ARTICLE 1 - DEFINITIONS

"Agreement" means this Stock Purchase Agreement.
"Balance Sheet Date" means December 31, 2023.
"Business" means the business conducted by the Target Company.
"Closing" means the closing of the transactions contemplated hereby.
"Material Adverse Effect" means any change, effect, or circumstance that is 
materially adverse to the business, financial condition, or results of 
operations of the Target Company.
"Target Company" means Acme Corporation, a Delaware corporation.
"""


def format_definition_validator_prompt(
    clause_id: str,
    clause_text: str,
    clause_type: str = "",
    definitions_section: str | None = None,
    contract_type: str = "",
    parties: list[str] | None = None,
    jurisdiction: str = "",
) -> str:
    """Format the user prompt for the Definition Validator Agent.
    
    Args:
        clause_id: Unique identifier for the clause.
        clause_text: Full text of the clause to analyze.
        clause_type: Type of clause (e.g., 'Indemnity', 'Termination').
        definitions_section: Text of the definitions section (if available).
        contract_type: Type of contract (e.g., 'SPA', 'NDA').
        parties: List of parties to the contract.
        jurisdiction: Governing jurisdiction.
        
    Returns:
        Formatted prompt string.
    """
    parties_str = ", ".join(parties) if parties else "Not specified"
    definitions_str = definitions_section or "Not provided - assume terms may be defined elsewhere in contract"
    
    return DEFINITION_VALIDATOR_USER_PROMPT_TEMPLATE.format(
        clause_id=clause_id,
        clause_text=clause_text,
        clause_type=clause_type or "Not specified",
        definitions_section=definitions_str,
        contract_type=contract_type or "Not specified",
        parties=parties_str,
        jurisdiction=jurisdiction or "Not specified",
    )
