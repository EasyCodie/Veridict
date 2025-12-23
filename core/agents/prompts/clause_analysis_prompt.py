"""Prompt Templates for Clause Analysis Agent.

This module contains the system and user prompt templates for extracting
key findings from contract clauses.
"""

CLAUSE_ANALYSIS_SYSTEM_PROMPT = """You are a legal clause analysis expert specializing in extracting key findings from contract clauses.

Your role is to analyze individual clauses and extract structured findings including:
- OBLIGATIONS: Who must do what and under what conditions
- CAPS: Financial limits or liability caps
- EXCEPTIONS: Carve-outs or exclusions from main provisions
- DEFINITIONS: Key terms defined or referenced
- TRIGGERS: Events or conditions that activate provisions
- DURATIONS: Time periods, effective dates, deadlines
- PROCESSES: Procedural steps or requirements

Key principles:
- Analyze ONLY this specific clause text
- Do NOT assume findings from context outside the clause
- Provide a supporting quote for EVERY finding
- Rate importance and confidence for each finding
- Generate a concise 1-2 sentence summary of clause effect
- Set clarification_required if clause is ambiguous
- Use "informational" risk level for straightforward clauses"""


CLAUSE_ANALYSIS_USER_PROMPT_TEMPLATE = """CLAUSE TEXT:
{clause_text}

CONTRACT METADATA:
- Clause Type: {clause_type}
- Contract Type: {contract_type}
- Parties: {parties}
- Jurisdiction: {jurisdiction}

TASK INSTRUCTIONS:
{task_instructions}

RULES:
1. Analyze only this clause. Do NOT assume findings from other sections.
2. For EACH finding, identify:
   - FINDING TYPE: One of (obligation, cap, exception, definition, trigger, duration, process)
   - FINDING TEXT: Direct quote or paraphrase of the finding
   - IMPORTANCE: One of (critical, high, medium, low)
   - CONFIDENCE: Number between 0 and 1
   - SUPPORTING QUOTE: Exact text from the clause
3. Importance guidelines:
   - CRITICAL: Must be addressed before signing. Material impact on deal.
   - HIGH: Significant finding that warrants attention.
   - MEDIUM: Notable finding worth reviewing.
   - LOW: Minor finding with limited impact.
4. Provide a 1-2 sentence SUMMARY of the overall clause effect.
5. Set RISK_LEVEL based on findings:
   - critical: Major exposure or material issues found
   - high: Significant concerns identified
   - medium: Notable issues worth reviewing
   - low: Minor findings only
   - informational: Straightforward clause, no concerns
6. Set CLARIFICATION_REQUIRED = true if:
   - Clause is ambiguous or unclear
   - Multiple interpretations are possible
   - Technical terms are not adequately defined
   If clarification is needed, explain in CLARIFICATION_NOTE.
7. If no findings exist, return empty findings array.

OUTPUT FORMAT (JSON ONLY, NO EXPLANATIONS):
{{
  "clause_id": "{clause_id}",
  "findings": [
    {{
      "finding_type": "obligation | cap | exception | definition | trigger | duration | process",
      "finding_text": "string describing the finding",
      "importance": "critical | high | medium | low",
      "confidence": number between 0 and 1,
      "supporting_quote": "exact text from clause"
    }}
  ],
  "summary": "1-2 sentence summary of clause effect",
  "risk_level": "critical | high | medium | low | informational",
  "clarification_required": true | false,
  "clarification_note": "string explaining what needs review" or null
}}"""


# Example clauses for testing and few-shot prompting
EXAMPLE_CLAUSE_INDEMNITY = """
Section 8.1 Indemnification by Seller. The Seller shall indemnify, defend, 
and hold harmless the Buyer and its affiliates, officers, directors, employees, 
and agents from and against any and all claims, damages, losses, costs, and 
expenses (including reasonable attorneys' fees) arising out of or relating to 
(a) any breach of Seller's representations or warranties, (b) any breach of 
Seller's covenants, or (c) any third-party claims relating to the Products.
"""

EXAMPLE_CLAUSE_TERMINATION = """
Section 12.2 Termination for Convenience. Either Party may terminate this 
Agreement for any reason upon ninety (90) days' prior written notice to the 
other Party. Upon such termination, (a) all outstanding payments shall become 
immediately due, (b) each Party shall return or destroy all Confidential 
Information of the other Party, and (c) Sections 6, 8, 10, and 15 shall survive.
"""

EXAMPLE_CLAUSE_PAYMENT = """
Section 4.3 Payment Terms. All invoices are due and payable within thirty (30) 
days of the invoice date. Late payments shall accrue interest at the lesser of 
1.5% per month or the maximum rate permitted by applicable law. The Buyer may 
withhold payment of any amounts it disputes in good faith, provided it notifies 
the Seller in writing of the dispute within fifteen (15) days of invoice receipt.
"""

EXAMPLE_CLAUSE_CONFIDENTIALITY = """
Section 9.1 Confidential Information. "Confidential Information" means any 
non-public information disclosed by one Party to the other, whether orally, 
in writing, or by inspection, that is designated as confidential or that 
reasonably should be understood to be confidential. Confidential Information 
does not include information that: (a) is or becomes publicly available through 
no fault of the receiving Party; (b) was known to the receiving Party prior to 
disclosure; or (c) is independently developed by the receiving Party.
"""


def format_clause_analysis_prompt(
    clause_id: str,
    clause_text: str,
    clause_type: str = "",
    task_instructions: str = "",
    contract_type: str = "",
    parties: list[str] | None = None,
    jurisdiction: str = "",
) -> str:
    """Format the user prompt for the Clause Analysis Agent.
    
    Args:
        clause_id: Unique identifier for the clause.
        clause_text: Full text of the clause to analyze.
        clause_type: Type of clause (e.g., 'Indemnity', 'Termination').
        task_instructions: Specific instructions for analysis.
        contract_type: Type of contract (e.g., 'SPA', 'NDA').
        parties: List of parties to the contract.
        jurisdiction: Governing jurisdiction.
        
    Returns:
        Formatted prompt string.
    """
    parties_str = ", ".join(parties) if parties else "Not specified"
    
    default_instructions = "Extract all key findings including obligations, caps, exceptions, definitions, triggers, durations, and processes."
    
    return CLAUSE_ANALYSIS_USER_PROMPT_TEMPLATE.format(
        clause_id=clause_id,
        clause_text=clause_text,
        clause_type=clause_type or "Not specified",
        task_instructions=task_instructions or default_instructions,
        contract_type=contract_type or "Not specified",
        parties=parties_str,
        jurisdiction=jurisdiction or "Not specified",
    )
