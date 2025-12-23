"""Prompt templates for Micro-Agent Factory agents."""

from core.agents.prompts.obligation_extractor_prompt import (
    OBLIGATION_EXTRACTOR_SYSTEM_PROMPT,
    OBLIGATION_EXTRACTOR_USER_PROMPT_TEMPLATE,
    format_obligation_extractor_prompt,
)
from core.agents.prompts.cap_limitation_prompt import (
    CAP_LIMITATION_SYSTEM_PROMPT,
    CAP_LIMITATION_USER_PROMPT_TEMPLATE,
    format_cap_limitation_prompt,
)
from core.agents.prompts.risk_identifier_prompt import (
    RISK_IDENTIFIER_SYSTEM_PROMPT,
    RISK_IDENTIFIER_USER_PROMPT_TEMPLATE,
    format_risk_identifier_prompt,
)
from core.agents.prompts.clause_analysis_prompt import (
    CLAUSE_ANALYSIS_SYSTEM_PROMPT,
    CLAUSE_ANALYSIS_USER_PROMPT_TEMPLATE,
    format_clause_analysis_prompt,
)
from core.agents.prompts.definition_validator_prompt import (
    DEFINITION_VALIDATOR_SYSTEM_PROMPT,
    DEFINITION_VALIDATOR_USER_PROMPT_TEMPLATE,
    format_definition_validator_prompt,
)
from core.agents.prompts.comparison_agent_prompt import (
    COMPARISON_AGENT_SYSTEM_PROMPT,
    COMPARISON_AGENT_USER_PROMPT_TEMPLATE,
    format_comparison_agent_prompt,
)

__all__ = [
    "OBLIGATION_EXTRACTOR_SYSTEM_PROMPT",
    "OBLIGATION_EXTRACTOR_USER_PROMPT_TEMPLATE",
    "format_obligation_extractor_prompt",
    "CAP_LIMITATION_SYSTEM_PROMPT",
    "CAP_LIMITATION_USER_PROMPT_TEMPLATE",
    "format_cap_limitation_prompt",
    "RISK_IDENTIFIER_SYSTEM_PROMPT",
    "RISK_IDENTIFIER_USER_PROMPT_TEMPLATE",
    "format_risk_identifier_prompt",
    "CLAUSE_ANALYSIS_SYSTEM_PROMPT",
    "CLAUSE_ANALYSIS_USER_PROMPT_TEMPLATE",
    "format_clause_analysis_prompt",
    "DEFINITION_VALIDATOR_SYSTEM_PROMPT",
    "DEFINITION_VALIDATOR_USER_PROMPT_TEMPLATE",
    "format_definition_validator_prompt",
    "COMPARISON_AGENT_SYSTEM_PROMPT",
    "COMPARISON_AGENT_USER_PROMPT_TEMPLATE",
    "format_comparison_agent_prompt",
]

