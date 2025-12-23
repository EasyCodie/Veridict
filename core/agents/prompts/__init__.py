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

__all__ = [
    "OBLIGATION_EXTRACTOR_SYSTEM_PROMPT",
    "OBLIGATION_EXTRACTOR_USER_PROMPT_TEMPLATE",
    "format_obligation_extractor_prompt",
    "CAP_LIMITATION_SYSTEM_PROMPT",
    "CAP_LIMITATION_USER_PROMPT_TEMPLATE",
    "format_cap_limitation_prompt",
]
