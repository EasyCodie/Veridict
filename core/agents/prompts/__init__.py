"""Prompt templates for Micro-Agent Factory agents."""

from core.agents.prompts.obligation_extractor_prompt import (
    OBLIGATION_EXTRACTOR_SYSTEM_PROMPT,
    OBLIGATION_EXTRACTOR_USER_PROMPT_TEMPLATE,
    format_obligation_extractor_prompt,
)

__all__ = [
    "OBLIGATION_EXTRACTOR_SYSTEM_PROMPT",
    "OBLIGATION_EXTRACTOR_USER_PROMPT_TEMPLATE",
    "format_obligation_extractor_prompt",
]
