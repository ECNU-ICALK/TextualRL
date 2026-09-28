"""SearchQA adapter variant for structured causal context evolution."""
from __future__ import annotations

from skillopt.envs.searchqa.adapter import SearchQAAdapter


class SearchQAMCEAdapter(SearchQAAdapter):
    """Reuse SearchQA execution while loading searchqa_mce analyst prompts."""

