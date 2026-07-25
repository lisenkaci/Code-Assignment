from __future__ import annotations

import pytest
from pydantic import ValidationError

from codebase_analyzer.config import AnalyzerSettings


def test_prompt_budget_reserves_output_tokens() -> None:
    settings = AnalyzerSettings(
        max_context_tokens=10_000,
        reserved_output_tokens=2_000,
        chunk_token_budget=7_000,
    )

    assert settings.prompt_token_budget == 8_000


def test_chunk_budget_must_fit_available_prompt() -> None:
    with pytest.raises(ValidationError, match="chunk token budget exceeds"):
        AnalyzerSettings(
            max_context_tokens=10_000,
            reserved_output_tokens=2_000,
            chunk_token_budget=8_001,
        )
