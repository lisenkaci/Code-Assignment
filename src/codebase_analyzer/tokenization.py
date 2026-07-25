"""Model-aware token counting with deterministic fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken
from tiktoken import Encoding


@dataclass(frozen=True, slots=True)
class TokenSlice:
    text: str
    token_count: int


class TokenCounter:
    """Count and split text using the configured model's tokenizer."""

    def __init__(self, model: str, fallback_encoding: str = "o200k_base") -> None:
        self.model = model
        self.encoding_name: str
        try:
            self._encoding: Encoding = tiktoken.encoding_for_model(model)
            self.encoding_name = self._encoding.name
        except KeyError:
            self._encoding = tiktoken.get_encoding(fallback_encoding)
            self.encoding_name = fallback_encoding

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def split(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
    ) -> tuple[TokenSlice, ...]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be between zero and max_tokens")

        tokens = self._encoding.encode(text)
        if not tokens:
            return (TokenSlice(text="", token_count=0),)

        slices: list[TokenSlice] = []
        step = max_tokens - overlap_tokens
        for start in range(0, len(tokens), step):
            selected = tokens[start : start + max_tokens]
            slices.append(
                TokenSlice(
                    text=self._encoding.decode(selected),
                    token_count=len(selected),
                )
            )
            if start + max_tokens >= len(tokens):
                break
        return tuple(slices)
