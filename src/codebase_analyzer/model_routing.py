"""Deterministic model routing for cost-aware semantic analysis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import ClassVar

from codebase_analyzer.parsed_models import ParsedMethod


class ModelTier(StrEnum):
    """Semantic extraction tiers supported by the analyzer."""

    ROUTINE = "routine"
    COMPLEX = "complex"


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """Auditable routing result for one analysis chunk."""

    tier: ModelTier
    model: str
    reasons: tuple[str, ...]


class ModelRouter:
    """Select an extraction model using only deterministic source facts."""

    _COMPLEX_PATH_MARKERS: ClassVar[tuple[str, ...]] = (
        "/services/auth/",
        "/security/",
        "/config/",
    )
    _ROUTINE_OBJECT_METHODS: ClassVar[frozenset[str]] = frozenset(
        {"equals", "hashcode", "tostring"}
    )

    def __init__(
        self,
        *,
        routine_model: str,
        complex_model: str,
        large_chunk_tokens: int = 4_000,
        business_complexity_threshold: int = 6,
    ) -> None:
        self.routine_model = routine_model
        self.complex_model = complex_model
        self.large_chunk_tokens = large_chunk_tokens
        self.business_complexity_threshold = business_complexity_threshold

    def route(
        self,
        *,
        path: str,
        token_count: int,
        has_parser_warnings: bool,
        methods: Iterable[ParsedMethod],
    ) -> ModelRoute:
        """Route a chunk and retain the deterministic reasons for the decision."""

        normalized_path = f"/{path.casefold().strip('/')}"
        filename = PurePosixPath(normalized_path).name
        suffixes = PurePosixPath(normalized_path).suffixes
        reasons: list[str] = []

        if has_parser_warnings:
            reasons.append("parser_warning")
        if token_count >= self.large_chunk_tokens:
            reasons.append("large_chunk")
        if any(marker in normalized_path for marker in self._COMPLEX_PATH_MARKERS):
            reasons.append("architecture_sensitive_path")
        if "/repository/custom/" in normalized_path and filename.endswith("impl.java"):
            reasons.append("custom_query_implementation")
        if filename in {"build.gradle.kts", "settings.gradle.kts", "readme.md"}:
            reasons.append("architecture_context")
        if suffixes and suffixes[-1] == ".sql":
            reasons.append("query_definition")
        if filename.startswith("application.") or filename.startswith("bootstrap."):
            reasons.append("runtime_configuration")

        business_complexities = [
            method.cyclomatic_complexity
            for method in methods
            if method.name.casefold() not in self._ROUTINE_OBJECT_METHODS
        ]
        if (
            business_complexities
            and max(business_complexities) >= self.business_complexity_threshold
        ):
            reasons.append("business_control_flow")

        if reasons:
            return ModelRoute(
                tier=ModelTier.COMPLEX,
                model=self.complex_model,
                reasons=tuple(dict.fromkeys(reasons)),
            )
        return ModelRoute(
            tier=ModelTier.ROUTINE,
            model=self.routine_model,
            reasons=("default_routine",),
        )
