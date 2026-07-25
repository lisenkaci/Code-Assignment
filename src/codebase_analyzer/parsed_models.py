"""Internal, deterministic representations produced before any LLM call."""

from __future__ import annotations

from dataclasses import dataclass

from codebase_analyzer.schemas import TypeKind


@dataclass(frozen=True, slots=True)
class ParsedParameter:
    name: str
    type: str
    declaration: str


@dataclass(frozen=True, slots=True)
class ParsedMethod:
    name: str
    qualified_name: str
    signature: str
    return_type: str | None
    parameters: tuple[ParsedParameter, ...]
    modifiers: tuple[str, ...]
    annotations: tuple[str, ...]
    start_line: int
    end_line: int
    source: str
    lines_of_code: int
    cyclomatic_complexity: int


@dataclass(frozen=True, slots=True)
class ParsedType:
    name: str
    qualified_name: str
    kind: TypeKind
    annotations: tuple[str, ...]
    extends: tuple[str, ...]
    implements: tuple[str, ...]
    start_line: int
    end_line: int
    methods: tuple[ParsedMethod, ...]


@dataclass(frozen=True, slots=True)
class ParsedJavaFile:
    path: str
    package: str | None
    imports: tuple[str, ...]
    types: tuple[ParsedType, ...]
    source: str
    sha256: str
    has_parser_warnings: bool
