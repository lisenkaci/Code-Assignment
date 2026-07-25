"""Versioned, machine-readable output contracts for codebase analysis."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects unexpected LLM output fields."""

    model_config = ConfigDict(extra="forbid")


class TypeKind(StrEnum):
    CLASS = "class"
    INTERFACE = "interface"
    ENUM = "enum"
    RECORD = "record"
    ANNOTATION = "annotation"


class SourceLocation(StrictModel):
    path: str = Field(description="Repository-relative POSIX path")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class ComplexityMetrics(StrictModel):
    lines_of_code: int = Field(ge=0)
    cyclomatic_complexity: int = Field(ge=1)
    parameter_count: int = Field(ge=0)


class MethodKnowledge(StrictModel):
    name: str
    qualified_name: str
    signature: str
    description: str
    return_type: str | None = None
    parameters: list[str] = Field(default_factory=list)
    modifiers: list[str] = Field(default_factory=list)
    annotations: list[str] = Field(default_factory=list)
    complexity: ComplexityMetrics
    location: SourceLocation
    noteworthy_aspects: list[str] = Field(default_factory=list)


class TypeKnowledge(StrictModel):
    name: str
    qualified_name: str
    kind: TypeKind
    description: str
    annotations: list[str] = Field(default_factory=list)
    extends: list[str] = Field(default_factory=list)
    implements: list[str] = Field(default_factory=list)
    methods: list[MethodKnowledge] = Field(default_factory=list)
    location: SourceLocation
    noteworthy_aspects: list[str] = Field(default_factory=list)


class FileKnowledge(StrictModel):
    path: str
    language: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    summary: str
    has_parser_warnings: bool = False
    types: list[TypeKnowledge] = Field(default_factory=list)
    noteworthy_aspects: list[str] = Field(default_factory=list)


class ModuleKnowledge(StrictModel):
    name: str
    path: str
    purpose: str
    key_components: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    noteworthy_aspects: list[str] = Field(default_factory=list)


class ProjectOverview(StrictModel):
    name: str
    purpose: str
    architecture: str
    technologies: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    key_workflows: list[str] = Field(default_factory=list)
    complexity_summary: str
    noteworthy_aspects: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class TokenUsage(StrictModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ModelSelection(StrictModel):
    routine_extraction: str
    complex_extraction: str
    synthesis: str


class ModelRoutingStats(StrictModel):
    routine_chunk_count: int = Field(default=0, ge=0)
    complex_chunk_count: int = Field(default=0, ge=0)
    synthesis_artifact_count: int = Field(default=0, ge=0)
    complex_routing_reasons: dict[str, int] = Field(default_factory=dict)


class AnalysisMetadata(StrictModel):
    schema_version: Literal["1.1"] = "1.1"
    generated_at: datetime
    analyzer_version: str
    repository_name: str
    repository_commit: str | None = None
    models: ModelSelection
    model_routing: ModelRoutingStats = Field(default_factory=ModelRoutingStats)
    prompt_version: str
    parser: str
    tokenizer: str
    token_usage_estimated: bool = True
    token_usage_scope: str = "Uncached API calls in this run; cache hits are excluded."
    planned_chunk_count: int = Field(default=0, ge=0)
    planned_source_tokens: int = Field(default=0, ge=0)
    analyzed_file_count: int = Field(ge=0)
    skipped_file_count: int = Field(ge=0)
    parser_warning_file_count: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    token_usage_by_model: dict[str, TokenUsage] = Field(default_factory=dict)


class AnalysisReport(StrictModel):
    metadata: AnalysisMetadata
    overview: ProjectOverview
    modules: list[ModuleKnowledge] = Field(default_factory=list)
    files: list[FileKnowledge] = Field(default_factory=list)
