"""Strict intermediate contracts returned by the language model."""

from pydantic import Field

from codebase_analyzer.schemas import StrictModel


class MethodInterpretation(StrictModel):
    qualified_name: str
    signature: str
    description: str = Field(min_length=1)
    noteworthy_aspects: list[str] = Field(default_factory=list)


class TypeInterpretation(StrictModel):
    qualified_name: str
    description: str = Field(min_length=1)
    noteworthy_aspects: list[str] = Field(default_factory=list)


class ChunkInterpretation(StrictModel):
    file_path: str
    summary: str = Field(min_length=1)
    types: list[TypeInterpretation] = Field(default_factory=list)
    methods: list[MethodInterpretation] = Field(default_factory=list)
    noteworthy_aspects: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    architecture_signals: list[str] = Field(default_factory=list)


class ModuleInterpretation(StrictModel):
    name: str
    path: str
    purpose: str = Field(min_length=1)
    key_components: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    noteworthy_aspects: list[str] = Field(default_factory=list)


class RepositoryInterpretation(StrictModel):
    name: str
    purpose: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    technologies: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    key_workflows: list[str] = Field(default_factory=list)
    complexity_summary: str = Field(min_length=1)
    noteworthy_aspects: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
