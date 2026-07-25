from __future__ import annotations

from pathlib import Path
from typing import Any

from codebase_analyzer.config import AnalyzerSettings
from codebase_analyzer.llm import UsageTracker
from codebase_analyzer.llm_schemas import (
    ChunkInterpretation,
    MethodInterpretation,
    ModuleInterpretation,
    RepositoryInterpretation,
    TypeInterpretation,
)
from codebase_analyzer.model_routing import ModelTier
from codebase_analyzer.pipeline import AnalysisPipeline


class FakeExtractor:
    def __init__(self) -> None:
        self.usage = UsageTracker()
        self.call_count = 0

    def analyze_chunk(
        self,
        *,
        file_path: str,
        start_line: int,
        end_line: int,
        package: str | None,
        imports: tuple[str, ...],
        type_catalog: list[dict[str, Any]],
        method_catalog: list[dict[str, Any]],
        source: str,
        model_tier: ModelTier,
    ) -> ChunkInterpretation:
        self.call_count += 1
        return ChunkInterpretation(
            file_path=file_path,
            summary=f"Analyzes {file_path}.",
            types=[
                TypeInterpretation(
                    qualified_name=item["qualified_name"],
                    description=f"Type {item['qualified_name']}.",
                )
                for item in type_catalog
            ],
            methods=[
                MethodInterpretation(
                    qualified_name=item["qualified_name"],
                    signature=item["signature"],
                    description=f"Method {item['qualified_name']}.",
                )
                for item in method_catalog
            ],
        )

    def analyze_module(
        self,
        *,
        module_name: str,
        module_path: str,
        file_evidence: list[dict[str, Any]],
    ) -> ModuleInterpretation:
        self.call_count += 1
        return ModuleInterpretation(
            name=module_name,
            path=module_path,
            purpose=f"Groups {len(file_evidence)} analyzed files.",
        )

    def analyze_repository(
        self,
        *,
        repository_name: str,
        repository_metadata: dict[str, Any],
        module_evidence: list[dict[str, Any]],
        complexity_hotspots: list[dict[str, Any]],
    ) -> RepositoryInterpretation:
        self.call_count += 1
        return RepositoryInterpretation(
            name=repository_name,
            purpose="Test repository.",
            architecture="Layered test architecture.",
            complexity_summary=f"{len(complexity_hotspots)} measured hotspots.",
        )


def build_pipeline(tmp_path: Path, extractor: FakeExtractor) -> AnalysisPipeline:
    settings = AnalyzerSettings(
        routine_model="unknown-routine-model",
        complex_model="unknown-complex-model",
        synthesis_model="unknown-synthesis-model",
        max_context_tokens=4_000,
        reserved_output_tokens=500,
        chunk_token_budget=1_000,
        max_concurrency=1,
    )
    return AnalysisPipeline(
        settings=settings,
        extractor=extractor,
        cache_directory=tmp_path / "cache",
    )


def test_pipeline_builds_valid_report_and_reuses_cache(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    java_file = repository / "src/main/java/com/example/Example.java"
    java_file.parent.mkdir(parents=True)
    java_file.write_text(
        "package com.example;\n"
        "public class Example {\n"
        "  public int choose(boolean value) {\n"
        "    if (value) return 1;\n"
        "    return 0;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    (repository / "README.md").write_text("# Example\n", encoding="utf-8")

    first_extractor = FakeExtractor()
    first_report = build_pipeline(tmp_path, first_extractor).run(repository)

    assert first_report.metadata.analyzed_file_count == 2
    assert first_report.metadata.cache_misses > 0
    assert first_report.metadata.cache_hits == 0
    assert first_report.metadata.models.routine_extraction == "unknown-routine-model"
    assert first_report.metadata.model_routing.complex_chunk_count == 1
    assert first_report.metadata.model_routing.routine_chunk_count == 1
    method = next(
        method
        for file in first_report.files
        for parsed_type in file.types
        for method in parsed_type.methods
    )
    assert method.description == "Method com.example.Example.choose."
    assert method.complexity.cyclomatic_complexity == 2

    second_extractor = FakeExtractor()
    second_report = build_pipeline(tmp_path, second_extractor).run(repository)

    assert second_extractor.call_count == 0
    assert second_report.metadata.cache_hits == first_report.metadata.cache_misses
    assert second_report.overview == first_report.overview
