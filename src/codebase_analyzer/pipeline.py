"""End-to-end orchestration from repository bytes to validated JSON knowledge."""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, Protocol

from codebase_analyzer import __version__
from codebase_analyzer.cache import JsonModelCache
from codebase_analyzer.chunking import AnalysisChunk, ChunkPlanner
from codebase_analyzer.config import AnalyzerSettings
from codebase_analyzer.java_parser import parse_java_files
from codebase_analyzer.llm import PROMPT_VERSION, UsageTracker
from codebase_analyzer.llm_schemas import (
    ChunkInterpretation,
    MethodInterpretation,
    ModuleInterpretation,
    RepositoryInterpretation,
)
from codebase_analyzer.model_routing import ModelRoute, ModelRouter, ModelTier
from codebase_analyzer.parsed_models import ParsedJavaFile, ParsedMethod, ParsedType
from codebase_analyzer.scanner import FileRecord, RepositoryScanner, ScanPolicy, ScanResult
from codebase_analyzer.schemas import (
    AnalysisMetadata,
    AnalysisReport,
    ComplexityMetrics,
    FileKnowledge,
    MethodKnowledge,
    ModelRoutingStats,
    ModelSelection,
    ModuleKnowledge,
    ProjectOverview,
    SourceLocation,
    TokenUsage,
    TypeKnowledge,
)
from codebase_analyzer.tokenization import TokenCounter

ProgressCallback = Callable[[str], None]


def module_path_for(path: str) -> str:
    """Map a repository-relative file path to a stable analysis module."""

    parts = PurePosixPath(path).parts
    if "test" in parts or "tests" in parts:
        return "tests"
    if "services" in parts:
        index = parts.index("services")
        if index + 1 < len(parts):
            return f"services/{parts[index + 1]}"
    if "app" in parts:
        index = parts.index("app")
        if index + 1 < len(parts) and parts[index + 1] in {"common", "config"}:
            return parts[index + 1]
    if "resources" in parts:
        return "resources"
    return "project"


class KnowledgeExtractor(Protocol):
    usage: UsageTracker

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
    ) -> ChunkInterpretation: ...

    def analyze_module(
        self,
        *,
        module_name: str,
        module_path: str,
        file_evidence: list[dict[str, Any]],
    ) -> ModuleInterpretation: ...

    def analyze_repository(
        self,
        *,
        repository_name: str,
        repository_metadata: dict[str, Any],
        module_evidence: list[dict[str, Any]],
        complexity_hotspots: list[dict[str, Any]],
    ) -> RepositoryInterpretation: ...


@dataclass(frozen=True, slots=True)
class _Document:
    record: FileRecord
    parsed: ParsedJavaFile
    chunks: tuple[AnalysisChunk, ...]


@dataclass(frozen=True, slots=True)
class _ChunkTask:
    document: _Document
    chunk: AnalysisChunk
    types: tuple[ParsedType, ...]
    methods: tuple[ParsedMethod, ...]
    route: ModelRoute


class _CacheStats:
    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self._lock = Lock()

    def hit(self) -> None:
        with self._lock:
            self.hits += 1

    def miss(self) -> None:
        with self._lock:
            self.misses += 1


class AnalysisPipeline:
    """Coordinate deterministic analysis, bounded LLM calls, caching, and merging."""

    def __init__(
        self,
        *,
        settings: AnalyzerSettings,
        extractor: KnowledgeExtractor,
        cache_directory: Path,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings
        self.extractor = extractor
        self.progress = progress
        self.token_counter = TokenCounter(settings.routine_model)
        self.chunk_planner = ChunkPlanner(
            self.token_counter,
            chunk_token_budget=settings.chunk_token_budget,
        )
        self.model_router = ModelRouter(
            routine_model=settings.routine_model,
            complex_model=settings.complex_model,
        )
        self.cache = JsonModelCache(cache_directory)
        self.cache_stats = _CacheStats()

    def run(
        self,
        repository_root: Path,
        *,
        include_tests: bool = True,
        max_file_bytes: int = 1_000_000,
    ) -> AnalysisReport:
        root = repository_root.expanduser().resolve()
        scan = RepositoryScanner(
            ScanPolicy(
                include_tests=include_tests,
                max_file_bytes=max_file_bytes,
            )
        ).scan(root)
        self._notify(f"Discovered {scan.selected_file_count} analyzable files")

        documents = self._prepare_documents(root, scan)
        tasks = [
            self._build_chunk_task(document, chunk)
            for document in documents
            for chunk in document.chunks
        ]
        self._notify(f"Prepared {len(tasks)} token-bounded analysis chunks")
        chunk_results = self._analyze_chunks(tasks)

        files = self._build_file_knowledge(documents, chunk_results)
        modules = self._build_modules(files)
        overview = self._build_overview(root.name, files, modules)
        usage = self.extractor.usage.snapshot()
        usage_by_model = self.extractor.usage.snapshots_by_model()
        parser_warnings = sum(file.has_parser_warnings for file in files)
        routing_reasons = Counter(
            reason
            for task in tasks
            if task.route.tier is ModelTier.COMPLEX
            for reason in task.route.reasons
        )

        return AnalysisReport(
            metadata=AnalysisMetadata(
                generated_at=datetime.now(UTC),
                analyzer_version=__version__,
                repository_name=root.name,
                repository_commit=self._git_commit(root),
                models=ModelSelection(
                    routine_extraction=self.settings.routine_model,
                    complex_extraction=self.settings.complex_model,
                    synthesis=self.settings.synthesis_model,
                ),
                model_routing=ModelRoutingStats(
                    routine_chunk_count=sum(
                        task.route.tier is ModelTier.ROUTINE for task in tasks
                    ),
                    complex_chunk_count=sum(
                        task.route.tier is ModelTier.COMPLEX for task in tasks
                    ),
                    synthesis_artifact_count=len(modules) + 1,
                    complex_routing_reasons=dict(sorted(routing_reasons.items())),
                ),
                prompt_version=PROMPT_VERSION,
                parser="javalang + lizard",
                tokenizer=self.token_counter.encoding_name,
                token_usage_estimated=True,
                planned_chunk_count=len(tasks),
                planned_source_tokens=sum(task.chunk.token_count for task in tasks),
                analyzed_file_count=len(files),
                skipped_file_count=scan.skipped_file_count,
                parser_warning_file_count=parser_warnings,
                cache_hits=self.cache_stats.hits,
                cache_misses=self.cache_stats.misses,
                token_usage=TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                ),
                token_usage_by_model={
                    model: TokenUsage(
                        prompt_tokens=snapshot.prompt_tokens,
                        completion_tokens=snapshot.completion_tokens,
                        total_tokens=snapshot.total_tokens,
                    )
                    for model, snapshot in sorted(usage_by_model.items())
                },
            ),
            overview=overview,
            modules=modules,
            files=files,
        )

    def _prepare_documents(self, root: Path, scan: ScanResult) -> list[_Document]:
        documents: list[_Document] = []
        java_records = tuple(
            record for record in scan.files if Path(record.path).suffix.lower() == ".java"
        )
        parsed_java = parse_java_files(root, java_records)
        for record in scan.files:
            if Path(record.path).suffix.lower() == ".java":
                parsed = parsed_java[record.path]
                chunks = self.chunk_planner.plan_file(parsed)
            else:
                source = (root / record.path).read_text(encoding="utf-8", errors="replace")
                parsed = ParsedJavaFile(
                    path=record.path,
                    package=None,
                    imports=(),
                    types=(),
                    source=source,
                    sha256=record.sha256,
                    has_parser_warnings=False,
                )
                chunks = self.chunk_planner.plan_text(
                    file_path=record.path,
                    source=source,
                    sha256=record.sha256,
                )
            documents.append(_Document(record=record, parsed=parsed, chunks=chunks))
        return documents

    def _build_chunk_task(
        self,
        document: _Document,
        chunk: AnalysisChunk,
    ) -> _ChunkTask:
        types = tuple(self._overlapping_types(document.parsed.types, chunk))
        methods = tuple(
            method
            for parsed_type in types
            for method in parsed_type.methods
            if self._overlaps(method.start_line, method.end_line, chunk)
        )
        route = self.model_router.route(
            path=document.record.path,
            token_count=chunk.token_count,
            has_parser_warnings=document.parsed.has_parser_warnings,
            methods=methods,
        )
        return _ChunkTask(
            document=document,
            chunk=chunk,
            types=types,
            methods=methods,
            route=route,
        )

    def _analyze_chunks(
        self,
        tasks: list[_ChunkTask],
    ) -> dict[str, ChunkInterpretation]:
        results: dict[str, ChunkInterpretation] = {}
        if self.settings.max_concurrency == 1:
            for index, task in enumerate(tasks, start=1):
                results[task.chunk.chunk_id] = self._analyze_chunk(task)
                self._notify(f"Analyzed chunk {index}/{len(tasks)}")
            return results

        with ThreadPoolExecutor(max_workers=self.settings.max_concurrency) as executor:
            futures: dict[Future[ChunkInterpretation], _ChunkTask] = {
                executor.submit(self._analyze_chunk, task): task for task in tasks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                results[task.chunk.chunk_id] = future.result()
                if completed == len(tasks) or completed % 10 == 0:
                    self._notify(f"Analyzed chunk {completed}/{len(tasks)}")
        return results

    def _analyze_chunk(self, task: _ChunkTask) -> ChunkInterpretation:
        document = task.document
        chunk = task.chunk
        key = self.cache.build_key(
            "chunk",
            PROMPT_VERSION,
            task.route.model,
            chunk.chunk_id,
        )
        cached = self.cache.get(key, ChunkInterpretation)
        if cached is not None:
            self.cache_stats.hit()
            return cached

        result = self.extractor.analyze_chunk(
            file_path=document.record.path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            package=document.parsed.package,
            imports=document.parsed.imports,
            type_catalog=[
                {
                    "qualified_name": parsed_type.qualified_name,
                    "kind": parsed_type.kind.value,
                    "lines": [parsed_type.start_line, parsed_type.end_line],
                }
                for parsed_type in task.types
            ],
            method_catalog=[
                {
                    "qualified_name": method.qualified_name,
                    "signature": method.signature,
                    "lines": [method.start_line, method.end_line],
                    "cyclomatic_complexity": method.cyclomatic_complexity,
                }
                for method in task.methods
            ],
            source=chunk.content,
            model_tier=task.route.tier,
        )
        if result.file_path != document.record.path:
            result = result.model_copy(update={"file_path": document.record.path})
        self.cache.put(key, result)
        self.cache_stats.miss()
        return result

    def _build_file_knowledge(
        self,
        documents: list[_Document],
        chunk_results: dict[str, ChunkInterpretation],
    ) -> list[FileKnowledge]:
        files: list[FileKnowledge] = []
        for document in documents:
            interpretations = [chunk_results[chunk.chunk_id] for chunk in document.chunks]
            type_interpretations = {
                item.qualified_name: item
                for interpretation in interpretations
                for item in interpretation.types
            }
            method_interpretations = {
                (item.qualified_name, item.signature): item
                for interpretation in interpretations
                for item in interpretation.methods
            }

            types: list[TypeKnowledge] = []
            for parsed_type in document.parsed.types:
                interpreted_type = type_interpretations.get(parsed_type.qualified_name)
                nested_type_names = [
                    candidate.name
                    for candidate in document.parsed.types
                    if candidate.qualified_name.startswith(f"{parsed_type.qualified_name}.")
                ]
                fallback_description = (
                    f"Container {parsed_type.kind.value} grouping the nested types "
                    f"{', '.join(nested_type_names)}."
                    if nested_type_names
                    else f"Declares the {parsed_type.kind.value} {parsed_type.qualified_name}."
                )
                methods = [
                    self._build_method_knowledge(
                        document.record.path,
                        method,
                        method_interpretations.get((method.qualified_name, method.signature)),
                    )
                    for method in parsed_type.methods
                ]
                types.append(
                    TypeKnowledge(
                        name=parsed_type.name,
                        qualified_name=parsed_type.qualified_name,
                        kind=parsed_type.kind,
                        description=interpreted_type.description
                        if interpreted_type
                        else fallback_description,
                        annotations=list(parsed_type.annotations),
                        extends=list(parsed_type.extends),
                        implements=list(parsed_type.implements),
                        methods=methods,
                        location=SourceLocation(
                            path=document.record.path,
                            start_line=parsed_type.start_line,
                            end_line=parsed_type.end_line,
                        ),
                        noteworthy_aspects=interpreted_type.noteworthy_aspects
                        if interpreted_type
                        else [
                            "Description was derived from deterministic type structure "
                            "because the LLM omitted this declaration."
                        ],
                    )
                )

            summaries = self._unique(interpretation.summary for interpretation in interpretations)
            noteworthy = self._unique(
                aspect
                for interpretation in interpretations
                for aspect in interpretation.noteworthy_aspects
            )
            if document.parsed.has_parser_warnings:
                noteworthy.append(
                    "javalang used the Lizard fallback because the file contains newer Java syntax."
                )
            files.append(
                FileKnowledge(
                    path=document.record.path,
                    language=document.record.language,
                    sha256=document.record.sha256,
                    summary=" ".join(summaries),
                    has_parser_warnings=document.parsed.has_parser_warnings,
                    types=types,
                    noteworthy_aspects=noteworthy,
                )
            )
        return files

    @staticmethod
    def _build_method_knowledge(
        path: str,
        method: ParsedMethod,
        interpretation: MethodInterpretation | None,
    ) -> MethodKnowledge:
        return MethodKnowledge(
            name=method.name,
            qualified_name=method.qualified_name,
            signature=method.signature,
            description=interpretation.description
            if interpretation
            else f"Implements {method.signature}.",
            return_type=method.return_type,
            parameters=[parameter.declaration for parameter in method.parameters],
            modifiers=list(method.modifiers),
            annotations=list(method.annotations),
            complexity=ComplexityMetrics(
                lines_of_code=method.lines_of_code,
                cyclomatic_complexity=method.cyclomatic_complexity,
                parameter_count=len(method.parameters),
            ),
            location=SourceLocation(
                path=path,
                start_line=method.start_line,
                end_line=method.end_line,
            ),
            noteworthy_aspects=interpretation.noteworthy_aspects
            if interpretation
            else ["Semantic method description was not returned by the LLM."],
        )

    def _build_modules(self, files: list[FileKnowledge]) -> list[ModuleKnowledge]:
        grouped: dict[str, list[FileKnowledge]] = defaultdict(list)
        for file in files:
            grouped[module_path_for(file.path)].append(file)

        modules: list[ModuleKnowledge] = []
        for module_path, module_files in sorted(grouped.items()):
            module_name = module_path.rsplit("/", maxsplit=1)[-1]
            evidence = [self._file_evidence(file) for file in module_files]
            evidence_json = json.dumps(evidence, sort_keys=True)
            key = self.cache.build_key(
                "module",
                PROMPT_VERSION,
                self.settings.synthesis_model,
                module_path,
                evidence_json,
            )
            interpretation = self.cache.get(key, ModuleInterpretation)
            if interpretation is None:
                interpretation = self.extractor.analyze_module(
                    module_name=module_name,
                    module_path=module_path,
                    file_evidence=evidence,
                )
                self.cache.put(key, interpretation)
                self.cache_stats.miss()
            else:
                self.cache_stats.hit()
            modules.append(
                ModuleKnowledge(
                    name=module_name,
                    path=module_path,
                    purpose=interpretation.purpose,
                    key_components=interpretation.key_components,
                    dependencies=interpretation.dependencies,
                    noteworthy_aspects=interpretation.noteworthy_aspects,
                )
            )
        return modules

    def _build_overview(
        self,
        repository_name: str,
        files: list[FileKnowledge],
        modules: list[ModuleKnowledge],
    ) -> ProjectOverview:
        methods = [
            method
            for file in files
            for parsed_type in file.types
            for method in parsed_type.methods
        ]
        hotspots = [
            {
                "qualified_name": method.qualified_name,
                "signature": method.signature,
                "cyclomatic_complexity": method.complexity.cyclomatic_complexity,
                "lines_of_code": method.complexity.lines_of_code,
                "path": method.location.path,
            }
            for method in sorted(
                methods,
                key=lambda item: (
                    item.complexity.cyclomatic_complexity,
                    item.complexity.lines_of_code,
                ),
                reverse=True,
            )[:20]
        ]
        metadata = {
            "analyzed_files": len(files),
            "java_types": sum(len(file.types) for file in files),
            "methods": len(methods),
            "modules": len(modules),
            "parser_warning_files": sum(file.has_parser_warnings for file in files),
        }
        module_evidence = [module.model_dump(mode="json") for module in modules]
        key = self.cache.build_key(
            "repository",
            PROMPT_VERSION,
            self.settings.synthesis_model,
            repository_name,
            json.dumps(metadata, sort_keys=True),
            json.dumps(module_evidence, sort_keys=True),
            json.dumps(hotspots, sort_keys=True),
        )
        interpretation = self.cache.get(key, RepositoryInterpretation)
        if interpretation is None:
            interpretation = self.extractor.analyze_repository(
                repository_name=repository_name,
                repository_metadata=metadata,
                module_evidence=module_evidence,
                complexity_hotspots=hotspots,
            )
            self.cache.put(key, interpretation)
            self.cache_stats.miss()
        else:
            self.cache_stats.hit()
        return ProjectOverview(
            name=repository_name,
            purpose=interpretation.purpose,
            architecture=interpretation.architecture,
            technologies=interpretation.technologies,
            entry_points=interpretation.entry_points,
            key_workflows=interpretation.key_workflows,
            complexity_summary=interpretation.complexity_summary,
            noteworthy_aspects=interpretation.noteworthy_aspects,
            limitations=interpretation.limitations,
        )

    @staticmethod
    def _file_evidence(file: FileKnowledge) -> dict[str, Any]:
        methods = [method for parsed_type in file.types for method in parsed_type.methods]
        max_complexity = max(
            (method.complexity.cyclomatic_complexity for method in methods),
            default=0,
        )
        return {
            "path": file.path,
            "summary": file.summary,
            "types": [parsed_type.qualified_name for parsed_type in file.types],
            "max_cyclomatic_complexity": max_complexity,
            "noteworthy_aspects": file.noteworthy_aspects,
        }

    @staticmethod
    def _overlapping_types(
        types: tuple[ParsedType, ...],
        chunk: AnalysisChunk,
    ) -> list[ParsedType]:
        return [
            parsed_type
            for parsed_type in types
            if parsed_type.start_line <= chunk.end_line
            and parsed_type.end_line >= chunk.start_line
        ]

    @staticmethod
    def _overlaps(start_line: int, end_line: int, chunk: AnalysisChunk) -> bool:
        return start_line <= chunk.end_line and end_line >= chunk.start_line

    @staticmethod
    def _git_commit(root: Path) -> str | None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _notify(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)
