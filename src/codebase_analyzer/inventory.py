"""Deterministic preflight inventory produced without credentials or LLM calls."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from codebase_analyzer.chunking import ChunkPlanner
from codebase_analyzer.java_parser import parse_java_files
from codebase_analyzer.model_routing import ModelRouter, ModelTier
from codebase_analyzer.parsed_models import ParsedJavaFile, ParsedMethod
from codebase_analyzer.pipeline import module_path_for
from codebase_analyzer.scanner import RepositoryScanner, ScanPolicy
from codebase_analyzer.tokenization import TokenCounter


class InventoryBuilder:
    """Quantify scope, complexity, and expected LLM work before a paid run."""

    def __init__(
        self,
        *,
        routine_model: str,
        complex_model: str,
        synthesis_model: str,
        chunk_token_budget: int,
    ) -> None:
        self.routine_model = routine_model
        self.complex_model = complex_model
        self.synthesis_model = synthesis_model
        self.token_counter = TokenCounter(routine_model)
        self.chunk_planner = ChunkPlanner(
            self.token_counter,
            chunk_token_budget=chunk_token_budget,
        )
        self.model_router = ModelRouter(
            routine_model=routine_model,
            complex_model=complex_model,
        )

    def build(
        self,
        repository_root: Path,
        *,
        include_tests: bool = True,
        max_file_bytes: int = 1_000_000,
    ) -> dict[str, Any]:
        root = repository_root.expanduser().resolve()
        scan = RepositoryScanner(
            ScanPolicy(
                include_tests=include_tests,
                max_file_bytes=max_file_bytes,
            )
        ).scan(root)

        methods: list[tuple[str, ParsedMethod]] = []
        type_count = 0
        java_file_count = 0
        chunk_count = 0
        total_tokens = 0
        parser_warning_files: list[str] = []
        routing_tiers: Counter[str] = Counter()
        routing_reasons: Counter[str] = Counter()
        modules: dict[str, Counter[str]] = defaultdict(Counter)
        languages: Counter[str] = Counter()
        java_records = tuple(
            record for record in scan.files if Path(record.path).suffix.lower() == ".java"
        )
        parsed_java = parse_java_files(root, java_records)

        for record in scan.files:
            module = module_path_for(record.path)
            modules[module]["files"] += 1
            languages[record.language] += 1
            parsed: ParsedJavaFile | None
            if Path(record.path).suffix.lower() == ".java":
                java_file_count += 1
                parsed = parsed_java[record.path]
                chunks = self.chunk_planner.plan_file(parsed)
                type_count += len(parsed.types)
                file_methods = [
                    method for parsed_type in parsed.types for method in parsed_type.methods
                ]
                methods.extend((record.path, method) for method in file_methods)
                modules[module]["types"] += len(parsed.types)
                modules[module]["methods"] += len(file_methods)
                if parsed.has_parser_warnings:
                    parser_warning_files.append(record.path)
                has_parser_warnings = parsed.has_parser_warnings
            else:
                source = (root / record.path).read_text(encoding="utf-8", errors="replace")
                chunks = self.chunk_planner.plan_text(
                    file_path=record.path,
                    source=source,
                    sha256=record.sha256,
                )
                parsed = None
                has_parser_warnings = False
            for chunk in chunks:
                chunk_methods = (
                    (
                        method
                        for parsed_type in parsed.types
                        if parsed_type.start_line <= chunk.end_line
                        and parsed_type.end_line >= chunk.start_line
                        for method in parsed_type.methods
                        if method.start_line <= chunk.end_line
                        and method.end_line >= chunk.start_line
                    )
                    if parsed is not None
                    else ()
                )
                route = self.model_router.route(
                    path=record.path,
                    token_count=chunk.token_count,
                    has_parser_warnings=has_parser_warnings,
                    methods=chunk_methods,
                )
                routing_tiers[route.tier.value] += 1
                if route.tier is ModelTier.COMPLEX:
                    routing_reasons.update(route.reasons)
            chunk_count += len(chunks)
            total_tokens += sum(chunk.token_count for chunk in chunks)

        complexities = [method.cyclomatic_complexity for _, method in methods]
        hotspots = sorted(
            methods,
            key=lambda item: (
                item[1].cyclomatic_complexity,
                item[1].lines_of_code,
            ),
            reverse=True,
        )[:20]

        return {
            "repository": root.name,
            "scope": {
                "selected_files": scan.selected_file_count,
                "skipped_files": scan.skipped_file_count,
                "skipped_by_reason": dict(sorted(scan.skipped_by_reason.items())),
                "java_files": java_file_count,
                "types": type_count,
                "methods": len(methods),
                "parser_warning_files": sorted(parser_warning_files),
                "languages": dict(sorted(languages.items())),
            },
            "token_plan": {
                "tokenizer": self.token_counter.encoding_name,
                "source_tokens": total_tokens,
                "analysis_chunks": chunk_count,
                "chunk_token_budget": self.chunk_planner.chunk_token_budget,
            },
            "model_plan": {
                "models": {
                    "routine_extraction": self.routine_model,
                    "complex_extraction": self.complex_model,
                    "synthesis": self.synthesis_model,
                },
                "routine_extraction_chunks": routing_tiers[ModelTier.ROUTINE.value],
                "complex_extraction_chunks": routing_tiers[ModelTier.COMPLEX.value],
                "synthesis_artifacts": len(modules) + 1,
                "complex_routing_reasons": dict(sorted(routing_reasons.items())),
            },
            "complexity": {
                "average_cyclomatic_complexity": round(mean(complexities), 2)
                if complexities
                else 0,
                "maximum_cyclomatic_complexity": max(complexities, default=0),
                "low_1_to_5": sum(value <= 5 for value in complexities),
                "moderate_6_to_10": sum(6 <= value <= 10 for value in complexities),
                "high_11_plus": sum(value >= 11 for value in complexities),
                "hotspots": [
                    {
                        "path": path,
                        "qualified_name": method.qualified_name,
                        "signature": method.signature,
                        "cyclomatic_complexity": method.cyclomatic_complexity,
                        "lines_of_code": method.lines_of_code,
                        "start_line": method.start_line,
                    }
                    for path, method in hotspots
                ],
            },
            "modules": {
                name: {
                    "files": counts["files"],
                    "types": counts["types"],
                    "methods": counts["methods"],
                }
                for name, counts in sorted(modules.items())
            },
        }
