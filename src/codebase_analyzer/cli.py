"""Command-line interface for reproducible analyzer workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from codebase_analyzer import __version__
from codebase_analyzer.config import AnalyzerSettings
from codebase_analyzer.inventory import InventoryBuilder
from codebase_analyzer.llm import LangChainKnowledgeExtractor
from codebase_analyzer.pipeline import AnalysisPipeline
from codebase_analyzer.scanner import RepositoryScanner, ScanPolicy
from codebase_analyzer.schemas import AnalysisReport
from codebase_analyzer.tokenization import TokenCounter

app = typer.Typer(
    name="codebase-analyzer",
    help="Extract structured knowledge from a source repository.",
    no_args_is_help=True,
)
console = Console()


def _write_json(payload: dict[str, Any], output: Path) -> None:
    """Write JSON atomically so interrupted runs do not leave partial results."""

    resolved_output = output.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = resolved_output.with_suffix(f"{resolved_output.suffix}.tmp")
    temporary_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(resolved_output)


@app.command()
def scan(
    repository: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Repository root to inspect.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination for the deterministic manifest."),
    ] = Path("output/manifest.json"),
    exclude_tests: Annotated[
        bool,
        typer.Option("--exclude-tests", help="Exclude files under test directories."),
    ] = False,
    max_file_bytes: Annotated[
        int,
        typer.Option(min=1, help="Skip individual files larger than this limit."),
    ] = 1_000_000,
) -> None:
    """Create a stable manifest before parsing or making LLM requests."""

    policy = ScanPolicy(
        include_tests=not exclude_tests,
        max_file_bytes=max_file_bytes,
    )
    result = RepositoryScanner(policy).scan(repository)
    _write_json(result.to_dict(), output)
    console.print(
        f"Selected [bold]{result.selected_file_count}[/bold] files; "
        f"skipped [bold]{result.skipped_file_count}[/bold]."
    )
    console.print(f"Manifest: {output.expanduser().resolve()}")


@app.command()
def inspect(
    repository: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Repository root to inspect.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination for deterministic preflight metrics."),
    ] = Path("output/sakila-inventory.json"),
    exclude_tests: Annotated[
        bool,
        typer.Option("--exclude-tests", help="Exclude files under test directories."),
    ] = False,
) -> None:
    """Measure scope, complexity, and token demand without making LLM calls."""

    settings = AnalyzerSettings()
    inventory = InventoryBuilder(
        routine_model=settings.routine_model,
        complex_model=settings.complex_model,
        synthesis_model=settings.synthesis_model,
        chunk_token_budget=settings.chunk_token_budget,
    ).build(repository, include_tests=not exclude_tests)
    _write_json(inventory, output)
    console.print(
        f"Prepared [bold]{inventory['token_plan']['analysis_chunks']}[/bold] chunks "
        f"without making LLM calls."
    )
    console.print(f"Inventory: {output.expanduser().resolve()}")


@app.command()
def analyze(
    repository: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Repository root to analyze.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination for the validated analysis report."),
    ] = Path("output/sakila-analysis.json"),
    cache_directory: Annotated[
        Path,
        typer.Option(help="Content-addressed cache for validated LLM responses."),
    ] = Path("output/cache"),
    exclude_tests: Annotated[
        bool,
        typer.Option("--exclude-tests", help="Exclude files under test directories."),
    ] = False,
    routine_model: Annotated[
        str | None,
        typer.Option(help="Override CODE_ANALYZER_ROUTINE_MODEL for this run."),
    ] = None,
    complex_model: Annotated[
        str | None,
        typer.Option(help="Override CODE_ANALYZER_COMPLEX_MODEL for this run."),
    ] = None,
    synthesis_model: Annotated[
        str | None,
        typer.Option(help="Override CODE_ANALYZER_SYNTHESIS_MODEL for this run."),
    ] = None,
    max_concurrency: Annotated[
        int | None,
        typer.Option(min=1, max=32, help="Override bounded parallel LLM calls."),
    ] = None,
) -> None:
    """Run deterministic parsing, structured LLM extraction, and JSON validation."""

    settings = AnalyzerSettings()
    updates: dict[str, Any] = {}
    if routine_model is not None:
        updates["routine_model"] = routine_model
    if complex_model is not None:
        updates["complex_model"] = complex_model
    if synthesis_model is not None:
        updates["synthesis_model"] = synthesis_model
    if max_concurrency is not None:
        updates["max_concurrency"] = max_concurrency
    if updates:
        settings = settings.model_copy(update=updates)

    try:
        token_counter = TokenCounter(settings.routine_model)
        extractor = LangChainKnowledgeExtractor(settings, token_counter)
        pipeline = AnalysisPipeline(
            settings=settings,
            extractor=extractor,
            cache_directory=cache_directory,
            progress=lambda message: console.print(f"[dim]{message}[/dim]"),
        )
        report = pipeline.run(repository, include_tests=not exclude_tests)
    except ValueError as error:
        console.print(f"[red]Configuration error:[/red] {error}")
        raise typer.Exit(code=2) from error

    _write_json(report.model_dump(mode="json"), output)
    console.print(
        f"Analyzed [bold]{report.metadata.analyzed_file_count}[/bold] files with "
        f"[bold]{report.metadata.cache_hits}[/bold] cache hits."
    )
    console.print(f"Analysis: {output.expanduser().resolve()}")


@app.command("schema")
def export_schema(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination for the JSON Schema document."),
    ] = Path("output/analysis-schema.json"),
) -> None:
    """Export the versioned contract that all final analysis must satisfy."""

    _write_json(AnalysisReport.model_json_schema(), output)
    console.print(f"Schema: {output.expanduser().resolve()}")


@app.command()
def version() -> None:
    """Print the installed analyzer version."""

    console.print(__version__)
