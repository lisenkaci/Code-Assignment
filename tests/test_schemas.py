from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from codebase_analyzer.schemas import (
    AnalysisMetadata,
    AnalysisReport,
    ModelSelection,
    ProjectOverview,
)


def build_report() -> AnalysisReport:
    return AnalysisReport(
        metadata=AnalysisMetadata(
            generated_at=datetime.now(UTC),
            analyzer_version="0.1.0",
            repository_name="example",
            repository_commit="abc1234",
            models=ModelSelection(
                routine_extraction="test-routine-model",
                complex_extraction="test-complex-model",
                synthesis="test-synthesis-model",
            ),
            prompt_version="test.1",
            parser="test-parser",
            tokenizer="test-tokenizer",
            analyzed_file_count=0,
            skipped_file_count=0,
        ),
        overview=ProjectOverview(
            name="Example",
            purpose="Exercise the output schema.",
            architecture="Single module.",
            complexity_summary="No source files.",
        ),
    )


def test_report_round_trips_through_json() -> None:
    report = build_report()

    restored = AnalysisReport.model_validate_json(report.model_dump_json())

    assert restored == report
    assert restored.metadata.schema_version == "1.1"


def test_schema_rejects_unexpected_fields() -> None:
    payload = build_report().model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AnalysisReport.model_validate(payload)
