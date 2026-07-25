from __future__ import annotations

from codebase_analyzer.model_routing import ModelRouter, ModelTier
from codebase_analyzer.parsed_models import ParsedMethod


def build_router() -> ModelRouter:
    return ModelRouter(
        routine_model="routine-model",
        complex_model="complex-model",
    )


def method(name: str, complexity: int) -> ParsedMethod:
    return ParsedMethod(
        name=name,
        qualified_name=f"Example.{name}",
        signature=f"{name}()",
        return_type="void",
        parameters=(),
        modifiers=("public",),
        annotations=(),
        start_line=1,
        end_line=2,
        source="",
        lines_of_code=2,
        cyclomatic_complexity=complexity,
    )


def test_routine_file_uses_routine_model() -> None:
    route = build_router().route(
        path="src/main/java/example/Dto.java",
        token_count=500,
        has_parser_warnings=False,
        methods=(method("getName", 1), method("equals", 16)),
    )

    assert route.tier is ModelTier.ROUTINE
    assert route.model == "routine-model"
    assert route.reasons == ("default_routine",)


def test_semantically_sensitive_path_uses_complex_model() -> None:
    route = build_router().route(
        path="src/main/java/example/repository/custom/FilmRepositoryImpl.java",
        token_count=500,
        has_parser_warnings=False,
        methods=(),
    )

    assert route.tier is ModelTier.COMPLEX
    assert route.model == "complex-model"
    assert "custom_query_implementation" in route.reasons


def test_business_control_flow_uses_complex_model() -> None:
    route = build_router().route(
        path="src/main/java/example/Service.java",
        token_count=500,
        has_parser_warnings=False,
        methods=(method("calculateAvailability", 6),),
    )

    assert route.tier is ModelTier.COMPLEX
    assert "business_control_flow" in route.reasons
