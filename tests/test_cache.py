from __future__ import annotations

from pathlib import Path

import pytest

from codebase_analyzer.cache import JsonModelCache
from codebase_analyzer.llm_schemas import ChunkInterpretation


def test_cache_round_trips_validated_models(tmp_path: Path) -> None:
    cache = JsonModelCache(tmp_path / "cache")
    key = cache.build_key("model", "prompt", "content")
    value = ChunkInterpretation(file_path="Example.java", summary="Example source.")

    assert cache.get(key, ChunkInterpretation) is None
    cache.put(key, value)

    assert cache.get(key, ChunkInterpretation) == value


def test_cache_ignores_invalid_or_corrupt_entries(tmp_path: Path) -> None:
    cache = JsonModelCache(tmp_path)
    key = cache.build_key("corrupt")
    (tmp_path / f"{key}.json").write_text("{not-json", encoding="utf-8")

    assert cache.get(key, ChunkInterpretation) is None


def test_cache_rejects_unsafe_keys(tmp_path: Path) -> None:
    cache = JsonModelCache(tmp_path)

    with pytest.raises(ValueError, match="hexadecimal"):
        cache.get("../escape", ChunkInterpretation)
