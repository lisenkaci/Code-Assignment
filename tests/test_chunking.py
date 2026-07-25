from __future__ import annotations

import hashlib

import pytest

from codebase_analyzer.chunking import ChunkPlanner
from codebase_analyzer.java_parser import JavaParser
from codebase_analyzer.tokenization import TokenCounter


def parse(source: str):
    encoded = source.encode()
    return JavaParser().parse_source(
        "Example.java",
        encoded,
        hashlib.sha256(encoded).hexdigest(),
    )


def test_small_file_remains_one_chunk() -> None:
    parsed = parse("class Example { void run() {} }\n")
    counter = TokenCounter("unknown-test-model")

    chunks = ChunkPlanner(counter, chunk_token_budget=100).plan_file(parsed)

    assert len(chunks) == 1
    assert chunks[0].content == parsed.source
    assert chunks[0].token_count <= 100


def test_oversized_method_uses_bounded_fallback_chunks() -> None:
    statements = "\n".join(f"        int value{i} = {i};" for i in range(40))
    parsed = parse(
        "class Example {\n"
        "    void largeMethod() {\n"
        f"{statements}\n"
        "    }\n"
        "}\n"
    )
    counter = TokenCounter("unknown-test-model")

    chunks = ChunkPlanner(
        counter,
        chunk_token_budget=50,
        fallback_overlap_lines=2,
    ).plan_file(parsed)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 50 for chunk in chunks)
    assert [chunk.part_index for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.total_parts == len(chunks) for chunk in chunks)


def test_token_split_validates_budget_and_overlap() -> None:
    counter = TokenCounter("unknown-test-model")

    slices = counter.split("one two three four five six", max_tokens=3, overlap_tokens=1)

    assert len(slices) >= 2
    assert all(item.token_count <= 3 for item in slices)
    with pytest.raises(ValueError, match="between zero"):
        counter.split("text", max_tokens=3, overlap_tokens=3)
