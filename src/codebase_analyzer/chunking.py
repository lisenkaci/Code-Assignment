"""Token-bounded source chunks that prefer Java method boundaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from codebase_analyzer.parsed_models import ParsedJavaFile, ParsedMethod
from codebase_analyzer.tokenization import TokenCounter


@dataclass(frozen=True, slots=True)
class AnalysisChunk:
    chunk_id: str
    file_path: str
    part_index: int
    total_parts: int
    start_line: int
    end_line: int
    content: str
    token_count: int


@dataclass(frozen=True, slots=True)
class _Segment:
    start_line: int
    end_line: int
    content: str
    token_count: int
    overlaps_previous: bool = False


class ChunkPlanner:
    """Keep complete files or methods together whenever the token budget allows."""

    def __init__(
        self,
        token_counter: TokenCounter,
        *,
        chunk_token_budget: int,
        fallback_overlap_lines: int = 3,
    ) -> None:
        if chunk_token_budget <= 0:
            raise ValueError("chunk_token_budget must be positive")
        if fallback_overlap_lines < 0:
            raise ValueError("fallback_overlap_lines cannot be negative")
        self.token_counter = token_counter
        self.chunk_token_budget = chunk_token_budget
        self.fallback_overlap_lines = fallback_overlap_lines

    def plan_file(self, parsed_file: ParsedJavaFile) -> tuple[AnalysisChunk, ...]:
        source = parsed_file.source
        token_count = self.token_counter.count(source)
        line_count = max(len(source.splitlines()), 1)
        if token_count <= self.chunk_token_budget:
            return self._finalize(
                parsed_file,
                (
                    _Segment(
                        start_line=1,
                        end_line=line_count,
                        content=source,
                        token_count=token_count,
                    ),
                ),
            )

        methods = sorted(
            (method for parsed_type in parsed_file.types for method in parsed_type.methods),
            key=lambda method: (method.start_line, method.end_line),
        )
        semantic_segments = self._semantic_segments(source, methods)
        bounded_segments = tuple(
            bounded
            for segment in semantic_segments
            for bounded in self._bound_segment(segment)
        )
        packed_segments = self._pack_segments(source, bounded_segments)
        return self._finalize(parsed_file, packed_segments)

    def plan_text(
        self,
        *,
        file_path: str,
        source: str,
        sha256: str,
    ) -> tuple[AnalysisChunk, ...]:
        """Plan non-Java context using the same hard token ceiling."""

        document = ParsedJavaFile(
            path=file_path,
            package=None,
            imports=(),
            types=(),
            source=source,
            sha256=sha256,
            has_parser_warnings=False,
        )
        return self.plan_file(document)

    def _semantic_segments(
        self,
        source: str,
        methods: list[ParsedMethod],
    ) -> tuple[_Segment, ...]:
        lines = source.splitlines(keepends=True)
        segments: list[_Segment] = []
        cursor = 1

        for method in methods:
            if method.start_line < cursor:
                continue
            if cursor < method.start_line:
                segments.append(self._make_segment(lines, cursor, method.start_line - 1))
            segments.append(self._make_segment(lines, method.start_line, method.end_line))
            cursor = method.end_line + 1

        if cursor <= len(lines):
            segments.append(self._make_segment(lines, cursor, len(lines)))
        if not segments:
            segments.append(self._make_segment(lines, 1, max(len(lines), 1)))
        return tuple(segments)

    def _bound_segment(self, segment: _Segment) -> tuple[_Segment, ...]:
        if segment.token_count <= self.chunk_token_budget:
            return (segment,)

        lines = segment.content.splitlines(keepends=True)
        bounded: list[_Segment] = []
        index = 0
        while index < len(lines):
            end = index
            selected = ""
            while end < len(lines):
                candidate = selected + lines[end]
                candidate_tokens = self.token_counter.count(candidate)
                if candidate_tokens > self.chunk_token_budget:
                    break
                selected = candidate
                end += 1
                if candidate_tokens == self.chunk_token_budget:
                    break

            if end == index:
                token_slices = self.token_counter.split(
                    lines[index],
                    max_tokens=self.chunk_token_budget,
                    overlap_tokens=min(50, self.chunk_token_budget // 10),
                )
                for slice_index, token_slice in enumerate(token_slices):
                    bounded.append(
                        _Segment(
                            start_line=segment.start_line + index,
                            end_line=segment.start_line + index,
                            content=token_slice.text,
                            token_count=token_slice.token_count,
                            overlaps_previous=slice_index > 0,
                        )
                    )
                index += 1
                continue

            bounded.append(
                _Segment(
                    start_line=segment.start_line + index,
                    end_line=segment.start_line + end - 1,
                    content=selected,
                    token_count=self.token_counter.count(selected),
                    overlaps_previous=bool(bounded and index > 0),
                )
            )
            if end >= len(lines):
                break
            next_index = max(end - self.fallback_overlap_lines, index + 1)
            index = next_index

        return tuple(bounded)

    def _pack_segments(
        self,
        source: str,
        segments: tuple[_Segment, ...],
    ) -> tuple[_Segment, ...]:
        lines = source.splitlines(keepends=True)
        packed: list[_Segment] = []
        current: _Segment | None = None

        for segment in segments:
            if current is None:
                current = segment
                continue
            if segment.overlaps_previous or segment.start_line <= current.end_line:
                packed.append(current)
                current = segment
                continue

            combined_content = "".join(lines[current.start_line - 1 : segment.end_line])
            combined_tokens = self.token_counter.count(combined_content)
            if combined_tokens <= self.chunk_token_budget:
                current = _Segment(
                    start_line=current.start_line,
                    end_line=segment.end_line,
                    content=combined_content,
                    token_count=combined_tokens,
                )
            else:
                packed.append(current)
                current = segment

        if current is not None:
            packed.append(current)
        return tuple(packed)

    def _make_segment(self, lines: list[str], start_line: int, end_line: int) -> _Segment:
        content = "".join(lines[start_line - 1 : end_line])
        return _Segment(
            start_line=start_line,
            end_line=end_line,
            content=content,
            token_count=self.token_counter.count(content),
        )

    def _finalize(
        self,
        parsed_file: ParsedJavaFile,
        segments: tuple[_Segment, ...],
    ) -> tuple[AnalysisChunk, ...]:
        total = len(segments)
        chunks: list[AnalysisChunk] = []
        for index, segment in enumerate(segments, start=1):
            identity = (
                f"{parsed_file.sha256}:{segment.start_line}:{segment.end_line}:"
                f"{hashlib.sha256(segment.content.encode('utf-8')).hexdigest()}"
            )
            chunks.append(
                AnalysisChunk(
                    chunk_id=hashlib.sha256(identity.encode()).hexdigest()[:20],
                    file_path=parsed_file.path,
                    part_index=index,
                    total_parts=total,
                    start_line=segment.start_line,
                    end_line=segment.end_line,
                    content=segment.content,
                    token_count=segment.token_count,
                )
            )
        return tuple(chunks)
