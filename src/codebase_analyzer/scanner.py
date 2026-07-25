"""Deterministic, bounded repository file discovery."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class FileCategory(StrEnum):
    SOURCE = "source"
    CONTEXT = "context"


DEFAULT_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".gradle",
        ".idea",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "generated",
        "node_modules",
        "out",
        "target",
    }
)
DEFAULT_SOURCE_EXTENSIONS = frozenset(
    {".java", ".kt", ".py", ".sql", ".js", ".jsx", ".ts", ".tsx"}
)
DEFAULT_CONTEXT_EXTENSIONS = frozenset(
    {".gradle", ".kts", ".md", ".properties", ".toml", ".yaml", ".yml"}
)
DEFAULT_CONTEXT_FILENAMES = frozenset(
    {"dockerfile", "makefile", "pom.xml", "settings.xml"}
)
LANGUAGE_BY_EXTENSION = {
    ".gradle": "Gradle",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript JSX",
    ".kt": "Kotlin",
    ".kts": "Kotlin Script",
    ".md": "Markdown",
    ".properties": "Properties",
    ".py": "Python",
    ".sql": "SQL",
    ".toml": "TOML",
    ".ts": "TypeScript",
    ".tsx": "TypeScript JSX",
    ".yaml": "YAML",
    ".yml": "YAML",
}


@dataclass(frozen=True, slots=True)
class ScanPolicy:
    """Controls which repository files are eligible for analysis."""

    source_extensions: frozenset[str] = DEFAULT_SOURCE_EXTENSIONS
    context_extensions: frozenset[str] = DEFAULT_CONTEXT_EXTENSIONS
    context_filenames: frozenset[str] = DEFAULT_CONTEXT_FILENAMES
    excluded_directories: frozenset[str] = DEFAULT_EXCLUDED_DIRECTORIES
    max_file_bytes: int = 1_000_000
    include_tests: bool = True

    def __post_init__(self) -> None:
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")


@dataclass(frozen=True, slots=True)
class FileRecord:
    """Stable metadata for one file selected for analysis."""

    path: str
    category: FileCategory
    language: str
    byte_size: int
    line_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Repository manifest and explicit skip accounting."""

    repository_root: str
    files: tuple[FileRecord, ...]
    skipped_by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def selected_file_count(self) -> int:
        return len(self.files)

    @property
    def skipped_file_count(self) -> int:
        return sum(self.skipped_by_reason.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_root": self.repository_root,
            "selected_file_count": self.selected_file_count,
            "skipped_file_count": self.skipped_file_count,
            "skipped_by_reason": dict(sorted(self.skipped_by_reason.items())),
            "files": [file.to_dict() for file in self.files],
        }


class RepositoryScanner:
    """Walk a repository without following symlinks or generated directories."""

    def __init__(self, policy: ScanPolicy | None = None) -> None:
        self.policy = policy or ScanPolicy()

    def scan(self, repository_root: Path) -> ScanResult:
        root = repository_root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repository path is not a directory: {root}")

        records: list[FileRecord] = []
        skipped: dict[str, int] = {}

        for current_root, directories, filenames in os.walk(root, followlinks=False):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in self.policy.excluded_directories
            )
            current_path = Path(current_root)

            for filename in sorted(filenames):
                path = current_path / filename
                relative_path = path.relative_to(root)
                reason = self._skip_reason(path, relative_path)
                if reason is not None:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue

                records.append(self._build_record(path, relative_path))

        records.sort(key=lambda record: record.path)
        return ScanResult(
            repository_root=str(root),
            files=tuple(records),
            skipped_by_reason=skipped,
        )

    def _skip_reason(self, path: Path, relative_path: Path) -> str | None:
        if path.is_symlink():
            return "symlink"
        if not path.is_file():
            return "not_regular_file"
        if not self.policy.include_tests and self._is_test_path(relative_path):
            return "tests_excluded"
        if self._category_for(path) is None:
            return "unsupported_file_type"
        if path.stat().st_size > self.policy.max_file_bytes:
            return "file_too_large"
        return None

    def _category_for(self, path: Path) -> FileCategory | None:
        suffix = path.suffix.lower()
        if suffix in self.policy.source_extensions:
            return FileCategory.SOURCE
        if (
            suffix in self.policy.context_extensions
            or path.name.lower() in self.policy.context_filenames
        ):
            return FileCategory.CONTEXT
        return None

    def _build_record(self, path: Path, relative_path: Path) -> FileRecord:
        data = path.read_bytes()
        suffix = path.suffix.lower()
        category = self._category_for(path)
        if category is None:
            raise RuntimeError(f"unsupported file reached record creation: {path}")

        return FileRecord(
            path=relative_path.as_posix(),
            category=category,
            language=LANGUAGE_BY_EXTENSION.get(suffix, "Text"),
            byte_size=len(data),
            line_count=self._line_count(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    @staticmethod
    def _line_count(data: bytes) -> int:
        if not data:
            return 0
        return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)

    @staticmethod
    def _is_test_path(relative_path: Path) -> bool:
        lowered_parts = {part.lower() for part in relative_path.parts}
        return bool(lowered_parts.intersection({"test", "tests", "__tests__"}))
