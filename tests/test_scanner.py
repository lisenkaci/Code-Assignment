from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from codebase_analyzer.scanner import FileCategory, RepositoryScanner, ScanPolicy


def write_file(root: Path, relative_path: str, content: bytes) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_scan_selects_supported_files_in_stable_order(tmp_path: Path) -> None:
    java_content = b"class Main {\n  void run() {}\n}\n"
    write_file(tmp_path, "src/main/java/Main.java", java_content)
    write_file(tmp_path, "src/test/java/MainTest.java", b"class MainTest {}\n")
    write_file(tmp_path, "README.md", b"# Example\n")
    write_file(tmp_path, "asset.bin", b"\x00\x01")
    write_file(tmp_path, "large.sql", b"x" * 101)
    write_file(tmp_path, "build/generated/Generated.java", b"class Generated {}\n")

    result = RepositoryScanner(ScanPolicy(max_file_bytes=100)).scan(tmp_path)

    assert [record.path for record in result.files] == [
        "README.md",
        "src/main/java/Main.java",
        "src/test/java/MainTest.java",
    ]
    assert result.files[0].category is FileCategory.CONTEXT
    assert result.files[1].category is FileCategory.SOURCE
    assert result.files[1].line_count == 3
    assert result.files[1].sha256 == hashlib.sha256(java_content).hexdigest()
    assert result.skipped_by_reason == {
        "file_too_large": 1,
        "unsupported_file_type": 1,
    }


def test_scan_can_exclude_tests(tmp_path: Path) -> None:
    write_file(tmp_path, "src/main/java/Main.java", b"class Main {}\n")
    write_file(tmp_path, "src/test/java/MainTest.java", b"class MainTest {}\n")

    result = RepositoryScanner(ScanPolicy(include_tests=False)).scan(tmp_path)

    assert [record.path for record in result.files] == ["src/main/java/Main.java"]
    assert result.skipped_by_reason == {"tests_excluded": 1}


def test_scan_rejects_missing_repository(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(ValueError, match="not a directory"):
        RepositoryScanner().scan(missing)


def test_scan_policy_rejects_non_positive_file_limit() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ScanPolicy(max_file_bytes=0)
