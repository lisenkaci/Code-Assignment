"""Content-addressed cache for validated structured LLM responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class JsonModelCache:
    """Persist only schema-validated models; never cache credentials."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.expanduser().resolve()

    @staticmethod
    def build_key(*parts: str) -> str:
        digest = hashlib.sha256()
        for part in parts:
            encoded = part.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    def get(self, key: str, model_type: type[ModelT]) -> ModelT | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return model_type.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValidationError, json.JSONDecodeError):
            return None

    def put(self, key: str, model: BaseModel) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            model.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _path(self, key: str) -> Path:
        if not key or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("cache key must be a lowercase hexadecimal digest")
        return self.directory / f"{key}.json"
