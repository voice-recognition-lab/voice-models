"""Shared JSONL reading and atomic text and JSON writing."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator


def read_jsonl(path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield JSON objects with their source path and line number."""
    path = Path(path)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{location}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{location}: expected a JSON object")
            yield location, row


def write_atomic(path, text) -> None:
    """Replace a UTF-8 text file after its complete contents are written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(text)
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value, *, ensure_ascii=False) -> None:
    """Atomically write formatted JSON with a trailing newline."""
    text = json.dumps(value, ensure_ascii=ensure_ascii, indent=2) + "\n"
    write_atomic(path, text)
