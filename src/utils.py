import json
from typing import Any
from pathlib import Path

from .constants import DATA_CHUNKS


def dump_chunks(py: list[dict[str, Any]], md: list[dict[str, Any]]) -> None:

    target_dir = Path(DATA_CHUNKS)
    target_dir.mkdir(parents=True, exist_ok=True)

    with (target_dir / "py_chunks.json").open("w") as f:
        json.dump(py, f, indent=2)

    with (target_dir / "md_chunks.json").open("w") as f:
        json.dump(md, f, indent=2)


def reload_chunks() -> list[dict[str, Any]]:
    target = Path(DATA_CHUNKS)
    with Path(target / "py_chunks.json").open("r") as f:
        py_chunks = json.load(f)

    with Path(target / "md_chunks.json").open("r") as f:
        md_chunks = json.load(f)

    chunks_reloaded = py_chunks + md_chunks
    return chunks_reloaded


def load_files(dir: str) -> list[tuple[str, str]]:
    res = []
    dir_path = Path(dir)
    extension = {".py", ".md"}

    selected_files = [str(f) for f in dir_path.rglob("*")
                      if f.suffix in extension]
    for file in selected_files:
        with open(file, "r") as content:
            res.append((file, content.read()))
    return res
