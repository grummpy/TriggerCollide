"""Build a LoRA library from a local folder of .safetensors files (headers only)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from triggercollide.extract import LoraRecord, build_record, disambiguate, read_sidecar_triggers
from triggercollide.safetensors_meta import HeaderError, read_header

MODEL_SUFFIXES = (".safetensors",)


def find_models(folder: str | Path, recursive: bool = True, limit: int = 20000) -> list[Path]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")
    found: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                if name.lower().endswith(MODEL_SUFFIXES):
                    found.append(Path(dirpath) / name)
                    if len(found) >= limit:
                        return found
    else:
        found = sorted(p for p in root.iterdir() if p.is_file() and p.name.lower().endswith(MODEL_SUFFIXES))
    return found


def scan_folder(folder: str | Path, recursive: bool = True) -> tuple[list[LoraRecord], dict]:
    start = time.perf_counter()
    root = Path(folder).expanduser()
    records: list[LoraRecord] = []
    for path in find_models(root, recursive=recursive):
        rel = path.relative_to(root).as_posix()
        try:
            header = read_header(path)
        except (HeaderError, OSError) as exc:
            records.append(LoraRecord(name=path.stem, filename=rel, error=str(exc), origin="folder"))
            continue
        records.append(
            build_record(
                rel,
                header.metadata,
                tensor_names=header.tensor_names,
                shapes=header.shapes,
                sidecar=read_sidecar_triggers(path),
                origin="folder",
            )
        )
    disambiguate(records)
    stats = {
        "source": "folder",
        "location": str(root),
        "files": len(records),
        "errors": sum(1 for r in records if r.error),
        "seconds": round(time.perf_counter() - start, 3),
    }
    return records, stats
