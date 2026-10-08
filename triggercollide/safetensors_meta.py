"""Read only the JSON header of a .safetensors file. Tensor data is never loaded.

A .safetensors file starts with an 8-byte little-endian length N, followed by N
bytes of JSON. The JSON maps tensor names to dtype/shape/offsets, plus an
optional ``__metadata__`` object of string keys and string values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MAX_HEADER_BYTES = 100 * 1024 * 1024


class HeaderError(ValueError):
    """The file is not a readable .safetensors header."""


@dataclass
class Header:
    metadata: dict[str, str] = field(default_factory=dict)
    tensor_names: list[str] = field(default_factory=list)
    shapes: dict[str, list[int]] = field(default_factory=dict)
    header_bytes: int = 0


def header_length(prefix: bytes) -> int:
    if len(prefix) < 8:
        raise HeaderError("File is shorter than the 8-byte safetensors prefix")
    size = int.from_bytes(prefix[:8], "little")
    if size <= 1 or size > MAX_HEADER_BYTES:
        raise HeaderError(f"Header length {size} is not plausible for a safetensors file")
    return size


def parse_header_bytes(raw: bytes) -> Header:
    """Parse the JSON header (the bytes after the 8-byte prefix)."""
    try:
        data = json.loads(raw.decode("utf-8").rstrip(" \x00"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HeaderError(f"Header is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HeaderError("Header JSON is not an object")
    meta_raw = data.get("__metadata__") or {}
    metadata = {str(k): v if isinstance(v, str) else json.dumps(v) for k, v in meta_raw.items()} if isinstance(
        meta_raw, dict
    ) else {}
    names = [k for k in data if k != "__metadata__"]
    shapes = {
        k: list(v.get("shape") or [])
        for k, v in data.items()
        if k != "__metadata__" and isinstance(v, dict) and isinstance(v.get("shape"), list)
    }
    return Header(metadata=metadata, tensor_names=names, shapes=shapes, header_bytes=len(raw))


def read_header(path: str | Path) -> Header:
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        size = header_length(handle.read(8))
        if size > file_size - 8:
            raise HeaderError("Header length runs past the end of the file")
        raw = handle.read(size)
    if len(raw) != size:
        raise HeaderError("File ended inside the header")
    return parse_header_bytes(raw)


def write_safetensors(path: str | Path, metadata: dict[str, str], tensors: dict[str, list[float]] | None = None) -> Path:
    """Write a tiny but valid float32 .safetensors file. Used for demo data and tests."""
    import struct

    path = Path(path)
    tensors = tensors or {"lora_unet_demo.lora_down.weight": [0.0, 0.0]}
    header: dict[str, object] = {"__metadata__": {k: str(v) for k, v in metadata.items()}}
    blobs = []
    offset = 0
    for name, values in tensors.items():
        blob = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": [len(values)], "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)
    return path
