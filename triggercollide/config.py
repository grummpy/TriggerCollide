"""Optional local config: config.toml next to the app (gitignored) or environment variables."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def load_config(path: Path | None = None) -> dict:
    path = path or Path.cwd() / "config.toml"
    data: dict = {}
    if path.is_file():
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    comfy = os.environ.get("TRIGGERCOLLIDE_COMFY_URL") or data.get("comfy_url", "")
    folder = os.environ.get("TRIGGERCOLLIDE_LORA_FOLDER") or data.get("lora_folder", "")
    return {"comfy_url": str(comfy), "lora_folder": str(folder)}
