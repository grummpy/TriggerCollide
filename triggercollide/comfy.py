"""Read-only ComfyUI client.

TriggerCollide only ever sends HTTP GET requests to ComfyUI, and only to the
endpoints listed in ``ALLOWED_PATHS``. It never queues a prompt, never touches
the queue, and never uploads anything, so it cannot start a GPU job.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from triggercollide.extract import LoraRecord, build_record, disambiguate

ALLOWED_PATHS = ("/system_stats", "/models/loras", "/object_info/LoraLoader", "/view_metadata/loras")
FORBIDDEN_FRAGMENTS = ("/prompt", "/queue", "/interrupt", "/free", "/upload", "/history", "/api/")


class ComfyError(RuntimeError):
    pass


def normalize_base_url(value: str) -> str:
    value = (value or "").strip().rstrip("/")
    if not value:
        raise ComfyError("Enter your ComfyUI address, for example http://127.0.0.1:8188")
    if "://" not in value:
        value = "http://" + value
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ComfyError(f"Not a usable ComfyUI address: {value}")
    if parsed.path not in ("", "/") or parsed.query:
        raise ComfyError("Use just the host and port, for example http://127.0.0.1:8188")
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 6.0, workers: int = 8):
        self.base_url = normalize_base_url(base_url)
        self.timeout = timeout
        self.workers = max(1, min(int(workers), 16))
        self.requests_made: list[str] = []

    def _get(self, path: str, query: dict[str, str] | None = None):
        if not any(path == allowed or path.startswith(allowed + "?") for allowed in ALLOWED_PATHS):
            raise ComfyError(f"Refusing to call {path}: TriggerCollide only reads from ComfyUI")
        if any(fragment in path for fragment in FORBIDDEN_FRAGMENTS):
            raise ComfyError(f"Refusing to call {path}")
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        self.requests_made.append("GET " + path)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise ComfyError(f"ComfyUI answered {exc.code} for {path}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ComfyError(f"Could not reach ComfyUI at {self.base_url}: {reason}") from exc
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ComfyError(f"ComfyUI sent something that is not JSON for {path}") from exc

    def system_stats(self) -> dict:
        data = self._get("/system_stats")
        if not isinstance(data, dict):
            raise ComfyError("ComfyUI did not return system stats")
        return data

    def list_loras(self) -> list[str]:
        data = self._get("/models/loras")
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
        info = self._get("/object_info/LoraLoader")
        try:
            names = info["LoraLoader"]["input"]["required"]["lora_name"][0]
        except (TypeError, KeyError, IndexError) as exc:
            raise ComfyError("ComfyUI did not list any LoRAs") from exc
        return [n for n in names if isinstance(n, str)]

    def lora_metadata(self, filename: str) -> dict[str, str]:
        data = self._get("/view_metadata/loras", {"filename": filename})
        if not isinstance(data, dict):
            return {}
        return {str(k): v if isinstance(v, str) else json.dumps(v) for k, v in data.items()}


def scan_comfy(base_url: str, timeout: float = 6.0, workers: int = 8) -> tuple[list[LoraRecord], dict]:
    start = time.perf_counter()
    client = ComfyClient(base_url, timeout=timeout, workers=workers)
    stats = client.system_stats()
    names = client.list_loras()
    names = [n for n in names if n.lower().endswith(".safetensors")] + [
        n for n in names if not n.lower().endswith(".safetensors")
    ]

    def one(name: str) -> LoraRecord:
        if not name.lower().endswith(".safetensors"):
            return build_record(name, {}, origin="comfyui")
        try:
            meta = client.lora_metadata(name)
        except ComfyError as exc:
            return LoraRecord(name=PurePosixPath(name.replace("\\", "/")).stem, filename=name, error=str(exc), origin="comfyui")
        return build_record(name.replace("\\", "/"), meta, origin="comfyui")

    with ThreadPoolExecutor(max_workers=client.workers) as pool:
        records = disambiguate(list(pool.map(one, names)))
    version = (stats.get("system") or {}).get("comfyui_version", "")
    info = {
        "source": "comfyui",
        "location": client.base_url,
        "files": len(records),
        "errors": sum(1 for r in records if r.error),
        "no_trigger_info": sum(1 for r in records if not r.error and not r.tags and not r.triggers),
        "comfyui_version": version,
        "requests": len(client.requests_made),
        "methods": sorted({r.split(" ", 1)[0] for r in client.requests_made}),
        "seconds": round(time.perf_counter() - start, 3),
    }
    return records, info
