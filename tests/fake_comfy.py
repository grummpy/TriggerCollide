"""A tiny stand-in for ComfyUI that records every request it gets."""

from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LORAS = {
    "styles/glow_fox_xl.safetensors": {"ss_output_name": "Glow Fox", "ss_base_model_version": "sdxl_base_v1-0",
                                       "modelspec.trigger_phrase": "glowfox"},
    "glow_fox_v2_xl.safetensors": {"ss_output_name": "Glow Fox 2", "ss_base_model_version": "sdxl_base_v1-0",
                                   "modelspec.trigger_phrase": "glowfox"},
    "plain_flux.safetensors": None,
}


class FakeComfy:
    def __init__(self, list_endpoint: bool = True):
        self.requests: list[tuple[str, str]] = []
        self.list_endpoint = list_endpoint
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                owner.requests.append(("GET", self.path))
                url = urllib.parse.urlparse(self.path)
                if url.path == "/system_stats":
                    return self._send(200, {"system": {"comfyui_version": "0.3.test"}, "devices": []})
                if url.path == "/models/loras" and owner.list_endpoint:
                    return self._send(200, list(LORAS))
                if url.path == "/object_info/LoraLoader":
                    return self._send(200, {"LoraLoader": {"input": {"required": {"lora_name": [list(LORAS)]}}}})
                if url.path == "/view_metadata/loras":
                    name = urllib.parse.parse_qs(url.query).get("filename", [""])[0]
                    meta = LORAS.get(name)
                    if meta is None:
                        return self._send(404, {})
                    return self._send(200, meta)
                return self._send(404, {})

            def do_POST(self):  # noqa: N802
                owner.requests.append(("POST", self.path))
                return self._send(500, {"error": "tests forbid POST"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
