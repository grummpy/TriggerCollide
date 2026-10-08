"""Local web UI. Served on 127.0.0.1 only (see server.py)."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from triggercollide import __version__
from triggercollide.comfy import ComfyError, scan_comfy
from triggercollide.config import load_config
from triggercollide.demo import demo_workflow, write_demo_library
from triggercollide.library import scan_folder
from triggercollide.report import build_result, collisions_csv
from triggercollide.workflow import check_workflow


def web_dir() -> Path:
    return Path(__file__).resolve().parent / "web"


def create_app(data_dir: Path | None = None, scan_comfy_fn=scan_comfy) -> Flask:
    folder = Path(data_dir) if data_dir is not None else Path.cwd() / "data"
    folder.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__, static_folder=str(web_dir()), static_url_path="/static")
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
    scans: dict[str, dict] = {}
    lock = threading.Lock()

    def remember(records, stats, same_family_only=True) -> dict:
        payload = build_result(records, stats, same_family_only=same_family_only)
        scan_id = uuid.uuid4().hex[:12]
        payload["scan_id"] = scan_id
        with lock:
            scans[scan_id] = {"records": records, "payload": payload}
            while len(scans) > 8:
                scans.pop(next(iter(scans)))
        return payload

    @app.get("/")
    def index():
        return send_from_directory(web_dir(), "index.html")

    @app.get("/favicon.ico")
    def favicon():
        return send_from_directory(web_dir(), "favicon.ico")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "version": __version__})

    @app.get("/api/meta")
    def meta():
        cfg = load_config()
        return jsonify({"version": __version__, "comfy_url": cfg["comfy_url"], "lora_folder": cfg["lora_folder"],
                        "read_only": True})

    @app.post("/api/scan/folder")
    def scan_folder_route():
        body = request.get_json(force=True, silent=True) or {}
        path = str(body.get("path", "")).strip()
        if not path:
            return jsonify({"error": "Enter the folder that holds your LoRA .safetensors files."}), 400
        try:
            records, stats = scan_folder(path, recursive=bool(body.get("recursive", True)))
        except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(remember(records, stats, bool(body.get("same_family_only", True))))

    @app.post("/api/scan/comfy")
    def scan_comfy_route():
        body = request.get_json(force=True, silent=True) or {}
        try:
            records, stats = scan_comfy_fn(str(body.get("url", "")), timeout=float(body.get("timeout", 6)))
        except ComfyError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(remember(records, stats, bool(body.get("same_family_only", True))))

    @app.post("/api/scan/demo")
    def scan_demo_route():
        body = request.get_json(force=True, silent=True) or {}
        demo_dir = write_demo_library(folder / "demo-library")
        records, stats = scan_folder(demo_dir)
        stats["source"] = "demo"
        return jsonify(remember(records, stats, bool(body.get("same_family_only", True))))

    @app.get("/api/demo/workflow")
    def demo_workflow_route():
        return jsonify(demo_workflow())

    @app.post("/api/workflow/check")
    def workflow_route():
        body = request.get_json(force=True, silent=True) or {}
        scan = scans.get(str(body.get("scan_id", "")))
        if scan is None:
            return jsonify({"error": "Scan a library first, then check a workflow against it."}), 400
        raw = body.get("workflow")
        try:
            report = check_workflow(raw if isinstance(raw, (dict, str)) else "", scan["records"])
        except (ValueError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"That is not a ComfyUI workflow JSON: {exc}"}), 400
        return jsonify(report.to_dict())

    @app.get("/api/scan/<scan_id>/collisions.csv")
    def export_csv(scan_id: str):
        scan = scans.get(scan_id)
        if scan is None:
            abort(404)
        return Response(collisions_csv(scan["payload"]), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=triggercollide-collisions.csv"})

    @app.get("/api/scan/<scan_id>.json")
    def export_json(scan_id: str):
        scan = scans.get(scan_id)
        if scan is None:
            abort(404)
        return Response(json.dumps(scan["payload"], indent=2), mimetype="application/json",
                        headers={"Content-Disposition": "attachment; filename=triggercollide-scan.json"})

    return app
