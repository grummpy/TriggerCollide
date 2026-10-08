"""Command line: open the UI, scan a folder or ComfyUI, or check a workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from triggercollide import __version__
from triggercollide.comfy import ComfyError, scan_comfy
from triggercollide.config import load_config
from triggercollide.library import scan_folder
from triggercollide.report import build_result, collisions_csv, counts_only
from triggercollide.server import serve
from triggercollide.workflow import check_workflow

COMMANDS = {"serve", "scan", "check", "-h", "--help", "--version"}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        argv = ["serve", *argv]
    parser = argparse.ArgumentParser(prog="triggercollide",
                                     description="Find LoRA trigger words that overlap or clash.")
    parser.add_argument("--version", action="version", version=f"triggercollide {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("serve", help="Open the local UI")
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--no-browser", action="store_true")
    sp.add_argument("--data-dir", type=Path, default=None)

    for name, helptext in (("scan", "Scan a LoRA library"), ("check", "Check a ComfyUI workflow's LoRA stack")):
        p = sub.add_parser(name, help=helptext)
        src = p.add_mutually_exclusive_group()
        src.add_argument("--folder", type=Path, help="Folder of .safetensors LoRAs (headers only are read)")
        src.add_argument("--comfy", help="ComfyUI address, for example http://127.0.0.1:8188 (GET requests only)")
        p.add_argument("--timeout", type=float, default=6.0)
        p.add_argument("--all-families", action="store_true", help="Also compare LoRAs from different base models")
        p.add_argument("--out", type=Path, help="Write the full result as JSON")
        p.add_argument("--counts-only", action="store_true", help="Print counts only, no names or trigger words")
        if name == "scan":
            p.add_argument("--csv", type=Path, help="Write collisions as CSV")
        else:
            p.add_argument("--workflow", type=Path, required=True, help="ComfyUI workflow JSON (API or UI format)")

    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(port=args.port, open_browser=not args.no_browser, data_dir=args.data_dir)
        return 0
    try:
        records, stats = _load(args)
    except (ComfyError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if args.command == "scan":
        payload = build_result(records, stats, same_family_only=not args.all_families)
        if args.out:
            args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if args.csv:
            args.csv.write_text(collisions_csv(payload), encoding="utf-8")
        if args.counts_only:
            print(json.dumps(counts_only(payload), indent=2))
        else:
            _print_scan(payload)
        return 0
    report = check_workflow(args.workflow.read_text(encoding="utf-8"), records).to_dict()
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.counts_only:
        print(json.dumps({"stacked": len(report["stacked"]), "total_weight": report["total_weight"],
                          "issues": report["issue_counts"]}, indent=2))
    else:
        print(f"{len(report['stacked'])} LoRAs in the workflow, total weight {report['total_weight']:g}")
        for issue in report["issues"]:
            print(f"[{issue['level'].upper():6}] {issue['message']}")
        for miss in report["missing_triggers"]:
            print(f"[hint  ] {miss['lora']}: prompt has none of {', '.join(miss['triggers'])}")
    return 1 if report["issue_counts"]["high"] else 0


def _load(args):
    if args.folder:
        return scan_folder(args.folder)
    cfg = load_config()
    url = args.comfy or cfg["comfy_url"]
    if not url and cfg["lora_folder"]:
        return scan_folder(cfg["lora_folder"])
    if not url:
        raise ComfyError("Pass --folder or --comfy (or set comfy_url in config.toml)")
    return scan_comfy(url, timeout=args.timeout)


def _print_scan(payload: dict) -> None:
    s = payload["summary"]
    print(f"Scanned {s['loras']} LoRAs from {s['source']} in {s.get('seconds', 0)}s; "
          f"{s['collisions']} collisions (high {s['severity']['high']}, medium {s['severity']['medium']}, "
          f"low {s['severity']['low']}).")
    for col in payload["collisions"][:25]:
        first = col["findings"][0]["detail"] if col["findings"] else ""
        print(f"[{col['severity'].upper():6}] {col['score']:.2f}  {col['a_name']}  <->  {col['b_name']}: {first}")
    if len(payload["collisions"]) > 25:
        print(f"... and {len(payload['collisions']) - 25} more. Use --csv or the UI for the full list.")
