"""Serialize scan results for the UI, JSON, and CSV."""

from __future__ import annotations

import csv
import io

from triggercollide.collide import analyze, matrix


def build_result(records, stats, *, same_family_only: bool = True) -> dict:
    result = analyze(records, same_family_only=same_family_only)
    involvement = result["involvement"]
    library = []
    for idx, rec in enumerate(records):
        row = rec.to_dict()
        row["index"] = idx
        row["collisions"] = involvement.get(idx, 0)
        library.append(row)
    collisions = [c.to_dict(records) for c in result["collisions"]]
    summary = {
        **stats,
        "loras": len(records),
        "with_triggers": sum(1 for r in records if r.triggers),
        "with_tags": sum(1 for r in records if r.tags),
        "collisions": len(collisions),
        "severity": {k: result["severity_counts"].get(k, 0) for k in ("high", "medium", "low")},
        "families": result["families"],
        "skipped_cross_family_pairs": result["skipped_cross_family_pairs"],
        "common_tags_ignored": len(result["common_tags"]),
        "loras_in_collisions": len(involvement),
    }
    return {
        "summary": summary,
        "library": library,
        "collisions": collisions,
        "matrix": matrix(records, result),
        "common_tags": result["common_tags"],
    }


def collisions_csv(payload: dict) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["severity", "score", "lora_a", "file_a", "lora_b", "file_b", "findings"])
    for col in payload["collisions"]:
        writer.writerow([col["severity"], col["score"], col["a_name"], col["a_file"], col["b_name"], col["b_file"],
                         " | ".join(f["detail"] for f in col["findings"])])
    return buffer.getvalue()


def counts_only(payload: dict) -> dict:
    """Summary without any names, triggers, or tags. Safe to paste into a bug report."""
    s = payload["summary"]
    keys = ("source", "loras", "files", "errors", "no_trigger_info", "with_triggers", "with_tags", "collisions",
            "severity", "families", "skipped_cross_family_pairs", "common_tags_ignored", "loras_in_collisions",
            "seconds", "requests", "methods", "comfyui_version")
    return {k: s[k] for k in keys if k in s}
