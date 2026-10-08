"""Check the LoRAs stacked in a ComfyUI workflow. The workflow is only read, never queued."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath

from triggercollide.collide import analyze
from triggercollide.extract import FAMILY_LABELS, LoraRecord, _family_from_text, normalize

MODEL_EXT = re.compile(r"\.(safetensors|pt|ckpt|bin)$", re.IGNORECASE)
TOTAL_WARN = 1.5
TOTAL_HIGH = 2.5
SINGLE_WARN = 1.5


@dataclass
class StackedLora:
    node: str
    node_type: str
    lora: str
    strength_model: float = 1.0
    strength_clip: float | None = None
    enabled: bool = True


@dataclass
class WorkflowReport:
    stacked: list[StackedLora] = field(default_factory=list)
    checkpoints: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    total_weight: float = 0.0
    missing_triggers: list[dict] = field(default_factory=list)
    format: str = "api"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["issue_counts"] = {
            level: sum(1 for i in self.issues if i["level"] == level) for level in ("high", "medium", "low")
        }
        return data


def _num(value, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_on(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("off", "false", "0", "no", "disabled")
    return bool(value)


def load_workflow(text_or_obj) -> dict:
    data = json.loads(text_or_obj) if isinstance(text_or_obj, (str, bytes)) else text_or_obj
    if not isinstance(data, dict):
        raise ValueError("Workflow JSON must be an object")
    if isinstance(data.get("prompt"), dict) and "nodes" not in data:
        data = data["prompt"]  # a saved /prompt payload
    return data


def extract_stack(workflow: dict) -> tuple[list[StackedLora], list[dict], str]:
    if isinstance(workflow.get("nodes"), list):
        return _extract_ui(workflow) + ("ui",)
    return _extract_api(workflow) + ("api",)


def _extract_api(workflow: dict) -> tuple[list[StackedLora], list[dict]]:
    stacked: list[StackedLora] = []
    checkpoints: list[dict] = []
    for node_id, node in sorted(workflow.items(), key=lambda kv: str(kv[0])):
        if not isinstance(node, dict):
            continue
        ctype = str(node.get("class_type", ""))
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            continue
        mode = node.get("mode")
        node_enabled = mode not in (2, 4)
        for key in ("ckpt_name", "unet_name"):
            if isinstance(inputs.get(key), str):
                checkpoints.append({"node": str(node_id), "name": inputs[key], "family": _family_from_text(inputs[key])})
        stacked.extend(_loras_from_inputs(str(node_id), ctype, inputs, node_enabled))
    return stacked, checkpoints


def _loras_from_inputs(node_id: str, ctype: str, inputs: dict, node_enabled: bool) -> list[StackedLora]:
    out: list[StackedLora] = []
    if isinstance(inputs.get("lora_name"), str) and MODEL_EXT.search(inputs["lora_name"]):
        strength = _num(inputs.get("strength_model", inputs.get("strength", 1.0)))
        clip = inputs.get("strength_clip")
        out.append(
            StackedLora(node_id, ctype, inputs["lora_name"], strength, None if clip is None else _num(clip),
                        node_enabled and strength != 0)
        )
        return out
    # rgthree Power Lora Loader: lora_1 = {"on": true, "lora": "x.safetensors", "strength": 1}
    for value in inputs.values():
        if isinstance(value, dict) and isinstance(value.get("lora"), str) and MODEL_EXT.search(value["lora"]):
            strength = _num(value.get("strength", value.get("strengthTwo", 1.0)))
            out.append(StackedLora(node_id, ctype, value["lora"], strength, None,
                                   node_enabled and _is_on(value.get("on", True)) and strength != 0))
    # Comfyroll and similar stacks: lora_name_1, switch_1, model_weight_1, clip_weight_1
    for key, value in inputs.items():
        match = re.fullmatch(r"lora_name_(\d+)", key)
        if match and isinstance(value, str) and MODEL_EXT.search(value):
            n = match.group(1)
            strength = _num(inputs.get(f"model_weight_{n}", inputs.get(f"lora_wt_{n}", inputs.get(f"strength_{n}", 1.0))))
            switch = inputs.get(f"switch_{n}", True)
            clip = inputs.get(f"clip_weight_{n}")
            out.append(StackedLora(node_id, ctype, value, strength, None if clip is None else _num(clip),
                                   node_enabled and _is_on(switch) and strength != 0))
    if not out and "lora" in ctype.lower():
        for key, value in inputs.items():
            if "lora" in key.lower() and isinstance(value, str) and MODEL_EXT.search(value):
                out.append(StackedLora(node_id, ctype, value, 1.0, None, node_enabled))
    return out


def _extract_ui(workflow: dict) -> tuple[list[StackedLora], list[dict]]:
    """Best effort for the UI-format workflow (the one ComfyUI saves by default)."""
    stacked: list[StackedLora] = []
    checkpoints: list[dict] = []
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict):
            continue
        ctype = str(node.get("type", ""))
        widgets = node.get("widgets_values")
        node_id = str(node.get("id", ""))
        enabled = node.get("mode", 0) not in (2, 4)
        if isinstance(widgets, dict):
            stacked.extend(_loras_from_inputs(node_id, ctype, widgets, enabled))
            continue
        if not isinstance(widgets, list):
            continue
        if ctype in ("CheckpointLoaderSimple", "UNETLoader", "CheckpointLoader") and widgets and isinstance(widgets[0], str):
            checkpoints.append({"node": node_id, "name": widgets[0], "family": _family_from_text(widgets[0])})
            continue
        for idx, value in enumerate(widgets):
            if isinstance(value, dict) and isinstance(value.get("lora"), str) and MODEL_EXT.search(value["lora"]):
                strength = _num(value.get("strength", 1.0))
                stacked.append(StackedLora(node_id, ctype, value["lora"], strength, None,
                                           enabled and _is_on(value.get("on", True)) and strength != 0))
            elif isinstance(value, str) and MODEL_EXT.search(value) and "lora" in ctype.lower():
                rest = [w for w in widgets[idx + 1: idx + 3] if isinstance(w, (int, float)) and not isinstance(w, bool)]
                strength = _num(rest[0]) if rest else 1.0
                clip = _num(rest[1]) if len(rest) > 1 else None
                stacked.append(StackedLora(node_id, ctype, value, strength, clip, enabled and strength != 0))
    return stacked, checkpoints


def _key(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).name.lower()


def _path_key(name: str) -> str:
    return name.replace("\\", "/").strip("/").lower()


def _matcher(library: list[LoraRecord]):
    by_path = {_path_key(r.filename): r for r in library}
    by_base: dict[str, list[LoraRecord]] = {}
    for rec in library:
        by_base.setdefault(_key(rec.filename), []).append(rec)

    def find(name: str) -> LoraRecord | None:
        rec = by_path.get(_path_key(name))
        if rec is not None:
            return rec
        same = by_base.get(_key(name), [])
        return same[0] if len(same) == 1 else None

    return find


def prompt_texts(workflow: dict) -> list[str]:
    texts: list[str] = []
    if isinstance(workflow.get("nodes"), list):
        for node in workflow["nodes"]:
            if isinstance(node, dict) and "TextEncode" in str(node.get("type", "")):
                texts.extend(w for w in node.get("widgets_values") or [] if isinstance(w, str))
        return texts
    for node in workflow.values():
        if isinstance(node, dict) and "TextEncode" in str(node.get("class_type", "")):
            for key, value in (node.get("inputs") or {}).items():
                if isinstance(value, str) and key in ("text", "text_g", "text_l", "clip_l", "t5xxl", "prompt"):
                    texts.append(value)
    return texts


def check_workflow(workflow_input, library: list[LoraRecord], *, total_warn: float = TOTAL_WARN,
                   total_high: float = TOTAL_HIGH) -> WorkflowReport:
    workflow = load_workflow(workflow_input)
    stacked, checkpoints, fmt = extract_stack(workflow)
    report = WorkflowReport(stacked=stacked, checkpoints=checkpoints, format=fmt)
    active = [s for s in stacked if s.enabled]
    find = _matcher(library)
    report.total_weight = round(sum(abs(s.strength_model) for s in active), 3)

    if not stacked:
        report.issues.append({"level": "low", "kind": "no_loras", "message": "No LoRA loader nodes found in this workflow."})
        return report

    matched: list[LoraRecord] = []
    for s in active:
        rec = find(s.lora)
        if rec is None:
            report.issues.append({"level": "medium", "kind": "missing",
                                  "message": f"{s.lora} (node {s.node}) is not in the scanned library, "
                                             "or its file name matches more than one LoRA."})
        else:
            matched.append(rec)
        if abs(s.strength_model) > SINGLE_WARN:
            report.issues.append({"level": "medium", "kind": "strength",
                                  "message": f"{_key(s.lora)} runs at strength {s.strength_model:g}, above {SINGLE_WARN:g}."})

    if report.total_weight > total_high:
        report.issues.append({"level": "high", "kind": "total_weight",
                              "message": f"Total LoRA weight is {report.total_weight:g}. Above {total_high:g} images "
                                         "often fry or lose the base model's look."})
    elif report.total_weight > total_warn:
        report.issues.append({"level": "medium", "kind": "total_weight",
                              "message": f"Total LoRA weight is {report.total_weight:g}, above {total_warn:g}. "
                                         "Consider lowering the strongest LoRA."})

    families = {r.family for r in matched if r.family not in ("unknown",)}
    if len(families) > 1:
        names = ", ".join(sorted(FAMILY_LABELS.get(f, f) for f in families))
        report.issues.append({"level": "high", "kind": "base_mismatch",
                              "message": f"Stacked LoRAs were trained on different base models: {names}."})
    for ck in checkpoints:
        fam = ck["family"]
        if fam in ("unknown", "other"):
            continue
        for rec in matched:
            if rec.family not in ("unknown", "other") and rec.family != fam:
                report.issues.append({"level": "high", "kind": "base_mismatch",
                                      "message": f"{rec.filename} is {rec.family_label} but checkpoint "
                                                 f"{ck['name']} looks like {FAMILY_LABELS.get(fam, fam)}."})

    unique = list({id(r): r for r in matched}.values())
    if len(unique) > 1:
        result = analyze(unique, same_family_only=False)
        for col in result["collisions"]:
            a, b = unique[col.a], unique[col.b]
            first = col.findings[0].detail if col.findings else "overlap"
            report.issues.append({"level": col.severity, "kind": "collision",
                                  "message": f"{a.name} and {b.name}: {first}"
                                             + (f" (+{len(col.findings) - 1} more)" if len(col.findings) > 1 else ""),
                                  "score": round(col.score, 3)})

    prompt = normalize(" , ".join(prompt_texts(workflow)))
    if prompt:
        for rec in unique:
            explicit = [t for t in rec.triggers if not t.source.startswith("inferred")]
            if explicit and not any(t.norm in prompt for t in explicit):
                report.missing_triggers.append({"lora": rec.name, "triggers": [t.text for t in explicit[:5]]})
    order = {"high": 0, "medium": 1, "low": 2}
    report.issues.sort(key=lambda i: order.get(i["level"], 3))
    return report
