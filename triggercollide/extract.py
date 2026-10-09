"""Turn safetensors metadata (and optional sidecar files) into a LoRA record."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

FAMILY_LABELS = {
    "sd15": "SD 1.5",
    "sd2": "SD 2.x",
    "sdxl": "SDXL",
    "sd3": "SD 3",
    "flux": "Flux",
    "other": "Other",
    "unknown": "Unknown",
}

SIDECAR_SUFFIXES = (".txt", ".json", ".civitai.info", ".metadata.json", ".info")
SIDECAR_JSON_KEYS = ("trainedWords", "trained_words", "activation text", "activation_text", "trigger_words",
                     "triggerWords", "triggers", "trigger", "trigger_phrase")

_DIR_PREFIX = re.compile(r"^\s*\d+_(.+?)\s*$")
_WEIGHT_SYNTAX = re.compile(r"^\((.*?)(?::\s*[-\d.]+)?\)$")
_HASHLIKE = re.compile(r"^[0-9a-f]{16,}$|^[0-9a-f-]{32,}$", re.IGNORECASE)
_LORA_TAG = re.compile(r"<lora:[^>]*>", re.IGNORECASE)


@dataclass
class Trigger:
    text: str
    norm: str
    source: str


@dataclass
class LoraRecord:
    name: str
    filename: str
    trained_name: str = ""
    family: str = "unknown"
    family_source: str = ""
    base_detail: str = ""
    triggers: list[Trigger] = field(default_factory=list)
    tags: dict[str, int] = field(default_factory=dict)
    image_count: int | None = None
    origin: str = "folder"
    error: str = ""

    @property
    def family_label(self) -> str:
        return FAMILY_LABELS.get(self.family, self.family)

    def top_tags(self, limit: int = 25) -> list[str]:
        ranked = sorted(self.tags.items(), key=lambda item: (-item[1], item[0]))
        return [tag for tag, _ in ranked[:limit]]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["family_label"] = self.family_label
        data["top_tags"] = self.top_tags(12)
        data.pop("tags", None)
        data["tag_count"] = len(self.tags)
        return data


def normalize(text: str) -> str:
    """Lowercase, drop prompt weight syntax and lora tags, collapse spaces and underscores."""
    value = _LORA_TAG.sub(" ", str(text)).strip()
    match = _WEIGHT_SYNTAX.match(value)
    if match:
        value = match.group(1)
    value = value.replace("\\(", "(").replace("\\)", ")").replace("_", " ")
    value = re.sub(r"[\[\]{}]", " ", value)
    value = re.sub(r"\s+", " ", value).strip().strip(",.;:").strip().lower()
    return value


def split_phrases(text: str) -> list[str]:
    parts = re.split(r"[,\n;|]+", str(text))
    return [p.strip() for p in parts if normalize(p)]


def _decode_json_field(value: str | None):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def parse_tag_frequency(metadata: dict[str, str]) -> tuple[dict[str, int], list[str]]:
    """Return merged tag counts and the kohya dataset folder names from ss_tag_frequency."""
    raw = _decode_json_field(metadata.get("ss_tag_frequency"))
    tags: dict[str, int] = {}
    dirs: list[str] = []
    if isinstance(raw, dict):
        for folder, counts in raw.items():
            dirs.append(str(folder))
            if isinstance(counts, dict):
                for tag, count in counts.items():
                    norm = normalize(tag)
                    if not norm or len(norm) > 120:
                        continue
                    try:
                        tags[norm] = tags.get(norm, 0) + int(count)
                    except (TypeError, ValueError):
                        continue
    dataset_dirs = _decode_json_field(metadata.get("ss_dataset_dirs"))
    if isinstance(dataset_dirs, dict):
        for folder in dataset_dirs:
            if folder not in dirs:
                dirs.append(str(folder))
    return tags, dirs


def detect_family(
    metadata: dict[str, str],
    tensor_names: list[str] | None = None,
    filename: str = "",
    shapes: dict[str, list[int]] | None = None,
) -> tuple[str, str, str]:
    """Return (family, how we know, raw detail).

    The tensor layout is the strongest evidence, so it wins when the full header is available.
    Metadata comes next. Some trainers (ai-toolkit) write ss_base_model_version "sd_1.5" even for
    Flux LoRAs, so that particular combination is not trusted.
    """
    if tensor_names:
        family, detail = family_from_tensors(tensor_names, shapes or {})
        if family != "unknown":
            return family, "tensor layout", detail
    arch = (metadata.get("modelspec.architecture") or "").lower()
    base = (metadata.get("ss_base_model_version") or "").lower()
    model_name = (metadata.get("ss_sd_model_name") or metadata.get("ss_base_model") or "").lower()
    software = (metadata.get("software") or "").lower()
    v2 = (metadata.get("ss_v2") or "").lower() == "true"
    if arch:
        family = _family_from_text(arch)
        if family != "unknown":
            return family, "modelspec.architecture", arch
    unreliable_base = "ai-toolkit" in software and _family_from_text(base) == "sd15"
    if base and not unreliable_base:
        family = _family_from_text(base)
        if family != "unknown":
            return family, "ss_base_model_version", base
    if v2:
        return "sd2", "ss_v2", "true"
    if model_name:
        family = _family_from_text(model_name)
        if family != "unknown":
            return family, "ss_sd_model_name", model_name
    if filename:
        family = _family_from_text(Path(filename.replace("\\", "/")).name)
        if family not in ("unknown", "other"):
            return family, "filename guess", Path(filename).name
    return "unknown", "", arch or base or model_name


def _family_from_text(value: str) -> str:
    value = value.lower()
    if "flux" in value:
        return "flux"
    if "sd3" in value or "stable-diffusion-3" in value or "sd_3" in value:
        return "sd3"
    if any(k in value for k in ("sdxl", "xl-v1", "xl_v1", "-xl", "_xl", "pony", "illustrious", "noobai")):
        return "sdxl"
    if any(k in value for k in ("sd_v2", "v2-", "stable-diffusion-v2", "sd2", "sd_2")):
        return "sd2"
    if any(k in value for k in ("sd_v1", "v1-5", "v1-4", "stable-diffusion-v1", "sd15", "sd1.5", "sd_1", "sd-1")):
        return "sd15"
    if any(k in value for k in ("wan", "hunyuan", "ltx", "qwen", "hidream", "chroma", "lumina", "cascade", "pixart")):
        return "other"
    return "unknown"


CONTEXT_DIMS = {2048: "sdxl", 768: "sd15", 1024: "sd2"}


def family_from_tensors(names: list[str], shapes: dict[str, list[int]] | None = None) -> tuple[str, str]:
    """Infer the base model from tensor names and, for UNet LoRAs, the cross-attention width."""
    shapes = shapes or {}
    joined = "\n".join(names[:6000])
    if any(m in joined for m in ("double_blocks", "single_blocks", "single_transformer_blocks")):
        return "flux", "Flux block names"
    if "joint_blocks" in joined or ("transformer.transformer_blocks" in joined and "context_embedder" in joined):
        return "sd3", "SD3 block names"
    for name in names:
        lower = name.lower()
        if "attn2" in lower and "to_k" in lower and ("down" in lower or "lora_a" in lower):
            shape = shapes.get(name) or []
            if len(shape) >= 2 and shape[-1] in CONTEXT_DIMS:
                return CONTEXT_DIMS[shape[-1]], f"cross-attention width {shape[-1]}"
    if "lora_te2_" in joined or "text_encoder_2" in joined:
        return "sdxl", "second text encoder"
    # lora_unet_ is shared by multiple SD families, so it must not override
    # explicit metadata. Text-encoder keys remain useful SD 1.x evidence.
    if "lora_te_" in joined or "lora_te1_" in joined:
        return "sd15", "SD 1.x block names"
    return "unknown", ""


def family_from_tensor_names(names: list[str]) -> str:
    return family_from_tensors(names)[0]


def triggers_from_metadata(metadata: dict[str, str], dataset_dirs: list[str]) -> list[Trigger]:
    found: list[Trigger] = []
    phrase = metadata.get("modelspec.trigger_phrase") or metadata.get("trigger_phrase") or ""
    for part in split_phrases(phrase):
        found.append(Trigger(part, normalize(part), "modelspec.trigger_phrase"))
    for key in ("ss_trigger_words", "trigger_words", "activation_text"):
        for part in split_phrases(metadata.get(key, "")):
            found.append(Trigger(part, normalize(part), key))
    for folder in dataset_dirs:
        name = Path(folder.replace("\\", "/")).name
        match = _DIR_PREFIX.match(name)
        if match:
            text = match.group(1)
            if _HASHLIKE.match(text.strip()):
                continue  # trainers that name folders after a job id, not a trigger word
            found.append(Trigger(text, normalize(text), "dataset folder"))
    return found


GENERIC_CAPTION_TAGS = {
    "solo", "1girl", "1boy", "2girls", "2boys", "multiple girls", "multiple boys", "simple background",
    "white background", "black background", "looking at viewer", "smile", "upper body", "full body", "portrait",
    "outdoors", "indoors", "realistic", "photorealistic", "no humans", "monochrome", "greyscale", "short hair",
    "long hair", "blush", "open mouth", "closed mouth", "standing", "sitting", "day", "night", "sky", "a", "the",
}


def inferred_triggers(tags: dict[str, int], limit: int = 2) -> list[Trigger]:
    """Tags at (or within 3% of) the top count are usually the caption trigger. Generic caption tags are skipped."""
    tags = {t: c for t, c in tags.items() if t not in GENERIC_CAPTION_TAGS}
    if not tags:
        return []
    top = max(tags.values())
    if top < 3:
        return []
    ranked = sorted(tags.items(), key=lambda item: (-item[1], item[0]))
    candidates = [tag for tag, count in ranked if count >= top * 0.97]
    if len(candidates) >= 6:
        return []
    picked = candidates[:limit]
    return [Trigger(tag, tag, "inferred (top tag)") for tag in picked]


def read_sidecar_triggers(model_path: Path) -> list[Trigger]:
    found: list[Trigger] = []
    stem = model_path.with_suffix("")
    for suffix in SIDECAR_SUFFIXES:
        candidate = Path(str(stem) + suffix)
        if not candidate.is_file() or candidate.stat().st_size > 2_000_000:
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if suffix == ".txt":
            for part in split_phrases(text):
                found.append(Trigger(part, normalize(part), f"sidecar {candidate.name}"))
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        for part in _json_trigger_values(data):
            found.append(Trigger(part, normalize(part), f"sidecar {candidate.name}"))
    return found


def _json_trigger_values(data) -> list[str]:
    values: list[str] = []
    if isinstance(data, dict):
        for key in SIDECAR_JSON_KEYS:
            if key in data:
                item = data[key]
                if isinstance(item, str):
                    values.extend(split_phrases(item))
                elif isinstance(item, list):
                    for entry in item:
                        if isinstance(entry, str):
                            values.extend(split_phrases(entry))
        # civitai.info nests trainedWords at top level; model-version JSON may nest under "civitai"
        for nested_key in ("civitai", "modelVersion"):
            if isinstance(data.get(nested_key), dict):
                values.extend(_json_trigger_values(data[nested_key]))
    return values


def dedupe(triggers: list[Trigger]) -> list[Trigger]:
    seen: set[str] = set()
    out: list[Trigger] = []
    for trig in triggers:
        if not trig.norm or trig.norm in seen or len(trig.norm) > 200:
            continue
        seen.add(trig.norm)
        out.append(trig)
    return out


def build_record(
    filename: str,
    metadata: dict[str, str],
    *,
    tensor_names: list[str] | None = None,
    shapes: dict[str, list[int]] | None = None,
    sidecar: list[Trigger] | None = None,
    origin: str = "folder",
) -> LoraRecord:
    name = Path(filename.replace("\\", "/")).stem
    trained_name = metadata.get("ss_output_name") or metadata.get("modelspec.title") or metadata.get("name") or ""
    tags, dirs = parse_tag_frequency(metadata)
    triggers = list(sidecar or []) + triggers_from_metadata(metadata, dirs)
    if not triggers:
        triggers = inferred_triggers(tags)
    family, family_source, detail = detect_family(metadata, tensor_names, filename, shapes)
    images = None
    try:
        images = int(metadata["ss_num_train_images"]) if metadata.get("ss_num_train_images") else None
    except ValueError:
        images = None
    return LoraRecord(
        name=str(name),
        trained_name=str(trained_name)[:120],
        filename=filename,
        family=family,
        family_source=family_source,
        base_detail=detail[:120],
        triggers=dedupe(triggers),
        tags=tags,
        image_count=images,
        origin=origin,
    )


def disambiguate(records: list[LoraRecord]) -> list[LoraRecord]:
    """Give LoRAs that share a file stem (for example two 'lora.safetensors') their relative path as the name."""
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec.name.lower()] = counts.get(rec.name.lower(), 0) + 1
    for rec in records:
        if counts[rec.name.lower()] > 1:
            rec.name = rec.filename.replace("\\", "/").rsplit(".", 1)[0]
    return records
