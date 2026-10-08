"""Pairwise trigger collision analysis for a LoRA library."""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from triggercollide.extract import LoraRecord

# Words that show up in many triggers and mean little on their own.
GENERIC_TOKENS = {
    "a", "an", "the", "of", "and", "in", "on", "with", "by", "for", "to", "at", "style", "styles", "lora",
    "woman", "man", "girl", "boy", "person", "people", "character", "photo", "photograph", "art", "artstyle",
    "1girl", "1boy", "solo", "concept", "v1", "v2", "v3", "xl", "sdxl", "flux", "sd15", "model", "realistic",
}

WEIGHTS = {
    "exact": 1.0,
    "substring": 0.6,
    "cross_tag": 0.5,
    "token_overlap": 0.4,
    "shared_tag": 0.05,
}
SHARED_TAG_CAP = 0.5
INFERRED_FACTOR = 0.6  # triggers guessed from tag counts are less certain
HIGH = 1.0
MEDIUM = 0.5


@dataclass
class Finding:
    kind: str
    detail: str
    a: str = ""
    b: str = ""


@dataclass
class Collision:
    a: int
    b: int
    score: float
    severity: str
    findings: list[Finding] = field(default_factory=list)
    shared_tags: list[str] = field(default_factory=list)

    def to_dict(self, records: list[LoraRecord]) -> dict:
        data = asdict(self)
        data["a_name"] = records[self.a].name
        data["b_name"] = records[self.b].name
        data["a_file"] = records[self.a].filename
        data["b_file"] = records[self.b].filename
        data["score"] = round(self.score, 3)
        return data


def tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^\w]+", text.lower()) if t and t not in GENERIC_TOKENS and len(t) > 1}


def severity_for(score: float) -> str:
    if score >= HIGH:
        return "high"
    if score >= MEDIUM:
        return "medium"
    return "low"


def common_tags(records: list[LoraRecord], top_n: int = 25, share: float = 0.3) -> set[str]:
    """Tags that sit in the top list of many LoRAs (for example '1girl' or 'solo') are noise for collisions."""
    if len(records) < 4:
        return set()
    counts: Counter[str] = Counter()
    for rec in records:
        counts.update(set(rec.top_tags(top_n)))
    limit = max(2, int(len(records) * share))
    return {tag for tag, count in counts.items() if count >= limit}


def compatible(fa: str, fb: str) -> bool:
    if "unknown" in (fa, fb):
        return True
    return fa == fb


def analyze(
    records: list[LoraRecord],
    *,
    same_family_only: bool = True,
    top_n: int = 25,
    min_score: float = 0.0,
) -> dict:
    """Return collisions between every pair of LoRAs that share a trigger, a word, or a distinctive tag."""
    usable = [i for i, r in enumerate(records) if not r.error]
    noise = common_tags([records[i] for i in usable], top_n=top_n)
    top_sets = {i: set(records[i].top_tags(top_n)) - noise for i in usable}
    all_tags = {i: set(records[i].tags) for i in usable}
    trig_norms = {i: [t.norm for t in records[i].triggers] for i in usable}
    trig_tokens = {i: [tokens(t) for t in trig_norms[i]] for i in usable}

    candidates: set[tuple[int, int]] = set()
    by_key: dict[str, list[int]] = defaultdict(list)
    for i in usable:
        keys = set()
        for toks in trig_tokens[i]:
            keys.update("w:" + t for t in toks)
        keys.update("n:" + n for n in trig_norms[i])
        keys.update("t:" + t for t in top_sets[i])
        for key in keys:
            by_key[key].append(i)
    for members in by_key.values():
        if len(members) > 1:
            if len(members) > 400:
                continue  # a token shared by hundreds of LoRAs is generic, skip it
            for a, b in itertools.combinations(sorted(members), 2):
                candidates.add((a, b))
    # trigger appears as a training tag in another LoRA
    trigger_owner: dict[str, list[int]] = defaultdict(list)
    for i in usable:
        for norm in trig_norms[i]:
            trigger_owner[norm].append(i)
    for norm, owners in trigger_owner.items():
        for j in usable:
            if norm in all_tags[j]:
                for i in owners:
                    if i != j:
                        candidates.add((min(i, j), max(i, j)))
    # character-level substring (for example "ohwx" inside "ohwxman")
    unique = sorted({n for n in trigger_owner if len(n) >= 4}, key=len)
    if len(unique) <= 6000:
        for idx, short in enumerate(unique):
            for long in unique[idx + 1:]:
                if short != long and short in long:
                    for i in trigger_owner[short]:
                        for j in trigger_owner[long]:
                            if i != j:
                                candidates.add((min(i, j), max(i, j)))

    collisions: list[Collision] = []
    skipped_family = 0
    for a, b in sorted(candidates):
        ra, rb = records[a], records[b]
        if same_family_only and not compatible(ra.family, rb.family):
            skipped_family += 1
            continue
        found = compare_pair(ra, rb, top_sets[a], top_sets[b], all_tags[a], all_tags[b])
        if found is None:
            continue
        score, findings, shared = found
        if score <= min_score:
            continue
        collisions.append(Collision(a, b, score, severity_for(score), findings, shared))
    collisions.sort(key=lambda c: (-c.score, records[c.a].name.lower(), records[c.b].name.lower()))
    involvement: Counter[int] = Counter()
    for col in collisions:
        involvement[col.a] += 1
        involvement[col.b] += 1
    families = Counter(r.family for r in records if not r.error)
    return {
        "collisions": collisions,
        "involvement": involvement,
        "common_tags": sorted(noise),
        "skipped_cross_family_pairs": skipped_family,
        "families": dict(families),
        "severity_counts": dict(Counter(c.severity for c in collisions)),
    }


def compare_pair(ra, rb, top_a, top_b, tags_a, tags_b):
    findings: list[Finding] = []
    score = 0.0
    norms_a = {t.norm for t in ra.triggers}
    norms_b = {t.norm for t in rb.triggers}
    guessed = {t.norm for t in ra.triggers + rb.triggers if t.source.startswith("inferred")}
    exact = norms_a & norms_b
    for value in sorted(exact):
        if value in guessed:
            findings.append(Finding("exact", f'Both seem to use "{value}" (guessed from training tags)', value, value))
            score += WEIGHTS["exact"] * INFERRED_FACTOR
        else:
            findings.append(Finding("exact", f'Both use the trigger "{value}"', value, value))
            score += WEIGHTS["exact"]
    for x in sorted(norms_a - exact):
        for y in sorted(norms_b - exact):
            if len(x) >= 3 and len(y) >= 3 and (x in y or y in x):
                findings.append(Finding("substring", f'"{x}" and "{y}" contain each other', x, y))
                score += WEIGHTS["substring"]
            else:
                tx, ty = tokens(x), tokens(y)
                if tx and ty:
                    jac = len(tx & ty) / len(tx | ty)
                    if jac >= 0.5:
                        shared = ", ".join(sorted(tx & ty))
                        findings.append(Finding("token_overlap", f'"{x}" and "{y}" share words: {shared}', x, y))
                        score += WEIGHTS["token_overlap"] * jac
    for x in sorted(norms_a - exact):
        if x in tags_b:
            findings.append(Finding("cross_tag", f'Trigger "{x}" of {ra.name} is a training tag of {rb.name}', x, ""))
            score += WEIGHTS["cross_tag"] * (INFERRED_FACTOR if x in guessed else 1.0)
    for y in sorted(norms_b - exact):
        if y in tags_a:
            findings.append(Finding("cross_tag", f'Trigger "{y}" of {rb.name} is a training tag of {ra.name}', "", y))
            score += WEIGHTS["cross_tag"] * (INFERRED_FACTOR if y in guessed else 1.0)
    shared_tags = sorted((top_a & top_b) - norms_a - norms_b)
    if shared_tags:
        score += min(SHARED_TAG_CAP, WEIGHTS["shared_tag"] * len(shared_tags))
        findings.append(Finding("shared_tags", f"{len(shared_tags)} distinctive training tags in common"))
    if not findings:
        return None
    return score, findings, shared_tags[:20]


def matrix(records: list[LoraRecord], result: dict, limit: int = 40) -> dict:
    """Square score matrix for the LoRAs involved in the most collisions."""
    involvement = result["involvement"]
    picked = [i for i, _ in involvement.most_common(limit)]
    picked.sort(key=lambda i: records[i].name.lower())
    pos = {i: k for k, i in enumerate(picked)}
    grid = [[0.0] * len(picked) for _ in picked]
    for col in result["collisions"]:
        if col.a in pos and col.b in pos:
            grid[pos[col.a]][pos[col.b]] = round(col.score, 3)
            grid[pos[col.b]][pos[col.a]] = round(col.score, 3)
    return {"labels": [records[i].name for i in picked], "indexes": picked, "scores": grid}
