from triggercollide.collide import analyze, matrix
from triggercollide.extract import Trigger, build_record
from triggercollide.library import scan_folder


def _rec(name, triggers, family="sdxl", tags=None):
    rec = build_record(f"{name}.safetensors", {})
    rec.family = family
    rec.triggers = [Trigger(t, t, "test") for t in triggers]
    rec.tags = tags or {}
    return rec


def test_exact_substring_and_token_overlap():
    recs = [_rec("a", ["glowfox"]), _rec("b", ["glowfox"]), _rec("c", ["glowfox neon"]), _rec("d", ["blue neon sign"]),
            _rec("e", ["neon sign"])]
    result = analyze(recs)
    pairs = {(c.a, c.b): c for c in result["collisions"]}
    assert pairs[(0, 1)].severity == "high"
    assert any(f.kind == "substring" for f in pairs[(0, 2)].findings)
    assert any(f.kind == "substring" for f in pairs[(3, 4)].findings)


def test_cross_tag_finding():
    recs = [_rec("a", ["moonlit"]), _rec("b", ["sunburst"], tags={"moonlit": 9, "x": 1})]
    result = analyze(recs)
    assert result["collisions"][0].findings[0].kind == "cross_tag"


def test_cross_family_pairs_are_skipped_by_default():
    recs = [_rec("a", ["shared"], "sdxl"), _rec("b", ["shared"], "flux"), _rec("c", ["shared"], "unknown")]
    default = analyze(recs)
    assert {(c.a, c.b) for c in default["collisions"]} == {(0, 2), (1, 2)}
    assert default["skipped_cross_family_pairs"] == 1
    everything = analyze(recs, same_family_only=False)
    assert len(everything["collisions"]) == 3


def test_generic_words_do_not_collide():
    recs = [_rec("a", ["red style"]), _rec("b", ["blue style"])]
    assert analyze(recs)["collisions"] == []


def test_common_tags_are_ignored():
    tags = {"1girl": 50, "solo": 40}
    recs = [_rec(n, [n + "trig"], tags=dict(tags)) for n in "abcdef"]
    result = analyze(recs)
    assert result["collisions"] == []
    assert set(result["common_tags"]) == {"1girl", "solo"}


def test_demo_library_finds_the_planted_clashes(demo_dir):
    records, stats = scan_folder(demo_dir)
    assert stats["files"] == 11 and stats["errors"] == 0
    result = analyze(records)
    names = {(records[c.a].name, records[c.b].name) for c in result["collisions"] if c.severity == "high"}
    assert ("watercolor_fox_v2_xl", "watercolor_fox_xl") in names or ("watercolor_fox_xl", "watercolor_fox_v2_xl") in names
    by_name = {r.name: r for r in records}
    assert by_name["vintage_film_flux"].family == "flux"
    assert by_name["retro_pixel_hero_sd15"].family == "sd15"
    assert [t.norm for t in by_name["knight_armor_detail"].triggers] == ["knightarmor", "ornate armor"]
    m = matrix(records, result)
    assert len(m["labels"]) == len(m["scores"]) and all(m["scores"][i][i] == 0 for i in range(len(m["labels"])))


def test_scales_to_a_large_library():
    import time

    recs = [_rec(f"lora{i}", [f"trig{i}", f"style{i % 50} look"], tags={f"tag{i % 97}": 5, f"t{i}": 3})
            for i in range(1500)]
    start = time.perf_counter()
    result = analyze(recs)
    assert time.perf_counter() - start < 30
    assert result["collisions"]


def test_inferred_triggers_count_less_than_explicit():
    a = _rec("a", ["zork"])
    b = _rec("b", [])
    b.triggers = [Trigger("zork", "zork", "inferred (top tag)")]
    c = _rec("c", ["zork"])
    result = analyze([a, b, c])
    scores = {(x.a, x.b): x.score for x in result["collisions"]}
    assert scores[(0, 2)] == 1.0
    assert scores[(0, 1)] < 1.0
