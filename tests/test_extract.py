import json

from triggercollide.extract import (
    build_record,
    detect_family,
    family_from_tensor_names,
    family_from_tensors,
    normalize,
    read_sidecar_triggers,
)
from triggercollide.safetensors_meta import write_safetensors


def test_normalize_strips_weights_and_underscores():
    assert normalize("(Glow_Style:1.2)") == "glow style"
    assert normalize("  <lora:x:1> Foo  Bar, ") == "foo bar"


def test_dataset_folder_gives_trigger_and_tags_merge():
    meta = {
        "ss_tag_frequency": json.dumps({"10_zxq lamp": {"zxq": 5, "Lamp": 4}, "2_extra": {"lamp": 1}}),
        "ss_base_model_version": "sdxl_base_v1-0",
    }
    rec = build_record("dir/lamp.safetensors", meta)
    assert [t.norm for t in rec.triggers] == ["zxq lamp", "extra"]
    assert rec.tags == {"zxq": 5, "lamp": 5}
    assert rec.family == "sdxl"
    assert rec.name == "lamp"


def test_inferred_trigger_when_nothing_explicit():
    meta = {"ss_tag_frequency": json.dumps({"img": {"qwopstyle": 30, "tree": 12, "sky": 5}})}
    rec = build_record("x.safetensors", meta)
    assert [t.norm for t in rec.triggers] == ["qwopstyle"]
    assert rec.triggers[0].source.startswith("inferred")


def test_family_detection_sources():
    assert detect_family({"modelspec.architecture": "flux-1-dev/lora"})[0] == "flux"
    assert detect_family({"ss_base_model_version": "sd_v1"})[0] == "sd15"
    assert detect_family({"ss_v2": "True"})[0] == "sd2"
    assert detect_family({"modelspec.architecture": "stable-diffusion-xl-v1-base/lora"})[0] == "sdxl"
    assert detect_family({}, filename="cool_style_flux.safetensors")[:2] == ("flux", "filename guess")
    assert detect_family({})[0] == "unknown"
    assert family_from_tensor_names(["lora_te2_text_model.x"]) == "sdxl"
    assert family_from_tensor_names(["transformer.single_transformer_blocks.0.attn"]) == "flux"


def test_sidecar_txt_and_civitai_info(tmp_path):
    model = write_safetensors(tmp_path / "m.safetensors", {})
    (tmp_path / "m.txt").write_text("alpha one, beta\n", encoding="utf-8")
    (tmp_path / "m.civitai.info").write_text(json.dumps({"trainedWords": ["gamma, delta"]}), encoding="utf-8")
    found = sorted(t.norm for t in read_sidecar_triggers(model))
    assert found == ["alpha one", "beta", "delta", "gamma"]


def test_tensor_layout_beats_wrong_metadata():
    # ai-toolkit writes ss_base_model_version "sd_1.5" for Flux LoRAs
    meta = {"ss_base_model_version": "sd_1.5", "software": '{"name": "ai-toolkit"}'}
    names = ["transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight"]
    assert detect_family(meta, names)[:2] == ("flux", "tensor layout")
    # metadata only (the ComfyUI path): the unreliable value is not trusted
    assert detect_family(meta)[0] == "unknown"
    assert detect_family({"ss_base_model_version": "sd_1.5"})[0] == "sd15"


def test_cross_attention_width_identifies_sdxl_with_diffusers_names():
    name = "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight"
    assert family_from_tensors([name], {name: [8, 2048]}) == ("sdxl", "cross-attention width 2048")
    assert family_from_tensors([name], {name: [8, 768]})[0] == "sd15"
    assert family_from_tensors([name], {})[0] == "unknown"


def test_generic_unet_tensor_keys_do_not_override_sdxl_metadata():
    names = ["lora_unet_down_blocks_1_attentions_0.to_q.lora_down.weight"]
    assert detect_family({"ss_base_model_version": "sdxl_base_1.0"}, names)[0] == "sdxl"


def test_generic_caption_tags_are_not_inferred_as_triggers():
    meta = {"ss_tag_frequency": json.dumps({"img": {"solo": 40, "1girl": 40, "blorptoon": 39, "hat": 3}})}
    rec = build_record("x.safetensors", meta)
    assert [t.norm for t in rec.triggers] == ["blorptoon"]


def test_six_near_tied_tags_are_too_ambiguous_to_infer():
    meta = {"ss_tag_frequency": json.dumps({"img": {f"token{i}": 100 for i in range(6)}})}
    rec = build_record("x.safetensors", meta)
    assert rec.triggers == []


def test_hash_named_dataset_folders_are_not_triggers():
    meta = {"ss_tag_frequency": json.dumps({"1_66acad858845421286e736fba44c887f": {"logo": 3}})}
    rec = build_record("x.safetensors", meta)
    assert all("66acad" not in t.norm for t in rec.triggers)
