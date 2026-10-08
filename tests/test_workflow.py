import json

from triggercollide.demo import demo_workflow
from triggercollide.library import scan_folder
from triggercollide.workflow import check_workflow, extract_stack


def test_demo_workflow_flags_clash_base_mismatch_and_weight(demo_dir):
    records, _ = scan_folder(demo_dir)
    report = check_workflow(demo_workflow(), records).to_dict()
    kinds = {i["kind"] for i in report["issues"]}
    assert len(report["stacked"]) == 3
    assert report["total_weight"] == 2.3
    assert {"collision", "base_mismatch", "total_weight"} <= kinds
    assert report["issue_counts"]["high"] >= 2


def test_power_lora_loader_and_stack_nodes():
    wf = {
        "1": {"class_type": "Power Lora Loader (rgthree)", "inputs": {
            "lora_1": {"on": True, "lora": "a.safetensors", "strength": 0.7},
            "lora_2": {"on": False, "lora": "b.safetensors", "strength": 1.0}}},
        "2": {"class_type": "CR LoRA Stack", "inputs": {
            "switch_1": "On", "lora_name_1": "c.safetensors", "model_weight_1": 0.5, "clip_weight_1": 0.5,
            "switch_2": "Off", "lora_name_2": "d.safetensors", "model_weight_2": 1.0}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "e.safetensors", "strength_model": 0.4},
              "mode": 4},
    }
    stacked, _, fmt = extract_stack(wf)
    assert fmt == "api"
    enabled = sorted(s.lora for s in stacked if s.enabled)
    assert enabled == ["a.safetensors", "c.safetensors"]
    assert len(stacked) == 5


def test_ui_format_workflow(demo_dir):
    records, _ = scan_folder(demo_dir)
    wf = {"nodes": [
        {"id": 4, "type": "CheckpointLoaderSimple", "widgets_values": ["something_flux_dev.safetensors"]},
        {"id": 7, "type": "LoraLoader", "mode": 0, "widgets_values": ["watercolor_fox_xl.safetensors", 1.0, 1.0]},
        {"id": 8, "type": "CLIPTextEncode", "widgets_values": ["a cat"]},
    ], "links": []}
    report = check_workflow(json.dumps(wf), records)
    assert report.format == "ui"
    assert any(i["kind"] == "base_mismatch" for i in report.issues)
    assert report.missing_triggers and report.missing_triggers[0]["triggers"] == ["wcfox"]


def test_missing_lora_and_no_lora_cases(demo_dir):
    records, _ = scan_folder(demo_dir)
    report = check_workflow({"1": {"class_type": "LoraLoader", "inputs": {"lora_name": "nope.safetensors",
                                                                        "strength_model": 1.0}}}, records)
    assert report.issues[0]["kind"] == "missing"
    empty = check_workflow({"1": {"class_type": "KSampler", "inputs": {}}}, records)
    assert empty.issues[0]["kind"] == "no_loras"


def test_same_file_name_in_two_folders_is_matched_by_path(tmp_path):
    from triggercollide.safetensors_meta import write_safetensors

    write_safetensors(tmp_path / "a" / "lora.safetensors", {"modelspec.trigger_phrase": "alpha"})
    write_safetensors(tmp_path / "b" / "lora.safetensors", {"modelspec.trigger_phrase": "beta"})
    records, _ = scan_folder(tmp_path)
    assert sorted(r.name for r in records) == ["a/lora", "b/lora"]
    wf = {"1": {"class_type": "LoraLoader", "inputs": {"lora_name": "b\\lora.safetensors", "strength_model": 1}},
          "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "lora.safetensors", "strength_model": 1}}}
    report = check_workflow(wf, records)
    missing = [i for i in report.issues if i["kind"] == "missing"]
    assert len(missing) == 1 and "node 2" in missing[0]["message"]
