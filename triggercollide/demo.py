"""A small synthetic LoRA library with deliberate collisions, written as real .safetensors files."""

from __future__ import annotations

import json
from pathlib import Path

from triggercollide.safetensors_meta import write_safetensors

SDXL_TENSORS = {"lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": [0.0, 0.0],
                "lora_unet_input_blocks_4_1_proj_in.lora_down.weight": [0.0, 0.0]}
SD15_TENSORS = {"lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight": [0.0, 0.0],
                "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": [0.0, 0.0]}
FLUX_TENSORS = {"lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": [0.0, 0.0],
                "lora_unet_single_blocks_0_linear1.lora_down.weight": [0.0, 0.0]}


def _tags(folder: str, counts: dict[str, int]) -> str:
    return json.dumps({folder: counts})


DEMO = [
    ("copper_lantern_xl.safetensors", SDXL_TENSORS, {
        "ss_output_name": "Copper Lantern", "ss_base_model_version": "sdxl_base_v1-0",
        "ss_tag_frequency": _tags("10_cprlntrn lantern", {"cprlntrn": 40, "lantern": 38, "warm light": 30,
                                                          "copper": 28, "night": 12, "street": 9}),
        "ss_num_train_images": "400"}),
    ("paper_lantern_festival_xl.safetensors", SDXL_TENSORS, {
        "ss_output_name": "Paper Lantern Festival", "ss_base_model_version": "sdxl_base_v1-0",
        "modelspec.trigger_phrase": "lantern festival",
        "ss_tag_frequency": _tags("8_lantern festival", {"lantern festival": 32, "lantern": 30, "warm light": 22,
                                                         "crowd": 15, "night": 14, "paper": 12}),
        "ss_num_train_images": "256"}),
    ("watercolor_fox_xl.safetensors", SDXL_TENSORS, {
        "ss_output_name": "Watercolor Fox", "ss_base_model_version": "sdxl_base_v1-0",
        "modelspec.trigger_phrase": "wcfox",
        "ss_tag_frequency": _tags("12_wcfox", {"wcfox": 36, "fox": 34, "watercolor": 33, "forest": 10,
                                               "paper texture": 8}), "ss_num_train_images": "432"}),
    ("watercolor_fox_v2_xl.safetensors", SDXL_TENSORS, {
        "ss_output_name": "Watercolor Fox v2", "ss_base_model_version": "sdxl_base_v1-0",
        "modelspec.trigger_phrase": "wcfox",
        "ss_tag_frequency": _tags("12_wcfox", {"wcfox": 40, "fox": 39, "watercolor": 38, "snow": 11,
                                               "paper texture": 9}), "ss_num_train_images": "480"}),
    ("watercolor_wash_xl.safetensors", SDXL_TENSORS, {
        "ss_output_name": "Watercolor Wash", "ss_base_model_version": "sdxl_base_v1-0",
        "modelspec.trigger_phrase": "watercolor wash",
        "ss_tag_frequency": _tags("6_watercolor wash", {"watercolor": 50, "watercolor wash": 48,
                                                        "paper texture": 30, "soft edges": 20}),
        "ss_num_train_images": "300"}),
    ("clay_figurine_xl.safetensors", SDXL_TENSORS, {
        "ss_output_name": "Clay Figurine", "ss_base_model_version": "sdxl_base_v1-0",
        "modelspec.trigger_phrase": "clayfig",
        "ss_tag_frequency": _tags("10_clayfig", {"clayfig": 30, "clay": 29, "figurine": 25, "studio light": 12}),
        "ss_num_train_images": "300"}),
    ("retro_pixel_hero_sd15.safetensors", SD15_TENSORS, {
        "ss_output_name": "Retro Pixel Hero", "ss_base_model_version": "sd_v1",
        "ss_tag_frequency": _tags("10_pxhero", {"pxhero": 25, "pixel art": 24, "hero": 20, "sword": 8}),
        "ss_num_train_images": "250"}),
    ("ink_sketch_sd15.safetensors", SD15_TENSORS, {
        "ss_output_name": "Ink Sketch", "ss_base_model_version": "sd_v1",
        "modelspec.trigger_phrase": "inksketch",
        "ss_tag_frequency": _tags("10_inksketch", {"inksketch": 22, "monochrome": 21, "lineart": 20, "hero": 5}),
        "ss_num_train_images": "220"}),
    ("vintage_film_flux.safetensors", FLUX_TENSORS, {
        "modelspec.architecture": "flux-1-dev/lora", "modelspec.title": "Vintage Film",
        "modelspec.trigger_phrase": "vntgfilm, film grain"}),
    ("film_grain_boost_flux.safetensors", FLUX_TENSORS, {
        "modelspec.architecture": "flux-1-dev/lora", "modelspec.title": "Film Grain Boost",
        "modelspec.trigger_phrase": "film grain"}),
    ("knight_armor_detail.safetensors", SDXL_TENSORS, {}),
]

SIDECARS = {"knight_armor_detail.txt": "knightarmor, ornate armor\n"}


def write_demo_library(folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, tensors, meta in DEMO:
        write_safetensors(folder / name, meta, tensors)
    for name, text in SIDECARS.items():
        (folder / name).write_text(text, encoding="utf-8")
    (folder / "README.txt").write_text(
        "Synthetic demo LoRA headers made by TriggerCollide. They contain no real model weights.\n",
        encoding="utf-8",
    )
    return folder


def demo_workflow() -> dict:
    """A ComfyUI API-format workflow that stacks three demo LoRAs (one from the wrong base)."""
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "demo_sdxl_base.safetensors"}},
        "10": {"class_type": "LoraLoader", "inputs": {"lora_name": "watercolor_fox_xl.safetensors",
                                                      "strength_model": 0.9, "strength_clip": 0.9,
                                                      "model": ["4", 0], "clip": ["4", 1]}},
        "11": {"class_type": "LoraLoader", "inputs": {"lora_name": "watercolor_fox_v2_xl.safetensors",
                                                      "strength_model": 0.8, "strength_clip": 0.8,
                                                      "model": ["10", 0], "clip": ["10", 1]}},
        "12": {"class_type": "LoraLoader", "inputs": {"lora_name": "ink_sketch_sd15.safetensors",
                                                      "strength_model": 0.6, "strength_clip": 0.6,
                                                      "model": ["11", 0], "clip": ["11", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "wcfox in a snowy forest, watercolor",
                                                         "clip": ["12", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 6, "sampler_name": "euler",
                                                   "scheduler": "normal", "denoise": 1, "model": ["12", 0],
                                                   "positive": ["6", 0], "negative": ["6", 0],
                                                   "latent_image": ["5", 0]}},
    }
