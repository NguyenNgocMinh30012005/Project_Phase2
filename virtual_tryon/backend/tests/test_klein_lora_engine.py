from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from app.core.config import EngineConfig
from app.engines.base import TryOnInputs
from app.engines.klein_tryon_lora_engine import KleinTryOnLoraEngine
from app.utils.errors import ModelUnavailableError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(**overrides) -> EngineConfig:
    payload = {
        "enabled": True,
        "backend": "fal_api",
        "base_model": "black-forest-labs/FLUX.2-klein-9B",
        "lora_repo": "fal/flux-klein-9b-virtual-tryon-lora",
        "lora_weight_api": "flux-klein-tryon.safetensors",
        "fal_endpoint": "fal-ai/flux-2/klein/9b/base/edit/lora",
        "num_inference_steps": 28,
        "guidance_scale": 2.5,
        "lora_scale": 1.0,
        "require_three_images": True,
        "bottom_strategy": "crop_from_person",
        "bottom_crop": {
            "y_start_ratio": 0.50,
            "y_end_ratio": 0.98,
            "x_margin_ratio": 0.08,
            "save_debug": True,
        },
    }
    payload.update(overrides)
    return EngineConfig(**payload)


def _inputs(output_dir: Path) -> TryOnInputs:
    person = Image.new("RGB", (256, 384), (180, 160, 140))
    top = Image.new("RGB", (256, 384), (20, 80, 210))
    return TryOnInputs(
        person_image=person,
        garment_image=top,
        category="upper_body",
        agnostic_mask=Image.new("L", person.size, 255),
        prompt=None,
        seed=42,
        output_dir=output_dir,
        extra={"garment_top_image": top},
    )


def test_klein_lora_availability_missing_token(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    engine = KleinTryOnLoraEngine(_config())
    availability = engine.is_available()
    assert not availability
    assert availability.error_code in {"MISSING_FAL_KEY", "DEPENDENCY_MISSING"}
    assert "FAL_KEY" in availability.status


def test_klein_lora_bottom_crop_from_person(monkeypatch, tmp_path):
    monkeypatch.delenv("FAL_KEY", raising=False)
    engine = KleinTryOnLoraEngine(_config())
    try:
        engine.run(_inputs(tmp_path))
    except ModelUnavailableError:
        pass
    crop_path = tmp_path / "auto_bottom_reference.png"
    assert crop_path.exists()
    crop = Image.open(crop_path)
    assert crop.width < 256
    assert crop.height > 100
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["bottom_source"] == "auto_cropped_from_person"


def test_klein_lora_sanitizes_request_response(monkeypatch, tmp_path):
    monkeypatch.delenv("FAL_KEY", raising=False)
    engine = KleinTryOnLoraEngine(_config(api_key_env="FAL_KEY"))
    try:
        engine.run(_inputs(tmp_path))
    except ModelUnavailableError:
        pass
    request_payload = json.loads((tmp_path / "request_sanitized.json").read_text(encoding="utf-8"))
    serialized = json.dumps(request_payload)
    assert "hf_" not in serialized
    assert "FAL_KEY" not in serialized
    assert request_payload["loras"][0]["path"].endswith("flux-klein-tryon.safetensors")


def test_klein_lora_manual_rating_template(tmp_path):
    eval_set = tmp_path / "eval_set" / "sample_001"
    eval_set.mkdir(parents=True)
    Image.new("RGB", (128, 192), (180, 180, 180)).save(eval_set / "person.jpg")
    Image.new("RGB", (128, 192), (20, 80, 210)).save(eval_set / "garment_top.jpg")
    (eval_set / "metadata.json").write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
                "category": "upper_body",
                "difficulty": "easy",
                "expected_focus": ["identity"],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "ablation"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_klein_lora_ablation.py"),
            "--sample",
            str(eval_set),
            "--output",
            str(output_dir),
            "--mock",
            "--skip-idm",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    template = output_dir / "manual_ratings_klein_lora.csv"
    assert template.exists()
    rows = list(csv.DictReader(template.open(encoding="utf-8")))
    variants = {row["variant"] for row in rows}
    assert {"klein_lora_default_prompt", "klein_lora_strong_prompt"}.issubset(variants)
    assert all(row["identity_1_5"] == "" for row in rows)
