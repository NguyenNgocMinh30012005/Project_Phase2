from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from app.core.config import load_settings
from app.engines.base import TryOnInputs
from app.engines.idm_vton_engine import IDMVTonEngine, REQUIRED_CHECKPOINTS
from app.preprocessing.agnostic_mask import create_agnostic_mask


def _write_fake_file(path: Path, size: int = 2048) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _settings_for_idm(tmp_path):
    settings = load_settings()
    settings.pipeline.engine = "idm_vton"
    settings.storage.inputs_dir = tmp_path / "inputs"
    settings.storage.outputs_dir = tmp_path / "outputs"
    settings.storage.temp_dir = tmp_path / "temp"
    settings.idm_vton.repo_path = tmp_path / "IDM-VTON"
    settings.idm_vton.repo_path.mkdir()
    settings.idm_vton.entrypoint = settings.idm_vton.repo_path / "inference.py"
    settings.idm_vton.entrypoint.write_text("print('fake')\n", encoding="utf-8")
    settings.idm_vton.checkpoint_dir = tmp_path / "models" / "idm_vton" / "ckpt"
    return settings


def _make_inputs(output_dir: Path) -> TryOnInputs:
    person = Image.new("RGB", (256, 384), (180, 180, 180))
    garment = Image.new("RGB", (256, 384), (20, 80, 220))
    mask_result = create_agnostic_mask(person, "upper_body", load_settings().preprocessing)
    return TryOnInputs(
        person_image=person,
        garment_image=garment,
        category="upper_body",
        agnostic_mask=mask_result.soft_mask,
        agnostic_image=mask_result.agnostic_image,
        prompt="blue shirt",
        seed=123,
        output_dir=output_dir,
    )


def test_idm_vton_availability_missing_checkpoint(tmp_path):
    settings = _settings_for_idm(tmp_path)
    engine = IDMVTonEngine(settings.idm_vton)
    status = engine.status()
    assert status.startswith("missing:")
    assert "densepose/model_final_162be9.pkl" in status
    assert "humanparsing/parsing_atr.onnx" in status
    assert "openpose/ckpts/body_pose_model.pth" in status


def test_idm_vton_command_building(tmp_path):
    settings = _settings_for_idm(tmp_path)
    engine = IDMVTonEngine(settings.idm_vton)
    context = engine.build_dataset(_make_inputs(tmp_path / "job"))
    command = context.command
    assert "accelerate.commands.launch" in command
    assert str(settings.idm_vton.entrypoint) in command
    assert "--data_dir" in command
    assert str(context.data_dir) in command
    assert "--output_dir" in command
    assert str(context.output_dir) in command
    assert (context.data_dir / "test" / "vitonhd_test_tagged.json").exists()
    assert (context.data_dir / "test_pairs.txt").read_text(encoding="utf-8").strip() == "person_0001.jpg garment_0001.jpg"


def test_idm_vton_run_with_monkeypatched_subprocess(tmp_path, monkeypatch):
    settings = _settings_for_idm(tmp_path)
    for rel_path in REQUIRED_CHECKPOINTS:
        _write_fake_file(settings.idm_vton.checkpoint_dir / rel_path)

    def fake_run(command, cwd, capture_output, text, check):
        output_dir = Path(command[command.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (768, 1024), (30, 120, 220)).save(output_dir / "person_0001.jpg")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    engine = IDMVTonEngine(settings.idm_vton)
    monkeypatch.setattr(engine, "missing_requirements", lambda: [])
    response = engine.run(_make_inputs(tmp_path / "job"))
    assert response.image.size == (768, 1024)
    assert (tmp_path / "job" / "core_output.png").exists()
    assert (tmp_path / "job" / "idm_vton_command.txt").exists()
