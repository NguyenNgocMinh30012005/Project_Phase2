from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.core.paths import CONFIG_DIR, PROJECT_ROOT, resolve_project_path


class StorageConfig(BaseModel):
    inputs_dir: Path
    outputs_dir: Path
    temp_dir: Path
    public_outputs_prefix: str = "/outputs"


class ImageConfig(BaseModel):
    max_side: int = 1536
    output_width: int = 768
    output_height: int = 1024


class PreprocessingConfig(BaseModel):
    dilation_px: int = 18
    blur_radius: int = 8
    preserve_face: bool = True
    preserve_hands: bool = True
    preserve_hair: bool = True


class PipelineConfig(BaseModel):
    engine: str = "idm_vton"
    allow_mock_engine: bool = False
    save_intermediates: bool = True
    fail_on_missing_core_model: bool = True


class ModelRuntimeConfig(BaseModel):
    device: str = "cuda"
    precision: str = "bf16"


class EngineConfig(BaseModel):
    enabled: bool = True
    repo_path: Path | None = None
    checkpoint_dir: Path | None = None
    entrypoint: str | None = None
    model_name: str | None = None
    base_model: str | None = None
    lora_path: Path | None = None
    default_width: int = 768
    default_height: int = 1024
    steps: int = 30
    guidance_scale: float = 2.0
    default_strength: float = 0.35
    lora_scale: float = 1.0


class RepairConfig(BaseModel):
    enabled: bool = True
    mask_dilation_px: int = 16
    mask_blur_radius: int = 8


class QualityConfig(BaseModel):
    min_output_width: int = 256
    min_output_height: int = 256
    background_change_threshold: float = 0.18
    garment_change_threshold: float = 0.04
    artifact_threshold: float = 0.35


class RefinementConfig(BaseModel):
    refine_only_masked_region: bool = True
    preserve_face: bool = True
    preserve_background: bool = True
    default_prompt: str = "Refine garment boundaries while preserving identity."


class AppConfig(BaseModel):
    name: str = "Virtual Try-On"
    environment: str = "development"
    debug: bool = True


class Settings(BaseModel):
    app: AppConfig
    storage: StorageConfig
    image: ImageConfig
    preprocessing: PreprocessingConfig
    pipeline: PipelineConfig
    runtime: ModelRuntimeConfig
    idm_vton: EngineConfig
    flux_refiner: EngineConfig
    catvton: EngineConfig
    klein_tryon_lora: EngineConfig
    repair: RepairConfig
    quality: QualityConfig
    refinement: RefinementConfig
    repair_regions: list[str] = Field(default_factory=list)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    for key in ["inputs_dir", "outputs_dir", "temp_dir"]:
        if key in config.get("storage", {}):
            config["storage"][key] = resolve_project_path(config["storage"][key])

    for section in ["idm_vton", "flux_refiner", "catvton", "klein_tryon_lora"]:
        section_config = config.get(section, {})
        for key in ["repo_path", "checkpoint_dir", "lora_path"]:
            if section_config.get(key):
                section_config[key] = resolve_project_path(section_config[key])
    return config


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    engine = os.getenv("TRYON_ENGINE")
    if engine:
        config.setdefault("pipeline", {})["engine"] = engine

    allow_mock = os.getenv("TRYON_ALLOW_MOCK")
    if allow_mock is not None:
        config.setdefault("pipeline", {})["allow_mock_engine"] = allow_mock.lower() in {"1", "true", "yes", "on"}

    device = os.getenv("TRYON_DEVICE")
    if device:
        config["device"] = device
    return config


def load_settings() -> Settings:
    config: dict[str, Any] = {}
    for filename in ["default.yaml", "models.yaml", "pipeline.yaml"]:
        config = _deep_merge(config, _read_yaml(CONFIG_DIR / filename))

    config = _apply_env_overrides(config)
    config = _resolve_paths(config)

    return Settings(
        app=AppConfig(**config.get("app", {})),
        storage=StorageConfig(**config.get("storage", {})),
        image=ImageConfig(**config.get("image", {})),
        preprocessing=PreprocessingConfig(**config.get("preprocessing", {})),
        pipeline=PipelineConfig(**config.get("pipeline", {})),
        runtime=ModelRuntimeConfig(device=config.get("device", "cuda"), precision=config.get("precision", "bf16")),
        idm_vton=EngineConfig(**config.get("idm_vton", {})),
        flux_refiner=EngineConfig(**config.get("flux_refiner", {})),
        catvton=EngineConfig(**config.get("catvton", {})),
        klein_tryon_lora=EngineConfig(**config.get("klein_tryon_lora", {})),
        repair=RepairConfig(**config.get("repair", {})),
        quality=QualityConfig(**config.get("quality", {})),
        refinement=RefinementConfig(**config.get("refinement", {})),
        repair_regions=config.get("repair_regions", []),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
