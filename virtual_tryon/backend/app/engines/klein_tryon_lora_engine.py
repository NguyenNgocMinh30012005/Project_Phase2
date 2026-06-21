from __future__ import annotations

from app.core.config import EngineConfig
from app.engines.base import TryOnInputs, TryOnResult
from app.utils.errors import ModelUnavailableError


class KleinTryOnLoraEngine:
    name = "klein_tryon_lora"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def is_available(self) -> bool:
        return bool(
            self.config.enabled
            and self.config.base_model
            and self.config.lora_path
            and self.config.lora_path.exists()
        )

    def prepare(self) -> None:
        if not self.is_available():
            raise ModelUnavailableError(f"Flux Klein Try-On LoRA not found at {self.config.lora_path}")

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        self.prepare()
        prompt = inputs.prompt or ""
        if not prompt.startswith("TRYON"):
            prompt = f"TRYON {prompt}".strip()
        raise ModelUnavailableError(
            "Klein Try-On LoRA adapter is experimental. Wire a FLUX image generation backend before use. "
            f"Default settings: steps={self.config.steps}, guidance_scale={self.config.guidance_scale}, "
            f"lora_scale={self.config.lora_scale}. Prompt would be: {prompt}"
        )
