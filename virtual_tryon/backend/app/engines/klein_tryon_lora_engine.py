from __future__ import annotations

from pathlib import Path

from app.core.config import EngineConfig
from app.engines.base import TryOnInputs, TryOnResult
from app.utils.errors import ModelUnavailableError


def build_klein_tryon_prompt(
    user_prompt: str | None,
    *,
    person_description: str = "a person",
    top_description: str = "the top garment",
    bottom_description: str = "the bottom garment",
) -> str:
    prompt = (user_prompt or "").strip()
    if prompt.startswith("TRYON"):
        return prompt
    if prompt:
        return f"TRYON {prompt}"
    return (
        f"TRYON {person_description}. Replace the outfit with {top_description} and {bottom_description} "
        "as shown in the reference images. The final image is a full body shot."
    )


class KleinTryOnLoraEngine:
    name = "klein_tryon_lora"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        if not self.config.enabled:
            missing.append("klein_tryon_lora.enabled is false")
        if not self.config.base_model:
            missing.append("base_model is not configured")
        if not self.config.lora_path or not self.config.lora_path.exists():
            missing.append(f"lora_path not found: {self.config.lora_path}")
        return missing

    def status(self) -> str:
        missing = self.missing_requirements()
        if missing:
            return "unavailable: " + "; ".join(missing)
        return "available"

    def is_available(self) -> bool:
        return not self.missing_requirements()

    def prepare(self) -> None:
        missing = self.missing_requirements()
        if missing:
            raise ModelUnavailableError("Klein Try-On LoRA is not available. " + "; ".join(missing))

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        prompt = build_klein_tryon_prompt(inputs.prompt)
        if inputs.output_dir:
            output_dir = Path(inputs.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "klein_lora_prompt.txt").write_text(prompt, encoding="utf-8")
        self.prepare()
        if inputs.category == "full_outfit" and not inputs.extra.get("has_full_outfit_inputs"):
            raise ModelUnavailableError("Klein Try-On LoRA requires both top and bottom garments for full_outfit.")
        raise ModelUnavailableError(
            "Klein Try-On LoRA adapter is configured as a benchmark baseline, but no FLUX LoRA "
            f"execution backend is wired. steps={self.config.steps}, guidance_scale={self.config.guidance_scale}, "
            f"lora_scale={self.config.lora_scale}. Prompt would be: {prompt}"
        )
