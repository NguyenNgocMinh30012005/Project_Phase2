from __future__ import annotations

import importlib.util
import logging
import time

from PIL import Image

from app.core.config import EngineConfig
from app.engines.base import RefineResult
from app.utils.errors import ModelUnavailableError


logger = logging.getLogger(__name__)


class FluxRefinerEngine:
    name = "flux_refiner"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._pipe = None

    def is_available(self) -> bool:
        checkpoint_dir = self.config.checkpoint_dir
        deps = ["torch", "diffusers", "transformers", "accelerate"]
        if not all(importlib.util.find_spec(dep) for dep in deps):
            return False
        return bool(
            self.config.enabled
            and (
                self.config.model_name
                or (checkpoint_dir and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()))
            )
        )

    def prepare(self) -> None:
        if not self.is_available():
            raise ModelUnavailableError(f"FLUX refiner checkpoint/model not found at {self.config.checkpoint_dir}")
        if self._pipe is not None:
            return

        try:
            import torch
            import diffusers
        except Exception as exc:
            raise ModelUnavailableError(f"FLUX refiner dependencies are not importable: {exc}") from exc

        model_id = str(self.config.checkpoint_dir) if self.config.checkpoint_dir and self.config.checkpoint_dir.exists() else self.config.model_name
        if not model_id:
            raise ModelUnavailableError("FLUX refiner model_name/checkpoint_dir is not configured.")

        dtype = torch.bfloat16 if self.config.default_strength >= 0 and torch.cuda.is_available() else torch.float32
        pipeline_cls = getattr(diffusers, "FluxFillPipeline", None) or getattr(diffusers, "FluxInpaintPipeline", None)
        if pipeline_cls is None:
            pipeline_cls = getattr(diffusers, "AutoPipelineForInpainting", None)
        if pipeline_cls is None:
            raise ModelUnavailableError(
                "Installed diffusers does not expose a FLUX/inpainting pipeline class. "
                "Install a FLUX-compatible diffusers build or disable flux_refiner."
            )

        try:
            self._pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=dtype)
            if torch.cuda.is_available():
                self._pipe = self._pipe.to("cuda")
            if hasattr(self._pipe, "enable_attention_slicing"):
                self._pipe.enable_attention_slicing()
        except Exception as exc:
            self._pipe = None
            raise ModelUnavailableError(f"FLUX refiner failed to load model '{model_id}': {exc}") from exc

    def refine(
        self,
        image: Image.Image,
        mask: Image.Image | None,
        prompt: str,
        references: dict | None = None,
        seed: int | None = None,
    ) -> RefineResult:
        start = time.perf_counter()
        self.prepare()
        base = image.convert("RGB")
        mask_image = mask.convert("L") if mask is not None else Image.new("L", base.size, 255)

        try:
            import torch

            generator = None
            if seed is not None and torch.cuda.is_available():
                generator = torch.Generator("cuda").manual_seed(seed)
            elif seed is not None:
                generator = torch.Generator().manual_seed(seed)

            kwargs = {
                "prompt": prompt,
                "image": base,
                "mask_image": mask_image,
                "num_inference_steps": self.config.steps,
                "guidance_scale": self.config.guidance_scale,
                "generator": generator,
            }
            if self.config.default_strength is not None:
                kwargs["strength"] = self.config.default_strength
            if references and references.get("garment") is not None:
                kwargs["reference_image"] = references["garment"]

            try:
                output = self._pipe(**kwargs)
            except TypeError:
                kwargs.pop("reference_image", None)
                output = self._pipe(**kwargs)
        except Exception as exc:
            raise ModelUnavailableError(f"FLUX refiner execution failed: {exc}") from exc

        refined_region = output.images[0].convert("RGB")
        if refined_region.size != base.size:
            refined_region = refined_region.resize(base.size, Image.Resampling.LANCZOS)
        out = base.copy()
        out.paste(refined_region, mask=mask_image)
        elapsed = time.perf_counter() - start
        logger.info("FLUX refiner completed in %.2fs", elapsed)
        return RefineResult(
            out,
            {
                "engine": self.name,
                "runtime_seconds": elapsed,
                "prompt": prompt,
                "seed": seed,
                "model": self.config.model_name or str(self.config.checkpoint_dir),
            },
        )
