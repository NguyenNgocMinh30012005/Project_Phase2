from __future__ import annotations

import logging
import time

from PIL import Image, ImageEnhance, ImageFilter

from app.core.config import EngineConfig
from app.engines.base import RefineResult
from app.utils.errors import ModelUnavailableError


logger = logging.getLogger(__name__)


class FluxRefinerEngine:
    name = "flux_refiner"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def is_available(self) -> bool:
        checkpoint_dir = self.config.checkpoint_dir
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

        # Placeholder deterministic local repair until the FLUX image-edit backend is wired.
        sharpened = base.filter(ImageFilter.UnsharpMask(radius=1.2))
        refined_region = ImageEnhance.Sharpness(sharpened).enhance(1.08)
        if mask:
            out = base.copy()
            out.paste(refined_region, mask=mask.convert("L"))
        else:
            out = refined_region
        elapsed = time.perf_counter() - start
        logger.info("FLUX refiner placeholder completed in %.2fs", elapsed)
        return RefineResult(
            out,
            {
                "engine": self.name,
                "runtime_seconds": elapsed,
                "prompt": prompt,
                "seed": seed,
                "placeholder": True,
            },
        )
