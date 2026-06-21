from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import EngineConfig
from app.engines.base import TryOnInputs, TryOnResult
from app.utils.errors import ModelUnavailableError


logger = logging.getLogger(__name__)


class CatVTonEngine:
    name = "catvton"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def is_available(self) -> bool:
        checkpoint_dir = self.config.checkpoint_dir
        return bool(
            self.config.enabled
            and checkpoint_dir
            and checkpoint_dir.exists()
            and any(checkpoint_dir.iterdir())
        )

    def prepare(self) -> None:
        if not self.is_available():
            raise ModelUnavailableError(f"CatVTON checkpoint not found at {self.config.checkpoint_dir}")

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        self.prepare()
        raise ModelUnavailableError(
            "CatVTON adapter is reserved as a baseline. Configure catvton.entrypoint before running this engine."
        )
