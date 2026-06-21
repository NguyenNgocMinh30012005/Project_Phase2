from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from app.core.config import EngineConfig
from app.engines.base import TryOnInputs, TryOnResult
from app.utils.errors import EngineExecutionError, ModelUnavailableError


logger = logging.getLogger(__name__)


class IDMVTonEngine:
    name = "idm_vton"

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
            raise ModelUnavailableError(f"IDM-VTON checkpoint not found at {self.config.checkpoint_dir}")

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        start = time.perf_counter()
        self.prepare()

        if not self.config.entrypoint:
            raise ModelUnavailableError(
                "IDM-VTON checkpoint is present, but no entrypoint is configured in configs/models.yaml "
                "for idm_vton.entrypoint."
            )

        output_path = (inputs.output_dir or Path.cwd()) / "core_output.png"
        command = [
            "python",
            self.config.entrypoint,
            "--person",
            str(inputs.extra["person_path"]),
            "--garment",
            str(inputs.extra["garment_path"]),
            "--mask",
            str(inputs.extra["mask_path"]),
            "--category",
            inputs.category,
            "--output",
            str(output_path),
        ]
        if inputs.prompt:
            command.extend(["--prompt", inputs.prompt])
        if inputs.seed is not None:
            command.extend(["--seed", str(inputs.seed)])

        logger.info("Running IDM-VTON command: %s", " ".join(command))
        completed = subprocess.run(command, cwd=self.config.repo_path, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise EngineExecutionError(
                "IDM-VTON execution failed. "
                f"stdout={completed.stdout[-1000:]} stderr={completed.stderr[-1000:]}"
            )
        if not output_path.exists():
            raise EngineExecutionError(f"IDM-VTON finished but did not create output at {output_path}")

        from app.utils.image_io import open_rgb

        elapsed = time.perf_counter() - start
        logger.info("IDM-VTON completed in %.2fs", elapsed)
        return TryOnResult(open_rgb(output_path), {"engine": self.name, "runtime_seconds": elapsed})
