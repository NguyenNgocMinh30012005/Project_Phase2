from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import EngineConfig
from app.engines.base import TryOnInputs, TryOnResult
from app.preprocessing.image_loader import fit_to_canvas
from app.utils.errors import EngineExecutionError, ModelUnavailableError
from app.utils.image_io import open_rgb, save_image


logger = logging.getLogger(__name__)


REQUIRED_CHECKPOINTS = (
    "densepose/model_final_162be9.pkl",
    "humanparsing/parsing_atr.onnx",
    "humanparsing/parsing_lip.onnx",
    "openpose/ckpts/body_pose_model.pth",
)


@dataclass(frozen=True)
class IDMVTonRunContext:
    data_dir: Path
    output_dir: Path
    person_name: str
    garment_name: str
    expected_output: Path
    command: list[str]


class IDMVTonEngine:
    name = "idm_vton"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        if not self.config.enabled:
            missing.append("idm_vton.enabled is false")
        if not self.config.repo_path or not self.config.repo_path.exists():
            missing.append(f"repo_path not found: {self.config.repo_path}")
        if not self.config.entrypoint or not self.config.entrypoint.exists():
            missing.append(f"entrypoint not found: {self.config.entrypoint}")
        if not self.config.checkpoint_dir:
            missing.append("checkpoint_dir is not configured")
        else:
            if not self.config.checkpoint_dir.exists():
                missing.append(f"IDM-VTON checkpoint not found at {self.config.checkpoint_dir}")
            for rel_path in REQUIRED_CHECKPOINTS:
                checkpoint = self.config.checkpoint_dir / rel_path
                if not checkpoint.exists():
                    missing.append(f"missing checkpoint: {rel_path}")
                elif checkpoint.stat().st_size < 1024:
                    missing.append(f"checkpoint looks incomplete: {rel_path}")
        try:
            import accelerate  # noqa: F401
        except Exception:
            missing.append("python package missing: accelerate")
        return missing

    def status(self) -> str:
        missing = self.missing_requirements()
        if missing:
            return "missing: " + "; ".join(missing)
        return "available"

    def is_available(self) -> bool:
        return not self.missing_requirements()

    def prepare(self) -> None:
        missing = self.missing_requirements()
        if missing:
            raise ModelUnavailableError(
                "IDM-VTON is not available. " + "; ".join(missing)
            )

    def build_dataset(self, inputs: TryOnInputs) -> IDMVTonRunContext:
        if inputs.output_dir is None:
            raise EngineExecutionError("IDM-VTON requires inputs.output_dir for dataset staging.")
        job_dir = Path(inputs.output_dir)
        data_dir = job_dir / "idm_vton_dataset"
        output_dir = job_dir / "idm_vton_result"
        test_dir = data_dir / "test"

        for folder in ["image", "cloth", "agnostic-mask", "image-densepose"]:
            (test_dir / folder).mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        person_name = "person_0001.jpg"
        garment_name = "garment_0001.jpg"
        width = self.config.default_width
        height = self.config.default_height

        person = fit_to_canvas(inputs.person_image, width, height)
        garment = fit_to_canvas(inputs.garment_image, width, height)
        save_image(person, test_dir / "image" / person_name)
        save_image(garment, test_dir / "cloth" / garment_name)

        mask_name = person_name.replace(".jpg", "_mask.png")
        mask = inputs.agnostic_mask.convert("L").resize((width, height))
        save_image(mask, test_dir / "agnostic-mask" / mask_name)

        if inputs.densepose_image is not None:
            densepose = fit_to_canvas(inputs.densepose_image, width, height)
        else:
            logger.warning("No densepose image provided; using person image as densepose placeholder.")
            densepose = person
        save_image(densepose, test_dir / "image-densepose" / person_name)
        save_image(densepose, job_dir / "densepose.png")

        tags = self._tags_for_category(inputs.category, inputs.prompt)
        manifest = {"data": [{"file_name": garment_name, "category_name": "TOPS", "tag_info": tags}]}
        (test_dir / "vitonhd_test_tagged.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (data_dir / "test_pairs.txt").write_text(f"{person_name} {garment_name}\n", encoding="utf-8")

        command = self.build_command(data_dir, output_dir, inputs.seed)
        return IDMVTonRunContext(
            data_dir=data_dir,
            output_dir=output_dir,
            person_name=person_name,
            garment_name=garment_name,
            expected_output=output_dir / person_name,
            command=command,
        )

    def build_command(self, data_dir: Path, output_dir: Path, seed: int | None) -> list[str]:
        entrypoint = self.config.entrypoint
        if entrypoint is None:
            raise ModelUnavailableError("IDM-VTON entrypoint is not configured.")
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            str(entrypoint),
            "--pretrained_model_name_or_path",
            self.config.model_name or "yisol/IDM-VTON",
            "--width",
            str(self.config.default_width),
            "--height",
            str(self.config.default_height),
            "--num_inference_steps",
            str(self.config.steps),
            "--output_dir",
            str(output_dir),
            "--unpaired",
            "--data_dir",
            str(data_dir),
            "--seed",
            str(seed if seed is not None else 42),
            "--test_batch_size",
            "1",
            "--guidance_scale",
            str(self.config.guidance_scale),
        ]
        return command

    @staticmethod
    def _tags_for_category(category: str, prompt: str | None) -> list[dict[str, str | None]]:
        item = {
            "upper_body": "shirts",
            "lower_body": "pants",
            "dress": "dress",
            "full_outfit": "outfit",
        }.get(category, "shirts")
        prompt_value = prompt or item
        return [
            {"tag_name": "item", "tag_category": item},
            {"tag_name": "sleeveLength", "tag_category": "regular"},
            {"tag_name": "neckLine", "tag_category": "regular"},
            {"tag_name": "details", "tag_category": prompt_value[:80]},
            {"tag_name": "colors", "tag_category": None},
            {"tag_name": "textures", "tag_category": None},
        ]

    def run(self, inputs: TryOnInputs) -> TryOnResult:
        start = time.perf_counter()
        self.prepare()
        context = self.build_dataset(inputs)
        job_dir = inputs.output_dir or Path.cwd()
        command_text = " ".join(str(part) for part in context.command)
        (job_dir / "idm_vton_command.txt").write_text(command_text, encoding="utf-8")
        logger.info("Running IDM-VTON command: %s", command_text)
        completed = subprocess.run(
            context.command,
            cwd=self.config.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        (job_dir / "idm_vton_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
        (job_dir / "idm_vton_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            raise EngineExecutionError(
                "IDM-VTON execution failed. "
                f"stdout={completed.stdout[-1000:]} stderr={completed.stderr[-1000:]}"
            )
        if not context.expected_output.exists():
            raise EngineExecutionError(f"IDM-VTON finished but did not create output at {context.expected_output}")

        core_output = Path(job_dir) / "core_output.png"
        image = open_rgb(context.expected_output)
        save_image(image, core_output)
        elapsed = time.perf_counter() - start
        logger.info("IDM-VTON completed in %.2fs", elapsed)
        return TryOnResult(
            image,
            {
                "engine": self.name,
                "runtime_seconds": elapsed,
                "command": command_text,
                "data_dir": str(context.data_dir),
                "output_dir": str(context.output_dir),
            },
        )
