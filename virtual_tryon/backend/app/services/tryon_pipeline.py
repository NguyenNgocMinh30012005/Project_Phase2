from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.core.config import Settings
from app.core.paths import REPO_ROOT
from app.engines.base import TryOnInputs
from app.engines.factory import create_refiner, create_repair_engine, create_tryon_engine
from app.evaluation.quality_checks import build_quality_report, run_quality_checks
from app.preprocessing.agnostic_mask import create_agnostic_mask
from app.preprocessing.densepose import DensePoseEstimator
from app.preprocessing.garment_segmenter import GarmentSegmenter
from app.preprocessing.human_parser import HumanParser
from app.preprocessing.image_loader import fit_to_canvas
from app.preprocessing.refine_mask import build_refine_masks, select_refine_mask
from app.prompts.prompt_builder import build_prompt
from app.prompts.prompt_types import EngineMode, PromptBuildResult, PromptVariant
from app.prompts.testcase_prompt_library import get_testcase
from app.schemas.tryon import DebugUrls, QualityScores, TryOnCategory, TryOnResponse
from app.services.artifact_service import write_artifact_manifest
from app.services.storage_service import StorageService
from app.utils.errors import InputValidationError, ModelUnavailableError
from app.utils.image_io import save_image
from app.utils.seed import normalize_seed, set_seed


logger = logging.getLogger(__name__)


@dataclass
class PipelineRequest:
    job_id: str
    person_image: Image.Image
    garment_top: Image.Image | None
    garment_bottom: Image.Image | None
    garment_dress: Image.Image | None
    category: TryOnCategory
    prompt: str | None
    use_refiner: bool
    repair_mode: bool
    seed: int | None = None
    engine_mode: str | None = None
    testcase_id: str | None = None
    prompt_variant: str = "default"
    auto_prompt: bool = False


class TryOnPipeline:
    def __init__(self, settings: Settings, storage: StorageService) -> None:
        self.settings = settings
        self.storage = storage
        self.segmenter = GarmentSegmenter()
        self.human_parser = HumanParser()
        self.densepose = DensePoseEstimator()

    def _select_garment(self, request: PipelineRequest) -> Image.Image:
        if request.category == "upper_body" and request.garment_top:
            return request.garment_top
        if request.category == "lower_body" and request.garment_bottom:
            return request.garment_bottom
        if request.category == "dress" and request.garment_dress:
            return request.garment_dress
        if request.category == "full_outfit":
            return request.garment_dress or request.garment_top or request.garment_bottom  # type: ignore[return-value]
        raise InputValidationError(f"No garment image provided for category '{request.category}'.")

    def _settings_for_request(self, request: PipelineRequest) -> Settings:
        settings = self.settings
        if not request.engine_mode:
            return settings
        mode = request.engine_mode
        mode_settings = settings.model_copy(deep=True)
        if mode == "idm_vton":
            mode_settings.pipeline.engine = "mock" if settings.pipeline.engine == "mock" else "idm_vton"
        elif mode == "idm_mask_expanded":
            mode_settings.pipeline.engine = "mock" if settings.pipeline.engine == "mock" else "idm_vton"
            mode_settings.mask_experiments.upper_body_expand_hem.enabled = True
        elif mode == "idm_vton_flux":
            mode_settings.pipeline.engine = "mock" if settings.pipeline.engine == "mock" else "idm_vton"
            mode_settings.flux_refiner.enabled = True
            mode_settings.refinement.enabled = True
        elif mode == "idm_mask_expanded_flux":
            mode_settings.pipeline.engine = "mock" if settings.pipeline.engine == "mock" else "idm_vton"
            mode_settings.mask_experiments.upper_body_expand_hem.enabled = True
            mode_settings.flux_refiner.enabled = True
            mode_settings.refinement.enabled = True
        elif mode == "klein_lora":
            mode_settings.pipeline.engine = "klein_tryon_lora"
            mode_settings.klein_tryon_lora.enabled = True
        elif mode == "catvton":
            mode_settings.pipeline.engine = "catvton"
        else:
            raise InputValidationError(f"Unsupported engine_mode '{mode}'.")
        return mode_settings

    @staticmethod
    def _prompt_engine_mode(request: PipelineRequest, settings: Settings) -> EngineMode:
        mapping = {
            "idm_vton": EngineMode.IDM,
            "idm_mask_expanded": EngineMode.IDM_MASK_EXPANDED,
            "idm_vton_flux": EngineMode.IDM_MASK_EXPANDED_FLUX,
            "idm_mask_expanded_flux": EngineMode.IDM_MASK_EXPANDED_FLUX,
            "klein_lora": EngineMode.KLEIN_LORA,
            "catvton": EngineMode.CATVTON,
        }
        if request.engine_mode:
            return mapping[request.engine_mode]
        if settings.pipeline.engine == "klein_tryon_lora":
            return EngineMode.KLEIN_LORA
        if settings.pipeline.engine == "catvton":
            return EngineMode.CATVTON
        return EngineMode.IDM

    def _resolve_prompts(
        self,
        request: PipelineRequest,
        settings: Settings,
    ) -> tuple[str | None, str | None, PromptBuildResult | None]:
        engine_mode = self._prompt_engine_mode(request, settings)
        if request.auto_prompt:
            if not request.testcase_id:
                raise InputValidationError("testcase_id is required when auto_prompt=true.")
            try:
                testcase = get_testcase(request.testcase_id)
                variant = PromptVariant(request.prompt_variant)
            except (KeyError, ValueError) as exc:
                raise InputValidationError(str(exc)) from exc
            prompt_request = testcase.build_request(engine_mode, variant)
            if request.prompt:
                prompt_request.extra_user_instruction = request.prompt
            result = build_prompt(prompt_request)
            return result.core_prompt or result.positive_prompt, result.refine_prompt, result

        prompt = request.prompt
        if engine_mode == EngineMode.KLEIN_LORA and prompt:
            from app.engines.klein_prompt_builder import build_klein_tryon_prompt

            prompt = build_klein_tryon_prompt(
                None,
                None,
                None,
                request.category,
                extra_instruction=prompt,
            )
        if prompt is None and settings.pipeline.engine != "klein_tryon_lora":
            prompt = settings.refinement.default_prompt
        return prompt, None, None

    def _save_prompt_artifacts(
        self,
        job_dir: Path,
        core_prompt: str | None,
        refine_prompt: str | None,
        prompt_result: PromptBuildResult | None,
    ) -> dict[str, Path | None]:
        paths: dict[str, Path | None] = {
            "core": None,
            "refine": None,
            "metadata": None,
        }
        if core_prompt:
            paths["core"] = job_dir / "prompt_core.txt"
            paths["core"].write_text(core_prompt, encoding="utf-8")
            (job_dir / "prompt.txt").write_text(core_prompt, encoding="utf-8")
        if refine_prompt:
            paths["refine"] = job_dir / "prompt_refine.txt"
            paths["refine"].write_text(refine_prompt, encoding="utf-8")
        if prompt_result:
            if prompt_result.negative_prompt:
                (job_dir / "negative_prompt.txt").write_text(
                    prompt_result.negative_prompt,
                    encoding="utf-8",
                )
            paths["metadata"] = job_dir / "prompt_metadata.json"
            paths["metadata"].write_text(
                json.dumps(prompt_result.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return paths

    def validate_inputs(self, request: PipelineRequest) -> None:
        if request.person_image is None:
            raise InputValidationError("person_image is required.")
        if not any([request.garment_top, request.garment_bottom, request.garment_dress]):
            raise InputValidationError("At least one garment image is required.")
        self._select_garment(request)

    @staticmethod
    def _config_hash(payload: dict) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _commit_sha() -> str:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def run(self, request: PipelineRequest) -> TryOnResponse:
        self.validate_inputs(request)
        settings = self._settings_for_request(request)
        seed = normalize_seed(request.seed)
        set_seed(seed)

        job_dir = self.storage.job_dir(request.job_id)
        width = settings.image.output_width
        height = settings.image.output_height
        prompt, refine_prompt, prompt_result = self._resolve_prompts(request, settings)
        prompt_paths = self._save_prompt_artifacts(job_dir, prompt, refine_prompt, prompt_result)

        person = fit_to_canvas(request.person_image, width, height)
        garment = fit_to_canvas(self._select_garment(request), width, height)
        garment_top = fit_to_canvas(request.garment_top, width, height) if request.garment_top else None
        garment_bottom = fit_to_canvas(request.garment_bottom, width, height) if request.garment_bottom else None
        garment_dress = fit_to_canvas(request.garment_dress, width, height) if request.garment_dress else None
        save_image(person, job_dir / "person.png")
        save_image(garment, job_dir / "garment.png")
        if garment_top is not None:
            save_image(garment_top, job_dir / "garment_top.png")
        if garment_bottom is not None:
            save_image(garment_bottom, job_dir / "garment_bottom.png")
        if garment_dress is not None:
            save_image(garment_dress, job_dir / "garment_dress.png")

        human_parse = self.human_parser.parse(person, job_dir)
        densepose = self.densepose.estimate(person, job_dir)
        mask_experiment = settings.mask_experiments.upper_body_expand_hem
        mask_result = create_agnostic_mask(
            person,
            request.category,
            settings.preprocessing,
            mask_experiment,
        )
        garment_seg = self.segmenter.segment(garment, (width, height))

        save_image(mask_result.raw_mask, job_dir / "raw_mask.png")
        save_image(mask_result.dilated_mask, job_dir / "agnostic_mask.png")
        save_image(mask_result.soft_mask, job_dir / "soft_mask.png")
        save_image(mask_result.preview, job_dir / "mask_preview.png")
        save_image(mask_result.agnostic_image, job_dir / "agnostic.png")
        experiment_debug_images = {
            "mask_original_upper_body.png": mask_result.original_upper_body_mask,
            "mask_expanded_upper_body.png": mask_result.expanded_upper_body_mask,
            "mask_diff_upper_body.png": mask_result.diff_upper_body_mask,
            "mask_original_upper_body_overlay.png": mask_result.original_upper_body_overlay,
            "mask_expanded_upper_body_overlay.png": mask_result.expanded_upper_body_overlay,
            "mask_diff_upper_body_overlay.png": mask_result.diff_upper_body_overlay,
        }
        for filename, image in experiment_debug_images.items():
            if image is not None:
                save_image(image, job_dir / filename)
        save_image(garment_seg.cloth_mask, job_dir / "cloth_mask.png")
        save_image(garment_seg.normalized_crop, job_dir / "garment_normalized.png")
        if densepose.densepose_path is None:
            save_image(person, job_dir / "densepose_placeholder.png")

        engine = create_tryon_engine(settings)
        inputs = TryOnInputs(
            person_image=person,
            garment_image=garment_seg.normalized_crop,
            category=request.category,
            agnostic_mask=mask_result.soft_mask,
            agnostic_image=mask_result.agnostic_image,
            prompt=prompt,
            seed=seed,
            output_dir=job_dir,
            extra={
                "person_path": job_dir / "person.png",
                "garment_path": job_dir / "garment.png",
                "garment_top_image": garment_top,
                "garment_bottom_image": garment_bottom,
                "garment_dress_image": garment_dress,
                "garment_top_path": job_dir / "garment_top.png" if garment_top is not None else None,
                "garment_bottom_path": job_dir / "garment_bottom.png" if garment_bottom is not None else None,
                "garment_dress_path": job_dir / "garment_dress.png" if garment_dress is not None else None,
                "mask_path": job_dir / "agnostic_mask.png",
                "human_parse": human_parse.warning,
                "densepose": densepose.warning,
            },
        )

        try:
            core = engine.run(inputs)
        except ModelUnavailableError:
            raise

        core_path = save_image(core.image, job_dir / "core_output.png")
        core_image = core.image
        current_image = core_image

        refine_masks = build_refine_masks(person, mask_result.soft_mask, settings.refinement)
        save_image(refine_masks.garment_refine_mask, job_dir / "garment_refine_mask.png")
        save_image(refine_masks.boundary_refine_mask, job_dir / "boundary_refine_mask.png")
        save_image(refine_masks.safe_refine_mask, job_dir / "safe_refine_mask.png")
        save_image(refine_masks.garment_overlay, job_dir / "garment_refine_mask_overlay.png")
        save_image(refine_masks.boundary_overlay, job_dir / "boundary_refine_mask_overlay.png")
        save_image(refine_masks.safe_overlay, job_dir / "safe_refine_mask_overlay.png")
        active_refine_mask = select_refine_mask(refine_masks, settings.refinement.mask_mode)

        quality: QualityScores = run_quality_checks(
            person,
            core_image,
            garment_seg.normalized_crop,
            active_refine_mask,
            settings.quality,
        )

        refined_path: Path | None = None
        refined_image: Image.Image | None = None
        refine_notes = list(refine_masks.notes)
        core_engine_name = getattr(engine, "name", "unknown")
        engine_status = {
            "idm_vton": "success" if core_engine_name in {"idm_vton", "mock"} else "skipped",
            "flux_refiner": "skipped",
            "catvton": "success" if core_engine_name == "catvton" else "skipped",
            "klein_lora": "success" if core_engine_name == "klein_tryon_lora" else "skipped",
        }
        refiner_status = "skipped"
        use_refiner = request.use_refiner or request.engine_mode in {
            "idm_vton_flux",
            "idm_mask_expanded_flux",
        }
        if use_refiner and settings.refinement.enabled and settings.flux_refiner.enabled:
            refiner = create_refiner(settings)
            try:
                refined = refiner.refine(
                    core_image,
                    active_refine_mask,
                    refine_prompt or prompt,
                    references={"person": person, "garment": garment_seg.normalized_crop},
                    seed=seed,
                )
                refined_image = refined.image
                refined_path = save_image(refined_image, job_dir / "refined_output.png")
                refiner_status = "success"
                engine_status["flux_refiner"] = "success"
            except ModelUnavailableError as exc:
                message = f"Refiner unavailable; returning core output. {exc}"
                quality.notes.append(message)
                refine_notes.append(message)
                (job_dir / "flux_refiner_error.txt").write_text(message, encoding="utf-8")
                refiner_status = "skipped"
                engine_status["flux_refiner"] = "skipped"
                logger.warning("Skipping refiner: %s", exc)
            except Exception as exc:
                message = f"Refiner failed; returning core output. {exc}"
                quality.notes.append(message)
                refine_notes.append(message)
                (job_dir / "flux_refiner_error.txt").write_text(message, encoding="utf-8")
                refiner_status = "failed"
                engine_status["flux_refiner"] = "failed"
                logger.exception("Refiner failed; falling back to core output.")

        quality_report = build_quality_report(
            person,
            core_image,
            refined_image,
            active_refine_mask,
            settings.quality,
            refine_notes=refine_notes,
            engine_status=engine_status,
        )
        if quality_report["final_choice"] == "refined" and refined_image is not None:
            current_image = refined_image
        else:
            current_image = core_image

        if request.repair_mode and settings.repair.enabled and refined_image is not None and quality_report["final_choice"] == "refined":
            repair_engine = create_repair_engine(settings)
            repaired = repair_engine.refine(current_image, active_refine_mask, prompt, seed=seed)
            current_image = repaired.image
            refined_path = save_image(current_image, job_dir / "refined_output.png")
            quality_report["repair"] = repaired.metadata

        result_path = save_image(current_image, job_dir / "result.png")
        quality_report_path = self.storage.save_json(request.job_id, "quality_report.json", quality_report)
        mask_config = {
            "preprocessing": settings.preprocessing.model_dump(mode="json"),
            "upper_body_expand_hem": mask_experiment.model_dump(mode="json"),
        }
        engine_config = {
            "pipeline_engine": settings.pipeline.engine,
            "runtime": settings.runtime.model_dump(mode="json"),
            "engine": (
                settings.idm_vton.model_dump(mode="json")
                if settings.pipeline.engine in {"idm_vton", "mock"}
                else getattr(settings, settings.pipeline.engine).model_dump(mode="json")
            ),
        }
        metadata = {
            "job_id": request.job_id,
            "seed": seed,
            "mask_config_hash": self._config_hash(mask_config),
            "engine_config_hash": self._config_hash(engine_config),
            "mask_config": mask_config,
            "engine_config": engine_config,
            "commit_sha": self._commit_sha(),
            "category": request.category,
            "engine": getattr(engine, "name", "unknown"),
            "prompt": prompt,
            "prompt_variant": request.prompt_variant,
            "testcase_id": request.testcase_id,
            "auto_prompt": request.auto_prompt,
            "prompt_hash": prompt_result.metadata.get("prompt_hash") if prompt_result else None,
            "quality": quality.model_dump(),
            "quality_report": quality_report,
            "refiner_status": refiner_status,
            "engine_status": engine_status,
            "core_metadata": core.metadata,
        }
        self.storage.save_json(request.job_id, "metadata.json", metadata)
        write_artifact_manifest(
            request.job_id,
            job_dir,
            settings.storage.public_outputs_prefix,
        )

        return TryOnResponse(
            job_id=request.job_id,
            status="completed",
            result_url=self.storage.public_url(result_path),
            debug=DebugUrls(
                mask_url=self.storage.public_url(job_dir / "mask_preview.png"),
                mask_urls=[
                    url
                    for url in [
                        self.storage.public_url(job_dir / "mask_preview.png"),
                        self.storage.public_url(job_dir / "garment_refine_mask_overlay.png"),
                        self.storage.public_url(job_dir / "boundary_refine_mask_overlay.png"),
                        self.storage.public_url(job_dir / "safe_refine_mask_overlay.png"),
                    ]
                    if url
                ],
                agnostic_url=self.storage.public_url(job_dir / "agnostic.png"),
                core_output_url=self.storage.public_url(core_path),
                refined_output_url=self.storage.public_url(refined_path),
                quality_report_url=self.storage.public_url(quality_report_path),
                refine_mask_url=self.storage.public_url(job_dir / "safe_refine_mask_overlay.png"),
                prompt_core_url=self.storage.public_url(prompt_paths["core"]),
                prompt_refine_url=self.storage.public_url(prompt_paths["refine"]),
                prompt_metadata_url=self.storage.public_url(prompt_paths["metadata"]),
            ),
            quality=quality,
            seed=seed,
        )
