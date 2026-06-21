from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.core.config import Settings
from app.engines.base import TryOnInputs
from app.engines.factory import create_refiner, create_repair_engine, create_tryon_engine
from app.evaluation.quality_checks import run_quality_checks
from app.preprocessing.agnostic_mask import create_agnostic_mask
from app.preprocessing.densepose import DensePoseEstimator
from app.preprocessing.garment_segmenter import GarmentSegmenter
from app.preprocessing.human_parser import HumanParser
from app.preprocessing.image_loader import fit_to_canvas
from app.schemas.tryon import DebugUrls, QualityScores, TryOnCategory, TryOnResponse
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

    def validate_inputs(self, request: PipelineRequest) -> None:
        if request.person_image is None:
            raise InputValidationError("person_image is required.")
        if not any([request.garment_top, request.garment_bottom, request.garment_dress]):
            raise InputValidationError("At least one garment image is required.")
        self._select_garment(request)

    def run(self, request: PipelineRequest) -> TryOnResponse:
        self.validate_inputs(request)
        seed = normalize_seed(request.seed)
        set_seed(seed)

        job_dir = self.storage.job_dir(request.job_id)
        width = self.settings.image.output_width
        height = self.settings.image.output_height
        prompt = request.prompt or self.settings.refinement.default_prompt

        person = fit_to_canvas(request.person_image, width, height)
        garment = fit_to_canvas(self._select_garment(request), width, height)
        save_image(person, job_dir / "person.png")
        save_image(garment, job_dir / "garment.png")

        human_parse = self.human_parser.parse(person, job_dir)
        densepose = self.densepose.estimate(person, job_dir)
        mask_result = create_agnostic_mask(person, request.category, self.settings.preprocessing)
        garment_seg = self.segmenter.segment(garment, (width, height))

        save_image(mask_result.raw_mask, job_dir / "raw_mask.png")
        save_image(mask_result.dilated_mask, job_dir / "agnostic_mask.png")
        save_image(mask_result.soft_mask, job_dir / "soft_mask.png")
        save_image(mask_result.preview, job_dir / "mask_preview.png")
        save_image(mask_result.agnostic_image, job_dir / "agnostic.png")
        save_image(garment_seg.cloth_mask, job_dir / "cloth_mask.png")
        save_image(garment_seg.normalized_crop, job_dir / "garment_normalized.png")

        engine = create_tryon_engine(self.settings)
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
        current_image = core.image

        quality: QualityScores = run_quality_checks(
            person,
            current_image,
            garment_seg.normalized_crop,
            mask_result.soft_mask,
            self.settings.quality,
        )

        refined_path: Path | None = None
        if request.use_refiner and (quality.needs_refine or self.settings.flux_refiner.enabled):
            refiner = create_refiner(self.settings)
            try:
                refined = refiner.refine(
                    current_image,
                    mask_result.soft_mask,
                    prompt,
                    references={"person": person, "garment": garment_seg.normalized_crop},
                    seed=seed,
                )
                current_image = refined.image
                refined_path = save_image(current_image, job_dir / "refined_output.png")
            except ModelUnavailableError as exc:
                quality.notes.append(str(exc))
                logger.warning("Skipping refiner: %s", exc)

        if request.repair_mode and self.settings.repair.enabled:
            repair_engine = create_repair_engine(self.settings)
            repaired = repair_engine.refine(current_image, mask_result.dilated_mask, prompt, seed=seed)
            current_image = repaired.image
            refined_path = save_image(current_image, job_dir / "refined_output.png")

        result_path = save_image(current_image, job_dir / "result.png")
        metadata = {
            "job_id": request.job_id,
            "seed": seed,
            "category": request.category,
            "engine": getattr(engine, "name", "unknown"),
            "prompt": prompt,
            "quality": quality.model_dump(),
            "core_metadata": core.metadata,
        }
        self.storage.save_json(request.job_id, "metadata.json", metadata)

        return TryOnResponse(
            job_id=request.job_id,
            status="completed",
            result_url=self.storage.public_url(result_path),
            debug=DebugUrls(
                mask_url=self.storage.public_url(job_dir / "mask_preview.png"),
                agnostic_url=self.storage.public_url(job_dir / "agnostic.png"),
                core_output_url=self.storage.public_url(core_path),
                refined_output_url=self.storage.public_url(refined_path),
            ),
            quality=quality,
            seed=seed,
        )
