from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.engines.factory import create_refiner
from app.preprocessing.image_loader import load_image_from_bytes, validate_mime
from app.schemas.tryon import RefineResponse, TryOnCategory, TryOnResponse, TryOnStatusResponse
from app.services.container import get_job_service, get_storage_service
from app.services.tryon_pipeline import PipelineRequest
from app.utils.errors import InputValidationError, ModelUnavailableError
from app.utils.image_io import save_image
from app.utils.seed import normalize_seed, set_seed


router = APIRouter(tags=["tryon"])


async def _read_upload(file: UploadFile | None):
    if file is None:
        return None
    validate_mime(file.content_type, file.filename)
    data = await file.read()
    settings = get_settings()
    return load_image_from_bytes(data, max_side=settings.image.max_side)


@router.post("/tryon", response_model=TryOnStatusResponse)
async def create_tryon(
    person_image: Annotated[UploadFile, File(...)],
    background_tasks: BackgroundTasks,
    garment_top: Annotated[UploadFile | None, File()] = None,
    garment_bottom: Annotated[UploadFile | None, File()] = None,
    garment_dress: Annotated[UploadFile | None, File()] = None,
    category: Annotated[TryOnCategory, Form()] = "upper_body",
    prompt: Annotated[str | None, Form()] = None,
    use_refiner: Annotated[bool, Form()] = True,
    repair_mode: Annotated[bool, Form()] = True,
    run_mode: Annotated[str | None, Form()] = None,
    seed: Annotated[int | None, Form()] = None,
) -> TryOnStatusResponse:
    if not any([garment_top, garment_bottom, garment_dress]):
        raise HTTPException(status_code=400, detail="At least one garment image is required.")

    try:
        person = await _read_upload(person_image)
        top = await _read_upload(garment_top)
        bottom = await _read_upload(garment_bottom)
        dress = await _read_upload(garment_dress)
    except InputValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_service = get_job_service()
    job_id = job_service.new_job_id()
    request = PipelineRequest(
        job_id=job_id,
        person_image=person,
        garment_top=top,
        garment_bottom=bottom,
        garment_dress=dress,
        category=category,
        prompt=prompt,
        use_refiner=use_refiner,
        repair_mode=repair_mode,
        seed=seed,
    )
    settings = get_settings()
    selected_run_mode = (run_mode or settings.api.run_mode).lower()
    if selected_run_mode not in {"sync", "async"}:
        raise HTTPException(status_code=400, detail="run_mode must be 'sync' or 'async'.")
    if selected_run_mode == "async":
        queued = job_service.queue_tryon_job(request)
        background_tasks.add_task(job_service.run_queued_job, request)
        return queued
    return job_service.create_tryon_job(request)


@router.get("/tryon/{job_id}", response_model=TryOnStatusResponse)
def get_tryon(job_id: str) -> TryOnStatusResponse:
    job = get_job_service().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.delete("/tryon/{job_id}", response_model=TryOnStatusResponse)
def cancel_tryon(job_id: str) -> TryOnStatusResponse:
    job = get_job_service().cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.post("/tryon/refine", response_model=RefineResponse)
async def refine_image(
    image: Annotated[UploadFile, File(...)],
    mask: Annotated[UploadFile | None, File()] = None,
    prompt: Annotated[str, Form()] = "Refine garment boundary while preserving identity, pose, face, and background.",
    seed: Annotated[int | None, Form()] = None,
) -> RefineResponse:
    settings = get_settings()
    storage = get_storage_service()
    job_id = get_job_service().new_job_id()
    job_dir = storage.job_dir(job_id)
    normalized_seed = normalize_seed(seed)
    set_seed(normalized_seed)

    try:
        base_image = await _read_upload(image)
        mask_image = await _read_upload(mask)
    except InputValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    save_image(base_image, job_dir / "refine_input.png")
    if mask_image:
        save_image(mask_image.convert("L"), job_dir / "refine_mask.png")

    refiner = create_refiner(settings)
    try:
        result = refiner.refine(base_image, mask_image, prompt, seed=normalized_seed)
    except ModelUnavailableError as exc:
        return RefineResponse(job_id=job_id, status="failed", error=str(exc), seed=normalized_seed)

    result_path = save_image(result.image, job_dir / "refined_output.png")
    storage.save_json(job_id, "metadata.json", {"prompt": prompt, "seed": normalized_seed, "metadata": result.metadata})
    return RefineResponse(
        job_id=job_id,
        status="completed",
        result_url=storage.public_url(result_path),
        seed=normalized_seed,
    )
