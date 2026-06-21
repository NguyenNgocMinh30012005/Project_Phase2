from __future__ import annotations

import pytest
from PIL import Image

from app.core.config import load_settings
from app.schemas.tryon import TryOnResponse
from app.services.storage_service import StorageService
from app.services.job_service import JobService
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline
from app.utils.errors import ModelUnavailableError


def make_request(job_id: str = "test_job") -> PipelineRequest:
    return PipelineRequest(
        job_id=job_id,
        person_image=Image.new("RGB", (256, 384), (180, 180, 180)),
        garment_top=Image.new("RGB", (256, 384), (40, 90, 220)),
        garment_bottom=None,
        garment_dress=None,
        category="upper_body",
        prompt="preserve identity and pose",
        use_refiner=True,
        repair_mode=True,
        seed=123,
    )


def configure_temp_storage(settings, tmp_path):
    settings.storage.inputs_dir = tmp_path / "inputs"
    settings.storage.outputs_dir = tmp_path / "outputs"
    settings.storage.temp_dir = tmp_path / "temp"
    return settings


def test_pipeline_returns_tryon_result(tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "mock"
    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)
    response = pipeline.run(make_request())
    assert isinstance(response, TryOnResponse)
    assert response.status == "completed"
    assert response.result_url


def test_missing_model_gives_clear_error(tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "idm_vton"
    settings.idm_vton.checkpoint_dir = tmp_path / "missing_idm_vton"
    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)
    with pytest.raises(ModelUnavailableError, match="IDM-VTON checkpoint not found"):
        pipeline.run(make_request("missing_model"))


def test_debug_paths_are_created(tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "mock"
    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)
    response = pipeline.run(make_request("debug_job"))
    job_dir = settings.storage.outputs_dir / "debug_job"
    assert (job_dir / "mask_preview.png").exists()
    assert (job_dir / "agnostic.png").exists()
    assert (job_dir / "core_output.png").exists()
    assert response.debug.core_output_url


def test_pipeline_real_engine_fallback_returns_failed_job(tmp_path):
    settings = configure_temp_storage(load_settings(), tmp_path)
    settings.pipeline.engine = "idm_vton"
    settings.idm_vton.checkpoint_dir = tmp_path / "missing_idm_vton"
    storage = StorageService(settings.storage)
    service = JobService(TryOnPipeline(settings, storage))
    response = service.create_tryon_job(make_request("real_engine_missing"))
    assert response.status == "failed"
    assert response.error
    assert "IDM-VTON" in response.error
