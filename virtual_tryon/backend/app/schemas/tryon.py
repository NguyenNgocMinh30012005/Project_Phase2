from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TryOnCategory = Literal["upper_body", "lower_body", "dress", "full_outfit"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled", "cancel_requested"]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


class DebugUrls(BaseModel):
    mask_url: str | None = None
    mask_urls: list[str] = Field(default_factory=list)
    agnostic_url: str | None = None
    core_output_url: str | None = None
    refined_output_url: str | None = None
    quality_report_url: str | None = None
    refine_mask_url: str | None = None
    prompt_core_url: str | None = None
    prompt_refine_url: str | None = None
    prompt_metadata_url: str | None = None


class QualityScores(BaseModel):
    identity_score: float | None = None
    garment_similarity_score: float | None = None
    background_preservation_score: float | None = None
    artifact_score: float | None = None
    needs_refine: bool = False
    notes: list[str] = Field(default_factory=list)


class TryOnResponse(BaseModel):
    job_id: str
    status: JobStatus
    result_url: str | None = None
    debug: DebugUrls = Field(default_factory=DebugUrls)
    quality: QualityScores | None = None
    error: str | None = None
    seed: int | None = None


class TryOnStatusResponse(TryOnResponse):
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str | None = None
    cancel_requested: bool = False
    engine_status: dict[str, str] = Field(default_factory=dict)
    artifact_manifest: dict | None = None
    error_code: str | None = None
    retry_count: int = 0


class HealthResponse(BaseModel):
    status: str
    device: str
    models: dict[str, str]


class RefineResponse(BaseModel):
    job_id: str
    status: JobStatus
    result_url: str | None = None
    error: str | None = None
    seed: int | None = None
