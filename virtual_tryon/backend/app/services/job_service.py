from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.tryon import TryOnResponse, TryOnStatusResponse
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline
from app.utils.errors import InputValidationError, TryOnError


class JobService:
    def __init__(self, pipeline: TryOnPipeline) -> None:
        self.pipeline = pipeline
        self.jobs: dict[str, TryOnStatusResponse] = {}

    @staticmethod
    def new_job_id() -> str:
        return uuid4().hex

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_tryon_job(self, request: PipelineRequest) -> TryOnStatusResponse:
        now = self._now()
        running = TryOnStatusResponse(job_id=request.job_id, status="running", created_at=now, updated_at=now)
        self.jobs[request.job_id] = running
        try:
            response = self.pipeline.run(request)
            completed = TryOnStatusResponse(**response.model_dump(), created_at=now, updated_at=self._now())
            self.jobs[request.job_id] = completed
            return completed
        except (TryOnError, InputValidationError) as exc:
            failed = TryOnStatusResponse(
                job_id=request.job_id,
                status="failed",
                error=str(exc),
                created_at=now,
                updated_at=self._now(),
            )
            self.jobs[request.job_id] = failed
            return failed

    def get_job(self, job_id: str) -> TryOnStatusResponse | None:
        return self.jobs.get(job_id)
