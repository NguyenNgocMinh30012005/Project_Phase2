from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.tryon import TryOnStatusResponse
from app.services.storage_service import StorageService
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline
from app.utils.errors import InputValidationError, TryOnError


class JobService:
    def __init__(self, pipeline: TryOnPipeline, storage: StorageService | None = None) -> None:
        self.pipeline = pipeline
        self.storage = storage or pipeline.storage
        self.jobs: dict[str, TryOnStatusResponse] = {}
        self._lock = threading.Lock()
        self._jobs_guard = threading.Lock()

    @staticmethod
    def new_job_id() -> str:
        return uuid4().hex

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _job_json_path(self, job_id: str):
        return self.storage.outputs_dir / job_id / "job.json"

    def _save_job(self, job: TryOnStatusResponse) -> TryOnStatusResponse:
        with self._jobs_guard:
            self.jobs[job.job_id] = job
            self.storage.save_json(job.job_id, "job.json", job.model_dump(mode="json"))
        return job

    def _load_job_from_disk(self, job_id: str) -> TryOnStatusResponse | None:
        path = self._job_json_path(job_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            job = TryOnStatusResponse(**payload)
        except Exception:
            return None
        with self._jobs_guard:
            self.jobs[job_id] = job
        return job

    def _response_to_completed(self, response, *, created_at: str, started_at: str | None) -> TryOnStatusResponse:
        now = self._now()
        payload = response.model_dump()
        engine_status: dict[str, str] = {}
        return TryOnStatusResponse(
            **payload,
            created_at=created_at,
            started_at=started_at,
            finished_at=now,
            updated_at=now,
            engine_status=engine_status,
        )

    def _attach_engine_status(self, job: TryOnStatusResponse) -> TryOnStatusResponse:
        quality_report = self.storage.job_dir(job.job_id) / "quality_report.json"
        if quality_report.exists():
            try:
                report = json.loads(quality_report.read_text(encoding="utf-8"))
                if isinstance(report.get("engine_status"), dict):
                    job.engine_status = report["engine_status"]
            except Exception:
                return job
        return job

    def create_tryon_job(self, request: PipelineRequest) -> TryOnStatusResponse:
        now = self._now()
        running = TryOnStatusResponse(
            job_id=request.job_id,
            status="running",
            created_at=now,
            started_at=now,
            updated_at=now,
        )
        self._save_job(running)
        try:
            with self._lock:
                response = self.pipeline.run(request)
            completed = self._response_to_completed(response, created_at=now, started_at=now)
            completed = self._attach_engine_status(completed)
            return self._save_job(completed)
        except (TryOnError, InputValidationError) as exc:
            failed = TryOnStatusResponse(
                job_id=request.job_id,
                status="failed",
                error=str(exc),
                created_at=now,
                started_at=now,
                finished_at=self._now(),
                updated_at=self._now(),
            )
            return self._save_job(failed)

    def queue_tryon_job(self, request: PipelineRequest) -> TryOnStatusResponse:
        now = self._now()
        queued = TryOnStatusResponse(job_id=request.job_id, status="queued", created_at=now, updated_at=now)
        return self._save_job(queued)

    def run_queued_job(self, request: PipelineRequest) -> None:
        job = self.get_job(request.job_id)
        if job is None:
            return
        if job.cancel_requested:
            job.status = "failed"
            job.error = "Job cancelled before start."
            job.finished_at = self._now()
            job.updated_at = job.finished_at
            self._save_job(job)
            return

        with self._lock:
            job = self.get_job(request.job_id) or job
            if job.cancel_requested:
                job.status = "failed"
                job.error = "Job cancelled before start."
                job.finished_at = self._now()
                job.updated_at = job.finished_at
                self._save_job(job)
                return
            job.status = "running"
            job.started_at = self._now()
            job.updated_at = job.started_at
            self._save_job(job)
            try:
                response = self.pipeline.run(request)
                completed = self._response_to_completed(
                    response,
                    created_at=job.created_at or job.started_at or self._now(),
                    started_at=job.started_at,
                )
                completed.cancel_requested = job.cancel_requested
                completed = self._attach_engine_status(completed)
                self._save_job(completed)
            except (TryOnError, InputValidationError) as exc:
                failed = TryOnStatusResponse(
                    job_id=request.job_id,
                    status="failed",
                    error=str(exc),
                    created_at=job.created_at,
                    started_at=job.started_at,
                    finished_at=self._now(),
                    updated_at=self._now(),
                    cancel_requested=job.cancel_requested,
                )
                self._save_job(failed)

    def cancel_job(self, job_id: str) -> TryOnStatusResponse | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        if job.status == "queued":
            job.status = "failed"
            job.cancel_requested = True
            job.error = "Job cancelled before start."
            job.finished_at = self._now()
            job.updated_at = job.finished_at
        elif job.status == "running":
            job.cancel_requested = True
            job.updated_at = self._now()
        return self._save_job(job)

    def get_job(self, job_id: str) -> TryOnStatusResponse | None:
        with self._jobs_guard:
            job = self.jobs.get(job_id)
        if job is not None:
            return job
        return self._load_job_from_disk(job_id)
