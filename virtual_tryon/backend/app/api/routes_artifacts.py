from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import get_settings


router = APIRouter(tags=["artifacts"])


def _safe_artifact_path(artifact_path: str) -> Path:
    settings = get_settings()
    root = settings.storage.outputs_dir.resolve()
    candidate = (root / artifact_path).resolve()
    if candidate == root or root not in candidate.parents:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return candidate


@router.get("/artifacts/{artifact_path:path}")
def get_artifact(artifact_path: str) -> FileResponse:
    return FileResponse(_safe_artifact_path(artifact_path))
