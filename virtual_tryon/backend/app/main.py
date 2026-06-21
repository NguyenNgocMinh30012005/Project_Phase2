from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_artifacts import router as artifacts_router
from app.api.routes_health import router as health_router
from app.api.routes_tryon import router as tryon_router
from app.core.config import get_settings
from app.core.logging import configure_logging


configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app.name,
    version="0.1.0",
    debug=settings.app.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.storage.outputs_dir.mkdir(parents=True, exist_ok=True)

app.include_router(artifacts_router)
app.include_router(health_router)
app.include_router(tryon_router)
