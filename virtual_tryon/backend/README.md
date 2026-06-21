# Virtual Try-On Backend

FastAPI backend for the Virtual Try-On pipeline.

## Run

```bash
cd virtual_tryon/backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Important Behavior

- Core try-on engine defaults to `idm_vton`.
- Missing core checkpoints return a clear failed job message.
- Mock engine is available for tests and pipeline validation via `TRYON_ENGINE=mock`.
- FLUX refiner and ADetailer-like repair are post-processing modules only.
- Every job writes intermediates to `data/outputs/{job_id}/`.
- `use_refiner=true` is best-effort: FLUX load/OOM/runtime failures are logged and the job falls back to `core_output.png`.
- Every completed core job writes `quality_report.json` and refine mask overlays for debugging.
- `/health` returns detailed model status strings; benchmark baselines can be unavailable without affecting the default IDM-VTON API.
