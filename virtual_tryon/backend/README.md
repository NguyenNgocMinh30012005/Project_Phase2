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
