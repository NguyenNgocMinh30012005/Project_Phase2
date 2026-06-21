# Project_Phase2 - Virtual Try-On

Production-grade Virtual Try-On scaffold with an IDM-VTON-style core engine, optional FLUX refiner, baseline engine adapters, FastAPI backend, React frontend MVP, CLI scripts, and tests.

## Repository Layout

```text
virtual_tryon/
  backend/        FastAPI API, pipeline, engines, preprocessing, tests
  frontend/       React/TypeScript upload and result viewer MVP
  configs/        Runtime, model, and pipeline config
  scripts/        CLI runners and model setup helpers
  data/           Local inputs, outputs, temp files
  docs/           Architecture, API, model setup, evaluation notes
```

## Backend Setup

```bash
cd virtual_tryon/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

On RunPod, use the prepared environment:

```bash
source /workspace/Project_Phase2/env.sh
cd /workspace/Project_Phase2/virtual_tryon/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

Health:

```bash
curl http://127.0.0.1:8000/health
```

Try-on:

```bash
curl -X POST http://127.0.0.1:8000/tryon \
  -F "person_image=@data/examples/person.jpg" \
  -F "garment_top=@data/examples/top.jpg" \
  -F "category=upper_body" \
  -F "prompt=replace the shirt with the reference garment, preserve face, pose, body shape" \
  -F "use_refiner=true" \
  -F "repair_mode=true"
```

Outputs are saved under:

```text
virtual_tryon/data/outputs/{job_id}/
```

## CLI

Mock end-to-end run without checkpoints:

```bash
cd virtual_tryon
python scripts/run_tryon.py \
  --person data/examples/person.jpg \
  --garment data/examples/top.jpg \
  --category upper_body \
  --prompt "replace the shirt with the reference garment, preserve face, pose, body shape" \
  --mock \
  --output data/outputs/demo.png
```

Production run uses `configs/models.yaml`. If IDM-VTON checkpoint is missing, the API/job error is explicit:

```text
IDM-VTON checkpoint not found at ...
```

## Frontend

```bash
cd virtual_tryon/frontend
npm install
npm run dev
```

Set the backend URL if needed:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Tests

```bash
cd virtual_tryon/backend
TRYON_ENGINE=mock pytest
```

## Model Setup

Create model folders:

```bash
cd virtual_tryon
bash scripts/download_models.sh
```

Then place checkpoints according to `configs/models.yaml`.
