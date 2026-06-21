# API

## GET /health

Returns service status, detected device, and model availability.

```json
{
  "status": "ok",
  "device": "cuda:NVIDIA GeForce RTX 3090 Ti",
  "models": {
    "idm_vton": "available",
    "flux_refiner": "unavailable: license/access not accepted or model is private",
    "catvton": "unavailable: catvton.enabled is false"
  }
}
```

Model status strings may include detailed skip reasons. IDM-VTON is the default core API engine; CatVTON and Klein LoRA are benchmark baselines unless explicitly configured.

## POST /tryon

Multipart form fields:

- `person_image`: required image file.
- `garment_top`: optional image file.
- `garment_bottom`: optional image file.
- `garment_dress`: optional image file.
- `category`: `upper_body`, `lower_body`, `dress`, or `full_outfit`.
- `prompt`: optional text.
- `use_refiner`: boolean, default `true`.
- `repair_mode`: boolean, default `true`.
- `run_mode`: optional `sync` or `async`; defaults to `configs/pipeline.yaml`.
- `seed`: optional integer.

Response:

```json
{
  "job_id": "abc",
  "status": "completed",
  "result_url": "/artifacts/abc/result.png",
  "debug": {
    "mask_url": "/artifacts/abc/mask_preview.png",
    "mask_urls": ["/artifacts/abc/mask_preview.png"],
    "agnostic_url": "/artifacts/abc/agnostic.png",
    "core_output_url": "/artifacts/abc/core_output.png",
    "refined_output_url": "/artifacts/abc/refined_output.png",
    "quality_report_url": "/artifacts/abc/quality_report.json",
    "refine_mask_url": "/artifacts/abc/safe_refine_mask_overlay.png"
  },
  "seed": 123
}
```

When `run_mode=async`, `POST /tryon` returns quickly:

```json
{
  "job_id": "abc",
  "status": "queued",
  "result_url": null,
  "debug": {}
}
```

Poll `GET /tryon/{job_id}` until `completed` or `failed`.

If the core model is missing, the job returns `status: failed` and a clear `error` string.

If `use_refiner=true` and the FLUX refiner is missing, incompatible, or runs out of memory, the job still returns `status: completed` with `result_url` pointing to the IDM-VTON core output. The output folder includes `flux_refiner_error.txt` and `quality_report.json` explaining the fallback. Raw stack traces are not returned in the API response. Repair runs only after a refined output is created and accepted by the quality gate.

Important output files:

```text
core_output.png
result.png
quality_report.json
garment_refine_mask.png
boundary_refine_mask.png
safe_refine_mask.png
garment_refine_mask_overlay.png
boundary_refine_mask_overlay.png
safe_refine_mask_overlay.png
idm_vton_command.txt
idm_vton_stdout.txt
idm_vton_stderr.txt
```

`quality_report.json` includes `engine_status`, `final_choice`, and `final_choice_reason`. For model comparison across CatVTON/Klein baselines, use `scripts/benchmark_pipeline.py` instead of the default `/tryon` API.

## GET /artifacts/{path}

Serves files under `data/outputs` only:

```text
/artifacts/{job_id}/result.png
/artifacts/{job_id}/core_output.png
/artifacts/{job_id}/quality_report.json
```

Path traversal and files outside `data/outputs` return clean 404 responses. Models, checkpoints, `third_party`, `.env`, and tokens are never served by this route.

Example missing-model response:

```json
{
  "job_id": "abc",
  "status": "failed",
  "result_url": null,
  "error": "IDM-VTON is not available. missing checkpoint: densepose/model_final_162be9.pkl"
}
```

## GET /tryon/{job_id}

Returns the stored job status. Job metadata is also written to:

```text
data/outputs/{job_id}/job.json
```

`job.json` includes `queued`, `running`, `completed`, or `failed`, timestamps, clean error text, result URL, debug URLs, and engine status.

## DELETE /tryon/{job_id}

Cancels a queued job. Running jobs are marked with `cancel_requested`; the current local executor does not kill an active IDM-VTON subprocess.

## POST /tryon/refine

Multipart form fields:

- `image`: required image file.
- `mask`: optional image file.
- `prompt`: required or default prompt.
- `seed`: optional integer.
