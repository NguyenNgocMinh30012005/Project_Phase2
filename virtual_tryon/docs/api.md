# API

## GET /health

Returns service status, detected device, and model availability.

```json
{
  "status": "ok",
  "device": "cuda:NVIDIA GeForce RTX 3090 Ti",
  "models": {
    "idm_vton": "missing",
    "flux_refiner": "available",
    "catvton": "missing"
  }
}
```

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
- `seed`: optional integer.

Response:

```json
{
  "job_id": "abc",
  "status": "completed",
  "result_url": "/outputs/abc/result.png",
  "debug": {
    "mask_url": "/outputs/abc/mask_preview.png",
    "agnostic_url": "/outputs/abc/agnostic.png",
    "core_output_url": "/outputs/abc/core_output.png",
    "refined_output_url": "/outputs/abc/refined_output.png",
    "quality_report_url": "/outputs/abc/quality_report.json",
    "refine_mask_url": "/outputs/abc/safe_refine_mask_overlay.png"
  },
  "seed": 123
}
```

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

Returns the stored job status for the current API process.

## POST /tryon/refine

Multipart form fields:

- `image`: required image file.
- `mask`: optional image file.
- `prompt`: required or default prompt.
- `seed`: optional integer.
