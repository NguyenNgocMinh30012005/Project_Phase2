# Model Setup

## Directory Convention

```text
virtual_tryon/models/
  idm_vton/
    ckpt/
      densepose/
        model_final_162be9.pkl
      humanparsing/
        parsing_atr.onnx
        parsing_lip.onnx
      openpose/
        ckpts/
          body_pose_model.pth
  flux2/
  catvton/
  loras/
    flux-klein-tryon.safetensors

virtual_tryon/third_party/
  IDM-VTON/
  CatVTON/
```

Run:

```bash
cd virtual_tryon
bash scripts/setup_idm_vton.sh
```

This clones the official IDM-VTON implementation into `third_party/IDM-VTON`, creates the checkpoint folders, and verifies required preprocessing files. It does not download gated/private weights.

## Config

Edit:

```text
configs/models.yaml
```

Important fields:

- `idm_vton.checkpoint_dir`
- `idm_vton.repo_path`
- `idm_vton.entrypoint`
- `idm_vton.model_name`
- `flux_refiner.model_name`
- `flux_refiner.checkpoint_dir`
- `klein_tryon_lora.lora_path`

## Check IDM-VTON

```bash
cd virtual_tryon
python scripts/run_idm_vton.py --check
```

## Real CLI Run

```bash
cd virtual_tryon
python scripts/run_tryon.py \
  --person data/examples/person_001.jpg \
  --garment data/examples/top_001.jpg \
  --category upper_body \
  --prompt "replace the shirt with the reference garment, preserve face, pose, and body shape" \
  --output data/outputs/real_idm_test.png
```

Do not pass `--mock` for the real engine. The adapter creates a one-sample VITON-HD-style dataset under:

```text
data/outputs/{job_id}/idm_vton_dataset/
```

Debug files include:

```text
core_output.png
idm_vton_command.txt
idm_vton_stdout.txt
idm_vton_stderr.txt
mask_preview.png
agnostic.png
cloth_mask.png
densepose.png or densepose_placeholder.png
```

## Real API Test

```bash
curl -X POST http://127.0.0.1:8000/tryon \
  -F "person_image=@data/examples/person_001.jpg" \
  -F "garment_top=@data/examples/top_001.jpg" \
  -F "category=upper_body" \
  -F "prompt=replace the shirt with the reference garment, preserve face, pose, body shape" \
  -F "use_refiner=false" \
  -F "repair_mode=false"
```

## Common Errors

`IDM-VTON is not available. missing checkpoint: densepose/model_final_162be9.pkl; ...`

The checkpoint folder is missing required files. Place preprocessing checkpoints under `models/idm_vton/ckpt` or update `configs/models.yaml`.

`entrypoint not found: .../third_party/IDM-VTON/inference.py`

Run `bash scripts/setup_idm_vton.sh` or update `idm_vton.entrypoint`.

`checkpoint looks incomplete: ...`

The file exists but is probably a Git LFS pointer or placeholder. Replace it with the real checkpoint file.

`CatVTON checkpoint not found at ...`

CatVTON is a baseline engine. Keep it disabled until weights and an entrypoint are configured.
