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
bash scripts/setup_idm_vton_runtime.sh
bash scripts/download_idm_vton_ckpt.sh
```

`setup_idm_vton.sh` clones the official IDM-VTON implementation into `third_party/IDM-VTON` and creates checkpoint folders.

`setup_idm_vton_runtime.sh` installs the IDM-VTON-compatible runtime pins used by the official environment. It also removes `peft` from the active venv because newer PEFT releases can conflict with `accelerate==0.25.0` during IDM-VTON pipeline loading.

`download_idm_vton_ckpt.sh` downloads `ckpt/**` from the Hugging Face space `yisol/IDM-VTON`, copies it into `models/idm_vton/ckpt`, and verifies the four required preprocessing checkpoint files. It uses a temporary folder and does not write tokens to the repo.

## Token And Model Safety

- Never commit Hugging Face tokens, `.env` files, shell history, checkpoints, model weights, generated outputs, or third-party source mirrors.
- Keep `HF_TOKEN` in the shell environment only for the current session, or use the Hugging Face CLI login cache outside the repository.
- If a token is pasted into a chat, terminal transcript, issue, or any external system, revoke or rotate it in Hugging Face settings before continuing production work.
- `.gitignore` must keep `virtual_tryon/models/`, `virtual_tryon/data/outputs/`, `virtual_tryon/data/temp/`, and `virtual_tryon/third_party/` out of Git.
- Before commit, run a quick safety check such as `git status --short` and `rg -n "hf_[A-Za-z0-9_]+" .`.

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
- `flux_refiner.backend`
- `flux_refiner.model_name`
- `flux_refiner.model_path`
- `flux_refiner.checkpoint_dir`
- `flux_refiner.api_url_env`
- `flux_refiner.api_key_env`
- `klein_tryon_lora.lora_path`

`flux_refiner.backend` supports:

- `disabled`
- `flux2_dev`
- `flux2_klein`
- `flux2_api`
- `fal_tryon_lora`

For API backends, set credentials through environment variables only:

```bash
export FLUX_REFINER_API_URL="https://example.invalid/refine"
export FLUX_REFINER_API_KEY="..."
```

Do not store API keys or Hugging Face tokens in the repo. If a token was pasted into an external system, revoke or rotate it.

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

## Baseline Suite

After IDM-VTON is available, preserve a fixed baseline before adding refiners:

```bash
cd virtual_tryon
python scripts/run_idm_baseline_suite.py
```

The script discovers paired `data/examples/person_*` and `data/examples/top_*` files, runs IDM-VTON without FLUX or repair, and writes:

```text
data/outputs/baseline_suite/{sample_id}/
  input_person.png
  input_garment.png
  core_output.png
  mask_preview.png
  idm_vton_command.txt
  idm_vton_stdout.txt
  idm_vton_stderr.txt
  run_metadata.json
data/outputs/baseline_suite/baseline_summary.json
```

If only one or two paired samples exist, the script still runs and prints guidance to add more examples.

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

## CatVTON Baseline

CatVTON is benchmark-only by default:

```yaml
catvton:
  enabled: false
  repo_path: "./third_party/CatVTON"
  checkpoint_dir: "./models/catvton"
  entrypoint: null
```

Clone or install CatVTON under `third_party/CatVTON`, place weights under `models/catvton`, and set `entrypoint` before enabling it. Both `third_party/` and `models/` are ignored by Git.

## Klein Try-On LoRA Baseline

Klein Try-On LoRA is also benchmark-only by default:

```yaml
klein_tryon_lora:
  enabled: false
  base_model: "black-forest-labs/FLUX.2-klein-9B"
  lora_path: "./models/loras/flux-klein-tryon.safetensors"
```

Benchmark logs include the final `TRYON ...` prompt. If the LoRA file or execution backend is missing, the benchmark row is marked unavailable instead of crashing.
