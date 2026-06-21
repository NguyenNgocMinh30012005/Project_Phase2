# Model Setup

## Directory Convention

```text
virtual_tryon/models/
  idm_vton/
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
bash scripts/download_models.sh
```

This creates folders only. It does not download gated or private weights.

## Config

Edit:

```text
configs/models.yaml
```

Important fields:

- `idm_vton.checkpoint_dir`
- `idm_vton.repo_path`
- `idm_vton.entrypoint`
- `flux_refiner.model_name`
- `flux_refiner.checkpoint_dir`
- `klein_tryon_lora.lora_path`

## Check IDM-VTON

```bash
cd virtual_tryon
python scripts/run_idm_vton.py --check
```

## Common Errors

`IDM-VTON checkpoint not found at ...`

The folder does not exist or is empty. Place weights under `models/idm_vton` or update `configs/models.yaml`.

`IDM-VTON checkpoint is present, but no entrypoint is configured`

Set `idm_vton.entrypoint` to the third-party runner script that accepts person, garment, mask, category, and output flags.

`CatVTON checkpoint not found at ...`

CatVTON is a baseline engine. Keep it disabled until weights and an entrypoint are configured.
