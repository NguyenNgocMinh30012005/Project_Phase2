# Evaluation Set Expansion Guide

## Goal

Expand `data/eval_set/` from the single smoke sample to 5-10 consented, license-compatible samples. Keep source images private when their redistribution terms are unclear; the validator and benchmark can operate on local untracked samples.

Include a deliberate mix:

- Easy upper-body transfer with a clear frontal pose and unobstructed torso.
- Logo, typography, stripe, plaid, or other fine-pattern garment.
- Long hair crossing the collar or upper garment.
- One or both hands occluding the garment.
- Side or three-quarter pose.
- Lower-body or dress samples only where the selected engine and inputs support them.

## Folder Layout

```text
data/eval_set/sample_002/
  person.jpg
  garment_top.jpg
  metadata.json
```

Use `garment_bottom.jpg` for `lower_body`, `garment_dress.jpg` for `dress`, and either a dress image or both top and bottom images for `full_outfit`. Accepted image extensions are `.jpg`, `.jpeg`, `.png`, and `.webp`.

## Metadata Schema

```json
{
  "sample_id": "sample_002",
  "category": "upper_body",
  "difficulty": "hard",
  "expected_focus": ["logo_fidelity", "hand_occlusion"],
  "notes": "Right hand overlaps the shirt graphic."
}
```

Rules:

- `sample_id` must match the folder name.
- `category` must be `upper_body`, `lower_body`, `dress`, or `full_outfit`.
- `difficulty` must be `easy`, `medium`, or `hard`.
- `expected_focus` must be a JSON list of strings.
- `notes` is optional but should record the visual challenge and review target.

## Validate

From `virtual_tryon/` run:

```bash
python scripts/validate_eval_set.py --eval-set data/eval_set
```

Resolve every warning before benchmarking. Also inspect that each person/garment pair has the intended category, readable orientation, and no accidental duplicate.

## Benchmark

With local model dependencies available, run:

```bash
python scripts/benchmark_pipeline.py \
  --eval-set data/eval_set \
  --modes idm,idm_flux \
  --limit 5
```

Review `summary.json`, `summary.csv`, the generated grid/gallery, and `manual_ratings.csv`. Compare identity, garment fidelity, realism, pose preservation, and artifacts rather than selecting a winner from one automated score alone.
