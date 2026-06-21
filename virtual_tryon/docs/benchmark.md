# Benchmark And Review

## Golden Eval Set

Eval samples live under:

```text
data/eval_set/{sample_id}/
  person.jpg
  garment_top.jpg
  garment_bottom.jpg
  garment_dress.jpg
  metadata.json
```

`metadata.json`:

```json
{
  "sample_id": "sample_001",
  "category": "upper_body",
  "difficulty": "easy",
  "expected_focus": ["identity", "garment_texture"],
  "notes": ""
}
```

Validate:

```bash
cd virtual_tryon
python scripts/validate_eval_set.py --eval-set data/eval_set
```

The validator reports warnings instead of crashing when the folder is empty or a sample is incomplete.

## Model Comparison Benchmark

Run:

```bash
cd virtual_tryon
python scripts/benchmark_pipeline.py \
  --eval-set data/eval_set \
  --modes idm,idm_flux,catvton,klein_lora \
  --limit 1 \
  --output data/outputs/benchmark_phase6_test
```

Modes:

- `idm`: IDM-VTON core only.
- `idm_flux`: IDM-VTON plus optional FLUX refiner.
- `repair`: IDM-VTON plus accepted FLUX output plus repair.
- `catvton`: CatVTON baseline.
- `klein_lora`: Klein Try-On LoRA baseline.

If a baseline engine is unavailable, its row is marked `unavailable` and the benchmark continues.

Output:

```text
data/outputs/benchmark_{timestamp}/
  summary.csv
  summary.json
  grid.png
  index.html
  manual_ratings.csv
  sample_001/
    input_person.png
    input_garment_top.png
    quality_report.json
    run_metadata.json
    idm/
    idm_flux/
    catvton/
    klein_lora/
```

## Review Gallery

Generate or rebuild:

```bash
python scripts/build_review_gallery.py \
  --benchmark-dir data/outputs/benchmark_phase6_test
```

Open `index.html` offline. Skipped modes are shown as placeholders. Use `manual_ratings.csv` for human scoring with columns for identity, garment fidelity, realism, pose preservation, artifact score, winner, and notes.

Generated benchmark folders are regular outputs and can be cleaned with:

```bash
python scripts/cleanup_outputs.py --older-than-hours 24 --keep-latest 5 --dry-run
```
