# Evaluation

The first quality checker is intentionally lightweight and deterministic. It returns:

```json
{
  "identity_score": null,
  "garment_similarity_score": 0.0,
  "background_preservation_score": 0.0,
  "artifact_score": 0.0,
  "needs_refine": true,
  "notes": []
}
```

Implemented checks:

- Output exists through pipeline storage.
- Output resolution is above configured minimum.
- Background preservation via difference outside the garment mask.
- Garment region change via difference inside the garment mask.
- Boundary artifact heuristic using mask edges.
- Rough garment similarity when a garment reference is available.

Future upgrades:

- Face embedding similarity.
- Human parsing based region preservation.
- Garment CLIP/DINO similarity.
- Per-region artifact detection for collar, sleeve, hands, and hemline.
