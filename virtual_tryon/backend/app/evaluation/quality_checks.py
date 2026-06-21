from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from app.core.config import QualityConfig
from app.preprocessing.mask_utils import invert, to_l_mask
from app.schemas.tryon import QualityScores


def _mean_abs_diff(a: Image.Image, b: Image.Image, mask: Image.Image | None = None) -> float:
    arr_a = np.array(a.convert("RGB"), dtype=np.float32)
    arr_b = np.array(b.convert("RGB").resize(a.size), dtype=np.float32)
    diff = np.abs(arr_a - arr_b).mean(axis=2) / 255.0
    if mask is not None:
        mask_arr = np.array(to_l_mask(mask).resize(a.size), dtype=np.float32) / 255.0
        denom = float(mask_arr.sum())
        if denom <= 1e-6:
            return 0.0
        return float((diff * mask_arr).sum() / denom)
    return float(diff.mean())


def run_quality_checks(
    person_image: Image.Image,
    output_image: Image.Image,
    garment_image: Image.Image | None,
    garment_mask: Image.Image,
    config: QualityConfig,
) -> QualityScores:
    notes: list[str] = []
    needs_refine = False

    if output_image.width < config.min_output_width or output_image.height < config.min_output_height:
        notes.append("Output resolution is below configured minimum.")
        needs_refine = True

    background_mask = invert(garment_mask)
    background_change = _mean_abs_diff(person_image, output_image, background_mask)
    background_preservation = max(0.0, 1.0 - background_change)
    if background_change > config.background_change_threshold:
        notes.append("Background changed more than expected.")
        needs_refine = True

    garment_change = _mean_abs_diff(person_image, output_image, garment_mask)
    if garment_change < config.garment_change_threshold:
        notes.append("Garment region changed too little; core try-on may have failed.")
        needs_refine = True

    boundary = garment_mask.convert("L").filter(ImageFilter.FIND_EDGES)
    artifact_score = min(1.0, _mean_abs_diff(person_image, output_image, boundary) * 2.0)
    if artifact_score > config.artifact_threshold:
        notes.append("Visible mask boundary artifacts detected.")
        needs_refine = True

    garment_similarity = None
    if garment_image is not None:
        garment_similarity = max(0.0, 1.0 - _mean_abs_diff(garment_image, output_image, garment_mask))

    return QualityScores(
        identity_score=None,
        garment_similarity_score=garment_similarity,
        background_preservation_score=background_preservation,
        artifact_score=artifact_score,
        needs_refine=needs_refine,
        notes=notes,
    )
