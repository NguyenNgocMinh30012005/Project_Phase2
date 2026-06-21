from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.core.config import MaskExperimentConfig, PreprocessingConfig
from app.preprocessing.mask_utils import MaskBundle, blur, dilate, overlay_mask_preview
from app.schemas.tryon import TryOnCategory


@dataclass(frozen=True)
class AgnosticMaskResult:
    raw_mask: Image.Image
    dilated_mask: Image.Image
    soft_mask: Image.Image
    preview: Image.Image
    agnostic_image: Image.Image
    original_upper_body_mask: Image.Image | None = None
    expanded_upper_body_mask: Image.Image | None = None
    diff_upper_body_mask: Image.Image | None = None
    original_upper_body_overlay: Image.Image | None = None
    expanded_upper_body_overlay: Image.Image | None = None
    diff_upper_body_overlay: Image.Image | None = None


def _region_bbox(width: int, height: int, category: TryOnCategory) -> tuple[int, int, int, int]:
    if category == "upper_body":
        return int(width * 0.18), int(height * 0.22), int(width * 0.82), int(height * 0.68)
    if category == "lower_body":
        return int(width * 0.20), int(height * 0.50), int(width * 0.80), int(height * 0.95)
    if category == "dress":
        return int(width * 0.16), int(height * 0.22), int(width * 0.84), int(height * 0.95)
    return int(width * 0.15), int(height * 0.20), int(width * 0.85), int(height * 0.95)


def _protected_region_mask(
    person_image: Image.Image,
    category: TryOnCategory,
    *,
    preserve_face: bool,
    preserve_hair: bool,
    preserve_hands: bool,
) -> Image.Image:
    width, height = person_image.size
    protected = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(protected)
    if preserve_face or preserve_hair:
        face_clearance = int(height * 0.22)
        draw.rectangle((0, 0, width, face_clearance), fill=255)
    if preserve_hands and category in {"upper_body", "dress", "full_outfit"}:
        ycbcr = np.array(person_image.convert("YCbCr"), dtype=np.uint8)
        cb = ycbcr[:, :, 1]
        cr = ycbcr[:, :, 2]
        skin = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
        yy, xx = np.indices((height, width))
        side_region = (
            ((xx < width * 0.32) | (xx > width * 0.68))
            & (yy > height * 0.28)
            & (yy < height * 0.92)
        )
        hand_mask = Image.fromarray((skin & side_region).astype(np.uint8) * 255, mode="L")
        hand_mask = hand_mask.filter(ImageFilter.MaxFilter(21))
        protected = ImageChops.lighter(protected, hand_mask)
    return protected


def _clear_protected(mask: Image.Image, protected: Image.Image) -> Image.Image:
    values = np.array(mask.convert("L"), dtype=np.uint8)
    protected_values = np.array(protected.convert("L"), dtype=np.uint8)
    values[protected_values > 0] = 0
    return Image.fromarray(values, mode="L")


def _expand_upper_body_hem(
    raw_mask: Image.Image,
    config: MaskExperimentConfig,
) -> Image.Image:
    width, height = raw_mask.size
    x0, _, x1, y1 = _region_bbox(width, height, "upper_body")
    extension = max(0, int(round(height * config.torso_down_extension_ratio)))
    bottom = min(height - 1, y1 + extension)
    waist_extra = max(0, config.waist_extra_dilation_px)
    waist_x0 = max(0, x0 - waist_extra)
    waist_x1 = min(width - 1, x1 + waist_extra)

    expanded = raw_mask.copy()
    draw = ImageDraw.Draw(expanded)
    overlap = max(2, min(waist_extra // 2, height // 32))
    draw.rounded_rectangle(
        (waist_x0, max(0, y1 - overlap), waist_x1, bottom),
        radius=max(8, width // 30),
        fill=255,
    )
    return expanded


def create_agnostic_mask(
    person_image: Image.Image,
    category: TryOnCategory,
    config: PreprocessingConfig,
    upper_body_experiment: MaskExperimentConfig | None = None,
) -> AgnosticMaskResult:
    width, height = person_image.size
    original_raw = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(original_raw)
    draw.rounded_rectangle(_region_bbox(width, height, category), radius=max(8, width // 30), fill=255)

    if config.preserve_face or config.preserve_hair:
        face_clearance = int(height * 0.22)
        draw.rectangle((0, 0, width, face_clearance), fill=0)

    if config.preserve_hands and category in {"upper_body", "dress", "full_outfit"}:
        hand_w = int(width * 0.13)
        hand_y0 = int(height * 0.38)
        hand_y1 = int(height * 0.72)
        draw.rectangle((0, hand_y0, hand_w, hand_y1), fill=0)
        draw.rectangle((width - hand_w, hand_y0, width, hand_y1), fill=0)

    experiment_enabled = bool(
        category == "upper_body"
        and upper_body_experiment is not None
        and upper_body_experiment.enabled
    )
    selected_raw = original_raw
    debug_expanded = None
    debug_diff = None
    original_overlay = None
    expanded_overlay = None
    diff_overlay = None
    protected = None
    if experiment_enabled and upper_body_experiment is not None:
        protected = _protected_region_mask(
            person_image,
            category,
            preserve_face=upper_body_experiment.preserve_face,
            preserve_hair=upper_body_experiment.preserve_hair,
            preserve_hands=upper_body_experiment.preserve_hands,
        )
        candidate = _expand_upper_body_hem(original_raw, upper_body_experiment)
        added_region = _clear_protected(
            ImageChops.subtract(candidate, original_raw),
            protected,
        )
        debug_expanded = ImageChops.lighter(original_raw, added_region)
        debug_diff = added_region
        selected_raw = debug_expanded
        if upper_body_experiment.save_debug_overlays:
            original_overlay = overlay_mask_preview(person_image, original_raw, (59, 130, 246))
            expanded_overlay = overlay_mask_preview(person_image, debug_expanded, (16, 185, 129))
            diff_overlay = overlay_mask_preview(person_image, debug_diff, (239, 68, 68))

    if protected is not None and debug_diff is not None:
        base_expanded = dilate(original_raw, config.dilation_px)
        added_expanded = _clear_protected(
            dilate(debug_diff, config.dilation_px),
            protected,
        )
        expanded = ImageChops.lighter(base_expanded, added_expanded)
        base_soft = blur(base_expanded, config.blur_radius)
        added_soft = _clear_protected(
            blur(added_expanded, config.blur_radius),
            protected,
        )
        soft = ImageChops.lighter(base_soft, added_soft)
    else:
        expanded = dilate(selected_raw, config.dilation_px)
        soft = blur(expanded, config.blur_radius)
    preview = overlay_mask_preview(person_image, expanded)

    agnostic = person_image.convert("RGB").copy()
    overlay = Image.new("RGB", person_image.size, (224, 224, 224))
    agnostic.paste(overlay, mask=soft)
    return AgnosticMaskResult(
        selected_raw,
        expanded,
        soft,
        preview,
        agnostic,
        original_upper_body_mask=original_raw if experiment_enabled else None,
        expanded_upper_body_mask=debug_expanded,
        diff_upper_body_mask=debug_diff,
        original_upper_body_overlay=original_overlay,
        expanded_upper_body_overlay=expanded_overlay,
        diff_upper_body_overlay=diff_overlay,
    )


def bundle_from_result(result: AgnosticMaskResult) -> MaskBundle:
    return MaskBundle(
        raw_mask=result.raw_mask,
        dilated_mask=result.dilated_mask,
        soft_mask=result.soft_mask,
        preview=result.preview,
    )
