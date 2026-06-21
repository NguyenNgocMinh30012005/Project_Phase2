from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

from app.core.config import PreprocessingConfig
from app.preprocessing.mask_utils import MaskBundle, blur, dilate, overlay_mask_preview
from app.schemas.tryon import TryOnCategory


@dataclass(frozen=True)
class AgnosticMaskResult:
    raw_mask: Image.Image
    dilated_mask: Image.Image
    soft_mask: Image.Image
    preview: Image.Image
    agnostic_image: Image.Image


def _region_bbox(width: int, height: int, category: TryOnCategory) -> tuple[int, int, int, int]:
    if category == "upper_body":
        return int(width * 0.18), int(height * 0.22), int(width * 0.82), int(height * 0.68)
    if category == "lower_body":
        return int(width * 0.20), int(height * 0.50), int(width * 0.80), int(height * 0.95)
    if category == "dress":
        return int(width * 0.16), int(height * 0.22), int(width * 0.84), int(height * 0.95)
    return int(width * 0.15), int(height * 0.20), int(width * 0.85), int(height * 0.95)


def create_agnostic_mask(
    person_image: Image.Image,
    category: TryOnCategory,
    config: PreprocessingConfig,
) -> AgnosticMaskResult:
    width, height = person_image.size
    raw = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(raw)
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

    expanded = dilate(raw, config.dilation_px)
    soft = blur(expanded, config.blur_radius)
    preview = overlay_mask_preview(person_image, expanded)

    agnostic = person_image.convert("RGB").copy()
    overlay = Image.new("RGB", person_image.size, (224, 224, 224))
    agnostic.paste(overlay, mask=soft)
    return AgnosticMaskResult(raw, expanded, soft, preview, agnostic)


def bundle_from_result(result: AgnosticMaskResult) -> MaskBundle:
    return MaskBundle(
        raw_mask=result.raw_mask,
        dilated_mask=result.dilated_mask,
        soft_mask=result.soft_mask,
        preview=result.preview,
    )
