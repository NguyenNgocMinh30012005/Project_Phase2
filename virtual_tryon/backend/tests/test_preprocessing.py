from __future__ import annotations

from PIL import Image

from app.core.config import PreprocessingConfig
from app.preprocessing.agnostic_mask import create_agnostic_mask
from app.preprocessing.garment_segmenter import GarmentSegmenter
from app.preprocessing.mask_utils import mask_area


def test_agnostic_mask_has_expected_outputs():
    image = Image.new("RGB", (256, 384), (180, 180, 180))
    result = create_agnostic_mask(image, "upper_body", PreprocessingConfig())
    assert result.raw_mask.size == image.size
    assert result.dilated_mask.size == image.size
    assert result.soft_mask.size == image.size
    assert mask_area(result.dilated_mask) >= mask_area(result.raw_mask)


def test_garment_segmenter_returns_normalized_crop():
    image = Image.new("RGB", (128, 128), (255, 255, 255))
    garment = Image.new("RGB", (80, 100), (20, 80, 200))
    image.paste(garment, (24, 14))
    result = GarmentSegmenter().segment(image, (256, 384))
    assert result.normalized_crop.size == (256, 384)
    assert result.cloth_mask.size == image.size
