# Architecture

## Why IDM-VTON Is The Core

IDM-VTON-style methods are designed for garment-conditioned person image synthesis. They are the right core because the task is not general image editing; it must preserve pose, body shape, face, hair, and background while replacing only the garment region with a reference garment.

## Why FLUX Is A Refiner

FLUX.2 dev/base is used as a post-process image-edit/inpaint refiner. Its job is to repair boundary quality, fabric folds, texture, lighting, hand overlap, collar, sleeves, and hemline. It should not replace the whole try-on engine because unrestricted editing can drift identity, pose, or garment geometry.

## Why ADetailer Is Not Core

ADetailer-like logic is a repair module. It can detect or mask local problem regions and inpaint them, but it does not solve garment transfer. It is useful after core try-on and optional FLUX refinement.

## Pipeline Diagram

```text
person image + garment image + category + prompt
  -> validate inputs
  -> normalize person and garment
  -> human parser stub or model
  -> densepose stub or model
  -> agnostic mask and agnostic person image
  -> garment segmentation
  -> core try-on engine, default IDM-VTON
  -> quality checks
  -> optional FLUX refiner on garment/mask region
  -> optional ADetailer-like local repair
  -> save result and debug intermediates
```

## Engine Contract

All core engines implement:

```python
is_available() -> bool
prepare() -> None
run(inputs: TryOnInputs) -> TryOnResult
```

All refiners implement:

```python
is_available() -> bool
prepare() -> None
refine(image, mask, prompt, references=None, seed=None) -> RefineResult
```
