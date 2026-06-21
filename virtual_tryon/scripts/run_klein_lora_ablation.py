from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from app.core.config import Settings, load_settings  # noqa: E402
from app.engines.base import TryOnInputs  # noqa: E402
from app.engines.klein_prompt_builder import build_klein_tryon_prompt  # noqa: E402
from app.engines.klein_tryon_lora_engine import KleinTryOnLoraEngine  # noqa: E402
from app.preprocessing.image_loader import load_image_from_path  # noqa: E402
from app.services.storage_service import StorageService  # noqa: E402
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline  # noqa: E402
from app.utils.errors import TryOnError  # noqa: E402
from app.utils.image_io import save_image  # noqa: E402
from validate_eval_set import EvalSample, validate_sample  # noqa: E402


SUMMARY_COLUMNS = [
    "sample_id",
    "variant",
    "status",
    "runtime_seconds",
    "result_path",
    "prompt_path",
    "status_path",
    "auto_bottom_reference_path",
    "engine_status",
    "error_code",
    "notes",
]
MANUAL_RATING_COLUMNS = [
    "sample_id",
    "variant",
    "result_path",
    "identity_1_5",
    "garment_fidelity_1_5",
    "old_garment_removed_1_5",
    "realism_1_5",
    "pose_preservation_1_5",
    "body_shape_preservation_1_5",
    "background_preservation_1_5",
    "overedit_1_5",
    "winner",
    "notes",
]
DEFAULT_PROMPT = build_klein_tryon_prompt(None, None, None, "upper_body")
STRONG_PROMPT = (
    "TRYON blonde woman standing front-facing. Replace the upper outfit completely with the blue velvet "
    "wrap V-neck short-sleeve top shown in the reference image. Remove every remaining pink or magenta "
    "shirt fragment, including the lower hem. Preserve the person's face, hair, hands, body shape, "
    "black pants, pose, and background. The final image is a full body shot."
)


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path | None, output_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _settings_for_klein(base_settings: Settings, output_dir: Path, bottom_strategy: str) -> Settings:
    settings = base_settings.model_copy(deep=True)
    settings.storage.outputs_dir = output_dir
    settings.pipeline.engine = "klein_tryon_lora"
    settings.klein_tryon_lora.enabled = True
    settings.klein_tryon_lora.bottom_strategy = bottom_strategy
    return settings


def _run_idm_original(
    sample: EvalSample,
    output_dir: Path,
    base_settings: Settings,
    *,
    seed: int,
    mock: bool,
) -> dict[str, Any]:
    variant = "idm_original"
    started = time.perf_counter()
    settings = base_settings.model_copy(deep=True)
    settings.storage.outputs_dir = output_dir
    settings.pipeline.engine = "mock" if mock else "idm_vton"
    storage = StorageService(settings.storage)
    request = PipelineRequest(
        job_id=variant,
        person_image=load_image_from_path(sample.person_path, max_side=settings.image.max_side),
        garment_top=load_image_from_path(sample.garment_top, max_side=settings.image.max_side) if sample.garment_top else None,
        garment_bottom=load_image_from_path(sample.garment_bottom, max_side=settings.image.max_side) if sample.garment_bottom else None,
        garment_dress=load_image_from_path(sample.garment_dress, max_side=settings.image.max_side) if sample.garment_dress else None,
        category=sample.category,
        prompt="replace the shirt with the reference garment, preserve face, pose, and body shape",
        use_refiner=False,
        repair_mode=False,
        seed=seed,
    )
    try:
        response = TryOnPipeline(settings, storage).run(request)
        result_path = storage.file_path_from_public_url(response.result_url) if response.result_url else None
        return {
            "sample_id": sample.sample_id,
            "variant": variant,
            "status": response.status,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "result_path": _display_path(result_path, output_dir),
            "prompt_path": None,
            "status_path": None,
            "auto_bottom_reference_path": None,
            "engine_status": "completed",
            "error_code": None,
            "notes": "",
        }
    except Exception as exc:
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        (variant_dir / "ablation_error.txt").write_text(str(exc), encoding="utf-8")
        return {
            "sample_id": sample.sample_id,
            "variant": variant,
            "status": "failed",
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "result_path": None,
            "prompt_path": None,
            "status_path": _display_path(variant_dir / "ablation_error.txt", output_dir),
            "auto_bottom_reference_path": None,
            "engine_status": "failed",
            "error_code": "TRYON_ERROR",
            "notes": str(exc),
        }


def _run_klein_variant(
    sample: EvalSample,
    output_dir: Path,
    base_settings: Settings,
    *,
    variant: str,
    prompt: str,
    seed: int,
    bottom_strategy: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    settings = _settings_for_klein(base_settings, output_dir, bottom_strategy)
    person = load_image_from_path(sample.person_path, max_side=settings.image.max_side)
    top = load_image_from_path(sample.garment_top, max_side=settings.image.max_side) if sample.garment_top else None
    bottom = load_image_from_path(sample.garment_bottom, max_side=settings.image.max_side) if sample.garment_bottom else None
    if top is None:
        raise ValueError(f"{sample.sample_id} has no garment_top image")

    engine = KleinTryOnLoraEngine(settings.klein_tryon_lora)
    status = "completed"
    error_code = None
    notes = ""
    result_path: Path | None = None
    try:
        result = engine.run(
            TryOnInputs(
                person_image=person,
                garment_image=top,
                category=sample.category,
                agnostic_mask=Image.new("L", person.size, 255),
                prompt=prompt,
                seed=seed,
                output_dir=variant_dir,
                extra={
                    "garment_top_image": top,
                    "garment_bottom_image": bottom,
                },
            )
        )
        result_path = save_image(result.image, variant_dir / "result.png")
    except TryOnError as exc:
        status = "unavailable" if "not available" in str(exc).lower() else "failed"
        error_code = "ENGINE_UNAVAILABLE" if status == "unavailable" else "ENGINE_EXECUTION_FAILED"
        notes = str(exc)
    except Exception as exc:
        status = "failed"
        error_code = "ENGINE_EXECUTION_FAILED"
        notes = f"{type(exc).__name__}: {exc}"
        (variant_dir / "ablation_error.txt").write_text(notes, encoding="utf-8")

    status_path = variant_dir / "status.json"
    prompt_path = variant_dir / "prompt.txt"
    auto_bottom_path = variant_dir / "auto_bottom_reference.png"
    engine_status = "unknown"
    if status_path.exists():
        try:
            engine_status = json.loads(status_path.read_text(encoding="utf-8")).get("status", "unknown")
        except json.JSONDecodeError:
            engine_status = "invalid_status_json"
    return {
        "sample_id": sample.sample_id,
        "variant": variant,
        "status": status,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "result_path": _display_path(result_path, output_dir),
        "prompt_path": _display_path(prompt_path, output_dir) if prompt_path.exists() else None,
        "status_path": _display_path(status_path, output_dir) if status_path.exists() else None,
        "auto_bottom_reference_path": _display_path(auto_bottom_path, output_dir) if auto_bottom_path.exists() else None,
        "engine_status": engine_status,
        "error_code": error_code,
        "notes": notes,
    }


def _write_grid(sample: EvalSample, output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    row_by_variant = {row["variant"]: row for row in rows}
    cells: list[tuple[str, Path | None]] = [
        ("Person input", sample.person_path),
        ("Top garment reference", sample.garment_top),
    ]
    auto_bottom = None
    for row in rows:
        if row.get("auto_bottom_reference_path"):
            auto_bottom = output_dir / row["auto_bottom_reference_path"]
            break
    cells.append(("Auto bottom reference", auto_bottom))
    for label, variant in [
        ("IDM original", "idm_original"),
        ("Klein LoRA default prompt", "klein_lora_default_prompt"),
        ("Klein LoRA strong prompt", "klein_lora_strong_prompt"),
    ]:
        row = row_by_variant.get(variant)
        cells.append((label, output_dir / row["result_path"] if row and row.get("result_path") else None))

    cell_w, cell_h, header_h = 300, 400, 60
    grid = Image.new("RGB", (cell_w * len(cells), cell_h + header_h), "white")
    draw = ImageDraw.Draw(grid)
    font = _load_font(18)
    for index, (label, path) in enumerate(cells):
        x0 = index * cell_w
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x0 + (cell_w - (bbox[2] - bbox[0])) // 2, 18), label, fill=(20, 20, 20), font=font)
        if path and path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            grid.paste(image, (x0 + (cell_w - image.width) // 2, header_h + (cell_h - image.height) // 2))
        else:
            draw.rectangle((x0 + 20, header_h + 20, x0 + cell_w - 20, header_h + cell_h - 20), outline=(170, 170, 170), width=2)
            draw.text((x0 + 92, header_h + cell_h // 2), "Unavailable", fill=(110, 110, 110), font=font)
        if index:
            draw.line((x0, 0, x0, header_h + cell_h), fill=(210, 210, 210), width=2)
    path = output_dir / "comparison_grid.png"
    grid.save(path)
    return path


def _write_html(sample: EvalSample, output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    cards = []
    for row in rows:
        result = row.get("result_path")
        image_html = (
            f'<img src="{html.escape(result)}" alt="{html.escape(row["variant"])}">'
            if result
            else '<div class="missing">Unavailable or skipped</div>'
        )
        cards.append(
            "<article>"
            f"<h2>{html.escape(row['variant'])}</h2>{image_html}"
            f"<p>Status: {html.escape(str(row.get('status')))}</p>"
            f"<p>Runtime: {html.escape(str(row.get('runtime_seconds')))}s</p>"
            f"<p>Notes: {html.escape(str(row.get('notes') or ''))}</p>"
            "</article>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Klein LoRA ablation</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; color: #161616; background: #f5f6f8; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
    article {{ padding: 14px; background: #fff; border: 1px solid #d9dee7; border-radius: 6px; }}
    img {{ display: block; max-width: 100%; height: auto; background: #fff; }}
    .missing {{ display: grid; min-height: 280px; place-items: center; border: 1px dashed #aab3c0; color: #596273; }}
    p {{ font-size: 13px; line-height: 1.4; }}
  </style>
</head>
<body>
  <h1>Klein LoRA ablation: {html.escape(sample.sample_id)}</h1>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
    path = output_dir / "comparison_index.html"
    path.write_text(document, encoding="utf-8")
    return path


def _write_manual_template(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "manual_ratings_klein_lora.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_RATING_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "variant": row["variant"],
                    "result_path": row.get("result_path"),
                }
            )
    return path


def run_ablation(
    *,
    sample_dir: Path,
    seed: int,
    bottom_strategy: str,
    output_dir: Path,
    include_idm: bool = True,
    mock: bool = False,
) -> dict[str, Any]:
    sample, issues = validate_sample(sample_dir)
    if sample is None:
        raise ValueError("Invalid eval sample: " + "; ".join(issues))
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    rows: list[dict[str, Any]] = []

    person = load_image_from_path(sample.person_path, max_side=settings.image.max_side)
    save_image(person, output_dir / "input_person.png")
    if sample.garment_top:
        top = load_image_from_path(sample.garment_top, max_side=settings.image.max_side)
        save_image(top, output_dir / "input_garment_top.png")

    idm_result = output_dir / "idm_original" / "result.png"
    if include_idm or not idm_result.exists():
        rows.append(_run_idm_original(sample, output_dir, settings, seed=seed, mock=mock))

    rows.append(
        _run_klein_variant(
            sample,
            output_dir,
            settings,
            variant="klein_lora_default_prompt",
            prompt=DEFAULT_PROMPT,
            seed=seed,
            bottom_strategy=bottom_strategy,
        )
    )
    rows.append(
        _run_klein_variant(
            sample,
            output_dir,
            settings,
            variant="klein_lora_strong_prompt",
            prompt=STRONG_PROMPT,
            seed=seed,
            bottom_strategy=bottom_strategy,
        )
    )

    summary = {
        "sample_id": sample.sample_id,
        "seed": seed,
        "bottom_strategy": bottom_strategy,
        "rows": rows,
        "notes": [
            "Klein LoRA is an experimental baseline and is not the production default.",
            "Subjective observations belong in manual_ratings_klein_lora.csv.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_grid(sample, output_dir, rows)
    _write_html(sample, output_dir, rows)
    _write_manual_template(rows, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Klein Try-On LoRA experimental ablation.")
    parser.add_argument("--sample", default="data/eval_set/sample_001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bottom-strategy", default="crop_from_person", choices=["crop_from_person", "blank_placeholder", "skip"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-idm", action="store_true", default=True)
    parser.add_argument("--skip-idm", action="store_false", dest="include_idm")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_project_path(args.output) if args.output else PROJECT_ROOT / "data" / "outputs" / f"klein_lora_ablation_{timestamp}"
    summary = run_ablation(
        sample_dir=_resolve_project_path(args.sample),
        seed=args.seed,
        bottom_strategy=args.bottom_strategy,
        output_dir=output_dir,
        include_idm=args.include_idm,
        mock=args.mock,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"comparison_grid={output_dir / 'comparison_grid.png'}")
    print(f"comparison_index={output_dir / 'comparison_index.html'}")
    print(f"summary_csv={output_dir / 'summary.csv'}")
    print(f"summary_json={output_dir / 'summary.json'}")
    print(f"manual_ratings={output_dir / 'manual_ratings_klein_lora.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
