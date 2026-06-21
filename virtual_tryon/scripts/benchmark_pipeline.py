from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import load_settings  # noqa: E402
from app.preprocessing.image_loader import load_image_from_path  # noqa: E402
from app.services.storage_service import StorageService  # noqa: E402
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CSV_COLUMNS = [
    "sample_id",
    "mode",
    "status",
    "runtime_seconds",
    "output_path",
    "background_preservation_score",
    "face_preservation_score",
    "garment_change_score",
    "over_edit_score",
    "final_choice",
    "notes",
]


def _discover_samples(examples_dir: Path, max_samples: int) -> list[tuple[str, Path, Path]]:
    persons = sorted(
        path
        for path in examples_dir.glob("person_*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    tops = sorted(
        path
        for path in examples_dir.glob("top_*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [(f"sample_{idx:03d}", person, top) for idx, (person, top) in enumerate(zip(persons, tops), start=1)][
        :max_samples
    ]


def _quality_columns(report: dict) -> dict:
    final_choice = report.get("final_choice") or "core"
    chosen = report.get(final_choice) or report.get("core") or {}
    notes = chosen.get("notes") or []
    return {
        "background_preservation_score": chosen.get("background_preservation_score"),
        "face_preservation_score": chosen.get("face_preservation_score"),
        "garment_change_score": chosen.get("garment_change_score"),
        "over_edit_score": chosen.get("over_edit_score"),
        "final_choice": final_choice,
        "notes": "; ".join(str(note) for note in notes),
    }


def _write_grid(rows: list[dict], grid_path: Path) -> None:
    images: list[tuple[str, Image.Image]] = []
    for row in rows:
        output_path = row.get("output_path")
        if not output_path:
            continue
        path = Path(output_path)
        if not path.exists():
            continue
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            continue
        label = f"{row['sample_id']} | {row['mode']} | {row.get('final_choice') or 'n/a'}"
        images.append((label, image))

    if not images:
        return

    cell_w, cell_h, label_h = 256, 342, 30
    cols = min(3, len(images))
    rows_count = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_w, rows_count * (cell_h + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    for idx, (label, image) in enumerate(images):
        col = idx % cols
        row = idx // cols
        thumb = image.copy()
        thumb.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        x = col * cell_w + (cell_w - thumb.width) // 2
        y = row * (cell_h + label_h) + label_h + (cell_h - thumb.height) // 2
        draw.text((col * cell_w + 8, row * (cell_h + label_h) + 8), label[:42], fill=(30, 30, 30))
        grid.paste(thumb, (x, y))
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(grid_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark IDM-VTON pipeline variants.")
    parser.add_argument("--person", default=None, help="Optional single person image.")
    parser.add_argument("--garment", default=None, help="Optional single garment image.")
    parser.add_argument("--examples-dir", default=str(PROJECT_ROOT / "data" / "examples"))
    parser.add_argument("--category", default="upper_body", choices=["upper_body", "lower_body", "dress", "full_outfit"])
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--prompt", default="replace the shirt with the reference garment, preserve face, pose, and body shape")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    if args.mock:
        settings.pipeline.engine = "mock"
        settings.pipeline.allow_mock_engine = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_name = f"benchmark_{timestamp}"
    output_dir = Path(args.output_dir) if args.output_dir else settings.storage.outputs_dir / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.person and args.garment:
        samples = [("sample_001", Path(args.person), Path(args.garment))]
    else:
        samples = _discover_samples(Path(args.examples_dir), args.max_samples)

    if not samples:
        message = (
            "No benchmark samples found. Provide --person and --garment, or add "
            "person_001.jpg/top_001.jpg style pairs under data/examples."
        )
        (output_dir / "summary.json").write_text(
            json.dumps({"rows": [], "notes": [message]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(message)
        return 0

    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)
    modes = [
        ("idm_vton_only", False, False),
        ("idm_vton_flux_refiner", True, False),
        ("idm_vton_flux_refiner_repair", True, True),
    ]
    rows: list[dict] = []

    for sample_id, person_path, garment_path in samples:
        person = load_image_from_path(person_path, max_side=settings.image.max_side)
        garment = load_image_from_path(garment_path, max_side=settings.image.max_side)
        for mode, use_refiner, repair_mode in modes:
            job_id = f"{output_dir.name}/{sample_id}_{mode}"
            started = time.perf_counter()
            row = {
                "sample_id": sample_id,
                "mode": mode,
                "status": "failed",
                "runtime_seconds": None,
                "output_path": None,
                "background_preservation_score": None,
                "face_preservation_score": None,
                "garment_change_score": None,
                "over_edit_score": None,
                "final_choice": None,
                "notes": "",
            }
            try:
                request = PipelineRequest(
                    job_id=job_id,
                    person_image=person,
                    garment_top=garment if args.category in {"upper_body", "full_outfit"} else None,
                    garment_bottom=garment if args.category == "lower_body" else None,
                    garment_dress=garment if args.category == "dress" else None,
                    category=args.category,
                    prompt=args.prompt,
                    use_refiner=use_refiner,
                    repair_mode=repair_mode,
                    seed=0,
                )
                response = pipeline.run(request)
                row["status"] = response.status
                if response.result_url:
                    row["output_path"] = str(storage.file_path_from_public_url(response.result_url))
                quality_report_path = storage.outputs_dir / job_id / "quality_report.json"
                if quality_report_path.exists():
                    report = json.loads(quality_report_path.read_text(encoding="utf-8"))
                    row.update(_quality_columns(report))
            except Exception as exc:
                row["notes"] = f"Pipeline failed: {exc}"
                error_path = storage.outputs_dir / job_id / "benchmark_error.txt"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(str(exc), encoding="utf-8")
            finally:
                row["runtime_seconds"] = round(time.perf_counter() - started, 3)
                rows.append(row)

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _write_grid(rows, output_dir / "grid.png")

    print(f"summary_json={summary_json}")
    print(f"summary_csv={summary_csv}")
    print(f"grid={output_dir / 'grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
