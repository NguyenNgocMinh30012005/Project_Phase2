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
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from app.core.config import load_settings  # noqa: E402
from app.engines.factory import create_tryon_engine  # noqa: E402
from app.preprocessing.image_loader import load_image_from_path  # noqa: E402
from app.services.storage_service import StorageService  # noqa: E402
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline  # noqa: E402
from app.utils.errors import ModelUnavailableError, TryOnError  # noqa: E402
from app.utils.image_io import save_image  # noqa: E402
from build_review_gallery import build_gallery  # noqa: E402
from validate_eval_set import EvalSample, discover_eval_samples  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_MODES = {"idm", "idm_flux", "catvton", "klein_lora", "repair"}
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


def _display_path(path: Path, benchmark_dir: Path) -> str:
    try:
        return path.resolve().relative_to(benchmark_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_grid(rows: list[dict], benchmark_dir: Path) -> None:
    images: list[tuple[str, Image.Image]] = []
    for row in rows:
        output_path = row.get("output_path")
        if not output_path:
            continue
        path = benchmark_dir / output_path
        if not path.exists():
            continue
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            continue
        label = f"{row['sample_id']} | {row['mode']} | {row.get('final_choice') or row.get('status')}"
        images.append((label, image))

    if not images:
        return

    cell_w, cell_h, label_h = 256, 342, 30
    cols = min(4, len(images))
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
        draw.text((col * cell_w + 8, row * (cell_h + label_h) + 8), label[:44], fill=(30, 30, 30))
        grid.paste(thumb, (x, y))
    grid.save(benchmark_dir / "grid.png")


def _quality_columns(report: dict | None) -> dict:
    if not report:
        return {
            "background_preservation_score": None,
            "face_preservation_score": None,
            "garment_change_score": None,
            "over_edit_score": None,
            "final_choice": None,
            "notes": "",
        }
    final_choice = report.get("final_choice") or "core"
    chosen = report.get(final_choice) or report.get("core") or {}
    notes = chosen.get("notes") or []
    if report.get("final_choice_reason"):
        notes = [report["final_choice_reason"], *notes]
    return {
        "background_preservation_score": chosen.get("background_preservation_score"),
        "face_preservation_score": chosen.get("face_preservation_score"),
        "garment_change_score": chosen.get("garment_change_score"),
        "over_edit_score": chosen.get("over_edit_score"),
        "final_choice": final_choice,
        "notes": "; ".join(str(note) for note in notes),
    }


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in SUPPORTED_MODES]
    if invalid:
        raise SystemExit(f"Unsupported benchmark mode(s): {', '.join(invalid)}")
    return modes or ["idm", "idm_flux", "catvton", "klein_lora"]


def _load_legacy_sample(args: argparse.Namespace) -> list[EvalSample]:
    if not args.person or not args.garment:
        return []
    metadata = {
        "sample_id": "sample_001",
        "category": args.category,
        "difficulty": "medium",
        "expected_focus": ["identity", "garment_texture"],
        "notes": "Temporary sample created from --person/--garment arguments.",
    }
    garment_top = Path(args.garment) if args.category in {"upper_body", "full_outfit"} else None
    garment_bottom = Path(args.garment) if args.category == "lower_body" else None
    garment_dress = Path(args.garment) if args.category == "dress" else None
    return [
        EvalSample(
            sample_id="sample_001",
            root=Path(args.person).parent,
            person_path=Path(args.person),
            garment_top=garment_top,
            garment_bottom=garment_bottom,
            garment_dress=garment_dress,
            category=args.category,
            difficulty="medium",
            metadata=metadata,
        )
    ]


def _mode_settings(settings, mode: str, mock: bool):
    mode_settings = settings.model_copy(deep=True)
    if mode in {"idm", "idm_flux", "repair"}:
        mode_settings.pipeline.engine = "mock" if mock else "idm_vton"
    elif mode == "catvton":
        mode_settings.pipeline.engine = "catvton"
    elif mode == "klein_lora":
        mode_settings.pipeline.engine = "klein_tryon_lora"
    return mode_settings


def _mode_flags(mode: str) -> tuple[bool, bool]:
    if mode == "idm_flux":
        return True, False
    if mode == "repair":
        return True, True
    return False, False


def _copy_inputs(sample: EvalSample, sample_dir: Path, benchmark_dir: Path, settings) -> dict[str, str | None]:
    person = load_image_from_path(sample.person_path, max_side=settings.image.max_side)
    paths: dict[str, str | None] = {}
    paths["input_person_path"] = _display_path(save_image(person, sample_dir / "input_person.png"), benchmark_dir)
    for attr, filename in [
        ("garment_top", "input_garment_top.png"),
        ("garment_bottom", "input_garment_bottom.png"),
        ("garment_dress", "input_garment_dress.png"),
    ]:
        source = getattr(sample, attr)
        if source:
            image = load_image_from_path(source, max_side=settings.image.max_side)
            paths[f"input_{attr}_path"] = _display_path(save_image(image, sample_dir / filename), benchmark_dir)
    paths["input_garment_path"] = paths.get("input_garment_top_path") or paths.get("input_garment_bottom_path") or paths.get("input_garment_dress_path")
    return paths


def _request_for_sample(
    sample: EvalSample,
    *,
    job_id: str,
    prompt: str | None,
    use_refiner: bool,
    repair_mode: bool,
    seed: int,
    settings,
) -> PipelineRequest:
    person = load_image_from_path(sample.person_path, max_side=settings.image.max_side)
    garment_top = load_image_from_path(sample.garment_top, max_side=settings.image.max_side) if sample.garment_top else None
    garment_bottom = load_image_from_path(sample.garment_bottom, max_side=settings.image.max_side) if sample.garment_bottom else None
    garment_dress = load_image_from_path(sample.garment_dress, max_side=settings.image.max_side) if sample.garment_dress else None
    return PipelineRequest(
        job_id=job_id,
        person_image=person,
        garment_top=garment_top,
        garment_bottom=garment_bottom,
        garment_dress=garment_dress,
        category=sample.category,
        prompt=prompt,
        use_refiner=use_refiner,
        repair_mode=repair_mode,
        seed=seed,
    )


def _skip_row(sample: EvalSample, mode: str, sample_dir: Path, benchmark_dir: Path, input_paths: dict, reason: str, started: float) -> dict:
    mode_dir = sample_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "skip_reason.txt").write_text(reason, encoding="utf-8")
    return {
        "sample_id": sample.sample_id,
        "mode": mode,
        "status": "unavailable",
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "output_path": None,
        "mode_dir": _display_path(mode_dir, benchmark_dir),
        "sample_dir": _display_path(sample_dir, benchmark_dir),
        "quality_report_path": None,
        "run_metadata_path": None,
        **_quality_columns(None),
        "final_choice": None,
        **input_paths,
        "notes": reason,
    }


def _run_mode(
    sample: EvalSample,
    mode: str,
    sample_dir: Path,
    benchmark_dir: Path,
    input_paths: dict,
    settings,
    *,
    mock: bool,
    prompt: str | None,
    seed: int,
) -> tuple[dict, dict | None]:
    started = time.perf_counter()
    mode_settings = _mode_settings(settings, mode, mock)
    storage = StorageService(mode_settings.storage)
    mode_dir = sample_dir / mode
    job_id = f"{benchmark_dir.name}/{sample.sample_id}/{mode}"
    engine = create_tryon_engine(mode_settings)
    if not engine.is_available():
        status = engine.status() if hasattr(engine, "status") else f"{getattr(engine, 'name', mode)} unavailable"
        return _skip_row(sample, mode, sample_dir, benchmark_dir, input_paths, status, started), None

    use_refiner, repair_mode = _mode_flags(mode)
    try:
        request = _request_for_sample(
            sample,
            job_id=job_id,
            prompt=prompt,
            use_refiner=use_refiner,
            repair_mode=repair_mode,
            seed=seed,
            settings=mode_settings,
        )
        response = TryOnPipeline(mode_settings, storage).run(request)
        result_path = storage.file_path_from_public_url(response.result_url) if response.result_url else None
        quality_path = mode_dir / "quality_report.json"
        metadata_path = mode_dir / "metadata.json"
        report = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
        row = {
            "sample_id": sample.sample_id,
            "mode": mode,
            "status": response.status,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "output_path": _display_path(result_path, benchmark_dir) if result_path else None,
            "mode_dir": _display_path(mode_dir, benchmark_dir),
            "sample_dir": _display_path(sample_dir, benchmark_dir),
            "quality_report_path": _display_path(quality_path, benchmark_dir) if quality_path.exists() else None,
            "run_metadata_path": _display_path(metadata_path, benchmark_dir) if metadata_path.exists() else None,
            **input_paths,
            **_quality_columns(report),
        }
        return row, report
    except ModelUnavailableError as exc:
        return _skip_row(sample, mode, sample_dir, benchmark_dir, input_paths, str(exc), started), None
    except TryOnError as exc:
        mode_dir.mkdir(parents=True, exist_ok=True)
        (mode_dir / "benchmark_error.txt").write_text(str(exc), encoding="utf-8")
        row = _skip_row(sample, mode, sample_dir, benchmark_dir, input_paths, f"failed: {exc}", started)
        row["status"] = "failed"
        return row, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Virtual Try-On engines on a golden eval set.")
    parser.add_argument("--eval-set", default=None, help="Golden evaluation set folder.")
    parser.add_argument("--modes", default="idm,idm_flux,catvton,klein_lora")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--person", default=None, help="Legacy single person image fallback.")
    parser.add_argument("--garment", default=None, help="Legacy single garment image fallback.")
    parser.add_argument("--category", default="upper_body", choices=["upper_body", "lower_body", "dress", "full_outfit"])
    parser.add_argument("--prompt", default="replace the shirt with the reference garment, preserve face, pose, and body shape")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modes = _parse_modes(args.modes)
    settings = load_settings()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) if args.output else settings.storage.outputs_dir / f"benchmark_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    settings.storage.outputs_dir = output_dir.parent

    issues: list[dict] = []
    if args.eval_set:
        samples, issues = discover_eval_samples(Path(args.eval_set))
    else:
        samples = _load_legacy_sample(args)
        if not samples:
            samples, issues = discover_eval_samples(PROJECT_ROOT / "data" / "eval_set")
    if args.limit is not None:
        samples = samples[: args.limit]

    rows: list[dict] = []
    sample_reports: dict[str, dict] = {}
    if not samples:
        message = "No valid eval samples found. Run scripts/validate_eval_set.py for details."
        rows = []
        (output_dir / "summary.json").write_text(
            json.dumps({"rows": rows, "issues": issues, "notes": [message]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        build_gallery(output_dir)
        print(message)
        return 0

    for sample in samples:
        sample_dir = output_dir / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        input_paths = _copy_inputs(sample, sample_dir, output_dir, settings)
        sample_metadata = {"sample": sample.metadata, "modes": {}, "issues": []}
        sample_quality = {"sample_id": sample.sample_id, "modes": {}, "engine_status": {}}
        for mode in modes:
            row, report = _run_mode(
                sample,
                mode,
                sample_dir,
                output_dir,
                input_paths,
                settings,
                mock=args.mock,
                prompt=args.prompt,
                seed=0,
            )
            rows.append(row)
            sample_metadata["modes"][mode] = row
            sample_quality["modes"][mode] = report
            sample_quality["engine_status"][mode] = row["status"]
        (sample_dir / "run_metadata.json").write_text(json.dumps(sample_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        (sample_dir / "quality_report.json").write_text(json.dumps(sample_quality, indent=2, ensure_ascii=False), encoding="utf-8")
        sample_reports[sample.sample_id] = sample_quality

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(
        json.dumps({"rows": rows, "issues": issues, "sample_reports": sample_reports}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_grid(rows, output_dir)
    gallery_path = build_gallery(output_dir)

    print(f"summary_json={summary_json}")
    print(f"summary_csv={summary_csv}")
    print(f"grid={output_dir / 'grid.png'}")
    print(f"index={gallery_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
