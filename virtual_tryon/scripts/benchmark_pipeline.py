from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import load_settings  # noqa: E402
from app.preprocessing.image_loader import load_image_from_path  # noqa: E402
from app.services.storage_service import StorageService  # noqa: E402
from app.services.tryon_pipeline import PipelineRequest, TryOnPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark pipeline variants on repeated samples.")
    parser.add_argument("--person", required=True)
    parser.add_argument("--garment", required=True)
    parser.add_argument("--category", default="upper_body", choices=["upper_body", "lower_body", "dress", "full_outfit"])
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "outputs" / "benchmarks"))
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    if args.mock:
        settings.pipeline.engine = "mock"
    storage = StorageService(settings.storage)
    pipeline = TryOnPipeline(settings, storage)
    person = load_image_from_path(args.person, max_side=settings.image.max_side)
    garment = load_image_from_path(args.garment, max_side=settings.image.max_side)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    variants = [
        ("core_only", False, False),
        ("core_plus_refine", True, False),
        ("core_plus_refine_repair", True, True),
    ]

    for idx in range(args.n):
        for variant, use_refiner, repair_mode in variants:
            job_id = f"bench_{variant}_{idx}"
            request = PipelineRequest(
                job_id=job_id,
                person_image=person,
                garment_top=garment if args.category in {"upper_body", "full_outfit"} else None,
                garment_bottom=garment if args.category == "lower_body" else None,
                garment_dress=garment if args.category == "dress" else None,
                category=args.category,
                prompt=None,
                use_refiner=use_refiner,
                repair_mode=repair_mode,
                seed=idx,
            )
            response = pipeline.run(request)
            metrics.append({"variant": variant, **response.model_dump()})

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
