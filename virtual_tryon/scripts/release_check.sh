#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

failures=0
PYTHON_BIN="${TRYON_PYTHON_BIN:-python}"
if [[ -x /workspace/venvs/project_phase2/bin/python && -z "${TRYON_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN=/workspace/venvs/project_phase2/bin/python
fi

run_required() {
  local label="$1"
  shift
  echo
  echo "==> $label"
  if "$@"; then
    echo "PASS: $label"
  else
    echo "FAIL: $label"
    failures=$((failures + 1))
  fi
}

echo "Virtual Try-On release check"
echo "commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
git status --short

staged_forbidden="$(git diff --cached --name-only | grep -E '(^|/)(models|third_party|node_modules|dist|data/outputs|data/temp)/|(^|/)\.env($|\.)' || true)"
if [[ -n "$staged_forbidden" ]]; then
  echo "FAIL: forbidden generated or sensitive paths are staged:"
  echo "$staged_forbidden"
  failures=$((failures + 1))
else
  echo "PASS: no forbidden generated or sensitive paths staged"
fi

run_required "Backend mock tests" env PYTHONPATH=. TRYON_ENGINE=mock "$PYTHON_BIN" -m pytest
run_required "Evaluation set validation" "$PYTHON_BIN" scripts/validate_eval_set.py --eval-set data/eval_set

if command -v npm >/dev/null 2>&1; then
  (
    cd frontend
    run_required "Frontend build" npm run build
  )
else
  echo "WARN: npm is unavailable; frontend build skipped"
fi

if [[ -n "${TRYON_BACKEND_URL:-}" ]]; then
  run_required \
    "API E2E smoke" \
    "$PYTHON_BIN" scripts/e2e_smoke_test.py \
      --api-base "$TRYON_BACKEND_URL" \
      --sample data/eval_set/sample_001 \
      --use-refiner false \
      --timeout "${TRYON_E2E_TIMEOUT:-900}"
else
  echo "WARN: TRYON_BACKEND_URL is not set; API E2E smoke skipped"
fi

echo
echo "Release checklist"
echo "[ ] Real IDM-VTON smoke passed"
echo "[ ] Benchmark eval set passed"
echo "[ ] Playwright E2E passed"
echo "[ ] /health, /system, and /metrics checked"
echo "[ ] Artifact cleanup dry-run checked"

if (( failures > 0 )); then
  echo "Release check failed with $failures required check(s)."
  exit 1
fi
echo "Required automated release checks passed."
