#!/usr/bin/env bash
# Fresh checkout -> locked install (or supplied existing Python) -> tests -> five-video API check.
# --with-data uses local videos and weights; never copies them or starts an external AI reviewer.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
source=$(git -C "$root" remote get-url origin 2>/dev/null || echo "$root")
real=0
for arg in "$@"; do
  case "$arg" in --local) source="$root" ;; --with-data) real=1 ;; *) echo "Unknown option: $arg"; exit 2 ;; esac
done
work=$(mktemp -d "${TMPDIR:-/tmp}/dt-smoke.XXXXXX")
echo "Fresh checkout and logs: $work"
git clone --quiet --no-hardlinks "$source" "$work/repo"
cd "$work/repo"
if [ -n "${SMOKE_PYTHON:-}" ]; then
  py=("$SMOKE_PYTHON")
else
  uv sync --frozen
  py=(uv run python)
fi
"${py[@]}" -m ruff check .
"${py[@]}" -m pytest -q
if [ "$real" = 1 ]; then
  export MODELS_DIR="${MODELS_DIR:-$root/../models}"
  "${py[@]}" -m vision.evaluate --videos "$root/data/raw/open/videos" --labels "$root/data/annotations.csv"
  "${py[@]}" -m service.smoke --real --videos "$root/data/raw/open/videos" --report "$work/acceptance.json"
else
  "${py[@]}" -m service.smoke --report "$work/acceptance.json"
fi
echo "SMOKE PASS; report: $work/acceptance.json (real=$real)"
