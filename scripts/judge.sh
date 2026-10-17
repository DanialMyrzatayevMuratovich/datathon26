#!/usr/bin/env bash
# Build docs/aggregate review input; never copy bank data into an AI agent checkout.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
work=$(mktemp -d "${TMPDIR:-/tmp}/dt-judge.XXXXXX")
uv run python -m service.review_packet --out "$work/review.txt"
echo "Review only $work/review.txt. No external reviewer has been launched."
echo "Real reproduction: scripts/smoke.sh --with-data (local only, separate from AI review)."
