#!/usr/bin/env bash
# What the jury does with our repo: fresh clone -> install -> lint -> tests -> the pipeline end to end -> API.
# Without organizer data (it is not in git) the pipeline runs on the synthetic demo set; with --with-data the
# working copy's data/raw is copied into the clone and the real `auto` from task.toml is reproduced.
# The working copy and its results are never changed.
# Usage: scripts/smoke.sh [--local] [--with-data]   (--local: last local commit instead of origin)
set -euo pipefail
root=$(git rev-parse --show-toplevel)
source=$(git -C "$root" remote get-url origin 2>/dev/null || echo "$root")
with_data=0
for a in "$@"; do
  case "$a" in --local) source="$root" ;; --with-data) with_data=1 ;; esac
done
work=$(mktemp -d "${TMPDIR:-/tmp}/dt-smoke.XXXXXX")
server_pid=""
trap '[ -z "$server_pid" ] || kill "$server_pid" 2>/dev/null || true' EXIT
step() {
  local name="$1"; shift
  if "$@" >"$work/$name.log" 2>&1; then
    echo "✔ $name"
  else
    echo "✖ $name — $work/$name.log"
    tail -n 20 "$work/$name.log"
    exit 1
  fi
}
echo "Логи и отдельный клон: $work"
[ -n "$(git -C "$root" status --porcelain)" ] && echo "Внимание: незакоммиченные правки не входят в проверку."
step clone git clone --quiet --no-hardlinks "$source" "$work/repo"
cd "$work/repo"
{ git rev-parse HEAD; uname -srm; date -u '+%Y-%m-%dT%H:%M:%SZ'; } > "$work/environment.txt"
export DEMO_MODE=1
step install uv sync --frozen
uv run python --version >> "$work/environment.txt"
step lint uv run ruff check .
step tests uv run pytest -q
if [ "$with_data" = 1 ]; then
  mkdir -p data/raw && cp -R "$root/data/raw/." data/raw/
  step auto uv run python -m app.cli auto --note "smoke: fresh clone"
else
  step demo uv run python -m app.cli demo --task binary
fi
uv run python - "$work/port" >"$work/server.log" 2>&1 <<'PY' &
import socket
import sys
from pathlib import Path
import uvicorn
sock = socket.socket()
sock.bind(('127.0.0.1', 0))
Path(sys.argv[1]).write_text(str(sock.getsockname()[1]))
uvicorn.Server(uvicorn.Config('app.main:app', log_level='warning')).run(sockets=[sock])
PY
server_pid=$!
up=0
for _ in $(seq 1 60); do
  if [ -s "$work/port" ] && curl -fsS "http://127.0.0.1:$(cat "$work/port")/api/health" >/dev/null 2>&1; then up=1; break; fi
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 1
done
[ "$up" = 1 ] || { echo "✖ server — $work/server.log"; exit 1; }
step http uv run python - "$(cat "$work/port")" <<'PY'
import json
import sys
from urllib.request import urlopen
base = f'http://127.0.0.1:{sys.argv[1]}'
get = lambda p: json.load(urlopen(base + p, timeout=15))
assert get('/api/health')['best'], 'no best model'
runs = get('/api/runs')
assert runs and get('/api/runs/' + runs[0]['run_id'])['steps']
print('health, runs, run detail: OK')
PY
printf 'SMOKE PASS: install lint tests pipeline http\n' | tee "$work/result.txt"
echo "Отчёт: $work/result.txt; окружение: $work/environment.txt"
