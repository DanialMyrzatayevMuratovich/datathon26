#!/usr/bin/env bash
# Run from Git Bash on Windows. Ctrl+C stops only processes started by this script.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
use_tunnel=0
case "${1:-}" in
  --tunnel) use_tunnel=1 ;;
  "") ;;
  *) echo "Usage: bash scripts/run_service.sh [--tunnel]"; exit 2 ;;
esac
for command in uv ffmpeg ffprobe; do
  command -v "$command" >/dev/null || { echo "Missing: $command (see README.md)"; exit 1; }
done
if [ "$use_tunnel" = 1 ]; then
  command -v cloudflared >/dev/null || { echo "Missing: cloudflared (see README.md)"; exit 1; }
fi
uv run --frozen python -m service.cli doctor --require-ui
server_pid=""; tunnel_pid=""; awake_pid=""
cleanup() {
  trap - EXIT INT TERM
  for pid in "$tunnel_pid" "$server_pid" "$awake_pid"; do
    if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if command -v caffeinate >/dev/null; then
  caffeinate -dims & awake_pid=$!
fi
uv run --frozen uvicorn service.main:app --host 127.0.0.1 --port 8000 --workers 1 & server_pid=$!
if ! uv run --frozen python - <<'PY'
import time
import urllib.error
import urllib.request
for _ in range(60):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=1) as response:
            if response.status == 200:
                break
    except (OSError, urllib.error.URLError):
        time.sleep(0.5)
else:
    raise SystemExit('API did not start within 30 seconds')
PY
then exit 1; fi
if ! kill -0 "$server_pid" 2>/dev/null; then echo "API exited; check port 8000."; exit 1; fi
echo "API ready: http://127.0.0.1:8000/api/v1/health"
if [ "$use_tunnel" = 1 ]; then
  bash scripts/tunnel.sh & tunnel_pid=$!
  while kill -0 "$server_pid" 2>/dev/null && kill -0 "$tunnel_pid" 2>/dev/null; do sleep 1; done
  if ! kill -0 "$tunnel_pid" 2>/dev/null; then
    echo "Tunnel stopped. Check network/cloudflared output, then restart; no public URL is active."
    exit 1
  fi
fi
wait "$server_pid"
