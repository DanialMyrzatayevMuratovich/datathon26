#!/usr/bin/env bash
# Transport only: AI inference stays on the host. No tokens in command arguments.
set -euo pipefail
command -v cloudflared >/dev/null || { echo "Install cloudflared first (see README.md)."; exit 1; }
if [ -n "${TUNNEL_NAME:-}" ]; then
  # Requires an existing local cloudflared config with ingress pointing to localhost:8000.
  exec cloudflared tunnel --no-autoupdate run "$TUNNEL_NAME"
fi
exec cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 --protocol http2
