#!/usr/bin/env bash
# Pinned model weights for the video pipeline (docs/ML_SPEC.md §7). Weights live NEXT TO the repo, never in git:
# ~/Desktop/хакатон-дататон/models (override: MODELS_DIR=...). Pinned revisions = the same weights on every laptop,
# in Docker and in both evaluation runs. All licenses: Apache-2.0 / MIT (checked on the model cards 17.10.2026).
# Usage: scripts/download_models.sh [core|all]      core ≈ 7.9 GB (default), all ≈ 14 GB
# Needs: brew install hf
set -euo pipefail
command -v hf >/dev/null 2>&1 || { echo "✖ нет команды hf — поставь: brew install hf"; exit 1; }
root=$(cd "$(dirname "$0")/.." && pwd)
dest="${MODELS_DIR:-$root/../models}"
mkdir -p "$dest"
dest=$(cd "$dest" && pwd)
set_="${1:-core}"

dl() {  # dl <repo> <revision> <folder> [hf download options...]
  local repo=$1 rev=$2 dir=$3; shift 3
  echo "→ $repo @ ${rev:0:8} → $dest/$dir"
  hf download "$repo" --revision "$rev" --local-dir "$dest/$dir" "$@"
}

# core — needed for the first experiments (Э1–Э3) and the RAG indexes
dl mlx-community/Qwen3-VL-4B-Instruct-4bit 2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b qwen3-vl-4b-instruct-mlx-4bit  # VLM, 3.1 GB
dl PekingU/rtdetr_v2_r50vd 282494075698cab9faa1096ae26856890030c817 rtdetr-v2-r50                                 # detector, 0.2 GB
dl BAAI/bge-m3 5617a9f61b028005a4858fdac845db406aefb181 bge-m3 --exclude "onnx/*"                                  # text embeddings (RAG), 2.3 GB
dl google/siglip2-base-patch16-224 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 siglip2-base                          # image embeddings, 1.5 GB
dl IDEA-Research/grounding-dino-tiny a2bb814dd30d776dcf7e30523b00659f4f141c71 grounding-dino-tiny --exclude "pytorch_model.bin"  # open-vocab, 0.7 GB

if [ "$set_" = "all" ]; then
  dl mlx-community/Qwen3-VL-8B-Instruct-4bit defcdea7cc7a4b0858fea563cbbce171d328e457 qwen3-vl-8b-instruct-mlx-4bit  # VLM quality, 5.8 GB
  dl usyd-community/vitpose-base-simple a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45 vitpose-base                     # pose, 0.3 GB
fi
# the pinned MLX Qwen revision ships a stale model.safetensors.index.json (it names two shards that are not in the repo,
# the weights are one model.safetensors); mlx-vlm then falls back to *.safetensors, but the service preflight refuses
# the set. Put such an index aside (kept as *.stale).
for idx in "$dest"/*/model.safetensors.index.json; do
  [ -f "$idx" ] || continue
  d=$(dirname "$idx")
  if python3 -c 'import json,sys,os; d=sys.argv[1]; m=json.load(open(os.path.join(d,"model.safetensors.index.json")))["weight_map"]; sys.exit(0 if all(os.path.exists(os.path.join(d,f)) for f in set(m.values())) else 1)' "$d"; then :; else
    mv "$idx" "$idx.stale" && echo "… $d: устаревший индекс весов отложен (*.stale)"
  fi
done
echo "✔ веса в $dest"
du -sh "$dest"/* 2>/dev/null
