#!/usr/bin/env bash
# Prints the Colab link for notebooks/colab_train.ipynb of the pushed repo, so a long or GPU run moves off the laptop
# in one click. Push first: Colab clones what is on GitHub, not your working copy.
# Usage: scripts/colab.sh [branch]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
url=$(git remote get-url origin 2>/dev/null) || { echo "✖ нет origin — сначала подключи GitHub-репо"; exit 1; }
slug=$(echo "$url" | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')
branch="${1:-$(git rev-parse --abbrev-ref HEAD)}"
[ -n "$(git status --porcelain)" ] && echo "⚠ есть незакоммиченные правки — в Colab их не будет (scripts/checkpoint.sh)"
[ "$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)" != "0" ] && echo "⚠ есть незапушенные коммиты — scripts/checkpoint.sh"
echo "https://colab.research.google.com/github/$slug/blob/$branch/notebooks/colab_train.ipynb"
echo "Приватный репо: в Colab → Secrets добавь GH_TOKEN (fine-grained, только этот репо, Contents: Read and write)."
