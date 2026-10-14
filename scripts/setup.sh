#!/usr/bin/env bash
# One-time setup for each teammate right after `git clone`: who you are (auto-detected from your GitHub login),
# commit identity, hooks, local env, Python deps and a check that every model library imports.
# Usage: scripts/setup.sh [id] [--web]   (ids — строки MEMBER в docs/ZONES.md; лид добавляет их scripts/team.sh add)
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
web=0; args=()
for a in "$@"; do [ "$a" = "--web" ] && web=1 || args+=("$a"); done
set -- ${args[@]+"${args[@]}"}

team=$(scripts/team.sh ids)
if [ -z "$(echo $team)" ]; then
  echo "✖ в docs/ZONES.md ещё нет участников. Лид, один раз: scripts/team.sh add <id> <Имя> <github-login> (для обоих),"
  echo "  затем впиши LEAD: <id> и сделай первый commit + push. Напарник после git pull запускает scripts/setup.sh."
  exit 1
fi
# who is at this keyboard: the GitHub account gh is logged into, else the noreply commit email
login=$(gh api user --jq .login 2>/dev/null || true)
[ -z "$login" ] && login=$(git config user.email 2>/dev/null | sed -n 's/^[0-9]*+\([^@]*\)@users\.noreply\.github\.com$/\1/p')
detected=""; [ -n "$login" ] && detected=$(scripts/team.sh whois "$login")
role="${1:-${detected:-$(git config dt.role 2>/dev/null || true)}}"
known=0; for t in $team; do [ "$t" = "$role" ] && known=1; done
if [ "$known" != 1 ]; then
  if [ -n "${1:-}" ]; then why="нет такого id: $1"
  elif [ -n "$login" ]; then why="GitHub @$login нет в docs/ZONES.md (строки MEMBER)"
  else why="gh не залогинен, id не указан"; fi
  echo "✖ не понял, кто ты ($why). Укажи: scripts/setup.sh <$(echo $team | tr ' ' '|')>"; exit 1
fi
if [ -n "$detected" ] && [ "$detected" != "$role" ]; then
  echo "⚠ gh залогинен как @$login ($detected), а выбран $role (@$(scripts/team.sh login "$role")) — проверь, что это твой ноутбук"
fi
git config dt.role "$role"
git config user.name "$(scripts/team.sh name "$role")"
email=$(scripts/team.sh email "$role"); [ -n "$email" ] && git config user.email "$email"
echo "✔ ты: $(scripts/team.sh whoami)"
echo "✔ коммиты: $(git config user.name) <$(git config user.email)> — засчитываются аккаунту @$(scripts/team.sh login "$role")"
git config core.hooksPath .githooks
git config pull.rebase true
git config rebase.autoStash true
chmod +x .githooks/* scripts/*.sh 2>/dev/null || true
echo "✔ pre-commit: проверка секретов, сырых данных и зон включена; git pull всегда с rebase"
# the team channel lives in this repo (branch team-chat); a clone pointed elsewhere earlier switches back and rereads it
if [ "$(git config dt.chatRemote 2>/dev/null || echo origin)" != origin ]; then
  git config --unset dt.chatRemote; git remote remove teamchat 2>/dev/null || true; git config dt.chatSeen 0
fi
echo "✔ канал команды: scripts/msg.sh (ветка team-chat в этом репо)"
if [ -z "$login" ]; then echo "⚠ gh не залогинен: gh auth login (иначе push пойдёт через обычный git-логин, проверь git push --dry-run)"; fi

if [ ! -f .env.local ] && [ -f .env.example ]; then
  cp .env.example .env.local && echo "✔ создан .env.local (ключи пустые → советы агента по правилам, без LLM)"
fi
# Team keys never enter the repo: a file passed by hand (TEAM_ENV=path). Values are copied, never printed.
team_env="${TEAM_ENV:-}"
if [ -n "$team_env" ] && [ -f "$team_env" ]; then
  touch .env.local; chmod 600 .env.local
  added=0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    key="${line%%=*}"; val="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [ -n "$val" ] || continue
    grep -q "^$key=." .env.local && continue
    grep -v "^$key=" .env.local > .env.local.tmp; mv .env.local.tmp .env.local
    printf '%s\n' "$line" >> .env.local; added=$((added + 1))
  done < "$team_env"
  echo "✔ ключи команды: добавлено в .env.local — $added (из $team_env, значения не печатаю)"
fi

if [ "$(uname)" = Darwin ]; then
  prefix=$(brew --prefix 2>/dev/null || echo /opt/homebrew)
  [ -e "$prefix/opt/libomp/lib/libomp.dylib" ] || echo "⚠ нет libomp — LightGBM/XGBoost не загрузятся: brew install libomp"
fi
echo "… uv sync (Python 3.12 и библиотеки; первый раз 1–3 мин)"
uv sync -q || { echo "✖ uv sync не прошёл — смотри вывод выше"; exit 1; }
uv run python -c "
from app.models import available
bad = {k: v for k, v in available().items() if v != 'ok'}
print('✔ модели: все библиотеки загружаются' if not bad else '⚠ недоступны: ' + str(bad))
"
if [ "$web" = 1 ] && [ -f web/package.json ]; then (cd web && pnpm install --prefer-offline) && echo "✔ web: pnpm install"; fi
for t in claude codex; do
  command -v "$t" >/dev/null 2>&1 && echo "✔ $t $("$t" --version 2>/dev/null | head -1)"
done
echo "Codex: при первом запуске в этой папке выбери «Trust» — подхватится .codex/config.toml (сеть в песочнице)."
echo "Дальше: docs/PLAYBOOK.md — «Старт агента» и план на 24 часа. Проверка окружения: uv run python -m app.cli demo"
