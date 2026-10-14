#!/usr/bin/env bash
# Safety net for 24 hours: two laptops, long trainings and a night shift — nobody should lose work or leave the
# teammate on a stale base. Start once in a spare terminal tab and forget about it (costs zero tokens).
# Every :25 and :55 — if you have unpushed work and your own last push is older than 25 min, runs
# scripts/checkpoint.sh "wip: auto checkpoint". Reminds of the model freeze and the final push. Stops at the end.
# Usage: scripts/autopush.sh [--once]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
# shellcheck disable=SC1091
. scripts/timeline.sh
# an empty email would make --author match everyone, so the teammate's pushes would count as yours
[ -n "$(git config user.email)" ] || { echo "✖ user.email пуст — сначала scripts/setup.sh"; exit 1; }

notify() {
  osascript -e "display notification \"$1\" with title \"Datathon\" sound name \"Glass\"" >/dev/null 2>&1 ||
    powershell.exe -NoProfile -Command "[console]::beep(880,400)" >/dev/null 2>&1 || true  # Windows (Git Bash): sound only
  echo "$(date +%H:%M) $1"
}
my_push_age() {
  local t
  t=$(git log -1 --format=%ct --author="$(git config user.email)" '@{u}' 2>/dev/null)
  if [ -n "$t" ]; then echo $(( ($(date +%s) - t) / 60 )); else echo 999; fi
}
has_work() {
  [ -n "$(git status --porcelain)" ] || [ "$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)" != "0" ]
}
min_age="${AUTOPUSH_MIN_AGE:-25}"
tick() {
  local age; age=$(my_push_age)
  if has_work && [ "$age" -ge "$min_age" ]; then
    notify "авто-checkpoint: твой последний push ${age}м назад"
    scripts/checkpoint.sh "wip: auto checkpoint $(date +%H:%M)" || notify "авто-checkpoint НЕ прошёл — смотри терминал"
  else
    echo "$(date +%H:%M) ок: твой push ${age}м назад, срочного нет"
  fi
}

if [ "${1:-}" = "--once" ]; then tick; exit 0; fi
echo "autopush: авто-checkpoint в :25 и :55, напоминания о фризе ($DT_FREEZE) и финальном push ($DT_FINAL), стоп в $DT_END. Выход — Ctrl-C."
last=""
while :; do
  if dt_past "$DT_END"; then echo "$DT_END — autopush остановлен"; exit 0; fi
  hm=$(date +%H:%M)
  if [ "$hm" != "$last" ]; then
    last=$hm
    case "$hm" in
      *:25|*:55) dt_past "$DT_START" && tick ;;
    esac
    now_full=$(date '+%Y-%m-%d %H:%M')
    if [ "$now_full" = "$DT_FREEZE" ]; then
      notify "ФРИЗ МОДЕЛЕЙ: финальный сабмит, дальше только фиксы, питч и README"
    fi
    if [ "$now_full" = "$DT_FINAL" ]; then
      notify "ФИНАЛЬНЫЙ push сейчас: scripts/checkpoint.sh \"final\""
    fi
  fi
  sleep 20
done
