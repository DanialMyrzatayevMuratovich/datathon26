#!/usr/bin/env bash
# UserPromptSubmit hook: one line of context per prompt — where we are in the 24 hours, your role and zone, push
# freshness, what the teammate is doing — so agents size the work to the clock and stay inside their zone.
# cut -c counts bytes in the C locale and would split Cyrillic mid-character
export LC_ALL="${LC_ALL:-en_US.UTF-8}"
root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
# shellcheck disable=SC1091
. "$root/scripts/timeline.sh"
now=$(date +%s)

head="⏱ $(date '+%d.%m %H:%M')"
role=$(git config dt.role 2>/dev/null || true)
team_line=""
if [ -n "$role" ] && [ -x "$root/scripts/team.sh" ]; then
  T="$root/scripts/team.sh"
  zone=$( [ -f "$root/docs/ZONES.md" ] && grep -E "^$role:" "$root/docs/ZONES.md" | head -1 | cut -d: -f2- )
  head="$head · ты работаешь на $("$T" name "$role") (@$("$T" login "$role"))${zone:+, зона:$zone}"
  for id in $("$T" ids); do
    [ "$id" = "$role" ] && continue
    cur=$(awk '$0 == "## Сейчас" {on=1; next} /^## /{on=0} on && NF' "$root/docs/tasks/$id.md" 2>/dev/null | head -1 | cut -c1-70)
    team_line="$team_line${team_line:+; }$("$T" name "$id"): ${cur:-—}"
  done
  [ -n "$team_line" ] && team_line=" · напарник: $team_line"
fi
push=""
if t=$(git log -1 --format=%ct '@{u}' 2>/dev/null); then
  push=" · последний push команды $(( (now - t) / 60 ))м назад"
  n=$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo 0)
  [ "$n" != "0" ] && push="$push · у тебя не запушено: $n"
fi
push="$push$team_line"
# unread team-channel messages ride along with the prompt, so the agent sees them without being asked
chat=""; [ -n "$role" ] && [ -x "$root/scripts/msg.sh" ] && chat=$("$root/scripts/msg.sh" hook 2>/dev/null)

start=$(dt_epoch "$DT_START"); mid=$(dt_epoch "$DT_MID"); freeze=$(dt_epoch "$DT_FREEZE")
final=$(dt_epoch "$DT_FINAL"); end=$(dt_epoch "$DT_END"); pitch=$(dt_epoch "$DT_PITCH")
if [ "$now" -lt $((start - 86400 * 7)) ] || [ "$now" -gt $((pitch + 6 * 3600)) ]; then
  echo "$head$push"
elif [ "$now" -lt "$start" ]; then
  echo "$head · до старта ($DT_START) $(dt_left "$start") — готовим окружение, без кода под задачу$push"
elif [ "$now" -lt "$freeze" ]; then
  hour=$(( (now - start) / 3600 + 1 ))
  stage="итерации"; [ "$now" -lt $((start + 3600)) ] && stage="kickoff и бейзлайн"
  [ "$now" -ge "$mid" ] && stage="после разворота: углубляем выбранное, без новых направлений"
  echo "$head · час $hour из 24 ($stage) · до фриза моделей (${DT_FREEZE#* }) $(dt_left "$freeze") · до финального push (${DT_FINAL#* }) $(dt_left "$final")$push"
elif [ "$now" -lt "$final" ]; then
  echo "$head · ФРИЗ МОДЕЛЕЙ: только фиксы, финальный сабмит, питч, README · до финального push (${DT_FINAL#* }) $(dt_left "$final")$push"
elif [ "$now" -lt "$end" ]; then
  echo "$head · ФИНАЛ: до ${DT_END#* } $(dt_left "$end") — только push и сдача$push"
elif [ "$now" -lt "$pitch" ]; then
  echo "$head · код сдан; питч в ${DT_PITCH#* } через $(dt_left "$pitch") — репетиция, ответы на вопросы$push"
else
  echo "$head · питч идёт: код не меняем$push"
fi
[ -n "$chat" ] && echo "$chat"
exit 0
