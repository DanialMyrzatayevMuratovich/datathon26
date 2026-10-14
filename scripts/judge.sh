#!/usr/bin/env bash
# Independent "jury" pass on a fresh clone: reads README literally, reproduces, scores against the rubric and lists
# claims the repo does not back. Prefers Codex (a different model family, its own quota), falls back to Claude Code.
# 3–8 min; fine in the background. Usage: scripts/judge.sh
set -uo pipefail
root=$(git rev-parse --show-toplevel) || exit 1
url=$(git -C "$root" remote get-url origin 2>/dev/null || echo "$root")
work=$(mktemp -d "${TMPDIR:-/tmp}/dt-judge.XXXXXX")
git clone -q --depth 1 "$url" "$work/repo" || { echo "✖ clone"; exit 1; }
[ -d "$root/data/raw" ] && mkdir -p "$work/repo/data/raw" && cp -R "$root/data/raw/." "$work/repo/data/raw/"

prompt='Ты — член жюри дататона (кейсодатель — банк) и видишь этот репозиторий впервые.
1. Прочитай README.md, затем docs/TASK.md, docs/REQUIREMENTS.md, docs/INSIGHTS.md и docs/PITCH.md (если есть).
2. Воспроизведи основной результат СТРОГО по README, ничего не додумывая. Для каждого шага: команда → ок/ошибка (одна строка).
3. Сверь цифры: метрика в README/питче против runs/*/metrics.json и того, что получилось у тебя.
4. Оцени по критериям из docs/REQUIREMENTS.md («Карта баллов»); если их нет — качество модели и валидации 25,
   инсайты и их обоснованность данными 25, бизнес-ценность для банка 20, подача и визуализация 15, воспроизводимость 15.
   По каждому: балл и одно предложение почему.
5. Перечисли утверждения (README, инсайты, питч), которые не подтверждаются данными, кодом или запуском.
6. Задай 5 самых неудобных вопросов жюри (утечка, валидация, применимость, стоимость ошибок, этика данных).
7. Топ-5 исправлений по убыванию влияния на оценку: файл → что именно поменять.
Не меняй файлы репозитория (кроме артефактов запуска). Ответ — на русском, кратко, markdown.'

if command -v codex >/dev/null 2>&1; then
  model="${JUDGE_MODEL:-}"
  extra=()
  for d in "$HOME/.cache/uv"; do [ -d "$d" ] && extra+=(--add-dir "$d"); done
  echo "→ Codex проверяет свежий clone: $work/repo (3–8 мин)…"
  codex exec -C "$work/repo" ${model:+-m "$model"} -s workspace-write \
    -c sandbox_workspace_write.network_access=true -c model_reasoning_effort=medium \
    ${extra[@]+"${extra[@]}"} --ephemeral -o "$work/report.md" "$prompt" >"$work/agent.log" 2>&1
elif command -v claude >/dev/null 2>&1; then
  echo "→ Claude Code проверяет свежий clone: $work/repo (3–8 мин)…"
  (cd "$work/repo" && claude -p "$prompt" \
    --allowedTools "Read" "Grep" "Glob" "Bash(uv *)" "Bash(ls *)" "Bash(cat *)" "Bash(head *)" "Bash(git log*)" \
    >"$work/report.md" 2>"$work/agent.log")
else
  echo "✖ нет ни codex, ни claude CLI"; exit 1
fi
status=$?
if [ -s "$work/report.md" ]; then
  cat "$work/report.md"; echo; echo "(отчёт: $work/report.md)"
else
  echo "✖ судья не дал отчёт (код $status). Лог: $work/agent.log"; exit 1
fi
