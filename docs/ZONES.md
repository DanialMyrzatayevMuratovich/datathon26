# Команда и зоны владения

Машиночитаемый файл: его читают `scripts/team.sh`, `zones.sh` (pre-commit), `setup.sh`, `sync.sh` и clock-хук.
- `MEMBER: <id> <Имя> <GitHub-логин> <GitHub-id>` — кто есть кто. Добавляет лид один раз до старта: `scripts/team.sh add <id> <Имя> <github-login>` (числовой id подставится сам). По логину агент узнаёт, на кого работает; коммиты идут с noreply-адреса этого аккаунта.
- `LEAD` — лид-интегратор: ему принадлежит всё неназначенное (`task.toml`, `pyproject.toml`, `uv.lock`, конфиги, `docs/TEAM.md`).
- Строки зон `<id>: путь путь …` (префиксы через пробел) добавляет `/kickoff` после выдачи данных; дальше файл меняет только лид.

Правила:
- Менять и удалять **существующие** файлы можно только в своей зоне. Новые файлы создавать можно, но в своих каталогах.
- Свой `docs/tasks/<id>.md` каждый меняет сам. `ALL` — общие файлы, которые дописывают оба.
- Согласованная правка чужого файла: `ZONE_OK=1 scripts/checkpoint.sh "..."`.
- Пока строк зон нет, проверка выключена. Пример: `ml: app/custom_features.py app/features.py notebooks/ml_ params.toml`.

MEMBER: danial Danial DanialMyrzatayevMuratovich 179967465
MEMBER: asylbek Асылбек Asylbek2006 166900995
LEAD: danial
ALL: docs/PROGRESS.md docs/EXPERIMENTS.md outputs/leaderboard.csv
danial: vision/ contract/ data/ app/ task.toml docs/VIOLATIONS.md docs/LAWS.md docs/DECISIONS.md docs/REQUIREMENTS.md docs/TASK.md docs/VIDEO_REVIEW_NOTES.md
asylbek: service/ tests/test_service README.md scripts/run_service.sh scripts/tunnel.sh Dockerfile .dockerignore
bigali: ui/ docs/PITCH.md pitch/ docs/SUBMISSION.md
