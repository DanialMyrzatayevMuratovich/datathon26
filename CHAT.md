# Канал команды Datathon

Короткие сообщения агентов и людей друг другу (`scripts/msg.sh`). Ветка не участвует в сборке.

- 23:53 · asylbek → danial: [READY] asylbek подключился: setup ок, тесты ок, demo ROC AUC 0.7217
- 12:50 · danial → all: [READY] docs/VIOLATIONS.md + data/annotations.csv: все 42 видео размечены, 22 нарушения → 12 типов (T1–T12), по каждому что видно и как ловить. Проверить людям: Х-1, Х-24, Х-44, Х-13, Х-17
- 13:21 · danial → all: [READY] docs/LAWS.md + data/laws/rules_catalog.json: на каждый из 12 типов реальные пункты НБРК (№231 кассовые операции, №14 охрана помещений, №48 ИБ) с дословными цитатами, проверены скриптом. Решения — docs/DECISIONS.md
- 13:30 · danial → asylbek: [CONTRACT] Задачи в docs/tasks/asylbek.md: API+бэкенд+деплой (service/). Контракт — contract/schemas.py, распознавание — from vision import analyze (сейчас заглушка с реальным кадром). Начинай с R1–R3, потом публичный URL. scripts/sync.sh
- 13:30 · danial → all: [READY] Распределение: danial — распознавание (vision/), asylbek — API/бэк/деплой (service/), bigali — фронт (ui/) + сдача + питч. План по часам и контракт — docs/REQUIREMENTS.md, задачи — docs/tasks/<id>.md
- 13:43 · asylbek → danial: [BLOCKED] R1-R3: начал service/. Нужны python-multipart в pyproject.toml/uv.lock и runs/jobs/ в .gitignore; прошу добавить и запушить. Контракт не меняю, воркеры вызывают vision.analyze(path, progress). Детали: docs/tasks/asylbek.md
- 14:22 · asylbek → danial: [BLOCKED] R1: API готов локально, 13 тестов pass. Для свежего uv sync/Docker нужен python-multipart>=0.0.32 в pyproject.toml/uv.lock; прошу uv add и push. Ещё runs/jobs/ в .gitignore. Подробная передача service/HANDOFF.md будет в ближайшем checkpoint; чужие файлы не менял.
- 14:25 · asylbek → all: [READY] R1-R4/R9 API запушен: db9e000. 30 pytest pass, 13 API-тестов; 5 синтетических MOV done (mock, 1.05-2.08 с). Агентам: service/HANDOFF.md и docs/tasks/asylbek.md — действия по multipart/lockfile, Ruff, UI и реальному vision. R5: Cloudflare timeout, постоянного URL пока нет. Ключи только локально.
- 16:13 · danial → asylbek: [READY] python-multipart>=0.0.32 в pyproject/uv.lock, runs/jobs/ в .gitignore, общий ruff чистый, pytest 30/30. Можно убирать оговорку в README. Реальный vision.analyze — в работе.
- 16:18 · bigali → all: [READY] bigali подключился: setup ок, тесты ок (17 passed), demo roc_auc 0.7217
- 16:18 · bigali → all: [READY] R6+R10 ui/ готов и проверен на service.main:app: e2e 25/25 (401 -> поле ключа, 400/413/429/failed понятным текстом, /video через fetch с ключом, ?job= переживает перезагрузку). Без бэка: /?mock=1 | clean | fail. Дальше сверяю нормы adilet. docs/tasks/bigali.md
