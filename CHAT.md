# Канал команды Datathon

Короткие сообщения агентов и людей друг другу (`scripts/msg.sh`). Ветка не участвует в сборке.

- 23:53 · asylbek → danial: [READY] asylbek подключился: setup ок, тесты ок, demo ROC AUC 0.7217
- 12:50 · danial → all: [READY] docs/VIOLATIONS.md + data/annotations.csv: все 42 видео размечены, 22 нарушения → 12 типов (T1–T12), по каждому что видно и как ловить. Проверить людям: Х-1, Х-24, Х-44, Х-13, Х-17
- 13:21 · danial → all: [READY] docs/LAWS.md + data/laws/rules_catalog.json: на каждый из 12 типов реальные пункты НБРК (№231 кассовые операции, №14 охрана помещений, №48 ИБ) с дословными цитатами, проверены скриптом. Решения — docs/DECISIONS.md
- 13:30 · danial → asylbek: [CONTRACT] Задачи в docs/tasks/asylbek.md: API+бэкенд+деплой (service/). Контракт — contract/schemas.py, распознавание — from vision import analyze (сейчас заглушка с реальным кадром). Начинай с R1–R3, потом публичный URL. scripts/sync.sh
- 13:30 · danial → all: [READY] Распределение: danial — распознавание (vision/), asylbek — API/бэк/деплой (service/), bigali — фронт (ui/) + сдача + питч. План по часам и контракт — docs/REQUIREMENTS.md, задачи — docs/tasks/<id>.md
