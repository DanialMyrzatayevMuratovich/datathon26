# asylbek — задачи

<!-- Меняет только asylbek (или его агент). Напарники читают: scripts/sync.sh показывает «Сейчас» и «Нужно от других». -->

Роль: **API + бэкенд + деплой** (зона: `service/`, `tests/test_service*`, `README.md`, `scripts/run_service.sh`, `scripts/tunnel.sh`, `Dockerfile`). Контракт — `contract/schemas.py` и раздел «Контракт» в `docs/REQUIREMENTS.md` (меняет только danial; не хватает поля — пиши `[BLOCKED]`/просьбу). Распознавание вызывать только как `from vision import analyze` (сейчас заглушка, `VISION_ENGINE=mock`). Пуш — `scripts/checkpoint.sh` каждые ≤ 30 мин. Ноутбук Windows → скрипты в Git Bash.

## Сейчас
- R1–R3 — FastAPI-сервис `service/main.py` (`uv run uvicorn service.main:app --port 8000`): `POST /api/v1/analyze` (multipart `file`, .mov, лимит 200 МБ, сохранение в `runs/jobs/<job_id>/`) → `{"job_id"}`; `GET /api/v1/jobs/{id}` → `contract.schemas.to_public(...)` строго по ТЗ; `GET /api/v1/jobs/{id}/details` → `JobDetails`; `GET /api/v1/jobs/{id}/video`; `GET /api/v1/health`. Ключ `X-API-Key` из `.env.local` (`API_KEY=…`), без ключа/неверный → 401 (кроме health и статики UI). Обработка в фоне: пул на 2 воркера (ThreadPool/ProcessPool), статус `processing` → `done`/`failed`, прогресс из колбэка `analyze(path, progress)`; таймаут 4 мин 30 с → `failed`. Результат и статус сохранять на диск (`runs/jobs/<id>/result.json`), чтобы перезапуск сервиса не терял задачи. Статика UI: смонтировать папку `ui/` на `/`. · Готово, когда: curl из раздела 10 ТЗ на `data/raw/open/videos/X-1.mov` даёт job_id, через пару секунд `done` с кадром; без ключа — 401; тесты `tests/test_service_api.py` зелёные.

## Дальше
- R5 — публичный URL: `scripts/tunnel.sh` (cloudflared quick tunnel или named tunnel → `localhost:8000`), `scripts/run_service.sh` (uvicorn + туннель + `caffeinate` на macOS), проверить с телефона через мобильный интернет. Хост — Мак Даниала (M4, 16 ГБ); запасной — твой ноутбук с тем же кодом. Отписать URL в канал `[READY] URL …` (ключ — не в канал, лично).
- R4 — нагрузка: 5 видео разом (скрипт `scripts/load_test.sh`), время каждого < 5 мин, сервис не падает; лимит одновременных задач, очередь.
- Надёжность: проверка расширения и того, что это видео (`ffprobe`), понятные ошибки 400/413/404, логирование, очистка старых задач.
- R9 — README: как запустить локально (≤ 3 команды), URL, как получить ключ, примеры curl, архитектура (mermaid), раздел «Ограничения» — со слов danial.

## Нужно от других
- danial: реальный `vision.analyze` (вместо заглушки) — по мере готовности, контракт тот же.
- bigali: UI в `ui/` ходит на `/api/v1/jobs/{id}/details` и `/api/v1/jobs/{id}/video`.

## Готово
- Подключился к команде: setup, тесты, demo (16.10 23:52).
