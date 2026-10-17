# asylbek — задачи

<!-- Меняет только asylbek (или его агент); остальные читают. -->

Роль: **API + бэкенд + деплой**. Зона: `service/`, `tests/test_service*`, `README.md`, `scripts/run_service.sh`, `scripts/tunnel.sh`, `Dockerfile`, `.dockerignore`.

## Сейчас

- R1–R4/R9: повторная приёмка после `d2819ea`, дефекты API исправлены; готовлю checkpoint. Аудит всех потоков: [service/AUDIT.md](../../service/AUDIT.md).
- R5: отдельный Apple Silicon Mac с HTTPS/автозапуском подготовлен: [service/deploy/README.md](../../service/deploy/README.md). Хост ещё не арендован, URL не выдан; нужны бюджет и доступы человека.

## Дальше

- Завершить Docker/API-приёмку и полный pytest, отправить checkpoint.
- На выделенном хосте развернуть real, проверить с Даниалом 5 видео и ~145-секундный ролик, HTTPS/UI вне сети команды.
- Передать Бигали проверенный URL, ключ — лично человеку. После этой приёмки закрыть R5.

## Нужно от других

- **danial / R4/R7:** времена 5 параллельных запросов, длинного ролика, память двух процессов, финальные веса/commit и оценка. Его ответ 17:07 подтверждает проверку нашего mock и ожидание real.
- **bigali / R6:** same-origin вместо `?api=`, таймауты fetch/XHR, точный текст остановки ожидания, приёмка real. CSP уже блокирует внешний origin. Детали: [service/HANDOFF.md](../../service/HANDOFF.md).
- **Асылбек / R5:** одобрить конкретный тариф Mac, дать доступ к хосту/DNS. Ничего платного не создано.

## Готово

- R1–R3: MOV до 200 MiB, ffprobe, X-API-Key, публичный JSON, details/video/health, Range/HEAD.
- R4: 2 процесса, 8 мест, очередь+анализ 270 с, upload/idle deadlines, дисковые ошибки без потери потока, восстановление, TTL включая orphan-каталоги. POSIX watchdog при падении API; Windows taskkill при управляемой остановке.
- R9: README/handoff актуальны; старые запросы multipart/ignore/Ruff/UI сняты — это уже сделали danial/bigali.
- Синтетика: **5/5 done**, **1.844–3.750 с**, mock, 8 проверок 401, JPEG, UI/CSP, перезапуск, Range; локально `runs/jobs/acceptance-local.json`.
- `service.cli check --strict`, запрет redirect с ключом, preflight doctor. Windows/Linux real отклоняется до запуска.
- Git: 42 достижимых коммита проверены по шаблонам секретов, приватных путей/blob >50 MiB и конфликтов нет, fsck чистый. Push разрешён человеком, только checkpoint.
- Подключение к команде: setup, тесты, demo — 16.10 23:52.
