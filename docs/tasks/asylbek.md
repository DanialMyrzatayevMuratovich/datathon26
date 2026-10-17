# asylbek — задачи

<!-- Меняет только asylbek (или его агент); остальные читают. -->

Роль: **API + бэкенд + деплой**. Зона: `service/`, `tests/test_service*`, `README.md`, `scripts/run_service.sh`, `scripts/tunnel.sh`, `Dockerfile`, `.dockerignore`.

## Сейчас

- 22:17: по прямому разрешению Асылбека работаю над всем проектом, включая ML. Pull до `bce14cc` выполнен без конфликтов. Critics: валидация, код, воспроизводимость и утверждения; исправляю память кадров, кэш и чистый eval, ожидание thermal-monitor. Изменения правил требуют реального прогона, 98% не заявлены.
- R5 остаётся открытым: конфигурация отдельного Apple Silicon Mac с HTTPS/автозапуском подготовлена, но хост ещё не арендован и URL не выдан. Ориентир человека — 4500 ₸; кандидат M4 32 GB / 24 ч требует измерения RAM/времени, срока проверки жюри и одобрения оплаты. [План деплоя](../../service/deploy/README.md).
- Новый аудит: [service/AUDIT.md](../../service/AUDIT.md). Чистый real-прогон, API/RAM и внешний URL ещё не подтверждены; по сообщению Даниала в 21:30 чистая оценка запущена.

## Дальше

- После отчёта Даниала о памяти/скорости и одобрения/доступа человека развернуть real на выбранном хосте.
- Проверить с Даниалом 5 реальных запросов, включая ~145-секундный ролик, HTTPS/UI вне сети команды, автозапуск и сохранение результата.
- Передать Бигали проверенный URL, ключ — лично человеку. После этой приёмки закрыть R5.

## Нужно от других

- **danial / R4/R7:** чистый real API-прогон без DT_CACHE: 5 запросов, длинный ролик, peak RAM; финальные веса/commit, воспроизводимость и другой ракурс. `2123-fb5e`: accuracy 80%, recall 14/21, FP 1/19; joint type+time ±2с 8/22. Вернуть потерянный X-18 и проверить QUALITY_PLAN. `--expect-engine real` обязателен для нашей реальной приёмки.
- **danial / документы и скиллы:** METRICS/ARCHITECTURE уже обновлены; остаются согласование REQUIREMENTS и адаптация табличных autopilot/ship/smoke к видео до автоматического цикла.
- **bigali / R6:** code-review замечания закрыты; остаются real 3 видео/телефон и финальные URL/метрики SUBMISSION. Детали: [service/HANDOFF.md](../../service/HANDOFF.md).
- **Команда / люди:** срок проверки жюри, одобрение конкретного тарифа/аккаунт/хост/DNS, свои видео и спорная разметка, телефонная проверка, выступающий и сдача. Ничего платного не создано.

## Готово

- 22:30 critics: 4 read-only ревью; fixes `vision/frames`, detector boxes, cache identity, eval coverage/joint/provenance, stale thermal wait. Ruff pass; **68 passed / 2 skipped**. Синтетический decode 1080p: peak RSS 371.71 → 167.42 MiB, все пиксели совпадают. Метрика ML после изменений не измерена.
- Pull `bce14cc`, аудит `2207-cebf`: 85%, recall 15/21, FP0/19, joint9/22. Кэшированный подбор правил. Конфликт типа X24 v4/v5 отправлен danial; метка не менялась. Человек предложил локальный ZIP, SSH к Mac нет.
- Новый full suite: Ruff pass, **59 passed / 2 skipped**. Пять разных синтетических MOV через CLI: **5/5**, **0.812–1.953 с**, 8 проверок 401; mock отвергнут при `--expect-engine real`. Отчёт `runs/jobs/acceptance-a2b74fc/mixed.json`.
- `service.quality` + 10 тестов: one-to-one type/time, coverage, дубликаты, валидация. Два отчёта без кадров/описаний — `service/reports/`; воспроизводимый план для агента Даниала — `service/QUALITY_PLAN.md`.
- CLI разных видео и защиты engine + 6 тестов. API-контракт не менялся; отчёт содержит вход/тип/таймкод для сопоставления с прямым ML-прогоном.
- Принят body-timeout `510882e` (4/4 локальных проверок), SUBMISSION `8e5e7c0` абзац «Сервис» получил REVIEW-OK.
- R1–R3: MOV до 200 MiB, ffprobe, X-API-Key, публичный JSON, details/video/health, Range/HEAD.
- R4: 2 процесса, 8 мест, очередь+анализ 270 с, upload/idle deadlines, дисковые ошибки без потери потока, восстановление, TTL включая orphan-каталоги. POSIX watchdog при падении API; Windows taskkill при управляемой остановке.
- R9: README/handoff и аудит обновлены после `4dbab5a`; старые запросы multipart/ignore/Ruff/UI сняты — это уже сделали danial/bigali.
- Новая приёмка: **5/5 done**, **1.188–2.406 с**, mock, 8 проверок 401, JPEG, UI/CSP; `runs/jobs/audit-4dbab5a/acceptance.json`. Перезапуск, Range, Docker/mock и два POSIX-watchdog сценария проверены на предыдущем `a84b77d`.
- Исправлена гонка теста disk-full из REVIEW Даниала; API-worker отключает developer DT_CACHE, добавлен тест. Два целевых теста: 8 повторов подряд успешно. Итог: **Ruff pass, 43 passed / 2 skipped** (POSIX на Windows); checkpoint — в `docs/PROGRESS.md`.
- `service.cli check --strict`, запрет redirect с ключом, preflight doctor. Windows/Linux real отклоняется до запуска.
- Git: новые `191a428`/`4dbab5a` получены без конфликтов, fsck чистый; проверены все 48 достижимых коммитов: известных шаблонов секретов/приватных путей/blob >50 MiB не найдено. Секреты и зоны также проверяются checkpoint. Push разрешён человеком, только checkpoint.
- Подключение к команде: setup, тесты, demo — 16.10 23:52.
