# Отдельный хост для сдачи, R5

Это план развёртывания **текущего** MLX-движка на отдельном Apple Silicon Mac в дата-центре. Личные ноутбуки можно выключить. Аренда и публичный URL ещё не созданы; генератор конфигурации сам ничего не покупает и не запускает. Развёртывание на macOS нужно отдельно проверить на выбранном хосте.

## Конкретный вариант до покупки

Текущий ориентир бюджета Асылбека — **4500 ₸**, это ещё не разрешение на покупку. Кандидат: Scaleway M4-M, 32 GB RAM, €0.29/час без налогов, минимум 24 часа (€6.96). Проверено 17.10.2026 по [тарифам](https://www.scaleway.com/en/pricing/apple-silicon/) и [FAQ минимальной аренды](https://www.scaleway.com/en/docs/apple-silicon/faq/). Итог в тенге зависит от курса банка, временной авторизации карты, налогов/условий аккаунта; его проверяет человек перед оплатой. Уже предложенный M4-XL 64 GB на 48 часов (€23.52) в этот ориентир не укладывается.

**32 ГБ ещё не приняты по памяти/скорости.** Сначала Даниал проверяет real API без кэша: пять запросов, самый длинный ролик, два worker, пиковая память всего дерева процессов. Успех mock или cached-eval этого не доказывает. Если проверка не проходит, кандидат не считается готовым решением. До заказа также выяснить окно доступности у жюри: 24 часа должны покрывать развёртывание и проверку; автоудаление до её окончания неприемлемо.

Нужны одобрение бюджета человеком, аккаунт с картой и проверкой личности, доступная машина и DNS-имя команды. Срок проверки аккаунта и наличие машины не гарантированы. Выбирать почасовой **No commitment**, не месячный план. Остановка питания не прекращает оплату: после проверки жюри сервер удаляют в консоли, с учётом минимальных 24 часов. Источники: [создание Mac](https://www.scaleway.com/en/docs/apple-silicon/how-to/create-mac-mini/), [квоты и идентификация](https://www.scaleway.com/en/docs/organizations-and-projects/organization/organization-quotas/), [оплата](https://www.scaleway.com/en/docs/billing/faq/).

Сначала переносим только код и публичные веса, проверяем синтетическим видео. Обработка банковских видео — только в согласованном с командой контуре. Внешние AI API не используются: модель выполняется на самом сервере.

## 1. Подготовить арендованный Mac

Подключение по SSH — по [инструкции Scaleway](https://www.scaleway.com/en/docs/apple-silicon/quickstart/). Приватный ключ и пароли не помещать в Git, сообщения или URL. Проверить fingerprint сервера. В сетевых правилах разрешить 80/443 для жюри и SSH только с адресов команды; 8000 наружу не открывать.

После одобренной аренды на **удалённом Mac**, под обычным пользователем, установить инструменты (Homebrew при его отсутствии — по официальной инструкции):

```bash
brew install uv ffmpeg libomp caddy
# Клонировать командный репозиторий по SSH, затем перейти в его корень.
uv sync --frozen
uv run python -m service.cli init
chmod 600 .env.local
```

Загрузить закреплённые публичные веса командой лида: `uv run bash scripts/download_models.sh core` (она использует `hf` из окружения). Либо безопасно скопировать уже загруженные **веса**, не видео, в `../models`. В `.env.local` задать `VISION_ENGINE=real`, `MODELS_DIR` — абсолютный путь, `SERVICE_WORKERS=2`. Генератор также задаёт эти значения в launchd, поэтому при изменении числа процессов нужно менять соответствующий plist. API_KEY создан локально и не выводится.

```bash
uv run python -m service.cli doctor --engine real --require-ui
uv run ruff check . && uv run pytest -q
```

Doctor проверяет платформу, инструменты и наличие файлов весов; он не заменяет реальный прогон модели. Сохранить для отчёта commit: `git rev-parse HEAD`.

## 2. Подготовить HTTPS и автозапуск

Создать A-запись **своего** DNS-имени на публичный IPv4 сервера. Ниже заменить `video.your-domain.kz` настоящим именем. Убрать ошибочную AAAA-запись, если IPv6 сервера не настроен.

```bash
uv run python -m service.deploy.render video.your-domain.kz
plutil -lint runs/jobs/deploy/*.plist
caddy validate --config runs/jobs/deploy/Caddyfile --adapter caddyfile
```

Просмотреть три созданных файла. API работает от текущего обычного пользователя и слушает только localhost:8000; Caddy обслуживает 80/443 и управляет TLS. Ключ остаётся в `.env.local`, в plist его нет. Caddy admin API отключён, access-log запросов не включён. Рабочие логи — в `runs/jobs/deploy/` и `/var/log/fortebank-https*.log`, не публиковать их целиком.

После успешной проверки на новом сервере установить именно эти конфигурации:

```bash
sudo install -d -m 755 '/Library/Application Support/ForteBank'
sudo install -d -m 700 /var/db/fortebank-caddy
sudo install -o root -g wheel -m 644 runs/jobs/deploy/Caddyfile '/Library/Application Support/ForteBank/Caddyfile'
sudo install -o root -g wheel -m 644 runs/jobs/deploy/kz.fortebank.api.plist /Library/LaunchDaemons/kz.fortebank.api.plist
sudo install -o root -g wheel -m 644 runs/jobs/deploy/kz.fortebank.https.plist /Library/LaunchDaemons/kz.fortebank.https.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/kz.fortebank.api.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/kz.fortebank.https.plist
```

launchd запускает процессы при старте ОС и восстанавливает их после выхода. Для TLS DNS должен указывать на этот хост, порты 80/443 должны быть доступны. [Документация Caddy](https://caddyserver.com/docs/automatic-https).

## 3. Приёмка до выдачи URL

На клиентском ноутбуке в игнорируемом `.env.local` использовать ключ **удалённого** сервера, переданный лично. Не добавлять ключ параметром URL.

```bash
curl -fsS https://video.your-domain.kz/api/v1/health
uv run python -m service.cli check --synthetic --count 5 --strict --url https://video.your-domain.kz --report runs/jobs/remote-synthetic.json
```

Синтетика проверяет доставку, очередь, JSON и доступ, а не качество модели. Далее Даниал выполняет аналогичный прогон на согласованном реальном ролике, включая самый длинный (~145 с), пять параллельных задач; сохраняет отчёт, CPU/RAM и проверяет, что **все done и каждая <300 с от начала загрузки**. Один короткий ролик или health этого не доказывает. Если два процесса не укладываются по памяти/времени, конфигурация не принята: сообщить лиду, не скрывать провал сменой метрики.

Проверить UI с телефона вне Wi-Fi команды: загрузка, прогресс, результат, кадр, перемотка, правило. Затем отключиться от SSH: сайт должен продолжить работу. При пустой очереди перезапустить API и проверить сохранённый done-результат:

```bash
sudo launchctl kickstart -k system/kz.fortebank.api
curl -fsS https://video.your-domain.kz/api/v1/health
```

Только после этого bigali заполняет URL в `docs/SUBMISSION.md`, ключ передаётся жюри отдельно. Рабочий адрес нельзя объявлять по одной успешно сгенерированной конфигурации.

## 4. Обновление и завершение

В согласованное окно без processing-задач: остановить API, `git pull --ff-only`, `uv sync --frozen`, doctor и тесты, запустить. Не запускать второй Uvicorn поверх launchd.

```bash
sudo launchctl bootout system/kz.fortebank.api
# Обновить код и пройти проверки, затем:
sudo launchctl bootstrap system /Library/LaunchDaemons/kz.fortebank.api.plist
```

При отказе новой версии сохранить логи, вернуть ранее проверенный commit в отдельную чистую копию и заново подготовить plist; не использовать force-push и не сбрасывать чужие изменения. Старые конфигурации хранить до приёмки обновления.

После разрешения команды завершить сдачу: сохранить только разрешённые агрегаты/код, остановить обе службы (`launchctl bootout`), удалить арендованный Mac в консоли и проверить прекращение начислений. Не удалять сервер до окончания проверки жюри.
