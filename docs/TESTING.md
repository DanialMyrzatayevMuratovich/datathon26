# Журнал проверок

<!-- Только то, что реально выполнено. Заполняет владелец README (скилл /ship). -->

## Команды
```bash
uv run ruff check . && uv run pytest -q     # автотесты
scripts/smoke.sh                            # свежий clone, демо-данные, API
scripts/smoke.sh --with-data                # свежий clone, реальные данные, та же метрика
scripts/judge.sh                            # независимая проверка «как жюри»
```

## Что покрыто автотестами
- загрузка CSV (`;`, cp1251, десятичная запятая), Excel, Parquet; роли колонок; угадывание target и id
- план признаков: обучен на train, одинаково применяется к test и строкам API; новые категории и пропавшие колонки
- оркестратор: binary / regression / multiclass, временная валидация, упавшая модель не роняет прогон, ясная ошибка без target, флаг «подозрительно высоко», сохранённая модель предсказывает как сабмит
- тюнинг пишет params.toml только при улучшении; API: прогоны, файлы, защита пути, предсказания; CLI profile

## Журнал
| Время | Что проверили | Результат |
|---|---|---|
| 17.10 22:48 | Финальный full suite после video workflows/probe/provenance/arrival | Ruff pass; **73 passed, 2 skipped**, 133.81с |
| 17.10 22:40 | `service.smoke` | 5 разных synthetic MOV: 5/5 done, 0.890–2.000с, 8/8 auth; сервер остановлен; `runs/jobs/phase2-smoke.json` |
| 17.10 22:43 | Безопасный `scripts/judge.sh`, синтаксис shell, skill-ссылки | Создан пакет только docs/агрегаты, внешний AI не запускался; shell syntax pass, обе Claude-ссылки совпадают; autopilot quick_validate pass, ship YAML valid с сохранённым Claude extension |
| 17.10 22:30 | Full Ruff + pytest после fixes frames/cache/eval/thermal | Ruff pass; 68 passed, 2 skipped (POSIX на Windows), 154.88с |
| 17.10 22:30 | 20с 1080p синтетика, 40 кадров, отдельные процессы старого/нового decode | Peak RSS 371.71 → 167.42 MiB; SHA256 RGB одинаковый `a81eb4ecb6f9b6140fd650e600605687319c7e3c147ac022cb9b4fcf66f9c37e`; локальные артефакты `runs/jobs/runtime-check/` |
| 17.10 22:17 | Critics: 4 независимых read-only ревью | Исправлены память, invalid boxes, кэш, bounded thermal wait, coverage/joint metrics, provenance. Разметка X24 и реальные ML-регрессии требуют данных; automation и pitch продолжаются |

## Известные ограничения

Текущие 85% относятся к кэшированному development-прогону `2207-cebf` до этих исправлений. Новые правила не заявлены улучшенными без реального прогона. Входных видео и Apple MLX на Windows нет. Даниал сообщил 22:25 об одном worker 46–108с и swap двух worker на 16GB; агрегаты артефакта ещё запрошены. Пять одновременных real API задач, новый ракурс и публичный HTTPS остаются непроверенными.
