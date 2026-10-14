# Заранее подготовленные компоненты

До начала соревновательной части (17.10.2026, 12:00) команда подготовила только инструменты разработки и универсальный каркас без логики задачи: данные и условия кейса были неизвестны до открытия. Всё, что относится к задаче (контракт `task.toml`, признаки в `app/custom_features.py`, параметры, эксперименты, инсайты, презентация), сделано в ходе соревнования — см. историю коммитов, `docs/PROGRESS.md` и `docs/EXPERIMENTS.md`.

| Что | Где | Зачем |
|---|---|---|
| Инструкции для AI-агентов | `AGENTS.md`, `CLAUDE.md`, `.claude/`, `.agents/skills/`, `.codex/` | единые правила и сценарии работы Claude Code и Codex для двух человек |
| Скрипты командной работы | `scripts/`, `.githooks/` | синхронизация, коммиты с проверкой секретов, сырых данных и зон, канал сообщений агентов, расписание, smoke-тест, независимая проверка |
| Универсальный AutoML-каркас | `app/` | загрузка табличных файлов, профиль колонок, авто-EDA, план признаков, лестница моделей (sklearn, LightGBM, CatBoost, XGBoost), схемы валидации и метрики, ансамбль, объяснение, сабмит, журнал решений агента, Optuna, API для демо |
| Тесты каркаса | `tests/` | на синтетических данных (`app/demo.py`), без данных кейса |
| Ноутбуки | `notebooks/` | шаблон EDA и запуск в Google Colab |
| Шаблоны документации и питча | `README.md`, `docs/`, `pitch/deck.md` | структура README, требований, инсайтов, презентации |
| Справка по домену «Финансы» | `docs/DOMAIN-FINANCE.md` | регуляторика, открытые данные, метрики, паттерны — опора для разбора ТЗ, не решения |
| Каркас веб-демо (по желанию) | `web/` | Next.js + Tailwind + shadcn/ui: пустой шаблон |

## Сторонние компоненты
- Скилл `ui-craft` — объединение и адаптация скиллов `frontend-design` (Anthropic, Apache-2.0) и `emil-design-eng` (Emil Kowalski, MIT); лицензии — в `.agents/skills/ui-craft/`.
- Скилл `grilling` — Matt Pocock, github.com/mattpocock/skills, MIT (`.agents/skills/grilling/LICENSE`).
- Командные скрипты, правила агентов и скиллы kickoff/checkpoint/critics/ship адаптированы из хакатонного шаблона команды CtrlX (HackAlem AI 2026).
- Python: pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib, seaborn — BSD; LightGBM, XGBoost, pyarrow, CatBoost — MIT/Apache-2.0; Optuna, FastAPI, pydantic, uvicorn, polars, duckdb, shap — MIT/BSD; plotly — MIT; openai-python — Apache-2.0.

## AI-инструменты разработки
Claude Code (Anthropic), OpenAI Codex.
