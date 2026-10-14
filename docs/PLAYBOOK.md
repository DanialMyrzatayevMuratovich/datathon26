# Плейбук: DATATHON'26 вдвоём с AI-агентами

Сб 17 – вс 18 октября 2026, Nazarbayev University (Астана), оффлайн. Открытие и выдача датасета — сб 11:00, старт 24 часов — 12:00, стоп — вс 12:00, питч — вс 13:00–17:00. Партнёр — ForteBank, призовой фонд 3 000 000 ₸. Расписание для скриптов — `scripts/timeline.sh`.

**Как устроено:** у каждого из двоих свой агент (Claude Code или Codex) в своей копии репо. Оба пушат в `main` через `scripts/checkpoint.sh`, видят друг друга через доску (`scripts/sync.sh`), строку «⏱» в каждом запросе и канал `scripts/msg.sh` (ветка `team-chat`). Зоны (`docs/ZONES.md`) не дают двум агентам править один файл. Задачу решает оркестратор `app.cli auto`: даёшь датасет — получаешь модели, ансамбль, объяснение, сабмит и отчёт. Агент крутит его в цикле `/autopilot`.

---

## 1. До хакатона (до пятницы вечера)

### Лид, один раз
1. Создать **приватный** репозиторий на GitHub и запушить эту папку:
   ```bash
   gh auth login                      # один раз на ноутбуке
   git remote add origin git@github.com:<owner>/<repo>.git   # или gh repo create <repo> --private --source . --push
   git push -u origin main
   ```
2. Добавить напарника в Collaborators (Settings → Collaborators).
3. Вписать обоих в команду и назначить лида:
   ```bash
   scripts/team.sh add <id1> <Имя1> <github-login-1>
   scripts/team.sh add <id2> <Имя2> <github-login-2>
   # в docs/ZONES.md: LEAD: <id лида>; в docs/TEAM.md — сильные стороны обоих
   scripts/setup.sh && scripts/checkpoint.sh "chore: team members"
   ```

### Оба, на своих ноутбуках
1. Поставить: git, [gh](https://cli.github.com), [uv](https://docs.astral.sh/uv/), Claude Code и/или Codex, VS Code (расширения Python, Jupyter, Marp for VS Code). macOS: `brew install libomp` (без него не грузятся LightGBM и XGBoost).
2. `gh auth login` → `git clone <repo>` → `cd <repo>` → `scripts/setup.sh` (определит тебя по GitHub, поставит Python 3.12 и библиотеки, включит хуки).
3. Проверка окружения: `uv run python -m app.cli demo` — синтетические банковские данные, полный цикл за ~30 с, отчёт в `runs/…/report.md`. Потом `uv run uvicorn app.main:app --port 8000` → http://localhost:8000.
4. Colab: `scripts/colab.sh` → открыть ссылку → выполнить ячейки на демо-данных (раздел 4). В Colab → Secrets добавить `GH_TOKEN` (fine-grained токен только на этот репо, Contents: Read and write).
5. Открыть агента в папке репо (`claude`), убедиться, что видны строка «⏱» и статус-строка, а `/kickoff`, `/autopilot` и другие скиллы есть в списке.

### Репетиция (желательно, 1–2 часа в четверг или пятницу)
Сдвиньте расписание на сейчас: создайте `scripts/timeline.local.sh` (в git не идёт) с `DT_START="<сегодня> 20:00"` и т. п. Вставьте в `docs/TASK.md` выдуманное ТЗ («предсказать дефолт клиента по данным data/demo»), скопируйте `data/demo/*.csv` в `data/raw/` и пройдите: `/kickoff` → `/autopilot` ×2 → `/insights` → `/pitch` → `/ship`. После — удалите `timeline.local.sh`, верните шаблоны: `git checkout -- docs task.toml`, очистите `data/raw/`.

### Пятница 19:00 — pre-datathon воркшоп (ауд. 7E.329): питчинг, тайм-менеджмент. Сходить.

### Что взять
Зарядки, удлинитель (пилот на 3+ розетки), мышь, наушники, переходник HDMI/USB-C, пауэрбанк, флешку (копия презентации), мобильный интернет (раздать с телефона, если Wi-Fi на 1200 человек ляжет), воду и еду, свитер на ночь, зубную щётку.

---

## 2. День X: кто что делает

| Время | Лид (по умолчанию ML) | Напарник (по умолчанию Insights & Pitch) |
|---|---|---|
| **сб 09:00–11:00** | регистрация, место у розетки, `scripts/sync.sh`, `scripts/autopush.sh` в отдельной вкладке терминала | то же |
| **11:00–12:00** открытие | записать ТЗ дословно в `docs/TASK.md` (фото слайдов → текст), спросить про формат сдачи, критерии и можно ли отдавать данные внешним сервисам | скачать данные в `data/raw/` на оба ноутбука (флешка/AirDrop, **не через git**), `uv run python -m app.cli profile data/raw/<файл>` |
| **12:00–12:40** | агенту: `/kickoff` (запустит бейзлайн в фоне, совет из 4 субагентов, контракт, план, зоны) | читать «как я понял» и отвечать на grill вместе с лидом; агенту: «Сделай auto-EDA: `uv run python -m app.cli eda` и выпиши 5 фактов» |
| **12:40–15:00** | `/autopilot`: бейзлайн → полный auto → первые признаки | `/insights`: первые 3 факта с графиками |
| **15:00** синк (5 мин голосом) | метрика, сабмит есть? | 3 факта есть? |
| **15:00–22:00** | `/autopilot` по кругу; агрегаты таблиц, `tune` в фоне, Colab для тяжёлого | `/insights` глубже: сегменты, бизнес-эффект; скелет `pitch/deck.md`; демо-страница (`ui-craft`) |
| **22:00 разворот** | `/critics` по своей зоне; решаем финальное направление | `/critics` по инсайтам; `/pitch` — первая версия |
| **22:00–02:00** | последние большие гипотезы | инсайты → слайды, речь |
| **02:00–05:00** | 😴 спит; перед сном: `/loop /autopilot` (агент сам ведёт эксперименты) + `tune` или Colab | дежурит: следит за агентом напарника (`scripts/msg.sh log`), полирует питч |
| **05:00–08:00** | дежурит: разбирает ночные прогоны, лучший → финальный кандидат | 😴 спит |
| **вс 08:00 ФРИЗ** | финальная модель и сабмит (`/ship` шаг 1) | — |
| **08:00–10:30** | `/ship`: README, smoke, судья | `/pitch`: графики, цифры сверены, PDF |
| **10:30–11:30** | репетиция питча ×2 с таймером, Q&A | то же |
| **11:30** | финальный `scripts/checkpoint.sh "final"` | PDF презентации на флешку и в Drive |
| **12:00** стоп · **13:00** питч | | |

### Старт агента (первое сообщение в начале смены)
Лиду после kickoff и напарнику — скопировать «Сейчас» из своего `docs/tasks/<id>.md`, например:
> Ты работаешь на <Имя>, поток ML. Прочитай AGENTS.md, task.toml, docs/REQUIREMENTS.md и мой docs/tasks/<id>.md. Запусти /autopilot и веди эксперименты до моей команды стоп. Каждые 30 минут — checkpoint; новый лучший скор — [SCORE] напарнику.

> Ты работаешь на <Имя>, поток Insights & Pitch. Прочитай AGENTS.md, docs/TASK.md, outputs/eda/report.md и последний runs/*/report.md. Запусти /insights: 5 инсайтов с цифрами и графиками в docs/INSIGHTS.md. Потом /pitch.

### Ночной режим
- ML-агент: `/loop /autopilot` — агент сам выбирает гипотезу, проверяет, пишет вывод и пушит. Перед сном проверь, что `scripts/autopush.sh` запущен и ноутбук **не уснёт** (macOS: `caffeinate -dims` в отдельной вкладке; питание от сети).
- Долгое и GPU — в Colab (раздел 4); вкладку Colab держать открытой.
- Утром первым делом: `scripts/sync.sh`, `uv run python -m app.cli runs`, хвост `docs/EXPERIMENTS.md`.

---

## 3. Шпаргалка команд

| Что | Команда |
|---|---|
| подтянуть работу напарника, доска, входящие | `scripts/sync.sh` |
| сохранить и запушить | `scripts/checkpoint.sh "what was done (R1)"` |
| написать напарнику | `scripts/msg.sh <id\|all> "[READY] …"` · входящие `scripts/msg.sh inbox` · история `scripts/msg.sh log` |
| страховочный автопуш | `scripts/autopush.sh` (отдельная вкладка) |
| роли колонок файла | `uv run python -m app.cli profile data/raw/train.csv` |
| auto-EDA | `uv run python -m app.cli eda` → `outputs/eda/report.md` |
| решить целиком | `uv run python -m app.cli auto --note "гипотеза"` (`--quick` — быстро, `--eda` — с EDA) |
| тюнинг | `uv run python -m app.cli tune --model lgbm --minutes 30` |
| лидерборд прогонов | `uv run python -m app.cli runs` |
| предсказать новый файл | `uv run python -m app.cli predict --data data/raw/new.csv` |
| демо для жюри | `uv run uvicorn app.main:app --port 8000` → http://localhost:8000 |
| Colab | `scripts/colab.sh` |
| проверка как у жюри | `scripts/smoke.sh --with-data` · `scripts/judge.sh` |
| тесты | `uv run ruff check . && uv run pytest -q` |

## 4. Ноутбук или Colab

| | Ноутбук | Google Colab |
|---|---|---|
| когда | итерации до ~20 мин, EDA, отладка, демо | долгие прогоны, тюнинг на часы, GPU (CatBoost/XGBoost на CUDA, нейросети), ночь без ноутбука |
| запуск | `uv run python -m app.cli …` | `notebooks/colab_train.ipynb` (ссылка — `scripts/colab.sh`) |
| данные | `data/raw/` | Google Drive `MyDrive/datathon26/raw/` |
| результаты | сразу в репо | ноутбук пушит `runs/`, сабмиты, `EXPERIMENTS.md`, `params.toml`; модели — в Drive |
| риски | ноутбук занят и греется | сессия рвётся (~90 мин без активности, лимит GPU) — держать вкладку открытой |

Подробно — `docs/COLAB.md`.

## 5. Если что-то сломалось

- **Конфликт при checkpoint** — коммит остался локально; следуй подсказке скрипта (своя зона — правь, чужая — бери версию напарника).
- **`Library not loaded: libomp.dylib`** — `brew install libomp`; до этого оркестратор сам пропустит LightGBM/XGBoost и пойдёт дальше по лестнице.
- **Wi-Fi лёг** — раздать интернет с телефона; без сети работает всё, кроме push, Colab и LLM. Коммиты копятся локально, `checkpoint.sh` запушит при появлении сети.
- **Агент застрял или ходит по кругу** — остановить (Esc), `scripts/sync.sh`, дать одну конкретную задачу из `docs/tasks/<id>.md`. Дважды одна ошибка → `scripts/new-skill.sh`.
- **Скор подозрительно высокий** — это почти всегда утечка: `/critics` и раздел «Предупреждения» в отчёте.
- **Сессия Colab оборвалась** — результаты завершённых прогонов уже в Drive/репо; перезапустить с ячейки «Запуск».
