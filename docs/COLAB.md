# Обучение в Google Colab

Тот же код и тот же контракт (`task.toml`), что на ноутбуке, поэтому скоры сравнимы. Colab запускает `app.cli` в своём `.venv` (`uv sync --frozen`), а не в системном Python Colab: версии библиотек совпадают с `uv.lock`, и предустановленные пакеты Colab не мешают.

## Один раз
1. Репо на GitHub, свежий код запушен (`scripts/checkpoint.sh`).
2. Colab → значок ключа (Secrets) → `GH_TOKEN`: fine-grained токен GitHub только на этот репозиторий, права Contents: Read and write. Включить доступ к ноутбуку.
3. Google Drive: папка `MyDrive/datathon26/raw/` — сюда файлы организаторов (те же, что в `data/raw/`).

## Каждый прогон
1. `scripts/colab.sh` → ссылка на `notebooks/colab_train.ipynb` текущей ветки → открыть.
2. Runtime → Change runtime type → **T4 GPU**, если нужен GPU (CatBoost/XGBoost `--gpu`). LightGBM в pip-сборке работает на CPU.
3. Заполнить первую ячейку (`REPO`, `COMMAND`, `NOTE`) и выполнить все ячейки по порядку.
4. Результаты: ноутбук коммитит и пушит `runs/<run_id>/`, `outputs/submissions/`, `outputs/leaderboard.csv`, `docs/EXPERIMENTS.md`, `params.toml` (после `git pull --rebase`). Модели `.joblib` — в `MyDrive/datathon26/models/` (в git не идут).
5. На ноутбуке: `scripts/sync.sh` → `uv run python -m app.cli runs`.

## Что запускать в Colab
- полный `auto` на всём наборе: `python -m app.cli auto --gpu --budget-min 120 --note "colab full"`;
- тюнинг на час: `python -m app.cli tune --model catboost --minutes 60 --gpu`;
- нейросети/эмбеддинги текста — отдельным ноутбуком в `notebooks/`, результат как признаки в `data/processed/` (+ строка в `docs/EXPERIMENTS.md`).

## Ограничения
- Бесплатный Colab рвёт сессию после ~90 минут без активности и ограничивает GPU по времени — держите вкладку открытой, долгие прогоны дробите (`tune --minutes 45`).
- Данные организаторов — только в вашем Drive, не в публичных местах. Если организаторы запретили облака — Colab не использовать, только ноутбук.
- Агент может вести ноутбук сам, если у него подключён инструмент Colab (MCP); иначе он даёт ссылку и инструкции.
