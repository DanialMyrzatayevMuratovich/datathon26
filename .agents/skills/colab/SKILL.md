---
name: colab
description: Вынести долгое или GPU-обучение в Google Colab — запушить код, дать ссылку на notebooks/colab_train.ipynb, объяснить, как подложить данные и вернуть результаты в репо. Использовать, когда прогон на ноутбуке дольше ~20 минут, нужен GPU (CatBoost/XGBoost на CUDA, нейросети) или ноутбук нужен для другой работы.
---

# Colab: тяжёлое — в облако, ноутбук — для работы

Два режима одного и того же кода: **ноутбук** (быстрые итерации, `uv run python -m app.cli …`) и **Colab** (долгие прогоны, GPU, ночь). Скор сравним, потому что контракт один — `task.toml`, а Colab запускает тот же `app.cli` в своём `.venv` (`uv sync --frozen`).

1. Решение: оценка времени на выборке (`--sample 50000`) × масштаб; > 20 мин или нужен GPU → Colab. Иначе — фоновый прогон на ноутбуке.
2. Код должен быть на GitHub: `scripts/checkpoint.sh "..."`, затем `scripts/colab.sh` → ссылка на `notebooks/colab_train.ipynb` нужной ветки. Дай её человеку.
3. Человек в Colab (один раз за сессию): Runtime → Change runtime type → GPU (T4) при необходимости; Secrets → `GH_TOKEN` для приватного репо (fine-grained токен только на этот репо, Contents: Read and write); данные — в Google Drive `MyDrive/datathon26/raw/` (организаторские файлы в git не кладём). Если у агента подключён инструмент Colab (MCP) — можно открыть и вести ноутбук самому, иначе — инструкции человеку.
4. В ноутбуке ячейки идут по порядку: clone → `uv sync` → данные из Drive → проверка GPU → `app.cli auto --gpu --note "colab: …"` или `tune` → результаты обратно.
5. Возврат результатов: ноутбук коммитит `runs/<run_id>/`, `outputs/submissions/`, `docs/EXPERIMENTS.md`, `params.toml` и пушит (`git pull --rebase` перед push). Модели `.joblib` в git не идут — копируются в Drive `MyDrive/datathon26/models/`; на ноутбук — скачать, если нужны для `predict`/демо.
6. На ноутбуке: `scripts/sync.sh` → `uv run python -m app.cli runs`. Сессия Colab оборвалась → результаты последнего завершённого прогона уже в Drive/репо; перезапусти с той же ячейки.
