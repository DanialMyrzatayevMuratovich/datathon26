# asylbek — задачи

<!-- Меняет только asylbek (или его агент). Напарник читает: scripts/sync.sh показывает «Сейчас» и «Нужно от других». -->

Роль: <!-- поток после /kickoff --> · человек: Асылбек

## Сейчас

- Локальная подготовка завершена. Ожидаю /kickoff Даниала 17 октября в 12:00 и назначение задачи; до выдачи ТЗ код под задачу не пишу.

## Дальше

- 17 октября в 12:00, после /kickoff Даниала: scripts/sync.sh, прочитать назначенную задачу и свою зону. До выдачи ТЗ код под задачу не писать.

## Нужно от других

- Даниал: заполнить LEAD: danial в docs/ZONES.md и свой профиль в docs/TEAM.md до /kickoff.

## Готово

- 16 октября: прочитаны AGENTS.md, CLAUDE.md и docs/PLAYBOOK.md; git pull и git push --dry-run успешны.
- GitHub: Asylbek2006, права WRITE на командный репозиторий; scripts/team.sh add asylbek Асылбек Asylbek2006 выполнен. scripts/setup.sh asylbek завершился сообщением «✔ модели: все библиотеки загружаются»; Git-личность, хуки, канал команды и локальное окружение настроены.
- Установлены GitHub CLI и uv; Python 3.12 и зависимости из uv.lock. Все модели доступны: dummy, linear, hgb, lgbm, catboost, xgb. Claude Code пропущен по выбору Асылбека.
- uv run ruff check . — успешно; uv run pytest -q — 17 passed.
- Синтетическое demo: run_id 20261016T2324-user-3d92, ensemble(lgbm+hgb), OOF ROC AUC 0.721668403956583 (0.7217). Отчёт был в runs/20261016T2324-user-3d92/report.md; артефакты демо удалены, docs/EXPERIMENTS.md и outputs/leaderboard.csv восстановлены. Это проверка окружения, не результат на данных организаторов.
- API /api/health вернул ok=true; сервер остановлен, порт 8000 свободен.
- Включён Developer Mode и локальный core.symlinks=true; ссылки .claude/skills восстановлены, kickoff/SKILL.md читается.
- scripts/colab.sh выдаёт ссылку на notebooks/colab_train.ipynb.
