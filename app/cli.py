"""Command line: `uv run python -m app.cli <command>` — the same entry point for people, agents and Colab.

  profile <file>   column roles, types, missing values (instead of reading rows by eye)
  eda              auto-EDA report with charts -> outputs/eda/report.md
  auto             autonomous solve: models, ensemble, explanation, submission, report
  tune             Optuna for one model on the same folds -> params.toml
  predict          predictions of a saved model for a new file
  runs             local leaderboard of all runs
  demo             synthetic bank data in data/demo + a quick auto run (rehearsal)
"""

import argparse
import shlex
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from app.config import OUTPUTS, ROOT, load_spec, resolve


def _spec_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("контракт задачи (по умолчанию из task.toml)")
    for name in ("train", "test", "sample-submission", "target", "id", "task", "metric", "cv",
                 "group", "time", "drop", "models", "submission-target"):
        g.add_argument(f"--{name}", default=None)
    g.add_argument("--folds", type=int, default=None)
    g.add_argument("--budget-min", type=float, default=None)
    g.add_argument("--sample", type=int, default=None)
    g.add_argument("--threshold", type=float, default=None)
    g.add_argument("--gpu", action="store_true", default=None)
    g.add_argument("--shap", action="store_true", default=None)
    g.add_argument("--labels", dest="predict_proba", action="store_false", default=None,
                   help="binary: сабмитить 0/1, а не вероятности")


def _spec(a: argparse.Namespace):
    keys = ("train", "test", "sample_submission", "target", "id", "task", "metric", "cv", "group",
            "time", "drop", "models", "submission_target", "folds", "budget_min", "sample",
            "threshold", "gpu", "shap", "predict_proba")
    return load_spec(**{k: getattr(a, k, None) for k in keys})


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="app.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("profile", help="роли колонок файла")
    s.add_argument("path")
    s.add_argument("--target", default="")
    s.add_argument("--nrows", type=int, default=None)

    s = sub.add_parser("eda", help="авто-EDA → outputs/eda/report.md")
    _spec_args(s)
    s.add_argument("--out", default=str(OUTPUTS / "eda"))

    s = sub.add_parser("auto", help="автономное решение: модели → ансамбль → сабмит → отчёт")
    _spec_args(s)
    s.add_argument("--note", default="", help="что проверяет этот прогон (в журнал экспериментов)")
    s.add_argument("--quick", action="store_true", help="3 фолда, ≤100k строк, dummy/hgb/lgbm")
    s.add_argument("--eda", action="store_true", help="сначала auto-EDA")

    s = sub.add_parser("tune", help="Optuna для одной модели → params.toml")
    _spec_args(s)
    s.add_argument("--model", required=True, choices=["lgbm", "catboost", "xgb", "hgb"])
    s.add_argument("--trials", type=int, default=50)
    s.add_argument("--minutes", type=float, default=30)

    s = sub.add_parser("predict", help="предсказания сохранённой модели")
    s.add_argument("--data", required=True)
    s.add_argument("--model", default=None, help="run_id или путь .joblib; по умолчанию models/best.json")
    s.add_argument("--out", default=None)

    s = sub.add_parser("runs", help="локальный лидерборд прогонов")
    s.add_argument("-n", type=int, default=15)

    s = sub.add_parser("demo", help="синтетические банковские данные + быстрый прогон")
    s.add_argument("--task", choices=["binary", "regression", "multiclass"], default="binary")
    s.add_argument("--rows", type=int, default=6000)
    s.add_argument("--no-run", action="store_true", help="только сгенерировать данные")
    return p


def _load_frames(spec):
    from app.agent.orchestrator import TEST_PATTERNS, TRAIN_PATTERNS
    from app.config import find_raw
    from app.data import guess_target, load_table

    train_path = resolve(spec.train) if spec.train else find_raw(TRAIN_PATTERNS)
    if not train_path:
        raise SystemExit("✖ нет train: положи файл в data/raw/ или передай --train")
    test_path = resolve(spec.test) if spec.test else find_raw(TEST_PATTERNS)
    train = load_table(train_path)
    test = load_table(test_path) if test_path and Path(test_path).exists() else None
    return train, test, spec.target or guess_target(train, test) or ""


def main(argv: list[str]) -> int:
    a = _parser().parse_args(argv)
    command = "uv run python -m app.cli " + " ".join(shlex.quote(x) for x in argv)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 40)

    if a.cmd == "profile":
        from app.data import load_table, profile_frame

        df = load_table(a.path, nrows=a.nrows)
        print(f"{a.path}: {len(df):,} строк × {df.shape[1]} колонок")
        print(profile_frame(df, a.target).to_string(index=False))
        return 0

    if a.cmd == "eda":
        from app.eda import run_eda

        spec = _spec(a)
        train, test, target = _load_frames(spec)
        report = run_eda(train, test, target, Path(a.out), time_col=spec.time)
        print(f"✔ {report.relative_to(ROOT) if report.is_relative_to(ROOT) else report}")
        return 0

    if a.cmd == "auto":
        from app.agent.orchestrator import TaskError, run_auto

        spec = _spec(a)
        if a.eda:
            from app.eda import run_eda

            train, test, target = _load_frames(spec)
            print(f"✔ EDA: {run_eda(train, test, target, OUTPUTS / 'eda', time_col=spec.time)}")
        try:
            s = run_auto(spec, note=a.note, quick=a.quick, command=command)
        except TaskError as exc:
            print(f"✖ {exc}")
            return 2
        print(f"\n✔ {s.used_for_submission}: {s.metric} OOF {s.submission_score:.4f} "
              f"(лучшая одиночная {s.best_model} {s.best_score:.4f} ± {s.best_std:.4f})")
        print(f"  отчёт: runs/{s.run_id}/report.md" + (f" · сабмит: {s.submission}"
                                                       if s.submission else ""))
        for w in s.warnings:
            print(f"  ⚠ {w}")
        return 0

    if a.cmd == "tune":
        from app.agent.tune import run_tune

        r = run_tune(_spec(a), a.model, trials=a.trials, minutes=a.minutes)
        verdict = "→ params.toml обновлён" if r["improved"] else "→ не лучше текущих, params.toml не тронут"
        print(f"\n✔ {r['model']}: {r['metric']} {r['baseline']:.4f} → {r['best']:.4f} "
              f"за {r['trials']} trials {verdict}")
        return 0

    if a.cmd == "predict":
        from app import bundle
        from app.data import load_table

        b = bundle.load(a.model)
        df = load_table(a.data)
        preds = b.predict(df)
        out = pd.DataFrame({b.id_col: df[b.id_col]} if b.id_col and b.id_col in df else {})
        if preds.ndim == 2:
            for i, c in enumerate(b.classes):
                out[str(c)] = preds[:, i]
        else:
            out[b.target] = preds
        path = Path(a.out) if a.out else OUTPUTS / "predictions" / f"{b.run_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        print(f"✔ {len(out):,} строк → {path}")
        return 0

    if a.cmd == "runs":
        board = OUTPUTS / "leaderboard.csv"
        if not board.exists():
            print("прогонов ещё нет — uv run python -m app.cli auto")
            return 0
        df = pd.read_csv(board)
        if df.empty:
            print("прогонов ещё нет — uv run python -m app.cli auto")
            return 0
        for metric, part in df.groupby("metric"):
            from app.evaluate import METRICS

            asc = not METRICS[metric].greater_is_better if metric in METRICS else False
            print(f"\n{metric}:")
            print(part.sort_values("oof_score", ascending=asc).head(a.n)[
                ["time", "author", "run_id", "used", "oof_score", "cv_std", "note"]
            ].to_string(index=False))
        return 0

    if a.cmd == "demo":
        from app import demo
        from app.agent.orchestrator import run_auto

        out_dir = ROOT / "data" / "demo"
        paths = demo.write(out_dir, a.task, a.rows)
        print("✔ данные: " + ", ".join(str(p.relative_to(ROOT)) for p in paths.values()))
        if a.no_run:
            return 0
        spec = replace(load_spec(path=Path("/nonexistent")),
                       train=str(paths["train"]), test=str(paths["test"]),
                       sample_submission=str(paths["sample_submission"]), id="client_id",
                       budget_min=5)
        s = run_auto(spec, note=f"demo {a.task}", quick=True, command=command)
        print(f"\n✔ demo {a.task}: {s.used_for_submission} {s.metric} {s.submission_score:.4f} "
              f"· отчёт runs/{s.run_id}/report.md")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
