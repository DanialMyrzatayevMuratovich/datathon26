"""Autonomous AutoML loop — hand it a dataset, it solves end to end and explains every decision.

plan -> load -> validate (target, id, task, metric) -> cv scheme -> features -> model ladder with fallbacks and a time
budget -> compare + sanity checks (no signal / too good to be true) -> ensemble if it helps -> explain -> submission
(checked against test and sample_submission) -> save -> advise next experiments -> report.
Every step is one JSONL line with its decision and reason: runs/<run_id>/agent_log.jsonl.
"""

import json
import secrets
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from getpass import getuser
from pathlib import Path

import numpy as np
import pandas as pd

from app import custom_features
from app.agent.log import RunLog
from app.agent.planner import advise
from app.bundle import Bundle, best_pointer
from app.config import ROOT, SEED, TaskSpec, find_raw, load_params, resolve
from app.data import guess_id, guess_target, load_table, to_datetime
from app.evaluate import (
    DEFAULT_METRIC,
    METRICS,
    best_threshold,
    better,
    choose_cv,
    detect_task,
    score,
    splits,
)
from app.explain import fold_importance, plot_importance, segments, segments_markdown, shap_summary
from app.features import apply_plan, fit_plan
from app.models import Adapter, available, candidates
from app.schemas import Advice, ModelScore, RunSummary

TRAIN_PATTERNS = ("train*.csv", "train*.parquet", "train*.xlsx", "*train*.*")
TEST_PATTERNS = ("test*.csv", "test*.parquet", "test*.xlsx", "*test*.*")
SUBMISSION_PATTERNS = ("sample_submission*.*", "*submission*.*")
QUICK_MODELS = ("dummy", "hgb", "lgbm")
LINEAR_MAX_ROWS = 300_000
TOO_GOOD = {"roc_auc": 0.995, "gini": 0.99, "accuracy": 0.995, "r2": 0.99, "f1_macro": 0.995}


class TaskError(RuntimeError):
    """The run cannot continue; the message says what to fix (shown to people and agents)."""


def whoami() -> str:
    try:
        role = subprocess.run(["git", "config", "dt.role"], capture_output=True, text=True,
                              cwd=ROOT, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        role = ""
    return role or getuser()


def _cv_model(name, X, y, folds, task, n_classes, cats, spec, metric, X_test, params=None):
    shape = (len(y), n_classes) if task == "multiclass" else (len(y),)
    oof = np.full(shape, np.nan)
    test_pred, scores, adapters, iters = None, [], [], []
    for tr, va in folds:
        model = Adapter(name, task, n_classes, cats, gpu=spec.gpu,
                        params=params).fit(
            X.iloc[tr], y[tr], X.iloc[va], y[va])
        p = model.predict(X.iloc[va])
        oof[va] = p
        scores.append(score(metric, y[va], p, spec.threshold))
        if X_test is not None:
            tp = model.predict(X_test)
            test_pred = tp if test_pred is None else test_pred + tp
        adapters.append(model)
        if model.best_iteration:
            iters.append(model.best_iteration)
    if test_pred is not None:
        test_pred = test_pred / len(folds)
    return oof, test_pred, scores, adapters, (int(np.mean(iters)) if iters else None)


def _oof_score(metric, y, oof, mask, threshold):
    return score(metric, y[mask], oof[mask], threshold)


def _submission(spec, test, preds_raw, task, classes, target, id_col, sample_sub):
    col = spec.submission_target or target
    if sample_sub is not None and sample_sub.shape[1] == 2 and not spec.submission_target:
        col = sample_sub.columns[1]
    out = pd.DataFrame(index=test.index)
    if id_col and id_col in test.columns:
        out[id_col] = test[id_col].to_numpy()
    if task == "binary":
        out[col] = preds_raw if spec.predict_proba else np.asarray(classes, dtype=object)[
            (preds_raw >= spec.threshold).astype(int)]
    elif task == "multiclass":
        wide = sample_sub is not None and sample_sub.shape[1] > 2 and {
            str(c) for c in classes} <= {str(c) for c in sample_sub.columns}
        if wide:  # one probability column per class, named as in sample_submission
            names = {str(c): c for c in sample_sub.columns}
            for i, cls in enumerate(classes):
                out[names[str(cls)]] = preds_raw[:, i]
        else:
            out[col] = np.asarray(classes, dtype=object)[preds_raw.argmax(axis=1)]
    else:
        out[col] = preds_raw
    if sample_sub is not None and id_col and id_col in sample_sub.columns and id_col in out:
        ordered = sample_sub[[id_col]].merge(out, on=id_col, how="left")
        if len(ordered) == len(out) and ordered.notna().all().all():
            out = ordered[[c for c in sample_sub.columns if c in ordered.columns]]
    return out.reset_index(drop=True)


@dataclass
class Ctx:
    """Everything a model run needs; `prepare` builds it identically for `auto` and `tune`."""

    spec: TaskSpec
    train: pd.DataFrame
    test: pd.DataFrame | None
    sample_sub: pd.DataFrame | None
    target: str
    id_col: str | None
    task: str
    metric: str
    classes: list | None
    y: np.ndarray
    n_classes: int
    positive_share: float | None
    kind: str
    why_cv: str
    folds: list
    validated: np.ndarray
    plan: object
    X: pd.DataFrame
    X_test: pd.DataFrame | None
    cats: list[str]


def prepare(spec: TaskSpec, log: RunLog, train_df: pd.DataFrame | None = None,
            test_df: pd.DataFrame | None = None) -> Ctx:
    # load ------------------------------------------------------------------------------------
    t = time.perf_counter()
    paths = {}
    if train_df is None:
        train_path = resolve(spec.train) if spec.train else find_raw(TRAIN_PATTERNS)
        if not train_path or not Path(train_path).exists():
            log.step("load", "fail", "нет train", decision="стоп")
            raise TaskError("Нет train: положи файл в data/raw/ и впиши [task] train в task.toml")
        paths["train"] = str(train_path)
        train_df = load_table(train_path)
    if test_df is None and paths:
        test_path = resolve(spec.test) if spec.test else find_raw(TEST_PATTERNS)
        if test_path and Path(test_path).exists():
            paths["test"] = str(test_path)
            test_df = load_table(test_path)
    sample_sub = None
    sub_path = resolve(spec.sample_submission) if spec.sample_submission else (
        find_raw(SUBMISSION_PATTERNS) if paths else None)
    if sub_path and Path(sub_path).exists():
        paths["sample_submission"] = str(sub_path)
        sample_sub = load_table(sub_path)
    n_before = train_df.shape[1]
    train = custom_features.add_features(train_df.copy())
    test = custom_features.add_features(test_df.copy()) if test_df is not None else None
    added = train.shape[1] - n_before
    if spec.sample and len(train) > spec.sample:
        train = train.sample(spec.sample, random_state=SEED)
    train = train.reset_index(drop=True)
    log.step("load", "ok", f"train {len(train):,}×{train.shape[1]}"
             + (f", test {len(test):,}×{test.shape[1]}" if test is not None else ", test нет")
             + (f", custom_features +{added}" if added else ""),
             decision="sample_submission найден" if sample_sub is not None else None,
             args=paths, started=t)

    # validate --------------------------------------------------------------------------------
    t = time.perf_counter()
    target = spec.target or guess_target(train, test)
    if not target or target not in train.columns:
        log.step("validate", "fail", f"target не найден ({spec.target or 'не задан'})")
        raise TaskError("Не знаю target: впиши [task] target в task.toml (или передай --target)")
    why_target = "задан" if spec.target else "единственная колонка train, которой нет в test"
    missing = int(train[target].isna().sum())
    if missing:
        train = train[train[target].notna()].reset_index(drop=True)
    id_col = spec.id or guess_id(train, test) or (
        sample_sub.columns[0] if sample_sub is not None and test is not None
        and sample_sub.columns[0] in test.columns else None)
    if spec.task != "auto":
        task, why_task = spec.task, "задано"
    else:
        task, why_task = detect_task(train[target])
    metric = spec.metric if spec.metric != "auto" else DEFAULT_METRIC[task]
    if metric not in METRICS or task not in METRICS[metric].tasks:
        log.step("validate", "fail", f"метрика {metric} не подходит к задаче {task}")
        raise TaskError(f"Метрика {metric!r} не для {task}; подходят: "
                        f"{[m for m, v in METRICS.items() if task in v.tasks]}")
    classes = None
    if task in {"binary", "multiclass"}:
        values = train[target].unique().tolist()
        try:
            classes = sorted(values)
        except TypeError:  # mixed types
            classes = sorted(values, key=str)
        y = train[target].map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=int)
    else:
        y = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
    n_classes = len(classes) if classes else 1
    positive_share = float(y.mean()) if task == "binary" else None
    log.step("validate", "warn" if missing else "ok",
             f"target `{target}` ({why_target}), задача {task} ({why_task}), метрика {metric}"
             + (f", id `{id_col}`" if id_col else "")
             + (f", класс 1 = {classes[1]!r} ({positive_share:.1%})" if task == "binary" else "")
             + (f"; выброшено {missing} строк без target" if missing else ""), started=t)

    # cv --------------------------------------------------------------------------------------
    kind, why_cv = choose_cv(spec, task, len(train))
    groups = train[spec.group].to_numpy() if spec.group else None
    order = to_datetime(train[spec.time]).to_numpy() if spec.time else None
    folds = splits(kind, spec.folds, y, groups, order)
    validated = np.zeros(len(y), dtype=bool)
    for _, va in folds:
        validated[va] = True
    log.step("cv", "ok", f"{kind}, {len(folds)} фолдов, валидируется {validated.mean():.0%} строк",
             decision=kind, reason=why_cv)

    # features --------------------------------------------------------------------------------
    t = time.perf_counter()
    if spec.group and spec.group not in spec.drop:
        spec = replace(spec, drop=(*spec.drop, spec.group))
    plan = fit_plan(train, spec, target, id_col)
    X = apply_plan(plan, train)
    X_test = apply_plan(plan, test) if test is not None else None
    cats = plan.categorical_features
    log.step("features", "ok",
             f"{len(plan.columns)} признаков: {len(plan.numeric)} числ., {len(cats)} кат., "
             f"{len(plan.frequency)} частотных, {len(plan.datetime)} дат, {len(plan.text)} текст.; "
             f"отброшено {len(plan.dropped)}", args={"dropped": plan.dropped}, started=t)

    return Ctx(spec=spec, train=train, test=test, sample_sub=sample_sub, target=target,
               id_col=id_col, task=task, metric=metric, classes=classes, y=y,
               n_classes=n_classes, positive_share=positive_share, kind=kind, why_cv=why_cv,
               folds=folds, validated=validated, plan=plan, X=X, X_test=X_test, cats=cats)


def run_auto(spec: TaskSpec, *, note: str = "", author: str | None = None, root: Path = ROOT,
             train_df: pd.DataFrame | None = None, test_df: pd.DataFrame | None = None,
             quick: bool = False, echo: bool = True, command: str = "") -> RunSummary:
    t_run, started = time.perf_counter(), datetime.now(UTC).astimezone()
    author = author or whoami()
    run_id = f"{datetime.now():%Y%m%dT%H%M}-{author}-{secrets.token_hex(2)}"
    run_dir = root / "runs" / run_id
    log = RunLog(run_id, run_dir, echo=echo)
    if quick:
        spec = replace(spec, folds=min(spec.folds, 3), models=spec.models or QUICK_MODELS,
                       sample=spec.sample or 100_000)
    log.step("plan", "ok", f"автопилот{' (quick)' if quick else ''}: {note or 'без заметки'}",
             decision="load → validate → cv → features → models → ensemble → explain → submit",
             args={k: v for k, v in spec.__dict__.items() if v not in ("", (), None)})

    c = prepare(spec, log, train_df, test_df)
    spec, train, test, sample_sub = c.spec, c.train, c.test, c.sample_sub
    target, id_col, task, metric, classes = c.target, c.id_col, c.task, c.metric, c.classes
    y, n_classes, positive_share = c.y, c.n_classes, c.positive_share
    kind, why_cv, folds, validated = c.kind, c.why_cv, c.folds, c.validated
    plan, X, X_test, cats = c.plan, c.X, c.X_test, c.cats

    # models ----------------------------------------------------------------------------------
    avail, budget_s = available(), spec.budget_min * 60
    tuned = load_params(root / "params.toml")
    results: dict[str, ModelScore] = {}
    oofs, test_preds, fold_models = {}, {}, {}
    t_models = time.perf_counter()
    for name in candidates(spec.models):
        if avail.get(name) != "ok":
            results[name] = ModelScore(model=name, status="skip", error=avail.get(name))
            log.step(f"model:{name}", "skip", "библиотека недоступна", reason=avail.get(name))
            continue
        if name == "linear" and len(X) > LINEAR_MAX_ROWS:
            results[name] = ModelScore(model=name, status="skip", error="много строк")
            log.step(f"model:{name}", "skip", f"{len(X):,} строк — линейная модель долгая")
            continue
        if name != "dummy" and time.perf_counter() - t_models > budget_s:
            results[name] = ModelScore(model=name, status="skip", error="бюджет времени")
            log.step(f"model:{name}", "skip", f"исчерпан бюджет {spec.budget_min:g} мин",
                     reason="увеличь budget_min или запусти модель отдельно: --models")
            continue
        t = time.perf_counter()
        try:
            oof, tp, fold_scores, adapters, iters = _cv_model(
                name, X, y, folds, task, n_classes, cats, spec, metric, X_test,
                params=tuned.get(name))
        except Exception as exc:  # fallback ladder: one broken model never stops the run
            results[name] = ModelScore(model=name, status="fail",
                                       error=f"{type(exc).__name__}: {str(exc)[:300]}")
            log.step(f"model:{name}", "fail", f"{type(exc).__name__}: {str(exc)[:160]}",
                     decision="следующая модель лестницы", started=t)
            continue
        oofs[name], test_preds[name], fold_models[name] = oof, tp, adapters
        results[name] = ModelScore(model=name, status="ok", mean=float(np.mean(fold_scores)),
                                   std=float(np.std(fold_scores)), folds=fold_scores,
                                   seconds=round(time.perf_counter() - t, 1), best_iteration=iters)
        r = results[name]
        log.step(f"model:{name}", "ok", f"{metric} {r.mean:.4f} ± {r.std:.4f} за {r.seconds:.0f} с"
                 + (f", итераций ~{iters}" if iters else ""), started=t)

    # compare ---------------------------------------------------------------------------------
    ok = {n: r for n, r in results.items() if r.status == "ok"}
    real = {n: r for n, r in ok.items() if n != "dummy"} or ok
    if not real:
        log.step("compare", "fail", "ни одна модель не обучилась")
        raise TaskError("Ни одна модель не обучилась — смотри ошибки в runs/<run_id>/agent_log.jsonl")
    gib = METRICS[metric].greater_is_better
    ranked = sorted(real, key=lambda n: real[n].mean, reverse=gib)
    best = ranked[0]
    warnings: list[str] = []
    gain = None
    if "dummy" in ok and best != "dummy":
        gain = abs(real[best].mean - ok["dummy"].mean)
        relative = gain / max(abs(ok["dummy"].mean), 1e-9)
        if (metric in {"roc_auc", "gini", "pr_auc", "accuracy", "f1", "f1_macro"} and gain < 0.01) or (
                metric in {"rmse", "mae", "logloss"} and relative < 0.01):
            warnings.append("лучшая модель почти не лучше dummy — проверь target и признаки")
    if metric in TOO_GOOD and gib and real[best].mean >= TOO_GOOD[metric]:
        warnings.append(f"{metric} = {real[best].mean:.4f} — подозрительно высоко, ищи утечку")
    std_ratio = real[best].std / max(abs(real[best].mean), 1e-9)
    if std_ratio > 0.05:
        warnings.append(f"разброс по фолдам {std_ratio:.0%} от среднего — скор нестабилен")
    log.step("compare", "warn" if warnings else "ok",
             " · ".join(f"{n} {real[n].mean:.4f}" for n in ranked), decision=f"лучшая {best}",
             reason="; ".join(warnings) or "скор лучше dummy, разброс по фолдам приемлемый")

    # ensemble --------------------------------------------------------------------------------
    blend_weights = {best: 1.0}
    best_oof = _oof_score(metric, y, oofs[best], validated, spec.threshold)
    chosen_score, chosen_name = best_oof, best
    top = [n for n in ranked if n in oofs][:3]
    for k in (2, 3):
        if len(top) < k:
            break
        members = top[:k]
        blend = np.mean([oofs[m] for m in members], axis=0)
        s = _oof_score(metric, y, blend, validated, spec.threshold)
        if better(metric, s, chosen_score) and abs(s - chosen_score) > 1e-4:
            chosen_score, chosen_name = s, f"ensemble({'+'.join(members)})"
            blend_weights = {m: 1.0 / k for m in members}
    log.step("ensemble", "ok", f"OOF {best} {best_oof:.4f}; выбран {chosen_name} {chosen_score:.4f}",
             decision=chosen_name,
             reason="усреднение лучше одиночной модели" if len(blend_weights) > 1
             else "усреднение не улучшило OOF — берём одну модель")

    # explain ---------------------------------------------------------------------------------
    t = time.perf_counter()
    tree = next((n for n in ranked if n in {"lgbm", "catboost", "xgb", "hgb"} and n in fold_models),
                best)
    sample_idx = np.random.default_rng(SEED).choice(len(X), size=min(len(X), 3000), replace=False)
    importance = fold_importance(fold_models[tree], X.iloc[sample_idx], y[sample_idx])
    figures = []
    if not importance.empty:
        figures.append(plot_importance(importance, run_dir / "importance.png").name)
    if spec.shap and tree in {"lgbm", "catboost", "xgb"}:
        shap_path = shap_summary(fold_models[tree][0], X, run_dir / "shap.png")
        if shap_path:
            figures.append(shap_path.name)
    source_cols = []
    for feat in importance.index:
        base = feat.split("__")[0]
        if base in train.columns and base not in source_cols and base != target:
            source_cols.append(base)
    y_series = pd.Series(y, index=train.index)
    segs = segments(train, y_series, source_cols[:5], task)
    threshold_note = ""
    if task == "binary":
        thr, f1 = best_threshold(y[validated], oofs[best][validated])
        threshold_note = f"порог, максимизирующий F1 на OOF: {thr:.3f} (F1 {f1:.3f})"
    log.step("explain", "ok", f"важность по {tree}; топ: {', '.join(importance.index[:5])}",
             args={"figures": figures}, started=t)

    # submit ----------------------------------------------------------------------------------
    sub_path = None
    if test is not None:
        preds = sum(w * test_preds[m] for m, w in blend_weights.items())
        sub = _submission(spec, test, preds, task, classes, target, id_col, sample_sub)
        problems = []
        if len(sub) != len(test):
            problems.append(f"строк {len(sub)} ≠ test {len(test)}")
        if sub.isna().any().any():
            problems.append("есть пропуски в предсказаниях")
        if sample_sub is not None and list(sub.columns) != list(sample_sub.columns):
            problems.append(f"колонки {list(sub.columns)} ≠ sample_submission "
                            f"{list(sample_sub.columns)}")
        sub_dir = root / "outputs" / "submissions"
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(run_dir / "submission.csv", index=False)
        sub_path = sub_dir / f"{run_id}.csv"
        sub.to_csv(sub_path, index=False)
        log.step("submit", "warn" if problems else "ok",
                 f"{len(sub):,} строк, колонки {list(sub.columns)}", reason="; ".join(problems) or None,
                 args={"path": str(sub_path.relative_to(root))})
        warnings += [f"сабмит: {p}" for p in problems]
    else:
        log.step("submit", "skip", "нет test — только CV")

    # save ------------------------------------------------------------------------------------
    bundle = Bundle(run_id=run_id, plan=plan,
                    adapters={m: fold_models[m] for m in blend_weights}, weights=blend_weights,
                    task=task, target=target, id_col=id_col, classes=classes,
                    predict_proba=spec.predict_proba, threshold=spec.threshold, metric=metric,
                    score=chosen_score)
    model_path = bundle.save(root / "models")
    pointer = best_pointer(root / "models")
    is_best = pointer is None or pointer.get("metric") != metric or better(
        metric, chosen_score, pointer["score"])
    if is_best:
        (root / "models" / "best.json").write_text(json.dumps(
            {"run_id": run_id, "metric": metric, "score": chosen_score, "model": chosen_name,
             "author": author}, ensure_ascii=False, indent=2), encoding="utf-8")
    log.step("save", "ok", f"{model_path.relative_to(root)}",
             decision="новый лучший → models/best.json" if is_best else "лучший прежний")

    # advise ----------------------------------------------------------------------------------
    facts = {
        "task": task, "metric": metric, "cv": kind, "n_train": len(train), "n_features": X.shape[1],
        "leaderboard": [r.model_dump(include={"model", "status", "mean", "std"})
                        for r in results.values()],
        "best": best, "chosen": chosen_name, "score": chosen_score, "gain_over_dummy": gain,
        "cv_std_ratio": std_ratio, "positive_share": positive_share,
        "top_features": importance.index[:10].tolist(),
        "suspicious": any("утечку" in w for w in warnings),
        "high_card": list(plan.frequency), "datetime": list(plan.datetime),
        "time_col": spec.time or (list(plan.datetime)[0] if plan.datetime else ""),
        "warnings": warnings,
    }
    advice, mode = advise(facts)
    advice = Advice(ideas=advice.ideas[:8], risks=advice.risks[:5])
    log.step("advise", "ok", f"{len(advice.ideas)} идей, {len(advice.risks)} рисков ({mode})")

    # report ----------------------------------------------------------------------------------
    summary = RunSummary(
        run_id=run_id, author=author, note=note, started=started,
        seconds=round(time.perf_counter() - t_run, 1), task=task, metric=metric,
        greater_is_better=gib, cv=kind, folds=len(folds), n_train=len(train),
        n_test=0 if test is None else len(test), n_features=X.shape[1], best_model=best,
        best_score=real[best].mean, best_std=real[best].std, used_for_submission=chosen_name,
        submission_score=chosen_score, leaderboard=list(results.values()),
        top_features=[(f, round(float(v), 4)) for f, v in importance.head(20).items()],
        warnings=warnings, advice=advice, advice_mode=mode,
        submission=str(sub_path.relative_to(root)) if sub_path else None,
        model_path=str(model_path.relative_to(root)), command=command,
    )
    (run_dir / "metrics.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    _write_report(summary, log, plan, why_cv, figures, segs, threshold_note, run_dir)
    _append_tables(summary, root)
    log.step("report", "ok", f"{(run_dir / 'report.md').relative_to(root)}")
    return summary


def _write_report(s: RunSummary, log: RunLog, plan, why_cv, figures, segs, threshold_note,
                  run_dir: Path) -> None:
    m = METRICS[s.metric]
    md = [f"# Прогон {s.run_id}", "", f"> {s.note}" if s.note else "", "",
          f"- кто: {s.author} · {s.started:%Y-%m-%d %H:%M} · {s.seconds:.0f} с",
          f"- задача **{s.task}**, метрика **{m.title}** ({'больше' if m.greater_is_better else 'меньше'}"
          f" — лучше), валидация **{s.cv}** × {s.folds} — {why_cv}",
          f"- train {s.n_train:,} строк, test {s.n_test:,}, признаков {s.n_features}",
          "", "## Лидерборд (среднее по фолдам)", "",
          f"| модель | {s.metric} | ± | время, с | итераций | статус |", "|---|---:|---:|---:|---:|---|"]
    for r in sorted(s.leaderboard, key=lambda r: (r.mean is None, -(r.mean or 0)
                                                   if m.greater_is_better else (r.mean or 0))):
        md.append(f"| {r.model} | {'' if r.mean is None else f'{r.mean:.4f}'} | "
                  f"{'' if r.std is None else f'{r.std:.4f}'} | {r.seconds:.0f} | "
                  f"{r.best_iteration or ''} | {r.status}{' — ' + r.error if r.error else ''} |")
    md += ["", "## Итог",
           f"- лучшая одиночная модель: **{s.best_model}** {s.best_score:.4f} ± {s.best_std:.4f}",
           f"- в сабмит: **{s.used_for_submission}**, OOF {s.metric} = {s.submission_score:.4f}"]
    if threshold_note:
        md.append(f"- {threshold_note}")
    if s.submission:
        md.append(f"- сабмит: `{s.submission}` · модель: `{s.model_path}`")
    md += ["", "## Предупреждения", ""] + ([f"- ⚠ {w}" for w in s.warnings] or ["- нет"])
    md += ["", "## Важность признаков", ""]
    md += [f"{i}. `{f}` — {v:.1%}" for i, (f, v) in enumerate(s.top_features[:15], 1)]
    md += [f"\n![{f}]({f})" for f in figures]
    if plan.dropped:
        md += ["", "Отброшены: " + ", ".join(f"`{c}` ({r})" for c, r in plan.dropped.items())]
    if segs:
        md += ["", "## Сегменты по топ-признакам (сырьё для инсайтов)", "",
               segments_markdown(segs, s.task)]
    md += ["", f"## Что попробовать дальше ({s.advice_mode})", ""]
    md += [f"- {i}" for i in s.advice.ideas] + [f"- ⚠ {r}" for r in s.advice.risks]
    if s.command:
        md += ["", "## Как воспроизвести", "", f"```bash\n{s.command}\n```"]
    md += ["", "## Журнал агента", "", "| # | шаг | статус | итог | решение | причина |",
           "|---:|---|---|---|---|---|"]
    for st in log.steps:
        cells = [st.summary, st.decision or "", st.reason or ""]
        cells = [c.replace("|", "/").replace("\n", " ") for c in cells]
        md.append(f"| {st.step} | {st.tool} | {st.status} | {cells[0]} | {cells[1]} | {cells[2]} |")
    (run_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _append_tables(s: RunSummary, root: Path) -> None:
    row = {"run_id": s.run_id, "time": f"{s.started:%Y-%m-%d %H:%M}", "author": s.author,
           "note": s.note, "task": s.task, "metric": s.metric, "best_model": s.best_model,
           "cv_mean": round(s.best_score, 5), "cv_std": round(s.best_std, 5),
           "used": s.used_for_submission, "oof_score": round(s.submission_score, 5),
           "n_features": s.n_features, "submission": s.submission or ""}
    board = root / "outputs" / "leaderboard.csv"
    board.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(board, mode="a", header=not board.exists(), index=False)
    journal = root / "docs" / "EXPERIMENTS.md"
    journal.parent.mkdir(parents=True, exist_ok=True)
    if not journal.exists():
        journal.write_text("# Журнал экспериментов\n\n", encoding="utf-8")
    line = (f"- {datetime.now():%d.%m %H:%M} · {s.author} · `{s.run_id}` · {s.best_model} {s.metric} "
            f"{s.best_score:.4f}±{s.best_std:.4f} → {s.used_for_submission} {s.submission_score:.4f}"
            + (f" · {s.note}" if s.note else "") + "\n")
    with journal.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)
