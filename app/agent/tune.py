"""Optuna tuning of one model on exactly the same folds as `auto`, so the score is comparable.

Writes the best parameters to params.toml only when they beat the current ones; the next `auto` run picks them up.
Good for unattended night runs: `uv run python -m app.cli tune --model lgbm --minutes 60`.
"""

import json
import secrets
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna

from app.agent.log import RunLog
from app.agent.orchestrator import _cv_model, prepare, whoami
from app.config import PARAMS_FILE, ROOT, TaskSpec, load_params, save_params
from app.evaluate import METRICS, better

SPACES = {
    "lgbm": lambda t: {
        "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10, log=True),
    },
    "catboost": lambda t: {
        "depth": t.suggest_int("depth", 4, 10),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 20, log=True),
        "random_strength": t.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 1.0),
    },
    "xgb": lambda t: {
        "max_depth": t.suggest_int("max_depth", 3, 10),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_weight": t.suggest_float("min_child_weight", 1, 20, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True),
    },
    "hgb": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 15, 255, log=True),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 200, log=True),
        "l2_regularization": t.suggest_float("l2_regularization", 1e-4, 10, log=True),
    },
}


def run_tune(spec: TaskSpec, model: str, *, trials: int = 50, minutes: float = 30,
             root: Path = ROOT, params_path: Path | None = None, echo: bool = True) -> dict:
    if model not in SPACES:
        raise ValueError(f"tune умеет {sorted(SPACES)}")
    params_path = params_path or (PARAMS_FILE if root == ROOT else root / "params.toml")
    run_id = f"{datetime.now():%Y%m%dT%H%M}-{whoami()}-tune-{model}-{secrets.token_hex(2)}"
    log = RunLog(run_id, root / "runs" / run_id, echo=echo)
    c = prepare(spec, log)
    gib = METRICS[c.metric].greater_is_better

    def cv(params: dict) -> float:
        _, _, scores, _, _ = _cv_model(model, c.X, c.y, c.folds, c.task, c.n_classes, c.cats,
                                       c.spec, c.metric, None, params=params)
        return float(np.mean(scores))

    current = load_params(params_path).get(model, {})
    t = time.perf_counter()
    baseline = cv(current)
    log.step("baseline", "ok", f"{model} {c.metric} {baseline:.4f} на текущих параметрах",
             args={"params": current}, started=t)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize" if gib else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    if current:
        study.enqueue_trial(current)

    def objective(trial):
        value = cv(SPACES[model](trial))
        if echo:
            print(f"  trial {trial.number:>3}: {value:.4f}", flush=True)
        return value

    t = time.perf_counter()
    study.optimize(objective, n_trials=trials, timeout=minutes * 60, catch=(Exception,))
    done = [tr for tr in study.trials if tr.value is not None]
    if not done:
        log.step("optuna", "fail", "ни один trial не завершился")
        raise RuntimeError("tune: ни один trial не завершился — смотри ошибки модели")
    best, value = study.best_params, study.best_value
    improved = better(c.metric, value, baseline) and abs(value - baseline) > 1e-5
    log.step("optuna", "ok", f"{len(done)} trials за {time.perf_counter() - t:.0f} с: "
             f"{c.metric} {baseline:.4f} → {value:.4f}",
             decision="записать в params.toml" if improved else "оставить текущие параметры",
             args={"best_params": best})
    if improved:
        save_params(model, best, note=f"{run_id}: {c.metric} {baseline:.4f} → {value:.4f}",
                    path=params_path)
    result = {"run_id": run_id, "model": model, "metric": c.metric, "baseline": baseline,
              "best": value, "improved": improved, "params": best, "trials": len(done)}
    (log.dir / "tune.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return result
