"""Task type, metrics and the validation scheme — the part that decides whether a CV score means anything.

Predictions everywhere: binary -> P(class 1), multiclass -> (n, k) probabilities, regression -> values.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from app.config import SEED, TaskSpec


@dataclass(frozen=True)
class Metric:
    name: str
    tasks: tuple[str, ...]
    greater_is_better: bool
    title: str


METRICS = {
    m.name: m
    for m in (
        Metric("roc_auc", ("binary",), True, "ROC AUC"),
        Metric("gini", ("binary",), True, "Gini = 2·AUC − 1 (банковский стандарт скоринга)"),
        Metric("pr_auc", ("binary",), True, "PR AUC (average precision), для редкого класса"),
        Metric("logloss", ("binary", "multiclass"), False, "Log loss"),
        Metric("f1", ("binary",), True, "F1 по порогу"),
        Metric("accuracy", ("binary", "multiclass"), True, "Accuracy"),
        Metric("f1_macro", ("multiclass",), True, "Macro F1"),
        Metric("f1_weighted", ("multiclass",), True, "Weighted F1"),
        Metric("roc_auc_ovr", ("multiclass",), True, "ROC AUC one-vs-rest"),
        Metric("rmse", ("regression",), False, "RMSE"),
        Metric("mae", ("regression",), False, "MAE"),
        Metric("r2", ("regression",), True, "R²"),
        Metric("mape", ("regression",), False, "MAPE"),
        Metric("rmsle", ("regression",), False, "RMSLE"),
    )
}
DEFAULT_METRIC = {"binary": "roc_auc", "multiclass": "f1_macro", "regression": "rmse"}


def detect_task(y: pd.Series) -> tuple[str, str]:
    """(task, reason)."""
    n = y.nunique(dropna=True)
    if n == 2:
        return "binary", "у target ровно 2 значения"
    numeric = pd.api.types.is_numeric_dtype(y)
    if numeric:
        integer_like = bool(np.all(np.mod(y.dropna().to_numpy(dtype=float), 1) == 0))
        if not integer_like or n > 20:
            return "regression", f"числовой target, {n} уникальных значений"
    if n <= 50:
        return "multiclass", f"{n} классов"
    return "regression", f"{n} уникальных значений — считаем регрессией"


def score(metric: str, y_true: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> float:
    if metric == "roc_auc":
        return float(skm.roc_auc_score(y_true, pred))
    if metric == "gini":
        return float(2 * skm.roc_auc_score(y_true, pred) - 1)
    if metric == "pr_auc":
        return float(skm.average_precision_score(y_true, pred))
    if metric == "logloss":
        return float(skm.log_loss(y_true, np.clip(pred, 1e-15, 1 - 1e-15),
                                  labels=None if pred.ndim == 1 else np.arange(pred.shape[1])))
    if metric in {"f1", "accuracy"} and pred.ndim == 1:
        labels = (pred >= threshold).astype(int)
        fn = skm.f1_score if metric == "f1" else skm.accuracy_score
        return float(fn(y_true, labels))
    if metric in {"accuracy", "f1_macro", "f1_weighted"}:
        labels = pred.argmax(axis=1)
        if metric == "accuracy":
            return float(skm.accuracy_score(y_true, labels))
        return float(skm.f1_score(y_true, labels, average=metric.removeprefix("f1_")))
    if metric == "roc_auc_ovr":
        return float(skm.roc_auc_score(y_true, pred, multi_class="ovr",
                                       labels=np.arange(pred.shape[1])))
    if metric == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y_true, pred)))
    if metric == "mae":
        return float(skm.mean_absolute_error(y_true, pred))
    if metric == "r2":
        return float(skm.r2_score(y_true, pred))
    if metric == "mape":
        return float(skm.mean_absolute_percentage_error(y_true, pred))
    if metric == "rmsle":
        return float(np.sqrt(skm.mean_squared_log_error(np.clip(y_true, 0, None),
                                                        np.clip(pred, 0, None))))
    raise ValueError(f"неизвестная метрика {metric!r}; есть: {sorted(METRICS)}")


def better(metric: str, a: float, b: float) -> bool:
    """Is score a better than b."""
    return a > b if METRICS[metric].greater_is_better else a < b


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Threshold that maximises F1 on out-of-fold predictions (report it, do not trust it blindly)."""
    grid = np.unique(np.quantile(proba, np.linspace(0.01, 0.99, 99)))
    scores = [skm.f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def choose_cv(spec: TaskSpec, task: str, n_rows: int) -> tuple[str, str]:
    """(kind, reason) — the reason goes to the log and to the pitch."""
    if spec.cv != "auto":
        return spec.cv, "задано в task.toml"
    if spec.time:
        return "time", f"есть время ({spec.time}): обучаемся на прошлом, проверяем на будущем"
    if spec.group:
        return "group", f"группы по {spec.group}: один объект не попадает и в train, и в valid"
    if task in {"binary", "multiclass"}:
        return "stratified", "классификация: доли классов одинаковы во всех фолдах"
    return "kfold", "регрессия без времени и групп"


def splits(kind: str, folds: int, y: np.ndarray, groups: np.ndarray | None,
           order: np.ndarray | None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Index pairs. For `time`, `order` sorts rows by time; the earliest chunk is never validated."""
    idx = np.arange(len(y))
    if kind == "time":
        sorted_idx = idx[np.argsort(order, kind="stable")] if order is not None else idx
        return [(sorted_idx[tr], sorted_idx[va])
                for tr, va in TimeSeriesSplit(n_splits=folds).split(sorted_idx)]
    if kind == "group":
        if groups is None:
            raise ValueError("cv = group, но не задана колонка group")
        if len(np.unique(y)) <= 50 and pd.api.types.is_integer_dtype(y):
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
            return list(splitter.split(idx, y, groups))
        return list(GroupKFold(n_splits=folds).split(idx, y, groups))
    if kind == "stratified":
        return list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED).split(idx, y))
    return list(KFold(n_splits=folds, shuffle=True, random_state=SEED).split(idx))
