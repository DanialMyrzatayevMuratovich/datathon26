"""Why the model decides what it decides — the raw material for insights and the pitch.

Feature importance (native gain averaged over folds, permutation as a fallback), optional SHAP, and segment tables:
target rate / mean by bins of the top features with lift over the overall level, which reads as a business story
("clients with utilisation > 80% default 3.1x more often").
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from app.data import is_text  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})


def fold_importance(adapters: list, X_sample: pd.DataFrame | None = None,
                    y_sample: np.ndarray | None = None) -> pd.Series:
    parts = [s for s in (a.importance() for a in adapters) if s is not None]
    if parts:
        return pd.concat(parts, axis=1).mean(axis=1).sort_values(ascending=False)
    if X_sample is None or y_sample is None:
        return pd.Series(dtype=float)
    model = adapters[-1]
    n = min(len(X_sample), 3000)

    class _Wrap:  # permutation_importance needs fit/predict-style estimator
        def fit(self, X, y):
            return self

        def predict(self, X):
            return model.predict(X)

    result = permutation_importance(_Wrap(), X_sample.iloc[:n], y_sample[:n], n_repeats=3,
                                    random_state=0, scoring=_scorer(model.task))
    s = pd.Series(result.importances_mean, index=X_sample.columns).clip(lower=0)
    return (s / s.sum() if s.sum() > 0 else s).sort_values(ascending=False)


def _scorer(task: str):
    from sklearn import metrics as skm

    if task == "binary":
        return lambda est, X, y: skm.roc_auc_score(y, est.predict(X))
    if task == "multiclass":
        return lambda est, X, y: skm.accuracy_score(y, est.predict(X).argmax(axis=1))
    return lambda est, X, y: -np.sqrt(skm.mean_squared_error(y, est.predict(X)))


def plot_importance(imp: pd.Series, path: Path, top: int = 20) -> Path:
    s = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.28 * len(s) + 0.8)))
    ax.barh(s.index.astype(str), s.values, color="#3b6fd8")
    ax.set_xlabel("доля важности")
    ax.set_title("Важность признаков (среднее по фолдам)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def shap_summary(adapter, X: pd.DataFrame, path: Path, n: int = 2000) -> Path | None:
    """SHAP beeswarm for tree models; None when shap or the model type is not supported."""
    try:
        import shap

        sample = X.sample(min(n, len(X)), random_state=0)
        explainer = shap.TreeExplainer(adapter.model)
        values = explainer.shap_values(adapter._frame(sample))
        if isinstance(values, list):  # older shap, binary -> [neg, pos]
            values = values[-1]
        if getattr(values, "ndim", 2) == 3:  # (n, features, classes)
            values = values[:, :, -1]
        plt.figure()
        shap.summary_plot(values, sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")
        return path
    except Exception:
        plt.close("all")
        return None


def _bin_key(s: pd.Series, bins: int) -> pd.Categorical | None:
    """Ordered segments: quantile bins with readable edges for numbers, top-8 values for categories."""
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
        q = pd.qcut(s, q=bins, duplicates="drop")
        labels = [f"{(s.min() if i == 0 else iv.left):.4g} – {iv.right:.4g}"
                  for i, iv in enumerate(q.cat.categories)]
        if len(set(labels)) == len(labels):
            q = q.cat.rename_categories(labels)
        return q.cat.add_categories(["пропуск"]).fillna("пропуск")
    if not (is_text(s) or isinstance(s.dtype, pd.CategoricalDtype) or s.nunique() <= 20):
        return None
    if s.nunique() > 50:  # dates as strings, ids, free text: segments would be noise
        return None
    sv = pd.Series([None if pd.isna(v) else str(v) for v in s], index=s.index, dtype=object)
    top = sv.value_counts().head(8).index.tolist()
    labels = sv.where(sv.isin(top) | sv.isna(), "другое").fillna("пропуск")
    return pd.Categorical(labels, categories=[*top, "другое", "пропуск"])


def segments(df: pd.DataFrame, target: pd.Series, columns: list[str], task: str,
             bins: int = 5) -> list[dict]:
    """Per column: table of bin -> rows, share, target level and lift over the overall level."""
    if task == "multiclass":
        return []
    y = target.astype(float)
    overall = float(y.mean())
    out = []
    for col in columns:
        if col not in df.columns:
            continue
        key = _bin_key(df[col], bins)
        if key is None:
            continue
        g = pd.DataFrame({"bin": key, "y": y}).groupby("bin", observed=True)["y"]
        table = pd.DataFrame({"rows": g.size(), "target": g.mean()})
        table["share"] = table["rows"] / table["rows"].sum()
        table["lift"] = table["target"] / overall if overall else np.nan
        table = table.reset_index()
        solid = table[table["rows"] >= max(30, 0.02 * len(df))]  # tiny segments are noise, not insight
        spread = float(solid["target"].max() - solid["target"].min()) if len(solid) > 1 else 0.0
        out.append({"column": col, "overall": overall, "spread": spread,
                    "table": table.round(4).to_dict(orient="records")})
    return sorted(out, key=lambda d: -d["spread"])


def segments_markdown(segs: list[dict], task: str) -> str:
    level = "доля target=1" if task == "binary" else "среднее target"
    lines = []
    for seg in segs:
        lines.append(f"**{seg['column']}** (в среднем {seg['overall']:.4g}, разброс {seg['spread']:.4g})\n")
        lines.append(f"| сегмент | строк | доля | {level} | lift |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in seg["table"]:
            lines.append(f"| {r['bin']} | {r['rows']} | {r['share']:.1%} | {r['target']:.4g} | "
                         f"{r['lift']:.2f}× |")
        lines.append("")
    return "\n".join(lines)
