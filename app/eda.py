"""Auto-EDA in one command: a markdown report with charts that both people (and their agents) read first.

Covers shape, column roles, target, missing values, univariate signal of every feature, categorical target rates,
train/test drift and leakage suspects. Agents read the report, not the raw rows.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from app.data import is_text, memory_mb, profile_frame, to_datetime  # noqa: E402
from app.evaluate import detect_task  # noqa: E402

SAMPLE = 200_000  # rows used for the heavier statistics
LEAK_AUC = 0.97
DRIFT_KS = 0.1


def _fig(path: Path, fig) -> str:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path.name


def univariate_signal(X: pd.DataFrame, y: pd.Series, task: str) -> pd.DataFrame:
    """Signal of each column alone: AUC (binary), mutual information (multiclass), Spearman (regression)."""
    rows = []
    for col in X.columns:
        s = X[col]
        if is_text(s) or isinstance(s.dtype, pd.CategoricalDtype):
            codes = s.astype("category").cat.codes.replace(-1, np.nan)
            num = codes.astype(float)
            kind = "cat"
        elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            num, kind = s.astype(float), "num"
        else:
            continue
        mask = num.notna() & y.notna()
        if mask.sum() < 30 or num[mask].nunique() < 2:
            continue
        xv, yv = num[mask], y[mask]
        try:
            if task == "binary":
                if kind == "cat":  # category codes have no order: score by target rate per category
                    rate = yv.groupby(xv).transform("mean")
                    auc = roc_auc_score(yv, rate)
                else:
                    auc = roc_auc_score(yv, xv)
                value, name = max(auc, 1 - auc), "AUC"
            elif task == "multiclass":
                value = float(mutual_info_classif(xv.to_frame(), yv, random_state=0,
                                                  discrete_features=kind == "cat")[0])
                name = "MI"
            else:
                value, name = abs(stats.spearmanr(xv, yv).statistic), "|Spearman|"
        except ValueError:
            continue
        rows.append({"feature": col, "kind": kind, name: round(float(value), 4),
                     "null_share": round(float(s.isna().mean()), 3)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(df.columns[2], ascending=False).reset_index(drop=True)


def drift(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        a, b = train[col].dropna(), test[col].dropna()
        if a.empty or b.empty:
            continue
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            value, kind = stats.ks_2samp(a.astype(float), b.astype(float)).statistic, "KS"
        else:
            pa = a.astype(str).value_counts(normalize=True)
            pb = b.astype(str).value_counts(normalize=True)
            value, kind = 0.5 * pa.subtract(pb, fill_value=0).abs().sum(), "TV"
            unseen = 1 - b.astype(str).isin(set(a.astype(str))).mean()
            if unseen > 0.05:
                kind += f", {unseen:.0%} новых значений"
        rows.append({"feature": col, "drift": round(float(value), 3), "measure": kind})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("drift", ascending=False).reset_index(drop=True)


def run_eda(train: pd.DataFrame, test: pd.DataFrame | None, target: str, out_dir: Path,
            time_col: str = "") -> Path:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    md: list[str] = ["# Auto-EDA", ""]
    tr = train.sample(SAMPLE, random_state=0) if len(train) > SAMPLE else train
    md += [
        "## Данные",
        f"- train: {len(train):,} строк × {train.shape[1]} колонок, {memory_mb(train):.1f} МБ, "
        f"полных дублей {int(train.duplicated().sum()):,}",
    ]
    if test is not None:
        md.append(f"- test: {len(test):,} строк × {test.shape[1]} колонок; "
                  f"колонки только в train: {[c for c in train.columns if c not in test.columns]}")
    if len(tr) < len(train):
        md.append(f"- тяжёлые статистики считаются по выборке {len(tr):,} строк")
    md.append("")

    prof = profile_frame(train, target)
    md += ["## Колонки", "", prof.to_markdown(index=False), ""]

    task = None
    if target and target in train.columns:
        y = tr[target]
        task, why = detect_task(train[target].dropna())
        md += ["## Target", f"- `{target}`: задача **{task}** ({why}); "
               f"пропусков в target {train[target].isna().mean():.2%}"]
        fig, ax = plt.subplots(figsize=(6, 3))
        if task == "regression":
            ax.hist(train[target].dropna(), bins=50, color="#3b6fd8")
            skew = float(train[target].skew())
            md.append(f"- асимметрия {skew:.2f}" + (" → попробуй log1p(target)" if skew > 1.5 else ""))
        else:
            counts = train[target].value_counts()
            ax.bar(counts.index.astype(str), counts.values, color="#3b6fd8")
            shares = (counts / counts.sum()).round(4).to_dict()
            md.append(f"- доли классов: {shares}")
        ax.set_title(f"Распределение {target}")
        md += ["", f"![target]({_fig(out_dir / 'target.png', fig)})", ""]

        if task == "binary" and y.nunique() == 2:
            y_enc = (y == sorted(y.dropna().unique())[-1]).astype(int)
        elif task == "multiclass":
            y_enc = y.astype("category").cat.codes
        else:
            y_enc = pd.to_numeric(y, errors="coerce")
        feats = tr.drop(columns=[target])
        signal = univariate_signal(feats, y_enc, task)
        if not signal.empty:
            md += ["## Сигнал каждого признака по отдельности", "",
                   signal.head(25).to_markdown(index=False), ""]
            metric = signal.columns[2]
            top = signal.head(15)[::-1]
            fig, ax = plt.subplots(figsize=(7, 0.3 * len(top) + 1))
            ax.barh(top["feature"].astype(str), top[metric], color="#3b6fd8")
            ax.set_title(f"Сила признака в одиночку ({metric})")
            md += [f"![signal]({_fig(out_dir / 'signal.png', fig)})", ""]
            if metric == "AUC":
                leaks = signal[signal["AUC"] >= LEAK_AUC]["feature"].tolist()
                if leaks:
                    md += [f"> ⚠ **Подозрение на утечку**: {leaks} в одиночку дают AUC ≥ {LEAK_AUC}. "
                           "Проверь, не известен ли признак только после события.", ""]

        cats = [c for c in feats.columns
                if (is_text(feats[c]) or isinstance(feats[c].dtype, pd.CategoricalDtype))
                and 2 <= feats[c].nunique() <= 30]
        if cats and task != "multiclass":
            rows = []
            for c in cats:
                rate = y_enc.groupby(feats[c].astype(str)).agg(["mean", "size"])
                rate = rate[rate["size"] >= 30]
                if len(rate) >= 2:
                    rows.append({"feature": c, "min": rate["mean"].min(), "max": rate["mean"].max(),
                                 "spread": rate["mean"].max() - rate["mean"].min(),
                                 "best_segment": rate["mean"].idxmax()})
            if rows:
                table = pd.DataFrame(rows).sort_values("spread", ascending=False).round(4)
                md += ["## Категории: разброс target между значениями", "",
                       table.head(15).to_markdown(index=False), ""]

    na = train.isna().mean().sort_values(ascending=False)
    na = na[na > 0].head(25)
    if not na.empty:
        fig, ax = plt.subplots(figsize=(7, 0.3 * len(na) + 1))
        ax.barh(na.index.astype(str)[::-1], na.values[::-1], color="#d8833b")
        ax.set_title("Доля пропусков")
        md += ["## Пропуски", "", f"![missing]({_fig(out_dir / 'missing.png', fig)})", ""]

    if test is not None:
        common = [c for c in train.columns if c in test.columns and c != target]
        d = drift(tr, test.sample(min(len(test), SAMPLE), random_state=0), common)
        if not d.empty:
            md += ["## Сдвиг train → test", "", d.head(15).to_markdown(index=False), ""]
            strong = d[d["drift"] > DRIFT_KS]["feature"].tolist()
            if strong:
                md += [f"> ⚠ Сильный сдвиг в {strong[:8]}: модель видит в test другие значения. "
                       "Проверь время (test позже train?) и не стоит ли убрать признак.", ""]

    time_cols = [time_col] if time_col else prof[prof["role"] == "datetime"]["name"].tolist()
    for col in time_cols[:2]:
        t = to_datetime(train[col])
        line = f"- `{col}`: train {t.min()} … {t.max()}"
        if test is not None and col in test.columns:
            tt = to_datetime(test[col])
            line += f"; test {tt.min()} … {tt.max()}"
            if pd.notna(tt.min()) and pd.notna(t.max()) and tt.min() >= t.max():
                line += " → **test целиком позже train: нужна временная валидация (cv = time)**"
        md += ["## Время", line]
        if target and task in {"binary", "regression"}:
            y_num = pd.to_numeric(train[target], errors="coerce") if task == "regression" else (
                train[target] == sorted(train[target].dropna().unique())[-1]).astype(int)
            monthly = y_num.groupby(t.dt.to_period("M")).mean()
            if len(monthly) > 1:
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(monthly.index.astype(str), monthly.values, marker="o", color="#3b6fd8")
                ax.set_title(f"{target} по месяцам ({col})")
                ax.tick_params(axis="x", rotation=60)
                md += ["", f"![time]({_fig(out_dir / f'time_{col}.png', fig)})"]
        md.append("")

    md.append(f"_EDA заняла {time.perf_counter() - started:.1f} с._")
    report = out_dir / "report.md"
    report.write_text("\n".join(md), encoding="utf-8")
    return report
