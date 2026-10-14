"""Generic, leak-safe feature preparation.

`fit_plan` looks at train only and records every decision (types, categories, frequency maps, dropped columns
with a reason); `apply_plan` replays it on train, test and single API rows, so all three get identical columns.
Dataset-specific features go to app/custom_features.py, multi-table aggregates use `aggregate_by`.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.config import TaskSpec
from app.data import column_role, is_text, to_datetime

MAX_NULL_SHARE = 0.98
TOP_CATEGORY_COUNTS = 10


@dataclass
class FeaturePlan:
    numeric: list[str] = field(default_factory=list)
    categorical: dict[str, list[str]] = field(default_factory=dict)  # column -> categories seen in train
    frequency: dict[str, dict[str, float]] = field(default_factory=dict)  # high-cardinality column
    datetime: dict[str, dict] = field(default_factory=dict)  # column -> {"origin": iso | None, "hour": bool}
    text: list[str] = field(default_factory=list)  # free text -> length / word count
    dropped: dict[str, str] = field(default_factory=dict)  # column -> reason
    columns: list[str] = field(default_factory=list)  # final feature order

    @property
    def categorical_features(self) -> list[str]:
        return list(self.categorical)


def _naive(dt: pd.Series) -> pd.Series:
    return dt.dt.tz_convert(None) if getattr(dt.dt, "tz", None) is not None else dt


def fit_plan(train: pd.DataFrame, spec: TaskSpec, target: str, id_col: str | None) -> FeaturePlan:
    plan = FeaturePlan()
    for col in spec.drop:
        if col in train.columns:
            plan.dropped[col] = "в drop (task.toml)"
    if id_col:
        plan.dropped[id_col] = "id"
    for col in train.columns:
        if col == target or col in plan.dropped:
            continue
        s = train[col]
        if s.isna().mean() > MAX_NULL_SHARE:
            plan.dropped[col] = f"пропусков {s.isna().mean():.0%}"
            continue
        role, _ = column_role(s, len(train))
        if role == "constant":
            plan.dropped[col] = "константа"
        elif role == "id":
            plan.dropped[col] = "похоже на id"
        elif role == "datetime":
            dt = _naive(to_datetime(s))
            origin = dt.min()
            plan.datetime[col] = {
                "origin": None if pd.isna(origin) else origin.isoformat(),
                "hour": bool((dt.dt.hour.fillna(0) != 0).any()),
            }
        elif role == "text":
            plan.text.append(col)
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            plan.numeric.append(col)
        elif role == "high_card":
            plan.frequency[col] = s.astype(str).value_counts(normalize=True).to_dict()
        else:
            plan.categorical[col] = sorted(s.dropna().astype(str).unique().tolist())
    plan.columns = list(_build(plan, train.head(50)).columns)
    return plan


def _build(plan: FeaturePlan, df: pd.DataFrame) -> pd.DataFrame:
    def col_or_nan(col: str) -> pd.Series:
        return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)

    out: dict[str, pd.Series] = {}
    for col in plan.numeric:
        s = col_or_nan(col)
        out[col] = pd.to_numeric(s.astype(float) if pd.api.types.is_bool_dtype(s) else s,
                                 errors="coerce")
    for col, cats in plan.categorical.items():
        s = col_or_nan(col)
        values = s.astype(str).where(s.notna())
        out[col] = pd.Series(pd.Categorical(values, categories=cats), index=df.index)
    for col, freq in plan.frequency.items():
        out[f"{col}__freq"] = col_or_nan(col).astype(str).map(freq).astype(float).fillna(0.0)
    for col, meta in plan.datetime.items():
        dt = _naive(to_datetime(col_or_nan(col)))
        out[f"{col}__year"] = dt.dt.year
        out[f"{col}__month"] = dt.dt.month
        out[f"{col}__day"] = dt.dt.day
        out[f"{col}__dow"] = dt.dt.dayofweek
        out[f"{col}__doy"] = dt.dt.dayofyear
        if meta["hour"]:
            out[f"{col}__hour"] = dt.dt.hour
        if meta["origin"]:
            out[f"{col}__days"] = (dt - pd.Timestamp(meta["origin"])).dt.total_seconds() / 86400
    for col in plan.text:
        s = col_or_nan(col).fillna("").astype(str)
        out[f"{col}__len"] = s.str.len()
        out[f"{col}__words"] = s.str.split().str.len()
    return pd.DataFrame(out, index=df.index)


def apply_plan(plan: FeaturePlan, df: pd.DataFrame) -> pd.DataFrame:
    X = _build(plan, df)
    for col in plan.columns:
        if col not in X.columns:
            X[col] = np.nan
    return X[plan.columns]


def aggregate_by(df: pd.DataFrame, key: str, time_col: str | None = None,
                 ref_time: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per `key` from an event table (transactions, logins, payments): numeric stats, shares of the
    most frequent categories, recency and tenure. Join the result to the client table on `key`."""
    g = df.groupby(key)
    parts = [g.size().rename("n_events")]
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != key]
    if numeric:
        stats = g[numeric].agg(["mean", "sum", "min", "max", "std"])
        stats.columns = [f"{c}__{a}" for c, a in stats.columns]
        parts.append(stats)
    for col in df.columns:
        if col in (key, time_col) or not is_text(df[col]):
            continue
        top = df[col].value_counts().head(TOP_CATEGORY_COUNTS).index
        if df[col].nunique() > 50:
            parts.append(g[col].nunique().rename(f"{col}__nunique"))
            continue
        shares = pd.crosstab(df[key], df[col].where(df[col].isin(top)), normalize="index")
        shares.columns = [f"{col}=={v}__share" for v in shares.columns]
        parts.append(shares)
    if time_col:
        t = _naive(to_datetime(df[time_col]))
        ref = pd.Timestamp(ref_time) if ref_time is not None else t.max()
        tg = t.groupby(df[key])
        parts.append(((ref - tg.max()).dt.total_seconds() / 86400).rename("days_since_last"))
        parts.append(((tg.max() - tg.min()).dt.total_seconds() / 86400).rename("active_days"))
    return pd.concat(parts, axis=1).reset_index()
