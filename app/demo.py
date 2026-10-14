"""Synthetic bank datasets for rehearsals and tests — the real data appears only at the opening.

`clients`: credit-default style table (binary), `income` (regression), `segment` (multiclass), plus an event table
of transactions to practise multi-table aggregation. Deterministic (seed), no real people.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CITIES = ["Астана", "Алматы", "Шымкент", "Караганда", "Актобе", "Атырау", "Павлодар", "Костанай"]
PRODUCTS = ["card", "cash_loan", "auto_loan", "mortgage", "deposit"]
CHANNELS = ["app", "branch", "partner", "web"]
MCC = ["grocery", "fuel", "restaurants", "travel", "electronics", "pharmacy", "transfer", "cash"]


def clients(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 72, n)
    income = np.round(rng.lognormal(12.6, 0.55, n), -3)  # tenge per month
    limit = np.round(income * rng.uniform(0.5, 4, n), -3)
    utilization = np.clip(rng.beta(2, 3, n) * 1.3, 0, 1.3)
    open_date = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1300, n), unit="D")
    city = rng.choice(CITIES, n, p=[0.22, 0.25, 0.12, 0.09, 0.08, 0.08, 0.08, 0.08])
    product = rng.choice(PRODUCTS, n)
    channel = rng.choice(CHANNELS, n, p=[0.55, 0.2, 0.15, 0.1])
    overdue = rng.poisson(0.4 + 1.5 * utilization, n)
    df = pd.DataFrame({
        "client_id": np.arange(100000, 100000 + n),
        "age": age,
        "income": income,
        "credit_limit": limit,
        "utilization": utilization.round(3),
        "overdue_count_12m": overdue,
        "city": city,
        "product": product,
        "channel": channel,
        "open_date": open_date.strftime("%Y-%m-%d"),
        "has_salary_project": rng.random(n) < 0.35,
    })
    df.loc[rng.random(n) < 0.08, "income"] = np.nan  # missing income, as in real exports
    logit = (-3.2 + 2.2 * utilization + 0.45 * overdue - 0.5 * df["has_salary_project"]
             + 0.6 * (channel == "partner") - 0.012 * (age - 40)
             + 0.4 * np.isin(city, ["Шымкент", "Актобе"]))
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    seg = np.where(income > 600_000, "premium", np.where(utilization > 0.8, "risk", "mass"))
    df["segment"] = np.where(rng.random(n) < 0.1, rng.choice(["premium", "risk", "mass"], n), seg)
    return df


def transactions(cl: pd.DataFrame, per_client: float = 12, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(per_client, len(cl))
    ids = np.repeat(cl["client_id"].to_numpy(), counts)
    n = len(ids)
    return pd.DataFrame({
        "client_id": ids,
        "tx_date": (pd.Timestamp("2026-04-01") + pd.to_timedelta(rng.integers(0, 180, n), unit="D")
                    ).strftime("%Y-%m-%d"),
        "amount": np.round(rng.lognormal(9, 1.1, n), 0),
        "mcc_group": rng.choice(MCC, n),
    })


def write(out_dir: Path, task: str = "binary", n: int = 6000) -> dict[str, Path]:
    """train.csv / test.csv (target column dropped) / sample_submission.csv / transactions.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = clients(n)
    target = {"binary": "default", "regression": "income", "multiclass": "segment"}[task]
    others = {"default", "segment"} - {target}
    df = df.drop(columns=list(others))
    if task == "regression":
        df = df[df["income"].notna()]
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:].drop(columns=[target])
    paths = {
        "train": out_dir / "train.csv",
        "test": out_dir / "test.csv",
        "sample_submission": out_dir / "sample_submission.csv",
        "transactions": out_dir / "transactions.csv",
    }
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    pd.DataFrame({"client_id": test["client_id"], target: 0}).to_csv(
        paths["sample_submission"], index=False)
    transactions(df).to_csv(paths["transactions"], index=False)
    return paths
