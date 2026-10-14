"""Dataset-specific features — the ML stream fills this during the datathon.

`add_features` receives a copy of train and, separately, of test, so it must not use the target or statistics of the
whole column (those belong in app/features.py, which fits on train only). Return the frame with new columns: the
generic plan then types and encodes them. Every feature needs a one-line "why" — it goes into the pitch.
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Examples for bank data (delete what does not fit):
    # df["debt_to_income"] = df["debt"] / df["income"].where(df["income"] > 0)  # платёжная нагрузка
    # df["utilization"] = df["balance"] / df["credit_limit"].where(df["credit_limit"] > 0)  # загрузка лимита
    # Event tables (transactions): app.features.aggregate_by(tx, key="client_id", time_col="tx_date")
    return df
