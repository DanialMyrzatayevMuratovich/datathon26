import pandas as pd

from app import demo
from app.config import TaskSpec
from app.features import aggregate_by, apply_plan, fit_plan


def test_plan_is_fitted_on_train_and_replayed(bank):
    train, test = bank
    plan = fit_plan(train, TaskSpec(drop=("channel",)), "default", "client_id")
    assert plan.dropped["client_id"] == "id"
    assert plan.dropped["channel"].startswith("в drop")
    assert "open_date" in plan.datetime
    X, X_test = apply_plan(plan, train), apply_plan(plan, test)
    assert list(X.columns) == list(X_test.columns) == plan.columns
    assert "default" not in X.columns
    assert isinstance(X["city"].dtype, pd.CategoricalDtype)


def test_unseen_category_and_missing_column_become_nan(bank):
    train, _ = bank
    plan = fit_plan(train, TaskSpec(), "default", "client_id")
    row = pd.DataFrame([{"city": "Марс", "age": 30}])  # API row: one unknown city, most columns absent
    X = apply_plan(plan, row)
    assert list(X.columns) == plan.columns
    assert pd.isna(X.loc[0, "city"])
    assert X.loc[0, "age"] == 30


def test_aggregate_by_events():
    cl = demo.clients(50)
    tx = demo.transactions(cl, per_client=5)
    agg = aggregate_by(tx, "client_id", time_col="tx_date")
    assert agg["client_id"].is_unique
    assert {"n_events", "amount__sum", "days_since_last", "active_days"} <= set(agg.columns)
    assert any(c.startswith("mcc_group==") for c in agg.columns)
