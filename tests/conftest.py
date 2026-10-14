import numpy as np
import pandas as pd
import pytest

from app import demo
from app.config import TaskSpec

FAST = ("dummy", "linear", "hgb")


@pytest.fixture
def root(tmp_path):
    """Isolated project root: runs/, models/, outputs/, docs/ of a test never touch the repo."""
    return tmp_path


@pytest.fixture
def bank():
    df = demo.clients(1500, seed=3).drop(columns=["segment"])
    return df.iloc[:1200].reset_index(drop=True), df.iloc[1200:].drop(columns=["default"]).reset_index(
        drop=True)


@pytest.fixture
def fast_spec():
    return TaskSpec(models=FAST, folds=3, budget_min=5)


@pytest.fixture
def regression_frame():
    rng = np.random.default_rng(1)
    n = 800
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(size=n),
                       "cat": rng.choice(list("abc"), n)})
    df["y"] = 3 * df.x1 + 2 * (df.cat == "b") + rng.normal(scale=0.3, size=n)
    return df
