"""Model zoo behind one interface: fit(X, y, X_valid, y_valid) / predict(X) / importance().

Ladder from the cheapest to the strongest: dummy (reference) -> linear -> hgb (scikit-learn, works everywhere) ->
lgbm / catboost / xgb (used when importable: LightGBM and XGBoost need libomp on macOS — `brew install libomp`).
"""

import importlib
import inspect
from functools import cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from app.config import SEED

LADDER = ("dummy", "linear", "hgb", "lgbm", "catboost", "xgb")
LIBRARY = {"lgbm": "lightgbm", "catboost": "catboost", "xgb": "xgboost"}
EARLY_STOP = 100
N_ESTIMATORS = 3000


@cache
def available() -> dict[str, str]:
    """name -> "ok" or the import error (shown in the log, so a missing libomp is obvious)."""
    out = {name: "ok" for name in ("dummy", "linear", "hgb")}
    for name, module in LIBRARY.items():
        try:
            importlib.import_module(module)
            out[name] = "ok"
        except Exception as exc:  # ImportError or OSError: libomp.dylib not found
            out[name] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return out


def candidates(requested: tuple[str, ...] = ()) -> list[str]:
    names = requested or LADDER
    unknown = [n for n in names if n not in LADDER]
    if unknown:
        raise ValueError(f"неизвестные модели {unknown}; есть: {LADDER}")
    return list(names)


def _as_str(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype("object").where(X.notna(), "NA").astype(str)


class Adapter:
    def __init__(self, name: str, task: str, n_classes: int, categorical: list[str],
                 gpu: bool = False, seed: int = SEED, params: dict | None = None):
        self.name, self.task, self.n_classes = name, task, n_classes
        self.categorical, self.gpu, self.seed = categorical, gpu, seed
        self.params = dict(params or {})  # tuned values from params.toml / optuna
        self.model = None
        self.feature_names: list[str] = []
        self.best_iteration: int | None = None

    @property
    def classification(self) -> bool:
        return self.task in {"binary", "multiclass"}

    def _build(self, X: pd.DataFrame):
        clf, cats = self.classification, [c for c in self.categorical if c in X.columns]
        if self.name == "dummy":
            return DummyClassifier(strategy="prior") if clf else DummyRegressor()
        if self.name == "linear":
            nums = [c for c in X.columns if c not in cats]
            pre = ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
                ("cat", make_pipeline(
                    FunctionTransformer(_as_str),
                    OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                  max_categories=30, sparse_output=False)), cats),
            ])
            head = LogisticRegression(max_iter=2000) if clf else Ridge(alpha=1.0)
            return make_pipeline(pre, head)
        if self.name == "hgb":
            cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
            return cls(learning_rate=0.06, max_iter=800, early_stopping=True,
                       validation_fraction=0.1, n_iter_no_change=50,
                       categorical_features="from_dtype", random_state=self.seed)
        if self.name == "lgbm":
            import lightgbm as lgb

            cls = lgb.LGBMClassifier if clf else lgb.LGBMRegressor
            return cls(n_estimators=N_ESTIMATORS, learning_rate=0.03, num_leaves=63,
                       min_child_samples=20, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, reg_lambda=1.0, random_state=self.seed, verbose=-1)
        if self.name == "catboost":
            import catboost as cb

            cls = cb.CatBoostClassifier if clf else cb.CatBoostRegressor
            loss = {"binary": "Logloss", "multiclass": "MultiClass"}.get(self.task, "RMSE")
            return cls(iterations=N_ESTIMATORS, learning_rate=0.05, depth=6, loss_function=loss,
                       random_seed=self.seed, verbose=False, allow_writing_files=False,
                       task_type="GPU" if self.gpu else "CPU")
        if self.name == "xgb":
            import xgboost as xgb

            cls = xgb.XGBClassifier if clf else xgb.XGBRegressor
            return cls(n_estimators=N_ESTIMATORS, learning_rate=0.03, max_depth=6, subsample=0.8,
                       colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                       max_cat_to_onehot=1, early_stopping_rounds=EARLY_STOP,
                       device="cuda" if self.gpu else "cpu", random_state=self.seed)
        raise ValueError(self.name)

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.name != "catboost":
            return X
        X = X.copy()  # CatBoost wants strings in categorical columns and no NaN there
        for c in self.categorical:
            if c in X.columns:
                X[c] = X[c].astype("object").where(X[c].notna(), "NA").astype(str)
        return X

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_valid: pd.DataFrame | None = None,
            y_valid: np.ndarray | None = None) -> "Adapter":
        self.feature_names = list(X.columns)
        self.model = self._build(X)
        if self.params and self.name not in {"dummy", "linear"}:
            self.model.set_params(**self.params)
        Xf, has_valid = self._frame(X), X_valid is not None and y_valid is not None
        if self.name == "lgbm":
            import lightgbm as lgb

            kwargs = {}
            if has_valid:  # LightGBM >= 4.7 takes eval_X / eval_y, older versions eval_set
                new_api = "eval_X" in inspect.signature(self.model.fit).parameters
                kwargs = ({"eval_X": (X_valid,), "eval_y": (y_valid,)} if new_api
                          else {"eval_set": [(X_valid, y_valid)]})
                kwargs["callbacks"] = [lgb.early_stopping(EARLY_STOP, verbose=False)]
            self.model.fit(Xf, y, **kwargs)
            self.best_iteration = getattr(self.model, "best_iteration_", None)
        elif self.name == "catboost":
            cats = [c for c in self.categorical if c in X.columns]
            kwargs = {"cat_features": cats}
            if has_valid:
                kwargs |= {"eval_set": (self._frame(X_valid), y_valid),
                           "early_stopping_rounds": EARLY_STOP}
            self.model.fit(Xf, y, **kwargs)
            self.best_iteration = self.model.get_best_iteration()
        elif self.name == "xgb":
            if not has_valid:  # no early stopping without a validation set
                self.model.set_params(early_stopping_rounds=None, n_estimators=600)
            kwargs = {"eval_set": [(X_valid, y_valid)], "verbose": False} if has_valid else {}
            self.model.fit(Xf, y, **kwargs)
            self.best_iteration = getattr(self.model, "best_iteration", None)
        else:
            self.model.fit(Xf, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = self._frame(X[self.feature_names])
        if not self.classification:
            return np.asarray(self.model.predict(Xf), dtype=float).ravel()
        proba = np.asarray(self.model.predict_proba(Xf), dtype=float)
        if self.task == "binary":
            return proba[:, 1]
        if proba.shape[1] < self.n_classes:  # a fold that missed a rare class
            full = np.zeros((len(X), self.n_classes))
            full[:, self.model.classes_.astype(int)] = proba
            proba = full
        return proba

    def importance(self) -> pd.Series | None:
        values = getattr(self.model, "feature_importances_", None)
        if self.name == "catboost":
            values = self.model.get_feature_importance()
        if values is None:
            return None
        s = pd.Series(np.asarray(values, dtype=float), index=self.feature_names)
        return s / s.sum() if s.sum() > 0 else s
