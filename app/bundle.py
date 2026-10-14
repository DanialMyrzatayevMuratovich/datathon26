"""Everything needed to predict new rows: the feature plan, fold models and blend weights (models/<run_id>.joblib)."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app import custom_features
from app.config import MODELS_DIR
from app.features import FeaturePlan, apply_plan
from app.models import Adapter


@dataclass
class Bundle:
    run_id: str
    plan: FeaturePlan
    adapters: dict[str, list[Adapter]]  # model -> its fold models
    weights: dict[str, float]  # model -> blend weight (one model -> 1.0)
    task: str
    target: str
    id_col: str | None
    classes: list | None  # original labels, index = encoded class
    predict_proba: bool
    threshold: float
    metric: str
    score: float

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        X = apply_plan(self.plan, custom_features.add_features(df.copy()))
        return sum(w * np.mean([a.predict(X) for a in self.adapters[m]], axis=0)
                   for m, w in self.weights.items())

    def labels(self, raw: np.ndarray) -> np.ndarray:
        if self.task == "binary":
            return np.asarray(self.classes, dtype=object)[(raw >= self.threshold).astype(int)]
        if self.task == "multiclass":
            return np.asarray(self.classes, dtype=object)[raw.argmax(axis=1)]
        return raw

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """What goes into the submission: probability (binary, predict_proba), label or value."""
        raw = self.raw(df)
        if self.task == "binary" and self.predict_proba:
            return raw
        return self.labels(raw)

    def save(self, models_dir: Path = MODELS_DIR) -> Path:
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f"{self.run_id}.joblib"
        joblib.dump(self, path, compress=3)
        return path


def best_pointer(models_dir: Path = MODELS_DIR) -> dict | None:
    path = models_dir / "best.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load(ref: str | None = None, models_dir: Path = MODELS_DIR) -> Bundle:
    """ref: run_id, path to .joblib, or None for models/best.json."""
    if ref and Path(ref).suffix == ".joblib":
        return joblib.load(ref)
    if not ref:
        pointer = best_pointer(models_dir)
        if not pointer:
            raise FileNotFoundError("нет models/best.json — сначала uv run python -m app.cli auto")
        ref = pointer["run_id"]
    path = models_dir / f"{ref}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"нет {path} (модели не в git: обучи заново или возьми у напарника)")
    return joblib.load(path)
