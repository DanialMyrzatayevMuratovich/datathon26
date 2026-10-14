"""Paths and the task contract.

The contract lives in task.toml (the lead fills it at kickoff); CLI flags override it for one run.
Both teammates read the same file, so their CV scores are comparable.
"""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS = ROOT / "outputs"
RUNS_DIR = ROOT / "runs"
TASK_FILE = ROOT / "task.toml"
PARAMS_FILE = ROOT / "params.toml"  # tuned hyper-parameters per model (app.cli tune writes it)
EXPERIMENTS_MD = ROOT / "docs" / "EXPERIMENTS.md"

SEED = 42
TASK_TYPES = ("auto", "binary", "multiclass", "regression")
CV_KINDS = ("auto", "kfold", "stratified", "group", "time")


@dataclass(frozen=True)
class TaskSpec:
    train: str = ""  # path relative to the repo root or absolute
    test: str = ""  # optional: no test -> CV only, no submission
    sample_submission: str = ""  # optional: column names and order of the submission
    target: str = ""  # empty -> the column that is in train but not in test
    id: str = ""  # empty -> guessed (unique column named like id)
    task: str = "auto"  # auto | binary | multiclass | regression
    metric: str = "auto"  # see app.evaluate.METRICS
    cv: str = "auto"  # auto | kfold | stratified | group | time
    folds: int = 5
    group: str = ""  # column for GroupKFold (client_id: one client never in train and valid at once)
    time: str = ""  # column for a time split (no future in train)
    drop: tuple[str, ...] = ()  # columns never used as features (leaks, ids, free text)
    submission_target: str = ""  # target column name in the submission (default: target)
    predict_proba: bool = True  # binary: submit probabilities (True) or 0/1 labels
    threshold: float = 0.5  # binary labels: probability cut-off
    models: tuple[str, ...] = ()  # empty -> every available model
    budget_min: float = 20.0  # soft time budget for the model zoo
    sample: int = 0  # subsample train rows for speed (0 = all)
    gpu: bool = False  # CatBoost / XGBoost on CUDA (Colab)
    shap: bool = False  # SHAP summary plot for the best tree model (slower)


def load_spec(path: Path = TASK_FILE, **overrides) -> TaskSpec:
    """task.toml [task] table + non-empty overrides (CLI passes None for flags that were not given)."""
    data: dict = {}
    if path.exists():
        with path.open("rb") as f:
            data = tomllib.load(f).get("task", {})
    known = {f.name for f in fields(TaskSpec)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"task.toml: неизвестные ключи {unknown}; допустимые: {sorted(known)}")
    merged = {**data, **{k: v for k, v in overrides.items() if v is not None}}
    for key in ("drop", "models"):
        value = merged.get(key)
        if isinstance(value, str):
            merged[key] = tuple(x.strip() for x in value.split(",") if x.strip())
        elif value is not None:
            merged[key] = tuple(value)
    spec = TaskSpec(**merged)
    if spec.task not in TASK_TYPES:
        raise ValueError(f"task = {spec.task!r}; допустимо: {TASK_TYPES}")
    if spec.cv not in CV_KINDS:
        raise ValueError(f"cv = {spec.cv!r}; допустимо: {CV_KINDS}")
    return spec


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def find_raw(patterns: tuple[str, ...]) -> Path | None:
    """First file in data/raw matching one of the glob patterns (train*.csv, sample_submission*, ...)."""
    if not DATA_RAW.exists():
        return None
    for pattern in patterns:
        hits = sorted(p for p in DATA_RAW.rglob(pattern) if p.is_file())
        if hits:
            return hits[0]
    return None


def load_params(path: Path | None = None) -> dict[str, dict]:
    """{model: {param: value}} from params.toml; applied on top of the defaults in app/models.py."""
    path = path or PARAMS_FILE
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return {k: dict(v) for k, v in tomllib.load(f).items() if isinstance(v, dict)}


def save_params(model: str, params: dict, note: str = "", path: Path | None = None) -> Path:
    """Replace one [model] table in params.toml, keep the others (values: int, float, str, bool)."""
    path = path or PARAMS_FILE
    tables = load_params(path)
    tables[model] = params

    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return '"' + v.replace('"', '\\"') + '"'
        return repr(float(v)) if isinstance(v, float) else str(v)

    lines = ["# Гиперпараметры поверх значений по умолчанию из app/models.py.",
             "# Пишет `uv run python -m app.cli tune`; убери таблицу, чтобы вернуть умолчания.", ""]
    for name, values in tables.items():
        if name == model and note:
            lines.append(f"# {note}")
        lines.append(f"[{name}]")
        lines += [f"{k} = {fmt(v)}" for k, v in values.items()]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
