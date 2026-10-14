"""Load whatever tabular file the organizers hand out and describe its columns without reading rows by eye.

Local bank exports are often `;`-separated, cp1251-encoded and use a decimal comma; all three are handled here
so nobody loses the first hour of the datathon to parsing.
"""

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ENCODINGS = ("utf-8-sig", "cp1251", "latin-1")
ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|guid|^index$)", re.IGNORECASE)
DATE_NAME = re.compile(r"(date|time|dt$|_dt|month|дата|время|период)", re.IGNORECASE)
DECIMAL_COMMA = re.compile(r"^-?\d+(,\d+)?$")
HIGH_CARDINALITY = 200  # above this a string column is frequency-encoded, not one-hot / native category


def _sniff(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()[:65536]
    for enc in ENCODINGS:
        try:
            sample = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # latin-1 never fails, kept for type checkers
        enc, sample = "latin-1", raw.decode("latin-1")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    return enc, sep


def is_text(s: pd.Series) -> bool:
    """String-like column: object in pandas 2, `str` dtype in pandas 3."""
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _fix_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]
        if not is_text(s):
            continue
        values = s.dropna().astype(str).str.strip()
        if len(values) and values.str.match(DECIMAL_COMMA).mean() > 0.95:
            df[col] = pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df


def load_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"нет файла {path}")
    suffixes = [s.lower() for s in path.suffixes]
    kind = suffixes[-2] if suffixes and suffixes[-1] in {".gz", ".zip", ".bz2", ".xz"} else (
        suffixes[-1] if suffixes else ""
    )
    if kind in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        df = df.head(nrows) if nrows else df
    elif kind in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path, nrows=nrows)
    elif kind in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True, nrows=nrows)
    elif kind == ".json":
        df = pd.read_json(path)
    elif kind == ".feather":
        df = pd.read_feather(path)
    else:  # csv / tsv / txt and their compressed variants
        if kind in {".csv", ".tsv", ".txt"} and suffixes[-1] == kind:
            enc, sep = _sniff(path)
        else:
            enc, sep = "utf-8", "\t" if kind == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, encoding=enc, nrows=nrows, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return _fix_decimal_commas(df)


def _looks_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not is_text(s):
        return False
    values = s.dropna().astype(str).head(500)
    if values.empty or not values.str.contains(r"[-/.:]").mean() > 0.9:
        return False
    if values.str.len().median() < 6:
        return False
    parsed = pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=True)
    return parsed.notna().mean() > 0.9


def to_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True)


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    role: str  # target id constant datetime binary numeric categorical high_card text
    n_unique: int
    null_share: float
    examples: str
    note: str = ""


def column_role(s: pd.Series, n_rows: int) -> tuple[str, str]:
    n_unique = s.nunique(dropna=True)
    if n_unique <= 1:
        return "constant", "одно значение"
    if ID_NAME.search(s.name or "") and n_unique > 0.95 * s.notna().sum():
        return "id", "уникален и назван как id"
    if _looks_datetime(s) or (DATE_NAME.search(str(s.name)) and _looks_datetime(s.astype(str))):
        return "datetime", ""
    if n_unique == 2:
        return "binary", ""
    if pd.api.types.is_numeric_dtype(s):
        return "numeric", ""
    if is_text(s) or isinstance(s.dtype, pd.CategoricalDtype):
        mean_len = s.dropna().astype(str).str.len().mean()
        if mean_len > 40 and n_unique > 0.5 * n_rows:
            return "text", f"средняя длина {mean_len:.0f} символов"
        if n_unique > HIGH_CARDINALITY:
            return "high_card", f"{n_unique} значений → частотное кодирование"
        return "categorical", ""
    return "numeric", ""


def profile(df: pd.DataFrame, target: str = "") -> list[ColumnInfo]:
    out = []
    for col in df.columns:
        s = df[col]
        role, note = ("target", "") if col == target else column_role(s, len(df))
        examples = ", ".join(map(str, s.dropna().unique()[:3]))[:60]
        out.append(ColumnInfo(
            name=col,
            dtype=str(s.dtype),
            role=role,
            n_unique=int(s.nunique(dropna=True)),
            null_share=round(float(s.isna().mean()), 4),
            examples=examples,
            note=note,
        ))
    return out


def profile_frame(df: pd.DataFrame, target: str = "") -> pd.DataFrame:
    return pd.DataFrame([asdict(c) for c in profile(df, target)])


def guess_target(train: pd.DataFrame, test: pd.DataFrame | None) -> str | None:
    """Competition layout: the target is the one column present in train but absent from test."""
    if test is None:
        return None
    extra = [c for c in train.columns if c not in set(test.columns)]
    return extra[0] if len(extra) == 1 else None


def guess_id(train: pd.DataFrame, test: pd.DataFrame | None) -> str | None:
    for col in train.columns:
        if ID_NAME.search(col) and train[col].is_unique and (test is None or col in test.columns):
            return col
    return None


def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 2**20)


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Halve memory of wide numeric frames: float64 -> float32, int64 -> smallest int."""
    for col in df.select_dtypes(include=[np.floating]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=[np.integer]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df
