"""Validate saved-per-compound prediction files before scoring."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scripts.metrics import ALL_ENDPOINTS
except ModuleNotFoundError:  # direct invocation from within scripts directory
    from metrics import ALL_ENDPOINTS

REQUIRED = {"compound_id", "endpoint", "y_true", "y_pred"}


def read_predictions(path: str | Path, with_seed: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"compound_id": str, "endpoint": str})
    req = REQUIRED | ({"seed"} if with_seed else set())
    missing = sorted(req - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df = df.copy()
    if df[list(req)].isna().any().any():
        raise ValueError(f"Missing values in required columns of {path}")
    if not set(df.endpoint).issubset(ALL_ENDPOINTS):
        raise ValueError(f"Unsupported endpoints: {sorted(set(df.endpoint) - ALL_ENDPOINTS)}")
    for field in ("y_true", "y_pred"):
        df[field] = pd.to_numeric(df[field], errors="raise")
        if not np.isfinite(df[field]).all():
            raise ValueError(f"Non-finite entries in {field} in {path}")
    keys = ["compound_id", "endpoint"] + (["seed"] if with_seed else [])
    if with_seed:
        df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    if df.duplicated(keys).any():
        raise ValueError(f"Duplicate rows by {keys} in {path}")
    return df
