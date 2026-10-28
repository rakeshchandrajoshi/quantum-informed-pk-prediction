"""Endpoint-specific metrics for pKa and human pharmacokinetic predictions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PKA_ENDPOINTS = {"pka_acidic", "pka_basic"}
PK_ENDPOINTS = {"CL", "VDss", "Fu"}
ALL_ENDPOINTS = PKA_ENDPOINTS | PK_ENDPOINTS


def _as_numeric(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) < 2:
        raise ValueError("Need at least two matched, one-dimensional observations")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("All true and predicted values must be finite")
    return y, p


def compute_metrics(y_true, y_pred, endpoint: str, scale: str = "log10") -> dict:
    """Return endpoint-specific metrics.

    pKa: y_true/y_pred in pKa units; negatives are permitted.
    CL/VDss/Fu: by default input in log10 units; `scale=linear` accepts
    strictly positive real-scale values and converts them before scoring.
    """
    if endpoint not in ALL_ENDPOINTS:
        raise ValueError(f"Unrecognized endpoint {endpoint!r}; expected {sorted(ALL_ENDPOINTS)}")
    if scale not in {"log10", "linear"}:
        raise ValueError("scale must be 'log10' or 'linear'")
    y, p = _as_numeric(y_true, y_pred)
    if endpoint in PK_ENDPOINTS:
        if scale == "linear":
            if np.any(y <= 0) or np.any(p <= 0):
                raise ValueError("PK endpoint linear-scale values must be >0")
            if endpoint == "Fu" and (np.any(y > 1) or np.any(p > 1)):
                raise ValueError("Fu fractions supplied in linear scale must be in (0, 1]")
            y, p = np.log10(y), np.log10(p)
        residual = p - y
        return {
            "R2": float(r2_score(y, p)),
            "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "GMFE": float(10 ** np.mean(np.abs(residual))),
            "W2F": float(np.mean(np.abs(residual) <= np.log10(2) + 1e-12)),
        }
    return {
        "R2": float(r2_score(y, p)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "W1": float(np.mean(np.abs(p - y) <= 1 + 1e-12)),
    }
