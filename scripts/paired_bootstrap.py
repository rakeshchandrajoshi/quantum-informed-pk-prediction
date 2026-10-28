#!/usr/bin/env python3
"""Paired compound-level percentile-bootstrap differences between two models.

Compares matched per-compound predictions, not aggregate manuscript metrics.
The CI is exploratory and uncorrected for multiple metric comparisons.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from scripts.io_utils import read_predictions
    from scripts.metrics import compute_metrics
except ModuleNotFoundError:
    from io_utils import read_predictions
    from metrics import compute_metrics


def align(proposed: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keys = ["compound_id", "endpoint"]
    keyset = lambda df: set(map(tuple, df[keys].to_numpy()))
    if keyset(proposed) != keyset(reference):
        only_a = len(keyset(proposed) - keyset(reference))
        only_b = len(keyset(reference) - keyset(proposed))
        raise ValueError(f"Model compound sets differ: proposed-only={only_a}, reference-only={only_b}. Refusing silent intersection.")
    merged = proposed.merge(reference, on=keys, suffixes=("_proposed", "_reference"), validate="one_to_one", sort=True)
    if not np.allclose(merged.y_true_proposed, merged.y_true_reference, atol=1e-8, rtol=1e-8):
        raise ValueError("Reference values differ between proposed and reference predictions")
    return merged


def paired_bootstrap_metrics(df: pd.DataFrame, endpoint: str, scale: str, resamples: int, seed: int) -> list[dict]:
    if resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are recommended")
    y = df.y_true_proposed.to_numpy(float)
    a = df.y_pred_proposed.to_numpy(float)
    b = df.y_pred_reference.to_numpy(float)
    metric_a = compute_metrics(y, a, endpoint, scale)
    metric_b = compute_metrics(y, b, endpoint, scale)
    n = len(y)
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in metric_a}
    # Bootstrap identical resampled compound indices for both models.
    for _ in range(resamples):
        indices = rng.integers(0, n, size=n)
        ma = compute_metrics(y[indices], a[indices], endpoint, scale)
        mb = compute_metrics(y[indices], b[indices], endpoint, scale)
        for metric in metric_a:
            v = ma[metric] - mb[metric]
            if np.isfinite(v):
                diffs[metric].append(v)
    rows = []
    for metric in metric_a:
        d = np.asarray(diffs[metric], dtype=float)
        if len(d) < 0.9 * resamples:
            raise ValueError(f"Too many undefined bootstrap values for {endpoint} {metric}")
        low, high = np.percentile(d, [2.5, 97.5])
        rows.append({
            "endpoint": endpoint, "metric": metric, "n": n,
            "proposed": metric_a[metric], "reference": metric_b[metric],
            "difference_proposed_minus_reference": metric_a[metric] - metric_b[metric],
            "CI95_low": low, "CI95_high": high,
            "CI_excludes_zero": bool(low > 0 or high < 0),
            "bootstrap_resamples_valid": len(d),
            "note": "Exploratory unadjusted paired bootstrap; not a multiple-testing-adjusted p-value",
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposed", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", choices=["log10", "linear"], default="log10")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    proposed = read_predictions(args.proposed)
    reference = read_predictions(args.reference)
    both = align(proposed, reference)
    rows = []
    for endpoint, subset in both.groupby("endpoint", sort=True):
        rows.extend(paired_bootstrap_metrics(subset, endpoint, args.scale, args.resamples, args.seed))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} paired comparisons to {out}")


if __name__ == "__main__":
    main()
