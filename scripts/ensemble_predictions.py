#!/usr/bin/env python3
"""Compute an arithmetic mean of saved predictions across matched seeds.

Inputs must have all compounds for all expected seeds. It never trains models.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from scripts.io_utils import read_predictions
except ModuleNotFoundError:
    from io_utils import read_predictions


def average_seeds(df: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    expected = set(seeds)
    if not expected or len(expected) != len(seeds):
        raise ValueError("Provide unique expected seeds")
    observed = set(df.seed)
    if observed != expected:
        raise ValueError(f"Seed mismatch: expected {sorted(expected)}, observed {sorted(observed)}")
    group = df.groupby(["compound_id", "endpoint"], sort=True)
    for key, subset in group:
        if set(subset.seed) != expected or len(subset) != len(expected):
            raise ValueError(f"Incomplete or duplicated seed predictions for {key}")
        if not np.allclose(subset.y_true, subset.y_true.iloc[0], atol=1e-12, rtol=0):
            raise ValueError(f"Inconsistent ground truth across seeds for {key}")
    result = group.agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")).reset_index()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--expected-seeds", default="42,43,44,45,46,47,48,49,50,51")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.expected_seeds.split(",")]
    predictions = read_predictions(args.input, with_seed=True)
    averaged = average_seeds(predictions, seeds)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    averaged.to_csv(dest, index=False)
    print(f"Wrote {len(averaged)} averaged predictions to {dest}")


if __name__ == "__main__":
    main()
