#!/usr/bin/env python3
"""Score saved predictions; this program does not perform model inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
try:
    from scripts.metrics import compute_metrics
    from scripts.io_utils import read_predictions
except ModuleNotFoundError:
    from metrics import compute_metrics
    from io_utils import read_predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CSV: compound_id,endpoint,y_true,y_pred")
    ap.add_argument("--output", required=True, help="CSV destination")
    ap.add_argument("--scale", choices=["log10", "linear"], default="log10", help="Input scale for PK endpoints; pKa is always in pKa units")
    args = ap.parse_args()
    df = read_predictions(args.input)
    rows = []
    for endpoint, subset in df.groupby("endpoint", sort=True):
        metrics = compute_metrics(subset.y_true, subset.y_pred, endpoint, scale=args.scale)
        rows.append({"endpoint": endpoint, "n": len(subset), **metrics})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} endpoint metrics to {out}")


if __name__ == "__main__":
    main()
