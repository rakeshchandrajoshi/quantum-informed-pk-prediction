"""Unit tests on synthetic values only; no benchmark results are reproduced."""
import numpy as np
import pandas as pd
import pytest

from scripts.metrics import compute_metrics
from scripts.ensemble_predictions import average_seeds
from scripts.paired_bootstrap import align, paired_bootstrap_metrics


def test_perfect_pka():
    result = compute_metrics([7.0, 8.0, 9.0], [7.0, 8.0, 9.0], "pka_basic")
    assert result["MAE"] == 0
    assert result["RMSE"] == 0
    assert result["W1"] == 1.0
    assert result["R2"] == 1.0


def test_pk_fold_error_log_scale():
    result = compute_metrics([-1, 0, 1], [-1 + np.log10(2), 0, 1], "CL")
    assert result["MAE"] == pytest.approx(np.log10(2) / 3)
    assert result["GMFE"] == pytest.approx(2 ** (1/3))
    assert result["W2F"] == 1.0


def test_pk_linear_matches_log10():
    y = np.array([0.1, 1.0, 10.0]); p = np.array([0.2, 0.5, 10.0])
    a = compute_metrics(y, p, "VDss", scale="linear")
    b = compute_metrics(np.log10(y), np.log10(p), "VDss")
    for key in a:
        assert a[key] == pytest.approx(b[key])


def test_fu_invalid_linear():
    with pytest.raises(ValueError):
        compute_metrics([0.1, 1.2], [0.2, 0.8], "Fu", scale="linear")


def test_ensemble_seed_coverage_and_averaging():
    df = pd.DataFrame({
        "compound_id": ["m1", "m1", "m2", "m2"],
        "endpoint": ["pka_basic"]*4,
        "seed": [42,43,42,43],
        "y_true": [7,7,8,8],
        "y_pred": [6,8,7,9],
    })
    out = average_seeds(df, [42,43])
    assert list(out.y_pred) == [7,8]
    with pytest.raises(ValueError):
        average_seeds(df.query("not (compound_id == 'm1' and seed == 42)"), [42,43])


def test_paired_bootstrap_superior_predictions():
    ids = [f"m{i}" for i in range(30)]
    y = np.arange(30, dtype=float)
    proposed = pd.DataFrame({"compound_id": ids, "endpoint": "pka_acidic", "y_true": y, "y_pred": y})
    benchmark = pd.DataFrame({"compound_id": ids, "endpoint": "pka_acidic", "y_true": y, "y_pred": y + 1.5})
    aligned = align(proposed, benchmark)
    scores = paired_bootstrap_metrics(aligned, "pka_acidic", "log10", 200, 123)
    mae = next(row for row in scores if row["metric"] == "MAE")
    assert mae["difference_proposed_minus_reference"] == pytest.approx(-1.5)
    assert mae["CI_excludes_zero"]


def test_mismatched_compound_sets_rejected():
    a = pd.DataFrame({"compound_id": ["m1"], "endpoint":["CL"], "y_true":[-1],"y_pred":[-1]})
    b = pd.DataFrame({"compound_id": ["m2"], "endpoint":["CL"], "y_true":[-1],"y_pred":[-1]})
    with pytest.raises(ValueError):
        align(a,b)
