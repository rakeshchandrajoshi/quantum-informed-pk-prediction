# Quantum-Informed PK Prediction — Evaluation Tools

Python utilities for evaluating molecular-property and human pharmacokinetic predictions. The tools support the five endpoints examined in *Quantum-Informed Graph and Mechanism-Factorized Networks for Structure-Based Prediction of Human Pharmacokinetic Parameters*.

## Features

- **Prediction metrics:** R², MAE, RMSE, pKa accuracy within one unit (W1), geometric mean fold error (GMFE), and accuracy within two-fold (W2F).
- **Ensemble evaluation:** Arithmetic averaging of compound-aligned predictions across random seeds (42–51 by default), with validation of seed coverage and reference values.
- **Statistical comparison:** Compound-aligned paired bootstrap with percentile 95% confidence intervals (10,000 resamples by default).
- **Input validation:** Checks for duplicate records, missing values, non-finite predictions, mismatched compounds, and unsupported endpoints.

Supported endpoint identifiers: `pka_acidic`, `pka_basic`, `CL`, `VDss`, and `Fu`.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and run the test suite:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Input files

### Single-model predictions

A CSV file with the following columns:

```csv
compound_id,endpoint,y_true,y_pred
molecule_001,pka_acidic,7.50,7.42
molecule_002,pka_acidic,8.10,8.28
```

The rows above illustrate the CSV structure; they are not experimental observations.

- `compound_id`: unique molecular identifier.
- `endpoint`: one of the five supported endpoint identifiers.
- `y_true`: reference value.
- `y_pred`: model prediction, in the same scale as the reference.

For pKa, values are expressed in pKa units. For `CL`, `VDss`, and `Fu`, use **log10-transformed values** by default. If these pharmacokinetic parameters are supplied as positive, untransformed values, specify `--scale linear` when scoring. Values for `Fu` must lie in (0, 1] when supplied on the linear scale.

### Ensemble predictions

Include an additional `seed` column:

```csv
compound_id,endpoint,seed,y_true,y_pred
molecule_001,pka_basic,42,8.00,8.10
molecule_001,pka_basic,43,8.00,7.90
```

Provide the same compound IDs for each selected seed. The default seed set is 42 through 51.

## Commands

### Evaluate predictions

```bash
python scripts/evaluate_predictions.py --input /path/to/predictions.csv --output metrics.csv
```

For original-scale PK values:

```bash
python scripts/evaluate_predictions.py --input /path/to/predictions.csv --output metrics.csv --scale linear
```

### Average predictions across seeds

```bash
python scripts/ensemble_predictions.py --input /path/to/seed_predictions.csv --output ensemble.csv --expected-seeds 42,43,44,45,46,47,48,49,50,51
```

### Compare two models with paired bootstrap

```bash
python scripts/paired_bootstrap.py --proposed /path/to/ensemble.csv --reference /path/to/comparator.csv --output comparison.csv --resamples 10000
```

The comparison requires identical compound–endpoint records and identical reference values in both input files. Confidence intervals are unadjusted for multiple comparisons.

## Repository contents

```text
quantum-informed-pk-prediction/
├── .github/workflows/tests.yml
├── scripts/
│   ├── ensemble_predictions.py
│   ├── evaluate_predictions.py
│   ├── io_utils.py
│   ├── metrics.py
│   └── paired_bootstrap.py
├── tests/test_metrics.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Scope

This repository provides executable **evaluation and statistical-analysis utilities**. It does not contain the GraphMPNN or MFMN training and inference implementations, model checkpoints, benchmark datasets, or experimental prediction files. Predictions are supplied by the user through the CSV interfaces described above. No synthetic example is presented as a research result.

Benchmark data should be obtained from its original provider in accordance with applicable access and redistribution conditions.
