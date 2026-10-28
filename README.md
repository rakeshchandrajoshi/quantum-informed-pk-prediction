# Quantum-Informed PK Prediction

PyTorch code for GraphMPNN, mechanism-factorized molecular networks (MFMN), and conventional MLP baselines, with training, ensemble prediction, and evaluation utilities.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q
```

## Training

Use prepared benchmark features and the official train/test partitions.

```bash
# MFMN: clearance
python scripts/train_models.py --model mfmn --endpoint CL --config configs/mfmn_cl.json --input CL_features.csv --output runs/cl_mfmn

# MLP baseline: clearance
python scripts/train_models.py --model mlp --endpoint CL --config configs/mlp_cl.json --input CL_features.csv --output runs/cl_mlp

# GraphMPNN: acidic pKa
python scripts/train_models.py --model graphmpnn --endpoint pka_acidic --config configs/graphmpnn_acidic.json --input acidic_graphs.jsonl --output runs/pka_acidic
```

Additional configurations for basic pKa, VDss, and Fu are in `configs/`.

## Input format

- **MFMN/MLP:** CSV with `compound_id`, `split`, `target`, `ctx__*`, and seven factor groups (`size__*`, `lipophilicity__*`, `ionization__*`, `polarity__*`, `binding__*`, `electronic__*`, `structural__*`). CL/VDss additionally require `aux__<factor>` targets.
- **GraphMPNN:** JSONL with `compound_id`, `split`, `target`, `node_features` (36 values per atom), `edges` (8 values per bond), and `site_index`.
- Targets for CL, VDss, and Fu are in log10 units. The internal validation fraction is 15%; ensemble seeds are 42–51 by default.

Prepared descriptors, quantum-chemical feature generation, and benchmark data are not included. The model code was independently implemented from the manuscript description; the reported experimental scores have not been independently reproduced using this package.
