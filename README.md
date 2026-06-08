# FedCGA

This repository contains an anonymized artifact for **FedCGA**, a context-guided aggregation framework for privacy-preserving dynamic pricing in ride-hailing services.



## Repository structure

```text
FedCGA/
├── configs/                 # Training and preprocessing configurations
├── data/                    # Data format description only
├── data_processing/         # CSV-to-client-feature preprocessing scripts
├── federated_scripts/       # FedAvg, FedProx, FedALA, FedCGA and ablation training scripts
├── pricing_model/           # Pricing model definitions
├── examples/                # Schema-only example file
├── requirements.txt
└── README.md
```

## Environment

Python 3.9 or later is recommended.

```bash
pip install -r requirements.txt
```

## Data preparation

The original trip-level data are not included. Prepare the following files locally:

```text
data/raw/train_set.csv
data/raw/val_set.csv
data/raw/test_set.csv
```

Expected columns are listed in `data/README.md` and `examples/sample_schema.txt`.

Run preprocessing:

```bash
python data_processing/train_data_processor.py
python data_processing/val_data_processor.py
python data_processing/test_data_processor.py
```

This generates client-level feature files under `data/processed/`.

## Training

FedAvg:

```bash
python federated_scripts/fedavg_training.py --config configs/fedavg_config.yaml
```

FedProx:

```bash
python federated_scripts/fedprox_training.py --config configs/fedprox_config.yaml
```

FedALA:

```bash
python federated_scripts/fedala_training.py --config configs/fedala_config.yaml
```

FedCGA:

```bash
python federated_scripts/fedcga_training.py --config configs/context_aware_fedala.yaml
```

## Ablation runs

```bash
python federated_scripts/data_feature_only_fedala.py --config configs/data_feature_fedala.yaml
python federated_scripts/history_feature_only_fedala.py --config configs/history_feature_fedala.yaml
python federated_scripts/model_feature_only_fedala.py --config configs/model_feature_fedala.yaml
python federated_scripts/state_feature_only_fedala.py --config configs/state_feature_fedala.yaml
```

## Evaluation

Each training script evaluates the global model on validation and test clients and saves the final metrics under:

```text
outputs/<method_name>/training_results/final_training_result.json
```

## Main results

The following values were extracted from the original local experiment logs before removing generated output files.

| Method | Model | RMSE | MAE | R2 |
|---|---|---:|---:|---:|
| FedAvg | Transformer | 0.146124 | 0.115606 | - |
| FedProx | Transformer | 0.152310 | 0.120956 | - |
| FedALA | Transformer | 0.172175 | 0.135840 | - |
| FedCGA | Transformer | 0.189680 | 0.149470 | - |


## Notes for anonymous artifact review

- Raw CSV files are excluded.
- Preprocessed per-client feature files are excluded.
- Checkpoints and trained `.pth` files are excluded.
- Local IDE settings and virtual environments are excluded.
- All generated files should be recreated by following the commands above.
