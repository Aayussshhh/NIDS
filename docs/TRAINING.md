# Training Guide

Complete reference for the training pipeline — what it does, how long it takes, expected metrics, and how to tune it.

---

## The pipeline in one picture

```
                 ┌──────────────────┐
                 │  Unified dataset │  (preprocessor.py)
                 │     ~100M rows   │
                 └────────┬─────────┘
                          │
         ┌────────────────┼────────────────┬────────────────┐
         ▼                ▼                ▼                ▼
     TRAIN (60%)     VAL (10%)        STACK (15%)      TEST (15%)
         │                │                │                │
         ├──────┐         │                │                │
         │      │         │                │                │
         ▼      ▼         ▼                │                │
     XGBoost  Deep      (validation used   │                │
         │    CNN+      for early stop)    │                │
         │   BiLSTM                        │                │
         │    +Attn                        │                │
         │                                 │                │
         ▼                                 │                │
      Autoencoder (benign rows only) ◀─────┘                │
         │                                                  │
         │        STACK predictions ◀──── held out          │
         │                                                  │
         ▼                                                  │
    Meta-learner                                            │
         │                                                  │
         └──────────── Final evaluation ◀──────────────────┘
```

---

## Stage 1 — Preprocessing

The `build_unified_dataset()` call:

1. Loads each of the three CSVs.
2. Maps each dataset's features onto the 43-feature NetFlow-v2 schema.
3. Harmonizes labels into 9 unified classes.
4. Drops identifier columns (IPs, ports) to prevent label leakage.
5. Applies `log1p + RobustScaler` to numerics and `LabelEncoder` to categoricals.
6. Saves scalers + encoders to `artifacts/` for reuse at inference.

**Output:** one DataFrame with 39 features + `Label` + `LabelIdx`.

---

## Stage 2 — XGBoost

- Trained on the **train split** with early stopping on **val**.
- Uses `tree_method="hist"` + `device="cuda"` → native GPU on Windows.
- Sample weights from inverse class frequency handle imbalance.
- Saves: `xgboost.json` + `xgboost.params.joblib`.

**Typical timings on RTX 3070:**
- 200K rows: ~1 min
- 2M rows: ~10 min
- 20M rows: ~45 min

---

## Stage 3 — CNN+BiLSTM+Attention

- Data reshaped into sequences of 50 consecutive flows.
- Focal Loss (γ=2) handles severe imbalance (Heartbleed, Infiltration).
- Class weights applied on top.
- Mixed precision (AMP) training halves memory use and ~1.5×'s speed.
- Cosine LR schedule + AdamW + gradient clipping.
- Early stopping on val loss, patience 5.
- Saves best weights as `cnn_bilstm_attn.pt`.

**Typical timings on RTX 3070:**
- 200K rows, 30 epochs: ~15 min (usually stops around epoch 18)
- 2M rows, 30 epochs: ~2 hours
- 20M rows, 30 epochs: ~10 hours (consider 10–15 epochs instead)

---

## Stage 4 — Convolutional Autoencoder

- Trained on **benign-only** rows from the train split.
- Symmetric encoder/decoder with BatchNorm + LeakyReLU.
- MSE loss.
- Anomaly threshold = 99th percentile of reconstruction error on val benign.
- Saves `autoencoder.pt` + `autoencoder_threshold.pt`.

**Typical timings on RTX 3070:**
- 100K benign rows, 50 epochs: ~8 min (usually stops around epoch 30)
- 1M benign rows: ~45 min

---

## Stage 5 — Meta-learner

- Builds stacked features on the **held-out stack split**:
  - XGBoost class probabilities (9 values)
  - Deep model class probabilities (9 values)
  - Autoencoder reconstruction error (1 value)
- LogisticRegression with multinomial loss.
- Saves `meta_learner.joblib`.

**Typical timing:** under 1 minute.

---

## Stage 6 — Final evaluation

Runs the full stacked pipeline on the untouched **test split** and reports:

- Accuracy
- Macro-F1
- Per-class precision / recall / F1
- Confusion matrix
- Metrics saved to `artifacts/logs/test_metrics.json`

---

## Expected metrics

On the full unified dataset (no `--max-rows` cap), realistic production-grade numbers:

| Metric | Expected range |
|---|---|
| Test accuracy | 0.96 – 0.99 |
| Test macro-F1 | 0.88 – 0.95 |
| Benign precision | > 0.99 |
| DDoS / DoS recall | > 0.97 |
| Rare classes (Exfiltration, Other) F1 | 0.60 – 0.85 |

**Red flags that indicate a problem:**

- Test accuracy > 0.999 → probable label leakage (check that you dropped IPs + ports).
- Val loss goes up while train loss goes down → overfitting, reduce model capacity or increase dropout.
- XGBoost using CPU (slow) → verify `xgboost>=2.0` is installed and `device="cuda"` in config.
- Autoencoder MSE below 1e-6 → collapsed to identity, increase dropout or reduce latent dim.

---

## Running the pipeline

### Full run (use this after validating on a subset)

```cmd
python -m backend.training.train_pipeline ^
  --nf-uq-path datasets\NF-UQ-NIDS-v2\NF-UQ-NIDS-v2.csv ^
  --edge-iiot-path datasets\Edge-IIoTset\DNN-EdgeIIoT-dataset.csv ^
  --cicids-path datasets\CICIDS2017-corrected
```

### Fast development run (10–15 minutes total)

```cmd
python -m backend.training.train_pipeline ^
  --nf-uq-path datasets\NF-UQ-NIDS-v2\NF-UQ-NIDS-v2.csv ^
  --edge-iiot-path datasets\Edge-IIoTset\DNN-EdgeIIoT-dataset.csv ^
  --cicids-path datasets\CICIDS2017-corrected ^
  --max-rows 200000
```

### Retrain only one component

If you change a hyperparameter and only want to retrain one model:

```cmd
:: Retrain only the deep model, reuse XGBoost + AE + Meta
python -m backend.training.train_pipeline ^
  --nf-uq-path ... ^
  --skip-xgb --skip-ae --skip-meta
```

Note: if you change the deep model, you should also retrain the meta-learner (don't skip it) because its weights depend on the deep model's outputs.

---

## Hyperparameter tuning

All tunable knobs are in `backend/config.py`:

### `XGB_PARAMS`
- `max_depth`: 6–10. Deeper = more capacity, more overfitting risk.
- `n_estimators`: 300–1000. Early stopping protects against too-large values.
- `learning_rate`: 0.05–0.3. Lower needs more rounds.

### `DEEP_PARAMS`
- `sequence_length`: 30–100. Longer captures multi-stage attacks better but uses more VRAM.
- `batch_size`: drop to 128 if you hit OOM; raise to 512 if you have spare VRAM.
- `lstm_hidden`: 64–256. Bigger = more capacity.
- `focal_loss_gamma`: 1.0–3.0. Higher down-weights easy examples more aggressively.
- `dropout`: 0.2–0.5. Higher if you see overfitting.

### `AE_PARAMS`
- `latent_dim`: 8–32. Smaller forces the AE to compress harder, better anomaly detection but higher benign reconstruction error.
- `anomaly_percentile`: 99.0 (default). Lower to 95 for a noisier/more-sensitive detector, raise to 99.5 for fewer false alarms.

---

## Training on a CPU (not recommended)

If you don't have a GPU, the code still works — PyTorch and XGBoost both fall back to CPU.

Expected slowdowns:
- XGBoost: ~5–10× slower
- Deep model: ~20–50× slower
- Autoencoder: ~10–20× slower

On CPU, use `--max-rows 100000` or you won't finish this decade.

---

## Reading the training log

All output goes to stdout AND to `artifacts/logs/training.log`. Example of a healthy run:

```
2026-04-23 14:12:08 | INFO    | backend.training.train_pipeline | DEVICE: cuda, AMP: True
2026-04-23 14:12:08 | INFO    | backend.training.train_pipeline | Loading and preprocessing datasets...
2026-04-23 14:12:42 | INFO    | backend.data.preprocessor      | Final dataset: 14,238,112 rows, 41 cols
2026-04-23 14:12:42 | INFO    | backend.training.train_pipeline |   train : 9,966,678 (70.0%)
2026-04-23 14:12:42 | INFO    | backend.training.train_pipeline |   val   : 1,423,811 (10.0%)
2026-04-23 14:12:42 | INFO    | backend.training.train_pipeline |   stack : 1,423,811 (10.0%)
2026-04-23 14:12:42 | INFO    | backend.training.train_pipeline |   test  : 1,423,812 (10.0%)
2026-04-23 14:12:42 | INFO    | backend.training.train_pipeline | STAGE 1/4: Training XGBoost
2026-04-23 14:13:15 | INFO    | backend.models.xgboost_model   |   Best iteration: 347, best score: 0.1203
```

If you see warnings about "CUDA out of memory" or "backend=cpu", investigate immediately — the training will still finish but the result won't be production-grade.
