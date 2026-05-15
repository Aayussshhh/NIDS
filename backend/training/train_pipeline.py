"""
Master training pipeline.

Orchestrates training of all four model components in the correct order:
    1. Preprocess and split data (train / val / stacking / test)
    2. Train XGBoost on the supervised split
    3. Train CNN+BiLSTM+Attention on the supervised split
    4. Train Convolutional Autoencoder on benign-only subset
    5. Generate stacked features on the held-out stacking split
    6. Train Meta-learner on stacked features
    7. Final evaluation on the held-out test split

Why a separate "stacking split":
    If we train the meta-learner on predictions from the same data the
    base learners trained on, those predictions are overconfident
    (the base learners have memorized the training data). The meta-
    learner would then learn the wrong weights. By holding out a slice
    of data that NO base learner sees during training, we get honest
    base predictions for the meta-learner to learn from.

Run from the project root:
    python -m backend.training.train_pipeline \\
        --nf-uq-path datasets/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv \\
        --edge-iiot-path datasets/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv \\
        --cicids-path datasets/CICIDS2017-corrected/all_days.csv

Use --max-rows 100000 during development to iterate fast on a subset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from backend.config import (
    AE_PARAMS,
    DEEP_PARAMS,
    DEVICE,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    LOGS_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    TRAINING_FEATURES,
    USE_AMP,
    UNIFIED_LABELS,
)
from backend.data.preprocessor import build_unified_dataset
from backend.models.autoencoder import (
    AE_MODEL_PATH,
    AE_THRESHOLD_PATH,
    build_default_autoencoder,
    compute_anomaly_threshold,
)
from backend.models.cnn_bilstm_attn import (
    DEEP_MODEL_PATH,
    FocalLoss,
    build_default_model,
)
from backend.models.ensemble import META_MODEL_PATH, MetaLearner, stack_features
from backend.models.xgboost_model import XGB_MODEL_PATH, XGBoostClassifier
from backend.utils.logger import get_logger

log = get_logger(__name__, "training.log")


# ---------------------------------------------------------------------------
# DATA SPLITTING
# ---------------------------------------------------------------------------
def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    stack_size: float = 0.15,
    val_size: float = 0.10,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Stratified 4-way split: train / val / stacking / test.

    All splits are stratified by Label to keep class distributions
    consistent. Returns a dict for clarity.
    """
    log.info("Splitting dataset...")
    # First peel off the test split.
    train_pool, df_test = train_test_split(
        df, test_size=test_size, stratify=df["LabelIdx"], random_state=random_state
    )
    # Then peel off the stacking split.
    relative_stack = stack_size / (1.0 - test_size)
    train_pool, df_stack = train_test_split(
        train_pool,
        test_size=relative_stack,
        stratify=train_pool["LabelIdx"],
        random_state=random_state,
    )
    # Finally peel off validation.
    relative_val = val_size / (1.0 - test_size - stack_size)
    df_train, df_val = train_test_split(
        train_pool,
        test_size=relative_val,
        stratify=train_pool["LabelIdx"],
        random_state=random_state,
    )

    splits = {"train": df_train, "val": df_val, "stack": df_stack, "test": df_test}
    for name, sp in splits.items():
        log.info(f"  {name:6s}: {len(sp):>10,} rows "
                 f"({100*len(sp)/len(df):5.1f}%)")
    return splits


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and label vector from a split DataFrame."""
    return df[TRAINING_FEATURES].values.astype(np.float32), df["LabelIdx"].values


# ---------------------------------------------------------------------------
# SEQUENCE BUILDING (for the deep model)
# ---------------------------------------------------------------------------
def make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape flat (n_flows, n_features) into (n_windows, seq_len, n_features).

    Each window of seq_len consecutive flows becomes one training example.
    The label of the window is the label of its LAST flow (most recent).
    This matches how the live inference path will work: when a new flow
    arrives, classify it in the context of the previous seq_len-1 flows.

    Drops the last (n_flows % seq_len) flows so all windows are full.
    """
    n_full = (len(X) // seq_len) * seq_len
    X = X[:n_full]
    y = y[:n_full]
    X_seq = X.reshape(-1, seq_len, X.shape[1])
    y_seq = y.reshape(-1, seq_len)[:, -1]
    return X_seq, y_seq


# ---------------------------------------------------------------------------
# CLASS WEIGHTS (for handling imbalance)
# ---------------------------------------------------------------------------
def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights, normalized to mean=1.

    XGBoost's `sample_weight` and PyTorch's `CrossEntropyLoss(weight=...)`
    both accept these directly.
    """
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid division by zero for absent classes
    weights = 1.0 / counts
    weights /= weights.mean()
    return weights


# ---------------------------------------------------------------------------
# XGBOOST
# ---------------------------------------------------------------------------
def train_xgboost(splits: dict[str, pd.DataFrame]) -> XGBoostClassifier:
    log.info("=" * 70)
    log.info("STAGE 1/4: Training XGBoost")
    log.info("=" * 70)
    X_train, y_train = to_xy(splits["train"])
    X_val, y_val = to_xy(splits["val"])

    cls_weights = compute_class_weights(y_train)
    sample_weights = cls_weights[y_train]

    model = XGBoostClassifier()
    t0 = time.time()
    model.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weights)
    log.info(f"  XGBoost training took {time.time() - t0:.1f}s")
    model.save()
    return model


# ---------------------------------------------------------------------------
# DEEP MODEL
# ---------------------------------------------------------------------------
def train_deep(splits: dict[str, pd.DataFrame]) -> torch.nn.Module:
    log.info("=" * 70)
    log.info("STAGE 2/4: Training CNN+BiLSTM+Attention")
    log.info("=" * 70)

    X_train, y_train = to_xy(splits["train"])
    X_val, y_val = to_xy(splits["val"])

    seq_len = DEEP_PARAMS["sequence_length"]
    X_train_seq, y_train_seq = make_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = make_sequences(X_val, y_val, seq_len)
    log.info(f"  Train sequences: {X_train_seq.shape}, "
             f"Val sequences: {X_val_seq.shape}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_seq).float(),
                      torch.from_numpy(y_train_seq).long()),
        batch_size=DEEP_PARAMS["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows-friendly: avoids spawn issues
        pin_memory=DEVICE == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val_seq).float(),
                      torch.from_numpy(y_val_seq).long()),
        batch_size=DEEP_PARAMS["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE == "cuda",
    )

    model = build_default_model().to(DEVICE)
    log.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    cls_weights = compute_class_weights(y_train_seq)
    weight_tensor = torch.from_numpy(cls_weights).float().to(DEVICE)
    criterion = FocalLoss(gamma=DEEP_PARAMS["focal_loss_gamma"], weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEEP_PARAMS["learning_rate"],
        weight_decay=DEEP_PARAMS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DEEP_PARAMS["epochs"]
    )
    # Mixed-precision: GradScaler + autocast halve memory and speed up
    # training on Ampere GPUs without harming accuracy.
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(1, DEEP_PARAMS["epochs"] + 1):
        # ---- TRAIN
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(X_batch)
            train_n += len(X_batch)

        # ---- VALIDATE
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE, non_blocking=True)
                y_batch = y_batch.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                val_loss_sum += loss.item() * len(X_batch)
                val_n += len(X_batch)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        scheduler.step()
        train_loss = train_loss_sum / train_n
        val_loss = val_loss_sum / val_n
        val_f1 = f1_score(np.concatenate(all_labels), np.concatenate(all_preds),
                          average="macro", zero_division=0)
        log.info(f"  Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                 f"val_loss {val_loss:.4f} | val_macroF1 {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), DEEP_MODEL_PATH)
            log.info(f"    [+] Saved best model (val_loss {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= DEEP_PARAMS["early_stop_patience"]:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    # Reload best weights before returning.
    model.load_state_dict(torch.load(DEEP_MODEL_PATH, map_location=DEVICE, weights_only=True))
    return model


# ---------------------------------------------------------------------------
# AUTOENCODER
# ---------------------------------------------------------------------------
def train_autoencoder(splits: dict[str, pd.DataFrame]) -> torch.nn.Module:
    log.info("=" * 70)
    log.info("STAGE 3/4: Training Convolutional Autoencoder (benign only)")
    log.info("=" * 70)

    benign_idx = LABEL_TO_IDX["Benign"]
    train_benign = splits["train"][splits["train"]["LabelIdx"] == benign_idx]
    val_benign = splits["val"][splits["val"]["LabelIdx"] == benign_idx]
    log.info(f"  Benign training rows: {len(train_benign):,}")
    log.info(f"  Benign validation rows: {len(val_benign):,}")

    X_train = train_benign[TRAINING_FEATURES].values.astype(np.float32)
    X_val = val_benign[TRAINING_FEATURES].values.astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=AE_PARAMS["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val)),
        batch_size=AE_PARAMS["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE == "cuda",
    )

    model = build_default_autoencoder().to(DEVICE)
    log.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=AE_PARAMS["learning_rate"])
    criterion = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val = float("inf")
    patience_counter = 0
    for epoch in range(1, AE_PARAMS["epochs"] + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for (X_batch,) in train_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                X_hat = model(X_batch)
                loss = criterion(X_hat, X_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(X_batch)
            n += len(X_batch)

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for (X_batch,) in val_loader:
                X_batch = X_batch.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    X_hat = model(X_batch)
                    val_loss_sum += criterion(X_hat, X_batch).item() * len(X_batch)
                val_n += len(X_batch)

        train_loss = loss_sum / n
        val_loss = val_loss_sum / val_n
        log.info(f"  Epoch {epoch:3d} | train_mse {train_loss:.6f} | "
                 f"val_mse {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), AE_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= AE_PARAMS["early_stop_patience"]:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE, weights_only=True))

    # Compute the anomaly threshold on the validation benign set.
    threshold = compute_anomaly_threshold(
        model, val_loader, DEVICE, percentile=AE_PARAMS["anomaly_percentile"]
    )
    torch.save({"threshold": threshold}, AE_THRESHOLD_PATH)
    log.info(f"  Saved AE threshold ({threshold:.6f}) to {AE_THRESHOLD_PATH}")
    return model


# ---------------------------------------------------------------------------
# META-LEARNER
# ---------------------------------------------------------------------------
def train_meta(
    splits: dict[str, pd.DataFrame],
    xgb_model: XGBoostClassifier,
    deep_model: torch.nn.Module,
    ae_model: torch.nn.Module,
) -> MetaLearner:
    log.info("=" * 70)
    log.info("STAGE 4/4: Training Meta-Learner on stacked predictions")
    log.info("=" * 70)

    df_stack = splits["stack"]
    X_stack, y_stack = to_xy(df_stack)

    # XGBoost predictions on the held-out stacking split.
    log.info(f"  Generating XGBoost predictions on {len(X_stack):,} samples...")
    xgb_probs = xgb_model.predict_proba(X_stack)

    # Deep model predictions. Need sequences first.
    seq_len = DEEP_PARAMS["sequence_length"]
    X_stack_seq, y_stack_seq = make_sequences(X_stack, y_stack, seq_len)
    log.info(f"  Generating deep predictions on {len(X_stack_seq):,} sequences...")
    deep_model.eval()
    deep_probs_list = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_stack_seq).float()),
        batch_size=DEEP_PARAMS["batch_size"], shuffle=False, num_workers=0,
        pin_memory=DEVICE == "cuda",
    )
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            probs = torch.softmax(deep_model(X_batch), dim=-1).cpu().numpy()
            deep_probs_list.append(probs)
    deep_probs = np.concatenate(deep_probs_list, axis=0)

    # Truncate XGBoost predictions to match the sequence count (we lose
    # the partial last window when reshaping).
    n_kept = len(deep_probs)
    xgb_probs = xgb_probs[seq_len - 1::seq_len][:n_kept]
    y_stack_aligned = y_stack_seq[:n_kept]

    # Autoencoder reconstruction errors (one per sample, last in window).
    log.info(f"  Generating AE reconstruction errors...")
    X_stack_last = X_stack_seq[:, -1, :]  # last flow of each window
    ae_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_stack_last).float()),
        batch_size=AE_PARAMS["batch_size"], shuffle=False, num_workers=0,
        pin_memory=DEVICE == "cuda",
    )
    ae_model.eval()
    ae_errors_list = []
    with torch.no_grad():
        for (X_batch,) in ae_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            err = ae_model.reconstruction_error(X_batch).cpu().numpy()
            ae_errors_list.append(err)
    ae_errors = np.concatenate(ae_errors_list)[:n_kept]

    stacked = stack_features(xgb_probs, deep_probs, ae_errors)
    log.info(f"  Stacked feature matrix shape: {stacked.shape}")

    meta = MetaLearner()
    meta.fit(stacked, y_stack_aligned)
    meta.save()
    return meta


# ---------------------------------------------------------------------------
# FINAL EVALUATION
# ---------------------------------------------------------------------------
def evaluate(
    splits: dict[str, pd.DataFrame],
    xgb_model: XGBoostClassifier,
    deep_model: torch.nn.Module,
    ae_model: torch.nn.Module,
    meta_model: MetaLearner,
) -> dict:
    log.info("=" * 70)
    log.info("FINAL EVALUATION on held-out test split")
    log.info("=" * 70)

    df_test = splits["test"]
    X_test, y_test = to_xy(df_test)
    seq_len = DEEP_PARAMS["sequence_length"]
    X_test_seq, y_test_seq = make_sequences(X_test, y_test, seq_len)

    # Base predictions.
    xgb_probs = xgb_model.predict_proba(X_test)[seq_len - 1::seq_len][:len(X_test_seq)]

    deep_probs_list = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test_seq).float()),
        batch_size=DEEP_PARAMS["batch_size"], shuffle=False, num_workers=0,
        pin_memory=DEVICE == "cuda",
    )
    deep_model.eval()
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            probs = torch.softmax(deep_model(X_batch), dim=-1).cpu().numpy()
            deep_probs_list.append(probs)
    deep_probs = np.concatenate(deep_probs_list, axis=0)

    X_test_last = X_test_seq[:, -1, :]
    ae_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test_last).float()),
        batch_size=AE_PARAMS["batch_size"], shuffle=False, num_workers=0,
        pin_memory=DEVICE == "cuda",
    )
    ae_model.eval()
    ae_errors_list = []
    with torch.no_grad():
        for (X_batch,) in ae_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            ae_errors_list.append(ae_model.reconstruction_error(X_batch).cpu().numpy())
    ae_errors = np.concatenate(ae_errors_list)

    stacked = stack_features(xgb_probs, deep_probs, ae_errors)
    y_pred = meta_model.predict(stacked)

    acc = accuracy_score(y_test_seq, y_pred)
    macro_f1 = f1_score(y_test_seq, y_pred, average="macro", zero_division=0)
    log.info(f"  Test accuracy:  {acc:.4f}")
    log.info(f"  Test macro-F1:  {macro_f1:.4f}")

    target_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    log.info("\n" + classification_report(
        y_test_seq, y_pred, target_names=target_names, zero_division=0
    ))

    cm = confusion_matrix(y_test_seq, y_pred, labels=list(range(NUM_CLASSES)))
    log.info(f"\nConfusion matrix:\n{cm}")

    metrics = {
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "classes": target_names,
    }
    metrics_path = LOGS_DIR / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info(f"  Metrics saved to {metrics_path}")
    return metrics


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the hybrid NIDS pipeline end-to-end."
    )
    parser.add_argument("--nf-uq-path", type=str, default=None,
                        help="Path to NF-UQ-NIDS-v2.csv")
    parser.add_argument("--edge-iiot-path", type=str, default=None,
                        help="Path to DNN-EdgeIIoT-dataset.csv")
    parser.add_argument("--cicids-path", type=str, default=None,
                        help="Path to corrected CICIDS2017 CSV or directory")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows per dataset for fast development")
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--skip-ae", action="store_true")
    parser.add_argument("--skip-meta", action="store_true")
    args = parser.parse_args()

    log.info(f"DEVICE: {DEVICE}, AMP: {USE_AMP}")
    log.info(f"Loading and preprocessing datasets...")
    df = build_unified_dataset(
        nf_uq_path=args.nf_uq_path,
        edge_iiot_path=args.edge_iiot_path,
        cicids_path=args.cicids_path,
        max_rows_per_dataset=args.max_rows,
        fit_transformers=True,
    )

    splits = split_dataset(df)

    # Train each component (allowing selective skips for re-runs).
    xgb_model = XGBoostClassifier()
    if not args.skip_xgb:
        xgb_model = train_xgboost(splits)
    else:
        xgb_model.load()

    deep_model = build_default_model().to(DEVICE)
    if not args.skip_deep:
        deep_model = train_deep(splits)
    else:
        deep_model.load_state_dict(torch.load(DEEP_MODEL_PATH, map_location=DEVICE, weights_only=True))

    ae_model = build_default_autoencoder().to(DEVICE)
    if not args.skip_ae:
        ae_model = train_autoencoder(splits)
    else:
        ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE, weights_only=True))

    meta_model = MetaLearner()
    if not args.skip_meta:
        meta_model = train_meta(splits, xgb_model, deep_model, ae_model)
    else:
        meta_model.load()

    evaluate(splits, xgb_model, deep_model, ae_model, meta_model)
    log.info("=" * 70)
    log.info("TRAINING COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
