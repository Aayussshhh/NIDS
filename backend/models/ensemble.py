"""
Ensemble meta-learner.

Combines three signals into a final classification:
    1. XGBoost class probabilities         -> NUM_CLASSES values per sample
    2. Deep model class probabilities      -> NUM_CLASSES values per sample
    3. Autoencoder reconstruction error    -> 1 value per sample

Total stacked feature dimension per sample = 2 * NUM_CLASSES + 1 = 19.

We use LogisticRegression as the meta-learner. It's intentionally simple:
the heavy lifting is done by the base learners, and a complex meta-learner
just invites overfitting on the (already overfit) base predictions.

Training the meta-learner correctly requires HOLDOUT predictions from the
base models - never the predictions on data they were trained on. The
training script handles this by reserving a separate "stacking split".
"""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from backend.config import META_PARAMS, MODELS_DIR, NUM_CLASSES
from backend.utils.logger import get_logger

log = get_logger(__name__, "training.log")

META_MODEL_PATH = MODELS_DIR / "meta_learner.joblib"


def stack_features(
    xgb_probs: np.ndarray,
    deep_probs: np.ndarray,
    ae_errors: np.ndarray,
) -> np.ndarray:
    """Concatenate the three signal streams into one feature matrix.

    Args:
        xgb_probs: shape (n_samples, NUM_CLASSES)
        deep_probs: shape (n_samples, NUM_CLASSES)
        ae_errors: shape (n_samples,) - reconstruction errors

    Returns:
        Stacked features of shape (n_samples, 2 * NUM_CLASSES + 1).
    """
    if xgb_probs.shape != deep_probs.shape:
        raise ValueError(
            f"Shape mismatch: xgb_probs {xgb_probs.shape} vs deep_probs {deep_probs.shape}"
        )
    if ae_errors.ndim == 1:
        ae_errors = ae_errors.reshape(-1, 1)
    return np.concatenate([xgb_probs, deep_probs, ae_errors], axis=1)


class MetaLearner:
    """Logistic regression that learns the optimal weighting of the three
    base-model signals.

    Note on multi-class: we use multinomial LR with the lbfgs solver,
    which directly optimizes the multi-class log-loss without one-vs-rest
    decomposition. This matches the metric XGBoost is also optimizing.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = {**META_PARAMS, **(params or {})}
        self.model: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        log.info(f"Training meta-learner on {len(X):,} stacked samples...")
        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y)
        train_acc = self.model.score(X, y)
        log.info(f"  Meta-learner training accuracy: {train_acc:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Meta-learner not trained.")
        probs = self.model.predict_proba(X)
        
        # Pad with zeros if the model missed some classes during training
        if probs.shape[1] < NUM_CLASSES:
            full_probs = np.zeros((probs.shape[0], NUM_CLASSES), dtype=probs.dtype)
            for i, c in enumerate(self.model.classes_):
                full_probs[:, c] = probs[:, i]
            return full_probs
            
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path: str | Path = META_MODEL_PATH) -> None:
        joblib.dump(self.model, path)
        log.info(f"  Saved meta-learner to {path}")

    def load(self, path: str | Path = META_MODEL_PATH) -> None:
        self.model = joblib.load(path)
        log.info(f"  Loaded meta-learner from {path}")
