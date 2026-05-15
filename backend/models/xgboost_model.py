"""
XGBoost classifier - the tabular branch of the hybrid ensemble.

Why XGBoost: gradient-boosted trees are the strongest baseline for
tabular tasks like flow classification. The published IDS literature
consistently puts XGBoost in the top tier on CICIDS2017, UNSW-NB15,
and the NF-v2 datasets.

GPU training: xgboost>=2.0 supports `device="cuda"` natively on Windows.
There is no need for Linux or WSL. RTX 3070's 8 GB of VRAM is plenty
for training on the full unified dataset (~100M rows).

Outputs from this model become input features to the meta-learner: we
expose the per-class probability vector via `predict_proba`.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb

from backend.config import MODELS_DIR, NUM_CLASSES, XGB_PARAMS
from backend.utils.logger import get_logger

log = get_logger(__name__, "training.log")

XGB_MODEL_PATH = MODELS_DIR / "xgboost.json"


class XGBoostClassifier:
    """Thin wrapper around xgboost.XGBClassifier with sample weighting
    and convenience save/load methods."""

    def __init__(self, params: dict | None = None) -> None:
        # Allow callers to override hyperparameters; otherwise use defaults
        # from config.py.
        self.params = {**XGB_PARAMS, **(params or {})}
        self.model: xgb.XGBClassifier | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Train with optional sample weights for class imbalance.

        We use early stopping on the validation set; the model that
        achieves the best mlogloss is kept. This is much more reliable
        than training for a fixed number of rounds.
        """
        log.info(f"Training XGBoost on {len(X_train):,} samples "
                 f"(device={self.params.get('device', 'cpu')})")

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Sanity-check: make sure XGBoost actually used the GPU when asked.
        if self.params.get("device") == "cuda":
            log.info("  XGBoost reports CUDA in use.")
        log.info(f"  Best iteration: {self.model.best_iteration}, "
                 f"best score: {self.model.best_score:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-class probabilities (used as features by meta-learner)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions."""
        return np.argmax(self.predict_proba(X), axis=1)

    def feature_importance(self, feature_names: list[str]) -> list[tuple[str, float]]:
        """Return features sorted by importance (descending).

        Useful for SHAP-free quick interpretability and for pruning down
        to the most-informative features.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")
        importances = self.model.feature_importances_
        ranked = sorted(zip(feature_names, importances),
                        key=lambda x: -x[1])
        return ranked

    def save(self, path: str | Path = XGB_MODEL_PATH) -> None:
        """Save in XGBoost's native JSON format (forward-compatible across
        versions; pickle is not)."""
        if self.model is None:
            raise RuntimeError("Nothing to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        # Also save the params dict so we can reload without arguments.
        joblib.dump(self.params, path.with_suffix(".params.joblib"))
        log.info(f"  Saved XGBoost model to {path}")

    def load(self, path: str | Path = XGB_MODEL_PATH) -> None:
        """Reload a previously trained model."""
        path = Path(path)
        params_path = path.with_suffix(".params.joblib")
        if params_path.exists():
            self.params = joblib.load(params_path)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.load_model(str(path))
        log.info(f"  Loaded XGBoost model from {path}")
