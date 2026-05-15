"""
SHAP-based explainability.

We expose two complementary explainers:
    1. TreeSHAP for XGBoost - exact, fast (microseconds per sample).
    2. GradientSHAP for the deep model - approximate but tractable.

What the API surfaces:
    For each prediction returned to the React UI, we also return the
    top-K features that pushed the model toward its decision. This gives
    analysts an answer to "WHY did you flag this flow as DDoS?", which
    is the difference between an alert that gets triaged and one that
    gets ignored.

Performance note: TreeSHAP is essentially free even at line rate.
GradientSHAP is more expensive (~50ms per sample with 50 steps), so we
compute it lazily - only when the user clicks a flow in the UI to see
details, not on every flow that comes through the wire.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import shap
import torch

from backend.config import IDX_TO_LABEL, TRAINING_FEATURES
from backend.utils.logger import get_logger

log = get_logger(__name__, "inference.log")


class HybridExplainer:
    """Holds explainers for both XGBoost and the deep model.

    Initialize once at startup with a small "background" sample (e.g.
    1000 random benign rows) - SHAP needs this as a baseline to compute
    Shapley values against.
    """

    def __init__(self):
        self._tree_explainer: shap.TreeExplainer | None = None
        self._deep_explainer: shap.GradientExplainer | None = None

    def fit_tree_explainer(self, xgb_model) -> None:
        """Initialize TreeSHAP for the XGBoost model. Constant-time setup."""
        log.info("Initializing TreeSHAP explainer for XGBoost...")
        try:
            # shap.TreeExplainer accepts the underlying booster directly.
            self._tree_explainer = shap.TreeExplainer(xgb_model.model)
        except Exception as e:
            log.error(f"Failed to initialize TreeExplainer: {e}. Explainability for XGBoost will be disabled.")
            self._tree_explainer = None

    def fit_deep_explainer(
        self,
        deep_model: torch.nn.Module,
        background: torch.Tensor,
    ) -> None:
        """Initialize GradientSHAP for the deep model.

        Args:
            deep_model: a trained CNNBiLSTMAttention.
            background: a tensor of shape (n_background, seq_len, input_dim)
                drawn from the training set. 100-1000 samples is typical;
                more = more accurate Shapley estimates but slower.
        """
        log.info(f"Initializing GradientSHAP for deep model "
                 f"(background size {len(background)})")
        deep_model.eval()
        self._deep_explainer = shap.GradientExplainer(deep_model, background)

    def explain_xgboost(
        self,
        X: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """Per-sample TreeSHAP explanations.

        Returns a list of dicts, one per input row:
            {
                "predicted_class": "DDoS",
                "top_features": [
                    {"name": "FLOW_DURATION_MS", "shap_value": 0.34},
                    ...
                ]
            }
        """
        if self._tree_explainer is None:
            raise RuntimeError("TreeSHAP not initialized. Call fit_tree_explainer().")

        # shap returns (n_samples, n_features, n_classes) for multi-class.
        shap_values = self._tree_explainer.shap_values(X)

        results = []
        for i in range(len(X)):
            # Determine predicted class from absolute total contribution.
            per_class_total = np.abs(shap_values[i]).sum(axis=0)
            pred_class = int(np.argmax(per_class_total))

            # Pull SHAP values for the predicted class only.
            class_shap = shap_values[i, :, pred_class]
            top_idx = np.argsort(np.abs(class_shap))[::-1][:top_k]

            top_features = [
                {"name": TRAINING_FEATURES[j], "shap_value": float(class_shap[j])}
                for j in top_idx
            ]
            results.append({
                "predicted_class": IDX_TO_LABEL[pred_class],
                "top_features": top_features,
            })
        return results

    def explain_deep(
        self,
        X: torch.Tensor,
        top_k: int = 5,
        n_samples: int = 25,
    ) -> list[dict]:
        """Per-sample GradientSHAP for the deep model.

        Args:
            X: input tensor of shape (batch, seq, input_dim).
            top_k: number of features to surface per sample.
            n_samples: SHAP integration steps. Lower = faster, less accurate.
                25 is a good default for interactive use.
        """
        if self._deep_explainer is None:
            raise RuntimeError("GradientSHAP not initialized.")

        shap_values = self._deep_explainer.shap_values(X, nsamples=n_samples)
        # shap_values is a list of arrays (one per output class) of shape
        # (batch, seq, input_dim). Aggregate across the seq dimension to
        # get per-feature importance.
        results = []
        for i in range(len(X)):
            per_class = np.array([np.abs(sv[i]).sum() for sv in shap_values])
            pred_class = int(np.argmax(per_class))

            class_shap = shap_values[pred_class][i]  # (seq, input_dim)
            feature_importance = np.abs(class_shap).sum(axis=0)  # (input_dim,)
            top_idx = np.argsort(feature_importance)[::-1][:top_k]

            top_features = [
                {"name": TRAINING_FEATURES[j],
                 "shap_value": float(feature_importance[j])}
                for j in top_idx
            ]
            results.append({
                "predicted_class": IDX_TO_LABEL[pred_class],
                "top_features": top_features,
            })
        return results
