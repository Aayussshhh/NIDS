"""
Convolutional Autoencoder - the unsupervised anomaly branch.

Why we need this on top of the supervised classifiers:
    XGBoost and the deep model can only recognize attack patterns they
    were trained on. A genuinely novel zero-day attack will look like
    none of the labeled classes - it will probably get classified as
    "Benign" with low confidence. The autoencoder is trained ONLY on
    benign traffic, so it learns what "normal" looks like. A flow that
    the autoencoder cannot reconstruct well (high MSE) is anomalous,
    even if no classifier has ever seen it before.

How it fits into the ensemble:
    The reconstruction error becomes a third feature stream into the
    meta-learner. The meta-learner can then weigh "looks like a known
    attack" (XGBoost + deep model) against "looks unlike anything I've
    seen as benign" (autoencoder). This catches zero-days the
    classifiers miss.

Architecture: a symmetric encoder/decoder. We use a "convolutional"
autoencoder loosely - since each input is a single flow vector (not
an image), we use 1D convolutions over the feature dimension. This
is more parameter-efficient than a fully-connected AE and learns
spatial patterns between adjacent features (e.g. fwd packet sizes).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from backend.config import AE_PARAMS, MODELS_DIR
from backend.utils.logger import get_logger

log = get_logger(__name__, "training.log")

AE_MODEL_PATH = MODELS_DIR / "autoencoder.pt"
AE_THRESHOLD_PATH = MODELS_DIR / "autoencoder_threshold.pt"


class ConvAutoencoder(nn.Module):
    """Symmetric encoder/decoder for unsupervised anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        encoder_dims: list[int] = (128, 64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim

        # --- Encoder: progressively compress to latent representation.
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in encoder_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder: mirror image of the encoder.
        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for dim in reversed(encoder_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        # Final layer back to input dimension. NO activation - we want
        # to be able to reconstruct any value, including negative ones
        # (after RobustScaler the inputs are mean-centered).
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the reconstruction. Same shape as input."""
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Just the latent representation (for downstream use if needed)."""
        return self.encoder(x)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction.

        Returns a 1D tensor of shape (batch,). Higher = more anomalous.
        """
        self.eval()
        x_hat = self(x)
        return ((x - x_hat) ** 2).mean(dim=-1)


def compute_anomaly_threshold(
    model: ConvAutoencoder,
    benign_loader: torch.utils.data.DataLoader,
    device: str,
    percentile: float = 99.0,
) -> float:
    """Find the reconstruction-error threshold that flags `percentile`% of
    benign traffic as normal.

    Run after training, on a held-out benign set. Anything above this
    threshold at inference time is considered anomalous.
    """
    log.info(f"Computing anomaly threshold at p{percentile} of benign...")
    errors: list[float] = []
    model.eval()
    for batch in benign_loader:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        err = model.reconstruction_error(x).cpu().numpy()
        errors.extend(err.tolist())
    threshold = float(np.percentile(errors, percentile))
    log.info(f"  Threshold: {threshold:.6f} "
             f"(min={min(errors):.6f}, max={max(errors):.6f}, "
             f"mean={np.mean(errors):.6f})")
    return threshold


def build_default_autoencoder() -> ConvAutoencoder:
    """Build the AE with hyperparameters from config.py."""
    return ConvAutoencoder(
        input_dim=AE_PARAMS["input_dim"],
        latent_dim=AE_PARAMS["latent_dim"],
        encoder_dims=AE_PARAMS["encoder_dims"],
        dropout=AE_PARAMS["dropout"],
    )
