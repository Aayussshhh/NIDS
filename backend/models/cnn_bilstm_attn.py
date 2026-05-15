"""
CNN + BiLSTM + Attention - the sequential deep branch of the ensemble.

Architecture overview:
    1D-CNN block: extracts local spatial patterns across feature dimensions
    BiLSTM block: captures forward+backward temporal dependencies between
                  consecutive flows in a sliding window
    Multi-head attention: lets the model focus on the most informative
                          timesteps in the window (e.g. the SYN flood
                          burst inside an otherwise normal session)
    Classifier head: maps the attention-pooled representation to one of
                     our 9 unified attack classes

Why this combination beats either component alone:
    - CNN alone treats each timestep independently; misses temporal
      patterns crucial for multi-stage attacks
    - LSTM alone is slow to train on long sequences and tends to lose
      fine-grained byte/flag patterns that CNNs find easily
    - Attention on top of BiLSTM gives interpretability "for free":
      the attention weights tell us which timestep mattered most

Hardware notes for RTX 3070:
    With sequence_length=50, batch_size=256, and the dimensions in
    DEEP_PARAMS, this model uses about 1.5 GB of VRAM during training
    and 400 MB at inference. Comfortable headroom on 8 GB.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config import DEEP_PARAMS, MODELS_DIR, NUM_CLASSES
from backend.utils.logger import get_logger

log = get_logger(__name__, "training.log")

DEEP_MODEL_PATH = MODELS_DIR / "cnn_bilstm_attn.pt"


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) handles severe class imbalance better
    than plain cross-entropy. Critical here because Heartbleed and
    Infiltration are rarer than 1:10000 in our combined dataset.

    gamma controls how aggressively easy examples are down-weighted.
    gamma=0 reduces to standard cross-entropy; gamma=2 is the canonical
    starting point.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # log_softmax is numerically stable; we then exponentiate to get pt.
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class AttentionPool(nn.Module):
    """Multi-head self-attention followed by mean pooling.

    Implements the attention mechanism we use to summarize the BiLSTM
    output sequence into a single vector for classification. The
    attention weights are exposed via `last_attention_weights` so the
    explainability layer can visualize which timesteps the model
    focused on for each prediction.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, hidden). Self-attention -> same shape, plus weights.
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True,
                                           average_attn_weights=True)
        # Stash weights for explainability. Detach so it doesn't pin the
        # graph in memory between forward passes.
        self.last_attention_weights = attn_weights.detach()
        # Residual + LayerNorm (transformer-style).
        x = self.norm(x + attn_out)
        # Mean-pool over the sequence dimension.
        return x.mean(dim=1)


class CNNBiLSTMAttention(nn.Module):
    """The complete deep model.

    Input shape: (batch, sequence_length, input_dim) - each row is a
                 sequence of flow feature vectors.
    Output shape: (batch, NUM_CLASSES) - logits, NOT softmaxed.

    Treat the input as a 1D signal where the "channels" axis is the
    feature dimension (input_dim) and the "length" axis is the sequence
    of flows. Conv1d expects (batch, channels, length), so we transpose.
    """

    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 50,
        cnn_channels: list[int] = (64, 128),
        cnn_kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim

        # --- CNN block: 2 stacked Conv1d layers with batch norm + ReLU.
        cnn_layers: list[nn.Module] = []
        in_ch = input_dim
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel_size, padding="same"),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        cnn_out_channels = cnn_channels[-1]

        # --- BiLSTM block: bidirectional, multi-layer.
        # batch_first=True so we don't have to keep transposing.
        self.bilstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        bilstm_out_dim = lstm_hidden * 2  # *2 for bidirectional

        # --- Attention block.
        self.attention = AttentionPool(
            hidden_dim=bilstm_out_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )

        # --- Classifier head: 2-layer MLP.
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features). Transpose for Conv1d.
        x = x.transpose(1, 2)              # -> (batch, features, seq)
        x = self.cnn(x)                    # -> (batch, cnn_channels, seq)
        x = x.transpose(1, 2)              # -> (batch, seq, cnn_channels)
        x, _ = self.bilstm(x)              # -> (batch, seq, 2*lstm_hidden)
        x = self.attention(x)              # -> (batch, 2*lstm_hidden)
        return self.classifier(x)          # -> (batch, num_classes) logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that returns softmax probabilities."""
        self.eval()
        return F.softmax(self(x), dim=-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return the attention weights from the most recent forward pass.

        Shape: (batch, seq, seq). Used by the explainability API to
        visualize which timesteps the model attended to.
        """
        return self.attention.last_attention_weights


def build_default_model() -> CNNBiLSTMAttention:
    """Construct the model with all hyperparameters from config.py."""
    return CNNBiLSTMAttention(
        input_dim=DEEP_PARAMS["input_dim"],
        sequence_length=DEEP_PARAMS["sequence_length"],
        cnn_channels=DEEP_PARAMS["cnn_channels"],
        cnn_kernel_size=DEEP_PARAMS["cnn_kernel_size"],
        lstm_hidden=DEEP_PARAMS["lstm_hidden"],
        lstm_layers=DEEP_PARAMS["lstm_layers"],
        attention_heads=DEEP_PARAMS["attention_heads"],
        dropout=DEEP_PARAMS["dropout"],
    )
