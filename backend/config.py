"""
Central configuration for the Hybrid NIDS system.

All paths, model hyperparameters, training settings, and feature schemas
are defined here. Edit this file to point to your dataset locations and
to tune model behavior. Nothing else in the codebase should hardcode
paths or magic numbers.
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# Root of the project (the parent of the `backend` folder).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where downloaded raw datasets should live. Each dataset has its own
# subfolder. See `data/README.md` for the expected layout.
DATA_DIR = PROJECT_ROOT / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NF_UQ_NIDS_DIR = DATA_DIR / "NF-UQ-NIDS-v2"
EDGE_IIOT_DIR = DATA_DIR / "Edge-IIoTset"
CICIDS2017_DIR = DATA_DIR / "CICIDS2017-corrected"

# Where trained model artifacts, scalers, and encoders are saved.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
ENCODERS_DIR = ARTIFACTS_DIR / "encoders"
LOGS_DIR = ARTIFACTS_DIR / "logs"
for d in (MODELS_DIR, SCALERS_DIR, ENCODERS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------
# Auto-detect CUDA. On Windows with an RTX 3070 and the correct PyTorch
# wheel installed (cu118 or cu121), this should resolve to "cuda".
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mixed-precision training reduces VRAM use and speeds up training on
# Ampere GPUs (RTX 3070 is Ampere). Disable if you hit numerical issues.
USE_AMP = DEVICE == "cuda"

# ---------------------------------------------------------------------------
# UNIFIED FEATURE SCHEMA (NetFlow-v2, 43 features)
# ---------------------------------------------------------------------------
# Sarhan et al. (2022) defined a 43-feature NetFlow schema that is shared
# across NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2, and the
# merged NF-UQ-NIDS-v2 dataset. We adopt this exact schema as the lingua
# franca of the system. CICIDS2017 (corrected) and Edge-IIoTset features
# are mapped onto this schema by `data/preprocessor.py`.
NF_V2_FEATURES = [
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
]

# Identifier features are dropped before training to prevent label leakage
# (a model that memorizes attacker IPs is not learning to detect attacks).
IDENTIFIER_FEATURES = [
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
]

# Categorical features get label-encoded; everything else is treated numeric.
CATEGORICAL_FEATURES = ["PROTOCOL", "L7_PROTO"]

# These are the actual training features (43 total minus 4 identifiers = 39).
TRAINING_FEATURES = [f for f in NF_V2_FEATURES if f not in IDENTIFIER_FEATURES]

# Unified attack taxonomy. We harmonize the labels from all three datasets
# into these 9 classes (8 attack + 1 benign). See data/preprocessor.py for
# the per-dataset mapping logic.
UNIFIED_LABELS = [
    "Benign",          # 0
    "DoS",             # 1 - includes DoS Hulk, GoldenEye, Slowloris, Slowhttptest
    "DDoS",            # 2 - includes DDoS-LOIT, Mirai variants, reflection attacks
    "Reconnaissance",  # 3 - port scan, OS fingerprint, service enumeration
    "BruteForce",      # 4 - SSH, FTP, web login brute force
    "Injection",       # 5 - SQL injection, command injection, XSS
    "Botnet",          # 6 - C2 traffic, Ares, Mirai botnet activity
    "Exfiltration",    # 7 - infiltration, backdoor, data exfiltration
    "Other",           # 8 - everything else (worms, shellcode, ransomware, etc.)
]
NUM_CLASSES = len(UNIFIED_LABELS)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(UNIFIED_LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(UNIFIED_LABELS)}

# ---------------------------------------------------------------------------
# MODEL HYPERPARAMETERS
# ---------------------------------------------------------------------------
# XGBoost - tabular classifier. GPU training via tree_method="hist" +
# device="cuda" works natively on Windows once you install xgboost>=2.0.
XGB_PARAMS = {
    "tree_method": "hist",
    "device": DEVICE,
    "max_depth": 8,
    "learning_rate": 0.1,
    "n_estimators": 500,
    "objective": "multi:softprob",
    "num_class": NUM_CLASSES,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "random_state": 42,
}

# CNN + BiLSTM + Attention - sequential deep learner.
# Sequence length = number of consecutive flows per window. 50 is a good
# middle ground: long enough for multi-stage attack patterns, short enough
# to fit comfortably in 8 GB of VRAM at batch_size=256.
DEEP_PARAMS = {
    "sequence_length": 50,
    "input_dim": len(TRAINING_FEATURES),
    "cnn_channels": [64, 128],
    "cnn_kernel_size": 3,
    "lstm_hidden": 128,
    "lstm_layers": 2,
    "attention_heads": 4,
    "dropout": 0.3,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 30,
    "early_stop_patience": 5,
    "focal_loss_gamma": 2.0,  # focal loss handles class imbalance
}

# Convolutional Autoencoder - unsupervised anomaly detector.
# Trained ONLY on benign traffic. Anything with high reconstruction error
# is flagged as anomalous (the zero-day detection mechanism).
AE_PARAMS = {
    "input_dim": len(TRAINING_FEATURES),
    "latent_dim": 16,
    "encoder_dims": [128, 64, 32],
    "dropout": 0.2,
    "batch_size": 512,
    "learning_rate": 1e-3,
    "epochs": 50,
    "early_stop_patience": 7,
    # Threshold = 99th percentile of reconstruction error on the validation
    # benign set. Computed automatically during training.
    "anomaly_percentile": 99.0,
}

# Meta-learner (logistic regression) that combines the three signals:
# XGBoost class probabilities, deep model class probabilities, and AE score.
META_PARAMS = {
    "max_iter": 1000,
    "C": 1.0,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# LIVE INFERENCE
# ---------------------------------------------------------------------------
# How many seconds of packets to aggregate into one flow before classifying.
# Real production NIDS use a mix of expiration timers; we keep it simple
# with a single timeout. 60 s matches the typical NetFlow active timeout.
FLOW_TIMEOUT_SECONDS = 60.0
# Inactive flow eviction - flows that see no packets for this long are
# considered closed and emitted to the classifier.
FLOW_INACTIVE_TIMEOUT_SECONDS = 15.0

# Maximum number of flows to keep in the in-memory tracker. Prevents
# unbounded memory growth on high-volume links.
MAX_TRACKED_FLOWS = 100_000

# Default network interface for live capture. On Windows this is usually
# "Wi-Fi", "Ethernet", or a Npcap interface like
# r"\Device\NPF_{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}". Use
# `scripts/list_interfaces.py` to discover the names available on your
# machine. None means "let the user pick from the API".
DEFAULT_INTERFACE: str | None = None

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000

# CORS - allow the React dev server to talk to FastAPI in development.
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite default
    "http://localhost:3000",  # CRA default
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# WebSocket - how many recent classifications to broadcast per second.
# We rate-limit to avoid overwhelming the React UI on busy networks.
WEBSOCKET_BROADCAST_HZ = 10
