# Hybrid NIDS — Operations Console

Real-time Network Intrusion Detection System combining **XGBoost**, a **CNN+BiLSTM+Attention** sequential deep learner, a **Convolutional Autoencoder** for zero-day detection, and **SHAP** explainability — surfaced through a production-grade React dashboard.

```
┌────────────────────────────────────────────────────────────────┐
│  Live packets ──▶ Flow tracker ──▶ Hybrid Predictor ──▶ React  │
│                                          │                     │
│                                          ├─ XGBoost            │
│                                          ├─ CNN+BiLSTM+Attn    │
│                                          ├─ Conv Autoencoder   │
│                                          ├─ Meta-Learner       │
│                                          └─ SHAP explainer     │
└────────────────────────────────────────────────────────────────┘
```

---

## What it does

1. Captures live network traffic from a chosen interface using **Scapy + Npcap** on Windows.
2. Aggregates packets into bidirectional flows using a real-time flow tracker that produces **NetFlow-v2** feature vectors.
3. Runs each flow through a **hybrid ensemble** of four models, fuses the outputs, and produces a verdict: *Benign*, *DoS*, *DDoS*, *Reconnaissance*, *BruteForce*, *Injection*, *Botnet*, *Exfiltration*, or *Other*.
4. Includes a **zero-day fail-safe**: if the supervised models say *Benign* but the unsupervised autoencoder strongly disagrees, the flow is escalated as a novel attack candidate.
5. Streams classifications, confidence scores, model breakdowns, and SHAP explanations to a **React dashboard** over WebSockets.

---

## Project structure

```
nids-hybrid/
├── backend/
│   ├── config.py                  # All paths, hyperparameters, schemas
│   ├── api/main.py                # FastAPI server + WebSockets
│   ├── data/preprocessor.py       # Unifies all 3 datasets onto NF-v2
│   ├── models/
│   │   ├── xgboost_model.py       # XGBoost (GPU, native Windows)
│   │   ├── cnn_bilstm_attn.py     # PyTorch sequential deep model
│   │   ├── autoencoder.py         # Convolutional autoencoder
│   │   ├── ensemble.py            # Meta-learner that fuses signals
│   │   └── explainer.py           # TreeSHAP + GradientSHAP
│   ├── training/train_pipeline.py # End-to-end training orchestrator
│   ├── inference/
│   │   ├── packet_capture.py      # Scapy/Npcap live capture
│   │   ├── feature_extractor.py   # Real-time flow tracker
│   │   └── predictor.py           # Inference orchestrator
│   └── utils/logger.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Top-level layout
│   │   ├── components/            # Header, LiveTraffic, ThreatDetails…
│   │   ├── services/api.js        # REST + WebSocket client
│   │   └── hooks/useWebSocket.js  # Reconnecting socket hook
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   └── list_interfaces.py         # Discover Windows interface names
├── docs/
│   ├── INSTALL.md                 # Detailed installation
│   ├── DATASETS.md                # Dataset download + setup
│   ├── TRAINING.md                # Training-pipeline guide
│   └── ARCHITECTURE.md            # Design rationale
├── artifacts/                     # Trained models + scalers (created at runtime)
├── datasets/                      # Drop downloaded datasets here
├── requirements.txt
└── README.md
```

---

## Quick start (5 commands, after installation)

```bash
# 1. Discover your network interface name
python scripts/list_interfaces.py

# 2. Train the full pipeline (XGBoost + Deep + AE + Meta)
python -m backend.training.train_pipeline \
  --nf-uq-path datasets/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv \
  --edge-iiot-path datasets/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv \
  --cicids-path datasets/CICIDS2017-corrected/all_days.csv

# 3. Start the FastAPI backend (run as Administrator on Windows for capture)
python -m backend.api.main

# 4. In a second terminal, start the React frontend
cd frontend && npm install && npm run dev

# 5. Open the dashboard
#    http://localhost:5173
```

---

## Hardware & OS

- **Windows 10/11** (64-bit). No Linux/WSL needed.
- **NVIDIA GPU** with CUDA 12.x driver. Tested on **RTX 3070 (8 GB)**.
- Minimum 16 GB RAM (32 GB recommended for training the full unified dataset).
- ~50 GB free disk for the datasets, ~5 GB for trained models.

---

## See also

- [`docs/INSTALL.md`](docs/INSTALL.md) — full step-by-step installation
- [`docs/DATASETS.md`](docs/DATASETS.md) — exact dataset download and setup instructions
- [`docs/TRAINING.md`](docs/TRAINING.md) — training pipeline reference, hyperparameter tuning, expected metrics
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — why each component was chosen and how they fit together

---

## Capabilities at a glance

| Capability | Component |
|---|---|
| Live packet capture (Windows native) | `inference/packet_capture.py` (Scapy + Npcap) |
| Bidirectional flow tracking | `inference/feature_extractor.py` |
| Tabular classification | `models/xgboost_model.py` (XGBoost GPU) |
| Sequential pattern detection | `models/cnn_bilstm_attn.py` (PyTorch CUDA) |
| Zero-day / novel attack detection | `models/autoencoder.py` (Conv AE, benign-only) |
| Decision fusion | `models/ensemble.py` (LogReg meta-learner) |
| Explainability per prediction | `models/explainer.py` (TreeSHAP + GradientSHAP) |
| REST + WebSocket API | `api/main.py` (FastAPI) |
| Production-grade UI | `frontend/` (React + Vite + Recharts) |
