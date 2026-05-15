# Installation Guide — Windows 10/11 with RTX 3070

This guide gets the entire system running on a fresh Windows machine with no Linux, no WSL, no Docker.

---

## 1. Prerequisites

### NVIDIA driver

Install the latest **NVIDIA Game Ready** or **Studio** driver from <https://www.nvidia.com/Download/index.aspx>. The driver ships with the CUDA runtime; **you do not need to install the CUDA Toolkit separately** unless you plan to compile custom CUDA kernels.

Verify with:
```cmd
nvidia-smi
```
You should see your RTX 3070 listed and a CUDA version of 12.x.

### Python 3.10 or 3.11

Download from <https://www.python.org/downloads/>. **Tick "Add python.exe to PATH"** during install.

Verify:
```cmd
python --version
```

### Node.js 18 or newer (for the React frontend)

Download the LTS installer from <https://nodejs.org>.

Verify:
```cmd
node --version
npm --version
```

### Npcap (required for live packet capture)

Download from <https://npcap.com/#download>. During install, **tick "Install Npcap in WinPcap API-compatible Mode"** so Scapy can find it.

Without Npcap, the trained models still work but live capture won't.

### Git

<https://git-scm.com/download/win>

---

## 2. Clone the project

```cmd
git clone <your repo URL> nids-hybrid
cd nids-hybrid
```

---

## 3. Create a Python virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

You should see `(.venv)` in your prompt. Stay inside the venv for all Python commands below.

---

## 4. Install PyTorch with CUDA 12.1 (must come first)

**Do not use plain `pip install torch`** — that gives you the CPU-only build.

```cmd
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is visible:
```cmd
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected output:
```
CUDA: True
Device: NVIDIA GeForce RTX 3070
```

If you see `CUDA: False`, your driver is too old or you installed the CPU wheel. Re-check the previous step.

---

## 5. Install the rest of the Python dependencies

```cmd
pip install -r requirements.txt
```

This installs XGBoost, Scikit-learn, SHAP, Scapy, FastAPI, Pandas, NumPy, Joblib, Imbalanced-Learn, Uvicorn, and Pydantic.

Verify XGBoost can see the GPU:
```cmd
python -c "import xgboost as xgb; print(xgb.__version__)"
```

XGBoost 2.x ships with built-in GPU support — no extra install needed. The first XGBoost training run will print `Using device: cuda` if it works.

---

## 6. Install the frontend dependencies

```cmd
cd frontend
npm install
cd ..
```

This pulls React, Vite, Recharts, lucide-react, and framer-motion.

---

## 7. Verify everything works (without datasets yet)

Open one terminal and run:
```cmd
python -m backend.api.main
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The API will start even without trained models; endpoints that need models will return 503 until you train.

In a **second terminal**:
```cmd
cd frontend
npm run dev
```

Open <http://localhost:5173> in your browser. You should see the Operations Console UI loaded with "STANDBY" status.

Stop both processes (Ctrl-C) before continuing.

---

## 8. Download the datasets

See [`DATASETS.md`](DATASETS.md) for exact download URLs and expected directory layout.

---

## 9. Train the models

See [`TRAINING.md`](TRAINING.md) for the full training reference.

Quick version:
```cmd
python -m backend.training.train_pipeline ^
  --nf-uq-path datasets\NF-UQ-NIDS-v2\NF-UQ-NIDS-v2.csv ^
  --edge-iiot-path datasets\Edge-IIoTset\DNN-EdgeIIoT-dataset.csv ^
  --cicids-path datasets\CICIDS2017-corrected\all_days.csv ^
  --max-rows 200000
```

The `--max-rows 200000` cap is recommended for the first run so you can verify the full pipeline end-to-end in ~10–15 minutes. Remove it for the full training run (1–4 hours depending on dataset coverage).

Check `artifacts/models/` afterward — you should see:
```
xgboost.json
xgboost.params.joblib
cnn_bilstm_attn.pt
autoencoder.pt
autoencoder_threshold.pt
meta_learner.joblib
```

---

## 10. Run the live system

The packet sniffer needs **Administrator privileges**. Right-click your terminal (Command Prompt or PowerShell) and choose **Run as administrator**.

```cmd
cd C:\path\to\nids-hybrid
.venv\Scripts\activate

python scripts\list_interfaces.py
```

Pick the friendliest-looking name from the output (usually `Wi-Fi` or `Ethernet`).

Start the backend:
```cmd
python -m backend.api.main
```

In a regular terminal (no admin needed), start the frontend:
```cmd
cd frontend
npm run dev
```

Open <http://localhost:5173>, pick your interface from the dropdown, and click **START**. You should immediately see flows being classified.

---

## Common installation issues

### `ModuleNotFoundError: No module named 'backend'`
You're not at the project root, or you forgot to activate the venv. `cd` to the directory containing `requirements.txt` and run `.venv\Scripts\activate`.

### `OSError: [WinError 10013]` when starting capture
Run the backend terminal as Administrator. Packet capture requires elevated privileges on Windows.

### `'Npcap' is not installed`
Re-install Npcap from <https://npcap.com>. Make sure to tick the WinPcap-compatibility option.

### `RuntimeError: CUDA out of memory`
Reduce `DEEP_PARAMS["batch_size"]` and/or `AE_PARAMS["batch_size"]` in `backend/config.py`. On 8 GB VRAM, 256 / 512 are the defaults — try 128 / 256 if you also have other GPU processes running.

### XGBoost says "GPU not available"
Make sure you installed `xgboost>=2.0`. Older versions need separate `xgboost-gpu` packages and have different APIs.

### `npm install` warnings about peer dependencies
Safe to ignore — they're warnings about React 19 vs React 18, not errors.

### Frontend shows "Failed to fetch"
Check that the FastAPI backend is running on port 8000. The Vite dev server proxies `/api` and `/ws` to that port.
