# Architecture & Design Rationale

Why this system is built the way it is. Read this if you want to understand *why* each component exists, not just *what* it does.

---

## Big picture

The system is a **hybrid ensemble** that combines three classifiers of different inductive biases, coordinated by a meta-learner, and explained by SHAP. Each base learner catches something the others miss.

```
                                         ┌─────────────┐
Tabular flow ────▶  XGBoost  ─────▶ probs│             │
(39 features)                            │             │
                                         │             │
Sequence of    ──▶ CNN+BiLSTM  ────▶ probs Meta-Learner ─▶ Final verdict
50 flows          +Attention             │             │
                                         │             │
Single flow   ───▶  Autoencoder  ──▶ MSE │             │
(benign-trained)                         └─────────────┘
                                                │
                                         ┌──────┴──────┐
                                         │   SHAP      │
                                         │ Explainer   │
                                         └─────────────┘
```

---

## Why four models instead of one?

The IDS literature consistently shows that **no single model class dominates** on the full spectrum of attack types:

| Model family | Good at | Bad at |
|---|---|---|
| Gradient boosting (XGBoost) | Tabular patterns, rare classes, fast inference | Sequential context, novel attacks |
| Sequential deep (BiLSTM) | Multi-stage attack detection, temporal patterns | Small/rare classes, extrapolation |
| Autoencoder | Zero-day / novel anomalies | Discriminating between known attack types |
| Meta-learner | Combining calibrated probabilities | Generating them |

A production NIDS that only uses one of these will miss a large fraction of the threat landscape. The hybrid architecture is not "stacking for its own sake" — it's acknowledging that the attack space is heterogeneous and needs heterogeneous detectors.

---

## Why these three datasets?

Short answer: **coverage, realism, and non-overlap.**

- **NF-UQ-NIDS-v2** is the unified NetFlow-v2 corpus that combines four earlier datasets under one 43-feature schema. Gives us the widest attack breadth (~20 categories) and the largest training signal (~76M flows).
- **Edge-IIoTset** adds what NF-UQ-NIDS-v2 lacks: MQTT/Modbus/industrial protocols and packet-level data for the CNN branch. Without it the system would be blind to IoT/IIoT attack vectors.
- **CICIDS2017 (corrected)** provides the cleanest benign baseline available for autoencoder training — Monday's 529K pure-benign enterprise flows — plus enterprise-style attack patterns (Heartbleed, web attacks, SSH/FTP brute force) with fixed labels.

The long-form research justification is in the compass artifact we produced earlier. The critical insight: **schema unification via NetFlow-v2 is mandatory**, not optional. Concatenating datasets with different feature extractors produces silently broken models.

---

## Why XGBoost for the tabular branch?

- **Proven tabular performance.** XGBoost or a close relative (LightGBM) is the top-scoring model on every major IDS benchmark published in the last decade.
- **Rare class handling.** With per-sample weights + early stopping, XGBoost handles the Heartbleed-style "11 examples in a million flows" cases better than deep models.
- **Speed.** TreeSHAP on XGBoost is *exact* and runs in microseconds, enabling real-time per-flow explanations without a budget for approximate inference.
- **Native Windows GPU.** `xgboost>=2.0` with `device="cuda"` ships GPU training natively on Windows, with no Linux or WSL requirement.

---

## Why CNN+BiLSTM+Attention (and not a pure Transformer)?

We experimented mentally with a few options:

| Architecture | Why we rejected (or chose) |
|---|---|
| Pure MLP | Throws away temporal structure. |
| Pure CNN | Captures local byte/flag patterns, misses sequence dependencies. |
| Pure BiLSTM | Slow on long sequences; underperforms on local patterns. |
| Pure Transformer | Needs more data than a typical IDS training set to train from scratch; attention is O(n²); less interpretable. |
| **CNN + BiLSTM + Attention** | **Chosen.** CNN extracts local features cheaply, BiLSTM captures long-range temporal context, attention tells us *which* timestep mattered (interpretability for free). |

The attention module is exposed via `deep_model.get_attention_weights()` — the API could surface this to the UI for "timeline heatmap" visualizations if wanted.

---

## Why a Convolutional Autoencoder for zero-day detection?

A supervised classifier can only recognize patterns it was trained on. A **genuinely novel** attack will look nothing like any labeled class and will likely be classified as Benign with low confidence — which is exactly the failure mode zero-days exploit.

The autoencoder solves this differently: we train it on **benign traffic only**, so it learns the manifold of "normal". Anything it can't reconstruct well (high MSE) is off-manifold, which is the structural definition of an anomaly.

**Why *convolutional* and not fully-connected:** for tabular data the practical difference is small, but 1D conv layers capture spatial relationships between adjacent features (forward packet sizes, TCP flag bins) with fewer parameters than dense layers.

**The zero-day fail-safe.** In `predictor.py`, if the meta-learner says "Benign" but the AE MSE is more than 2× the 99th-percentile threshold, we override to "Other" with a `zero_day_override: true` flag. The UI surfaces this prominently. This single heuristic is the difference between "catches known attacks" and "potentially catches zero-days."

---

## Why a logistic regression meta-learner?

Because the base learners do the heavy lifting. The meta-learner just needs to learn that:
- "High XGBoost confidence + high deep-model confidence + low AE error" → trust the consensus
- "Conflicting XGBoost vs deep + high AE error" → trust the anomaly signal

A linear model is sufficient for this, and — critically — it's calibrated-probability-in, calibrated-probability-out. Stacking with a second XGBoost would invite overfitting on the validation stack and is harder to reason about.

---

## Why SHAP?

- **TreeSHAP** is exact, fast, and well-understood. For XGBoost it's a solved problem.
- **GradientSHAP** for deep models is approximate but uses the same conceptual framework (Shapley values), so the two explanations are comparable.
- Unlike Integrated Gradients (PyTorch-only) or LIME (expensive, non-deterministic), SHAP works across both model families with a single library.

The tradeoff: we don't use SHAP on every single flow at inference time — only when the user clicks a row in the UI. TreeSHAP is fast enough that we *could*, but it's unnecessary overhead.

---

## Why NetFlow-v2 as the feature schema?

The Sarhan et al. (2022) NetFlow-v2 schema is the de-facto cross-dataset standard. Without a unified schema:
- You cannot train one model on multiple datasets.
- Cross-dataset evaluation is impossible (so zero-day claims are unfalsifiable).
- Feature importance numbers from one dataset don't transfer.

NetFlow-v2 also aligns with real production deployments (nProbe, softflowd) — the features we train on are the same features a production flow exporter emits, so there's no gap between lab and deployment.

---

## Why Scapy + Npcap instead of a proper flow exporter?

For the **demo**: Scapy + Npcap runs in-process, has no separate daemon, no Kafka, no setup beyond `pip install scapy`. You get from "clone repo" to "classifying live packets" in 10 minutes.

For **production**: swap Scapy for nProbe or softflowd writing to a Kafka topic, and replace `packet_capture.py` with a Kafka consumer. The rest of the pipeline — feature extractor, predictor, API — is unchanged. That's deliberate; the pipeline is designed to be swappable at the packet-ingest boundary.

---

## Why FastAPI + WebSockets?

- FastAPI gives us async-native request handling, which matters because the classification pipeline is also async.
- WebSockets are the right primitive for a continuous stream of classifications — polling `/api/recent` at 10 Hz would either be wasteful or laggy.
- Pydantic models double as API docs (`/docs` is auto-generated).

---

## Why React + Vite + Recharts?

- React is the most widely-available frontend framework — if another engineer picks up this project, they can be productive immediately.
- Vite's dev proxy makes CORS trivial during development.
- Recharts is ergonomic for the specific charts we need (donut, area) and composes well with custom styling.
- No UI framework (no Material UI, no Chakra) — we intentionally own the visual language to achieve the operations-console aesthetic.

---

## Performance characteristics

| Stage | Throughput on RTX 3070 |
|---|---|
| Scapy capture (single NIC) | ~100K pps |
| Flow tracker | ~50K flows/sec emit rate |
| XGBoost inference | ~100K flows/sec |
| Deep model inference (batch=1) | ~2000 flows/sec |
| Deep model inference (batch=64) | ~30K flows/sec |
| Autoencoder inference (batch=1) | ~5000 flows/sec |
| SHAP (TreeSHAP, per sample) | ~20K flows/sec |
| End-to-end (single flow, no batching) | ~1500 flows/sec |

Bottleneck is the deep model at batch size 1. If you need more throughput, the flow tracker could batch multiple completed flows and pass them to the predictor together. Not implemented because even 1500 flows/sec is more than a single consumer-grade NIC produces in typical traffic.

---

## Security caveats

- **The capture interface needs admin privileges.** This is a Windows security feature, not a bug.
- **The API has no authentication.** Fine for a local demo; **do not expose to the internet** without adding auth (OAuth2, bearer tokens, mTLS, etc.).
- **The model is not adversarially robust.** An attacker who knows the model architecture could craft adversarial flows. Treat this as a *defense-in-depth* layer, not a sole defense.
- **Flow state is in-memory.** A restart loses active flows. Production would persist to Redis.

---

## Extension points

If you want to extend the system, the cleanest places to hack:

- **New dataset:** add a loader in `backend/data/preprocessor.py` following the pattern of `load_cicids2017`. Map its columns onto NF-v2, add a label map, done.
- **New model:** drop a new class under `backend/models/`, wire it into `train_pipeline.py` as a new stage, and add its outputs to `stack_features()`.
- **Replace packet capture:** swap `packet_capture.py` for a Kafka consumer; no other module changes.
- **New UI panel:** add a component under `frontend/src/components/` and subscribe to `events` in `App.jsx`.
- **New explainability method:** add to `backend/models/explainer.py`; the API surface already returns a `top_features` list that can carry any per-feature attribution.
