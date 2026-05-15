"""
Data preprocessing pipeline.

This module is responsible for taking the raw CSV files from each of our
three training datasets and producing a single, unified, model-ready
DataFrame that conforms to the NF-v2 schema defined in `config.py`.

Why this matters: the three datasets have completely different feature
sets - NF-UQ-NIDS-v2 has 43 NetFlow features, Edge-IIoTset has 61
mixed network/system features, and CICIDS2017 has 80 CICFlowMeter
features. Without a unification step, you cannot train one model on
all three. We follow Sarhan et al. (2022) and use the NF-v2 schema as
the lingua franca.

Usage:
    from backend.data.preprocessor import build_unified_dataset

    df = build_unified_dataset(
        nf_uq_path="datasets/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv",
        edge_iiot_path="datasets/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv",
        cicids_path="datasets/CICIDS2017-corrected/all_days.csv",
    )
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib

from backend.config import (
    CATEGORICAL_FEATURES,
    ENCODERS_DIR,
    IDENTIFIER_FEATURES,
    LABEL_TO_IDX,
    NF_V2_FEATURES,
    SCALERS_DIR,
    TRAINING_FEATURES,
)
from backend.utils.logger import get_logger

log = get_logger(__name__, "preprocessing.log")


# ---------------------------------------------------------------------------
# LABEL HARMONIZATION
# ---------------------------------------------------------------------------
# Each dataset uses its own attack labels. We map every original label to
# one of our 9 unified classes. Anything not in these maps is treated as
# "Other" (class 8) - this is intentional: an unknown attack type still
# gets classified as malicious rather than silently dropped.

NF_UQ_LABEL_MAP = {
    "Benign": "Benign",
    "DoS": "DoS",
    "DDoS": "DDoS",
    "Reconnaissance": "Reconnaissance",
    "Brute Force": "BruteForce",
    "Bruteforce": "BruteForce",
    "brute_force": "BruteForce",
    "Injection": "Injection",
    "XSS": "Injection",
    "SQL Injection": "Injection",
    "Bot": "Botnet",
    "Botnet": "Botnet",
    "Mirai": "Botnet",
    "Backdoor": "Exfiltration",
    "Infiltration": "Exfiltration",
    "Exploits": "Exfiltration",
    "Worms": "Other",
    "Shellcode": "Other",
    "Ransomware": "Other",
    "Fuzzers": "Other",
    "Generic": "Other",
    "Analysis": "Other",
    "Theft": "Exfiltration",
    "Password": "BruteForce",
    "MITM": "Other",
    "Scanning": "Reconnaissance",
}

EDGE_IIOT_LABEL_MAP = {
    "Normal": "Benign",
    "DDoS_UDP": "DDoS",
    "DDoS_ICMP": "DDoS",
    "DDoS_TCP": "DDoS",
    "DDoS_HTTP": "DDoS",
    "SQL_injection": "Injection",
    "Uploading": "Injection",
    "XSS": "Injection",
    "Vulnerability_scanner": "Reconnaissance",
    "Port_Scanning": "Reconnaissance",
    "Fingerprinting": "Reconnaissance",
    "Password": "BruteForce",
    "Backdoor": "Exfiltration",
    "Ransomware": "Other",
    "MITM": "Other",
}

CICIDS_LABEL_MAP = {
    "BENIGN": "Benign",
    "Benign": "Benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "Other",
    "DDoS": "DDoS",
    "PortScan": "Reconnaissance",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack \u2013 Brute Force": "BruteForce",
    "Web Attack \u2013 XSS": "Injection",
    "Web Attack \u2013 Sql Injection": "Injection",
    "Web Attack - Brute Force": "BruteForce",
    "Web Attack - XSS": "Injection",
    "Web Attack - Sql Injection": "Injection",
    "Bot": "Botnet",
    "Infiltration": "Exfiltration",
}


def _harmonize_labels(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Apply a label mapping. Unmapped labels become 'Other'."""
    # Strip whitespace because some CSV exports add trailing spaces.
    s = series.astype(str).str.strip()
    return s.map(mapping).fillna("Other")


# ---------------------------------------------------------------------------
# PER-DATASET LOADERS
# ---------------------------------------------------------------------------
# Each loader returns a DataFrame with exactly the NF-v2 column set
# plus a "Label" column with one of our 9 unified class names.


def load_nf_uq_nids(csv_path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load NF-UQ-NIDS-v2.

    This is the easy case: the dataset is already in the NF-v2 schema.
    We just need to harmonize the labels.

    Args:
        csv_path: Path to NF-UQ-NIDS-v2.csv.
        max_rows: If set, read only the first N rows. Use during development
            to iterate fast on a subset of the 76M-row file.
    """
    log.info(f"Loading NF-UQ-NIDS-v2 from {csv_path}")
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    log.info(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Some downloads include a "Dataset" column identifying the source
    # (CSE-CIC-IDS2018, ToN-IoT, etc.). We don't need it for training.
    df = df.drop(columns=["Dataset"], errors="ignore")

    # The label column is called "Attack" in NF-UQ-NIDS-v2; "Label" in some
    # older mirrors. Be defensive about both.
    label_col = "Attack" if "Attack" in df.columns else "Label"
    df["Label"] = _harmonize_labels(df[label_col], NF_UQ_LABEL_MAP)

    # Ensure every NF-v2 feature exists; fill missing ones with 0.
    for col in NF_V2_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[NF_V2_FEATURES + ["Label"]]

    log.info(f"  Class distribution:\n{df['Label'].value_counts()}")
    return df


def load_edge_iiotset(csv_path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load Edge-IIoTset.

    Edge-IIoTset uses a different feature set extracted from raw PCAPs.
    We map the closest-matching columns onto the NF-v2 schema. Features
    that don't have a natural Edge-IIoTset equivalent are filled with 0,
    which is fine because the model learns to ignore zero-only features
    via L2 regularization.

    The dataset's preferred CSV is 'DNN-EdgeIIoT-dataset.csv' (the
    pre-processed ML-ready version inside the official ZIP).
    """
    log.info(f"Loading Edge-IIoTset from {csv_path}")
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    log.info(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Build NF-v2 mapping from the columns Edge-IIoTset actually has.
    # Names follow Wireshark/tshark conventions (tcp.srcport, ip.proto, etc.).
    out = pd.DataFrame()
    out["IPV4_SRC_ADDR"] = df.get("ip.src_host", 0)
    out["IPV4_DST_ADDR"] = df.get("ip.dst_host", 0)
    out["L4_SRC_PORT"] = df.get("tcp.srcport", df.get("udp.srcport", 0))
    out["L4_DST_PORT"] = df.get("tcp.dstport", df.get("udp.dstport", 0))
    out["PROTOCOL"] = df.get("ip.proto", 0)
    out["L7_PROTO"] = 0  # Edge-IIoTset doesn't expose nDPI L7 detection.
    out["IN_BYTES"] = df.get("tcp.len", df.get("udp.length", 0))
    out["IN_PKTS"] = 1  # Edge-IIoTset is packet-level; each row is one pkt.
    out["OUT_BYTES"] = 0
    out["OUT_PKTS"] = 0
    out["TCP_FLAGS"] = df.get("tcp.flags", 0)
    out["CLIENT_TCP_FLAGS"] = df.get("tcp.flags", 0)
    out["SERVER_TCP_FLAGS"] = 0
    out["FLOW_DURATION_MILLISECONDS"] = df.get("tcp.time_delta", 0) * 1000
    out["MIN_TTL"] = df.get("ip.ttl", 64)
    out["MAX_TTL"] = df.get("ip.ttl", 64)
    out["LONGEST_FLOW_PKT"] = df.get("tcp.len", 0)
    out["SHORTEST_FLOW_PKT"] = df.get("tcp.len", 0)
    out["MIN_IP_PKT_LEN"] = df.get("tcp.len", 0)
    out["MAX_IP_PKT_LEN"] = df.get("tcp.len", 0)
    out["TCP_WIN_MAX_IN"] = df.get("tcp.window_size_value", 0)
    out["TCP_WIN_MAX_OUT"] = 0
    out["ICMP_TYPE"] = df.get("icmp.type", 0)
    out["ICMP_IPV4_TYPE"] = df.get("icmp.type", 0)
    out["DNS_QUERY_ID"] = df.get("dns.qry.name", 0)
    out["DNS_QUERY_TYPE"] = df.get("dns.qry.qu", 0)

    # Fill all remaining required columns with 0.
    for col in NF_V2_FEATURES:
        if col not in out.columns:
            out[col] = 0

    # Edge-IIoTset's label column is "Attack_label" or "Attack_type".
    label_col = "Attack_type" if "Attack_type" in df.columns else "Attack_label"
    out["Label"] = _harmonize_labels(df[label_col], EDGE_IIOT_LABEL_MAP)

    out = out[NF_V2_FEATURES + ["Label"]]
    log.info(f"  Class distribution:\n{out['Label'].value_counts()}")
    return out


def load_cicids2017(csv_path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load corrected CICIDS2017 (Engelen et al. WTMC 2021).

    CICIDS2017 uses 80 CICFlowMeter features. We map the conceptually
    closest ones onto NF-v2. The corrected version has cleaned labels
    and fixed CICFlowMeter bugs (FIN-termination, backward-packet bugs).

    Pass either the merged CSV or the per-day directory; if a directory
    is passed, all *.csv files inside are concatenated.
    """
    csv_path = Path(csv_path)
    if csv_path.is_dir():
        log.info(f"Loading CICIDS2017 from directory {csv_path}")
        frames = []
        for f in sorted(csv_path.glob("*.csv")):
            log.info(f"  Reading {f.name}")
            frames.append(pd.read_csv(f, nrows=max_rows, low_memory=False))
        df = pd.concat(frames, ignore_index=True)
    else:
        log.info(f"Loading CICIDS2017 from {csv_path}")
        df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    log.info(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # CICFlowMeter columns have leading spaces in the original release;
    # strip them for safety.
    df.columns = [c.strip() for c in df.columns]

    # Map CICFlowMeter -> NF-v2.
    out = pd.DataFrame()
    out["IPV4_SRC_ADDR"] = df.get("Source IP", df.get("Src IP", 0))
    out["IPV4_DST_ADDR"] = df.get("Destination IP", df.get("Dst IP", 0))
    out["L4_SRC_PORT"] = df.get("Source Port", df.get("Src Port", 0))
    out["L4_DST_PORT"] = df.get("Destination Port", df.get("Dst Port", 0))
    out["PROTOCOL"] = df.get("Protocol", 0)
    out["L7_PROTO"] = 0
    out["IN_BYTES"] = df.get("Total Length of Fwd Packets",
                             df.get("Fwd Pkts Tot", 0))
    out["IN_PKTS"] = df.get("Total Fwd Packets", df.get("Tot Fwd Pkts", 0))
    out["OUT_BYTES"] = df.get("Total Length of Bwd Packets",
                              df.get("Bwd Pkts Tot", 0))
    out["OUT_PKTS"] = df.get("Total Backward Packets", df.get("Tot Bwd Pkts", 0))
    # CICFlowMeter doesn't separate client/server TCP flags - use total.
    out["TCP_FLAGS"] = (df.get("FIN Flag Count", 0) + df.get("SYN Flag Count", 0)
                        + df.get("RST Flag Count", 0) + df.get("PSH Flag Count", 0)
                        + df.get("ACK Flag Count", 0) + df.get("URG Flag Count", 0))
    out["CLIENT_TCP_FLAGS"] = out["TCP_FLAGS"]
    out["SERVER_TCP_FLAGS"] = 0
    out["FLOW_DURATION_MILLISECONDS"] = df.get("Flow Duration", 0) / 1000
    out["MIN_TTL"] = 64
    out["MAX_TTL"] = 128
    out["LONGEST_FLOW_PKT"] = df.get("Fwd Packet Length Max",
                                     df.get("Fwd Pkt Len Max", 0))
    out["SHORTEST_FLOW_PKT"] = df.get("Fwd Packet Length Min",
                                      df.get("Fwd Pkt Len Min", 0))
    out["MIN_IP_PKT_LEN"] = out["SHORTEST_FLOW_PKT"]
    out["MAX_IP_PKT_LEN"] = out["LONGEST_FLOW_PKT"]
    out["SRC_TO_DST_AVG_THROUGHPUT"] = df.get("Flow Bytes/s", 0)
    out["DST_TO_SRC_AVG_THROUGHPUT"] = df.get("Flow Bytes/s", 0)
    out["TCP_WIN_MAX_IN"] = df.get("Init_Win_bytes_forward",
                                   df.get("Init Fwd Win Byts", 0))
    out["TCP_WIN_MAX_OUT"] = df.get("Init_Win_bytes_backward",
                                    df.get("Init Bwd Win Byts", 0))

    # Fill remaining columns.
    for col in NF_V2_FEATURES:
        if col not in out.columns:
            out[col] = 0

    label_col = "Label" if "Label" in df.columns else "label"
    out["Label"] = _harmonize_labels(df[label_col], CICIDS_LABEL_MAP)

    out = out[NF_V2_FEATURES + ["Label"]]
    log.info(f"  Class distribution:\n{out['Label'].value_counts()}")
    return out


# ---------------------------------------------------------------------------
# CLEANING + SCALING
# ---------------------------------------------------------------------------


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaNs, infinities, and obviously corrupt rows.

    CICIDS2017 in particular has rows with Inf values in throughput
    columns (zero-duration flows). We replace those with the column max
    or 0, then drop any rows that are still NaN.
    """
    log.info(f"Cleaning {len(df):,} rows...")
    # Replace +/-inf with NaN so we can handle uniformly.
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop string-typed garbage that snuck through (CICIDS sometimes has
    # the header row repeated mid-file).
    for col in TRAINING_FEATURES:
        if col in CATEGORICAL_FEATURES:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=TRAINING_FEATURES)
    log.info(f"  Dropped {before - len(df):,} rows with NaN/Inf values")
    return df.reset_index(drop=True)


def _encode_categoricals(
    df: pd.DataFrame, fit: bool = True
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns. Saves encoders to disk on fit.

    During inference we reuse the saved encoders so unseen categories
    map to a sentinel value rather than crashing.
    """
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        path = ENCODERS_DIR / f"{col}_encoder.joblib"
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            joblib.dump(le, path)
            encoders[col] = le
        else:
            le = joblib.load(path)
            # Map unseen labels to len(classes_) (a fresh integer).
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else len(le.classes_)
            )
            encoders[col] = le
    return df, encoders


def _scale_numeric(
    df: pd.DataFrame, fit: bool = True
) -> tuple[pd.DataFrame, RobustScaler]:
    """RobustScaler is more resistant to the heavy tails in network data
    (giant DDoS bursts, tiny Heartbleed flows) than StandardScaler.

    Apply log1p first to compress those tails further.
    """
    numeric_cols = [c for c in TRAINING_FEATURES if c not in CATEGORICAL_FEATURES]
    df[numeric_cols] = np.log1p(df[numeric_cols].clip(lower=0))

    scaler_path = SCALERS_DIR / "robust_scaler.joblib"
    if fit:
        scaler = RobustScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, scaler


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def build_unified_dataset(
    nf_uq_path: str | Path | None = None,
    edge_iiot_path: str | Path | None = None,
    cicids_path: str | Path | None = None,
    max_rows_per_dataset: int | None = None,
    fit_transformers: bool = True,
) -> pd.DataFrame:
    """Load, harmonize, clean, encode, and scale all available datasets.

    Returns a single DataFrame ready for model training. Sample/class
    weighting and SMOTE are applied later, inside training scripts, so
    that they can be done per-fold.

    Args:
        nf_uq_path: Path to NF-UQ-NIDS-v2.csv (or None to skip).
        edge_iiot_path: Path to DNN-EdgeIIoT-dataset.csv (or None to skip).
        cicids_path: Path to corrected CICIDS2017 CSV/dir (or None to skip).
        max_rows_per_dataset: Cap rows for fast development iteration.
        fit_transformers: True for training, False for inference (will
            reload saved scalers/encoders instead of fitting fresh ones).

    Returns:
        A DataFrame with TRAINING_FEATURES columns plus 'Label' (string)
        and 'LabelIdx' (int).
    """
    frames = []
    if nf_uq_path:
        frames.append(load_nf_uq_nids(nf_uq_path, max_rows_per_dataset))
    if edge_iiot_path:
        frames.append(load_edge_iiotset(edge_iiot_path, max_rows_per_dataset))
    if cicids_path:
        frames.append(load_cicids2017(cicids_path, max_rows_per_dataset))

    if not frames:
        raise ValueError("At least one dataset path must be provided.")

    log.info("Concatenating datasets...")
    df = pd.concat(frames, ignore_index=True)
    log.info(f"  Combined size: {len(df):,} rows")

    # Drop the four identifier columns - keeping them would let the model
    # cheat by memorizing attacker IPs.
    df = df.drop(columns=IDENTIFIER_FEATURES, errors="ignore")
    df = _clean(df)
    df, _ = _encode_categoricals(df, fit=fit_transformers)
    df, _ = _scale_numeric(df, fit=fit_transformers)

    # Attach integer labels for model consumption.
    df["LabelIdx"] = df["Label"].map(LABEL_TO_IDX)

    log.info(f"Final dataset: {len(df):,} rows, {df.shape[1]} cols")
    log.info(f"Final class distribution:\n{df['Label'].value_counts()}")
    return df
