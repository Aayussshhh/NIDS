# Datasets — Download & Setup

The system trains on three complementary datasets that together cover ~25 distinct attack families and provide ~100M+ flows. All three are freely available for academic use.

---

## Expected directory layout

Place every dataset under `datasets/` at the project root:

```
nids-hybrid/
└── datasets/
    ├── NF-UQ-NIDS-v2/
    │   └── NF-UQ-NIDS-v2.csv
    ├── Edge-IIoTset/
    │   └── DNN-EdgeIIoT-dataset.csv
    └── CICIDS2017-corrected/
        ├── monday.csv
        ├── tuesday.csv
        ├── wednesday.csv
        ├── thursday.csv
        ├── friday.csv
        └── (or a single all_days.csv)
```

The training script accepts either the directory or a merged CSV for CICIDS2017.

---

## 1. NF-UQ-NIDS-v2 (primary)

**Role:** Unified 43-feature NetFlow-v2 schema combining NF-CSE-CIC-IDS2018-v2 + NF-UNSW-NB15-v2 + NF-ToN-IoT-v2 + NF-BoT-IoT-v2.

**Size:** ~76M flows, ~10 GB CSV.

**Download options:**

- **Official (UQ Research):** <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>
  Click the "NF-UQ-NIDS-v2" link and accept the academic license.
- **Kaggle mirror:** <https://www.kaggle.com/datasets/dhoogla/nfuqnidsv2/data>
  Requires a Kaggle account; much faster download.

**Attack coverage:** Benign, Brute Force, Bot, DoS, DDoS, Infiltration, Web Attack, Fuzzers, Analysis, Backdoors, Exploits, Generic, Reconnaissance, Shellcode, Worms, Scanning, Injection, MITM, Password, Ransomware, XSS.

**Setup:**
```cmd
mkdir datasets\NF-UQ-NIDS-v2
:: place the CSV inside the folder
move C:\Downloads\NF-UQ-NIDS-v2.csv datasets\NF-UQ-NIDS-v2\
```

If you don't have 10 GB of spare RAM when training, either train on a subset with `--max-rows 500000` or download only one of the member datasets (NF-CSE-CIC-IDS2018-v2 alone is ~18M flows and roughly sufficient).

---

## 2. Edge-IIoTset (packet-level auxiliary)

**Role:** Adds IoT/IIoT protocol diversity (MQTT, Modbus, ARP/DNS spoofing, SlowITe) that's absent from the main NetFlow datasets. Also provides raw PCAPs for future packet-level extensions.

**Size:** ~21M records across multiple CSVs; ~2 GB after extraction.

**Download options:**

- **IEEE DataPort (official):** <https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications>
- **Kaggle mirror:** <https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot>

**Which file to use:** Inside the archive, the file `DNN-EdgeIIoT-dataset.csv` is the pre-processed ML-ready version. That's the one to feed into the training pipeline.

**Attack coverage:** Normal, DDoS_UDP, DDoS_ICMP, DDoS_TCP, DDoS_HTTP, SQL_injection, Uploading, Vulnerability_scanner, Port_Scanning, Fingerprinting, Password, XSS, Ransomware, Backdoor, MITM.

**Setup:**
```cmd
mkdir datasets\Edge-IIoTset
:: extract the archive, then copy the ML-ready CSV
copy "C:\Downloads\Edge-IIoTset\Selected dataset for ML and DL\DNN-EdgeIIoT-dataset.csv" datasets\Edge-IIoTset\
```

---

## 3. CICIDS2017 — Corrected (WTMC 2021 version)

**Role:** Clean benign baseline for autoencoder training (Monday = ~529K pure benign flows), plus enterprise-style attack traffic with corrected labels.

**Size:** ~2.8M flows, ~700 MB CSV total.

**⚠️ Use the CORRECTED version, not the original.** Engelen et al. (WTMC 2021) showed that the original CICIDS2017 has >20% label errors due to CICFlowMeter bugs and coarse time-window labeling. The corrected version fixes these.

**Download:**

- **Corrected dataset + fixed labels:** <https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html>
- **Fixed CICFlowMeter tool (optional, for re-extracting from raw PCAP):** <https://github.com/GintsEngelen/CICFlowMeter>

The distrinet page provides download links for each day's corrected CSV. You need all of them.

**Attack coverage:** BENIGN, DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed, DDoS, PortScan, FTP-Patator, SSH-Patator, Web Attack (Brute Force, XSS, SQL Injection), Bot, Infiltration.

**Setup:**
```cmd
mkdir datasets\CICIDS2017-corrected
:: place all day CSVs inside - the preprocessor will concatenate them
copy C:\Downloads\CICIDS2017-corrected\*.csv datasets\CICIDS2017-corrected\
```

Or, if you prefer a single file:
```cmd
:: from the project root, in a Python console:
python -c "import pandas as pd, glob; pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob('datasets/CICIDS2017-corrected/*.csv')], ignore_index=True).to_csv('datasets/CICIDS2017-corrected/all_days.csv', index=False)"
```

---

## Verifying the datasets

Once all three are in place, quickly smoke-test the preprocessor on a small subset:

```cmd
python -c "from backend.data.preprocessor import build_unified_dataset; df = build_unified_dataset(nf_uq_path='datasets/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv', edge_iiot_path='datasets/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv', cicids_path='datasets/CICIDS2017-corrected', max_rows_per_dataset=10000); print(df.shape); print(df['Label'].value_counts())"
```

Expected output: a DataFrame of ~30K rows (10K from each dataset) with a balanced-ish distribution across the 9 unified classes.

---

## Disk space budget

| Item | Size |
|---|---|
| NF-UQ-NIDS-v2 | ~10 GB |
| Edge-IIoTset | ~2 GB |
| CICIDS2017 corrected | ~700 MB |
| Trained artifacts | ~1 GB |
| Preprocessing temp | ~5 GB |
| **Total working set** | **~20 GB** |

---

## Legal / licensing

All three datasets are released for academic research. Check each dataset's license page before using the trained models in commercial products. Attribution to the original authors is required in any publication or production deployment:

- Sarhan et al. (2022) — NetFlow datasets
- Ferrag et al. (2022) — Edge-IIoTset
- Sharafaldin et al. (2018) — CICIDS2017; Engelen et al. (2021) — corrected labels
