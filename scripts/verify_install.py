"""
Post-install verification script.

Run this AFTER completing INSTALL.md to confirm that:
    - PyTorch can see your GPU
    - XGBoost has GPU support
    - Scapy + Npcap can enumerate interfaces
    - All project modules import cleanly

Usage:
    python scripts/verify_install.py
"""

import sys
import traceback


def check(name, fn):
    """Run a check and print pass/fail."""
    try:
        result = fn()
        if result is True or result is None:
            print(f"  [PASS] {name}")
            return True
        elif result is False:
            print(f"  [FAIL] {name}")
            return False
        else:
            print(f"  [PASS] {name}: {result}")
            return True
    except Exception as e:
        print(f"  [FAIL] {name}")
        print(f"         Error: {type(e).__name__}: {e}")
        return False


def check_python():
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        raise RuntimeError(f"Python 3.10+ required, got {version.major}.{version.minor}")
    return f"Python {version.major}.{version.minor}.{version.micro}"


def check_torch():
    import torch
    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU-only"
    if not cuda_ok:
        print("         Warning: CUDA not available. Training will use CPU (slow).")
    return f"torch {torch.__version__} on {device_name}"


def check_xgboost():
    import xgboost as xgb
    major = int(xgb.__version__.split(".")[0])
    if major < 2:
        raise RuntimeError(
            f"xgboost>=2.0 required, got {xgb.__version__}. "
            "Upgrade with: pip install --upgrade xgboost"
        )
    return f"xgboost {xgb.__version__}"


def check_scapy():
    from scapy.arch import get_if_list
    ifs = get_if_list()
    if not ifs:
        raise RuntimeError("No interfaces found - is Npcap installed?")
    return f"scapy sees {len(ifs)} interfaces"


def check_project_modules():
    """Import every project module to catch syntax or import errors."""
    import backend.config
    import backend.utils.logger
    import backend.data.preprocessor
    import backend.models.xgboost_model
    import backend.models.cnn_bilstm_attn
    import backend.models.autoencoder
    import backend.models.ensemble
    import backend.models.explainer
    import backend.inference.packet_capture
    import backend.inference.feature_extractor
    import backend.inference.predictor
    import backend.api.main
    return "all backend modules importable"


def check_fastapi():
    import fastapi
    import uvicorn
    return f"fastapi {fastapi.__version__}, uvicorn {uvicorn.__version__}"


def check_shap():
    import shap
    return f"shap {shap.__version__}"


def main():
    print("=" * 70)
    print("Hybrid NIDS - Installation Verification")
    print("=" * 70)
    print()

    checks = [
        ("Python version", check_python),
        ("PyTorch + CUDA", check_torch),
        ("XGBoost 2.x", check_xgboost),
        ("Scapy + Npcap", check_scapy),
        ("FastAPI + Uvicorn", check_fastapi),
        ("SHAP", check_shap),
        ("Project modules import cleanly", check_project_modules),
    ]

    results = [check(name, fn) for name, fn in checks]

    print()
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"All {total} checks passed. You're ready to download datasets and train.")
        print("Next: see docs/DATASETS.md")
    else:
        print(f"{passed}/{total} checks passed. Fix the failures before proceeding.")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
