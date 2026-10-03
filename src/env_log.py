"""env_log.py: pencatatan lingkungan eksekusi ke berkas (Subbab 3.13)."""

import os
import platform
import sys
from datetime import datetime
from importlib import metadata

from src import config, drive_io

PACKAGES = ["torch", "transformers", "scikit-learn", "cleanlab", "coral-pytorch",
            "scipy", "statsmodels", "numpy", "pandas", "matplotlib"]


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "TIDAK TERPASANG"


def _parse(v):
    out = []
    for p in v.split(".")[:3]:
        digits = "".join(ch for ch in p if ch.isdigit())
        out.append(int(digits) if digits else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


def check_coral_version():
    v = _version("coral-pytorch")
    if v == "TIDAK TERPASANG" or _parse(v) < _parse(config.MIN_CORAL_PYTORCH_VERSION):
        raise RuntimeError(
            f"[BERHENTI] coral-pytorch {v} tidak memenuhi syarat minimal "
            f"{config.MIN_CORAL_PYTORCH_VERSION}. Paket ini hanya dipakai untuk CORN "
            f"(corn_loss dan corn_label_from_logits). Pasang: pip install 'coral-pytorch>=1.3.0'."
        )
    return v


def collect_environment():
    info = {
        "waktu_catat": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "paket": {p: _version(p) for p in PACKAGES},
        "drive_root": config.DRIVE_ROOT,
        "gpu": "TIDAK ADA",
        "cuda": None,
    }
    try:
        import torch
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:  # pragma: no cover
        info["gpu"] = f"tidak terdeteksi ({e})"
    lex = {}
    for p in (config.SLANG_BASE_FILE, config.SLANG_DOMAIN_FILE):
        if os.path.exists(p):
            lex[os.path.basename(p)] = drive_io.sha256_file(p)
    info["sha256_lexicon"] = lex
    return info


def write_environment_log(run_tag):
    info = collect_environment()
    path = os.path.join(config.LOGS_DIR, f"environment__{run_tag}.json")
    res = drive_io.write_once_json(info, path)
    return info, res
