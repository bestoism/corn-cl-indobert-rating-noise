"""corpus.py: verifikasi penguncian korpus mentah (Subbab 3.2, Batasan Masalah)."""

import os

import pandas as pd

from src import config, drive_io


class CorpusMismatch(RuntimeError):
    pass


def verify_master(path=None):
    """
    Membandingkan SHA-256 dan jumlah baris berkas master terhadap nilai yang
    tertulis di kode (config.EXPECTED_MASTER_SHA256), bukan terhadap manifest
    yang dibuat ulang oleh run ini.
    """
    path = path or config.MASTER_FILE
    if not os.path.isfile(path):
        raise CorpusMismatch(f"[BERHENTI] Berkas master tidak ditemukan: {path}")
    sha = drive_io.sha256_file(path)
    n_rows = len(pd.read_csv(path))
    ok_sha = sha == config.EXPECTED_MASTER_SHA256
    ok_rows = n_rows == config.EXPECTED_MASTER_ROWS
    result = {
        "master_file": path,
        "sha256_dihitung": sha,
        "sha256_di_kode": config.EXPECTED_MASTER_SHA256,
        "sha256_cocok": bool(ok_sha),
        "n_baris_dihitung": int(n_rows),
        "n_baris_di_kode": int(config.EXPECTED_MASTER_ROWS),
        "n_baris_cocok": bool(ok_rows),
    }
    if not (ok_sha and ok_rows):
        raise CorpusMismatch(
            "[BERHENTI] Korpus master tidak sesuai dengan penguncian. "
            f"SHA-256 cocok: {ok_sha}, jumlah baris cocok: {ok_rows}. "
            "Jangan melanjutkan: seluruh eksperimen harus berjalan di atas korpus identik."
        )
    return result


def write_verification(result):
    path = os.path.join(config.SHARED_RESULTS_DIR, "corpus_verification.json")
    return drive_io.write_once_json(result, path)
