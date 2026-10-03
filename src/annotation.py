"""
annotation.py: pengelolaan berkas anotasi manual (Subbab 3.7, 3.10).

Aturan:
  - Berkas anotasi tidak pernah ditimpa. Bila berkas sudah ada, pembuatan sampel baru
    dihentikan dengan pesan, bukan menulis ulang.
  - Setiap sampel disertai sidecar JSON berisi SHA-256 dari daftar review_id terurut pada data
    yang menjadi dasar sampel (data uji untuk uji emas, baris ter-flag P4 untuk validasi manusia).
  - Sebelum anotasi dipakai, hash sidecar dibandingkan dengan hasil run saat ini.
  Alur A: anotasi dari eksekusi sebelumnya dipakai kembali setelah verifikasi hash berhasil.
  Alur B: anotasi dibuat baru (sampel kosong dibuat, peneliti mengisi, lalu perhitungan dijalankan).
"""

import json
import os
from datetime import datetime

import pandas as pd

from src import drive_io


class AnnotationExists(RuntimeError):
    pass


class AnnotationMismatch(RuntimeError):
    pass


def sidecar_path(csv_path):
    return csv_path + ".sidecar.json"


def has_any_verdict(csv_path, column):
    if not os.path.exists(csv_path):
        return False
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        return False
    vals = df[column].astype(str).str.strip().str.lower()
    return bool((~vals.isin(["", "nan", "none"])).any())


def assert_can_create(csv_path, verdict_column):
    """Menolak membuat berkas anotasi baru bila berkas sudah ada (berisi verdict atau tidak)."""
    if os.path.exists(csv_path):
        terisi = has_any_verdict(csv_path, verdict_column)
        raise AnnotationExists(
            f"[BERHENTI] Berkas anotasi sudah ada: {csv_path} "
            f"({'sudah berisi verdict' if terisi else 'masih kosong'}). Berkas anotasi tidak pernah "
            f"ditimpa. Gunakan alur A (verifikasi hash dan pakai kembali), atau pindahkan berkas "
            f"lama secara manual bila anotasi memang harus dibuat ulang (alur B)."
        )


def write_sidecar(csv_path, basis_ids, basis_name, extra=None):
    info = {
        "berkas": os.path.basename(csv_path),
        "dasar_sampel": basis_name,
        "n_review_id_dasar": len(basis_ids),
        "sha256_review_id_terurut": drive_io.ids_hash(basis_ids),
        "dibuat_pada": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **(extra or {}),
    }
    return drive_io.write_once_json(info, sidecar_path(csv_path))


def verify_sidecar(csv_path, current_basis_ids, label):
    """Menghentikan eksekusi bila hash sidecar tidak cocok dengan hasil run saat ini."""
    sp = sidecar_path(csv_path)
    if not os.path.exists(csv_path):
        raise AnnotationMismatch(f"[BERHENTI] Berkas anotasi tidak ditemukan: {csv_path}")
    if not os.path.exists(sp):
        raise AnnotationMismatch(
            f"[BERHENTI] Sidecar hash tidak ditemukan untuk {csv_path}. Anotasi tidak dapat "
            f"diverifikasi terhadap run saat ini.")
    with open(sp, "r", encoding="utf-8") as f:
        info = json.load(f)
    current = drive_io.ids_hash(current_basis_ids)
    if info.get("sha256_review_id_terurut") != current:
        raise AnnotationMismatch(
            f"[BERHENTI] Hash dasar sampel '{label}' tidak cocok. Anotasi lama dibuat dari kumpulan "
            f"review_id yang berbeda dari run ini (n lama={info.get('n_review_id_dasar')}, "
            f"n sekarang={len(set(map(str, current_basis_ids)))}). Anotasi harus dibuat ulang (alur B): "
            f"pindahkan berkas lama secara manual, lalu buat sampel baru."
        )
    return info


def verify_subset_of_sample(sample_csv, subset_csv):
    a = set(pd.read_csv(sample_csv)["review_id"].astype(str))
    b = set(pd.read_csv(subset_csv)["review_id"].astype(str))
    if not b.issubset(a):
        raise AnnotationMismatch(
            f"[BERHENTI] Berkas penilai kedua {subset_csv} memuat review_id di luar sampel {sample_csv}.")
    return True
