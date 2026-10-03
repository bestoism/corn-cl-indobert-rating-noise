"""
drive_io.py: penyimpanan berkas yang tahan terhadap perilaku mount Google Drive.

Aturan:
  1. Berkas di Drive tidak pernah ditimpa. Setiap unit kerja menulis satu berkas
     bernama unik, hanya sekali (mode pembuatan eksklusif 'xb').
  2. Setelah penulisan: flush, fsync, lalu baca ulang dan bandingkan isi serta
     ukuran. Pembacaan ulang diulang beberapa kali karena mount Drive dapat
     terlambat menampakkan berkas.
  3. Bila berkas sudah ada dan isinya identik, penulisan dilewati (aman untuk
     melanjutkan eksekusi). Bila berbeda, eksekusi dihentikan dengan pesan jelas.
  4. Setelah penulisan, folder diperiksa terhadap salinan ganda bernama
     "nama (1).ext" yang menandakan Drive membuat salinan terpisah.
  5. audit_drive_state() memindai folder proyek dan menghentikan eksekusi bila
     ada nama ganda atau artefak yang tidak diharapkan.
"""

import hashlib
import io
import json
import os
import re
import shutil
import time
import zipfile

import numpy as np
import pandas as pd

from src import config


class DriveWriteError(RuntimeError):
    pass


class DriveAuditError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# Hash
# ----------------------------------------------------------------------
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Pemeriksaan salinan ganda dan baca ulang
# ----------------------------------------------------------------------
_DUP_NAME_PATTERNS = [
    re.compile(r".* \(\d+\)(\.[^.]+)?$"),
    re.compile(r"^Salinan .*", re.IGNORECASE),
    re.compile(r"^Copy of .*", re.IGNORECASE),
    re.compile(r".* - Copy( \(\d+\))?(\.[^.]+)?$", re.IGNORECASE),
    re.compile(r".* - Salinan( \(\d+\))?(\.[^.]+)?$", re.IGNORECASE),
]


def is_duplicate_name(name):
    return any(p.match(name) for p in _DUP_NAME_PATTERNS)


def _check_no_duplicate_sibling(path):
    folder = os.path.dirname(path)
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    pattern = re.compile(rf"^{re.escape(stem)} \(\d+\){re.escape(ext)}$")
    for name in os.listdir(folder):
        if name != base and pattern.match(name):
            raise DriveWriteError(
                f"[BERHENTI] Terdeteksi salinan ganda di Drive: '{name}' di samping '{base}' "
                f"pada folder {folder}. Hapus salinan ganda secara manual di Drive web, lalu "
                f"jalankan ulang bagian ini."
            )


def _read_back(path, expected_len=None, retries=8, wait=2.0):
    last = None
    for _ in range(retries):
        try:
            if os.path.exists(path):
                if expected_len is None or os.path.getsize(path) == expected_len:
                    with open(path, "rb") as f:
                        return f.read()
            last = f"ukuran/keberadaan belum sesuai (ada={os.path.exists(path)})"
        except OSError as e:
            last = str(e)
        time.sleep(wait)
    raise DriveWriteError(f"[BERHENTI] Berkas tidak terbaca ulang dengan benar: {path} ({last})")


def write_once_bytes(path, data):
    """Menulis 'data' ke 'path' tepat sekali. Mengembalikan ringkasan penulisan."""
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        raise DriveWriteError(
            f"[BERHENTI] Folder belum dibuat: {folder}. Jalankan config.ensure_all_dirs() di awal."
        )

    if os.path.exists(path):
        existing = _read_back(path, retries=3, wait=1.0)
        if existing == data:
            return {"path": path, "status": "sudah_ada_identik", "bytes": len(data),
                    "sha256": sha256_bytes(data)}
        raise DriveWriteError(
            f"[BERHENTI] Berkas sudah ada dengan isi berbeda: {path}. Berkas di Drive tidak pernah "
            f"ditimpa. Jika berkas lama memang berasal dari eksekusi yang gagal, hapus manual di "
            f"Drive web, lalu jalankan ulang."
        )

    with open(path, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    back = _read_back(path, expected_len=len(data))
    if back != data:
        raise DriveWriteError(f"[BERHENTI] Isi berkas hasil baca ulang berbeda dari yang ditulis: {path}")
    _check_no_duplicate_sibling(path)
    return {"path": path, "status": "ditulis", "bytes": len(data), "sha256": sha256_bytes(data)}


def write_once_text(path, text):
    return write_once_bytes(path, text.encode("utf-8"))


def write_once_csv(df, path, **to_csv_kwargs):
    buf = io.StringIO()
    df.to_csv(buf, index=False, **to_csv_kwargs)
    return write_once_bytes(path, buf.getvalue().encode("utf-8"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Tidak dapat men-serialisasi tipe {type(o)}")


def write_once_json(obj, path):
    text = json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default)
    return write_once_bytes(path, text.encode("utf-8"))


def write_once_npz(path, **arrays):
    """npz tidak deterministik pada tingkat byte, sehingga perbandingan memakai isi larik."""
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        raise DriveWriteError(f"[BERHENTI] Folder belum dibuat: {folder}")
    if os.path.exists(path):
        old = np.load(path, allow_pickle=False)
        same = set(old.files) == set(arrays) and all(
            np.array_equal(old[k], np.asarray(arrays[k]), equal_nan=True)
            if np.asarray(arrays[k]).dtype.kind in "fc" else np.array_equal(old[k], np.asarray(arrays[k]))
            for k in arrays
        )
        if same:
            return {"path": path, "status": "sudah_ada_identik", "bytes": os.path.getsize(path)}
        raise DriveWriteError(f"[BERHENTI] Berkas npz sudah ada dengan isi berbeda: {path}")

    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    with open(path, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    back = _read_back(path, expected_len=len(data))
    if back != data:
        raise DriveWriteError(f"[BERHENTI] Isi npz hasil baca ulang berbeda: {path}")
    loaded = np.load(path, allow_pickle=False)
    for k in arrays:
        if loaded[k].shape != np.asarray(arrays[k]).shape:
            raise DriveWriteError(f"[BERHENTI] Bentuk larik '{k}' berubah setelah baca ulang: {path}")
    _check_no_duplicate_sibling(path)
    return {"path": path, "status": "ditulis", "bytes": len(data)}


# ----------------------------------------------------------------------
# Verifikasi berkas yang sudah ada (untuk melanjutkan eksekusi)
# ----------------------------------------------------------------------
def is_verified(path, expected_rows=None, required_cols=None):
    """True bila berkas ada, tidak kosong, dan dapat dibaca sesuai tipenya."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            if expected_rows is not None and len(df) != expected_rows:
                return False
            if required_cols is not None and not set(required_cols).issubset(df.columns):
                return False
        elif path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as z:
                _ = z.files
    except Exception:
        return False
    return True


def read_csv_verified(path, **kwargs):
    if not is_verified(path):
        raise FileNotFoundError(f"Berkas tidak ada atau tidak dapat dibaca: {path}")
    return pd.read_csv(path, **kwargs)


# ----------------------------------------------------------------------
# Audit keadaan Drive
# ----------------------------------------------------------------------
DIPERTAHANKAN = "DIPERTAHANKAN"
DIPERBOLEHKAN = "DIPERBOLEHKAN"
TIDAK_DIHARAPKAN = "TIDAK DIHARAPKAN"


def classify_path(rel_path, mode):
    """
    mode 'awal'    : keadaan bersih sebelum run murni. Hanya raw, lexicon, dan berkas
                     dipertahankan yang diperbolehkan.
    mode 'berjalan': untuk melanjutkan eksekusi yang sama. Artefak run (runs, shared,
                     logs) juga diperbolehkan.
    """
    parts = rel_path.replace("\\", "/").split("/")
    top = parts[0]
    if top == "data" and len(parts) >= 2 and parts[1] == "raw":
        return DIPERTAHANKAN
    if top == "data" and len(parts) == 1:
        return DIPERTAHANKAN
    if top in ("lexicon", "preserved", "annotations"):
        return DIPERTAHANKAN
    if top == "scrape_state":
        return DIPERBOLEHKAN
    if top in ("runs", "shared", "logs"):
        return DIPERBOLEHKAN if mode == "berjalan" else TIDAK_DIHARAPKAN
    return TIDAK_DIHARAPKAN


def audit_drive_state(root=None, mode="awal", verbose=True, stop_on_duplicate=True,
                      stop_on_unexpected=None):
    """
    Memindai folder proyek, melaporkan nama ganda, dan mengklasifikasikan isi.
    Menghentikan eksekusi (DriveAuditError) bila ada duplikat. Pada mode 'awal',
    berhenti pula bila ada item TIDAK DIHARAPKAN.
    """
    root = root or config.DRIVE_ROOT
    if mode not in ("awal", "berjalan"):
        raise ValueError("mode harus 'awal' atau 'berjalan'")
    if stop_on_unexpected is None:
        stop_on_unexpected = (mode == "awal")
    if not os.path.isdir(root):
        raise DriveAuditError(f"[BERHENTI] Folder proyek tidak ditemukan: {root}")

    duplicates, rows = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if is_duplicate_name(name):
                duplicates.append(rel)
            if os.path.isdir(full):
                continue
            rows.append({"path": rel, "kelas": classify_path(rel, mode),
                         "bytes": os.path.getsize(full)})

    df = pd.DataFrame(rows, columns=["path", "kelas", "bytes"])
    unexpected = df[df["kelas"] == TIDAK_DIHARAPKAN]

    if verbose:
        print(f"Folder proyek : {root}")
        print(f"Mode audit    : {mode}")
        if len(df):
            print(df.groupby("kelas").agg(jumlah=("path", "count"), total_mb=("bytes", lambda s: round(s.sum() / 1e6, 2))).to_string())
        else:
            print("(folder kosong)")
        if duplicates:
            print("\nNama ganda terdeteksi:")
            for d in duplicates:
                print("  -", d)
        if len(unexpected):
            print("\nItem TIDAK DIHARAPKAN (maksimal 50 baris):")
            for p in unexpected["path"].head(50):
                print("  -", p)

    problems = []
    if duplicates and stop_on_duplicate:
        problems.append(f"{len(duplicates)} nama ganda")
    if len(unexpected) and stop_on_unexpected:
        problems.append(f"{len(unexpected)} item tidak diharapkan")
    if problems:
        raise DriveAuditError("[BERHENTI] Audit Drive gagal: " + "; ".join(problems) + ".")
    if verbose:
        print("\n[OK] Audit Drive lulus.")
    return df, duplicates


def bersihkan_artefak(root=None, dry_run=True, konfirmasi=""):
    """
    Menghapus artefak run (runs, shared, logs) dan item tidak diharapkan. DRY RUN
    adalah bawaan: hanya menampilkan daftar. Penghapusan nyata memerlukan
    dry_run=False dan konfirmasi persis 'HAPUS SEKARANG'. Berkas DIPERTAHANKAN
    (raw, lexicon, preserved, annotations) tidak pernah dihapus.
    """
    root = root or config.DRIVE_ROOT
    targets = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if entry == "data":
            # hanya isi non-raw di dalam data yang dihapus
            for sub in os.listdir(full):
                if sub != "raw":
                    targets.append(os.path.join(full, sub))
            continue
        if entry in ("lexicon", "preserved", "annotations"):
            continue
        targets.append(full)

    print("Daftar yang akan dihapus:" if targets else "Tidak ada yang perlu dihapus.")
    for t in targets:
        print("  -", os.path.relpath(t, root))
    if dry_run:
        print("\n[DRY RUN] Tidak ada yang dihapus. Untuk menghapus: dry_run=False dan "
              "konfirmasi='HAPUS SEKARANG'.")
        return targets
    if konfirmasi != "HAPUS SEKARANG":
        raise DriveAuditError("[BERHENTI] Konfirmasi tidak sesuai. Tidak ada yang dihapus.")
    for t in targets:
        if os.path.isdir(t):
            shutil.rmtree(t)
        else:
            os.remove(t)
    print("[OK] Penghapusan selesai.")
    return targets


# ----------------------------------------------------------------------
# Ekspor cadangan (zip)
# ----------------------------------------------------------------------
def export_backup_zip(root, zip_name, include_ext=(".csv", ".json", ".md", ".npz"),
                      exclude_substrings=("ckpt", ".pt"), local_dir=None):
    """
    Mengemas seluruh CSV, JSON, MD, dan NPZ kecil di bawah 'root' ke zip lokal.
    Mengembalikan path zip dan hash. Tidak menulis ke Drive.
    """
    local_dir = local_dir or config.LOCAL_ROOT
    os.makedirs(local_dir, exist_ok=True)
    zip_path = os.path.join(local_dir, zip_name)
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, filenames in os.walk(root):
            for name in sorted(filenames):
                if not name.lower().endswith(include_ext):
                    continue
                full = os.path.join(dirpath, name)
                if any(s in full for s in exclude_substrings):
                    continue
                z.write(full, arcname=os.path.relpath(full, root))
                n += 1
    return {"zip_path": zip_path, "n_files": n, "bytes": os.path.getsize(zip_path),
            "sha256": sha256_file(zip_path)}


def download_if_colab(path):
    if config.IN_COLAB:
        from google.colab import files  # type: ignore
        files.download(path)
    else:
        print(f"(Bukan Colab) berkas cadangan ada di: {path}")


def ids_hash(ids, sort=True):
    """SHA-256 dari daftar review_id (terurut secara bawaan), dipakai untuk sidecar anotasi dan berkas fold."""
    items = [str(i) for i in ids]
    if sort:
        items = sorted(items)
    return sha256_bytes("\n".join(items).encode("utf-8"))
