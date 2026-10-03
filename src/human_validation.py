"""human_validation.py: validasi manusia atas baris ter-flag P4 (Subbab 3.7). Hanya split utama."""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src import annotation, config, drive_io, sampling

VALID = set(config.HUMAN_VERDICTS)
ANNOTATOR_COLS = ["review_id", "source_app", "review_text", "cleaned_text", "rating"]


def _paths():
    return config.ANNOT_FILES


def report_sample_size(df_noise):
    """Subbab 3.7: n Cochran dari N baris ter-flag P4, dibulatkan ke n perencanaan."""
    c = sampling.cochran_n(len(df_noise))
    print(f"N baris ter-flag P4 (data latih)     : {c['N']}")
    print(f"n0 = Z^2 p (1 - p) / e^2             : {c['n0']:.2f}")
    print(f"n setelah koreksi populasi terbatas  : {c['n_cochran']}")
    print(f"n perencanaan yang dipakai (Subbab 3.7): {config.HUMAN_VALIDATION_N}")
    return c


def create_sample(ctx, df_noise):
    """Alur B: membuat sampel buta kosong dan sidecar. Menolak bila berkas sudah ada."""
    ctx.require_main("validasi manusia")
    path = _paths()["human_validation"]
    annotation.assert_can_create(path, "human_verdict")

    cochran = report_sample_size(df_noise)
    sample, plan = sampling.stratified_human_sample(df_noise, n=config.HUMAN_VALIDATION_N, seed=42)
    print(f"Alokasi per bin jarak ordinal: {plan['take']} (total {plan['total']})")
    if plan["shortfall"]:
        print(f"[PERINGATAN] Kolam bin kurang dari target: {plan['shortfall']} (dilaporkan apa adanya).")

    blind = sample[ANNOTATOR_COLS].copy()
    blind["human_verdict"] = ""
    blind["human_note"] = ""
    drive_io.write_once_csv(blind, path)
    annotation.write_sidecar(path, df_noise["review_id"].astype(str).tolist(),
                             "baris ter-flag P4 K=5 pada data latih split utama",
                             extra={"n_cochran": cochran["n_cochran"], "n_sampel": len(blind),
                                    "alokasi": {str(k): int(v) for k, v in plan["take"].items()}})
    print(f"[OK] Sampel buta dibuat: {path}")
    print("Isi kolom 'human_verdict' dengan: noise / not_noise / ambiguous, lalu simpan di Drive.")
    return blind


def create_second_annotator(fraction=None, seed=99):
    fraction = fraction or config.SECOND_ANNOTATOR_FRACTION
    p1, p2 = _paths()["human_validation"], _paths()["human_validation_annotator2"]
    annotation.assert_can_create(p2, "human_verdict")
    df = pd.read_csv(p1)
    n_sub = max(1, int(round(len(df) * fraction)))
    subset = df.drop(columns=["human_verdict", "human_note"], errors="ignore").sample(n=n_sub, random_state=seed)
    subset["human_verdict"] = ""
    subset["human_note"] = ""
    drive_io.write_once_csv(subset, p2)
    with open(annotation.sidecar_path(p1), "r", encoding="utf-8") as f:
        import json
        parent = json.load(f)
    annotation.write_sidecar(p2, [], "subset sampel validasi manusia untuk penilai kedua",
                             extra={"sha256_review_id_terurut": parent["sha256_review_id_terurut"],
                                    "n_review_id_dasar": parent["n_review_id_dasar"], "n_sampel": n_sub})
    print(f"[OK] Sampel penilai kedua: {n_sub} baris -> {p2}")
    return p2


def verify_annotation(df_noise):
    """Alur A: verifikasi hash sidecar terhadap baris ter-flag P4 pada run saat ini."""
    path = _paths()["human_validation"]
    info = annotation.verify_sidecar(path, df_noise["review_id"].astype(str).tolist(), "validasi manusia")
    print(f"[OK] Hash sidecar validasi manusia cocok (n dasar = {info['n_review_id_dasar']}).")
    p2 = _paths()["human_validation_annotator2"]
    if os.path.exists(p2):
        annotation.verify_subset_of_sample(path, p2)
        print("[OK] Berkas penilai kedua konsisten dengan sampel.")
    return info


def _load_complete(df_noise):
    path = _paths()["human_validation"]
    df = pd.read_csv(path)
    df["human_verdict"] = df["human_verdict"].astype(str).str.strip().str.lower()
    empty = df["human_verdict"].isin(["", "nan", "none"])
    if empty.any():
        raise ValueError(f"Masih ada {int(empty.sum())} baris tanpa 'human_verdict'.")
    bad = ~df["human_verdict"].isin(VALID)
    if bad.any():
        raise ValueError(f"Nilai human_verdict tidak dikenali: {df.loc[bad, 'human_verdict'].unique().tolist()}")
    ref = df_noise[["review_id", "rating_diff"]].copy()
    ref["review_id"] = ref["review_id"].astype(str)
    df["review_id"] = df["review_id"].astype(str)
    df = df.merge(ref, on="review_id", how="left")
    if df["rating_diff"].isna().any():
        raise ValueError("Ada review_id sampel yang tidak ditemukan pada baris ter-flag run ini.")
    return df


def compute_agreement(ctx, df_noise):
    ctx.require_main("validasi manusia")
    verify_annotation(df_noise)
    df = _load_complete(df_noise)
    n = len(df)
    vc = df["human_verdict"].value_counts()
    agree, amb, dis = int(vc.get("noise", 0)), int(vc.get("ambiguous", 0)), int(vc.get("not_noise", 0))
    print(f"Total sampel: {n} | setuju (noise): {agree} ({agree / n * 100:.1f}%) | "
          f"ambigu: {amb} ({amb / n * 100:.1f}%) | tidak setuju (not_noise): {dis} ({dis / n * 100:.1f}%)")

    rows = []
    for diff_val, g in df.groupby("rating_diff"):
        m = len(g)
        c = g["human_verdict"].value_counts()
        rows.append({"rating_diff": int(diff_val), "n_sampel": m,
                     "pct_noise": round(c.get("noise", 0) / m * 100, 1),
                     "pct_not_noise": round(c.get("not_noise", 0) / m * 100, 1),
                     "pct_ambiguous": round(c.get("ambiguous", 0) / m * 100, 1)})
    breakdown = pd.DataFrame(rows).sort_values("rating_diff")
    print(breakdown.to_string(index=False))
    print("Acuan deskriptif (Northcutt dkk., 2021, domain gambar): sekitar 58 persen sampel terbukti isu. "
          "Dibahas deskriptif-komparatif, tanpa ambang lulus atau gagal.")

    summary = pd.DataFrame([{"n": n, "noise": agree, "ambiguous": amb, "not_noise": dis,
                             "agreement_rate": round(agree / n, 4)}])
    d = ctx.validation_dir
    drive_io.write_once_csv(df, os.path.join(d, "human_validation_result.csv"))
    drive_io.write_once_csv(breakdown, os.path.join(d, "human_validation_per_bin.csv"))
    drive_io.write_once_csv(summary, os.path.join(d, "human_validation_summary.csv"))
    return summary, breakdown


def compute_interannotator_kappa(ctx):
    """Cohen's kappa biasa (tiga kategori tidak berjenjang) pada baris overlap."""
    ctx.require_main("validasi manusia")
    p1, p2 = _paths()["human_validation"], _paths()["human_validation_annotator2"]
    m = pd.read_csv(p1).merge(pd.read_csv(p2), on="review_id", suffixes=("_1", "_2"))
    for c in ("human_verdict_1", "human_verdict_2"):
        m[c] = m[c].astype(str).str.strip().str.lower()
    m = m[m["human_verdict_1"].isin(VALID) & m["human_verdict_2"].isin(VALID)]
    if len(m) == 0:
        raise ValueError("Belum ada baris overlap yang lengkap diisi kedua penilai.")
    kappa = cohen_kappa_score(m["human_verdict_1"], m["human_verdict_2"], labels=sorted(VALID))
    print(f"Cohen's kappa antar-penilai (n={len(m)}): {kappa:.4f}")
    drive_io.write_once_csv(pd.DataFrame([{"n_overlap": len(m), "cohen_kappa": kappa}]),
                            os.path.join(ctx.validation_dir, "human_validation_kappa.csv"))
    return kappa


def create_retest_subset(fraction=None, seed=77):
    """Jalur cadangan satu penilai (test-retest, Subbab 3.7)."""
    fraction = fraction or config.TEST_RETEST_FRACTION
    p1, pr = _paths()["human_validation"], _paths()["human_validation_retest"]
    annotation.assert_can_create(pr, "human_verdict")
    df = pd.read_csv(p1)
    n_sub = max(1, int(round(len(df) * fraction)))
    subset = df.drop(columns=["human_verdict", "human_note"], errors="ignore").sample(n=n_sub, random_state=seed)
    subset["human_verdict"] = ""
    subset["human_note"] = ""
    drive_io.write_once_csv(subset, pr)
    return pr


def compute_test_retest(ctx):
    ctx.require_main("validasi manusia")
    p1, pr = _paths()["human_validation"], _paths()["human_validation_retest"]
    m = pd.read_csv(p1).merge(pd.read_csv(pr), on="review_id", suffixes=("_awal", "_retest"))
    for c in ("human_verdict_awal", "human_verdict_retest"):
        m[c] = m[c].astype(str).str.strip().str.lower()
    m = m[m["human_verdict_awal"].isin(VALID) & m["human_verdict_retest"].isin(VALID)]
    kappa = cohen_kappa_score(m["human_verdict_awal"], m["human_verdict_retest"], labels=sorted(VALID))
    print(f"Test-retest kappa (n={len(m)}): {kappa:.4f}. Hanya mengukur konsistensi diri (bias satu penilai).")
    drive_io.write_once_csv(pd.DataFrame([{"n": len(m), "test_retest_kappa": kappa}]),
                            os.path.join(ctx.validation_dir, "human_validation_test_retest.csv"))
    return kappa
