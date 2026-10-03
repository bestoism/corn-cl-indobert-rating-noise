"""diagnostics.py: distribusi jarak ordinal pada baris noise per kelas rating asli (Subbab 3.8)."""

import numpy as np
import pandas as pd

from src import config


def noise_distance_by_class(df_flagged, quality_df, threshold=None):
    """
    df_flagged: seluruh data latih dengan kolom rating, rating_diff, is_noise (metode terpilih).
    quality_df: keluaran calibration.per_class_quality untuk proxy yang sama.
    Menghasilkan tabel per kelas rating asli: jumlah baris, jumlah ter-flag, distribusi jarak
    ordinal 1 sampai 4 pada baris ter-flag, persentase jarak >= ambang severity, serta recall
    dan precision proxy sebagai rujukan pemeriksaan confound.
    """
    threshold = config.SEVERITY_THRESHOLD if threshold is None else threshold
    rows = []
    for r in range(1, config.NUM_CLASSES + 1):
        cls = df_flagged[df_flagged["rating"] == r]
        noisy = cls[cls["is_noise"]]
        n_cls, n_noise = len(cls), len(noisy)
        row = {"rating": r, "n_baris_kelas": n_cls, "n_flag": n_noise,
               "pct_flag_dalam_kelas": 100.0 * n_noise / n_cls if n_cls else float("nan")}
        for d in range(1, 5):
            row[f"n_jarak{d}"] = int((noisy["rating_diff"].clip(upper=4) == d).sum())
        n_sev = int((noisy["rating_diff"] >= threshold).sum())
        row["n_jarak_ge_ambang"] = n_sev
        row["pct_jarak_ge_ambang_dari_flag"] = 100.0 * n_sev / n_noise if n_noise else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)

    q = quality_df[["rating", "recall", "precision", "mae"]].rename(
        columns={"recall": "recall_proxy", "precision": "precision_proxy", "mae": "mae_proxy"})
    out = out.merge(q, on="rating", how="left")

    # Penanda confound: kelas yang masuk dua teratas pada persentase jarak berat sekaligus
    # dua terbawah pada recall proxy (operasionalisasi "bertepatan" pada Subbab 3.8).
    valid = out.dropna(subset=["pct_jarak_ge_ambang_dari_flag", "recall_proxy"])
    top_sev = set(valid.nlargest(2, "pct_jarak_ge_ambang_dari_flag")["rating"])
    low_rec = set(valid.nsmallest(2, "recall_proxy")["rating"])
    out["penanda_confound_recall"] = out["rating"].apply(lambda r: "YA" if (r in top_sev and r in low_rec) else "tidak")
    low_prec = set(valid.dropna(subset=["precision_proxy"]).nsmallest(2, "precision_proxy")["rating"])
    out["penanda_confound_precision"] = out["rating"].apply(lambda r: "YA" if (r in top_sev and r in low_prec) else "tidak")
    return out
