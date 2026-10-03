"""
consistency.py: aturan konsistensi lintas split partisi (Subbab 3.4, 3.12).

Aturan (ditulis juga di markdown notebook sebelum sel sensitivitas):
  - Arah dinormalisasi agar nilai positif berarti model A lebih baik. Untuk MAE,
    selisih (A minus B) negatif berarti A lebih baik. Untuk QWK, selisih positif
    berarti A lebih baik.
  - KONSISTEN: arah titik taksiran sama pada seluruh split, dan CI 95 persen bootstrap
    tidak melewati nol pada sekurangnya dua dari tiga split.
  - TIDAK KONSISTEN: arah berbalik pada sekurangnya satu split (tidak seragam), atau
    CI melewati nol pada sekurangnya dua split.
  - Dengan tiga split kedua kategori itu saling lengkap dan tidak tumpang tindih.
  - Hasil Wilcoxon dilaporkan per split berdampingan. P-value tidak digabung antar split.
"""

import numpy as np
import pandas as pd

METRIC_BETTER_IF_NEGATIVE = {"MAE": True, "QWK_ensemble": False, "QWK_per_seed": False}


def normalize_effect(metric, diff, ci_low, ci_high):
    """Mengembalikan (efek, low, high) sehingga efek > 0 berarti model A lebih baik."""
    if METRIC_BETTER_IF_NEGATIVE[metric]:
        return -diff, -ci_high, -ci_low
    return diff, ci_low, ci_high


def direction_label(metric, diff):
    eff, _, _ = normalize_effect(metric, diff, diff, diff)
    if eff > 0:
        return "A lebih baik"
    if eff < 0:
        return "B lebih baik"
    return "seri"


def evaluate_consistency(long_df, expected_splits=3):
    """
    long_df kolom wajib: split_seed, hypothesis, metric, diff, ci_low, ci_high.
    Keluaran: satu baris per (hypothesis, metric) dengan arah, status CI, dan putusan.
    """
    rows = []
    for (hyp, metric), g in long_df.groupby(["hypothesis", "metric"], sort=False):
        g = g.sort_values("split_seed")
        eff_rows = []
        for _, r in g.iterrows():
            eff, lo, hi = normalize_effect(metric, r["diff"], r["ci_low"], r["ci_high"])
            eff_rows.append({
                "split_seed": int(r["split_seed"]),
                "arah": "A" if eff > 0 else ("B" if eff < 0 else "seri"),
                "ci_tidak_melewati_nol": bool(lo > 0 or hi < 0),
            })
        n_splits = len(eff_rows)
        arah_set = {e["arah"] for e in eff_rows}
        arah_seragam = len(arah_set) == 1 and "seri" not in arah_set
        n_excl = sum(e["ci_tidak_melewati_nol"] for e in eff_rows)
        n_cross = n_splits - n_excl

        if n_splits < expected_splits:
            putusan = "TIDAK DAPAT DINILAI (split kurang)"
        elif arah_seragam and n_excl >= 2:
            putusan = "KONSISTEN"
        else:
            putusan = "TIDAK KONSISTEN"

        row = {"hypothesis": hyp, "metric": metric, "n_split": n_splits,
               "arah_seragam": arah_seragam,
               "n_split_ci_tidak_melewati_nol": n_excl,
               "n_split_ci_melewati_nol": n_cross, "putusan": putusan}
        for e in eff_rows:
            row[f"arah_split{e['split_seed']}"] = e["arah"]
            row[f"ci_excl_nol_split{e['split_seed']}"] = e["ci_tidak_melewati_nol"]
        if arah_seragam:
            row["arah_dominan"] = "A lebih baik" if "A" in arah_set else "B lebih baik"
        else:
            row["arah_dominan"] = "tidak seragam"
        rows.append(row)
    return pd.DataFrame(rows)
