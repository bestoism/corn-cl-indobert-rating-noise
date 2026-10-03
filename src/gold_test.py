"""
gold_test.py: subset uji emas (Subbab 3.10). Hanya split utama.

Evaluasi bekerja dari prediksi tersimpan (predictions/*.csv), bukan dari checkpoint.
Label uji emas final adalah label penilai pertama. Penilai kedua dipakai hanya untuk kappa
berbobot kuadratik pada baris overlap (lihat catatan perbedaan terhadap kalimat proposal).
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

from src import annotation, config, drive_io, sampling
from src.metrics import compute_metrics
from src import significance


def _paths():
    return config.ANNOT_FILES


def create_sample(ctx, seed=None):
    """Alur B: sampel buta kosong stratified berdasarkan rating asli. Menolak bila berkas sudah ada."""
    ctx.require_main("uji emas")
    path = _paths()["gold"]
    annotation.assert_can_create(path, "human_gold_rating")
    seed = config.GOLD_SAMPLE_SEED if seed is None else seed

    df_test = drive_io.read_csv_verified(ctx.split_file("test"))
    N = len(df_test)
    c = sampling.cochran_n(N, e=config.GOLD_MARGIN_ERROR)
    n = min(c["n_cochran"], N)
    print(f"Populasi data uji N={N} | n0={c['n0']:.2f} | n Cochran (e={config.GOLD_MARGIN_ERROR * 100:.0f}%)={c['n_cochran']}")

    sample, _ = train_test_split(df_test, train_size=n, random_state=seed, stratify=df_test["rating"])
    blind = sample[["review_id", "source_app", "cleaned_text"]].copy()
    blind["human_gold_rating"] = ""
    drive_io.write_once_csv(blind, path)
    annotation.write_sidecar(path, df_test["review_id"].astype(str).tolist(),
                             "seluruh data uji split utama",
                             extra={"n_cochran": c["n_cochran"], "n_sampel": len(blind), "N_populasi": N})
    print(f"[OK] Sampel uji emas (buta) dibuat: {path}")
    print("Isi 'human_gold_rating' dengan 1 sampai 5, atau 'ND' bila rating wajar bergantung pada "
          "informasi di luar teks. Rating asli dan prediksi model tidak ditampilkan.")
    return blind


def create_second_annotator(fraction=None, seed=None):
    fraction = fraction or config.SECOND_ANNOTATOR_FRACTION
    seed = config.GOLD_SECOND_ANNOTATOR_SEED if seed is None else seed
    p1, p2 = _paths()["gold"], _paths()["gold_annotator2"]
    annotation.assert_can_create(p2, "human_gold_rating")
    df = pd.read_csv(p1)
    n_sub = max(1, int(round(len(df) * fraction)))
    subset = df.drop(columns=["human_gold_rating"], errors="ignore").sample(n=n_sub, random_state=seed)
    subset["human_gold_rating"] = ""
    drive_io.write_once_csv(subset, p2)
    import json
    with open(annotation.sidecar_path(p1), "r", encoding="utf-8") as f:
        parent = json.load(f)
    annotation.write_sidecar(p2, [], "subset sampel uji emas untuk penilai kedua",
                             extra={"sha256_review_id_terurut": parent["sha256_review_id_terurut"],
                                    "n_review_id_dasar": parent["n_review_id_dasar"], "n_sampel": n_sub})
    print(f"[OK] Sampel penilai kedua uji emas: {n_sub} baris -> {p2}")
    return p2


def verify_annotation(ctx):
    df_test = drive_io.read_csv_verified(ctx.split_file("test"))
    info = annotation.verify_sidecar(_paths()["gold"], df_test["review_id"].astype(str).tolist(), "uji emas")
    print(f"[OK] Hash sidecar uji emas cocok dengan data uji run ini (n dasar = {info['n_review_id_dasar']}).")
    p2 = _paths()["gold_annotator2"]
    if os.path.exists(p2):
        annotation.verify_subset_of_sample(_paths()["gold"], p2)
    return info


def load_gold_labels(ctx):
    """Memuat label emas, mengeluarkan 'ND', dan melaporkan proporsinya."""
    df = pd.read_csv(_paths()["gold"])
    df["human_gold_rating"] = df["human_gold_rating"].astype(str).str.strip()
    is_und = df["human_gold_rating"].str.upper() == config.UNDETERMINED_LABEL
    n_und = int(is_und.sum())
    pct = n_und / len(df) * 100 if len(df) else 0.0
    print(f"Baris 'tidak dapat ditentukan dari teks': {n_und} ({pct:.2f}%)")
    drive_io.write_once_csv(pd.DataFrame([{"n_undetermined": n_und, "pct_undetermined": round(pct, 2),
                                           "n_total": len(df)}]),
                            os.path.join(ctx.gold_dir, "gold_test_undetermined_summary.csv"))
    det = df[~is_und].copy()
    det["human_gold_rating"] = pd.to_numeric(det["human_gold_rating"], errors="coerce")
    if det["human_gold_rating"].isna().any():
        raise ValueError(f"Masih ada {int(det['human_gold_rating'].isna().sum())} baris kosong atau tidak valid "
                         f"(di luar '{config.UNDETERMINED_LABEL}').")
    if (~det["human_gold_rating"].isin([1, 2, 3, 4, 5])).any():
        raise ValueError("Ada nilai di luar rentang 1 sampai 5.")
    det["human_gold_rating"] = det["human_gold_rating"].astype(int)
    det["review_id"] = det["review_id"].astype(str)
    return det


def compute_gold_kappa(ctx):
    """Kappa berbobot kuadratik antar penilai (rating ordinal 1 sampai 5), baris ND dikeluarkan."""
    p1, p2 = _paths()["gold"], _paths()["gold_annotator2"]
    d1, d2 = pd.read_csv(p1), pd.read_csv(p2)
    for d in (d1, d2):
        d["human_gold_rating"] = d["human_gold_rating"].astype(str).str.strip()
    d1 = d1[d1["human_gold_rating"].str.upper() != config.UNDETERMINED_LABEL]
    d2 = d2[d2["human_gold_rating"].str.upper() != config.UNDETERMINED_LABEL]
    m = d1.merge(d2, on="review_id", suffixes=("_1", "_2"))
    for c in ("human_gold_rating_1", "human_gold_rating_2"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["human_gold_rating_1", "human_gold_rating_2"])
    if len(m) == 0:
        raise ValueError("Belum ada baris overlap lengkap pada kedua penilai uji emas.")
    kappa = cohen_kappa_score(m["human_gold_rating_1"].astype(int), m["human_gold_rating_2"].astype(int),
                              weights="quadratic")
    print(f"Kappa berbobot kuadratik antar-penilai uji emas (n={len(m)}): {kappa:.4f}")
    drive_io.write_once_csv(pd.DataFrame([{"n_overlap": len(m), "quadratic_weighted_kappa": kappa}]),
                            os.path.join(ctx.gold_dir, "gold_test_kappa.csv"))
    return kappa


def evaluate_on_gold(ctx, seeds):
    """Metrik enam skenario pada subset emas dan arah serta effect size MAE (tanpa Wilcoxon kedua)."""
    ctx.require_main("uji emas")
    verify_annotation(ctx)
    gold = load_gold_labels(ctx)
    true_all, preds_all, ids = significance.load_predictions(ctx, seeds)
    pos = {rid: i for i, rid in enumerate(ids)}
    missing = [r for r in gold["review_id"] if r not in pos]
    if missing:
        raise RuntimeError(f"[BERHENTI] {len(missing)} review_id emas tidak ada pada data uji.")
    idx = np.array([pos[r] for r in gold["review_id"]])
    gold_labels = gold["human_gold_rating"].values
    n = len(gold)
    print(f"Evaluasi enam skenario pada subset uji emas (n={n}).")

    rows, preds_gold = [], {}
    for s in config.SCENARIOS:
        name = s["name"]
        preds_gold[name] = {sd: preds_all[name][sd][idx] for sd in seeds}
        ms = [compute_metrics(gold_labels, preds_gold[name][sd]) for sd in seeds]
        rows.append({"Model": name, "n_gold": n, "n_seed": len(seeds),
                     **{k.upper() if k in ("mae", "rmse", "qwk") else k: float(np.mean([m[k] for m in ms]))
                        for k in ("mae", "rmse", "accuracy", "off_by_one", "qwk")}})
    result = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    agg = significance.aggregate_errors_across_seeds(gold_labels, preds_gold)
    eff_rows = []
    for hyp, a, b in config.HYPOTHESES:
        es = significance.bootstrap_effect_size(agg, a, b)
        arah = "A lebih baik" if es["mean_diff"] < 0 else ("B lebih baik" if es["mean_diff"] > 0 else "seri")
        eff_rows.append({"hypothesis": hyp, **es, "arah": arah,
                         "ci_melewati_nol": bool(es["ci_95_low"] <= 0 <= es["ci_95_high"])})
    effect = pd.DataFrame(eff_rows)

    drive_io.write_once_csv(result, os.path.join(ctx.gold_dir, "gold_test_evaluation.csv"))
    drive_io.write_once_csv(effect, os.path.join(ctx.gold_dir, "gold_test_effect_sizes.csv"))
    print(result.to_string(index=False))
    print(effect.to_string(index=False))
    print("Bandingkan arah di atas dengan hasil H1 sampai H5 pada data uji penuh. Arah yang konsisten "
          "menguatkan kesimpulan, arah yang tidak konsisten dilaporkan sebagai temuan yang membatasi generalisasi.")
    return result, effect
