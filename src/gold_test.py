"""
gold_test.py -- Pembentukan dan evaluasi subset uji emas (gold test subset),
sesuai Bab III Subbab 3.9.4. Ini adalah pemeriksaan ketahanan (robustness
check) SEKUNDER, tidak menggantikan data uji penuh dan tidak dipakai untuk
uji hipotesis formal H1-H5 (itu tetap di significance.py, data uji penuh).
"""

import os
import math
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from coral_pytorch.dataset import corn_label_from_logits

from src import config
from src.data import ReviewDataset
from src.models import build_model
from src.metrics import compute_metrics

UNDETERMINED_LABEL = "ND"  # opsi "tidak dapat ditentukan dari teks" (Subbab 3.9.4)


def _cochran_n(N, p=0.5, z=1.96, e=None):
    e = e or config.GOLD_TEST_MARGIN_ERROR
    n0 = (z ** 2) * p * (1 - p) / (e ** 2)
    n = n0 / (1 + (n0 - 1) / N)
    return math.ceil(n)


def sample_gold_test_subset(seed=42, n_override=None):
    """
    Sampling stratified berdasarkan RATING ASLI (bukan rating_diff -- tidak
    semua baris test set punya prediksi proxy), ukuran dihitung via Cochran.
    Kalau n_override diisi manual (mis. karena keterbatasan waktu), penelitian
    ini akan melaporkan n aktual apa adanya di Bab III/IV, bukan target ideal
    yang tidak tercapai (Subbab 1.5, 3.9.4).
    """
    config.require_primary_split("Gold test")
    df_test = pd.read_csv(config.TEST_FILE)
    N = len(df_test)
    n_target = n_override or _cochran_n(N)
    n_actual = min(n_target, N)

    print(f"📐 Populasi data uji: {N} baris")
    print(f"   Target Cochran (e={config.GOLD_TEST_MARGIN_ERROR*100:.0f}%): {_cochran_n(N)} baris")
    print(f"   Ukuran sampel dipakai: {n_actual} baris"
          + (" (diturunkan manual dari target)" if n_override else ""))

    sample, _ = train_test_split(
        df_test, train_size=n_actual, random_state=seed, stratify=df_test["rating"]
    )

    # File BUTA untuk anotator: rating asli disembunyikan, prediksi model
    # apa pun TIDAK disertakan (mencegah evaluasi sirkular, Subbab 3.9.4).
    blind = sample[["review_id", "source_app", "cleaned_text"]].copy()
    blind["human_gold_rating"] = ""
    blind.to_csv(config.GOLD_TEST_SAMPLE_FILE, index=False)

    # Referensi internal (rating asli) -- BUKAN diberikan ke anotator, hanya
    # dipakai peneliti untuk pemeriksaan pelengkap, bukan bagian dari isian anotator.
    internal_path = config.GOLD_TEST_SAMPLE_FILE.replace(".csv", "_internal_reference.csv")
    sample[["review_id", "rating"]].to_csv(internal_path, index=False)

    print(f"\n📝 Sample uji emas (BUTA): {len(blind)} baris -> {config.GOLD_TEST_SAMPLE_FILE}")
    print("   -> Isi 'human_gold_rating' manual (1-5), atau isi 'ND' kalau rating yang")
    print("      wajar tampak bergantung pada info eksternal di luar teks (Subbab 3.9.4),")
    print("      lalu jalankan gold_test.evaluate_on_gold_subset() setelah training")
    print("      6 skenario (Cell 20) dan uji signifikansi (Cell 22) selesai.")
    return sample


def export_second_annotator_gold_subset(fraction=None, seed=99):
    config.require_primary_split("Gold test")
    fraction = fraction or config.SECOND_ANNOTATOR_FRACTION
    if not os.path.exists(config.GOLD_TEST_SAMPLE_FILE):
        print("⚠️ Sample uji emas belum ada -- jalankan sample_gold_test_subset() dulu.")
        return None

    df = pd.read_csv(config.GOLD_TEST_SAMPLE_FILE)
    n_sub = max(1, int(round(len(df) * fraction)))
    subset = df.drop(columns=["human_gold_rating"], errors="ignore").sample(n=n_sub, random_state=seed)
    subset["human_gold_rating"] = ""
    out_path = config.GOLD_TEST_SAMPLE_FILE.replace(".csv", "_annotator2.csv")
    subset.to_csv(out_path, index=False)
    print(f"📝 Sample anotator kedua (uji emas): {n_sub} baris -> {out_path}")
    return out_path


def compute_gold_interannotator_kappa():
    """Quadratic-weighted kappa (bukan kappa polos) karena rating di sini
    ordinal 1-5, bukan kategori noise/not_noise/ambiguous."""
    config.require_primary_split("Gold test")
    path2 = config.GOLD_TEST_SAMPLE_FILE.replace(".csv", "_annotator2.csv")
    if not (os.path.exists(config.GOLD_TEST_RESULT_FILE) and os.path.exists(path2)):
        print("⚠️ File anotator 1 (hasil final) dan/atau anotator 2 belum lengkap.")
        return None

    df1 = pd.read_csv(config.GOLD_TEST_RESULT_FILE)
    df2 = pd.read_csv(path2)

    # Filter/drop opsi ND sebelum menghitung kappa
    df1 = df1[df1["human_gold_rating"].astype(str).str.strip().str.upper() != UNDETERMINED_LABEL]
    df2 = df2[df2["human_gold_rating"].astype(str).str.strip().str.upper() != UNDETERMINED_LABEL]

    merged = df1.merge(df2, on="review_id", suffixes=("_1", "_2"))
    merged["human_gold_rating_1"] = pd.to_numeric(merged["human_gold_rating_1"], errors="coerce")
    merged["human_gold_rating_2"] = pd.to_numeric(merged["human_gold_rating_2"], errors="coerce")
    merged = merged.dropna(subset=["human_gold_rating_1", "human_gold_rating_2"])

    if len(merged) == 0:
        print("⚠️ Belum ada baris overlap yang lengkap.")
        return None

    kappa = cohen_kappa_score(
        merged["human_gold_rating_1"].astype(int),
        merged["human_gold_rating_2"].astype(int),
        weights="quadratic",
    )
    print(f"🤝 Quadratic-weighted kappa anotator (uji emas, n={len(merged)}): {kappa:.4f}")
    return kappa


def _load_gold_labels():
    if not os.path.exists(config.GOLD_TEST_RESULT_FILE):
        print("⚠️ File sample uji emas belum ada. Jalankan sample_gold_test_subset() dulu.")
        return None

    df = pd.read_csv(config.GOLD_TEST_RESULT_FILE)
    df["human_gold_rating"] = df["human_gold_rating"].astype(str).str.strip()

    is_und = df["human_gold_rating"].str.upper() == UNDETERMINED_LABEL
    n_und = int(is_und.sum())
    pct_und = n_und / len(df) * 100 if len(df) else 0
    print(f"🚫 Baris 'tidak dapat ditentukan dari teks': {n_und} ({pct_und:.2f}%)")
    pd.DataFrame([{"n_undetermined": n_und, "pct_undetermined": round(pct_und, 2),
                    "n_total": len(df)}]).to_csv(
        os.path.join(config.RESULTS_DIR, "gold_test_undetermined_summary.csv"), index=False
    )

    df_det = df[~is_und].copy()
    df_det["human_gold_rating"] = pd.to_numeric(df_det["human_gold_rating"], errors="coerce")

    n_missing = df_det["human_gold_rating"].isna().sum()
    if n_missing > 0:
        print(f"⚠️ Masih ada {n_missing} baris kosong/tidak valid (di luar opsi '{UNDETERMINED_LABEL}').")
        return None
    invalid = ~df_det["human_gold_rating"].isin([1, 2, 3, 4, 5])
    if invalid.any():
        print(f"⚠️ Ada {invalid.sum()} nilai di luar rentang 1-5 (dan bukan '{UNDETERMINED_LABEL}').")
        return None

    return df_det


def evaluate_on_gold_subset(scenarios):
    """
    Evaluasi keenam skenario (checkpoint dari Cell 20) pada subset uji emas.
    Melaporkan metrik + arah keunggulan/effect size (bootstrap CI), TANPA uji
    Wilcoxon dan koreksi Holm-Bonferroni formal kedua, sesuai Subbab 3.9.4 --
    ukuran sampel di sini tidak dirancang untuk uji berdaya penuh.
    """
    config.require_primary_split("Gold test")
    df_gold = _load_gold_labels()
    if df_gold is None:
        return None

    gold_labels = df_gold["human_gold_rating"].astype(int).values
    texts = df_gold["cleaned_text"].tolist()
    n = len(df_gold)
    print(f"🏅 Evaluasi 6 skenario pada subset uji emas (n={n}).")

    all_metrics_rows = []
    preds_per_seed_gold = {}

    for s in scenarios:
        name, loss_type = s["name"], s["loss"]
        preds_per_seed_gold[name] = {}
        seed_metrics = {"mae": [], "rmse": [], "accuracy": [], "off_by_one": [], "qwk": []}

        for seed in config.SEED_LIST:
            ckpt_path = os.path.join(config.MODEL_CKPT_DIR, name, f"seed{seed}_best.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint {ckpt_path} tidak ditemukan -- pastikan training 6 "
                    f"skenario (Cell 20) sudah selesai sebelum evaluasi gold subset."
                )
            model = build_model(loss_type)
            model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
            model.eval()

            loader = DataLoader(
                ReviewDataset(texts, [1] * n),  # label dummy, hanya untuk prediksi
                batch_size=config.BATCH_SIZE, shuffle=False,
            )
            preds = []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(config.DEVICE)
                    attention_mask = batch["attention_mask"].to(config.DEVICE)
                    logits = model(input_ids, attention_mask)
                    p = (torch.argmax(logits, dim=1) if loss_type == "ce"
                         else corn_label_from_logits(logits)).cpu().numpy() + 1
                    preds.extend(p)
            preds = np.array(preds)
            preds_per_seed_gold[name][seed] = preds

            m = compute_metrics(gold_labels, preds)
            for k in seed_metrics:
                seed_metrics[k].append(m[k])

        all_metrics_rows.append({
            "Model": name,
            "MAE": np.mean(seed_metrics["mae"]),
            "RMSE": np.mean(seed_metrics["rmse"]),
            "Accuracy": np.mean(seed_metrics["accuracy"]),
            "Off_by_one": np.mean(seed_metrics["off_by_one"]),
            "QWK": np.mean(seed_metrics["qwk"]),
            "n_gold": n,
        })

    df_result = pd.DataFrame(all_metrics_rows).sort_values("MAE")
    df_result.to_csv(config.GOLD_TEST_EVAL_FILE, index=False)
    print(f"\n💾 Hasil evaluasi subset uji emas -> {config.GOLD_TEST_EVAL_FILE}")
    print(df_result.to_string(index=False))

    from src.significance import PRE_REGISTERED_HYPOTHESES, aggregate_errors_across_seeds, bootstrap_effect_size

    aggregated_gold_errors = aggregate_errors_across_seeds(gold_labels, preds_per_seed_gold)
    rows = []
    for hyp_name, model_a, model_b in PRE_REGISTERED_HYPOTHESES:
        es = bootstrap_effect_size(aggregated_gold_errors, model_a, model_b)
        arah = "A lebih baik" if es["mean_diff"] < 0 else ("B lebih baik" if es["mean_diff"] > 0 else "seri")
        rows.append({"hypothesis": hyp_name, **es, "arah": arah})

    df_effect = pd.DataFrame(rows)
    df_effect.to_csv(config.GOLD_TEST_EFFECT_SIZE_FILE, index=False)
    print(f"\n💾 Arah & effect size subset uji emas -> {config.GOLD_TEST_EFFECT_SIZE_FILE}")
    print(df_effect.to_string(index=False))
    print("\n📌 Bandingkan arah di atas dengan hasil H1-H5 data uji penuh (Cell 22) --")
    print("   konsisten memperkuat validitas, tidak konsisten dilaporkan sebagai temuan")
    print("   yang membatasi generalisasi (Subbab 3.9.4).")

    return df_result, df_effect


def write_rubric(overwrite=False):
    """Rubrik operasional singkat per kelas rating (§3.10). Hanya klaim 'rubrik disediakan'
    di laporan jika anotasi memang memakainya."""
    config.require_primary_split("Gold test")
    if os.path.exists(config.GOLD_TEST_RUBRIC_FILE) and not overwrite:
        print(f"ℹ️ Rubrik sudah ada: {config.GOLD_TEST_RUBRIC_FILE}")
        return
    text = """# Rubrik anotasi uji emas (baca HANYA teks ulasan; rating asli disembunyikan)
| Rating | Kriteria operasional |
|---|---|
| 1 | Keluhan berat/tegas: aplikasi tidak berfungsi, merasa dirugikan, kemarahan, tidak merekomendasikan; tanpa apresiasi sama sekali. |
| 2 | Dominan negatif: masalah signifikan, tetapi ada nada kompromi atau apresiasi kecil. |
| 3 | Campuran/netral: pujian dan keluhan seimbang, atau teks tidak bermuatan evaluatif yang jelas. |
| 4 | Dominan positif: puas dan merekomendasikan, dengan keluhan kecil atau saran perbaikan. |
| 5 | Sangat positif: pujian tegas tanpa keluhan material. |
| ND | Tidak dapat ditentukan dari teks: rating yang wajar bergantung informasi eksternal (mis. status transaksi). |
Aturan: jangan menebak rating asli; jangan memakai prediksi model apa pun; isi ND bila ragu karena info eksternal."""
    with open(config.GOLD_TEST_RUBRIC_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"📝 Rubrik -> {config.GOLD_TEST_RUBRIC_FILE}")