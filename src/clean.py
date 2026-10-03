"""
clean.py: Confident learning untuk deteksi label noise (Subbab 3.6, 3.8).

detect_noise() murni (tanpa penulisan) dan dipakai untuk seluruh proxy dan nilai K.
Hanya P4 pada K = 5 yang boleh memanggil export_p4_outputs(), yang menulis tiga varian data
latih dan daftar baris ter-flag. P1 sampai P3 hanya menghasilkan metrik, diagnostik, dan
jumlah baris ter-flag.
"""

import os

import cleanlab
import numpy as np
import pandas as pd
from cleanlab.count import num_label_issues

from src import config, drive_io
from src.metrics import compute_metrics
from src.preprocess import drop_text_rating_conflicts


def select_filter_method(labels, pred_probs, results):
    """
    Memilih metode filter yang jumlah baris ter-flag-nya paling dekat dengan estimasi
    num_label_issues() varian off_diagonal_calibrated (Subbab 3.6). Varian bawaan
    off_diagonal identik dengan hitungan confident_learning sehingga tidak diskriminatif.
    """
    estimated_n = int(num_label_issues(labels=labels, pred_probs=pred_probs,
                                       estimation_method="off_diagonal_calibrated"))
    counts = {m: int(issues.sum()) for m, issues in results.items()}
    selected = min(counts, key=lambda m: abs(counts[m] - estimated_n))
    return selected, {**counts, "estimated_theoretical": estimated_n}


def detect_noise(df_train, pred_probs, verbose=True):
    """
    df_train: data latih dengan kolom rating (1 sampai 5). pred_probs: OOF (n, 5).
    Mengembalikan dict: df (dengan kolom prediksi dan flag), proxy_metrics, selected_method,
    method_info, labels0, preds0.
    """
    pred_probs = np.asarray(pred_probs)
    if pred_probs.shape != (len(df_train), config.NUM_CLASSES):
        raise ValueError(f"Bentuk pred_probs {pred_probs.shape} tidak sesuai, "
                         f"diharapkan ({len(df_train)}, {config.NUM_CLASSES}).")
    labels0 = df_train["rating"].values - 1
    preds0 = np.argmax(pred_probs, axis=1)
    proxy_metrics = compute_metrics(labels0, preds0)

    df = df_train.copy()
    df["predicted_rating"] = preds0 + 1
    df["rating_diff"] = (df["rating"] - df["predicted_rating"]).abs()

    results = {}
    for method in config.CLEANLAB_FILTER_METHODS:
        results[method] = cleanlab.filter.find_label_issues(
            labels=labels0, pred_probs=pred_probs, filter_by=method,
            min_examples_per_class=config.MIN_EXAMPLES_PER_CLASS)
    selected, info = select_filter_method(labels0, pred_probs, results)

    df["is_noise"] = results[selected]
    for method, issues in results.items():
        df[f"is_noise__{method}"] = issues
    df["is_noise_severe"] = df["is_noise"] & (df["rating_diff"] >= config.SEVERITY_THRESHOLD)

    if verbose:
        for m, n in info.items():
            tag = "  <-- DIPILIH" if m == selected else ""
            if m != "estimated_theoretical":
                print(f"   {m}: {n} baris ({n / len(df) * 100:.2f}%), selisih terhadap estimasi: "
                      f"{abs(n - info['estimated_theoretical'])}{tag}")
        print(f"   estimasi teoretis (off_diagonal_calibrated): {info['estimated_theoretical']} baris")
    return {"df": df, "proxy_metrics": proxy_metrics, "selected_method": selected,
            "method_info": info, "labels0": labels0, "preds0": preds0}


def export_p4_outputs(ctx, detection, proxy_id, K):
    """
    Menulis keluaran P4 (K = 5): daftar baris ter-flag, tiga varian data latih, dan ringkasan
    jumlah baris. Menolak dijalankan untuk proxy selain P4 atau K selain 5.
    """
    if proxy_id != config.FINAL_PROXY_ID or K != config.PROXY_CV_FOLDS:
        raise ValueError(
            f"[BERHENTI] Varian data bersih hanya dihasilkan oleh P4 pada K={config.PROXY_CV_FOLDS}. "
            f"Diminta proxy_id={proxy_id}, K={K}. P1 sampai P3 hanya menghasilkan metrik dan diagnostik.")
    df = detection["df"]
    df_noise = df[df["is_noise"]].copy()

    print("Distribusi rating_diff pada baris noise (sebelum resolusi):")
    print(df_noise["rating_diff"].value_counts().sort_index().to_string())

    n_before = len(df)
    df_resolved = drop_text_rating_conflicts(df, text_col="cleaned_text")
    n_after = len(df_resolved)
    print(f"Resolusi konflik teks identik (pasca deteksi): {n_before} -> {n_after} baris")

    raw = df_resolved.copy()
    hard = df_resolved[~df_resolved["is_noise"]].copy()
    severe = df_resolved[~df_resolved["is_noise_severe"]].copy()
    drop_cols = [c for c in raw.columns if c.startswith("is_noise") or c in ("predicted_rating", "rating_diff")]

    drive_io.write_once_csv(raw.drop(columns=drop_cols, errors="ignore"), ctx.variant_file("raw"))
    drive_io.write_once_csv(hard.drop(columns=drop_cols, errors="ignore"), ctx.variant_file("hard"))
    drive_io.write_once_csv(severe.drop(columns=drop_cols, errors="ignore"), ctx.variant_file("severe"))

    flag_cols = ["review_id", "rating", "predicted_rating", "rating_diff", "is_noise",
                 "is_noise_severe"] + [c for c in df.columns if c.startswith("is_noise__")]
    drive_io.write_once_csv(df[flag_cols], os.path.join(ctx.noise_dir, "flags_P4_K5.csv"))
    noise_cols = ["review_id", "source_app", "review_text", "cleaned_text", "rating",
                  "predicted_rating", "rating_diff"]
    drive_io.write_once_csv(df_noise[noise_cols], os.path.join(ctx.noise_dir, "flagged_rows_P4_K5.csv"))

    stats = {
        "split_seed": ctx.split_seed,
        "n_latih_awal": n_before, "n_setelah_resolusi_konflik": n_after,
        "n_dibuang_resolusi_konflik": n_before - n_after,
        "metode_filter_terpilih": detection["selected_method"],
        "n_flag_noise_sebelum_resolusi": int(len(df_noise)),
        "n_raw": len(raw), "n_hard": len(hard), "n_severe": len(severe),
        "n_dibuang_hard": int(df_resolved["is_noise"].sum()),
        "n_dibuang_severe": int(df_resolved["is_noise_severe"].sum()),
        **{f"n_{k}": v for k, v in detection["method_info"].items()},
    }
    drive_io.write_once_csv(pd.DataFrame([stats]), os.path.join(ctx.noise_dir, "variant_summary.csv"))
    print(f"Raw {len(raw)} | Hard-prune: buang {stats['n_dibuang_hard']}, sisa {len(hard)} | "
          f"Severity-aware: buang {stats['n_dibuang_severe']}, sisa {len(severe)}")
    return df_noise, stats
