"""
sensitivity_k.py: sensitivitas jumlah fold K = 3, 5, 10 pada proxy P4 (Subbab 3.5). Hanya split utama.

K = 5 berasal dari proses utama pada eksekusi yang sama (tidak dilatih dua kali).
Tambahan (di luar kalimat proposal): indeks Jaccard himpunan baris ter-flag terhadap K = 5 dan
CI bootstrap berpasangan untuk selisih QWK proxy antar K, berdasarkan OOF tersimpan.
"""

import os

import numpy as np
import pandas as pd

from src import ablation, config, drive_io
from src.metrics import qwk_fast


def jaccard(a, b):
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else float("nan")


def paired_qwk_diff_ci(labels0, preds_x, preds_y, n_boot=None, seed=None):
    """Selisih QWK (x minus y) dengan CI bootstrap berpasangan (indeks resampling sama)."""
    n_boot = n_boot or config.N_BOOT
    seed = config.BOOT_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    n = len(labels0)
    obs = qwk_fast(labels0, preds_x, offset=0) - qwk_fast(labels0, preds_y, offset=0)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[i] = qwk_fast(labels0[idx], preds_x[idx], offset=0) - qwk_fast(labels0[idx], preds_y[idx], offset=0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def run_k_sensitivity(ctx, df_train, main_eval_k5, Ks=None):
    """main_eval_k5: keluaran ablation.run_proxy_evaluation untuk P4 K=5 (proses utama)."""
    ctx.require_main("sensitivitas K")
    Ks = Ks or config.K_SENSITIVITY_VALUES
    evals = {5: main_eval_k5}
    for K in Ks:
        if K == 5:
            continue
        evals[K] = ablation.run_proxy_evaluation(ctx, df_train, config.FINAL_PROXY_ID, K, ctx.k_sensitivity_dir)

    flagged = {}
    for K, ev in evals.items():
        df = ev["detection"]["df"]
        flagged[K] = df.loc[df["is_noise"], "review_id"].astype(str).tolist()
        if K != 5 and not drive_io.is_verified(os.path.join(ctx.k_sensitivity_dir, f"flags__K{K}.csv")):
            drive_io.write_once_csv(df[["review_id", "is_noise"]],
                                    os.path.join(ctx.k_sensitivity_dir, f"flags__K{K}.csv"))

    rows = []
    for K in sorted(evals):
        r = evals[K]["result"]
        info = evals[K]["detection"]["method_info"]
        rows.append({
            "K": K, "qwk": r["qwk"], "mae": r["mae"], "accuracy": r["accuracy"], "off_by_one": r["off_by_one"],
            "ece_kumulatif_rantai_rata": r.get("ece_kumulatif_rantai_rata"),
            "metode_filter_terpilih": r["selected_filter_method"],
            "n_flag_confident_learning": info["confident_learning"],
            "n_flag_prune_by_noise_rate": info["prune_by_noise_rate"],
            "n_estimasi_teoretis": info["estimated_theoretical"],
            "n_flag_metode_terpilih": len(flagged[K]),
            "jaccard_vs_K5": jaccard(flagged[K], flagged[5]),
        })
    table = pd.DataFrame(rows)

    labels0 = evals[5]["labels0"]
    pairs = []
    ks = sorted(evals)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            kx, ky = ks[i], ks[j]
            d, lo, hi = paired_qwk_diff_ci(labels0, evals[kx]["preds0"], evals[ky]["preds0"])
            pairs.append({"K_x": kx, "K_y": ky, "qwk_diff_x_minus_y": d, "ci_95_low": lo, "ci_95_high": hi,
                          "ci_melewati_nol": bool(lo <= 0 <= hi)})
    pairs = pd.DataFrame(pairs)

    drive_io.write_once_csv(table, os.path.join(ctx.k_sensitivity_dir, "k_sensitivity_table.csv"))
    drive_io.write_once_csv(pairs, os.path.join(ctx.k_sensitivity_dir, "k_sensitivity_qwk_pairs.csv"))
    return table, pairs
