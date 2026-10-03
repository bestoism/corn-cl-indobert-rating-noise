"""
significance.py: uji signifikansi dan effect size untuk lima hipotesis pre-registered (Subbab 3.12).

Seluruh perhitungan bekerja dari berkas prediksi per sampel yang tersimpan saat pelatihan
(predictions/{skenario}__seed{seed}.csv), bukan dari checkpoint.

Tiga jalur pelaporan:
  1. Wilcoxon Signed-Rank atas rata-rata absolute error per sampel (agregasi seed), koreksi
     Holm-Bonferroni atas lima uji, dua arah, zero_method='zsplit'. Effect size MAE dengan CI
     bootstrap 95 persen.
  2. QWK ENSEMBLE (definisi utama): prediksi tiga seed dirata-rata lalu dibulatkan ke kelas
     terdekat, selisih QWK dengan CI bootstrap 95 persen.
  3. QWK PER SEED (analisis sensitivitas, TAMBAHAN): QWK dihitung per seed lalu dirata-rata
     antar seed. CI bootstrap memakai indeks resampling yang sama untuk semua seed (berpasangan).
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src import config, drive_io
from src.metrics import qwk_fast
from src.train import build_final_results_table


# ----------------------------------------------------------------------
# Memuat prediksi tersimpan
# ----------------------------------------------------------------------
def load_predictions(ctx, seeds, scenarios=None):
    scenarios = scenarios or config.SCENARIOS
    df_test = drive_io.read_csv_verified(ctx.split_file("test"))
    ref_ids = df_test["review_id"].astype(str).values
    true_labels = df_test["rating"].values.astype(int)

    preds_per_seed = {}
    for s in scenarios:
        preds_per_seed[s["name"]] = {}
        for seed in seeds:
            path = ctx.scenario_pred_file(s["name"], seed)
            df = drive_io.read_csv_verified(path)
            if not np.array_equal(df["review_id"].astype(str).values, ref_ids):
                raise RuntimeError(f"[BERHENTI] review_id pada {path} tidak selaras dengan data uji.")
            if not np.array_equal(df["y_true"].values.astype(int), true_labels):
                raise RuntimeError(f"[BERHENTI] y_true pada {path} berbeda dari data uji.")
            preds_per_seed[s["name"]][seed] = df["y_pred"].values.astype(int)
    return true_labels, preds_per_seed, ref_ids


def aggregate_errors_across_seeds(true_labels, preds_per_seed):
    """Absolute error per sampel per seed, dirata-rata antar seed (bukan antar sampel)."""
    out = {}
    for name, seed_preds in preds_per_seed.items():
        errs = np.stack([np.abs(true_labels - p) for p in seed_preds.values()])
        out[name] = errs.mean(axis=0)
    return out


# ----------------------------------------------------------------------
# Wilcoxon + Holm-Bonferroni
# ----------------------------------------------------------------------
def run_significance_test(aggregated_errors, alpha=None):
    alpha = config.ALPHA if alpha is None else alpha
    raw_p, rows = [], []
    for hyp, a, b in config.HYPOTHESES:
        ea, eb = aggregated_errors[a], aggregated_errors[b]
        try:
            _, p = wilcoxon(ea, eb, zero_method="zsplit")
        except ValueError:
            p = 1.0  # seluruh selisih nol
        raw_p.append(p)
        rows.append({"hypothesis": hyp, "model_a": a, "model_b": b,
                     "mean_error_a": float(ea.mean()), "mean_error_b": float(eb.mean()),
                     "p_value_raw": float(p)})
    reject, corrected, _, _ = multipletests(raw_p, alpha=alpha, method="holm")
    for i, r in enumerate(rows):
        r["p_value_corrected"] = float(corrected[i])
        r["signifikan"] = "Ya" if reject[i] else "Tidak"
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Effect size MAE
# ----------------------------------------------------------------------
def bootstrap_effect_size(aggregated_errors, model_a, model_b, n_boot=None, seed=None):
    n_boot = n_boot or config.N_BOOT
    seed = config.BOOT_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    diffs = aggregated_errors[model_a] - aggregated_errors[model_b]
    n = len(diffs)
    boot = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"model_a": model_a, "model_b": model_b, "mean_diff": float(diffs.mean()),
            "ci_95_low": float(lo), "ci_95_high": float(hi)}


def run_all_effect_sizes(aggregated_errors):
    rows = []
    for hyp, a, b in config.HYPOTHESES:
        rows.append({"hypothesis": hyp, **bootstrap_effect_size(aggregated_errors, a, b)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# QWK ensemble dan per seed
# ----------------------------------------------------------------------
def ensemble_predictions(seed_preds):
    """Rata-rata prediksi antar seed, dibulatkan ke kelas terdekat, dibatasi 1 sampai 5."""
    stacked = np.stack(list(seed_preds.values()))
    return np.clip(np.round(stacked.mean(axis=0)), 1, 5).astype(int)


def bootstrap_qwk_ensemble(true_labels, preds_per_seed, model_a, model_b, n_boot=None, seed=None):
    n_boot = n_boot or config.N_BOOT
    seed = config.BOOT_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    n = len(true_labels)
    pa, pb = ensemble_predictions(preds_per_seed[model_a]), ensemble_predictions(preds_per_seed[model_b])
    qa, qb = qwk_fast(true_labels, pa), qwk_fast(true_labels, pb)
    idx_all = np.arange(n)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        boot[i] = qwk_fast(true_labels[idx], pa[idx]) - qwk_fast(true_labels[idx], pb[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"model_a": model_a, "model_b": model_b, "qwk_a": qa, "qwk_b": qb,
            "qwk_diff": qa - qb, "ci_95_low": float(lo), "ci_95_high": float(hi)}


def bootstrap_qwk_per_seed(true_labels, preds_per_seed, model_a, model_b, n_boot=None, seed=None):
    n_boot = n_boot or config.N_BOOT
    seed = config.BOOT_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    n = len(true_labels)
    seeds = list(preds_per_seed[model_a].keys())
    qa = np.array([qwk_fast(true_labels, preds_per_seed[model_a][s]) for s in seeds])
    qb = np.array([qwk_fast(true_labels, preds_per_seed[model_b][s]) for s in seeds])
    idx_all = np.arange(n)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        yt = true_labels[idx]
        boot[i] = np.mean([qwk_fast(yt, preds_per_seed[model_a][s][idx]) -
                           qwk_fast(yt, preds_per_seed[model_b][s][idx]) for s in seeds])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    row = {"model_a": model_a, "model_b": model_b, "n_seed": len(seeds),
           "qwk_a_rata_seed": float(qa.mean()), "qwk_b_rata_seed": float(qb.mean()),
           "qwk_diff": float((qa - qb).mean()), "ci_95_low": float(lo), "ci_95_high": float(hi)}
    for s, d in zip(seeds, qa - qb):
        row[f"qwk_diff_seed{s}"] = float(d)
    return row


def run_all_qwk(true_labels, preds_per_seed):
    ens = pd.DataFrame([{"hypothesis": h, **bootstrap_qwk_ensemble(true_labels, preds_per_seed, a, b)}
                        for h, a, b in config.HYPOTHESES])
    per = pd.DataFrame([{"hypothesis": h, **bootstrap_qwk_per_seed(true_labels, preds_per_seed, a, b)}
                        for h, a, b in config.HYPOTHESES])
    return ens, per


# ----------------------------------------------------------------------
# Tabel panjang untuk aturan konsistensi
# ----------------------------------------------------------------------
def build_long_table(split_seed, n_weight_seeds, sig_df, eff_df, qwk_ens_df, qwk_per_df):
    rows = []
    sig = sig_df.set_index("hypothesis")
    for _, r in eff_df.iterrows():
        s = sig.loc[r["hypothesis"]]
        rows.append({"split_seed": split_seed, "hypothesis": r["hypothesis"], "metric": "MAE",
                     "model_a": r["model_a"], "model_b": r["model_b"], "diff": r["mean_diff"],
                     "ci_low": r["ci_95_low"], "ci_high": r["ci_95_high"],
                     "n_weight_seeds": n_weight_seeds, "wilcoxon_p_raw": s["p_value_raw"],
                     "wilcoxon_p_holm": s["p_value_corrected"], "wilcoxon_signifikan": s["signifikan"]})
    for metric, df in (("QWK_ensemble", qwk_ens_df), ("QWK_per_seed", qwk_per_df)):
        for _, r in df.iterrows():
            rows.append({"split_seed": split_seed, "hypothesis": r["hypothesis"], "metric": metric,
                         "model_a": r["model_a"], "model_b": r["model_b"], "diff": r["qwk_diff"],
                         "ci_low": r["ci_95_low"], "ci_high": r["ci_95_high"],
                         "n_weight_seeds": n_weight_seeds, "wilcoxon_p_raw": np.nan,
                         "wilcoxon_p_holm": np.nan, "wilcoxon_signifikan": ""})
    return pd.DataFrame(rows)


def run_significance_suite(ctx, seeds):
    """Menjalankan seluruh analisis untuk satu split dan menulis tabel unik ke significance_dir."""
    true_labels, preds, _ = load_predictions(ctx, seeds)
    agg = aggregate_errors_across_seeds(true_labels, preds)

    final_table = build_final_results_table(ctx, seeds)
    sig = run_significance_test(agg)
    eff = run_all_effect_sizes(agg)
    qwk_ens, qwk_per = run_all_qwk(true_labels, preds)
    long = build_long_table(ctx.split_seed, len(seeds), sig, eff, qwk_ens, qwk_per)

    d = ctx.significance_dir
    drive_io.write_once_csv(final_table, os.path.join(d, "final_results_table.csv"))
    drive_io.write_once_csv(sig, os.path.join(d, "significance_test.csv"))
    drive_io.write_once_csv(eff, os.path.join(d, "effect_sizes_mae.csv"))
    drive_io.write_once_csv(qwk_ens, os.path.join(d, "qwk_ensemble_effect.csv"))
    drive_io.write_once_csv(qwk_per, os.path.join(d, "qwk_per_seed_effect.csv"))
    drive_io.write_once_csv(long, os.path.join(d, "tabel_panjang_efek.csv"))
    return {"final": final_table, "sig": sig, "eff": eff, "qwk_ens": qwk_ens,
            "qwk_per": qwk_per, "long": long}
