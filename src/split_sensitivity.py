"""
split_sensitivity.py -- Analisis sensitivitas partisi data (3 split acak: 42 utama, 123, 2024).
Semua dilaporkan; tidak ada split yang 'dipilih'. Validasi manusia & gold test hanya split utama.
"""
import os
import json
import numpy as np
import pandas as pd

from src import config
from src.train import run_experiment
from src.data_split import create_splits
from src.clean import run_confident_learning
from src.significance import (
    collect_all_predictions, aggregate_errors_across_seeds, run_significance_test,
    run_all_effect_sizes, run_all_qwk_effect_sizes, run_all_qwk_per_seed,
)


def build_scenarios():
    """Panggil SETELAH config.set_split() -- path bergantung split aktif."""
    return [
        {"name": "M1_Baseline_CE",        "train_path": config.TRAIN_RAW_RESOLVED_FILE,   "loss": "ce"},
        {"name": "M2_CleanedHard_CE",     "train_path": config.TRAIN_CLEANED_HARD_FILE,   "loss": "ce"},
        {"name": "M3_CleanedSevere_CE",   "train_path": config.TRAIN_CLEANED_SEVERE_FILE, "loss": "ce"},
        {"name": "M4_Baseline_CORN",      "train_path": config.TRAIN_RAW_RESOLVED_FILE,   "loss": "corn"},
        {"name": "M5_CleanedHard_CORN",   "train_path": config.TRAIN_CLEANED_HARD_FILE,   "loss": "corn"},
        {"name": "M6_CleanedSevere_CORN", "train_path": config.TRAIN_CLEANED_SEVERE_FILE, "loss": "corn"},
    ]


def train_all_scenarios(scenarios):
    """Auto-resume memakai config.PROGRESS_FILE (format sama dengan Cell 20 lama)."""
    progress = {}
    if os.path.exists(config.PROGRESS_FILE):
        with open(config.PROGRESS_FILE) as f:
            progress = json.load(f)
    for s in scenarios:
        progress.setdefault(s["name"], {})
        for seed in config.SEED_LIST:
            if str(seed) in progress[s["name"]]:
                print(f"⏩ {s['name']} | seed {seed} (sudah selesai)")
                continue
            progress[s["name"]][str(seed)] = run_experiment(s["name"], s["train_path"], s["loss"], seed)
            with open(config.PROGRESS_FILE, "w") as f:
                json.dump(progress, f, indent=2)
    rows = [{"scenario": n, "seed": int(sd), **m} for n, d in progress.items() for sd, m in d.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.RESULTS_DIR, "per_seed_metrics.csv"), index=False)
    return df


def make_final_table():
    """Tabel mean ± std (format sama dengan Cell 20 lama), dari progress file."""
    with open(config.PROGRESS_FILE) as f:
        prog = json.load(f)
    labels = {"mae": "MAE (↓)", "rmse": "RMSE (↓)", "accuracy": "Acc (↑)",
              "off_by_one": "Off-by-1 (↑)", "qwk": "QWK (↑)"}
    rows = []
    for name, seeds in prog.items():
        ms = [m for sd, m in seeds.items() if int(sd) in config.SEED_LIST]
        if not ms:
            continue
        row = {"Model": name}
        for k, lab in labels.items():
            v = [m[k] for m in ms]
            row[lab] = f"{np.mean(v):.4f} ± {np.std(v):.4f}"
        row["_mae"] = np.mean([m["mae"] for m in ms])
        rows.append(row)
    df = pd.DataFrame(sorted(rows, key=lambda r: r["_mae"])).drop(columns=["_mae"])
    df.to_csv(config.FINAL_RESULTS_TABLE_FILE, index=False)
    return df


def run_split_pipeline(split_seed, seed_list=None):
    """
    split baru -> OOF P4 -> CL -> resolusi -> M1-M6 x seed -> Wilcoxon+Holm, effect size MAE,
    QWK (ensemble) dan QWK (per-seed). seed_list=[42] = opsi hemat (1 seed bobot).
    Split 42: idempotent -- tidak melatih ulang bila checkpoint/progress sudah ada.
    """
    old_seeds = list(config.SEED_LIST)
    if seed_list:
        config.SEED_LIST = list(seed_list)
    try:
        config.set_split(split_seed)
        config.set_proxy(3)
        if split_seed != config.PRIMARY_SPLIT_SEED and not os.path.exists(config.TRAIN_RAW_FILE):
            create_splits(config.CLEAN_TEXT_FILE, seed=split_seed)
        if not os.path.exists(config.TRAIN_CLEANED_SEVERE_FILE):
            run_confident_learning()

        scenarios = build_scenarios()
        train_all_scenarios(scenarios)

        true_labels, preds = collect_all_predictions(scenarios)
        errs = aggregate_errors_across_seeds(true_labels, preds)
        run_significance_test(errs)
        run_all_effect_sizes(errs).to_csv(os.path.join(config.RESULTS_DIR, "effect_sizes.csv"), index=False)
        run_all_qwk_effect_sizes(true_labels, preds).to_csv(
            os.path.join(config.RESULTS_DIR, "qwk_effect_sizes.csv"), index=False)
        run_all_qwk_per_seed(true_labels, preds).to_csv(
            os.path.join(config.RESULTS_DIR, "qwk_per_seed_effect_sizes.csv"), index=False)
        make_final_table()
        print(f"\n✅ Pipeline split {split_seed} selesai -> {config.RESULTS_DIR}")
    finally:
        config.SEED_LIST = old_seeds


def _count(df, lo, hi, lower_is_better):
    a = (df[hi] < 0) if lower_is_better else (df[lo] > 0)   # A lebih baik, CI tak melewati 0
    b = (df[lo] > 0) if lower_is_better else (df[hi] < 0)   # B lebih baik, CI tak melewati 0
    return pd.Series({"A_lebih_baik": int(a.sum()), "B_lebih_baik": int(b.sum()),
                      "tak_konklusif": int((~a & ~b).sum())})


def summarize_across_splits(split_seeds=None):
    """Gabungkan hasil per split. Tidak ada p-value gabungan (test set tumpang tindih, bukan replikasi independen)."""
    frames = []
    for s in (split_seeds or config.SPLIT_SEEDS):
        config.set_split(s, verbose=False)
        sig = pd.read_csv(config.SIGNIFICANCE_TEST_FILE)
        mae = pd.read_csv(os.path.join(config.RESULTS_DIR, "effect_sizes.csv"))
        qwk = pd.read_csv(os.path.join(config.RESULTS_DIR, "qwk_effect_sizes.csv"))
        qps = pd.read_csv(os.path.join(config.RESULTS_DIR, "qwk_per_seed_effect_sizes.csv"))
        key = ["model_a", "model_b"]
        df = (sig.merge(mae, on=key)
                 .merge(qwk[key + ["qwk_diff", "ci_95_low", "ci_95_high"]].rename(columns={
                     "qwk_diff": "qwk_diff_ens", "ci_95_low": "qwk_ens_lo", "ci_95_high": "qwk_ens_hi"}), on=key)
                 .merge(qps[key + ["qwk_diff", "ci_95_low", "ci_95_high"]].rename(columns={
                     "qwk_diff": "qwk_diff_seed", "ci_95_low": "qwk_seed_lo", "ci_95_high": "qwk_seed_hi"}), on=key))
        df.insert(0, "split_seed", s)
        frames.append(df)
    config.set_split(config.PRIMARY_SPLIT_SEED, verbose=False)

    long = pd.concat(frames, ignore_index=True).rename(
        columns={"ci_95_low": "mae_lo", "ci_95_high": "mae_hi", "mean_diff": "mae_diff"})
    long.to_csv(os.path.join(config.PRIMARY_RESULTS_DIR, "split_sensitivity_long.csv"), index=False)

    summ = []
    for hyp, g in long.groupby("hypothesis", sort=False):
        wil_a = int(((g["signifikan"] == "Ya") & (g["mean_error_a"] < g["mean_error_b"])).sum())
        wil_b = int(((g["signifikan"] == "Ya") & (g["mean_error_a"] > g["mean_error_b"])).sum())
        summ.append(pd.concat([
            pd.Series({"hypothesis": hyp, "n_split": len(g),
                       "Wilcoxon_signif_A_lebih_baik": wil_a, "Wilcoxon_signif_B_lebih_baik": wil_b}),
            _count(g, "mae_lo", "mae_hi", True).add_prefix("MAE_"),
            _count(g, "qwk_ens_lo", "qwk_ens_hi", False).add_prefix("QWKens_"),
            _count(g, "qwk_seed_lo", "qwk_seed_hi", False).add_prefix("QWKseed_"),
        ]))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(config.PRIMARY_RESULTS_DIR, "split_sensitivity_summary.csv"), index=False)
    return long, summ