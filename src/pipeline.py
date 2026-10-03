"""pipeline.py: orkestrasi langkah per split agar sel notebook tetap kecil dan satu tujuan."""

import json
import os
from datetime import datetime

import pandas as pd

from src import ablation, clean, config, consistency, data_split, drive_io, significance, train


def new_run_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def weight_seeds_for(split_seed, hemat=False):
    """Split utama selalu memakai tiga seed bobot. Opsi hemat hanya untuk split sensitivitas."""
    if split_seed == config.MAIN_SPLIT_SEED:
        return list(config.WEIGHT_SEEDS)
    return list(config.WEIGHT_SEEDS_LIGHT if hemat else config.WEIGHT_SEEDS)


def register_run_mode(ctx, seeds, hemat):
    """Mencatat seed bobot yang dipakai untuk split ini (sekali). Mengubah mode di tengah jalan ditolak."""
    path = os.path.join(ctx.training_dir, "run_mode.json")
    rec = {"split_seed": ctx.split_seed, "weight_seeds": list(seeds), "hemat": bool(hemat),
           "keterangan": ("OPSI HEMAT: satu seed bobot, wajib dinyatakan sebagai keterbatasan"
                          if hemat else "konfigurasi penuh: tiga seed bobot")}
    if drive_io.is_verified(path):
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
        if old["weight_seeds"] != rec["weight_seeds"]:
            raise RuntimeError(f"[BERHENTI] Mode seed untuk split {ctx.split_seed} sudah tercatat "
                               f"{old['weight_seeds']} dan tidak boleh diubah menjadi {rec['weight_seeds']}.")
        return old
    drive_io.write_once_json(rec, path)
    return rec


def read_run_mode(ctx):
    path = os.path.join(ctx.training_dir, "run_mode.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def step_p4(ctx, df_train):
    """P4 K=5 untuk satu split: evaluasi, diagnostik, dan tiga varian data latih."""
    ev = ablation.run_proxy_evaluation(ctx, df_train, config.FINAL_PROXY_ID, config.PROXY_CV_FOLDS,
                                       ctx.diagnostics_dir)
    noise_tab = ablation.run_noise_distance_diagnostic(ctx, ev, ctx.diagnostics_dir)
    variants_done = all(drive_io.is_verified(ctx.variant_file(v)) for v in ("raw", "hard", "severe"))
    if variants_done:
        print("[OK] Tiga varian data latih sudah ada.")
        df_noise = pd.read_csv(os.path.join(ctx.noise_dir, "flagged_rows_P4_K5.csv"))
        stats = pd.read_csv(os.path.join(ctx.noise_dir, "variant_summary.csv")).iloc[0].to_dict()
    else:
        df_noise, stats = clean.export_p4_outputs(ctx, ev["detection"], config.FINAL_PROXY_ID,
                                                  config.PROXY_CV_FOLDS)
    return {"eval": ev, "noise_by_class": noise_tab, "df_noise": df_noise, "variant_stats": stats}


def step_ablation_p1_p3(ctx, df_train, p4_eval):
    """Ablasi P1 sampai P3 (hanya metrik, diagnostik, dan jumlah flag) lalu tabel ablasi P1 sampai P4."""
    ctx.require_main("ablasi P1 sampai P3")
    results = []
    for pid in (0, 1, 2):
        ev = ablation.run_proxy_evaluation(ctx, df_train, pid, config.PROXY_CV_FOLDS, ctx.ablation_dir)
        results.append(ev["result"])
    results.append(p4_eval["result"])
    table = ablation.build_ablation_table(results)
    path = os.path.join(ctx.ablation_dir, "proxy_ablation_table.csv")
    drive_io.write_once_csv(table, path)
    return table


def step_train(ctx, seeds, copy_ckpt_to_drive=False):
    return train.run_all(ctx, seeds, copy_ckpt_to_drive=copy_ckpt_to_drive)


def step_analysis(ctx, seeds):
    return significance.run_significance_suite(ctx, seeds)


def consistency_summary(split_seeds=None):
    """Menggabungkan tabel panjang ketiga split (tanpa menggabungkan p-value) dan menilai konsistensi."""
    split_seeds = split_seeds or config.SPLIT_SEEDS
    frames = []
    for s in split_seeds:
        p = os.path.join(config.RunPaths(s).significance_dir, "tabel_panjang_efek.csv")
        frames.append(drive_io.read_csv_verified(p))
    long = pd.concat(frames, ignore_index=True)
    summary = consistency.evaluate_consistency(long, expected_splits=len(config.SPLIT_SEEDS))

    wil = long[long["metric"] == "MAE"][["hypothesis", "split_seed", "wilcoxon_p_raw", "wilcoxon_p_holm",
                                          "wilcoxon_signifikan"]]
    wide = wil.pivot(index="hypothesis", columns="split_seed",
                     values=["wilcoxon_p_holm", "wilcoxon_signifikan"])
    wide.columns = [f"{a}_split{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    d = config.SENSITIVITY_SUMMARY_DIR
    drive_io.write_once_csv(long, os.path.join(d, "tabel_panjang_semua_split.csv"))
    drive_io.write_once_csv(summary, os.path.join(d, "ringkasan_konsistensi.csv"))
    drive_io.write_once_csv(wide, os.path.join(d, "wilcoxon_berdampingan_per_split.csv"))
    return long, summary, wide
