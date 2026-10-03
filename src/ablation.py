"""
ablation.py: evaluasi satu proxy (P1 sampai P4) pada satu nilai K, dan tabel ablasi (Subbab 3.5, 3.6).

Satu pemanggilan run_proxy_evaluation menulis berkas unik berawalan '{P}__{nama}__K{K}' pada
out_dir: hasil (json), kualitas per kelas (csv), dan temperature per fold (csv, bila ada).
"""

import os

import numpy as np
import pandas as pd

from src import calibration, clean, config, diagnostics, drive_io, proxy


def run_proxy_evaluation(ctx, df_train, proxy_id, K, out_dir):
    info = config.PROXY_REGISTRY[proxy_id]
    prefix = f"{info['label']}__{info['name']}__K{K}"
    result_path = os.path.join(out_dir, f"{prefix}__hasil.json")

    print(f"\n=== {info['label']} {info['name']} (K={K}) ===")
    oof = proxy.get_oof(ctx, df_train, proxy_id, K)
    det = clean.detect_noise(df_train, oof["class_probs"])

    labels0, preds0 = det["labels0"], det["preds0"]
    m = det["proxy_metrics"]
    print(f"Kualitas proxy: acc={m['accuracy']:.4f} MAE={m['mae']:.4f} off-by-1={m['off_by_one']:.4f} "
          f"QWK={m['qwk']:.4f}")

    cal = calibration.calibration_report(oof["class_probs"], labels0, oof.get("q_chain"),
                                         n_bins=config.ECE_N_BINS)
    cal_flat = calibration.flatten_report(cal)
    per_class = calibration.per_class_quality(labels0, preds0)

    temps = oof.get("temperatures")
    result = {
        "proxy_id": proxy_id, "label": info["label"], "name": info["name"], "desc": info["desc"],
        "K": K, "n_train": len(df_train),
        **{k: v for k, v in m.items()},
        "pct_flagged_noise": 100.0 * int(det["df"]["is_noise"].sum()) / len(df_train),
        "selected_filter_method": det["selected_method"],
        "n_issues_confident_learning": det["method_info"]["confident_learning"],
        "n_issues_prune_by_noise_rate": det["method_info"]["prune_by_noise_rate"],
        "n_issues_estimated_theoretical": det["method_info"]["estimated_theoretical"],
        "temperature_rata": float(np.mean(temps)) if temps is not None else None,
        **cal_flat,
    }

    if not drive_io.is_verified(result_path):
        drive_io.write_once_json(result, result_path)
        drive_io.write_once_csv(per_class, os.path.join(out_dir, f"{prefix}__kualitas_per_kelas.csv"))
        if temps is not None:
            drive_io.write_once_csv(
                pd.DataFrame({"fold": np.arange(1, len(temps) + 1), "temperature": temps}),
                os.path.join(out_dir, f"{prefix}__temperature_per_fold.csv"))
    return {"result": result, "detection": det, "oof": oof, "per_class": per_class,
            "labels0": labels0, "preds0": preds0}


def run_noise_distance_diagnostic(ctx, evaluation, out_dir):
    """Subbab 3.8: distribusi jarak ordinal pada baris noise per kelas rating asli (P4, K=5)."""
    tab = diagnostics.noise_distance_by_class(evaluation["detection"]["df"], evaluation["per_class"])
    drive_io.write_once_csv(tab, os.path.join(out_dir, "P4__jarak_ordinal_noise_per_kelas.csv"))
    return tab


ABLATION_COLUMNS = [
    "proxy_id", "label", "name", "desc", "accuracy", "mae", "rmse", "off_by_one", "qwk",
    "pct_flagged_noise", "selected_filter_method", "n_issues_confident_learning",
    "n_issues_prune_by_noise_rate", "n_issues_estimated_theoretical", "temperature_rata",
    "ece_top_label", "ece_kumulatif_turunan_rata", "ece_kumulatif_rantai_rata",
]


def build_ablation_table(results):
    """results: list dict hasil (run_proxy_evaluation()['result']) untuk P1 sampai P4 pada K = 5."""
    df = pd.DataFrame(results)
    cols = [c for c in ABLATION_COLUMNS if c in df.columns]
    return df[cols].sort_values("proxy_id").reset_index(drop=True)


def load_result_json(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
