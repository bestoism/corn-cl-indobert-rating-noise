"""
diagnostics.py -- Diagnostik yang dijanjikan laporan:
  * §3.6 butir 2 : monotonisitas P(y>k) + Expected Calibration Error (ECE)
  * §3.6 butir 7 : kualitas proxy per kelas rating asli
  * §3.8         : distribusi jarak ordinal pada baris noise per kelas rating
  * §3.7         : ukuran sampel Cochran
Semua keluaran disimpan ke config.RESULTS_DIR (split-aware).
"""
import os
import math
import numpy as np
import pandas as pd

from src import config


def cochran_n(N, p=0.5, z=1.96, e=0.10):
    n0 = z ** 2 * p * (1 - p) / e ** 2
    return math.ceil(n0 / (1 + (n0 - 1) / N))


def cumulative_probs(pred_probs):
    """q[:, j] = P(y > j+1) untuk j=0..K-2, diturunkan dari probabilitas kelas."""
    return 1.0 - np.cumsum(np.asarray(pred_probs, dtype=float), axis=1)[:, :-1]


def _binary_ece(p, y, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def report_calibration(pred_probs, labels, n_bins=10, tol=1e-6, tag=""):
    """labels 0-indexed. ECE atas probabilitas kumulatif P(y>k) per ambang + ECE top-label."""
    pred_probs = np.asarray(pred_probs, dtype=float)
    labels = np.asarray(labels)
    q = cumulative_probs(pred_probs)

    rows = []
    for j in range(q.shape[1]):
        rows.append({"metric": f"ECE P(y>{j+1})",
                     "value": _binary_ece(q[:, j], (labels > j).astype(float), n_bins)})
    mean_ece = float(np.mean([r["value"] for r in rows]))
    rows.append({"metric": "ECE rata-rata P(y>k)", "value": mean_ece})

    conf = pred_probs.max(axis=1)
    correct = (pred_probs.argmax(axis=1) == labels).astype(float)
    rows.append({"metric": "ECE top-label", "value": _binary_ece(conf, correct, n_bins)})

    n_viol = int((np.diff(q, axis=1) > tol).sum())
    rows.append({"metric": f"pelanggaran monotonisitas P(y>k) (tol={tol})", "value": n_viol})
    rows.append({"metric": "n_baris", "value": len(labels)})

    out = pd.DataFrame(rows)
    path = os.path.join(config.RESULTS_DIR, f"calibration_report__{config.PROXY_NAME}{tag}.csv")
    out.to_csv(path, index=False)
    print(f"\n🌡️  Kalibrasi [{config.PROXY_NAME}{tag}]: ECE rata-rata P(y>k) = {mean_ece:.4f} | "
          f"ECE top-label = {rows[-4]['value']:.4f} | pelanggaran monotonisitas = {n_viol}")
    print(f"   -> {path}")
    print("   Catatan: probabilitas kumulatif diturunkan dari probabilitas kelas non-negatif, sehingga\n"
          "   monotonisitas terjamin secara konstruksi; pemeriksaan ini berfungsi sebagai guard numerik.")
    return {"mean_ece": mean_ece, "n_monotonic_violations": n_viol}


def per_class_proxy_quality(labels, preds, tag=""):
    """labels/preds 0-indexed. Recall, precision, MAE per kelas rating ASLI."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    rows = []
    for c in range(config.NUM_CLASSES):
        m_true = labels == c
        m_pred = preds == c
        rows.append({
            "rating_asli": c + 1,
            "n": int(m_true.sum()),
            "recall": float((preds[m_true] == c).mean()) if m_true.any() else np.nan,
            "precision": float((labels[m_pred] == c).mean()) if m_pred.any() else np.nan,
            "mae": float(np.abs(preds[m_true] - labels[m_true]).mean()) if m_true.any() else np.nan,
        })
    out = pd.DataFrame(rows)
    path = os.path.join(config.RESULTS_DIR, f"proxy_per_class__{config.PROXY_NAME}{tag}.csv")
    out.to_csv(path, index=False)
    print(f"\n📐 Kualitas proxy per kelas rating asli [{config.PROXY_NAME}]:")
    print(out.round(4).to_string(index=False))
    print(f"   -> {path}")
    return out


def noise_distribution_by_rating(df, tag=""):
    """df harus punya kolom rating, is_noise, rating_diff (hasil clean.py, sebelum resolusi)."""
    rows = []
    for r in sorted(df["rating"].unique()):
        sub = df[df["rating"] == r]
        flagged = sub[sub["is_noise"]]
        row = {"rating_asli": int(r), "n": len(sub), "n_flagged": len(flagged),
               "pct_flagged": round(len(flagged) / len(sub) * 100, 2)}
        for d in [1, 2, 3, 4]:
            row[f"diff_{d}"] = int((flagged["rating_diff"].clip(upper=4) == d).sum())
        row["pct_severe(diff>=thr)"] = (
            round((flagged["rating_diff"] >= config.SEVERITY_THRESHOLD).mean() * 100, 2)
            if len(flagged) else np.nan
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    path = os.path.join(config.RESULTS_DIR, f"noise_distribution_by_rating__{config.PROXY_NAME}{tag}.csv")
    out.to_csv(path, index=False)
    print(f"\n📊 Distribusi jarak ordinal pada baris noise per kelas rating asli (§3.8):")
    print(out.to_string(index=False))
    print(f"   -> {path}")
    return out