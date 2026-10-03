"""
calibration.py: diagnostik kalibrasi dan konversi rantai CORN (numpy murni).

Definisi yang dipakai (Subbab 2.1.8 dan 3.6):
  - Probabilitas kumulatif langsung dari rantai CORN (setelah temperature scaling,
    sebelum konversi ke probabilitas kelas): q_chain[k] = prod_{j<=k} sigmoid(z_j / T),
    untuk k = 1..K-1, yaitu P(y > k).
  - Probabilitas kumulatif turunan: q_derived[k] = sum_{c>k} P(y = c), dihitung dari
    probabilitas kelas hasil pembatasan minimal 1e-8 dan normalisasi ulang.
  - ECE kumulatif: ECE biner per ambang k pada kejadian {y > k} dengan probabilitas
    q[k], 10 bin lebar sama pada [0, 1], dirata-rata atas K-1 ambang. Dihitung
    terhadap label teramati (yang mungkin noisy), bukan label bersih.
  - ECE top-label (TAMBAHAN, bukan bagian proposal): ECE standar atas keyakinan
    maksimum dan ketepatan prediksi argmax, 10 bin lebar sama.

Catatan penting: kumulatif dari rantai adalah hasil kali bilangan pada [0, 1], dan
kumulatif turunan adalah jumlah ekor probabilitas non-negatif. Keduanya tidak naik
secara konstruksi, sehingga pemeriksaan monotonisitas bersifat uji kewarasan numerik
(pembulatan float), bukan bukti empiris tentang kualitas model. Bagian yang
informatif adalah selisih antara kedua kumulatif dan nilai ECE.
"""

import numpy as np


def _bin_index(prob, n_bins):
    idx = np.floor(np.asarray(prob, dtype=np.float64) * n_bins).astype(np.int64)
    return np.clip(idx, 0, n_bins - 1)


def ece_binary(prob, event, n_bins=10):
    prob = np.asarray(prob, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    n = len(prob)
    idx = _bin_index(prob, n_bins)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.sum() / n * abs(event[m].mean() - prob[m].mean())
    return float(ece)


def derived_cumulative(class_probs):
    """q_derived[:, k-1] = P(y > k) = jumlah probabilitas kelas k+1 sampai K."""
    cp = np.asarray(class_probs, dtype=np.float64)
    tail = np.cumsum(cp[:, ::-1], axis=1)[:, ::-1]  # tail[:, c] = sum_{j>=c} p_j
    return tail[:, 1:]


def ece_cumulative(q, labels0, n_bins=10):
    """q berbentuk (n, K-1) dengan kolom k-1 = P(y > k). labels0 pada skala 0 sampai K-1."""
    q = np.asarray(q, dtype=np.float64)
    labels0 = np.asarray(labels0)
    eces = []
    for k in range(q.shape[1]):
        eces.append(ece_binary(q[:, k], labels0 > k, n_bins))
    return float(np.mean(eces)), [float(e) for e in eces]


def ece_top_label(class_probs, labels0, n_bins=10):
    cp = np.asarray(class_probs, dtype=np.float64)
    conf = cp.max(axis=1)
    correct = (cp.argmax(axis=1) == np.asarray(labels0)).astype(np.float64)
    return ece_binary(conf, correct, n_bins)


def monotonicity_report(q, atol=0.0):
    """Proporsi baris dengan q[k+1] <= q[k] + atol untuk semua k."""
    q = np.asarray(q, dtype=np.float64)
    diffs = np.diff(q, axis=1)
    viol = diffs > atol
    n_rows_viol = int(viol.any(axis=1).sum())
    return {
        "n_baris": int(q.shape[0]),
        "n_baris_melanggar": n_rows_viol,
        "proporsi_monoton": float(1.0 - n_rows_viol / max(q.shape[0], 1)),
        "pelanggaran_maks": float(diffs.max()) if diffs.size else 0.0,
    }


def chain_vs_derived_gap(q_chain, class_probs):
    qd = derived_cumulative(class_probs)
    gap = np.abs(np.asarray(q_chain, dtype=np.float64) - qd)
    return {"selisih_maks": float(gap.max()), "selisih_rata": float(gap.mean())}


def calibration_report(class_probs, labels0, q_chain=None, n_bins=10):
    """Mengembalikan dict bernama. Nilai diambil berdasarkan nama metrik, bukan posisi."""
    cp = np.asarray(class_probs, dtype=np.float64)
    labels0 = np.asarray(labels0)
    q_der = derived_cumulative(cp)
    ece_der_mean, ece_der_each = ece_cumulative(q_der, labels0, n_bins)
    rep = {
        "ece_top_label": ece_top_label(cp, labels0, n_bins),
        "ece_kumulatif_turunan_rata": ece_der_mean,
        "monotonisitas_turunan": monotonicity_report(q_der),
    }
    for k, e in enumerate(ece_der_each, start=1):
        rep[f"ece_kumulatif_turunan_ambang{k}"] = e
    if q_chain is not None:
        q_chain = np.asarray(q_chain, dtype=np.float64)
        ece_ch_mean, ece_ch_each = ece_cumulative(q_chain, labels0, n_bins)
        rep["ece_kumulatif_rantai_rata"] = ece_ch_mean
        for k, e in enumerate(ece_ch_each, start=1):
            rep[f"ece_kumulatif_rantai_ambang{k}"] = e
        rep["monotonisitas_rantai"] = monotonicity_report(q_chain)
        rep["selisih_rantai_vs_turunan"] = chain_vs_derived_gap(q_chain, cp)
    return rep


def flatten_report(rep):
    """Meratakan dict bersarang menjadi satu baris datar untuk CSV."""
    flat = {}
    for k, v in rep.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}__{kk}"] = vv
        else:
            flat[k] = v
    return flat


def per_class_quality(labels0, preds0, n_classes=5):
    """Recall, precision, dan MAE per kelas rating asli (label teramati)."""
    import pandas as pd
    labels0 = np.asarray(labels0)
    preds0 = np.asarray(preds0)
    rows = []
    for c in range(n_classes):
        true_mask = labels0 == c
        pred_mask = preds0 == c
        n_true = int(true_mask.sum())
        n_pred = int(pred_mask.sum())
        tp = int((true_mask & pred_mask).sum())
        rows.append({
            "rating": c + 1,
            "n_label_asli": n_true,
            "n_prediksi": n_pred,
            "recall": tp / n_true if n_true else float("nan"),
            "precision": tp / n_pred if n_pred else float("nan"),
            "mae": float(np.abs(preds0[true_mask] - labels0[true_mask]).mean()) if n_true else float("nan"),
        })
    return pd.DataFrame(rows)
