"""metrics.py: lima metrik evaluasi ordinal (Subbab 2.1.7) dan QWK cepat untuk bootstrap."""

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             mean_absolute_error, mean_squared_error)


def compute_metrics(true_labels, predictions):
    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)
    if true_labels.shape != predictions.shape:
        raise ValueError(
            f"Ukuran true_labels {true_labels.shape} dan predictions {predictions.shape} tidak sama."
        )
    mae = mean_absolute_error(true_labels, predictions)
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    acc = accuracy_score(true_labels, predictions)
    off_by_one = float(np.mean(np.abs(true_labels - predictions) <= 1))
    qwk = cohen_kappa_score(true_labels, predictions, weights="quadratic")
    return {"mae": float(mae), "rmse": float(rmse), "accuracy": float(acc),
            "off_by_one": off_by_one, "qwk": float(qwk)}


def qwk_fast(y_true, y_pred, n_classes=5, offset=1):
    """
    QWK dari matriks konfusi (numpy murni) untuk resampling bootstrap yang cepat.
    Setara dengan sklearn.cohen_kappa_score(weights='quadratic') selama kelima kelas
    muncul pada gabungan label dan prediksi (terpenuhi pada data uji penelitian ini).
    Label dan prediksi pada skala 1 sampai 5 (offset=1) atau 0 sampai 4 (offset=0).
    """
    yt = np.asarray(y_true, dtype=np.int64) - offset
    yp = np.asarray(y_pred, dtype=np.int64) - offset
    O = np.bincount(yt * n_classes + yp, minlength=n_classes * n_classes)
    O = O.reshape(n_classes, n_classes).astype(np.float64)
    n = O.sum()
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / n
    idx = np.arange(n_classes)
    W = (idx[:, None] - idx[None, :]) ** 2 / (n_classes - 1) ** 2
    den = (W * E).sum()
    if den == 0:
        return float("nan")
    return float(1.0 - (W * O).sum() / den)
