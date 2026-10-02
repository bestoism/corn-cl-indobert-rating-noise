"""
k_sensitivity.py -- Sensitivitas jumlah fold K (3, 5, 10) pada proxy P4 (§3.5).
Konfirmatif terhadap keputusan apriori K=5, bukan pencarian optimum.
K=5 memakai cache P4 yang sudah ada (TIDAK dilatih ulang).
Hanya split utama. Estimasi waktu: K=3 dan K=10 ~130 menit total (GPU T4).
"""
import numpy as np
import pandas as pd

from src import config, diagnostics as diag
from src.metrics import compute_metrics
from src.clean import compute_filter_results, _select_filter_method
from src.proxy import _finetune_kfold_oof


def run_k_sensitivity(k_values=None):
    config.require_primary_split("Sensitivitas K")
    config.set_proxy(3)
    k_values = k_values or config.K_SENSITIVITY_VALUES

    df_train = pd.read_csv(config.TRAIN_RAW_FILE)
    labels = df_train["rating"].values - 1
    texts = df_train["cleaned_text"].tolist()

    rows, flagged = [], {}
    for k in k_values:
        tag = "" if k == config.PROXY_CV_FOLDS else f"K{k}"   # K=5 -> cache utama
        print(f"\n{'='*60}\n SENSITIVITAS K = {k} (cache_tag='{tag}')\n{'='*60}")
        probs = _finetune_kfold_oof(texts, labels, "corn", n_folds=k, cache_tag=tag)
        preds = probs.argmax(axis=1)
        m = compute_metrics(labels, preds)
        cal = diag.report_calibration(probs, labels, tag=f"__K{k}")
        results = compute_filter_results(labels, probs)
        method, info = _select_filter_method(labels, probs, results, verbose=False)
        flagged[k] = results[method]
        rows.append({
            "K": k, "qwk": m["qwk"], "mae": m["mae"], "accuracy": m["accuracy"], "off_by_one": m["off_by_one"],
            "mean_ece_cumulative": cal["mean_ece"], "selected_method": method,
            "n_flagged_confident_learning": info["confident_learning"],
            "n_flagged_prune_by_noise_rate": info["prune_by_noise_rate"],
            "n_flagged_selected": int(results[method].sum()),
        })

    base = flagged.get(config.PROXY_CV_FOLDS)
    for r in rows:
        if base is not None:
            a, b = flagged[r["K"]], base
            r["jaccard_flagged_vs_K5"] = float((a & b).sum() / max((a | b).sum(), 1))

    out = pd.DataFrame(rows)
    out.to_csv(config.K_SENSITIVITY_FILE, index=False)
    print("\n" + out.round(4).to_string(index=False))
    print(f"\n💾 -> {config.K_SENSITIVITY_FILE}")
    print("   Tafsir (§3.5): selisih QWK kecil dan tidak monoton => mendukung K=5 tanpa klaim optimalitas;\n"
          "   bila K lain unggul konsisten, nyatakan penyimpangan dari rencana apriori secara eksplisit.")
    return out