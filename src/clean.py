"""
clean.py -- Pipeline Confident Learning untuk deteksi label noise.
Satu skema data (tanpa versioning v1/v2) -- all_reviews_master.csv adalah
sumber kebenaran tunggal, dikunci sejak awal.
"""

import os
import numpy as np
import pandas as pd
import cleanlab
from cleanlab.count import num_label_issues
from sklearn.model_selection import train_test_split

from src import config
from src.metrics import compute_metrics
from src.proxy import get_proxy_pred_probs   # <-- SATU-SATUNYA titik integrasi dgn proxy.py
from src.preprocess import drop_text_rating_conflicts   # <-- tambahkan import ini


# ==========================================================
# 1. SAMPLING VALIDASI MANUSIA — STRATIFIED BY rating_diff
# ==========================================================
def _stratified_human_sample(df_noise, n=None, seed=42, min_per_severe_bin=15):
    """
    Alokasi TIDAK PROPORSIONAL (Subbab III.I.3): minimal `min_per_severe_bin`
    (15) baris untuk TIAP bin jarak ordinal >=2 (diff=2, diff=3, diff=4 --
    titik keputusan ambang severity-aware pruning), sisa n dialokasikan ke
    bin diff=1. Mengorbankan presisi agregat demi presisi di titik keputusan
    yang relevan bagi klaim utama penelitian.
    """
    n = n or config.HUMAN_VALIDATION_N
    df = df_noise.copy()
    df["_diff_bin"] = df["rating_diff"].clip(upper=4)

    sampled_parts = []
    n_allocated_severe = 0
    shortfall = {}
    rng_seed = seed

    for b in [2, 3, 4]:
        pool = df[df["_diff_bin"] == b]
        take = min(min_per_severe_bin, len(pool))
        if take < min_per_severe_bin:
            shortfall[b] = {"target": min_per_severe_bin, "available": len(pool)}
        if take > 0:
            sampled_parts.append(pool.sample(n=take, random_state=rng_seed))
        n_allocated_severe += take
        rng_seed += 1

    n_remaining = max(n - n_allocated_severe, 0)
    pool1 = df[df["_diff_bin"] == 1]
    take1 = min(n_remaining, len(pool1))
    if take1 < n_remaining:
        shortfall[1] = {"target": n_remaining, "available": len(pool1)}
    if take1 > 0:
        sampled_parts.append(pool1.sample(n=take1, random_state=rng_seed))

    sample = pd.concat(sampled_parts) if sampled_parts else df.iloc[0:0]
    sample = sample.drop(columns=["_diff_bin"]).sample(frac=1, random_state=seed)

    if len(sample) < n:
        print(f"⚠️ Ukuran sampel aktual ({len(sample)}) < target ({n}). Detail: {shortfall}")
        print("   Dilaporkan apa adanya (Batasan Masalah butir 5).")

    return sample


def export_human_validation_sample(df_noise, n=None):
    """
    PERBAIKAN: versi sebelumnya menyertakan predicted_rating dan rating_diff
    di file yang sama diberikan ke anotator -- itu MEMBOCORKAN informasi hasil
    flag Confident Learning secara implisit (anotator bisa menebak "ini pasti
    diflag" dari rating_diff besar), padahal Subbab 3.9.3 proposal menjanjikan
    penilaian buta penuh. Sekarang dipecah jadi dua file: file internal (untuk
    breakdown per-bin rating_diff nanti) dan file anotator yang benar-benar
    buta, disatukan kembali lewat review_id setelah verdict masuk.
    """
    n = n or config.HUMAN_VALIDATION_N
    sample = _stratified_human_sample(df_noise, n)

    internal_cols = ["review_id", "source_app", "review_text", "cleaned_text",
                      "rating", "predicted_rating", "rating_diff"]
    sample[internal_cols].to_csv(config.HUMAN_VALIDATION_INTERNAL_FILE, index=False)

    annotator_cols = ["review_id", "source_app", "review_text", "cleaned_text", "rating"]
    blind_sample = sample[annotator_cols].copy()
    blind_sample["human_verdict"] = ""
    blind_sample["human_note"] = ""
    blind_sample.to_csv(config.HUMAN_VALIDATION_FILE, index=False)

    print(f"\n📝 Sample validasi manusia ({len(blind_sample)} baris, BUTA -- "
          f"predicted_rating/rating_diff TIDAK disertakan):")
    print(f"   {config.HUMAN_VALIDATION_FILE}")
    print(f"   (referensi internal untuk breakdown per-bin -> {config.HUMAN_VALIDATION_INTERNAL_FILE})")
    print(f"   Distribusi rating_diff (internal): {sample['rating_diff'].value_counts().sort_index().to_dict()}")
    print("   -> Isi kolom 'human_verdict' manual, lalu jalankan src.human_validation.compute_agreement()")


# ==========================================================
# 2. LOG KUALITAS PROXY — akumulatif, idempotent per proxy_id
# ==========================================================
def _log_proxy_quality(proxy_metrics, pct_flagged, selected_method, method_selection_info):
    """
    Satu baris per proxy_id di results/proxy_ablation_table.csv (dipakai
    untuk pilot study Bab 1). Idempotent: kalau clean.py dijalankan ulang
    dengan PROXY_ID yang sama, baris lama diganti -- bukan duplikat.
    """
    row = {
        "proxy_id": config.PROXY_ID,
        "proxy_name": config.PROXY_NAME,
        "proxy_desc": config.PROXY_DESC,
        "accuracy": proxy_metrics["accuracy"],
        "mae": proxy_metrics["mae"],
        "off_by_one": proxy_metrics["off_by_one"],
        "qwk": proxy_metrics["qwk"],
        "pct_flagged_noise": pct_flagged,
        "selected_filter_method": selected_method,
        "n_issues_confident_learning": method_selection_info["confident_learning"],
        "n_issues_prune_by_noise_rate": method_selection_info["prune_by_noise_rate"],
        "n_issues_estimated_theoretical": method_selection_info["estimated_theoretical"],
    }
    log_df = pd.DataFrame([row])

    if os.path.exists(config.PROXY_QUALITY_LOG_FILE):
        existing = pd.read_csv(config.PROXY_QUALITY_LOG_FILE)
        existing = existing[existing["proxy_id"] != config.PROXY_ID]
        log_df = pd.concat([existing, log_df], ignore_index=True).sort_values("proxy_id")

    log_df.to_csv(config.PROXY_QUALITY_LOG_FILE, index=False)
    print(f"📄 Tabel ablasi proxy diperbarui -> {config.PROXY_QUALITY_LOG_FILE}")


# ==========================================================
# 3. PEMILIHAN METODE FILTER — kriteria eksplisit, bukan hardcode
# ==========================================================
def _select_filter_method(labels, pred_probs, results):
    """
    PERBAIKAN PENTING: versi sebelumnya memanggil num_label_issues() tanpa
    estimation_method eksplisit, yang berarti memakai default 'off_diagonal'.
    Dokumentasi resmi cleanlab menyatakan eksplisit bahwa 'off_diagonal'
    "Returns the same value as sum(find_label_issues(filter_by='confident_learning'))"
    -- artinya num_label_issues() dan filter_by='confident_learning' menghitung
    KUANTITAS YANG SAMA PERSIS secara definisi, bukan dua estimasi independen.

    Akibatnya, kriteria "pilih metode filter yang paling dekat dengan
    num_label_issues()" TIDAK PERNAH bisa memilih 'prune_by_noise_rate' --
    'confident_learning' otomatis menang dengan selisih 0 pada data apa pun.
    Ini terkonfirmasi secara empiris: pada seluruh proxy P1-P5 di pilot run,
    selisihnya selalu persis 0 tanpa kecuali.

    PERBAIKAN: memakai estimation_method='off_diagonal_calibrated', yang
    mengkalibrasi confident joint (np.sum(cj) == len(labels), dst.) sebelum
    menghitung off-diagonal -- angka ini BERBEDA secara matematis dari hitungan
    'confident_learning' mentah, sehingga perbandingan "paling dekat" kembali
    punya kekuatan diskriminatif yang sesungguhnya (bisa memilih salah satu
    dari dua metode filter, bukan otomatis satu-satunya).
    """
    estimated_n = int(num_label_issues(
        labels=labels, pred_probs=pred_probs,
        estimation_method="off_diagonal_calibrated",
    ))

    counts = {method: int(issues.sum()) for method, issues in results.items()}
    selected_method = min(counts, key=lambda m: abs(counts[m] - estimated_n))

    print(f"\n🎯 Pemilihan metode filter (kriteria: paling dekat dengan estimasi teoretis terkalibrasi):")
    print(f"   Estimasi teoretis (num_label_issues, off_diagonal_calibrated) : {estimated_n} baris")
    for method, n in counts.items():
        marker = " <-- DIPILIH" if method == selected_method else ""
        print(f"   '{method}': {n} baris (selisih: {abs(n - estimated_n)}){marker}")

    return selected_method, {**counts, "estimated_theoretical": estimated_n}


# ==========================================================
# 4. PIPELINE UTAMA CONFIDENT LEARNING
# ==========================================================
def run_confident_learning():
    print("=" * 60)
    print(f" CONFIDENT LEARNING — proxy aktif: [{config.PROXY_ID}] {config.PROXY_NAME} ")
    print("=" * 60)

    df_train = pd.read_csv(config.TRAIN_RAW_FILE)

    if config.DEBUG_MODE and len(df_train) > config.DEBUG_SAMPLE_SIZE:
        df_train, _ = train_test_split(
            df_train, train_size=config.DEBUG_SAMPLE_SIZE,
            random_state=42, stratify=df_train["rating"],
        )
        print(f"[DEBUG] subset -> {len(df_train)} baris")

    print(f"📥 Memuat {len(df_train)} baris data train.")
    labels = df_train["rating"].values - 1  # 0-indexed
    texts = df_train["cleaned_text"].tolist()

    # ---- INTEGRASI DENGAN proxy.py: satu panggilan, tidak perduli metodenya ----
    pred_probs = get_proxy_pred_probs(texts, labels)

    if pred_probs.shape != (len(texts), config.NUM_CLASSES):
        raise ValueError(
            f"Bentuk pred_probs dari proxy [{config.PROXY_NAME}] tidak sesuai: "
            f"dapat {pred_probs.shape}, expected ({len(texts)}, {config.NUM_CLASSES}). "
            f"Cek implementasi proxy ini di src/proxy.py."
        )

    proxy_preds = np.argmax(pred_probs, axis=1)
    # proxy_metrics["accuracy"] adalah satu-satunya sumber exact accuracy --
    # sebelumnya dihitung dua kali terpisah (proxy_acc manual + compute_metrics),
    # redundan dan berisiko dua angka beda kalau salah satu diedit nanti.
    proxy_metrics = compute_metrics(labels, proxy_preds)

    print(f"\n📐 Kualitas proxy [{config.PROXY_NAME}]:")
    print(f"   Exact Accuracy : {proxy_metrics['accuracy']:.4f}")
    print(f"   MAE            : {proxy_metrics['mae']:.4f}")
    print(f"   Off-by-1 Acc   : {proxy_metrics['off_by_one']:.4f}")
    print(f"   QWK            : {proxy_metrics['qwk']:.4f}")

    df_train = df_train.copy()
    df_train["predicted_rating"] = proxy_preds + 1
    df_train["rating_diff"] = (df_train["rating"] - df_train["predicted_rating"]).abs()

    # ---- filter cleanlab, >=2 metode dihitung, lalu dipilih via kriteria eksplisit ----
    # min_examples_per_class dipasang eksplisit: tanpa ini, kelas minoritas
    # (rating 2/3) berisiko diperlakukan tidak stabil oleh cleanlab saat
    # mengestimasi confident joint per kelas (lihat dokumentasi cleanlab
    # soal galat estimasi pada kelas kecil). Nilai 20 dipilih sebagai batas
    # bawah kasar -- laporkan di Bab IV kalau ada kelas yang jumlahnya
    # mendekati batas ini di data train.
    MIN_EXAMPLES_PER_CLASS = 20

    results = {}
    print("\n🔎 Analisis Metode Filter Cleanlab:")
    for method in config.CLEANLAB_FILTER_METHODS:
        issues = cleanlab.filter.find_label_issues(
            labels=labels,
            pred_probs=pred_probs,
            filter_by=method,
            min_examples_per_class=MIN_EXAMPLES_PER_CLASS,
        )
        results[method] = issues
        print(f"   '{method}': {issues.sum()} baris diflag ({issues.sum()/len(df_train)*100:.2f}%)")

    selected_method, method_selection_info = _select_filter_method(labels, pred_probs, results)

    df_train["is_noise"] = results[selected_method]
    for method, issues in results.items():
        df_train[f"is_noise__{method}"] = issues

    df_train["is_noise_severe"] = df_train["is_noise"] & (df_train["rating_diff"] >= config.SEVERITY_THRESHOLD)
    df_noise = df_train[df_train["is_noise"]].copy()   # untuk pelaporan & sampling manusia (unresolved)

    print(f"\n✅ Deteksi selesai (metode terpilih: {selected_method}).")
    print("\n📊 Distribusi rating_diff pada baris noise (sebelum resolusi):")
    print(df_noise["rating_diff"].value_counts().sort_index())

    # ================= BARU: RESOLUSI KONFLIK PASCA-DETEKSI (III.D butir 8) =================
    n_before = len(df_train)
    df_train_resolved = drop_text_rating_conflicts(df_train, text_col="cleaned_text")
    n_after = len(df_train_resolved)
    print(f"\n🗳️  Resolusi konflik teks-identik (voting mayoritas, PASCA-deteksi CL): "
          f"{n_before} -> {n_after} baris (dibuang {n_before - n_after})")

    # Tiga varian data latih diturunkan dari df_train_resolved (bukan df_train mentah):
    #   raw    = df_train_resolved TANPA pruning berbasis noise (baseline M1/M4)
    #   hard   = df_train_resolved MINUS seluruh baris is_noise
    #   severe = df_train_resolved MINUS baris is_noise & rating_diff >= SEVERITY_THRESHOLD
    df_train_raw_final = df_train_resolved.copy()
    df_cleaned_hard = df_train_resolved[~df_train_resolved["is_noise"]].copy()
    df_cleaned_severe = df_train_resolved[~df_train_resolved["is_noise_severe"]].copy()

    print(f"   Raw (resolved, baseline)     : {len(df_train_raw_final)}")
    print(f"   Hard-prune     : buang {df_train_resolved['is_noise'].sum()} / sisa {len(df_cleaned_hard)}")
    print(f"   Severity-aware : buang {df_train_resolved['is_noise_severe'].sum()} / sisa {len(df_cleaned_severe)}")

    _log_proxy_quality(proxy_metrics, pct_flagged=len(df_noise)/len(df_train)*100,
                        selected_method=selected_method, method_selection_info=method_selection_info)

    drop_cols = [c for c in df_train_raw_final.columns
                 if c.startswith("is_noise") or c in ("predicted_rating", "rating_diff")]
    df_train_raw_final.drop(columns=drop_cols, errors="ignore").to_csv(config.TRAIN_RAW_RESOLVED_FILE, index=False)
    df_cleaned_hard.drop(columns=drop_cols, errors="ignore").to_csv(config.TRAIN_CLEANED_HARD_FILE, index=False)
    df_cleaned_severe.drop(columns=drop_cols, errors="ignore").to_csv(config.TRAIN_CLEANED_SEVERE_FILE, index=False)
    print(f"\n💾 Raw (resolved) -> {config.TRAIN_RAW_RESOLVED_FILE}")
    print(f"💾 Cleaned (hard)   -> {config.TRAIN_CLEANED_HARD_FILE}")
    print(f"💾 Cleaned (severe) -> {config.TRAIN_CLEANED_SEVERE_FILE}")

    df_noise.to_csv(config.NOISE_SAMPLES_FILE, index=False)
    export_human_validation_sample(df_noise)

    return df_noise, proxy_metrics


if __name__ == "__main__":
    run_confident_learning()