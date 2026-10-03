"""
selfcheck.py: pemeriksaan mandiri pada akhir tiap bagian notebook.

Hanya pemeriksaan umum (keberadaan dan ukuran berkas, jumlah baris, overlap nol, kesesuaian
hash, kelengkapan sesi pelatihan). Tidak ada angka hasil dari run sebelumnya yang dipakai
sebagai nilai harapan.
"""

import json
import os

import numpy as np
import pandas as pd

from src import config, drive_io


class SelfCheckFailed(RuntimeError):
    pass


def _report(title, checks, warnings=None):
    gagal = [m for ok, m in checks if not ok]
    print(f"Pemeriksaan mandiri: {title}")
    for ok, m in checks:
        print(f"  [{'OK' if ok else 'BERHENTI'}] {m}")
    for w in (warnings or []):
        print(f"  [PERINGATAN] {w}")
    if gagal:
        raise SelfCheckFailed(f"[BERHENTI] {title}: " + "; ".join(gagal))
    print(f"[OK] {title}: seluruh pemeriksaan umum lulus.")
    return True


def _file_ok(path, min_bytes=1):
    return os.path.isfile(path) and os.path.getsize(path) >= min_bytes


def check_environment():
    from src import env_log
    checks = []
    try:
        v = env_log.check_coral_version()
        checks.append((True, f"coral-pytorch {v} memenuhi syarat minimal {config.MIN_CORAL_PYTORCH_VERSION}"))
    except RuntimeError as e:
        checks.append((False, str(e)))
    try:
        import torch
        checks.append((torch.cuda.is_available(), "GPU CUDA tersedia"))
    except Exception as e:
        checks.append((False, f"torch tidak dapat diimpor: {e}"))
    for d in config.all_shared_dirs():
        checks.append((os.path.isdir(d), f"folder ada: {os.path.relpath(d, config.DRIVE_ROOT)}"))
    return _report("1. Lingkungan dan Drive", checks)


def check_corpus():
    p = os.path.join(config.SHARED_RESULTS_DIR, "corpus_verification.json")
    checks = [(_file_ok(p), "berkas corpus_verification.json ada")]
    if _file_ok(p):
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        checks.append((r["sha256_dihitung"] == config.EXPECTED_MASTER_SHA256, "SHA-256 sama dengan nilai di kode"))
        checks.append((r["n_baris_dihitung"] == config.EXPECTED_MASTER_ROWS, "jumlah baris sama dengan nilai di kode"))
    return _report("2. Verifikasi korpus", checks)


def check_quality_report():
    names = ["00_ringkasan_kualitas_data.csv", "01_distribusi_rating_keseluruhan.csv",
             "01_distribusi_rating_per_aplikasi.csv", "02_distribusi_tanggal_per_app_rating.csv",
             "03_ringkasan_panjang_teks.csv", "03_panjang_teks_per_rating.csv",
             "04_teks_duplikat_persis_top200.csv", "07_simulasi_overlap_split_acak_biasa.csv"]
    checks = [(_file_ok(os.path.join(config.QUALITY_DIR, n)), f"ada dan tidak kosong: {n}") for n in names]
    pres = os.path.join(config.PRESERVED_DIR, "scraping_summary.csv")
    warn = [] if _file_ok(pres) else [
        "preserved/scraping_summary.csv tidak ditemukan. Angka filter bahasa wajib dilaporkan di Bab III "
        "dan hanya berasal dari berkas ini."]
    return _report("3. Laporan kualitas data", checks, warn)


def check_preprocessing():
    checks = [(_file_ok(config.CLEAN_TEXT_FILE), "reviews_clean.csv ada")]
    if _file_ok(config.CLEAN_TEXT_FILE):
        df = pd.read_csv(config.CLEAN_TEXT_FILE)
        checks.append((len(df) > 0, f"jumlah baris {len(df)}"))
        checks.append(({"review_id", "cleaned_text", "rating"}.issubset(df.columns), "kolom wajib ada"))
        checks.append((not df["cleaned_text"].astype(str).str.strip().eq("").any(), "tidak ada teks kosong"))
        checks.append((not df.duplicated(subset=["cleaned_text", "rating"]).any(), "tidak ada duplikat murni"))
        checks.append((df["review_id"].is_unique, "review_id unik"))
    return _report("4. Praproses", checks)


def check_split(ctx):
    sp = os.path.join(ctx.data_dir, "split_summary.csv")
    checks = [(_file_ok(sp), f"split_summary.csv ada (split {ctx.split_seed})")]
    if _file_ok(sp):
        s = pd.read_csv(sp).iloc[0]
        checks.append((s["n_train"] + s["n_val"] + s["n_test"] == s["n_total"], "jumlah train + val + test sama dengan total"))
        for c in [c for c in s.index if c.startswith("overlap_")]:
            checks.append((int(s[c]) == 0, f"{c} = {int(s[c])} (harus 0)"))
    for n in ("train", "val", "test"):
        checks.append((drive_io.is_verified(ctx.split_file(n)), f"berkas split {n} terverifikasi"))
    return _report(f"5. Pembagian data (split {ctx.split_seed})", checks)


def check_p4(ctx):
    checks = []
    n_train = len(pd.read_csv(ctx.split_file("train")))
    oof = os.path.join(ctx.proxy_dir, f"oof__finetuned_corn__K{config.PROXY_CV_FOLDS}.npz")
    checks.append((drive_io.is_verified(oof), "OOF P4 K=5 tersimpan"))
    if drive_io.is_verified(oof):
        z = np.load(oof, allow_pickle=False)
        cp = z["class_probs"]
        checks.append((cp.shape == (n_train, config.NUM_CLASSES), f"bentuk OOF {cp.shape}"))
        checks.append((np.allclose(cp.sum(axis=1), 1.0, atol=1e-4), "probabilitas kelas berjumlah satu"))
        checks.append(("q_chain" in z.files and z["q_chain"].shape == (n_train, config.NUM_CLASSES - 1),
                       "kumulatif rantai CORN tersimpan"))
        checks.append((len(z["temperatures"]) == config.PROXY_CV_FOLDS, "satu temperature per fold"))
    for v in ("raw", "hard", "severe"):
        checks.append((drive_io.is_verified(ctx.variant_file(v)), f"varian data latih '{v}' ada"))
    if all(drive_io.is_verified(ctx.variant_file(v)) for v in ("raw", "hard", "severe")):
        n = {v: len(pd.read_csv(ctx.variant_file(v))) for v in ("raw", "hard", "severe")}
        checks.append((n["hard"] <= n["severe"] <= n["raw"], f"urutan ukuran hard <= severe <= raw ({n})"))
    return _report(f"7. P4, deteksi noise, dan varian (split {ctx.split_seed})", checks)


def check_ablation(ctx):
    checks = []
    for pid in (0, 1, 2):
        i = config.PROXY_REGISTRY[pid]
        p = os.path.join(ctx.ablation_dir, f"{i['label']}__{i['name']}__K5__hasil.json")
        checks.append((drive_io.is_verified(p), f"hasil {i['label']} ada"))
    p = os.path.join(ctx.diagnostics_dir, "P4__finetuned_corn__K5__hasil.json")
    checks.append((drive_io.is_verified(p), "hasil P4 ada"))
    checks.append((drive_io.is_verified(os.path.join(ctx.ablation_dir, "proxy_ablation_table.csv")), "tabel ablasi ada"))
    extra = [f for f in os.listdir(ctx.cleaned_dir) if f.startswith("train_")]
    checks.append((sorted(extra) in ([], ["train_hard.csv", "train_raw.csv", "train_severe.csv"]),
                   "varian data bersih hanya berasal dari P4"))
    return _report("6. Ablasi proxy P1 sampai P4", checks)


def check_annotation(ctx, df_noise=None):
    from src import annotation
    checks, warn = [], []
    for key, basis in (("human_validation", None), ("gold", "test")):
        p = config.ANNOT_FILES[key]
        if not os.path.exists(p):
            warn.append(f"{key}: berkas anotasi belum ada (alur B belum dijalankan atau alur A belum disalin).")
            continue
        has = annotation.has_any_verdict(p, "human_verdict" if key == "human_validation" else "human_gold_rating")
        checks.append((os.path.exists(annotation.sidecar_path(p)), f"{key}: sidecar hash ada"))
        if not has:
            warn.append(f"{key}: verdict belum terisi (bagian ini dapat diselesaikan setelah pelatihan).")
    return _report("8. Anotasi manual", checks, warn)


def check_training(ctx):
    mode_path = os.path.join(ctx.training_dir, "run_mode.json")
    checks = [(drive_io.is_verified(mode_path), "run_mode.json ada")]
    warn = []
    if drive_io.is_verified(mode_path):
        mode = json.load(open(mode_path, "r", encoding="utf-8"))
        seeds = mode["weight_seeds"]
        expected = len(config.SCENARIOS) * len(seeds)
        done = 0
        for s in config.SCENARIOS:
            for sd in seeds:
                done += int(drive_io.is_verified(ctx.scenario_pred_file(s["name"], sd))
                            and drive_io.is_verified(ctx.scenario_metrics_file(s["name"], sd)))
        checks.append((done == expected, f"sesi pelatihan lengkap {done}/{expected}"))
        if mode["hemat"]:
            warn.append("OPSI HEMAT aktif (satu seed bobot). Nyatakan sebagai keterbatasan.")
    return _report(f"9. Pelatihan (split {ctx.split_seed})", checks, warn)


def check_significance(ctx):
    d = ctx.significance_dir
    names = ["final_results_table.csv", "significance_test.csv", "effect_sizes_mae.csv",
             "qwk_ensemble_effect.csv", "qwk_per_seed_effect.csv", "tabel_panjang_efek.csv"]
    checks = [(drive_io.is_verified(os.path.join(d, n)), f"ada: {n}") for n in names]
    if drive_io.is_verified(os.path.join(d, "significance_test.csv")):
        n = len(pd.read_csv(os.path.join(d, "significance_test.csv")))
        checks.append((n == len(config.HYPOTHESES), f"jumlah hipotesis pada tabel = {n}"))
    return _report(f"10. Uji signifikansi (split {ctx.split_seed})", checks)


def check_gold(ctx):
    names = ["gold_test_evaluation.csv", "gold_test_effect_sizes.csv", "gold_test_undetermined_summary.csv"]
    checks = [(drive_io.is_verified(os.path.join(ctx.gold_dir, n)), f"ada: {n}") for n in names]
    return _report("11. Uji emas", checks)


def check_k_sensitivity(ctx):
    d = ctx.k_sensitivity_dir
    checks = [(drive_io.is_verified(os.path.join(d, n)), f"ada: {n}")
              for n in ("k_sensitivity_table.csv", "k_sensitivity_qwk_pairs.csv")]
    if drive_io.is_verified(os.path.join(d, "k_sensitivity_table.csv")):
        t = pd.read_csv(os.path.join(d, "k_sensitivity_table.csv"))
        checks.append((sorted(t["K"].tolist()) == sorted(config.K_SENSITIVITY_VALUES), f"nilai K pada tabel: {t['K'].tolist()}"))
    return _report("12. Sensitivitas K", checks)


def check_partition_sensitivity():
    checks, warn = [], []
    for s in config.SPLIT_SEEDS:
        ctx = config.RunPaths(s)
        mode_path = os.path.join(ctx.training_dir, "run_mode.json")
        if not drive_io.is_verified(mode_path):
            checks.append((False, f"split {s}: run_mode.json tidak ada"))
            continue
        mode = json.load(open(mode_path, "r", encoding="utf-8"))
        n_exp = len(config.SCENARIOS) * len(mode["weight_seeds"])
        done = sum(int(drive_io.is_verified(ctx.scenario_pred_file(sc["name"], sd))) for sc in config.SCENARIOS
                   for sd in mode["weight_seeds"])
        checks.append((done == n_exp, f"split {s}: prediksi lengkap {done}/{n_exp}"))
        checks.append((drive_io.is_verified(os.path.join(ctx.significance_dir, "tabel_panjang_efek.csv")),
                       f"split {s}: tabel panjang efek ada"))
        if mode["hemat"]:
            warn.append(f"split {s}: OPSI HEMAT aktif (satu seed bobot). Nyatakan sebagai keterbatasan.")
    checks.append((drive_io.is_verified(os.path.join(config.SENSITIVITY_SUMMARY_DIR, "ringkasan_konsistensi.csv")),
                   "ringkasan konsistensi ada"))
    return _report("13. Sensitivitas partisi", checks, warn)
