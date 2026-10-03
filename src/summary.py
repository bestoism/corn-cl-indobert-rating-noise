"""summary.py: peta keluaran ke subbab proposal dan Bab IV, manifest hasil, dan cadangan."""

import os

import pandas as pd

from src import config, drive_io

# (pola relatif terhadap DRIVE_ROOT, subbab proposal, tabel atau bagian, pemakaian pada Bab IV)
OUTPUT_MAP = [
    ("shared/results/corpus_verification.json", "3.2", "Penguncian korpus", "Bab III: bukti SHA-256 dan jumlah baris"),
    ("preserved/scraping_summary.csv", "3.2", "Filter bahasa per aplikasi dan rating (DIPERTAHANKAN)", "Bab III: jumlah dan persentase ulasan digugurkan filter bahasa"),
    ("shared/results/quality/02_distribusi_tanggal_per_app_rating.csv", "3.2", "Confound temporal", "Bab IV: karakteristik dan limitasi data"),
    ("shared/results/quality/03_panjang_teks_per_rating.csv", "3.2", "Panjang teks per rating", "Bab IV: validasi dugaan penetapan kuota"),
    ("shared/results/quality/04_teks_duplikat_persis_top200.csv", "3.2", "Duplikat persis", "Bab IV: indikasi template atau spam"),
    ("shared/results/quality/05_konflik_sebaran_selisih_rating.csv", "3.2, 3.3", "Teks identik rating berbeda", "Bab IV: ambiguitas alami"),
    ("shared/results/preprocessing_summary.csv", "3.3", "Ringkasan praproses dan truncation", "Bab III dan IV"),
    ("runs/split_seed42/data/split_summary.csv", "3.4", "Ringkasan split dan bukti nol overlap", "Bab III: pembagian data"),
    ("runs/split_seed42/ablation/proxy_ablation_table.csv", "3.5, 3.6", "Tabel 3.1 (hasil ablasi proxy)", "Bab IV: ablasi P1 sampai P4, risiko sirkularitas"),
    ("runs/split_seed42/diagnostics/P4__finetuned_corn__K5__kualitas_per_kelas.csv", "3.6", "Kualitas proxy per kelas rating asli", "Bab IV: rujukan pemeriksaan confound"),
    ("runs/split_seed42/diagnostics/P4__finetuned_corn__K5__temperature_per_fold.csv", "2.1.8, 3.6", "Temperature per fold", "Bab IV: kalibrasi"),
    ("runs/split_seed42/diagnostics/P4__finetuned_corn__K5__hasil.json", "2.1.8, 3.6", "ECE, monotonisitas, selisih rantai vs turunan", "Bab IV: kalibrasi"),
    ("runs/split_seed42/diagnostics/P4__jarak_ordinal_noise_per_kelas.csv", "3.8", "Jarak ordinal pada baris noise per kelas", "Bab IV: validasi ambang severity"),
    ("runs/split_seed42/noise/variant_summary.csv", "3.6, 3.8", "Jumlah baris per varian data", "Bab IV: susut data per strategi pruning"),
    ("runs/split_seed42/validation/human_validation_summary.csv", "3.7", "Kesepakatan manusia vs Confident learning", "Bab IV: RQ2"),
    ("runs/split_seed42/validation/human_validation_per_bin.csv", "3.7, 3.8", "Kesepakatan per bin jarak ordinal", "Bab IV: dukungan atau penolakan ambang severity"),
    ("runs/split_seed42/validation/human_validation_kappa.csv", "3.7", "Cohen's kappa antar-penilai", "Bab IV: subjektivitas domain (RQ4)"),
    ("runs/split_seed42/significance/final_results_table.csv", "3.9, 3.11", "Tabel hasil enam skenario (rata-rata dan simpangan baku)", "Bab IV: RQ3 dan RQ4"),
    ("runs/split_seed42/significance/significance_test.csv", "3.12", "Tabel 3.4 (H1 sampai H5, Wilcoxon + Holm-Bonferroni)", "Bab IV: pengujian hipotesis"),
    ("runs/split_seed42/significance/effect_sizes_mae.csv", "3.12", "Effect size MAE dan CI bootstrap", "Bab IV: besaran efek"),
    ("runs/split_seed42/significance/qwk_ensemble_effect.csv", "3.12", "Selisih QWK ensemble dan CI (definisi utama)", "Bab IV: metrik keputusan utama"),
    ("runs/split_seed42/significance/qwk_per_seed_effect.csv", "3.12", "Selisih QWK rata-rata per seed (TAMBAHAN)", "Bab IV: analisis sensitivitas agregasi"),
    ("runs/split_seed42/gold/gold_test_evaluation.csv", "3.10", "Evaluasi enam skenario pada subset uji emas", "Bab IV: pemeriksaan ketahanan"),
    ("runs/split_seed42/gold/gold_test_effect_sizes.csv", "3.10", "Arah dan effect size subset emas", "Bab IV: pemeriksaan ketahanan"),
    ("runs/split_seed42/gold/gold_test_undetermined_summary.csv", "3.10", "Proporsi opsi ND", "Bab IV: batas keandalan prediksi berbasis teks"),
    ("runs/split_seed42/gold/gold_test_kappa.csv", "3.10", "Kappa berbobot kuadratik antar-penilai", "Bab IV: reliabilitas anotasi"),
    ("runs/split_seed42/k_sensitivity/k_sensitivity_table.csv", "3.5", "Sensitivitas K = 3, 5, 10", "Bab IV: konfirmasi K = 5"),
    ("runs/split_seed42/k_sensitivity/k_sensitivity_qwk_pairs.csv", "3.5", "CI selisih QWK antar K (TAMBAHAN)", "Bab IV: ukuran ketidakpastian"),
    ("shared/results/sensitivity_partisi/ringkasan_konsistensi.csv", "3.4, 3.12", "Konsistensi lintas tiga split", "Bab IV: stabilitas terhadap undian partisi"),
    ("shared/results/sensitivity_partisi/wilcoxon_berdampingan_per_split.csv", "3.12", "Wilcoxon per split (tidak digabung)", "Bab IV: sensitivitas partisi"),
    ("shared/results/sensitivity_partisi/tabel_panjang_semua_split.csv", "3.4, 3.12", "Tabel panjang per split", "Lampiran Bab IV"),
    ("logs/environment__*.json", "3.13", "Pencatatan lingkungan", "Bab III: alat dan lingkungan"),
]


def build_output_map():
    rows = []
    for rel, sub, tabel, bab4 in OUTPUT_MAP:
        full = os.path.join(config.DRIVE_ROOT, rel)
        if "*" in rel:
            import glob
            ada = len(glob.glob(full)) > 0
        else:
            ada = os.path.isfile(full) and os.path.getsize(full) > 0
        rows.append({"berkas": rel, "subbab_proposal": sub, "tabel_atau_bagian": tabel,
                     "pemakaian_bab_empat": bab4, "ada": "ya" if ada else "TIDAK"})
    return pd.DataFrame(rows)


def build_results_manifest(roots=None):
    """SHA-256 seluruh berkas hasil di bawah runs, shared, logs, annotations, dan preserved."""
    roots = roots or [config.RUNS_DIR, config.SHARED_DIR, config.LOGS_DIR, config.ANNOT_DIR, config.PRESERVED_DIR]
    manifest_path = os.path.join(config.SHARED_RESULTS_DIR, "results_manifest.csv")
    rows = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                if full == manifest_path:
                    continue
                rows.append({"berkas": os.path.relpath(full, config.DRIVE_ROOT),
                             "bytes": os.path.getsize(full), "sha256": drive_io.sha256_file(full)})
    df = pd.DataFrame(rows).sort_values("berkas").reset_index(drop=True)
    res = drive_io.write_once_csv(df, manifest_path)
    return df, res
