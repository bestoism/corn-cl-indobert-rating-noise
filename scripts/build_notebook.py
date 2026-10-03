"""
build_notebook.py: membangun notebooks/skripsi_final.ipynb dari definisi sel di bawah.
Menjalankan: python scripts/build_notebook.py
"""

import os
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CELLS = []


def md(text):
    CELLS.append(new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(new_code_cell(text.strip("\n")))


def section(nomor, judul, tujuan, rujukan, masukan, keluaran, waktu, baca, peringatan):
    md(f"""
## {nomor}. {judul}

**Tujuan.** {tujuan}

**Rujukan proposal.** {rujukan}

**Masukan.** {masukan}

**Keluaran.** {keluaran}

**Perkiraan waktu GPU.** {waktu} (perkiraan perencanaan mengikuti Subbab 3.13, bukan jaminan, dan bergantung pada GPU yang dialokasikan Colab).

**Cara membaca hasil.** {baca}

**Peringatan.** {peringatan}
""")


# ======================================================================
# SEL PERTAMA
# ======================================================================
md("""
# Integrasi Confident Learning dan CORN pada IndoBERT untuk Prediksi Rating Ulasan Google Play Store

**Notebook final (run murni)**

| | |
| --- | --- |
| Nama | Ryan Besto Saragih |
| NIM | 23051204205 |
| Pembimbing | Anita Qoiriah, S.Kom., M.Kom. |
| Repositori | https://github.com/bestoism/corn-cl-indobert-rating-noise |
| Folder Drive | /content/drive/MyDrive/SKRIPSI_CORN_CL_FINAL |

### Definisi run murni

Pada notebook ini, run murni berarti seluruh artefak komputasi (partisi data, probabilitas proxy,
varian data bersih, model, prediksi, tabel hasil, dan uji statistik) dibangun ulang dari data raw dan
lexicon dalam eksekusi ini. Notebook tidak bergantung pada cache, checkpoint, atau hasil eksekusi
sebelumnya. Cache di dalam satu eksekusi yang sama, yaitu berkas unik yang dipakai untuk melanjutkan
setelah runtime Colab terputus, tetap diperbolehkan. Input manusia (anotasi manual) hanya dipakai
kembali setelah verifikasi hash terhadap hasil run ini berhasil.

Berkas yang dipertahankan di Drive: data raw (all_reviews_master.csv), lexicon, scraping_summary.csv
dan manifest korpus asli (folder preserved), serta berkas anotasi beserta sidecar hash-nya (folder annotations).

### Daftar isi

1. Persiapan lingkungan dan audit keadaan Drive
2. Verifikasi korpus (hash terkunci)
3. Laporan kualitas data dan EDA (Subbab 3.2)
4. Praproses teks (Subbab 3.3)
5. Pembagian data dan bukti nol overlap (Subbab 3.4)
6. Ablasi proxy classifier P1 sampai P4, kalibrasi, dan diagnostik (Subbab 3.5, 3.6)
7. Penetapan P4, deteksi noise, pemilihan metode filter, resolusi konflik, dan tiga varian data (Subbab 3.6, 3.8)
8. Validasi manusia (Subbab 3.7)
9. Pelatihan enam skenario dan tiga seed (Subbab 3.9)
10. Evaluasi dan uji signifikansi H1 sampai H5 (Subbab 3.11, 3.12)
11. Subset uji emas (Subbab 3.10)
12. Sensitivitas jumlah fold K (Subbab 3.5)
13. Sensitivitas partisi data (Subbab 3.4, 3.12)
14. Ringkasan akhir
15. Manifest hasil dan unduhan cadangan

### Petunjuk menjalankan

- Jalankan sel dari atas ke bawah. Setiap bagian diakhiri sel pemeriksaan mandiri yang mencetak
  [OK] atau [BERHENTI]. Bila [BERHENTI], eksekusi dihentikan dan alasannya dicetak. Perbaiki dahulu, jangan dilewati.
- Sel yang menghentikan eksekusi: audit Drive (bagian 1), verifikasi korpus (bagian 2), pemeriksaan
  mandiri tiap bagian, serta pemeriksaan hash anotasi (bagian 8 dan 11).
- Sel yang boleh dilewati: sel pembersihan (bagian 1, dry run sebagai bawaan), sel cadangan zip, dan
  sel penampil tabel.
- Bila runtime putus: jalankan ulang sel konfigurasi (bagian 1.2) dengan LANJUTKAN_EKSEKUSI = True, lalu
  jalankan ulang sel yang terhenti. Unit kerja yang berkasnya sudah terverifikasi akan dilewati.
- Berkas di Drive tidak pernah ditimpa. Bila sebuah berkas sudah ada dengan isi berbeda, eksekusi berhenti.
- Anotasi manual (bagian 8 dan 11) dapat dikerjakan paralel dengan pelatihan. Pelatihan tidak menunggu anotasi.
""")

# ======================================================================
# 1. LINGKUNGAN
# ======================================================================
section(
    1, "Persiapan lingkungan dan audit keadaan Drive",
    "Memasang Drive dan repositori, memeriksa versi paket (coral-pytorch minimal 1.3.0), mencatat lingkungan, "
    "membuat seluruh folder sekali, dan memastikan Drive berada pada keadaan bersih.",
    "Subbab 3.13 (alat dan lingkungan).",
    "Drive berisi data/raw, lexicon, preserved, dan annotations.",
    "logs/environment__{RUN_TAG}.json; seluruh folder keluaran (runs, shared, logs) dibuat sekali.",
    "kurang dari 5 menit",
    "Audit membandingkan isi Drive dengan daftar yang diizinkan. Pada mode awal, hanya raw, lexicon, preserved, dan "
    "annotations yang boleh ada.",
    "Bila audit menemukan artefak lain atau nama ganda (misalnya akhiran ' (1)'), eksekusi berhenti. Setelah "
    "run dimulai, sel audit hanya lulus pada mode awal bila LANJUTKAN_EKSEKUSI = False dan belum ada artefak run."
)

md("### 1.1 Memasang Drive, repositori, dan paket")
code("""
import os, sys, subprocess

from google.colab import drive
drive.mount('/content/drive')

REPO_URL = "https://github.com/bestoism/corn-cl-indobert-rating-noise"
REPO_DIR = "/content/corn-cl-indobert-rating-noise"

if os.path.exists(REPO_DIR):
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(REPO_DIR, "requirements.txt")], check=True)
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print("[OK] Repositori dan paket siap.")
""")

md("### 1.2 Konfigurasi eksekusi dan impor")
code("""
import json
import numpy as np
import pandas as pd
from IPython.display import display

# ---- Pengaturan yang boleh diubah peneliti ----
LANJUTKAN_EKSEKUSI = False        # True hanya untuk melanjutkan eksekusi yang sama setelah runtime putus
HEMAT_SPLIT_SENSITIVITAS = False  # True: satu seed bobot per split sensitivitas (WAJIB dinyatakan sebagai keterbatasan)
ALUR_ANOTASI_VALIDASI = "B"       # "A": pakai kembali anotasi lama setelah verifikasi hash; "B": buat sampel baru
ALUR_ANOTASI_GOLD = "B"           # idem untuk uji emas
SALIN_CHECKPOINT_KE_DRIVE = False # checkpoint tidak perlu disalin ke Drive (prediksi sudah tersimpan)

from src import (config, drive_io, env_log, corpus, quality_report, preprocess, data_split, ablation, clean,
                 proxy, pipeline, human_validation, gold_test, sensitivity_k, significance, consistency,
                 summary, selfcheck, train)

RUN_TAG = pipeline.new_run_tag()
ctx_main = config.RunPaths(config.MAIN_SPLIT_SEED)
SEEDS_MAIN = pipeline.weight_seeds_for(config.MAIN_SPLIT_SEED)
print("RUN_TAG:", RUN_TAG)
print("Drive root:", config.DRIVE_ROOT)
""")

md("### 1.3 Pemeriksaan versi, audit keadaan Drive, pembuatan folder, dan pencatatan lingkungan")
code("""
coral_version = env_log.check_coral_version()
print("coral-pytorch:", coral_version)

audit_mode = "berjalan" if LANJUTKAN_EKSEKUSI else "awal"
df_audit, duplicates = drive_io.audit_drive_state(mode=audit_mode)

config.ensure_all_dirs()
env_info, _ = env_log.write_environment_log(RUN_TAG)
display(pd.Series({"python": env_info["python"][:40], "gpu": env_info["gpu"], **env_info["paket"]}))
""")

md("""
### 1.4 Pembersihan artefak (opsional, dry run sebagai bawaan)

Sel ini hanya menampilkan daftar yang akan dihapus. Berkas yang dipertahankan (raw, lexicon, preserved,
annotations) tidak pernah dihapus. Penghapusan nyata memerlukan dry_run=False dan konfirmasi persis
'HAPUS SEKARANG'.
""")
code("""
akan_dihapus = drive_io.bersihkan_artefak(dry_run=True)
""")

md("### 1.5 Pemeriksaan mandiri bagian 1")
code("""
selfcheck.check_environment()
""")

# ======================================================================
# 2. KORPUS
# ======================================================================
section(
    2, "Verifikasi korpus (hash terkunci)",
    "Memastikan korpus mentah identik dengan korpus yang dikunci, dengan membandingkan SHA-256 dan jumlah baris "
    "terhadap nilai yang tertulis di kode (src/config.py), bukan terhadap manifest yang dibuat ulang oleh run ini.",
    "Subbab 3.2 dan Batasan Masalah (penguncian berkas melalui hash SHA-256).",
    "data/raw/all_reviews_master.csv.",
    "shared/results/corpus_verification.json.",
    "kurang dari 1 menit",
    "Kedua status cocok (SHA-256 dan jumlah baris) harus bernilai True.",
    "Scraping tidak boleh dijalankan ulang. Bila hash tidak cocok, eksekusi berhenti karena seluruh eksperimen "
    "harus berjalan di atas korpus yang identik."
)
code("""
verif = corpus.verify_master()
display(pd.Series(verif))
corpus.write_verification(verif)
""")
code("""
selfcheck.check_corpus()
""")

# ======================================================================
# 3. KUALITAS DATA
# ======================================================================
section(
    3, "Laporan kualitas data dan EDA",
    "Menghasilkan karakterisasi korpus berkuota: distribusi rating, distribusi tanggal per aplikasi dan rating "
    "(confound temporal), distribusi panjang teks, teks duplikat persis, teks identik dengan rating berbeda, dan "
    "simulasi risiko kebocoran pada split acak biasa.",
    "Subbab 3.2 (kontrol kualitas data).",
    "all_reviews_master.csv (seluruhnya dapat dibuat ulang tanpa scraping).",
    "shared/results/quality/*.csv dan tiga gambar (fig01 sampai fig03).",
    "tanpa GPU, kurang dari 2 menit",
    "Tabel tanggal menunjukkan apakah kelas rating jarang (2 dan 3) ditelusuri lebih jauh ke belakang. Tabel panjang "
    "teks dan duplikat dipakai untuk memeriksa dugaan penetapan kuota pada Subbab 3.2. Hasil hanya mendeskripsikan "
    "data dan tidak mengubah keputusan desain.",
    "Jumlah ulasan yang diperiksa dan digugurkan filter bahasa (Bab III) TIDAK dapat dibuat ulang dari master. Angkanya "
    "hanya ada pada preserved/scraping_summary.csv."
)
code("""
qr = quality_report.run_quality_report()
display(qr["rating"]); display(qr["per_app"]); display(qr["severity"])
display(qr["length_per_rating"]); display(pd.Series(qr["summary"]))
""")
code("""
selfcheck.check_quality_report()
""")

# ======================================================================
# 4. PRAPROSES
# ======================================================================
section(
    4, "Praproses teks",
    "Membersihkan teks sesuai Subbab 3.3: case folding, penghapusan URL dan username, translasi emoji, normalisasi "
    "elongasi, normalisasi slang (Kamus Alay dan entri domain), dan penghapusan duplikat murni. Resolusi teks identik "
    "dengan rating berbeda sengaja ditunda sampai setelah deteksi noise (bagian 7).",
    "Subbab 3.3.",
    "all_reviews_master.csv dan lexicon/slang_base.csv (serta lexicon/slang_domain.csv bila ada).",
    "shared/processed/reviews_clean.csv dan shared/results/preprocessing_summary.csv.",
    "tanpa GPU, beberapa menit (termasuk menghitung tingkat pemotongan token)",
    "Perhatikan jumlah baris setelah pembersihan, persentase teks sangat pendek, dan persentase ulasan yang "
    "terpotong pada 128 token.",
    "Kamus tidak diunduh otomatis. Bila lexicon/slang_domain.csv tidak ada, hanya Kamus Alay yang dipakai dan "
    "peringatan dicetak (Subbab 3.3 menyebut entri domain dari PRDECT-ID)."
)
code("""
if drive_io.is_verified(config.CLEAN_TEXT_FILE):
    df_clean = pd.read_csv(config.CLEAN_TEXT_FILE)
    print(f"[OK] reviews_clean.csv sudah ada: {len(df_clean)} baris.")
else:
    df_clean, prep_summary = preprocess.run_preprocessing()
    display(pd.Series(prep_summary))
""")
code("""
selfcheck.check_preprocessing()
""")

# ======================================================================
# 5. SPLIT
# ======================================================================
section(
    5, "Pembagian data dan bukti nol overlap",
    "Membagi data 70:10:20 (train, validasi, uji) secara berkelompok berdasarkan teks, stratified menurut rating "
    "mayoritas grup, dengan seed partisi 42 (split utama). Mencatat bukti angka bahwa overlap teks dan review_id "
    "lintas partisi adalah nol.",
    "Subbab 3.4.",
    "shared/processed/reviews_clean.csv.",
    "runs/split_seed42/data/split_train.csv, split_val.csv, split_test.csv, split_summary.csv.",
    "tanpa GPU, kurang dari 1 menit",
    "Kolom overlap_* pada split_summary.csv harus bernilai 0. Proporsi baris dalam grup teks duplikat dilaporkan.",
    "Seed partisi (42, 123, 2024) berbeda dari seed bobot model (42, 123, 2024). Keduanya tidak boleh dicampur."
)
code("""
df_train_main, df_val_main, df_test_main = data_split.create_splits(ctx_main)
display(pd.read_csv(os.path.join(ctx_main.data_dir, "split_summary.csv")).T)
""")
code("""
selfcheck.check_split(ctx_main)
""")

# ======================================================================
# 6. ABLASI PROXY
# ======================================================================
section(
    6, "Ablasi proxy classifier P1 sampai P4, kalibrasi, dan diagnostik",
    "Menjalankan empat tahap proxy pada skema stratified K-Fold yang identik (K = 5, shuffle, seed 42): P1 (CLS beku "
    "dan Logistic Regression), P2 (mean-pooling beku dan Logistic Regression), P3 (IndoBERT fine-tuned, Cross-Entropy) "
    "dan P4 (IndoBERT fine-tuned, CORN, proxy final). P3 dan P4 dikalibrasi dengan temperature scaling per fold. "
    "Untuk tiap proxy dihitung kualitas (agregat dan per kelas rating asli), kalibrasi, dan jumlah baris ter-flag.",
    "Subbab 3.5 dan 3.6; kalibrasi pada Subbab 2.1.8.",
    "runs/split_seed42/data/split_train.csv.",
    "runs/split_seed42/ablation/ (P1 sampai P3 dan tabel ablasi), runs/split_seed42/diagnostics/ (P4), "
    "runs/split_seed42/proxy/ (berkas OOF dan berkas per fold).",
    "P1 dan P2 di bawah 15 menit; P3 sekitar 50 menit; P4 sekitar 50 menit",
    "Perbandingan P1 sampai P4 menjadi jalur utama pemeriksaan sirkularitas. ECE kumulatif dihitung sebagai ECE biner "
    "per ambang pada kejadian {y > k}, dengan 10 bin lebar sama, dirata-rata atas K-1 = 4 ambang, terhadap label "
    "teramati. Monotonisitas diperiksa pada kumulatif langsung dari rantai CORN (setelah temperature scaling, sebelum "
    "konversi ke probabilitas kelas). Karena kumulatif rantai adalah hasil kali bilangan pada [0, 1], pemeriksaan ini "
    "adalah uji kewarasan numerik; bagian yang informatif adalah selisih antara kumulatif rantai dan kumulatif turunan "
    "(akibat pembatasan minimal 1e-8 dan normalisasi ulang) serta nilai ECE.",
    "P1 sampai P3 tidak mengekspor sampel validasi manusia maupun varian data bersih. Hanya P4 yang melakukannya (bagian 7). "
    "Berkas per fold memungkinkan melanjutkan bila runtime putus."
)
code("""
df_train_main = pd.read_csv(ctx_main.split_file("train"))
ev_p1 = ablation.run_proxy_evaluation(ctx_main, df_train_main, 0, config.PROXY_CV_FOLDS, ctx_main.ablation_dir)
ev_p2 = ablation.run_proxy_evaluation(ctx_main, df_train_main, 1, config.PROXY_CV_FOLDS, ctx_main.ablation_dir)
""")
code("""
ev_p3 = ablation.run_proxy_evaluation(ctx_main, df_train_main, 2, config.PROXY_CV_FOLDS, ctx_main.ablation_dir)
""")
code("""
ev_p4 = ablation.run_proxy_evaluation(ctx_main, df_train_main, 3, config.PROXY_CV_FOLDS, ctx_main.diagnostics_dir)
""")
md("### 6.4 Tabel ablasi (Tabel 3.1) dan diagnostik P4")
code("""
ablation_table = pipeline.step_ablation_p1_p3(ctx_main, df_train_main, ev_p4)
display(ablation_table)
""")
code("""
res = ev_p4["result"]
cal_keys = [k for k in res if k.startswith("ece_") or k.startswith("monotonisitas_") or k.startswith("selisih_rantai")]
display(pd.Series({k: res[k] for k in cal_keys}, name="kalibrasi P4"))
display(pd.read_csv(os.path.join(ctx_main.diagnostics_dir, "P4__finetuned_corn__K5__temperature_per_fold.csv")))
display(ev_p4["per_class"])
""")
code("""
selfcheck.check_ablation(ctx_main)
""")
code("""
# Unduhan cadangan lokal (boleh dilewati)
bk = drive_io.export_backup_zip(config.DRIVE_ROOT, f"cadangan_bagian6_{RUN_TAG}.zip")
print(bk)
drive_io.download_if_colab(bk["zip_path"])
""")

# ======================================================================
# 7. P4 FINAL
# ======================================================================
section(
    7, "Penetapan P4 sebagai proxy final, deteksi noise, pemilihan metode filter, resolusi konflik, dan tiga varian data",
    "Menetapkan P4 sebagai proxy final (ditetapkan di muka), menjalankan Confident learning (min_examples_per_class = 20) "
    "dengan dua metode filter, memilih metode yang paling dekat dengan estimasi num_label_issues() varian "
    "off_diagonal_calibrated, menjalankan resolusi konflik teks identik pasca deteksi, dan menurunkan tiga varian data "
    "latih: raw (resolved), hard-pruned, dan severity-aware (jarak ordinal 2 atau lebih).",
    "Subbab 3.5 (aturan keputusan P4), 3.6, dan 3.8.",
    "OOF P4 dari bagian 6 (dimuat dari berkas, tidak dilatih ulang).",
    "runs/split_seed42/cleaned/train_{raw,hard,severe}.csv; runs/split_seed42/noise/ (flags, baris ter-flag, ringkasan "
    "varian); runs/split_seed42/diagnostics/P4__jarak_ordinal_noise_per_kelas.csv.",
    "kurang dari 5 menit (tanpa pelatihan ulang)",
    "Tabel jarak ordinal per kelas memuat persentase baris ter-flag dengan jarak di atas ambang. Kolom penanda_confound "
    "bernilai YA bila kelas itu berada di dua teratas pada persentase jarak berat sekaligus di dua terbawah pada recall "
    "(atau precision) proxy, sehingga terindikasi confound antara proxy lemah dan kelas yang memang banyak berlabel salah.",
    "Setelah penghapusan duplikat murni pada praproses, setiap grup teks konflik memuat satu baris per rating, sehingga "
    "seluruh grup konflik seri dan dibuang seluruhnya. Voting mayoritas karena itu tidak pernah memilih rating mayoritas. "
    "Hal ini dinyatakan apa adanya (lihat dokumen audit)."
)
code("""
out_p4 = pipeline.step_p4(ctx_main, df_train_main)
display(pd.Series(out_p4["variant_stats"]))
display(out_p4["noise_by_class"])
df_noise_main = out_p4["df_noise"]
""")
code("""
selfcheck.check_p4(ctx_main)
""")

# ======================================================================
# 8. VALIDASI MANUSIA
# ======================================================================
section(
    8, "Validasi manusia",
    "Mengukur kesepakatan antara Confident learning dan penilai manusia pada sampel baris ter-flag P4. Sampel dinilai "
    "buta (rating_diff dan prediksi tidak diperlihatkan), dengan tiga kategori: noise, not_noise, ambiguous.",
    "Subbab 3.7.",
    "Baris ter-flag P4 pada data latih split utama.",
    "annotations/human_validation_sample.csv (+ sidecar hash), annotations/human_validation_sample_annotator2.csv "
    "(+ sidecar), runs/split_seed42/validation/human_validation_{result,per_bin,summary,kappa}.csv.",
    "tanpa GPU; waktu anotasi bergantung pada penilai (dapat dikerjakan paralel dengan pelatihan)",
    "Kesepakatan dilaporkan keseluruhan dan per bin jarak ordinal, dibahas deskriptif-komparatif tanpa ambang lulus atau "
    "gagal. Cohen's kappa biasa dipakai karena tiga kategori tidak berjenjang. Bin jarak 3 dan 4 berukuran kecil secara "
    "struktural (Subbab 3.8).",
    "Dua alur. ALUR A: anotasi dari eksekusi sebelumnya dipakai kembali. Salin berkas sampel dan sidecar ke folder "
    "annotations, lalu sel verifikasi memeriksa hash daftar review_id ter-flag P4. Pada run murni, himpunan ter-flag "
    "dapat berbeda karena nondeterminisme GPU; bila hash tidak cocok, eksekusi berhenti dan anotasi harus dibuat ulang. "
    "ALUR B: sampel kosong dibuat, peneliti mengisi kolom human_verdict di Drive, lalu perhitungan dijalankan. Berkas "
    "anotasi tidak pernah ditimpa. Pelatihan (bagian 9) tidak perlu menunggu bagian ini selesai."
)
md("### 8.1 Ukuran sampel (Cochran) dan alokasi")
code("""
df_noise_main = pd.read_csv(os.path.join(ctx_main.noise_dir, "flagged_rows_P4_K5.csv"))
cochran_hv = human_validation.report_sample_size(df_noise_main)
""")
md("### 8.2 Alur A (verifikasi hash) atau alur B (membuat sampel baru)")
code("""
if ALUR_ANOTASI_VALIDASI == "A":
    human_validation.verify_annotation(df_noise_main)
elif ALUR_ANOTASI_VALIDASI == "B":
    human_validation.create_sample(ctx_main, df_noise_main)
    human_validation.create_second_annotator()
else:
    raise ValueError("ALUR_ANOTASI_VALIDASI harus 'A' atau 'B'")
""")
md("""
### 8.3 Hitung kesepakatan dan kappa

Jalankan sel ini setelah kolom human_verdict terisi lengkap. Boleh dijalankan setelah pelatihan selesai.
Penilai kedua bersifat opsional: bila tidak ada, gunakan jalur test-retest pada sub-bagian berikutnya.
""")
code("""
summary_hv, breakdown_hv = human_validation.compute_agreement(ctx_main, df_noise_main)
display(summary_hv); display(breakdown_hv)
""")
code("""
try:
    kappa_hv = human_validation.compute_interannotator_kappa(ctx_main)
except (FileNotFoundError, ValueError) as e:
    print("[PERINGATAN] Kappa antar-penilai belum dapat dihitung:", e)
    print("Bila hanya ada satu penilai, gunakan human_validation.create_retest_subset() lalu compute_test_retest(ctx_main).")
""")
code("""
selfcheck.check_annotation(ctx_main)
""")

# ======================================================================
# 9. PELATIHAN
# ======================================================================
section(
    9, "Pelatihan enam skenario dan tiga seed",
    "Melatih M1 sampai M6 (tiga varian data dikali dua fungsi loss) dengan tiga seed bobot (42, 123, 2024) pada konfigurasi "
    "Tabel 3.3: indobenchmark/indobert-base-p1, token maksimum 128, batch 16, learning rate 2e-5, AdamW weight decay 0,01, "
    "warmup linear 10 persen, maksimum 10 epoch, early stopping patience 3 (MAE validasi), mixed precision, checkpoint "
    "terbaik menurut QWK validasi (tie-breaker MAE), dropout 0,3, pooler_output.",
    "Subbab 3.9 (Tabel 3.2 dan 3.3).",
    "runs/split_seed42/cleaned/train_{raw,hard,severe}.csv, split_val.csv, split_test.csv.",
    "Per unit (skenario, seed): runs/split_seed42/predictions/{skenario}__seed{seed}.csv (review_id, y_true, y_pred pada "
    "data uji) dan runs/split_seed42/training/{skenario}__seed{seed}.json. Checkpoint ditulis ke disk lokal Colab.",
    "18 sesi pelatihan, sekitar 8 sampai 12 jam",
    "Setiap sel di bawah menjalankan satu skenario untuk tiga seed. Unit yang sudah terverifikasi dilewati otomatis.",
    "Sel panjang. Bila runtime putus, jalankan ulang sel konfigurasi (LANJUTKAN_EKSEKUSI = True) dan sel yang terhenti. "
    "Berkas prediksi dan metrik ditulis sekali per unit dan tidak pernah ditimpa. Kriteria seed bobot terpisah dari seed partisi."
)
code("""
mode_main = pipeline.register_run_mode(ctx_main, SEEDS_MAIN, hemat=False)
print(mode_main)
""")
for nama in ["M1_Baseline_CE", "M2_CleanedHard_CE", "M3_CleanedSevere_CE",
             "M4_Baseline_CORN", "M5_CleanedHard_CORN", "M6_CleanedSevere_CORN"]:
    code(f"""
status = train.run_scenario(ctx_main, "{nama}", SEEDS_MAIN, copy_ckpt_to_drive=SALIN_CHECKPOINT_KE_DRIVE)
display(status[status["scenario"] == "{nama}"])
""")
code("""
selfcheck.check_training(ctx_main)
""")

# ======================================================================
# 10. SIGNIFIKANSI
# ======================================================================
section(
    10, "Evaluasi dan uji signifikansi H1 sampai H5",
    "Menghitung tabel hasil enam skenario (rata-rata dan simpangan baku tiga seed), uji Wilcoxon Signed-Rank dua arah atas "
    "rata-rata absolute error per sampel (agregasi tiga seed) dengan koreksi Holm-Bonferroni atas lima hipotesis, effect "
    "size MAE dengan CI bootstrap 95 persen, selisih QWK ensemble dengan CI bootstrap (definisi utama), dan selisih QWK "
    "rata-rata per seed dengan CI bootstrap (analisis sensitivitas, tambahan).",
    "Subbab 3.11 dan 3.12 (Tabel 3.4: H1 M4 vs M1, H2 M6 vs M4, H3 M5 vs M4, H4 M6 vs M5, H5 M3 vs M1).",
    "Berkas prediksi tersimpan (tanpa memuat ulang checkpoint).",
    "runs/split_seed42/significance/{final_results_table, significance_test, effect_sizes_mae, qwk_ensemble_effect, "
    "qwk_per_seed_effect, tabel_panjang_efek}.csv.",
    "tanpa GPU, beberapa menit",
    "Arah: untuk MAE, selisih (A minus B) negatif berarti model A lebih baik. Untuk QWK, selisih positif berarti model A "
    "lebih baik. QWK ensemble: prediksi tiga seed dirata-rata lalu dibulatkan. QWK per seed: QWK dihitung per seed lalu "
    "dirata-rata. Selisih QWK antar fungsi loss dapat berbeda antara kedua cara agregasi. Bila Wilcoxon (setelah koreksi) "
    "dan CI bootstrap tidak sejalan, keduanya dilaporkan bersama.",
    "QWK tidak dapat didekomposisi per sampel sehingga tidak diuji dengan Wilcoxon. Data uji berpotensi masih mengandung noise."
)
code("""
out_sig = pipeline.step_analysis(ctx_main, SEEDS_MAIN)
display(out_sig["final"]); display(out_sig["sig"]); display(out_sig["eff"])
display(out_sig["qwk_ens"]); display(out_sig["qwk_per"])
""")
code("""
selfcheck.check_significance(ctx_main)
""")
code("""
bk = drive_io.export_backup_zip(config.DRIVE_ROOT, f"cadangan_bagian10_{RUN_TAG}.zip")
print(bk)
drive_io.download_if_colab(bk["zip_path"])
""")

# ======================================================================
# 11. UJI EMAS
# ======================================================================
section(
    11, "Subset uji emas",
    "Memeriksa ketahanan hasil H1 sampai H5 pada subset data uji yang dilabeli ulang oleh manusia. Sampel diambil stratified "
    "berdasarkan rating asli dari data uji split utama, dengan ukuran sesuai rumus Cochran (e = 10 persen).",
    "Subbab 3.10.",
    "runs/split_seed42/data/split_test.csv dan berkas prediksi tersimpan.",
    "annotations/gold_test_sample.csv (+ sidecar), annotations/gold_test_sample_annotator2.csv (+ sidecar), "
    "runs/split_seed42/gold/gold_test_{evaluation,effect_sizes,undetermined_summary,kappa}.csv.",
    "tanpa GPU; waktu anotasi bergantung pada penilai",
    "Opsi ND dikeluarkan dari metrik dan proporsinya dilaporkan terpisah. Hanya arah keunggulan dan CI bootstrap yang "
    "dilaporkan, tanpa Wilcoxon kedua. Label emas final adalah label penilai pertama. Penilai kedua dipakai untuk kappa "
    "berbobot kuadratik pada baris overlap.",
    "Rubrik operasional per kelas rating tersedia pada config_files/rubrik_anotasi_rating.md. Klaim 'rubrik disediakan' "
    "pada laporan hanya boleh dipertahankan bila penilai memang memakainya. Sampel dapat dibuat segera setelah bagian 5 "
    "dan diisi paralel dengan pelatihan. Komponen ini ditolak untuk split selain split utama."
)
md("### 11.1 Alur A (verifikasi hash) atau alur B (membuat sampel baru)")
code("""
if ALUR_ANOTASI_GOLD == "A":
    gold_test.verify_annotation(ctx_main)
elif ALUR_ANOTASI_GOLD == "B":
    gold_test.create_sample(ctx_main)
    gold_test.create_second_annotator()
else:
    raise ValueError("ALUR_ANOTASI_GOLD harus 'A' atau 'B'")
print(open(os.path.join(REPO_DIR, "config_files", "rubrik_anotasi_rating.md"), encoding="utf-8").read())
""")
md("### 11.2 Evaluasi (jalankan setelah kolom human_gold_rating terisi lengkap dan pelatihan selesai)")
code("""
gold_result, gold_effect = gold_test.evaluate_on_gold(ctx_main, SEEDS_MAIN)
display(gold_result); display(gold_effect)
""")
code("""
try:
    gold_kappa = gold_test.compute_gold_kappa(ctx_main)
except (FileNotFoundError, ValueError) as e:
    print("[PERINGATAN] Kappa uji emas belum dapat dihitung:", e)
""")
code("""
selfcheck.check_gold(ctx_main)
""")

# ======================================================================
# 12. K
# ======================================================================
section(
    12, "Sensitivitas jumlah fold K",
    "Memeriksa sensitivitas K = 3, 5, dan 10 pada proxy P4 sebagai pelengkap konfirmatif atas keputusan apriori K = 5, "
    "bukan pencarian nilai optimum. K = 5 berasal dari proses utama pada eksekusi ini (tidak dilatih dua kali).",
    "Subbab 3.5.",
    "runs/split_seed42/data/split_train.csv dan OOF P4 K = 5.",
    "runs/split_seed42/k_sensitivity/{k_sensitivity_table, k_sensitivity_qwk_pairs}.csv, berkas OOF dan fold K = 3 dan K = 10.",
    "sekitar 130 menit (K = 3 dan K = 10)",
    "Tabel melaporkan QWK, MAE, akurasi, off-by-one, ECE kumulatif rata-rata, metode filter terpilih, jumlah baris ter-flag "
    "per metode, dan indeks Jaccard himpunan ter-flag terhadap K = 5. TAMBAHAN di luar kalimat proposal: CI bootstrap "
    "berpasangan untuk selisih QWK antar K, supaya klaim 'selisih kecil dan tidak monoton' memiliki ukuran ketidakpastian.",
    "Komponen ini ditolak untuk split selain split utama."
)
code("""
ev_p4 = ablation.run_proxy_evaluation(ctx_main, df_train_main, 3, config.PROXY_CV_FOLDS, ctx_main.diagnostics_dir)
k_table, k_pairs = sensitivity_k.run_k_sensitivity(ctx_main, df_train_main, ev_p4)
display(k_table); display(k_pairs)
""")
code("""
selfcheck.check_k_sensitivity(ctx_main)
""")

# ======================================================================
# 13. SENSITIVITAS PARTISI
# ======================================================================
section(
    13, "Sensitivitas partisi data",
    "Menjawab permintaan dosen pembimbing agar kesimpulan tidak bergantung pada satu undian split. Selain split utama "
    "(seed partisi 42), dijalankan dua split acak tambahan (seed partisi 123 dan 2024) dengan rasio 70:10:20, stratifikasi "
    "dan pengelompokan yang sama. Total tiga skenario split.",
    "Subbab 3.4 dan 3.12.",
    "shared/processed/reviews_clean.csv.",
    "runs/split_seed123/ dan runs/split_seed2024/ (struktur seragam dengan split utama), serta "
    "shared/results/sensitivity_partisi/{tabel_panjang_semua_split, ringkasan_konsistensi, wilcoxon_berdampingan_per_split}.csv.",
    "per split tambahan: P4 lima fold sekitar 50 menit dan 18 sesi pelatihan sekitar 8 sampai 12 jam",
    "Lihat aturan konsistensi di bawah. Hasil Wilcoxon per split dilaporkan berdampingan dan tidak digabung.",
    "Ablasi P1 sampai P3, sensitivitas K, validasi manusia, dan uji emas hanya untuk split utama karena anotasi manual tidak "
    "dapat diulang per split. Kode menolak menjalankan komponen tersebut pada split lain."
)
md("""
### Dasar rancangan dan aturan konsistensi (ditetapkan SEBELUM menjalankan sel sensitivitas)

**Mengapa seed partisi, bukan rasio.** Keberatan yang dijawab adalah apakah kesimpulan stabil terhadap undian partisi.
Mengubah seed hanya mengubah komposisi sampel. Mengubah rasio mengubah ukuran data latih dan data uji sekaligus, sehingga
selisih hasil tidak dapat diatribusikan pada satu penyebab, dan metrik antar rasio tidak sebanding.

**Pipeline per split (identik).** Pembuatan split, proxy P4 lima fold dengan temperature scaling, Confident learning,
resolusi konflik, tiga varian data, enam skenario M1 sampai M6 dengan tiga seed bobot (18 sesi), uji Wilcoxon dengan koreksi
Holm-Bonferroni (lima hipotesis), effect size MAE dengan CI bootstrap, QWK ensemble dengan CI bootstrap, dan QWK per seed
dengan CI bootstrap. Opsi hemat (satu seed bobot per split, sakelar HEMAT_SPLIT_SENSITIVITAS) tersedia dan, bila dipakai,
wajib dinyatakan sebagai keterbatasan.

**Arah.** Untuk MAE, selisih (A minus B) negatif berarti model A lebih baik. Untuk QWK, selisih positif berarti model A lebih baik.

**Aturan konsistensi.** Suatu temuan dinyatakan KONSISTEN bila arahnya sama dan CI 95 persen bootstrap tidak melewati nol pada
sekurangnya dua dari tiga split. Ia dinyatakan TIDAK KONSISTEN bila arahnya berbalik atau CI melewati nol pada sekurangnya dua
split. Penerapan operasional: konsisten bila arah titik taksiran seragam pada ketiga split dan sekurangnya dua CI tidak melewati nol;
selain itu tidak konsisten (dengan tiga split, kedua kategori saling lengkap).

**Larangan.** Semua split dilaporkan seluruhnya, tidak ada pemilihan split berdasarkan hasil, dan p-value tidak digabung antar split.
Data uji tiap split berasal dari korpus yang sama dan saling tumpang tindih, sehingga ketiganya bukan replikasi independen.

**Batasan.** Validasi manusia dan uji emas hanya untuk split utama.
""")
code("""
SPLITS_TAMBAHAN = [s for s in config.SPLIT_SEEDS if s != config.MAIN_SPLIT_SEED]
ctx_sens = {s: config.RunPaths(s) for s in SPLITS_TAMBAHAN}
seeds_sens = {s: pipeline.weight_seeds_for(s, hemat=HEMAT_SPLIT_SENSITIVITAS) for s in SPLITS_TAMBAHAN}
print("Split tambahan:", SPLITS_TAMBAHAN, "| seed bobot:", seeds_sens)
if HEMAT_SPLIT_SENSITIVITAS:
    print("[PERINGATAN] OPSI HEMAT aktif. Wajib dinyatakan sebagai keterbatasan.")
""")
md("### 13.1 Pembuatan split tambahan")
code("""
df_train_sens = {}
for s, c in ctx_sens.items():
    tr, va, te = data_split.create_splits(c)
    df_train_sens[s] = tr
    pipeline.register_run_mode(c, seeds_sens[s], HEMAT_SPLIT_SENSITIVITAS)
    selfcheck.check_split(c)
""")
md("### 13.2 Proxy P4 lima fold, Confident learning, dan tiga varian data per split tambahan")
code("""
out_p4_sens = {}
for s, c in ctx_sens.items():
    out_p4_sens[s] = pipeline.step_p4(c, df_train_sens[s])
    display(pd.Series(out_p4_sens[s]["variant_stats"], name=f"split {s}"))
    selfcheck.check_p4(c)
""")
md("### 13.3 Pelatihan enam skenario per split tambahan")
for s in (123, 2024):
    code(f"""
status_{s} = pipeline.step_train(ctx_sens[{s}], seeds_sens[{s}], copy_ckpt_to_drive=False)
selfcheck.check_training(ctx_sens[{s}])
""")
md("### 13.4 Analisis per split tambahan")
code("""
out_sig_sens = {}
for s, c in ctx_sens.items():
    out_sig_sens[s] = pipeline.step_analysis(c, seeds_sens[s])
    print(f"Split {s}")
    display(out_sig_sens[s]["final"]); display(out_sig_sens[s]["sig"])
    selfcheck.check_significance(c)
""")
md("### 13.5 Ringkasan konsistensi (tanpa menggabungkan p-value)")
code("""
long_all, consistency_table, wilcoxon_wide = pipeline.consistency_summary()
display(consistency_table)
display(wilcoxon_wide)
display(long_all)
""")
code("""
selfcheck.check_partition_sensitivity()
""")
code("""
bk = drive_io.export_backup_zip(config.DRIVE_ROOT, f"cadangan_bagian13_{RUN_TAG}.zip")
print(bk)
drive_io.download_if_colab(bk["zip_path"])
""")

# ======================================================================
# 14. RINGKASAN AKHIR
# ======================================================================
section(
    14, "Ringkasan akhir",
    "Memetakan setiap berkas keluaran ke tabel atau subbab proposal dan bagian Bab IV yang akan memakainya, serta menampilkan "
    "tabel kunci dari berkas di Drive (bukan dari variabel sesi).",
    "Seluruh Bab III.",
    "Seluruh berkas keluaran.",
    "Tabel peta keluaran (ditampilkan).",
    "kurang dari 1 menit",
    "Kolom 'ada' harus bernilai ya untuk seluruh berkas yang relevan dengan run ini.",
    "Berkas preserved/scraping_summary.csv dan berkas anotasi berasal dari luar run ini."
)
code("""
output_map = summary.build_output_map()
display(output_map)
""")
code("""
for nama, path in [
    ("Tabel ablasi proxy (Tabel 3.1)", os.path.join(ctx_main.ablation_dir, "proxy_ablation_table.csv")),
    ("Hasil enam skenario (split utama)", os.path.join(ctx_main.significance_dir, "final_results_table.csv")),
    ("Uji signifikansi H1 sampai H5 (split utama)", os.path.join(ctx_main.significance_dir, "significance_test.csv")),
    ("Konsistensi lintas split", os.path.join(config.SENSITIVITY_SUMMARY_DIR, "ringkasan_konsistensi.csv")),
]:
    print("\\n" + nama)
    if drive_io.is_verified(path):
        display(pd.read_csv(path))
    else:
        print("[PERINGATAN] belum ada:", path)
""")

# ======================================================================
# 15. MANIFEST
# ======================================================================
section(
    15, "Manifest hasil dan unduhan cadangan",
    "Mencatat SHA-256 seluruh berkas hasil sebagai bukti integritas run, lalu mengunduh cadangan zip berisi seluruh CSV, JSON, "
    "dan prediksi kecil.",
    "Subbab 3.2 (prinsip penguncian) dan 3.13.",
    "Seluruh berkas di runs, shared, logs, annotations, dan preserved.",
    "shared/results/results_manifest.csv; berkas zip lokal.",
    "kurang dari 2 menit",
    "Manifest memuat jalur, ukuran, dan SHA-256 tiap berkas. Berkas manifest sendiri tidak ikut di dalamnya.",
    "Jalankan hanya setelah seluruh bagian selesai. Bila berkas hasil bertambah setelah manifest ditulis, manifest lama "
    "tidak ditimpa dan eksekusi berhenti dengan pesan; hapus manifest lama secara manual bila memang perlu dibuat ulang."
)
code("""
manifest_df, manifest_res = summary.build_results_manifest()
print(manifest_res)
display(manifest_df.head(30))
""")
code("""
bk = drive_io.export_backup_zip(config.DRIVE_ROOT, f"cadangan_final_{RUN_TAG}.zip")
print(bk)
drive_io.download_if_colab(bk["zip_path"])
print("[OK] Notebook selesai. Simpan berkas zip cadangan di luar Drive.")
""")


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "skripsi_final.ipynb")
    nb = new_notebook(cells=CELLS)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    }
    nbformat.validate(nb)
    nbformat.write(nb, out)
    print(f"[OK] Notebook ditulis: {out} ({len(CELLS)} sel)")


if __name__ == "__main__":
    sys.exit(main())
