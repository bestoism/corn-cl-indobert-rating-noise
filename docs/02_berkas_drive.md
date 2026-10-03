# Tugas 4. Daftar Berkas Drive

Analisis dilakukan terhadap kode. Berkas disebut DIPERTAHANKAN hanya bila tidak dapat dibuat ulang dari
data raw dan lexicon oleh notebook.

## DIPERTAHANKAN

| Berkas | Alasan |
| --- | --- |
| data/raw/all_reviews_master.csv (dan berkas per aplikasi bila ada) | Instruksi dosen. Scraping tidak boleh diulang (hasil berbeda, penguncian rusak) |
| lexicon/slang_base.csv, lexicon/slang_domain.csv (bila ada) | Instruksi dosen. Notebook tidak mengunduh kamus |
| preserved/scraping_summary.csv | Jumlah ulasan diperiksa dan dibuang oleh filter bahasa per (aplikasi, rating) hanya tercatat saat scraping. Wajib dilaporkan di Bab III |
| preserved/dataset_manifest_original.json | Catatan waktu penguncian korpus asli (locked_at). Hash dan jumlah baris sudah ada di kode, tetapi waktu penguncian tidak dapat direkonstruksi |
| annotations/human_validation_sample.csv dan .sidecar.json | Input manusia. Dipakai kembali hanya bila hash daftar review_id ter-flag P4 cocok (alur A) |
| annotations/human_validation_sample_annotator2.csv dan .sidecar.json | Input manusia penilai kedua |
| annotations/gold_test_sample.csv dan .sidecar.json | Input manusia uji emas. Hash dasar adalah daftar review_id data uji split utama, yang deterministik dari raw, lexicon, dan seed partisi |
| annotations/gold_test_sample_annotator2.csv dan .sidecar.json | Input manusia penilai kedua uji emas |

Opsional (bukan keharusan teknis): scrape_state dan data/raw/_checkpoints. Dipertahankan hanya bila peneliti ingin
dokumentasi proses scraping. Notebook tidak memakainya dan audit mengklasifikasikan scrape_state sebagai DIPERBOLEHKAN.

## DAPAT DIBUAT ULANG (aman dihapus sebelum run murni)

| Berkas lama | Mengapa dapat dibuat ulang |
| --- | --- |
| date_distribution_per_app_rating.csv, text_length_summary.csv, duplicate_review_texts.csv | Dihitung dari master (src/quality_report.py) |
| Seluruh results/eda (tabel dan gambar) | Dihitung dari master |
| dataset_manifest.json hasil run lama | Hash dan jumlah baris kini diverifikasi terhadap konstanta di kode. Manifest hasil run diganti shared/results/corpus_verification.json |
| Penanda bahasa terlalu pendek (n_too_short_for_detect) | Dapat dihitung dari kolom detected_lang pada master (06_penanda_bahasa_terlalu_pendek_dari_master.csv). Yang TIDAK dapat dibuat ulang hanyalah n_checked dan n_lang_dropped |
| data/processed (reviews_clean, split, preprocessing_summary, token_truncation_summary) | Deterministik dari raw dan lexicon |
| proxy_cache (embedding, OOF) | Dihitung ulang oleh run |
| cleaned (varian data per proxy) | Dibuat ulang; hanya P4 yang menghasilkan varian |
| models_ckpt | Checkpoint tidak dipakai lagi oleh analisis (prediksi tersimpan per unit) |
| results (ablasi, final_results_table, significance_test, effect size, gold_test_evaluation, dan lain-lain) | Dibangun ulang dari berkas unik per unit |
| logs/experiment_progress.json | Diganti berkas unik per unit |
| human_validation_internal, human_validation_result, human_validation_kappa | Diturunkan dari anotasi dan run |
| gold_test_internal_reference.csv | Hanya rating asli data uji; tersedia dari split |

## DIHAPUS

Semua berkas pada daftar "dapat dibuat ulang" di atas, ditambah:
- Folder skenario ganda dengan nama seperti "M4_Baseline_CORN (1)" dan salinan berakhiran " (1)" atau berawalan "Salinan".
- Berkas progress JSON bersalinan banyak.
- Folder lama SKRIPSI_CORN bila tidak dipakai lagi (pindahkan dahulu berkas pada daftar DIPERTAHANKAN ke folder baru).

Gunakan sel pembersihan pada notebook bagian 1.4. Sel itu dry run sebagai bawaan dan tidak pernah menghapus
raw, lexicon, preserved, dan annotations.

## Penataan di folder SKRIPSI_CORN_CL_FINAL sebelum run

```
SKRIPSI_CORN_CL_FINAL/
  data/raw/all_reviews_master.csv
  lexicon/slang_base.csv
  lexicon/slang_domain.csv        (bila ada)
  preserved/scraping_summary.csv
  preserved/dataset_manifest_original.json
  annotations/                    (berkas anotasi dan sidecar bila memakai alur A)
```

Catatan alur A: berkas anotasi lama tidak memiliki sidecar hash (sidecar baru diperkenalkan di versi ini). Untuk memakai
anotasi lama, buat sidecar dari daftar review_id yang berlaku pada saat sampel dibuat, atau jalankan alur B. Hash
validasi manusia bergantung pada himpunan baris ter-flag P4. Pada run murni himpunan itu dapat berbeda dari run lama
karena nondeterminisme GPU, sehingga anotasi lama kemungkinan harus dibuat ulang. Hash uji emas bergantung pada data
uji split utama dan seharusnya cocok bila praproses dan seed partisi tidak berubah.

## Paragraf untuk dosen

"Yang dimaksud run murni dalam penelitian ini adalah bahwa seluruh hasil komputasi, yaitu praproses, pembagian data,
probabilitas proxy classifier, deteksi label noise, varian data bersih, model, prediksi, dan uji statistik, dibangun
ulang dari awal pada satu eksekusi dari data raw dan kamus slang. Notebook tidak memuat cache, checkpoint, atau hasil
dari eksekusi sebelumnya. Berkas yang dipertahankan bukan cache karena tidak dihasilkan oleh komputasi pipeline. Data
raw dan kamus adalah masukan penelitian. Ringkasan penyaringan bahasa hanya dapat berasal dari proses scraping yang
tidak boleh diulang karena hasilnya akan berbeda. Berkas anotasi adalah penilaian manusia yang tidak dapat direproduksi
oleh komputasi, dan hanya dipakai kembali setelah hash daftar review_id pada berkas tersebut terbukti identik dengan
hasil eksekusi ini."
