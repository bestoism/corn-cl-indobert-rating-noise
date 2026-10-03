# corn-cl-indobert-rating-noise

Integrasi Confident learning dan CORN pada IndoBERT untuk deteksi label noise pada prediksi rating ulasan
Google Play Store (skripsi Ryan Besto Saragih, NIM 23051204205, pembimbing Anita Qoiriah, S.Kom., M.Kom.).

## Struktur

```
src/        modul inti (config, drive_io, corpus, quality_report, preprocess, data_split, data, models,
            calibration, proxy, ablation, clean, diagnostics, sampling, annotation, human_validation, gold_test,
            train, significance, consistency, sensitivity_k, pipeline, selfcheck, summary, env_log, metrics)
scripts/    build_notebook.py, check_style.py, scrape_google_play.py (dokumentasi, menolak berjalan)
notebooks/  skripsi_final.ipynb
config_files/ rubrik_anotasi_rating.md
tests/      test_pure.py (uji sintetis tanpa GPU)
docs/       audit, berkas Drive, usulan revisi proposal, catatan verifikasi
```

## Menjalankan

1. Susun folder Drive sesuai docs/02_berkas_drive.md.
2. Buka notebooks/skripsi_final.ipynb di Colab (GPU T4) dan jalankan dari atas ke bawah.
3. Uji lokal tanpa GPU: `python tests/test_pure.py` dan `python scripts/check_style.py .`
4. Membangun ulang notebook: `python scripts/build_notebook.py`
