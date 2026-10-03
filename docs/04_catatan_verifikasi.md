# Catatan Jujur: Apa yang Telah dan Belum Diverifikasi

## Telah dijalankan (di sandbox, Python 3.12, tanpa GPU, tanpa torch, tanpa akses HF Hub)

1. Pemeriksaan sintaks (ast.parse) seluruh berkas Python dan seluruh sel kode notebook (scripts/check_style.py).
2. Pencarian karakter em dash (U+2014) dan en dash (U+2013) di seluruh berkas teks proyek: jumlah nol.
3. Notebook divalidasi dengan nbformat.validate (valid).
4. tests/test_pure.py, 16 kelompok uji dengan data SINTETIS, seluruhnya lulus:
   - drive_io: tulis sekali, tolak menimpa isi berbeda, deteksi salinan ganda " (1)", audit Drive (mode awal dan berjalan),
     klasifikasi DIPERTAHANKAN, DIPERBOLEHKAN, TIDAK DIHARAPKAN, dry run pembersihan.
   - qwk_fast setara sklearn cohen_kappa_score (selisih di bawah 1e-12).
   - calibration: ECE nol pada probabilitas sempurna, deteksi pelanggaran monotonisitas, selisih rantai vs turunan,
     kunci dict bernama (ece_top_label berbeda dari ECE kumulatif), kualitas per kelas.
   - sampling: Cochran (n0 = 96,04; n = 93 untuk N = 2856; n = 97 untuk N besar), alokasi 15, 15, 15, 55, kasus kolam kurang.
   - consistency: arah MAE vs QWK, kasus konsisten, tidak konsisten (CI melewati nol; arah berbalik), split kurang.
   - clean.detect_noise dengan cleanlab 2.9.0 pada probabilitas sintetis (kedua metode filter, estimasi off_diagonal_calibrated).
   - preprocess: normalisasi slang di tepi tanda baca, emoji, dan pembuktian bahwa setelah duplikat murni dihapus semua grup
     konflik seri.
   - data_split: split berkelompok, nol overlap, seed partisi berbeda menghasilkan partisi berbeda.
   - quality_report dari master sintetis.
   - Orkestrasi P4, ablasi, sensitivitas K dengan OOF tiruan (monkeypatch proxy.get_oof): P1 sampai P3 tidak menghasilkan
     varian; export_p4_outputs menolak proxy selain P4 dan K selain 5; guard split non-utama.
   - Anotasi: alur B, tolak menimpa, hash sidecar cocok dan tidak cocok, agreement, kappa.
   - Signifikansi dari berkas prediksi tersimpan (QWK ensemble dan per seed dicocokkan dengan sklearn), uji emas
     (opsi ND dikeluarkan), sensitivitas partisi, opsi hemat dan penguncian mode.
   - Peta keluaran, manifest SHA-256, zip cadangan.

## BELUM diverifikasi (tidak dapat dijalankan di sini)

- Seluruh jalur yang memerlukan torch, transformers, GPU, dan unduhan IndoBERT: embedding beku P1 dan P2, fine-tuning
  P3 dan P4 (proxy._train_one_fold), temperature scaling LBFGS (proxy.fit_temperature), konversi rantai CORN dengan torch
  (proxy.corn_chain), pelatihan M1 sampai M6 (train.run_experiment), perhitungan tingkat pemotongan token (tokenizer).
  Logika yang dipindahkan dari kode lama sebisa mungkin disalin tanpa perubahan numerik, tetapi hasil eksekusi nyata belum diamati.
- Kompatibilitas dengan Python 3.13, torch 2.11, transformers 5.17, coral-pytorch 1.4.0 di Colab. Sandbox memakai Python 3.12
  dan pandas 3.0, sedangkan Colab dapat memakai pandas lain. Perilaku seri value_counts pada data_split bergantung pada versi pandas.
- torch.amp.GradScaler("cuda") dan torch.amp.autocast("cuda") menggantikan torch.cuda.amp.* yang deprecated. Perilakunya
  seharusnya identik, tetapi belum teramati di GPU.
- Perilaku mount Google Drive sebenarnya. Mekanisme penulisan sekali, baca ulang, deteksi duplikat, dan audit dirancang tahan
  terhadap gejala yang dilaporkan, tanpa mengandalkan dugaan penyebab, tetapi belum diuji pada Drive.
- Pemuatan lexicon nyata (slang_base.csv Kamus Alay), serta isi nyata master sebanyak 16740 baris.
- Eksekusi notebook secara utuh. Sel notebook hanya diperiksa sintaksnya dan alurnya dipetakan ke fungsi yang diuji terpisah.

## Asumsi yang diambil

1. Seed partisi 42 mereproduksi split lama bila praproses dan versi pandas sama. Tidak ada perbandingan dengan angka lama (sesuai instruksi).
2. K = 5 fold proxy memakai seed CV 42 pada semua split (seed CV tidak diubah mengikuti seed partisi).
3. Kata "split" pada "prediksi per sampel ... untuk setiap skenario, seed, dan split" ditafsirkan sebagai split partisi (42, 123, 2024).
   Prediksi disimpan untuk data uji saja.
4. Kriteria konsistensi dioperasionalkan sebagai: arah titik taksiran seragam pada ketiga split dan sekurangnya dua CI tidak
   melewati nol. Selain itu tidak konsisten. Dengan tiga split kedua kategori saling lengkap.
5. Opsi hemat hanya berlaku untuk split tambahan. Split utama selalu tiga seed bobot.
6. Eksekusi yang dilanjutkan dari fold tengah (proxy) memulai RNG dari seed 42 pada fold pertama yang dihitung. Hasil tidak
   identik bit per bit dengan eksekusi tanpa putus. Nondeterminisme GPU (mixed precision, cudnn.benchmark) pada dasarnya
   juga mencegah identitas bit.
7. Mode DEBUG pada kode lama dihapus (tidak ada di proposal dan berisiko mencemari hasil).
8. Kamus slang tidak diunduh otomatis. Folder lexicon dianggap masukan yang dipertahankan.
9. Berkas anotasi lama tidak memiliki sidecar hash. Alur A memerlukan sidecar yang dibuat dari daftar review_id yang berlaku.
10. ECE dihitung terhadap label teramati (noisy), bukan label bersih.
11. Pemeriksaan monotonisitas pada kumulatif rantai tetap tidak naik secara konstruksi (hasil kali bilangan pada [0, 1]).
    Diberi label uji kewarasan numerik, bukan bukti empiris.

## Tambahan di luar proposal (ditandai)

QWK rata-rata per seed (sensitivitas agregasi); ECE top-label; CI bootstrap berpasangan dan indeks Jaccard untuk sensitivitas K;
penanda confound otomatis pada diagnostik jarak ordinal; pemeriksaan audit Drive, sidecar hash anotasi, manifest hasil, dan zip
cadangan (infrastruktur, bukan metode).

## Hal yang perlu keputusan peneliti

- Kamus domain PRDECT-ID (temuan n): sediakan lexicon/slang_domain.csv atau ubah kalimat proposal.
- Sidecar untuk anotasi lama bila ingin memakai alur A.
- Apakah opsi hemat dipakai untuk split tambahan (keterbatasan wajib dinyatakan).
- Penyesuaian kalimat proposal pada dokumen 03.
