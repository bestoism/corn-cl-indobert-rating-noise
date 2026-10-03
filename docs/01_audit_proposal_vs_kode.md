# Tugas 1. Audit Proposal terhadap Kode

Prinsip: proposal adalah spesifikasi. Kode disesuaikan. Bila sebuah pernyataan proposal tidak bisa atau
tidak seharusnya dipenuhi kode, kode tidak dipaksa dan diusulkan kalimat pengganti (lihat dokumen 03).
Temuan bernomor a sampai l berasal dari daftar peneliti. Temuan m sampai t ditemukan saat audit.

Catatan penomoran: daftar isi proposal menulis "3.1 Alur Penelitian", sedangkan judul pada isi bab adalah
"3.1 Jenis Penelitian". Dipakai judul pada isi bab. Seluruh rujukan subbab pada kode baru mengikuti 3.1 sampai 3.13.

| No | Pernyataan di proposal (subbab) | Kondisi kode lama | Tindakan | Alasan |
| --- | --- | --- | --- | --- |
| a | 3.5: sensitivitas K = 3, 5, 10 pada P4 | Tidak ada | Ubah kode: src/sensitivity_k.py. K = 5 diambil dari proses utama. CI bootstrap berpasangan untuk selisih QWK antar K dan indeks Jaccard terhadap K = 5 ditandai sebagai TAMBAHAN | Klaim "selisih kecil dan tidak monoton" memerlukan ukuran ketidakpastian |
| b | 3.6 butir 2 dan 2.1.8: verifikasi monotonisitas dan ECE | Tidak ada | Ubah kode: src/calibration.py (lihat l untuk definisi) | Proposal menjanjikan pemeriksaan ini |
| c | 3.6 butir 7: kualitas proxy per kelas rating asli | Hanya agregat | Ubah kode: calibration.per_class_quality (recall, precision, MAE per kelas), disimpan per proxy | Rujukan pemeriksaan confound |
| d | 3.8: distribusi jarak ordinal pada baris noise per kelas dan confound dengan recall atau precision proxy | Tidak ada | Ubah kode: src/diagnostics.py. Penanda confound bernilai YA bila kelas berada di dua teratas pada persentase jarak berat sekaligus di dua terbawah pada recall (atau precision) | Operasionalisasi kata "bertepatan" pada 3.8 |
| e | 3.2 dan Batasan Masalah: penguncian SHA-256 | Hash ditulis ke manifest hasil run itu sendiri, tidak pernah diverifikasi | Ubah kode: src/corpus.py membandingkan hash dan jumlah baris dengan konstanta di src/config.py | Manifest yang ditulis ulang tidak dapat mendeteksi perubahan korpus |
| f | 3.10: rubrik operasional anotasi; ketidaksepakatan diselesaikan lewat majority vote atau diskusi | Tidak ada berkas rubrik. Kode memakai label penilai pertama sebagai label emas, penilai kedua hanya untuk kappa | Rubrik: config_files/rubrik_anotasi_rating.md ditambahkan. Perilaku adjudikasi TIDAK diubah. Usulan kalimat proposal ada di dokumen 03 | Dua penilai pada baris overlap saja tidak memungkinkan majority vote bermakna |
| g | 3.4: tidak ada bukti angka overlap teks nol | Hanya simulasi split acak biasa (EDA) | Ubah kode: data_split.create_splits menghitung overlap teks dan review_id lintas train, val, test, menulisnya ke split_summary.csv, dan berhenti bila tidak nol | Bukti angka menggantikan klaim |
| h | Notebook lama menyebut 4 hipotesis (H1 sampai H4); proposal 5 hipotesis | Kode lama sudah 5 hipotesis, teks notebook (dan pesan cetak gold_test) masih menyebut H1 sampai H4 | Ubah teks: notebook baru dan pesan cetak memakai H1 sampai H5 | Konsistensi dengan Tabel 3.4 |
| i | 3.12: selisih QWK dengan CI bootstrap, agregasi ensemble | Hanya ensemble (rata-rata prediksi tiga seed dibulatkan) | Pertahankan ensemble sebagai definisi utama. TAMBAHAN: QWK rata-rata per seed dengan CI bootstrap berpasangan (indeks resampling sama untuk semua seed), berdampingan | Selisih QWK antar fungsi loss dapat berbeda antar cara agregasi |
| j | Tabel 3.3 dan parameter lain | Lihat tabel verifikasi di bawah | Sesuai; beberapa butir ditambahkan pada proposal | Lihat bawah |
| k | Bug tampilan "ECE top-label" sama dengan ECE rata-rata karena indeks posisi | Kode diagnostik kalibrasi tersebut tidak ada pada lampiran yang saya terima (tidak ada ECE pada kode lama) | Pencegahan: calibration_report mengembalikan dict bernama, nilai diambil berdasarkan nama. Uji sintetis memastikan ece_top_label berbeda dari ECE kumulatif rata-rata | Tidak dapat memperbaiki kode yang tidak ada; desain baru menutup kelas bug ini |
| l | Pemeriksaan monotonisitas versi sederhana bersifat tautologi | Tidak ada | Ubah kode: kumulatif langsung dari rantai CORN (setelah temperature scaling, sebelum konversi) disimpan sebagai q_chain per fold dan di OOF. Monotonisitas, ECE, dan selisih maksimum terhadap kumulatif turunan (akibat pembatasan 1e-8 dan normalisasi ulang) dilaporkan | Lihat catatan di bawah tabel |
| m | 3.3 butir 8 dan 9: voting mayoritas lalu duplikat murni dihapus | Duplikat murni (teks dan rating identik) dihapus pada praproses SEBELUM voting. Akibatnya setiap grup konflik berisi satu baris per rating, semua grup seri, dan seluruhnya dibuang. Keluaran lama konsisten: "minoritas: 0, seri: 263". Voting mayoritas tidak pernah memilih mayoritas | Perilaku TIDAK diubah (mengubah urutan akan mengubah data). Usulan kalimat proposal pada dokumen 03. Docstring kode menyatakan ini | Menyesuaikan proposal lebih jujur daripada mengubah metodologi diam-diam |
| n | 3.3 butir 4: kamus domain dari PRDECT-ID | Kode memuat slang_domain.csv bila ada. Keluaran lama hanya mencetak "Kamus slang dasar: 15006 entri", tanpa baris kamus domain: pada run lama entri domain tampaknya TIDAK terpakai | Kode baru mencetak PERINGATAN bila slang_domain.csv tidak ada. Peneliti memutuskan: sediakan berkas itu di lexicon, atau ubah kalimat proposal | PRDECT-ID adalah dataset e-commerce, jangkauan domainnya terbatas |
| o | 3.4 kalimat pertama paragraf kedua: validasi menjadi sumber sinyal untuk "pencarian parameter suhu T" | T pada proxy dicari pada validasi INTERNAL di dalam bagian latih tiap fold (sesuai 3.4 kalimat berikutnya dan 3.6). Model final M1 sampai M6 tidak dikalibrasi | Kode tidak diubah. Usulan kalimat pada dokumen 03 | Dua kalimat pada 3.4 saling bertentangan |
| p | 3.12: Wilcoxon dua arah, bootstrap | Kode memakai zero_method="zsplit", 2000 resampel, seed 42; tidak tertulis di proposal | Kode tidak diubah. Usulan penambahan kalimat | Reprodusibilitas |
| q | 3.7: N = baris ter-flag P4; n dibulatkan 100 | Kode lama menetapkan HUMAN_VALIDATION_N = 100 tanpa menghitung dari N | Ubah kode: n0, n koreksi populasi terbatas dicetak dari N ter-flag, lalu n = 100 dipakai sebagai angka perencanaan | Proposal menjelaskan perhitungan itu |
| r | 2.1 dan 3.5: P1 sampai P3 hanya pembanding ablasi | Kode lama mengekspor sampel validasi manusia dan varian data bersih untuk tiap proxy yang dijalankan | Ubah kode: clean.export_p4_outputs menolak proxy selain P4 dan K selain 5 | Aturan run murni dan proposal |
| s | 3.13: checkpoint dan auto-resume | Satu berkas progress JSON ditimpa tiap seed | Ubah kode: satu berkas unik per unit kerja, ditulis sekali | Gejala duplikasi di Drive |
| t | 3.10: label dummy | Kode lama memakai label dummy [1] saat memuat model untuk prediksi uji emas | Ubah kode: evaluasi uji emas bekerja dari prediksi tersimpan, tanpa label dummy | Instruksi tugas |

## Catatan tentang monotonisitas (temuan l)

Kumulatif rantai q_k = hasil kali sigmoid(z_j / T), j sampai k, adalah hasil kali bilangan pada [0, 1] dan tidak naik
secara konstruksi. Kumulatif turunan (jumlah ekor probabilitas kelas non-negatif) juga tidak naik secara konstruksi.
Pemeriksaan monotonisitas karena itu hanya uji kewarasan numerik (galat pembulatan float). Pada notebook, hasilnya
dilaporkan apa adanya dan tidak dikutip sebagai bukti empiris kualitas model. Bagian yang informatif adalah:
(1) selisih maksimum dan rata-rata antara q_chain dan kumulatif turunan akibat pembatasan minimal 1e-8 dan normalisasi
ulang, dan (2) ECE kumulatif.

Varian ECE yang dipakai: ECE biner per ambang k = 1..4 pada kejadian {y > k} dengan probabilitas q_k, 10 bin lebar sama
pada [0, 1], dirata-rata atas K-1 = 4 ambang, dihitung terhadap label teramati (yang dapat noisy). Rumus pada Subbab 2.1.8
(ECE dengan acc dan conf per bin, M bin) cocok dengan varian ini bila acc(B_m) dibaca sebagai proporsi kejadian {y > k}
dalam bin. Kalimat penyesuaian ada pada dokumen 03. ECE top-label tersedia sebagai TAMBAHAN.

## Verifikasi butir j: hiperparameter dan parameter terhadap kode

| Parameter (rujukan proposal) | Nilai proposal | Kode lama | Kode baru |
| --- | --- | --- | --- |
| Backbone (Tabel 3.3) | indobenchmark/indobert-base-p1 | sama | sama |
| Token maksimum | 128 | 128 | 128 |
| Batch | 16 | 16 | 16 |
| Learning rate | 2e-5 | 2e-5 | 2e-5 |
| Optimizer | AdamW, weight decay 0,01 | AdamW, weight_decay 0,01 | sama, nilai eksplisit di config |
| Warmup | linear 10 persen | 10 persen dari total langkah | sama |
| Epoch maksimum | 10 | 10 | 10 |
| Early stopping | patience 3, MAE validasi | sama | sama |
| Presisi | mixed precision FP16 | torch.cuda.amp | torch.amp (perilaku sama) |
| Seed bobot | 42, 123, 2024 | sama | sama; dipisah dari seed partisi |
| Kriteria checkpoint | QWK validasi, tie-breaker MAE | sama | sama |
| Dropout dan pooler_output (3.9) | 0,3; pooler_output | 0,3; pooler_output | sama |
| Proxy fine-tuned (3.9) | 3 epoch, tanpa scheduler, tanpa early stopping | 3 epoch, AdamW lr 2e-5 (weight decay bawaan torch 0,01), tanpa scheduler | sama |
| min_examples_per_class (3.6) | 20 | 20 | 20 |
| K dan skema fold (3.5, 3.6) | 5, shuffle True, seed 42 | sama (juga untuk P1 dan P2) | sama |
| Ambang severity (3.8) | jarak 2 atau lebih | 2 | 2 |
| Rumus Cochran (3.7, 3.10) | Z 1,96; p 0,5; e 10 persen; koreksi populasi terbatas | benar pada gold_test; tidak dihitung pada validasi manusia | keduanya dihitung dan dicetak |
| Alokasi validasi manusia (3.7) | minimal 15 per bin jarak 2, 3, 4; sisa ke jarak 1 | sama | sama (urutan random_state dipertahankan) |
| Ukuran uji emas (3.10) | n sekitar 90 sampai 100 | 93 untuk N = 2856 | rumus sama |
| Split (3.4) | 70:10:20, grouped, seed 42 | sama | sama; ditambah seed 123 dan 2024 |

Catatan: ketergantungan pada versi pandas. Rating mayoritas grup pada data_split memakai value_counts().idxmax(). Untuk
grup seri, hasilnya bergantung pada urutan pandas saat seri. Versi pandas dicatat pada log lingkungan. Kode tidak diubah
agar partisi tidak bergeser.
