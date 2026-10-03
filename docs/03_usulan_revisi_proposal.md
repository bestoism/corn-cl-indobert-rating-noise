# Tugas 5. Usulan Revisi Kalimat Proposal

Draf berikut ditulis dalam gaya proposal. Tanda [ ] menandai angka yang diisi setelah hasil run tersedia.
Seluruh usulan mengikuti perilaku kode, bukan sebaliknya.

## Subbab 1.5 (Batasan Masalah), butir tambahan

"Sensitivitas terhadap undian partisi diperiksa pada dua split acak tambahan (seed partisi 123 dan 2024) di samping
split utama. Validasi manusia, subset uji emas, ablasi proxy P1 sampai P3, dan sensitivitas nilai K hanya dilakukan pada
split utama karena penilaian manual tidak dapat diulang pada setiap split. Data uji ketiga split berasal dari korpus yang
sama dan saling tumpang tindih, sehingga ketiganya bukan replikasi independen."

## Subbab 3.4 (Pembagian Data), paragraf tambahan

"Untuk memeriksa apakah kesimpulan stabil terhadap undian partisi, selain split utama (seed partisi 42) dijalankan dua
split acak dengan rasio 70:10:20, stratifikasi, dan pengelompokan teks yang sama, yaitu seed partisi 123 dan 2024.
Seed partisi dipilih sebagai sumbu sensitivitas, bukan rasio, karena perubahan seed hanya mengubah komposisi sampel,
sedangkan perubahan rasio mengubah ukuran data latih dan data uji sekaligus sehingga selisih hasil tidak dapat
diatribusikan pada satu penyebab dan metrik antar rasio tidak sebanding. Pada setiap split berlaku seluruh tahap yang
sama: proxy P4 lima fold dengan temperature scaling, Confident learning, resolusi konflik, tiga varian data, enam
skenario M1 sampai M6 dengan tiga seed bobot, dan seluruh pengujian pada Subbab 3.12. Seed partisi terpisah dari seed
bobot model. Seluruh split dilaporkan dan tidak ada pemilihan split berdasarkan hasil. Bukti bahwa tidak ada teks yang
sama dan tidak ada review_id yang sama lintas data latih, validasi, dan uji dilaporkan pada tabel ringkasan split."

Perbaikan kalimat pertama paragraf kedua Subbab 3.4 (temuan o): ganti "... dan pencarian parameter suhu T pada
kalibrasi." dengan "... dan pemilihan checkpoint terbaik berdasarkan QWK. Parameter suhu T pada proxy classifier dicari
pada validasi internal di dalam bagian latih tiap fold, sebagaimana dijelaskan berikut."

## Subbab 3.3 (Praproses Teks), perbaikan butir resolusi konflik (temuan m)

"Duplikat murni (teks dan rating identik) dihapus pada tahap praproses. Akibatnya, setiap grup teks identik dengan
rating berbeda memuat tepat satu baris per rating dan seluruh grup tidak memiliki rating mayoritas. Resolusi melalui
voting mayoritas yang dijalankan setelah deteksi label noise karena itu membuang seluruh baris pada grup konflik, dan
jumlah baris berating minoritas yang dibuang bernilai nol. Jumlah baris yang dibuang dilaporkan apa adanya."

Perbaikan butir kamus domain (temuan n), pilih salah satu: (1) bila berkas slang_domain.csv disediakan, pertahankan
kalimat saat ini dan laporkan jumlah entrinya; (2) bila tidak, ganti dengan "Penelitian ini memakai Kamus Alay [21].
Kamus domain dari PRDECT-ID [23] tidak diterapkan karena jangkauannya terbatas pada domain e-commerce."

## Subbab 2.1.8 dan 3.6 (definisi ECE dan monotonisitas)

Tambahan pada 2.1.8 setelah rumus ECE: "Pada probabilitas kumulatif, ECE dihitung sebagai ECE biner per ambang k pada
kejadian {y > k} dengan probabilitas q_k, memakai sepuluh bin berlebar sama pada [0, 1], lalu dirata-rata atas K-1
ambang. Pada kejadian biner tersebut, acc(B_m) adalah proporsi kejadian {y > k} dalam bin dan conf(B_m) adalah rata-rata
q_k dalam bin. ECE dihitung terhadap label teramati."

Ganti kalimat pemeriksaan pada 3.6 butir 2: "Kalibrasi diverifikasi post-hoc melalui tiga hal. Pertama, monotonisitas
probabilitas kumulatif langsung dari rantai CORN (setelah temperature scaling dan sebelum konversi ke probabilitas
kelas), yang secara konstruksi tidak naik sehingga berfungsi sebagai uji kewarasan numerik. Kedua, selisih maksimum
antara kumulatif dari rantai dan kumulatif yang diturunkan dari probabilitas kelas, akibat pembatasan minimal 1e-8 dan
normalisasi ulang. Ketiga, Expected Calibration Error kumulatif sebagaimana didefinisikan pada Subbab 2.1.8."

## Subbab 3.5 (sensitivitas K), tambahan

"Selisih QWK proxy antar nilai K dilaporkan bersama interval kepercayaan bootstrap berpasangan 95 persen berdasarkan
probabilitas out-of-sample yang tersimpan, serta indeks Jaccard himpunan baris ter-flag terhadap K = 5, agar klaim
bahwa selisih kecil dan tidak monoton memiliki ukuran ketidakpastian."

## Subbab 3.10 (Subset Uji Emas)

Ganti kalimat "Ketidaksepakatan diselesaikan melalui majority vote atau diskusi." dengan: "Label subset uji emas adalah
label penilai pertama. Penilai kedua menilai sebagian sampel yang overlap penuh semata-mata untuk menghitung
quadratic-weighted kappa sebagai ukuran reliabilitas, dan tidak dipakai untuk mengubah label emas."

Ganti butir (3) protokol: "Rubrik operasional singkat per kelas rating (lampiran [ ]) disediakan sebagai acuan
konsistensi dan dipakai oleh penilai." Kalimat ini hanya boleh dipertahankan bila penilai memang memakai rubrik; bila
tidak, nyatakan apa adanya.

## Subbab 3.12 (Pengujian Signifikansi Statistik)

Tambahan penjelasan teknis: "Uji Wilcoxon memakai zero_method zsplit karena banyak sampel memiliki selisih error nol
pada rating diskret. Interval kepercayaan bootstrap memakai 2.000 resampel dengan seed 42."

Penjelasan QWK: "Selisih QWK dilaporkan dengan dua cara agregasi yang berdampingan. Definisi utama memakai ensemble tiga
seed, yaitu prediksi dirata-rata lalu dibulatkan ke kelas terdekat. Sebagai analisis sensitivitas, QWK dihitung per seed
lalu dirata-rata, dengan interval kepercayaan bootstrap berpasangan yang memakai indeks resampling sama untuk seluruh
seed. Dua cara ini dilaporkan karena selisih QWK antar fungsi loss dapat berbeda di antara keduanya."

Aturan konsistensi lintas split: "Suatu temuan dinyatakan konsisten bila arahnya sama dan interval kepercayaan bootstrap
95 persen tidak melewati nol pada sekurangnya dua dari tiga split. Temuan dinyatakan tidak konsisten bila arahnya
berbalik atau interval kepercayaan melewati nol pada sekurangnya dua split. Arah dihitung sedemikian rupa sehingga untuk
MAE selisih negatif berarti model pertama lebih baik, sedangkan untuk QWK selisih positif berarti model pertama lebih
baik. Hasil Wilcoxon dilaporkan per split berdampingan dan p-value tidak digabung antar split."

## Subbab 3.13 (Estimasi Waktu), pembaruan

Tambahkan: "Sensitivitas partisi menambah dua split, masing-masing satu proxy P4 lima fold (sekitar 50 menit) dan 18 sesi
pelatihan (sekitar 8 sampai 12 jam). Penambahan total sekitar 18 sampai 26 jam GPU untuk dua split, sehingga total
estimasi murni komputasi menjadi sekitar 30 sampai 43 jam GPU di luar overhead antrean dan pemutusan sesi Colab."
Perhitungan: 12 sampai 17 jam (estimasi awal) ditambah 2 x (0,8 + 8 sampai 12) jam, yaitu kira-kira 17,6 sampai 25,6 jam,
dibulatkan 18 sampai 26 jam. Angka ini perkiraan perencanaan. Bila opsi hemat (satu seed bobot per split tambahan)
dipakai, tiap split tambahan memerlukan sekitar 3 sampai 4 jam pelatihan dan keterbatasannya wajib dinyatakan.
