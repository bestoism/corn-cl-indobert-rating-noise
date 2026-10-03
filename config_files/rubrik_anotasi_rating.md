# Rubrik Operasional Anotasi Rating (Subbab 3.10)

Rubrik ini disediakan sebagai acuan konsistensi penilai uji emas. Penilai membaca teks hasil
praproses (case folding, normalisasi slang, dan normalisasi emoji), bukan teks asli hasil scraping.
Rating asli dan prediksi model apa pun tidak diperlihatkan.

Catatan penggunaan: klaim "rubrik operasional disediakan" pada laporan hanya boleh dipertahankan
bila penilai memang memakai rubrik ini saat menganotasi. Bila tidak dipakai, nyatakan apa adanya.

Prinsip umum: nilai rating yang paling wajar diberikan pengguna berdasarkan ISI TEKS saja.
Jangan menebak transaksi, akun, atau konteks yang tidak tertulis.

| Rating | Pola isi teks yang sejalan |
| --- | --- |
| 1 | Kekecewaan berat atau kemarahan. Aplikasi tidak berfungsi, penipuan, dana atau pesanan hilang, layanan tidak dapat dipakai, tanpa sisi positif. |
| 2 | Dominan negatif. Masalah nyata yang mengganggu (sering error, lambat, bantuan tidak membantu), dengan sedikit pengakuan atas sisi baik atau tanpa kemarahan ekstrem. |
| 3 | Campuran atau netral. Sisi baik dan buruk disebut seimbang, atau komentar datar tanpa nada jelas ("biasa saja", "lumayan tapi ada kekurangan"). |
| 4 | Dominan positif dengan catatan kecil. Puas secara umum, ada saran atau keluhan ringan yang tidak menggugurkan kepuasan. |
| 5 | Puas penuh atau pujian tanpa keluhan berarti ("bagus", "sangat membantu", "mantap"). |

Opsi tambahan ND ("tidak dapat ditentukan dari teks"): gunakan hanya bila rating yang wajar bergantung
pada informasi eksternal yang tidak tertulis, misalnya status transaksi yang masih berjalan, atau teks
yang tidak bermakna sama sekali. Teks pendek yang tetap bermakna ("bagus", "jelek") tidak termasuk ND.

Aturan keputusan untuk teks ambigu:
1. Bila nada keseluruhan jelas, pilih rating sesuai nada, bukan sesuai kata tunggal.
2. Bila sarkasme tampak jelas dari konteks kalimat, nilai maksud sebenarnya.
3. Bila ragu antara dua rating bertetangga, pilih yang lebih dekat dengan nada keseluruhan, dan catat
   keraguan pada kolom catatan bila tersedia.

Keterangan format isian: kolom human_gold_rating diisi angka 1 sampai 5, atau ND.
