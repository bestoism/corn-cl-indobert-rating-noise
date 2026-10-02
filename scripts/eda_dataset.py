"""
EDA dan penguncian dataset (Exploratory Data Analysis).

Dijalankan SEKALI di atas korpus mentah hasil scraping, SETELAH scraping
selesai dan SEBELUM praproses. Tujuannya dua:

  1. Mengunci korpus sebagai sumber kebenaran tunggal (Batasan Masalah butir 6):
     dedup berbasis review_id, lalu catat jumlah baris + hash SHA-256 ke
     dataset_manifest.json. Setelah manifest ditulis, file master tidak
     boleh diubah lagi.

  2. Menghasilkan seluruh angka deskripsi data yang dijanjikan proposal:
       - Subbab III.B butir 1-4 (filter bahasa, distribusi tanggal per
         app-rating, distribusi panjang teks, proporsi teks duplikat persis)
       - Subbab III.D butir 7 (teks identik dengan rating berbeda)
       - Subbab III.D butir 9 (flag teks sangat pendek, <= 3 kata)
       - Subbab III.E (overlap teks lintas split, untuk memutuskan apakah
         perlu grouped split berbasis teks)

Semua output berupa CSV + PNG di RESULTS_DIR/eda/, siap disalin ke Bab IV.

CATATAN PENTING: skrip ini HANYA mendeskripsikan data. Tidak ada keputusan
desain (ambang severity, pemilihan proxy, hipotesis) yang boleh diubah
berdasarkan hasilnya -- keputusan itu sudah dikunci di muka lewat argumen
literatur. EDA di sini untuk melaporkan karakteristik korpus, bukan untuk
memilih konfigurasi yang hasilnya paling enak.

Cara pakai:
    python scripts/eda_dataset.py
atau di Colab:
    from scripts.eda_dataset import main
    main()
"""

import os
import json
import hashlib
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------
# Path: pakai src.config kalau ada, kalau tidak pakai default lokal
# ----------------------------------------------------------
try:
    from src import config
    RAW_DATA_FILE = config.RAW_DATA_FILE
    RESULTS_DIR = config.RESULTS_DIR
except Exception:
    RAW_DATA_FILE = "data/raw/all_reviews_master.csv"
    RESULTS_DIR = "results"

EDA_DIR = os.path.join(RESULTS_DIR, "eda")
MANIFEST_FILE = os.path.join(RESULTS_DIR, "dataset_manifest.json")

# Harus sama persis dengan Subbab III.E
SPLIT_TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42

# Subbab III.D butir 9
VERY_SHORT_MAX_WORDS = 3

TEXT_COL = "review_text"
RATING_COL = "rating"
APP_COL = "source_app"
DATE_COL = "date"
ID_COL = "review_id"


def _ensure_dirs():
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _save_table(df, name, index=False):
    path = os.path.join(EDA_DIR, f"{name}.csv")
    df.to_csv(path, index=index)
    print(f"   -> {path}")
    return path


def _save_fig(fig, name):
    path = os.path.join(EDA_DIR, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {path}")
    return path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _header(title):
    print("\n" + "=" * 68)
    print(f" {title} ")
    print("=" * 68)


# ==========================================================
# 0. MUAT + KUNCI KORPUS
# ==========================================================
def load_and_lock(path=RAW_DATA_FILE):
    _header("0. MUAT DAN KUNCI KORPUS MENTAH")

    df = pd.read_csv(path)
    n_raw = len(df)

    # Dedup review_id (Batasan Masalah butir 6). Idealnya scraper sudah
    # melakukannya; ini jaring pengaman sekaligus angka yang dilaporkan.
    df = df.drop_duplicates(subset=[ID_COL], keep="first")
    n_after_id_dedup = len(df)

    # Baris tanpa teks tidak bisa dipakai untuk tugas berbasis teks.
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.strip() != ""]
    df = df[df[TEXT_COL].str.lower() != "nan"]
    n_after_empty = len(df)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[RATING_COL] = df[RATING_COL].astype(int)
    df = df.reset_index(drop=True)

    print(f"Baris mentah dibaca            : {n_raw}")
    print(f"Setelah dedup {ID_COL:<16}: {n_after_id_dedup} "
          f"(dibuang {n_raw - n_after_id_dedup})")
    print(f"Setelah buang teks kosong      : {n_after_empty} "
          f"(dibuang {n_after_id_dedup - n_after_empty})")
    print(f"Rentang tanggal                : {df[DATE_COL].min().date()} "
          f"s.d. {df[DATE_COL].max().date()}")

    return df, {
        "n_raw": int(n_raw),
        "n_after_review_id_dedup": int(n_after_id_dedup),
        "n_after_empty_text_removal": int(n_after_empty),
    }


# ==========================================================
# 1. DISTRIBUSI RATING (keseluruhan + per aplikasi)
# ==========================================================
def report_rating_distribution(df):
    _header("1. DISTRIBUSI RATING")

    overall = df[RATING_COL].value_counts().sort_index()
    overall_pct = (overall / len(df) * 100).round(2)
    tab_overall = pd.DataFrame({"n": overall, "persen": overall_pct})
    tab_overall.index.name = "rating"
    print(tab_overall.to_string())
    _save_table(tab_overall.reset_index(), "01_distribusi_rating_keseluruhan")

    per_app = df.groupby([APP_COL, RATING_COL]).size().unstack(fill_value=0)
    print("\nPer aplikasi:")
    print(per_app.to_string())
    _save_table(per_app.reset_index(), "01_distribusi_rating_per_aplikasi")

    fig, ax = plt.subplots(figsize=(7, 4))
    per_app.T.plot(kind="bar", ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Jumlah ulasan")
    ax.set_title("Distribusi rating per aplikasi (korpus berkuota)")
    ax.legend(title="Aplikasi")
    _save_fig(fig, "01_distribusi_rating_per_aplikasi")

    print("\nCatatan Bab IV: distribusi ini hasil kuota per rating, BUKAN")
    print("distribusi alami Play Store. Sebutkan eksplisit agar angka MAE/QWK")
    print("absolut tidak salah dibaca sebagai performa pada distribusi lapangan.")
    return tab_overall, per_app


# ==========================================================
# 2. DISTRIBUSI TANGGAL PER (APP, RATING) -- confound temporal
#    Subbab III.B butir 2
# ==========================================================
def report_date_distribution(df):
    _header("2. DISTRIBUSI TANGGAL PER (APLIKASI, RATING) -- CONFOUND TEMPORAL")

    g = df.groupby([APP_COL, RATING_COL])[DATE_COL]
    tab = g.agg(
        n="count",
        tanggal_min="min",
        tanggal_median="median",
        tanggal_max="max",
    ).reset_index()
    tab["rentang_hari"] = (
        pd.to_datetime(tab["tanggal_max"]) - pd.to_datetime(tab["tanggal_min"])
    ).dt.days

    for c in ["tanggal_min", "tanggal_median", "tanggal_max"]:
        tab[c] = pd.to_datetime(tab[c]).dt.strftime("%Y-%m-%d")

    print(tab.to_string(index=False))
    _save_table(tab, "02_distribusi_tanggal_per_app_rating")

    # Ukuran keparahan confound: selisih rentang hari antara rating dengan
    # rentang terpanjang dan terpendek, per aplikasi.
    print("\nKeparahan confound temporal per aplikasi "
          "(selisih rentang hari terpanjang vs terpendek antar rating):")
    for app, sub in tab.groupby(APP_COL):
        span_max = sub["rentang_hari"].max()
        span_min = sub["rentang_hari"].min()
        r_max = int(sub.loc[sub["rentang_hari"].idxmax(), RATING_COL])
        r_min = int(sub.loc[sub["rentang_hari"].idxmin(), RATING_COL])
        print(f"  {app:<12}: rating {r_max} mundur {span_max} hari, "
              f"rating {r_min} hanya {span_min} hari "
              f"(selisih {span_max - span_min} hari)")

    fig, ax = plt.subplots(figsize=(7, 4))
    for app, sub in tab.groupby(APP_COL):
        ax.plot(sub[RATING_COL], sub["rentang_hari"], marker="o", label=app)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Rentang tanggal (hari)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_title("Rentang waktu pengumpulan per rating")
    ax.legend(title="Aplikasi")
    _save_fig(fig, "02_rentang_tanggal_per_rating")

    print("\nHipotesis yang dicek di sini: karena Sort.NEWEST digabung")
    print("filter_score_with, rating yang lebih jarang (2 dan 3) harus ditelusuri")
    print("lebih jauh ke belakang untuk memenuhi kuota, sehingga rentangnya lebih")
    print("panjang dan median tanggalnya lebih tua. Kalau pola itu terlihat,")
    print("laporkan sebagai confound temporal di Bab IV (tidak dikoreksi via")
    print("resampling, sesuai Batasan Masalah butir 7).")
    return tab


# ==========================================================
# 3. PANJANG TEKS -- Subbab III.B butir 3 + III.D butir 9
# ==========================================================
def report_text_length(df):
    _header("3. DISTRIBUSI PANJANG TEKS ULASAN")

    df = df.copy()
    df["n_kata"] = df[TEXT_COL].str.split().apply(len)
    df["n_karakter"] = df[TEXT_COL].str.len()

    desc = df[["n_kata", "n_karakter"]].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    ).round(2)
    print(desc.to_string())
    _save_table(desc.reset_index(), "03_ringkasan_panjang_teks")

    per_rating = df.groupby(RATING_COL)["n_kata"].agg(
        ["count", "mean", "median", "std"]
    ).round(2)
    print("\nPanjang teks (kata) per rating:")
    print(per_rating.to_string())
    _save_table(per_rating.reset_index(), "03_panjang_teks_per_rating")

    # Flag teks sangat pendek (penanda, BUKAN filter -- III.D butir 9)
    short_mask = df["n_kata"] <= VERY_SHORT_MAX_WORDS
    pct_short = short_mask.mean() * 100
    short_per_rating = (
        df[short_mask].groupby(RATING_COL).size()
        / df.groupby(RATING_COL).size() * 100
    ).round(2)
    print(f"\nUlasan sangat pendek (<= {VERY_SHORT_MAX_WORDS} kata): "
          f"{short_mask.sum()} baris ({pct_short:.2f}%)")
    print("Persentase per rating:")
    print(short_per_rating.to_string())
    _save_table(short_per_rating.reset_index(name="persen_teks_sangat_pendek"),
                "03_teks_sangat_pendek_per_rating")

    # Berapa persen yang melewati MAX_LEN=128 token (aproksimasi kasar via kata)
    pct_over_100_words = (df["n_kata"] > 100).mean() * 100
    print(f"\nUlasan > 100 kata (perkiraan kasar risiko truncation MAX_LEN=128): "
          f"{pct_over_100_words:.2f}%")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["n_kata"].clip(upper=100), bins=50)
    ax.set_xlabel("Jumlah kata (dipotong di 100 untuk keterbacaan)")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi panjang ulasan")
    _save_fig(fig, "03_histogram_panjang_teks")

    print("\nCatatan Bab IV: ulasan pendek TIDAK dibuang. Angka ini dipakai agar")
    print("deteksi noise pada ulasan pendek tidak salah ditafsirkan sebagai")
    print("ketidaksesuaian rating-teks, padahal informasinya memang minim.")
    return df


# ==========================================================
# 4. DUPLIKAT TEKS PERSIS -- Subbab III.B butir 4
# ==========================================================
def report_exact_duplicates(df, top_n=20):
    _header("4. TEKS DUPLIKAT PERSIS (indikasi template / spam)")

    counts = df[TEXT_COL].value_counts()
    dup = counts[counts > 1]
    n_dup_rows = int(dup.sum())
    n_dup_groups = int(len(dup))
    pct = n_dup_rows / len(df) * 100

    print(f"Grup teks yang muncul >1 kali : {n_dup_groups}")
    print(f"Baris yang terlibat           : {n_dup_rows} ({pct:.2f}%)")
    if not dup.empty:
        print(f"\n{min(top_n, len(dup))} teks paling sering berulang:")
        preview = dup.head(top_n).rename_axis("teks").reset_index(name="jumlah")
        preview["teks"] = preview["teks"].str.slice(0, 70)
        print(preview.to_string(index=False))

    _save_table(dup.head(200).rename_axis("teks").reset_index(name="jumlah"),
                "04_teks_duplikat_persis_top200")

    print("\nKeterbatasan: ini HANYA duplikat persis sama. Near-duplicate")
    print("(beda satu typo atau spasi) tidak tertangkap -- sebutkan di Bab IV")
    print("kalau angka ini dipakai sebagai bukti keberadaan template/spam.")
    return {"n_dup_groups": n_dup_groups, "n_dup_rows": n_dup_rows,
            "pct_dup_rows": round(pct, 2)}


# ==========================================================
# 5. TEKS IDENTIK DENGAN RATING BERBEDA -- Subbab III.D butir 7
# ==========================================================
def report_conflicting_duplicates(df):
    _header("5. TEKS IDENTIK DENGAN RATING BERBEDA (ambiguitas natural)")

    g = df.groupby(TEXT_COL)[RATING_COL]
    n_unique_rating = g.nunique()
    conflict_texts = n_unique_rating[n_unique_rating > 1].index

    conflict_rows = df[df[TEXT_COL].isin(conflict_texts)]
    n_groups = len(conflict_texts)
    n_rows = len(conflict_rows)
    pct = n_rows / len(df) * 100

    print(f"Grup teks dengan rating berbeda : {n_groups}")
    print(f"Baris yang terlibat             : {n_rows} ({pct:.2f}%)")

    if n_groups:
        # Sebaran selisih rating maksimum dalam tiap grup konflik
        spread = (
            conflict_rows.groupby(TEXT_COL)[RATING_COL]
            .agg(lambda s: int(s.max() - s.min()))
            .value_counts().sort_index()
        )
        spread.index.name = "selisih_rating_maks_dalam_grup"
        print("\nSebaran selisih rating maksimum per grup:")
        print(spread.to_string())
        _save_table(spread.reset_index(name="jumlah_grup"),
                    "05_konflik_sebaran_selisih_rating")

        # Berapa grup yang akan gugur total karena seri (III.D butir 8)
        def is_tie(s):
            vc = s.value_counts()
            return len(vc) > 1 and vc.iloc[0] == vc.iloc[1]

        ties = conflict_rows.groupby(TEXT_COL)[RATING_COL].apply(is_tie)
        n_ties = int(ties.sum())
        rows_in_ties = len(conflict_rows[conflict_rows[TEXT_COL].isin(
            ties[ties].index)])
        print(f"\nGrup yang seri (tidak ada rating mayoritas): {n_ties} grup, "
              f"{rows_in_ties} baris")
        print("-> Grup seri dibuang seluruhnya oleh resolusi voting mayoritas")
        print("   (III.D butir 8). Angka ini memperkirakan susut datanya.")

        contoh = (
            conflict_rows.sort_values(TEXT_COL)
            .groupby(TEXT_COL).head(5)
            .loc[:, [TEXT_COL, RATING_COL, APP_COL]]
            .head(40)
        )
        _save_table(contoh, "05_contoh_teks_konflik_rating")

    print("\nCatatan Bab IV: angka ini adalah bukti kuantitatif awal adanya")
    print("ambiguitas natural pada data, SEBELUM deteksi noise formal dijalankan.")
    return {"n_conflict_groups": int(n_groups), "n_conflict_rows": int(n_rows),
            "pct_conflict_rows": round(pct, 2)}


# ==========================================================
# 6. OVERLAP TEKS LINTAS SPLIT -- Subbab III.E
#    Menentukan apakah perlu pindah ke grouped split berbasis teks
# ==========================================================
def report_split_overlap(df):
    _header("6. OVERLAP TEKS LINTAS SPLIT LATIH-UJI (risiko kebocoran)")

    train_df, test_df = train_test_split(
        df, test_size=SPLIT_TEST_SIZE,
        stratify=df[RATING_COL], random_state=SPLIT_RANDOM_STATE,
    )

    train_texts = set(train_df[TEXT_COL])
    leaked_mask = test_df[TEXT_COL].isin(train_texts)
    n_leaked = int(leaked_mask.sum())
    pct_leaked = n_leaked / len(test_df) * 100

    print(f"Simulasi split stratified {int((1-SPLIT_TEST_SIZE)*100)}:"
          f"{int(SPLIT_TEST_SIZE*100)} (random_state={SPLIT_RANDOM_STATE})")
    print(f"  Baris latih : {len(train_df)}")
    print(f"  Baris uji   : {len(test_df)}")
    print(f"  Baris uji yang teksnya JUGA muncul di data latih: "
          f"{n_leaked} ({pct_leaked:.2f}%)")

    # Pecah: berapa di antaranya yang rating-nya juga sama vs berbeda
    if n_leaked:
        pair = train_df[[TEXT_COL, RATING_COL]].drop_duplicates()
        merged = test_df[leaked_mask].merge(
            pair, on=TEXT_COL, suffixes=("_uji", "_latih")
        )
        same = int((merged[f"{RATING_COL}_uji"] == merged[f"{RATING_COL}_latih"]).sum())
        print(f"    - dengan rating identik  : {same}")
        print(f"    - dengan rating berbeda  : {len(merged) - same}")

    print("\nAturan keputusan:")
    print("  <1 %  -> pertahankan split acak, laporkan angkanya, tambahkan")
    print("           pemeriksaan ketahanan pada subset uji tanpa kembaran.")
    print("  1-3 % -> masih bisa dipertahankan, tapi pemeriksaan ketahanan wajib.")
    print("  >3 %  -> pindah ke grouped split berbasis hash teks, agar semua")
    print("           baris berteks identik jatuh ke sisi split yang sama.")

    _save_table(pd.DataFrame([{
        "n_train": len(train_df), "n_test": len(test_df),
        "n_test_leaked": n_leaked, "pct_test_leaked": round(pct_leaked, 2),
    }]), "06_overlap_teks_lintas_split")

    return {"n_test_leaked": n_leaked, "pct_test_leaked": round(pct_leaked, 2)}


# ==========================================================
# 7. MANIFEST -- kunci korpus
# ==========================================================
def write_manifest(df, path, load_stats, extra):
    _header("7. PENGUNCIAN KORPUS (MANIFEST)")

    manifest = {
        "locked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "master_file": os.path.abspath(path),
        "sha256": _sha256(path),
        "n_rows_final": int(len(df)),
        "date_min": str(df[DATE_COL].min().date()),
        "date_max": str(df[DATE_COL].max().date()),
        "apps": sorted(df[APP_COL].unique().tolist()),
        "rating_counts": {str(k): int(v) for k, v in
                          df[RATING_COL].value_counts().sort_index().items()},
        "load_stats": load_stats,
        "eda_stats": extra,
        "split_config": {"test_size": SPLIT_TEST_SIZE,
                         "random_state": SPLIT_RANDOM_STATE},
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n-> {MANIFEST_FILE}")
    print("\nMULAI SEKARANG file master TIDAK BOLEH DIUBAH LAGI.")
    print("Commit manifest ini ke git bersama hash-nya. Kalau di kemudian hari")
    print("hash file master tidak cocok dengan manifest, berarti korpus berubah")
    print("dan seluruh hasil eksperimen harus dijalankan ulang.")
    return manifest


def verify_manifest(path=RAW_DATA_FILE, manifest_path=MANIFEST_FILE):
    """Verifikasi SHA-256 + jumlah baris korpus terhadap manifest (Batasan Masalah butir 6)."""
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    current = _sha256(path)
    n_rows = len(pd.read_csv(path))
    ok_hash = current == m["sha256"]
    ok_rows = n_rows == m["n_rows_final"]
    print(f"SHA-256 manifest : {m['sha256']}")
    print(f"SHA-256 saat ini : {current}  -> {'COCOK ✅' if ok_hash else 'TIDAK COCOK ❌'}")
    print(f"Jumlah baris     : {n_rows} (manifest {m['n_rows_final']}) -> {'COCOK ✅' if ok_rows else 'TIDAK COCOK ❌'}")
    if not (ok_hash and ok_rows):
        raise RuntimeError("Korpus berbeda dari manifest -- seluruh eksperimen harus dijalankan ulang.")
    return True


# ==========================================================
# MAIN
# ==========================================================
def main(path=RAW_DATA_FILE):
    _ensure_dirs()
    print("=" * 68)
    print(" EDA DAN PENGUNCIAN DATASET ")
    print("=" * 68)

    df, load_stats = load_and_lock(path)

    report_rating_distribution(df)
    report_date_distribution(df)
    report_text_length(df)
    dup_stats = report_exact_duplicates(df)
    conflict_stats = report_conflicting_duplicates(df)
    overlap_stats = report_split_overlap(df)

    extra = {}
    extra.update(dup_stats)
    extra.update(conflict_stats)
    extra.update(overlap_stats)
    write_manifest(df, path, load_stats, extra)

    _header("SELESAI")
    print(f"Semua tabel dan gambar tersimpan di: {EDA_DIR}")
    print("\nYang langsung bisa masuk proposal:")
    print("  III.B butir 1 -> angka filter bahasa (dari scraping_summary.csv)")
    print("  III.B butir 2 -> 02_distribusi_tanggal_per_app_rating.csv")
    print("  III.B butir 3 -> 03_ringkasan_panjang_teks.csv")
    print("  III.B butir 4 -> 04_teks_duplikat_persis_top200.csv")
    print("  III.D butir 7 -> 05_konflik_sebaran_selisih_rating.csv")
    print("  III.E         -> 06_overlap_teks_lintas_split.csv")


if __name__ == "__main__":
    main()