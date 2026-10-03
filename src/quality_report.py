"""
quality_report.py: laporan kualitas data dan EDA dari all_reviews_master.csv (Subbab 3.2).

Seluruh keluaran dapat dibuat ulang dari master tanpa scraping. Hal yang TIDAK dapat dibuat
ulang dari master adalah jumlah ulasan yang diperiksa dan dibuang oleh filter bahasa per
(aplikasi, rating), karena hanya tercatat saat scraping. Angka itu ada pada
preserved/scraping_summary.csv (berkas yang dipertahankan).

Modul ini hanya mendeskripsikan data. Tidak ada keputusan desain yang diubah oleh hasilnya.
"""

import io
import os

import numpy as np
import pandas as pd

from src import config, drive_io

TEXT_COL, RATING_COL, APP_COL, DATE_COL, ID_COL = "review_text", "rating", "source_app", "date", "review_id"
VERY_SHORT_MAX_WORDS = 3


def _out(name):
    return os.path.join(config.QUALITY_DIR, name)


def _save_fig(fig, name):
    import matplotlib
    matplotlib.use("Agg")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return drive_io.write_once_bytes(_out(name), buf.getvalue())


def load_master(path=None):
    df = pd.read_csv(path or config.MASTER_FILE)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[RATING_COL] = df[RATING_COL].astype(int)
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    return df


def rating_distribution(df):
    overall = df[RATING_COL].value_counts().sort_index()
    tab = pd.DataFrame({"rating": overall.index, "n": overall.values,
                        "persen": (overall.values / len(df) * 100).round(2)})
    per_app = df.groupby([APP_COL, RATING_COL]).size().unstack(fill_value=0).reset_index()
    drive_io.write_once_csv(tab, _out("01_distribusi_rating_keseluruhan.csv"))
    drive_io.write_once_csv(per_app, _out("01_distribusi_rating_per_aplikasi.csv"))
    return tab, per_app


def date_distribution(df):
    g = df.groupby([APP_COL, RATING_COL])[DATE_COL]
    tab = g.agg(n="count", tanggal_min="min", tanggal_median="median", tanggal_max="max").reset_index()
    tab["rentang_hari"] = (pd.to_datetime(tab["tanggal_max"]) - pd.to_datetime(tab["tanggal_min"])).dt.days
    for c in ["tanggal_min", "tanggal_median", "tanggal_max"]:
        tab[c] = pd.to_datetime(tab[c]).dt.strftime("%Y-%m-%d")
    drive_io.write_once_csv(tab, _out("02_distribusi_tanggal_per_app_rating.csv"))

    sev = []
    for app, sub in tab.groupby(APP_COL):
        smax, smin = sub["rentang_hari"].max(), sub["rentang_hari"].min()
        sev.append({"aplikasi": app,
                    "rating_rentang_terpanjang": int(sub.loc[sub["rentang_hari"].idxmax(), RATING_COL]),
                    "rentang_terpanjang_hari": int(smax),
                    "rating_rentang_terpendek": int(sub.loc[sub["rentang_hari"].idxmin(), RATING_COL]),
                    "rentang_terpendek_hari": int(smin),
                    "selisih_hari": int(smax - smin)})
    sev = pd.DataFrame(sev)
    drive_io.write_once_csv(sev, _out("02_keparahan_confound_temporal.csv"))
    return tab, sev


def text_length(df):
    d = df.copy()
    d["n_kata"] = d[TEXT_COL].str.split().apply(len)
    d["n_karakter"] = d[TEXT_COL].str.len()
    desc = d[["n_kata", "n_karakter"]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(2)
    per_rating = d.groupby(RATING_COL)["n_kata"].agg(["count", "mean", "median", "std"]).round(2)
    short = d["n_kata"] <= VERY_SHORT_MAX_WORDS
    short_per_rating = (d[short].groupby(RATING_COL).size() / d.groupby(RATING_COL).size() * 100).round(2)
    drive_io.write_once_csv(desc.reset_index(), _out("03_ringkasan_panjang_teks.csv"))
    drive_io.write_once_csv(per_rating.reset_index(), _out("03_panjang_teks_per_rating.csv"))
    drive_io.write_once_csv(short_per_rating.reset_index(name="persen_teks_sangat_pendek"),
                            _out("03_teks_sangat_pendek_per_rating.csv"))
    return d, desc, per_rating, short_per_rating, float(short.mean() * 100)


def exact_duplicates(df, top_n=200):
    counts = df[TEXT_COL].value_counts()
    dup = counts[counts > 1]
    n_rows = int(dup.sum())
    stats = {"n_grup_duplikat": int(len(dup)), "n_baris_duplikat": n_rows,
             "pct_baris_duplikat": round(n_rows / len(df) * 100, 2)}
    drive_io.write_once_csv(dup.head(top_n).rename_axis("teks").reset_index(name="jumlah"),
                            _out("04_teks_duplikat_persis_top200.csv"))
    return stats


def conflicting_duplicates(df):
    g = df.groupby(TEXT_COL)[RATING_COL]
    conflict_texts = g.nunique()
    conflict_texts = conflict_texts[conflict_texts > 1].index
    rows = df[df[TEXT_COL].isin(conflict_texts)]
    stats = {"n_grup_konflik": int(len(conflict_texts)), "n_baris_konflik": int(len(rows)),
             "pct_baris_konflik": round(len(rows) / len(df) * 100, 2)}
    if len(conflict_texts):
        spread = rows.groupby(TEXT_COL)[RATING_COL].agg(lambda s: int(s.max() - s.min())).value_counts().sort_index()
        drive_io.write_once_csv(spread.rename_axis("selisih_rating_maks_dalam_grup").reset_index(name="jumlah_grup"),
                                _out("05_konflik_sebaran_selisih_rating.csv"))
        contoh = (rows.sort_values(TEXT_COL).groupby(TEXT_COL).head(5)
                  .loc[:, [TEXT_COL, RATING_COL, APP_COL]].head(40))
        drive_io.write_once_csv(contoh, _out("05_contoh_teks_konflik_rating.csv"))
    return stats


def language_marker_summary(df):
    """Dapat dibuat ulang dari kolom detected_lang pada master (hanya baris yang lolos filter)."""
    if "detected_lang" not in df.columns:
        return None
    tab = df.groupby([APP_COL, RATING_COL])["detected_lang"].apply(
        lambda s: int((s == "undetected_short").sum())).reset_index(name="n_terlalu_pendek_untuk_deteksi")
    drive_io.write_once_csv(tab, _out("06_penanda_bahasa_terlalu_pendek_dari_master.csv"))
    return tab


def simulated_random_split_overlap(df, test_size=0.20, seed=42):
    """
    Simulasi split acak biasa per baris (BUKAN split yang dipakai) untuk menunjukkan besar
    risiko kebocoran teks bila tidak memakai grouped split (Subbab 3.4).
    """
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[RATING_COL], random_state=seed)
    leaked = test_df[TEXT_COL].isin(set(train_df[TEXT_COL]))
    stats = {"n_train_simulasi": len(train_df), "n_test_simulasi": len(test_df),
             "n_test_teks_juga_di_train": int(leaked.sum()),
             "pct_test_teks_juga_di_train": round(float(leaked.mean() * 100), 2)}
    drive_io.write_once_csv(pd.DataFrame([stats]), _out("07_simulasi_overlap_split_acak_biasa.csv"))
    return stats


def figures(df, per_app, date_tab, d_len):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    per_app.set_index(APP_COL).T.plot(kind="bar", ax=ax)
    ax.set_xlabel("Rating"); ax.set_ylabel("Jumlah ulasan")
    ax.set_title("Distribusi rating per aplikasi (korpus berkuota)")
    _save_fig(fig, "fig01_distribusi_rating_per_aplikasi.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for app, sub in date_tab.groupby(APP_COL):
        ax.plot(sub[RATING_COL], sub["rentang_hari"], marker="o", label=app)
    ax.set_xlabel("Rating"); ax.set_ylabel("Rentang tanggal (hari)")
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_title("Rentang waktu pengumpulan per rating"); ax.legend(title="Aplikasi")
    _save_fig(fig, "fig02_rentang_tanggal_per_rating.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d_len["n_kata"].clip(upper=100), bins=50)
    ax.set_xlabel("Jumlah kata (dipotong di 100)"); ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi panjang ulasan")
    _save_fig(fig, "fig03_histogram_panjang_teks.png")


def run_quality_report(path=None, make_figures=True):
    df = load_master(path)
    print(f"Baris master: {len(df)} | rentang tanggal: {df[DATE_COL].min().date()} s.d. {df[DATE_COL].max().date()}")
    tab_rating, per_app = rating_distribution(df)
    date_tab, sev = date_distribution(df)
    d_len, desc, per_rating, short_per_rating, pct_short = text_length(df)
    dup_stats = exact_duplicates(df)
    conf_stats = conflicting_duplicates(df)
    language_marker_summary(df)
    overlap_stats = simulated_random_split_overlap(df)
    if make_figures:
        figures(df, per_app, date_tab, d_len)

    summary = {"n_baris": len(df), "pct_teks_sangat_pendek": round(pct_short, 2),
               **dup_stats, **conf_stats, **overlap_stats}
    drive_io.write_once_csv(pd.DataFrame([summary]), _out("00_ringkasan_kualitas_data.csv"))
    return {"rating": tab_rating, "per_app": per_app, "date": date_tab, "severity": sev,
            "length_desc": desc, "length_per_rating": per_rating,
            "short_per_rating": short_per_rating, "summary": summary}
