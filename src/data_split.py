"""
data_split.py: pembagian data 70:10:20 (Subbab 3.4).

Pembagian berkelompok berdasarkan teks (grouped split) dengan stratifikasi pada tingkat
grup teks unik menurut rating mayoritas grup. random_state adalah seed partisi
(42 untuk split utama, 123 dan 2024 untuk sensitivitas). Seed partisi terpisah dari
seed bobot model (42, 123, 2024).

CATATAN REPRODUSIBILITAS: rating mayoritas grup diambil dengan value_counts().idxmax().
Untuk grup seri, hasilnya bergantung pada urutan pandas saat seri. Versi pandas dicatat
pada log lingkungan.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config, drive_io

TEXT_COL = "cleaned_text"
RATING_COL = "rating"


def _group_table(df):
    g = df.groupby(TEXT_COL)[RATING_COL].agg(lambda s: s.value_counts().idxmax())
    return g.reset_index().rename(columns={RATING_COL: "majority_rating"})


def make_splits(df, split_seed):
    """Murni (tanpa penulisan). Mengembalikan df_train, df_val, df_test."""
    train_prop = 1 - config.VAL_SIZE - config.TEST_SIZE
    group_tab = _group_table(df)

    train_groups, temp_groups = train_test_split(
        group_tab, train_size=train_prop, random_state=split_seed,
        stratify=group_tab["majority_rating"])
    val_relative = config.VAL_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    val_groups, test_groups = train_test_split(
        temp_groups, train_size=val_relative, random_state=split_seed,
        stratify=temp_groups["majority_rating"])

    train_texts = set(train_groups[TEXT_COL])
    val_texts = set(val_groups[TEXT_COL])
    test_texts = set(test_groups[TEXT_COL])
    df_train = df[df[TEXT_COL].isin(train_texts)].copy()
    df_val = df[df[TEXT_COL].isin(val_texts)].copy()
    df_test = df[df[TEXT_COL].isin(test_texts)].copy()
    return df_train, df_val, df_test, len(group_tab)


def overlap_evidence(df_train, df_val, df_test):
    """Bukti nol overlap teks dan review_id lintas partisi (Subbab 3.4)."""
    def inter(a, b, col):
        return len(set(a[col]) & set(b[col]))

    return {
        "overlap_teks_train_val": inter(df_train, df_val, TEXT_COL),
        "overlap_teks_train_test": inter(df_train, df_test, TEXT_COL),
        "overlap_teks_val_test": inter(df_val, df_test, TEXT_COL),
        "overlap_review_id_train_val": inter(df_train, df_val, "review_id"),
        "overlap_review_id_train_test": inter(df_train, df_test, "review_id"),
        "overlap_review_id_val_test": inter(df_val, df_test, "review_id"),
    }


def create_splits(ctx, df_clean=None):
    """
    Membuat tiga partisi untuk split 'ctx', menulis masing-masing satu berkas, dan menulis
    ringkasan beserta bukti nol overlap. Bila ketiga berkas sudah terverifikasi, dimuat saja.
    """
    paths = {n: ctx.split_file(n) for n in ("train", "val", "test")}
    summary_path = os.path.join(ctx.data_dir, "split_summary.csv")

    if all(drive_io.is_verified(p) for p in paths.values()) and drive_io.is_verified(summary_path):
        print(f"[OK] Split seed {ctx.split_seed} sudah ada, dimuat dari berkas.")
        return tuple(pd.read_csv(paths[n]) for n in ("train", "val", "test"))

    if df_clean is None:
        df_clean = drive_io.read_csv_verified(config.CLEAN_TEXT_FILE)
    df_train, df_val, df_test, n_groups = make_splits(df_clean, ctx.split_seed)

    ev = overlap_evidence(df_train, df_val, df_test)
    if any(v != 0 for v in ev.values()):
        raise RuntimeError(f"[BERHENTI] Terdapat overlap lintas partisi: {ev}")

    total = len(df_clean)
    dup_sizes = df_clean.groupby(TEXT_COL).size()
    n_dup_rows = int(dup_sizes[dup_sizes > 1].sum())

    summary = {
        "split_seed": ctx.split_seed,
        "n_total": total, "n_teks_unik": n_groups,
        "n_train": len(df_train), "n_val": len(df_val), "n_test": len(df_test),
        "pct_train": round(len(df_train) / total * 100, 2),
        "pct_val": round(len(df_val) / total * 100, 2),
        "pct_test": round(len(df_test) / total * 100, 2),
        "n_baris_grup_teks_duplikat": n_dup_rows,
        "pct_baris_grup_teks_duplikat": round(n_dup_rows / total * 100, 2),
        **ev,
    }
    for name, d in (("train", df_train), ("val", df_val), ("test", df_test)):
        dist = d[RATING_COL].value_counts(normalize=True).sort_index()
        for r in range(1, 6):
            summary[f"prop_rating{r}_{name}"] = round(float(dist.get(r, 0.0)), 4)

    print(f"Split seed {ctx.split_seed}: train={len(df_train)}, val={len(df_val)}, test={len(df_test)} "
          f"(dari {total} baris, {n_groups} teks unik)")
    print(f"Baris dalam grup teks duplikat: {n_dup_rows} ({summary['pct_baris_grup_teks_duplikat']}%)")
    print("Overlap lintas partisi (harus nol):", ev)

    drive_io.write_once_csv(df_train, paths["train"])
    drive_io.write_once_csv(df_val, paths["val"])
    drive_io.write_once_csv(df_test, paths["test"])
    drive_io.write_once_csv(pd.DataFrame([summary]), summary_path)
    return df_train, df_val, df_test
