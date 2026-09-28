"""
data_split.py -- Pembagian data latih (70%), validasi (10%), dan uji (20%)
sesuai Subbab III.E: stratified BERDASARKAN RATING, grouped BERDASARKAN
TEKS ULASAN (seluruh baris berteks identik jatuh di sisi split yang sama),
random_state tetap (42). Dijalankan SEKALI setelah praproses dan SEBELUM
Confident Learning.

CATATAN METODOLOGIS: stratifikasi murni per-baris tidak mungkin dipertahankan
bersamaan dengan syarat grouped split, sehingga stratifikasi dilakukan pada
level GRUP (rating mayoritas tiap teks unik) -- pendekatan pragmatis yang
perlu disebutkan eksplisit di Bab III/IV kalau ditanya penguji.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

TEXT_COL = "cleaned_text"
RATING_COL = "rating"
RANDOM_STATE = 42


def _group_table(df):
    g = df.groupby(TEXT_COL)[RATING_COL].agg(lambda s: s.value_counts().idxmax())
    return g.reset_index().rename(columns={RATING_COL: "majority_rating"})


def create_splits(input_path=None, seed=RANDOM_STATE):
    input_path = input_path or config.CLEAN_TEXT_FILE
    df = pd.read_csv(input_path)

    train_prop = 1 - config.VAL_SIZE - config.TEST_SIZE
    group_tab = _group_table(df)
    print(f"📦 Jumlah teks unik: {len(group_tab)} (dari {len(df)} baris)")

    train_groups, temp_groups = train_test_split(
        group_tab, train_size=train_prop, random_state=seed,
        stratify=group_tab["majority_rating"],
    )
    val_relative = config.VAL_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    val_groups, test_groups = train_test_split(
        temp_groups, train_size=val_relative, random_state=seed,
        stratify=temp_groups["majority_rating"],
    )

    train_texts = set(train_groups[TEXT_COL])
    val_texts = set(val_groups[TEXT_COL])
    test_texts = set(test_groups[TEXT_COL])

    df_train = df[df[TEXT_COL].isin(train_texts)].copy()
    df_val = df[df[TEXT_COL].isin(val_texts)].copy()
    df_test = df[df[TEXT_COL].isin(test_texts)].copy()

    total = len(df)
    print(f"✅ Train: {len(df_train)} baris ({len(df_train)/total*100:.1f}%)")
    print(f"✅ Val  : {len(df_val)} baris ({len(df_val)/total*100:.1f}%)")
    print(f"✅ Test : {len(df_test)} baris ({len(df_test)/total*100:.1f}%)")

    dup_sizes = df.groupby(TEXT_COL).size()
    n_dup_rows = int(dup_sizes[dup_sizes > 1].sum())
    pct_dup_rows = n_dup_rows / total * 100
    print(f"\n📎 Baris yang terlibat pengelompokan teks duplikat: "
          f"{n_dup_rows} ({pct_dup_rows:.2f}%) -- dijamin satu sisi split (Subbab III.E).")

    for name, d in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        dist = d[RATING_COL].value_counts(normalize=True).sort_index().round(3)
        print(f"   Distribusi rating {name}: {dist.to_dict()}")

    df_train.to_csv(config.TRAIN_RAW_FILE, index=False)
    df_val.to_csv(config.VAL_FILE, index=False)
    df_test.to_csv(config.TEST_FILE, index=False)

    pd.DataFrame([{
        "n_total": total, "n_train": len(df_train), "n_val": len(df_val),
        "n_test": len(df_test), "pct_rows_in_dup_groups": round(pct_dup_rows, 2),
    }]).to_csv(os.path.join(config.RESULTS_DIR, "split_summary.csv"), index=False)

    return df_train, df_val, df_test