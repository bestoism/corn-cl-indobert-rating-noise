"""
data_split.py -- Pembagian 70:10:20 (§3.4): stratified berdasarkan RATING, grouped
berdasarkan TEKS (semua baris berteks identik jatuh ke sisi split yang sama).
Stratifikasi dilakukan pada level grup teks unik (rating mayoritas grup).
Parameter seed menentukan partisi; seed 42 = split utama.
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


def create_splits(input_path=None, seed=RANDOM_STATE, overwrite=False):
    paths = (config.TRAIN_RAW_FILE, config.VAL_FILE, config.TEST_FILE)
    if not overwrite and all(os.path.exists(p) for p in paths):
        raise FileExistsError(
            f"Split untuk seed {seed} sudah ada ({config.TRAIN_RAW_FILE}). "
            f"Tidak ditimpa. Pakai overwrite=True hanya jika benar-benar disengaja."
        )

    input_path = input_path or config.CLEAN_TEXT_FILE
    df = pd.read_csv(input_path)

    train_prop = 1 - config.VAL_SIZE - config.TEST_SIZE
    group_tab = _group_table(df)
    print(f"📦 Jumlah teks unik: {len(group_tab)} (dari {len(df)} baris) | seed split = {seed}")

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

    # Bukti tidak ada kebocoran teks lintas split (angka ini masuk Bab III)
    overlap_tv = len(train_texts & val_texts)
    overlap_tt = len(train_texts & test_texts)
    overlap_vt = len(val_texts & test_texts)
    assert overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0, \
        f"Ada teks yang bocor lintas split! train∩val={overlap_tv}, train∩test={overlap_tt}, val∩test={overlap_vt}"
    print("✅ Verifikasi: overlap teks train∩val = train∩test = val∩test = 0")

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
        "split_seed": seed, "n_total": total, "n_train": len(df_train), "n_val": len(df_val),
        "n_test": len(df_test), "pct_rows_in_dup_groups": round(pct_dup_rows, 2),
        "overlap_train_val": overlap_tv, "overlap_train_test": overlap_tt, "overlap_val_test": overlap_vt,
    }]).to_csv(os.path.join(config.RESULTS_DIR, "split_summary.csv"), index=False)

    return df_train, df_val, df_test