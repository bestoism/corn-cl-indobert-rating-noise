import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src import config

VALID_VERDICTS = {"noise", "not_noise", "ambiguous"}


def _load_validated_sample():
    """Baca file anotator (BUTA), lalu join dengan referensi internal untuk
    mendapatkan kembali rating_diff (dibutuhkan untuk breakdown per-bin)."""
    if not os.path.exists(config.HUMAN_VALIDATION_FILE):
        print(f"⚠️ File belum ada: {config.HUMAN_VALIDATION_FILE}")
        print("   Jalankan clean.py dulu untuk men-generate file sample-nya.")
        return None

    df = None
    tried = []
    for sep in [";", ","]:
        try:
            candidate = pd.read_csv(config.HUMAN_VALIDATION_FILE, sep=sep)
            tried.append(f"'{sep}' -> kolom: {candidate.columns.tolist()}")
            if "human_verdict" in candidate.columns:
                df = candidate
                print(f"   (file dibaca dengan delimiter '{sep}')")
                break
        except pd.errors.ParserError:
            tried.append(f"'{sep}' -> ParserError")
            continue

    if df is None:
        print("⚠️ Gagal membaca file dengan delimiter ';' maupun ','.")
        for t in tried:
            print(f"   Percobaan: {t}")
        raise ValueError(f"Gagal membaca {config.HUMAN_VALIDATION_FILE}.")

    df["human_verdict"] = df["human_verdict"].astype(str).str.strip().str.lower()

    if os.path.exists(config.HUMAN_VALIDATION_INTERNAL_FILE):
        internal = pd.read_csv(config.HUMAN_VALIDATION_INTERNAL_FILE)[["review_id", "rating_diff"]]
        df = df.merge(internal, on="review_id", how="left")
    else:
        print("⚠️ File referensi internal tidak ditemukan -- breakdown per-bin rating_diff dilewati.")
        df["rating_diff"] = np.nan

    return df


def _check_completeness(df):
    empty_mask = df["human_verdict"].isin(["", "nan", "none"]) | df["human_verdict"].isna()
    n_empty = empty_mask.sum()

    if n_empty > 0:
        print(f"⚠️ Masih ada {n_empty} baris yang belum diisi 'human_verdict'.")
        print("   Isi manual semuanya, save, lalu jalankan lagi.")
        return False

    invalid_mask = ~df["human_verdict"].isin(VALID_VERDICTS)
    if invalid_mask.any():
        bad_values = df.loc[invalid_mask, "human_verdict"].unique().tolist()
        bad_rows = df.index[invalid_mask].tolist()
        print(f"⚠️ Ada nilai 'human_verdict' yang tidak dikenali: {bad_values}")
        print(f"   Baris ke-{bad_rows} (index dari 0). Nilai yang valid hanya: {sorted(VALID_VERDICTS)}")
        return False

    return True


def compute_agreement():
    df = _load_validated_sample()
    if df is None:
        return None
    if not _check_completeness(df):
        return None

    counts = df["human_verdict"].value_counts()
    total = len(df)
    agree = counts.get("noise", 0)
    ambiguous = counts.get("ambiguous", 0)
    disagree = counts.get("not_noise", 0)

    print("=" * 60)
    print(" HASIL VALIDASI MANUSIA vs CLEANLAB ")
    print("=" * 60)
    print(f"Total sample direview : {total}")
    print(f"Setuju (memang noise) : {agree} ({agree/total*100:.1f}%)")
    print(f"Ambigu                : {ambiguous} ({ambiguous/total*100:.1f}%)")
    print(f"Tidak setuju          : {disagree} ({disagree/total*100:.1f}%)")

    print("\n📊 Agreement rate per rating_diff:")
    breakdown_rows = []
    for diff_val, group in df.groupby("rating_diff"):
        n = len(group)
        vc = group["human_verdict"].value_counts()
        row = {
            "rating_diff": diff_val, "n_sample": n,
            "pct_noise": round(vc.get("noise", 0) / n * 100, 1),
            "pct_not_noise": round(vc.get("not_noise", 0) / n * 100, 1),
            "pct_ambiguous": round(vc.get("ambiguous", 0) / n * 100, 1),
        }
        breakdown_rows.append(row)
        print(f"   diff={diff_val}: n={n} | noise={row['pct_noise']}% | "
              f"not_noise={row['pct_not_noise']}% | ambiguous={row['pct_ambiguous']}%")

    breakdown_df = pd.DataFrame(breakdown_rows).sort_values("rating_diff")

    print("-" * 60)
    print("Acuan pembanding (Northcutt et al., 2021, ImageNet): ~58% sample")
    print("yang direview terbukti benar-benar issue -- acuan wajar, bukan standar mutlak.")
    print("=" * 60)

    df.to_csv(config.HUMAN_VALIDATION_RESULT_FILE, index=False)
    print(f"\n💾 Hasil lengkap -> {config.HUMAN_VALIDATION_RESULT_FILE}")

    return {
        "agree": int(agree), "ambiguous": int(ambiguous), "disagree": int(disagree),
        "total": int(total), "agreement_rate": round(agree / total, 4),
        "breakdown_by_rating_diff": breakdown_df.to_dict(orient="records"),
    }


# ==========================================================
# ANOTATOR KEDUA + COHEN'S KAPPA -- Subbab 3.9.3
# ==========================================================
def export_second_annotator_subset(fraction=None, seed=99):
    """Ekspor sebagian sample (default 30%) untuk dinilai independen oleh
    penilai kedua, tetap BUTA seperti anotator pertama."""
    fraction = fraction or config.SECOND_ANNOTATOR_FRACTION
    if not os.path.exists(config.HUMAN_VALIDATION_FILE):
        print("⚠️ Sample anotator pertama belum ada -- jalankan clean.py dulu.")
        return None

    df = pd.read_csv(config.HUMAN_VALIDATION_FILE)
    n_sub = max(1, int(round(len(df) * fraction)))
    subset = df.drop(columns=["human_verdict", "human_note"], errors="ignore").sample(
        n=n_sub, random_state=seed
    )
    subset["human_verdict"] = ""
    subset["human_note"] = ""
    subset.to_csv(config.HUMAN_VALIDATION_ANNOTATOR2_FILE, index=False)
    print(f"📝 Sample anotator kedua ({n_sub} baris, {fraction*100:.0f}% dari total) -> "
          f"{config.HUMAN_VALIDATION_ANNOTATOR2_FILE}")
    return config.HUMAN_VALIDATION_ANNOTATOR2_FILE


def compute_interannotator_kappa():
    """Cohen's kappa antara anotator pertama dan kedua pada baris overlap.
    Dilaporkan sebagai bukti kuantitatif reliabilitas anotasi (Subbab 3.9.3),
    sekaligus bukti langsung untuk RQ4 soal subjektivitas domain ini."""
    if not (os.path.exists(config.HUMAN_VALIDATION_FILE) and
            os.path.exists(config.HUMAN_VALIDATION_ANNOTATOR2_FILE)):
        print("⚠️ File anotator 1 dan/atau 2 belum lengkap.")
        return None

    df1 = pd.read_csv(config.HUMAN_VALIDATION_FILE)
    df2 = pd.read_csv(config.HUMAN_VALIDATION_ANNOTATOR2_FILE)
    merged = df1.merge(df2, on="review_id", suffixes=("_1", "_2"))
    merged["human_verdict_1"] = merged["human_verdict_1"].astype(str).str.strip().str.lower()
    merged["human_verdict_2"] = merged["human_verdict_2"].astype(str).str.strip().str.lower()
    merged = merged[
        merged["human_verdict_1"].isin(VALID_VERDICTS) &
        merged["human_verdict_2"].isin(VALID_VERDICTS)
    ]

    if len(merged) == 0:
        print("⚠️ Belum ada baris overlap yang lengkap diisi kedua anotator.")
        return None

    kappa = cohen_kappa_score(
        merged["human_verdict_1"], merged["human_verdict_2"], labels=sorted(VALID_VERDICTS)
    )
    print(f"🤝 Cohen's kappa antar-anotator (n={len(merged)}): {kappa:.4f}")

    pd.DataFrame([{"n_overlap": len(merged), "cohen_kappa": kappa}]).to_csv(
        config.HUMAN_VALIDATION_KAPPA_FILE, index=False
    )
    print(f"💾 Disimpan -> {config.HUMAN_VALIDATION_KAPPA_FILE}")
    return kappa


# ==========================================================
# FALLBACK SATU ANOTATOR -- TEST-RETEST RELIABILITY
# ==========================================================
def export_retest_subset(fraction=None, seed=77):
    """Fallback kalau hanya ada satu penilai: sebagian sample (default 10%)
    dinilai ulang mandiri tanpa melihat verdict sebelumnya (Subbab 3.9.3)."""
    fraction = fraction or config.TEST_RETEST_FRACTION
    if not os.path.exists(config.HUMAN_VALIDATION_FILE):
        print("⚠️ Sample anotator pertama belum ada -- jalankan clean.py dulu.")
        return None

    df = pd.read_csv(config.HUMAN_VALIDATION_FILE)
    n_sub = max(1, int(round(len(df) * fraction)))
    subset = df.drop(columns=["human_verdict", "human_note"], errors="ignore").sample(
        n=n_sub, random_state=seed
    )
    subset["human_verdict"] = ""
    subset["human_note"] = ""
    subset.to_csv(config.HUMAN_VALIDATION_RETEST_FILE, index=False)
    print(f"📝 Sample uji-ulang ({n_sub} baris) -> {config.HUMAN_VALIDATION_RETEST_FILE}")
    return config.HUMAN_VALIDATION_RETEST_FILE


def compute_test_retest_reliability():
    if not (os.path.exists(config.HUMAN_VALIDATION_RESULT_FILE) and
            os.path.exists(config.HUMAN_VALIDATION_RETEST_FILE)):
        print("⚠️ File hasil final dan/atau file uji-ulang belum lengkap.")
        return None

    df1 = pd.read_csv(config.HUMAN_VALIDATION_RESULT_FILE)
    df2 = pd.read_csv(config.HUMAN_VALIDATION_RETEST_FILE)
    merged = df1.merge(df2, on="review_id", suffixes=("_awal", "_retest"))
    merged["human_verdict_awal"] = merged["human_verdict_awal"].astype(str).str.strip().str.lower()
    merged["human_verdict_retest"] = merged["human_verdict_retest"].astype(str).str.strip().str.lower()
    merged = merged[
        merged["human_verdict_awal"].isin(VALID_VERDICTS) &
        merged["human_verdict_retest"].isin(VALID_VERDICTS)
    ]

    if len(merged) == 0:
        print("⚠️ Belum ada baris uji-ulang yang lengkap.")
        return None

    kappa = cohen_kappa_score(
        merged["human_verdict_awal"], merged["human_verdict_retest"], labels=sorted(VALID_VERDICTS)
    )
    print(f"🔁 Reliabilitas uji-ulang (test-retest kappa, n={len(merged)}): {kappa:.4f}")
    return kappa


if __name__ == "__main__":
    compute_agreement()