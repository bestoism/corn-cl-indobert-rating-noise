"""
preprocess.py: praproses teks ulasan untuk fine-tuning IndoBERT (Subbab 3.3).

Praproses dijaga minimal (tanpa stemming dan stopword removal). Filter bahasa sudah
dilakukan saat scraping (kolom detected_lang) dan tidak diulang di sini.

Berkas kamus slang (lexicon) adalah masukan yang dipertahankan di Drive. Modul ini
TIDAK mengunduh kamus secara otomatis, supaya run murni tidak bergantung pada isi
sumber daring yang dapat berubah. Hash kamus dicatat pada log lingkungan.
"""

import os
import re

import pandas as pd

from src import config, drive_io

# ----------------------------------------------------------------------
# Kamus emoji ke sentimen (Subbab 3.3, butir 2)
# ----------------------------------------------------------------------
EMOJI_SENTIMENT = {
    "\U0001F62D": " sedih ", "\U0001F622": " sedih ", "\U0001F614": " kecewa ", "\U0001F61E": " kecewa ",
    "\U0001F61F": " khawatir ", "\U0001F629": " lelah ", "\U0001F62B": " lelah ", "\U0001F616": " kesal ",
    "\u2639\ufe0f": " sedih ", "\U0001F641": " sedih ", "\U0001F494": " kecewa ",
    "\U0001F621": " marah ", "\U0001F92C": " marah ", "\U0001F624": " kesal ", "\U0001F611": " kesal ",
    "\U0001F620": " marah ", "\U0001F44A": " marah ", "\U0001F644": " kesal ",
    "\U0001F60D": " suka ", "\u2764\ufe0f": " suka ", "\U0001F9E1": " suka ", "\U0001F49B": " suka ",
    "\U0001F49A": " suka ", "\U0001F499": " suka ", "\U0001F49C": " suka ", "\U0001F5A4": " suka ",
    "\U0001F90D": " suka ", "\U0001F90E": " suka ", "\U0001F60A": " senang ", "\U0001F601": " senang ",
    "\U0001F606": " senang ", "\U0001F970": " suka ", "\U0001F618": " suka ", "\U0001F64C": " senang ",
    "\U0001F389": " senang ", "\U0001F38A": " senang ", "\u2728": " bagus ", "\U0001F525": " bagus ",
    "\U0001F4AF": " bagus ", "\U0001F44F": " bagus ", "\U0001F44C": " bagus ", "\U0001F4AA": " semangat ",
    "\U0001F91D": " terima_kasih ", "\U0001F602": " lucu ", "\U0001F923": " lucu ",
    "\U0001F64F": " terima_kasih ",
    "\U0001F44D": " bagus ", "\U0001F44E": " buruk ",
    "\U0001F631": " kaget ", "\U0001F628": " takut ", "\U0001F630": " cemas ",
    "\U0001F605": " canggung ", "\U0001F609": " bercanda ", "\U0001F61C": " bercanda ",
    "\U0001F914": " bingung ", "\U0001F610": " biasa_saja ",
}

# Emoji yang tidak tercakup kamus dihapus setelah translasi.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002190-\U000021FF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)

REQUIRED_SLANG_COLUMNS = ["slang", "formal"]
KAMUS_ALAY_RUJUKAN = "https://github.com/nasalsabila/kamus-alay (colloquial-indonesian-lexicon.csv)"


def _validate_slang_columns(df, source_path):
    """Validasi eksplisit kolom kamus (Subbab 3.3, butir 5)."""
    missing = [c for c in REQUIRED_SLANG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kamus slang di '{source_path}' tidak memiliki kolom yang diharapkan: {missing}. "
            f"Kolom tersedia: {df.columns.tolist()}. Kemungkinan format sumber kamus berubah "
            f"(rujukan: {KAMUS_ALAY_RUJUKAN})."
        )


def load_slang_lexicon(base_path=None, domain_path=None):
    """
    Memuat dan menggabungkan dua kamus: slang_base.csv (Kamus Alay, wajib) dan
    slang_domain.csv (entri khusus domain dari PRDECT-ID, opsional).
    """
    base_path = base_path or config.SLANG_BASE_FILE
    domain_path = domain_path or config.SLANG_DOMAIN_FILE
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"[BERHENTI] Kamus dasar tidak ditemukan: {base_path}. Folder lexicon adalah masukan "
            f"yang dipertahankan; salin kembali berkasnya ke Drive."
        )
    lexicon = {}
    df_base = pd.read_csv(base_path)
    _validate_slang_columns(df_base, base_path)
    df_base = df_base[REQUIRED_SLANG_COLUMNS]
    lexicon.update(dict(zip(df_base["slang"], df_base["formal"])))
    n_base = len(df_base)

    n_domain_new = 0
    if os.path.exists(domain_path):
        df_domain = pd.read_csv(domain_path)
        _validate_slang_columns(df_domain, domain_path)
        before = len(lexicon)
        lexicon.update(dict(zip(df_domain["slang"], df_domain["formal"])))
        n_domain_new = len(lexicon) - before
        print(f"[OK] Kamus domain dimuat: {len(df_domain)} entri ({n_domain_new} baru).")
    else:
        print(f"[PERINGATAN] Kamus domain tidak ditemukan ({domain_path}). Proposal Subbab 3.3 "
              f"menyebut entri domain dari PRDECT-ID; tanpa berkas ini hanya Kamus Alay yang dipakai.")
    print(f"[OK] Kamus dasar dimuat: {n_base} entri.")
    return lexicon


_SLANG_DICT = None


def get_slang_dict():
    global _SLANG_DICT
    if _SLANG_DICT is None:
        _SLANG_DICT = load_slang_lexicon()
    return _SLANG_DICT


# ----------------------------------------------------------------------
# Fungsi pembersihan
# ----------------------------------------------------------------------
def replace_emoji_sentiment(text):
    for emo, word in EMOJI_SENTIMENT.items():
        text = text.replace(emo, word)
    return text


def strip_unrecognized_emoji(text):
    return _EMOJI_PATTERN.sub(" ", text)


_WORD_STRIP_PATTERN = re.compile(r"^(\W*)(\w+)(\W*)$", flags=re.UNICODE)


def _normalize_word_with_slang(word, slang_dict):
    """Tanda baca di tepi kata dipisahkan sementara sebelum pencarian kamus (Subbab 3.3, butir 4)."""
    match = _WORD_STRIP_PATTERN.match(word)
    if not match:
        return word
    prefix, core, suffix = match.groups()
    return f"{prefix}{slang_dict.get(core, core)}{suffix}"


def clean_text_for_bert(text, slang_dict=None):
    if not isinstance(text, str):
        return ""
    slang_dict = get_slang_dict() if slang_dict is None else slang_dict

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)

    text = replace_emoji_sentiment(text)
    text = strip_unrecognized_emoji(text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    if slang_dict:
        words = text.split()
        text = " ".join(_normalize_word_with_slang(w, slang_dict) for w in words)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------
# Analitik dan pipeline
# ----------------------------------------------------------------------
def compute_slang_coverage(df, slang_dict, raw_col="review_text"):
    """Estimasi kasar cakupan kamus dari teks mentah (bukan angka final pipeline)."""
    all_words = " ".join(df[raw_col].dropna().astype(str).str.lower()).split()
    total = len(all_words)
    cleaned_words = [re.sub(r"[^a-z0-9]", "", w) for w in all_words]
    matched = sum(1 for w in cleaned_words if w in slang_dict)
    return {"total_words": total, "normalized_words": matched,
            "coverage_pct": round(matched / total * 100, 2) if total > 0 else 0.0}


def drop_text_rating_conflicts(df, text_col="cleaned_text", verbose=True):
    """
    Resolusi teks identik dengan rating berbeda melalui voting mayoritas (Subbab 3.3, butir 8):
    baris dengan rating minoritas dibuang, grup tanpa mayoritas jelas (seri) dibuang seluruhnya.

    CATATAN: karena duplikat murni (teks dan rating identik) sudah dihapus pada run_preprocessing,
    setiap grup konflik memuat tepat satu baris per rating. Semua grup konflik karena itu seri,
    dan seluruhnya dibuang. Jumlah baris minoritas yang dilaporkan akan selalu nol.
    """
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    df_dup = df[dup_mask]
    df_unique = df[~dup_mask]

    keep_groups = []
    n_dropped_minority = 0
    n_dropped_tie = 0

    for _, group in df_dup.groupby(text_col):
        rating_counts = group["rating"].value_counts()
        if len(rating_counts) == 1:
            keep_groups.append(group)
            continue
        top_count = rating_counts.iloc[0]
        modes = rating_counts[rating_counts == top_count].index.tolist()
        if len(modes) > 1:
            n_dropped_tie += len(group)
            continue
        majority_rating = modes[0]
        kept = group[group["rating"] == majority_rating]
        n_dropped_minority += len(group) - len(kept)
        keep_groups.append(kept)

    df_result = pd.concat([df_unique] + keep_groups, ignore_index=True) if keep_groups else df_unique
    if verbose:
        print(f"Baris dibuang (rating minoritas dalam grup konflik): {n_dropped_minority}")
        print(f"Baris dibuang (seri, tidak ada mayoritas jelas)    : {n_dropped_tie}")
        print(f"Total dibuang                                      : {n_dropped_minority + n_dropped_tie}")
    return df_result


def compute_token_truncation_rate(df, text_col="cleaned_text", max_len=None, tokenizer=None):
    from transformers import AutoTokenizer
    max_len = max_len or config.MAX_LEN
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    token_lengths = df[text_col].astype(str).apply(
        lambda t: len(tokenizer.encode(t, add_special_tokens=True)))
    pct_truncated = (token_lengths > max_len).mean() * 100
    return {"max_len": max_len, "pct_truncated": round(float(pct_truncated), 2),
            "median_token_len": int(token_lengths.median())}


def run_preprocessing(master_path=None, tokenizer=None, write=True):
    master_path = master_path or config.MASTER_FILE
    slang_dict = get_slang_dict()

    df = pd.read_csv(master_path)
    initial_len = len(df)
    print(f"Jumlah data awal: {initial_len} baris")

    df = df.dropna(subset=["review_text"])
    df["cleaned_text"] = df["review_text"].apply(lambda t: clean_text_for_bert(t, slang_dict))

    slang_coverage = compute_slang_coverage(df, slang_dict)
    print(f"Cakupan kamus slang (estimasi dari teks mentah): "
          f"{slang_coverage['normalized_words']}/{slang_coverage['total_words']} kata "
          f"({slang_coverage['coverage_pct']}%)")
    df = df[df["cleaned_text"].str.strip() != ""]

    word_counts = df["cleaned_text"].astype(str).str.split().apply(len)
    is_very_short = word_counts <= 3
    df["cleaned_text_word_count"] = word_counts
    df["is_very_short_text"] = is_very_short
    print(f"Teks sangat pendek (tiga kata atau kurang): {is_very_short.mean() * 100:.2f}%")

    dup_mask = df.duplicated(subset=["cleaned_text"], keep=False)
    conflicting = df[dup_mask].groupby("cleaned_text")["rating"].nunique()
    n_conflicting = int((conflicting > 1).sum())
    print(f"Teks identik dengan rating berbeda (setelah pembersihan): {n_conflicting} grup")
    print("Resolusi voting mayoritas dijalankan setelah deteksi Confident learning (Subbab 3.3, 3.6).")

    df = df.drop_duplicates(subset=["cleaned_text", "rating"])  # hanya duplikat murni
    final_len = len(df)
    print(f"Setelah dibersihkan: {final_len} baris (terbuang: {initial_len - final_len})")

    truncation = compute_token_truncation_rate(df, tokenizer=tokenizer) if tokenizer is not False else {}
    if truncation:
        print(f"Ulasan yang terpotong (> {truncation['max_len']} token): {truncation['pct_truncated']}%")

    summary = {
        "initial_rows": initial_len, "final_rows": final_len,
        "rows_dropped": initial_len - final_len,
        "text_rating_conflict_groups": n_conflicting,
        "pct_very_short_text": round(float(is_very_short.mean() * 100), 2),
        **{f"truncation_{k}": v for k, v in truncation.items()},
        **{f"slang_{k}": v for k, v in slang_coverage.items()},
    }
    if write:
        drive_io.write_once_csv(df, config.CLEAN_TEXT_FILE)
        drive_io.write_once_csv(pd.DataFrame([summary]),
                                os.path.join(config.SHARED_RESULTS_DIR, "preprocessing_summary.csv"))
    return df, summary
