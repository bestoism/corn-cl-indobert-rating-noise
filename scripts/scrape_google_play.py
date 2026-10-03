"""
Scraper Google Play Store (stratified per rating, resumable, filter bahasa inline).

PERINGATAN: skrip ini TIDAK dijalankan oleh notebook final. Korpus mentah
(data/raw/all_reviews_master.csv, 16740 baris) sudah terkunci dengan SHA-256 yang tertulis di
src/config.py. Menjalankan scraping ulang menghasilkan data berbeda dan merusak penguncian.
Skrip ini disimpan sebagai dokumentasi prosedur pengumpulan data (Subbab 3.2) dan menolak
berjalan bila berkas master sudah ada.

Laporan kualitas data (distribusi tanggal, panjang teks, duplikat) dipindahkan ke
src/quality_report.py yang berjalan dari berkas master. Yang tetap hanya dapat berasal dari
scraping adalah scraping_summary.csv (jumlah diperiksa dan dibuang oleh filter bahasa).

Memerlukan: pip install -r requirements-scraping.txt
"""

import os
import pickle
import random
import time
from datetime import datetime

import pandas as pd

from src import config

APPS_CONFIG = [
    {"app_id": "id.co.bankbkemobile.digitalbank", "name": "SeaBank"},
    {"app_id": "com.tokopedia.tkpd", "name": "Tokopedia"},
    {"app_id": "com.gojek.app", "name": "Gojek"},
]
TARGET_PER_SCORE = {1: 1200, 2: 900, 3: 900, 4: 1000, 5: 1200}
BATCH_SIZE = 200
MAX_RETRIES = 5
SLEEP_RANGE = (2, 5)
ACCEPTED_LANG = "id"

RAW_DIR = config.RAW_DIR
STATE_DIR = os.path.join(config.DRIVE_ROOT, "scrape_state")
ERROR_LOG_FILE = os.path.join(STATE_DIR, "scrape_errors.log")


def _log_error(app_name, score, message):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {app_name} | rating={score} | {message}\n")


def _detect_language(text):
    """
    py3langid dipilih karena lebih stabil untuk teks pendek dan informal. Keterbatasan: ulasan
    campur kode berat dapat terdeteksi sebagai 'en' meski secara substansi berbahasa Indonesia.
    """
    import py3langid as langid
    text = (text or "").strip()
    if len(text) < 3:
        return None
    try:
        lang, _ = langid.classify(text)
        return lang
    except Exception:
        return None


def _safe_scrape_call(fetch_fn, app_name, score, max_retries=MAX_RETRIES):
    last_error = None
    for attempt in range(max_retries):
        try:
            return fetch_fn()
        except Exception as e:
            last_error = e
            _log_error(app_name, score, f"Percobaan {attempt + 1} gagal: {e}")
            time.sleep(attempt + 2)
    _log_error(app_name, score, f"GAGAL TOTAL setelah {max_retries} percobaan: {last_error}")
    raise RuntimeError(f"Gagal setelah {max_retries} percobaan: {last_error}")


def _state_path(app_name, score):
    return os.path.join(STATE_DIR, f"{app_name}_score{score}.pkl")


def _load_state(app_name, score):
    sp = _state_path(app_name, score)
    ck = os.path.join(RAW_DIR, "_checkpoints", f"{app_name}_score{score}.csv")
    if os.path.exists(sp) and os.path.exists(ck):
        with open(sp, "rb") as f:
            state = pickle.load(f)
        return state.get("continuation_token"), pd.read_csv(ck).to_dict("records")
    return None, []


def _save_checkpoint(app_name, score, token, collected):
    os.makedirs(os.path.join(RAW_DIR, "_checkpoints"), exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(_state_path(app_name, score), "wb") as f:
        pickle.dump({"continuation_token": token, "collected": len(collected)}, f)
    pd.DataFrame(collected).to_csv(os.path.join(RAW_DIR, "_checkpoints", f"{app_name}_score{score}.csv"), index=False)


def scrape_one_score(app_id, app_name, score, target_count, lang="id", country="id"):
    from google_play_scraper import Sort, reviews
    token, collected = _load_state(app_name, score)
    checkpoint_marker = len(collected)
    n_checked, n_lang_dropped, n_too_short = len(collected), 0, 0

    while len(collected) < target_count:
        def fetch():
            return reviews(app_id=app_id, lang=lang, country=country, sort=Sort.NEWEST,
                           count=BATCH_SIZE, filter_score_with=score, continuation_token=token)
        try:
            result, new_token = _safe_scrape_call(fetch, app_name, score)
        except RuntimeError:
            break
        if not result:
            break
        for r in result:
            n_checked += 1
            text = r.get("content", "") or ""
            detected = _detect_language(text)
            if detected is None:
                n_too_short += 1
                keep = True
            elif detected == ACCEPTED_LANG:
                keep = True
            else:
                keep = False
                n_lang_dropped += 1
            if not keep:
                continue
            collected.append({
                "source_app": app_name, "review_text": text, "rating": r.get("score", score),
                "date": r.get("at"), "helpful_votes": r.get("thumbsUpCount", 0),
                "review_id": r.get("reviewId", ""),
                "detected_lang": detected if detected is not None else "undetected_short",
                "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        token = new_token
        if len(collected) - checkpoint_marker >= 500:
            _save_checkpoint(app_name, score, token, collected)
            checkpoint_marker = len(collected)
        time.sleep(random.uniform(*SLEEP_RANGE))
        if token is None:
            break

    _save_checkpoint(app_name, score, token, collected)
    df = pd.DataFrame(collected)
    if not df.empty:
        df = df.drop_duplicates(subset=["review_id"])
        df = df[df["review_text"].str.strip() != ""]
    status = "PENUH" if len(df) >= target_count else "KURANG"
    if df.empty:
        status = "KOSONG TOTAL"
        _log_error(app_name, score, "Selesai dengan 0 baris terkumpul.")
    return df, {"source_app": app_name, "rating": score, "target": target_count,
                "n_collected": len(df), "n_checked": n_checked, "n_lang_dropped": n_lang_dropped,
                "n_too_short_for_detect": n_too_short, "status": status}


def main():
    if os.path.exists(config.MASTER_FILE):
        raise SystemExit(
            f"[BERHENTI] {config.MASTER_FILE} sudah ada dan terkunci. Scraping ulang tidak dijalankan "
            f"karena akan merusak penguncian SHA-256 (src/config.py).")
    all_data, summaries = [], []
    for app_cfg in APPS_CONFIG:
        frames = []
        for score, target in TARGET_PER_SCORE.items():
            df_score, summary = scrape_one_score(app_cfg["app_id"], app_cfg["name"], score, target)
            frames.append(df_score)
            summaries.append(summary)
        df_app = pd.concat(frames, ignore_index=True)
        os.makedirs(RAW_DIR, exist_ok=True)
        df_app.to_csv(os.path.join(RAW_DIR, f"{app_cfg['name']}_reviews.csv"), index=False)
        all_data.append(df_app)
    final_df = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=["review_id"])
    final_df["date"] = pd.to_datetime(final_df["date"]).dt.strftime("%Y-%m-%d")
    final_df.to_csv(config.MASTER_FILE, index=False)
    os.makedirs(config.PRESERVED_DIR, exist_ok=True)
    pd.DataFrame(summaries).to_csv(os.path.join(config.PRESERVED_DIR, "scraping_summary.csv"), index=False)
    print(f"Selesai: {len(final_df)} ulasan. Catat SHA-256 baru ke config.EXPECTED_MASTER_SHA256.")


if __name__ == "__main__":
    main()
