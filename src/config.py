"""
config.py -- Konfigurasi terpusat untuk seluruh pipeline skripsi.
Satu skema data (tidak ada versioning v1/v2) -- RAW_DATA_FILE, CLEAN_TEXT_FILE,
TRAIN_RAW_FILE, TEST_FILE semuanya tunggal dan dikunci sejak awal.
Satu backbone (IndoBERT), sesuai Batasan Masalah proposal Bab 1.6.
"""

import os
import sys
import torch

# ==========================================================
# 1. DETEKSI LINGKUNGAN & PATH DASAR
# ==========================================================
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    DRIVE_ROOT = '/content/drive/MyDrive/SKRIPSI_CORN'
else:
    DRIVE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local_data")

DATA_RAW_DIR = os.path.join(DRIVE_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(DRIVE_ROOT, "data", "processed")
PROXY_CACHE_DIR = os.path.join(DRIVE_ROOT, "proxy_cache")
CLEANED_DIR = os.path.join(DRIVE_ROOT, "cleaned")
MODEL_CKPT_ROOT = os.path.join(DRIVE_ROOT, "models_ckpt")
HUMAN_VALIDATION_DIR = os.path.join(DRIVE_ROOT, "human_validation")
RESULTS_DIR = os.path.join(DRIVE_ROOT, "results")
LOGS_DIR = os.path.join(DRIVE_ROOT, "logs")
GOLD_TEST_DIR = os.path.join(DRIVE_ROOT, "gold_test")  # BARU -- Subbab 3.9.4

for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, PROXY_CACHE_DIR, CLEANED_DIR,
          MODEL_CKPT_ROOT, HUMAN_VALIDATION_DIR, RESULTS_DIR, LOGS_DIR, GOLD_TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================================
# 2. FILE DATA -- SUMBER KEBENARAN TUNGGAL, DIKUNCI SETELAH DIBUAT
# ==========================================================
RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, "all_reviews_master.csv")
CLEAN_TEXT_FILE = os.path.join(DATA_PROCESSED_DIR, "reviews_clean.csv")
TRAIN_RAW_FILE = os.path.join(DATA_PROCESSED_DIR, "split_train_raw.csv")
VAL_FILE = os.path.join(DATA_PROCESSED_DIR, "split_val.csv")
TEST_FILE = os.path.join(DATA_PROCESSED_DIR, "split_test.csv")
TRAIN_RAW_RESOLVED_FILE = os.path.join(CLEANED_DIR, "train_resolved.csv")

# ==========================================================
# 3. BACKBONE MODEL BAHASA -- TUNGGAL, SESUAI BATASAN MASALAH BAB 1.6
# ==========================================================
PRETRAINED_MODEL_NAME = "indobenchmark/indobert-base-p1"

# ==========================================================
# 4. REGISTRY PROXY CLASSIFIER -- 4 TAHAP (P1-P4), SESUAI TABEL 3.1
# ==========================================================
PROXY_REGISTRY = {
    0: {"name": "frozen_cls_lr",         "desc": "CLS embedding beku + Logistic Regression (P1)"},
    1: {"name": "frozen_meanpool_lr",    "desc": "Mean-pooling embedding beku + Logistic Regression (P2)"},
    2: {"name": "finetuned_ce",          "desc": "IndoBERT fine-tuned K-Fold, CE loss (P3)"},
    3: {"name": "finetuned_corn",        "desc": "IndoBERT fine-tuned K-Fold, CORN loss (P4) -- DEFAULT/FINAL"},
}


def set_proxy(proxy_id):
    """
    Ganti proxy aktif dengan AMAN di tengah sesi Python. Pakai ini di notebook:
    config.set_proxy(pid). JANGAN pakai importlib.reload(config).
    """
    global PROXY_ID, PROXY_NAME, PROXY_DESC
    global PROXY_PRED_PROBS_FILE, PROXY_PRED_PROBS_META_FILE
    global TRAIN_CLEANED_HARD_FILE, TRAIN_CLEANED_SEVERE_FILE, MODEL_CKPT_DIR
    global HUMAN_VALIDATION_FILE, HUMAN_VALIDATION_INTERNAL_FILE
    global HUMAN_VALIDATION_ANNOTATOR2_FILE, HUMAN_VALIDATION_RESULT_FILE
    global HUMAN_VALIDATION_KAPPA_FILE, HUMAN_VALIDATION_RETEST_FILE

    if proxy_id not in PROXY_REGISTRY:
        raise ValueError(
            f"PROXY_ID tidak dikenal: {proxy_id} (harus salah satu dari "
            f"{sorted(PROXY_REGISTRY.keys())})"
        )

    PROXY_ID = proxy_id
    PROXY_NAME = PROXY_REGISTRY[proxy_id]["name"]
    PROXY_DESC = PROXY_REGISTRY[proxy_id]["desc"]

    PROXY_PRED_PROBS_FILE = os.path.join(PROXY_CACHE_DIR, f"oof_pred_probs__{PROXY_NAME}.npy")
    PROXY_PRED_PROBS_META_FILE = os.path.join(PROXY_CACHE_DIR, f"oof_pred_probs_meta__{PROXY_NAME}.csv")
    TRAIN_CLEANED_HARD_FILE = os.path.join(CLEANED_DIR, f"train_cleaned_hard__{PROXY_NAME}.csv")
    TRAIN_CLEANED_SEVERE_FILE = os.path.join(CLEANED_DIR, f"train_cleaned_severe__{PROXY_NAME}.csv")
    MODEL_CKPT_DIR = os.path.join(MODEL_CKPT_ROOT, PROXY_NAME)
    os.makedirs(MODEL_CKPT_DIR, exist_ok=True)

    # File anotator (BUTA -- tidak ada predicted_rating/rating_diff, lihat clean.py)
    HUMAN_VALIDATION_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_sample__{PROXY_NAME}.csv")
    # File referensi INTERNAL (peneliti saja -- menyimpan predicted_rating/rating_diff
    # untuk breakdown per-bin setelah verdict masuk)
    HUMAN_VALIDATION_INTERNAL_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_internal__{PROXY_NAME}.csv")
    HUMAN_VALIDATION_ANNOTATOR2_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_sample__{PROXY_NAME}__annotator2.csv")
    HUMAN_VALIDATION_RETEST_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_sample__{PROXY_NAME}__retest.csv")
    HUMAN_VALIDATION_RESULT_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_result__{PROXY_NAME}.csv")
    HUMAN_VALIDATION_KAPPA_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_kappa__{PROXY_NAME}.csv")

    print(f"📌 Proxy aktif: [{PROXY_ID}] {PROXY_NAME} — {PROXY_DESC}")


EMBEDDING_CACHE_FILE_BASE = os.path.join(PROXY_CACHE_DIR, "embeddings.npy")
EMBEDDING_CACHE_META_FILE_BASE = os.path.join(PROXY_CACHE_DIR, "embeddings_meta.csv")
PROXY_QUALITY_LOG_FILE = os.path.join(RESULTS_DIR, "proxy_ablation_table.csv")

set_proxy(3)

# ==========================================================
# 5. MODEL & HYPERPARAMETER
# ==========================================================

MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_CLASSES = 5

EPOCHS = 10
SEED_LIST = [42, 123, 2024]
PATIENCE = 3

PROXY_CV_FOLDS = 5
PROXY_FINETUNE_EPOCHS = 3
PROXY_FINETUNE_LR = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 6. FAST DEV MODE
# ==========================================================
DEBUG_MODE = os.environ.get("SKRIPSI_DEBUG", "0") == "1"
DEBUG_SAMPLE_SIZE = 600
DEBUG_EPOCHS = 2
DEBUG_CV_FOLDS = 2
DEBUG_SEED_LIST = [42]

if DEBUG_MODE:
    EPOCHS = DEBUG_EPOCHS
    PROXY_CV_FOLDS = DEBUG_CV_FOLDS
    SEED_LIST = DEBUG_SEED_LIST
    print(f"⚠️  SKRIPSI_DEBUG=1 AKTIF — subset {DEBUG_SAMPLE_SIZE} baris, {EPOCHS} epoch, seed {SEED_LIST}")
    print("    HASIL DI MODE INI TIDAK BOLEH DIPAKAI UNTUK LAPORAN.")

# ==========================================================
# 7. SPLIT VALIDASI, CLEANING, SEVERITY
# ==========================================================
TEST_SIZE = 0.20
VAL_SIZE = 0.10  # <- sudah ada, sekarang dipakai data_split.py (bukan train.py lagi)
CLEANLAB_FILTER_METHODS = ["confident_learning", "prune_by_noise_rate"]
SEVERITY_THRESHOLD = 2

# ==========================================================
# 8. VALIDASI MANUSIA -- Subbab 3.9.3
# ==========================================================
# n=100 sesuai rumus Cochran (Z=1.96, p=0.5, e=10%, koreksi populasi terbatas),
# BUKAN 50 -- lihat Bab III Subbab 3.9.3. Sebelumnya kode ini tidak sinkron
# dengan proposal, sudah diperbaiki di sini.
HUMAN_VALIDATION_N = 100
SECOND_ANNOTATOR_FRACTION = 0.3   # porsi sample yang dinilai independen oleh penilai kedua
TEST_RETEST_FRACTION = 0.10       # fallback single-annotator: uji-ulang pada ~10% sample

# ==========================================================
# 9. SUBSET UJI EMAS (GOLD TEST SUBSET) -- Subbab 3.9.4
# ==========================================================
GOLD_TEST_MARGIN_ERROR = 0.10
GOLD_TEST_SAMPLE_FILE = os.path.join(GOLD_TEST_DIR, "gold_test_sample.csv")
GOLD_TEST_RESULT_FILE = os.path.join(GOLD_TEST_DIR, "gold_test_sample.csv")  # diisi manual di file yang sama
GOLD_TEST_EVAL_FILE = os.path.join(RESULTS_DIR, "gold_test_evaluation.csv")
GOLD_TEST_EFFECT_SIZE_FILE = os.path.join(RESULTS_DIR, "gold_test_effect_sizes.csv")

# ==========================================================
# 10. HASIL AKHIR
# ==========================================================
FINAL_RESULTS_TABLE_FILE = os.path.join(RESULTS_DIR, "final_results_table.csv")
SIGNIFICANCE_TEST_FILE = os.path.join(RESULTS_DIR, "significance_test.csv")
NOISE_SAMPLES_FILE = os.path.join(DATA_PROCESSED_DIR, "detected_noise_samples.csv")
PROGRESS_FILE = os.path.join(LOGS_DIR, "experiment_progress.json")