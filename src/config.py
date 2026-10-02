"""
config.py -- Konfigurasi terpusat. Satu skema data (all_reviews_master.csv dikunci),
satu backbone (IndoBERT). Mendukung analisis sensitivitas split lewat set_split().
Split utama (seed 42) memakai path lama; split lain masuk subfolder 'split<seed>'.
"""
import os
import sys
import torch

# ==========================================================
# 1. LINGKUNGAN & PATH DASAR
# ==========================================================
IN_COLAB = 'google.colab' in sys.modules
DRIVE_FOLDER_NAME = "SKRIPSI_CORN_CL_FINAL"

if IN_COLAB:
    DRIVE_ROOT = os.environ.get("SKRIPSI_DRIVE_ROOT", f"/content/drive/MyDrive/{DRIVE_FOLDER_NAME}")
else:
    DRIVE_ROOT = os.environ.get(
        "SKRIPSI_DRIVE_ROOT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local_data"),
    )

# Direktori PRIMER (split utama, seed 42)
DATA_RAW_DIR = os.path.join(DRIVE_ROOT, "data", "raw")
PRIMARY_DATA_PROCESSED_DIR = os.path.join(DRIVE_ROOT, "data", "processed")
PRIMARY_PROXY_CACHE_DIR = os.path.join(DRIVE_ROOT, "proxy_cache")
PRIMARY_CLEANED_DIR = os.path.join(DRIVE_ROOT, "cleaned")
PRIMARY_MODEL_CKPT_ROOT = os.path.join(DRIVE_ROOT, "models_ckpt")
PRIMARY_HUMAN_VALIDATION_DIR = os.path.join(DRIVE_ROOT, "human_validation")
PRIMARY_RESULTS_DIR = os.path.join(DRIVE_ROOT, "results")
PRIMARY_LOGS_DIR = os.path.join(DRIVE_ROOT, "logs")
GOLD_TEST_DIR = os.path.join(DRIVE_ROOT, "gold_test")  # hanya split utama

for d in [DATA_RAW_DIR, PRIMARY_DATA_PROCESSED_DIR, PRIMARY_PROXY_CACHE_DIR, PRIMARY_CLEANED_DIR,
          PRIMARY_MODEL_CKPT_ROOT, PRIMARY_HUMAN_VALIDATION_DIR, PRIMARY_RESULTS_DIR,
          PRIMARY_LOGS_DIR, GOLD_TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================================
# 2. FILE DATA TETAP (dibagi bersama oleh semua split)
# ==========================================================
RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, "all_reviews_master.csv")
CLEAN_TEXT_FILE = os.path.join(PRIMARY_DATA_PROCESSED_DIR, "reviews_clean.csv")
MANIFEST_FILE = os.path.join(PRIMARY_RESULTS_DIR, "dataset_manifest.json")

# ==========================================================
# 3. BACKBONE
# ==========================================================
PRETRAINED_MODEL_NAME = "indobenchmark/indobert-base-p1"

# ==========================================================
# 4. REGISTRY PROXY (P1-P4, Tabel 3.1)
# ==========================================================
PROXY_REGISTRY = {
    0: {"name": "frozen_cls_lr",      "desc": "CLS embedding beku + Logistic Regression (P1)"},
    1: {"name": "frozen_meanpool_lr", "desc": "Mean-pooling embedding beku + Logistic Regression (P2)"},
    2: {"name": "finetuned_ce",       "desc": "IndoBERT fine-tuned K-Fold, CE loss (P3)"},
    3: {"name": "finetuned_corn",     "desc": "IndoBERT fine-tuned K-Fold, CORN loss (P4) -- DEFAULT/FINAL"},
}
PROXY_ID = 3


def set_proxy(proxy_id, verbose=True):
    """Ganti proxy aktif. Pakai config.set_proxy(pid), JANGAN importlib.reload(config)."""
    global PROXY_ID, PROXY_NAME, PROXY_DESC
    global PROXY_PRED_PROBS_FILE, PROXY_PRED_PROBS_META_FILE
    global TRAIN_CLEANED_HARD_FILE, TRAIN_CLEANED_SEVERE_FILE, MODEL_CKPT_DIR
    global HUMAN_VALIDATION_FILE, HUMAN_VALIDATION_INTERNAL_FILE
    global HUMAN_VALIDATION_ANNOTATOR2_FILE, HUMAN_VALIDATION_RESULT_FILE
    global HUMAN_VALIDATION_KAPPA_FILE, HUMAN_VALIDATION_RETEST_FILE

    if proxy_id not in PROXY_REGISTRY:
        raise ValueError(f"PROXY_ID tidak dikenal: {proxy_id} (harus salah satu dari {sorted(PROXY_REGISTRY)})")

    PROXY_ID = proxy_id
    PROXY_NAME = PROXY_REGISTRY[proxy_id]["name"]
    PROXY_DESC = PROXY_REGISTRY[proxy_id]["desc"]

    PROXY_PRED_PROBS_FILE = os.path.join(PROXY_CACHE_DIR, f"oof_pred_probs__{PROXY_NAME}.npy")
    PROXY_PRED_PROBS_META_FILE = os.path.join(PROXY_CACHE_DIR, f"oof_pred_probs_meta__{PROXY_NAME}.csv")
    TRAIN_CLEANED_HARD_FILE = os.path.join(CLEANED_DIR, f"train_cleaned_hard__{PROXY_NAME}.csv")
    TRAIN_CLEANED_SEVERE_FILE = os.path.join(CLEANED_DIR, f"train_cleaned_severe__{PROXY_NAME}.csv")
    MODEL_CKPT_DIR = os.path.join(MODEL_CKPT_ROOT, PROXY_NAME)
    os.makedirs(MODEL_CKPT_DIR, exist_ok=True)

    HUMAN_VALIDATION_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_sample__{PROXY_NAME}.csv")
    HUMAN_VALIDATION_INTERNAL_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_internal__{PROXY_NAME}.csv")
    HUMAN_VALIDATION_ANNOTATOR2_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_sample__{PROXY_NAME}__annotator2.csv")
    HUMAN_VALIDATION_RETEST_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_sample__{PROXY_NAME}__retest.csv")
    HUMAN_VALIDATION_RESULT_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_result__{PROXY_NAME}.csv")
    HUMAN_VALIDATION_KAPPA_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_kappa__{PROXY_NAME}.csv")

    if verbose:
        print(f"📌 Proxy aktif: [{PROXY_ID}] {PROXY_NAME} — {PROXY_DESC}")


# ==========================================================
# 5. MODEL & HYPERPARAMETER (Tabel 3.3)
# ==========================================================
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_CLASSES = 5
EPOCHS = 10
SEED_LIST = [42, 123, 2024]          # seed BOBOT/inisialisasi (bukan seed split)
PATIENCE = 3

PROXY_CV_FOLDS = 5
PROXY_FINETUNE_EPOCHS = 3
PROXY_FINETUNE_LR = 2e-5
K_SENSITIVITY_VALUES = [3, 5, 10]    # §3.5

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
# 7. SPLIT, CLEANING, SEVERITY
# ==========================================================
TEST_SIZE = 0.20
VAL_SIZE = 0.10
CLEANLAB_FILTER_METHODS = ["confident_learning", "prune_by_noise_rate"]
SEVERITY_THRESHOLD = 2

# ==========================================================
# 8. VALIDASI MANUSIA (§3.7) -- hanya split utama
# ==========================================================
HUMAN_VALIDATION_N = 100
SECOND_ANNOTATOR_FRACTION = 0.3
TEST_RETEST_FRACTION = 0.10

# ==========================================================
# 9. SUBSET UJI EMAS (§3.10) -- hanya split utama
# ==========================================================
GOLD_TEST_MARGIN_ERROR = 0.10
GOLD_TEST_SAMPLE_FILE = os.path.join(GOLD_TEST_DIR, "gold_test_sample.csv")
GOLD_TEST_RESULT_FILE = os.path.join(GOLD_TEST_DIR, "gold_test_sample.csv")
GOLD_TEST_RUBRIC_FILE = os.path.join(GOLD_TEST_DIR, "gold_test_rubric.md")
GOLD_TEST_EVAL_FILE = os.path.join(PRIMARY_RESULTS_DIR, "gold_test_evaluation.csv")
GOLD_TEST_EFFECT_SIZE_FILE = os.path.join(PRIMARY_RESULTS_DIR, "gold_test_effect_sizes.csv")

# ==========================================================
# 10. FILE HASIL KHUSUS SPLIT UTAMA
# ==========================================================
K_SENSITIVITY_FILE = os.path.join(PRIMARY_RESULTS_DIR, "proxy_k_sensitivity.csv")

# ==========================================================
# 11. SENSITIVITAS SPLIT (seed PARTISI data; beda dari SEED_LIST)
# ==========================================================
PRIMARY_SPLIT_SEED = 42
SPLIT_SEEDS = [42, 123, 2024]
SPLIT_SEED = PRIMARY_SPLIT_SEED
SPLIT_TAG = ""


def set_split(split_seed, verbose=True):
    """
    Split 42 = path lama (tidak berubah). Split lain -> subfolder 'split<seed>' pada
    data/processed, proxy_cache, cleaned, models_ckpt, human_validation, results, logs.
    RAW_DATA_FILE, CLEAN_TEXT_FILE, lexicon, dan semua path gold test dibagi bersama.
    """
    global SPLIT_SEED, SPLIT_TAG
    global DATA_PROCESSED_DIR, PROXY_CACHE_DIR, CLEANED_DIR, MODEL_CKPT_ROOT
    global HUMAN_VALIDATION_DIR, RESULTS_DIR, LOGS_DIR
    global TRAIN_RAW_FILE, VAL_FILE, TEST_FILE, TRAIN_RAW_RESOLVED_FILE
    global EMBEDDING_CACHE_FILE_BASE, EMBEDDING_CACHE_META_FILE_BASE
    global PROXY_QUALITY_LOG_FILE, FINAL_RESULTS_TABLE_FILE
    global SIGNIFICANCE_TEST_FILE, NOISE_SAMPLES_FILE, PROGRESS_FILE

    SPLIT_SEED = split_seed
    SPLIT_TAG = "" if split_seed == PRIMARY_SPLIT_SEED else f"split{split_seed}"

    def _d(primary):
        return os.path.join(primary, SPLIT_TAG) if SPLIT_TAG else primary

    DATA_PROCESSED_DIR = _d(PRIMARY_DATA_PROCESSED_DIR)
    PROXY_CACHE_DIR = _d(PRIMARY_PROXY_CACHE_DIR)
    CLEANED_DIR = _d(PRIMARY_CLEANED_DIR)
    MODEL_CKPT_ROOT = _d(PRIMARY_MODEL_CKPT_ROOT)
    HUMAN_VALIDATION_DIR = _d(PRIMARY_HUMAN_VALIDATION_DIR)
    RESULTS_DIR = _d(PRIMARY_RESULTS_DIR)
    LOGS_DIR = _d(PRIMARY_LOGS_DIR)
    for d in [DATA_PROCESSED_DIR, PROXY_CACHE_DIR, CLEANED_DIR, MODEL_CKPT_ROOT,
              HUMAN_VALIDATION_DIR, RESULTS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    TRAIN_RAW_FILE = os.path.join(DATA_PROCESSED_DIR, "split_train_raw.csv")
    VAL_FILE = os.path.join(DATA_PROCESSED_DIR, "split_val.csv")
    TEST_FILE = os.path.join(DATA_PROCESSED_DIR, "split_test.csv")
    TRAIN_RAW_RESOLVED_FILE = os.path.join(CLEANED_DIR, "train_resolved.csv")
    NOISE_SAMPLES_FILE = os.path.join(DATA_PROCESSED_DIR, "detected_noise_samples.csv")
    EMBEDDING_CACHE_FILE_BASE = os.path.join(PROXY_CACHE_DIR, "embeddings.npy")
    EMBEDDING_CACHE_META_FILE_BASE = os.path.join(PROXY_CACHE_DIR, "embeddings_meta.csv")
    PROXY_QUALITY_LOG_FILE = os.path.join(RESULTS_DIR, "proxy_ablation_table.csv")
    FINAL_RESULTS_TABLE_FILE = os.path.join(RESULTS_DIR, "final_results_table.csv")
    SIGNIFICANCE_TEST_FILE = os.path.join(RESULTS_DIR, "significance_test.csv")
    PROGRESS_FILE = os.path.join(LOGS_DIR, "experiment_progress.json")

    set_proxy(PROXY_ID, verbose=False)   # turunkan ulang path yang bergantung proxy
    if verbose:
        print(f"🧩 Split aktif: seed={SPLIT_SEED} ({'UTAMA' if not SPLIT_TAG else SPLIT_TAG}) | "
              f"proxy [{PROXY_ID}] {PROXY_NAME}")


def require_primary_split(what):
    if SPLIT_SEED != PRIMARY_SPLIT_SEED:
        raise RuntimeError(
            f"{what} hanya valid untuk split utama (seed {PRIMARY_SPLIT_SEED}); aktif: {SPLIT_SEED}. "
            f"Jalankan config.set_split({PRIMARY_SPLIT_SEED})."
        )


set_split(PRIMARY_SPLIT_SEED, verbose=False)
print(f"📌 Proxy aktif: [{PROXY_ID}] {PROXY_NAME} — {PROXY_DESC}")