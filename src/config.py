"""
config.py: konfigurasi terpusat.

Prinsip penyimpanan (Subbab 3.13 dan catatan ketahanan Drive):
  - Drive hanya berisi berkas unik yang ditulis sekali (lihat drive_io.py).
  - Tidak ada folder per skenario. Seluruh berkas per skenario dan seed memakai
    nama datar, misalnya M4_Baseline_CORN__seed42.json, pada folder yang dibuat
    sekali di awal.
  - Checkpoint model (sekitar 500 MB) ditulis ke disk lokal Colab (LOCAL_ROOT).

Struktur folder keluaran per split partisi (seragam untuk split utama dan
sensitivitas):
  runs/split_seed{S}/{data,proxy,ablation,diagnostics,noise,cleaned,
                      training,predictions,significance,validation,
                      gold,k_sensitivity}
Folder ablation, validation, gold, dan k_sensitivity hanya terisi pada split
utama (seed partisi 42).
"""

import os
import sys

IN_COLAB = "google.colab" in sys.modules

_DEFAULT_COLAB_ROOT = "/content/drive/MyDrive/SKRIPSI_CORN_CL_FINAL"
_DEFAULT_LOCAL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local_data"
)

DRIVE_ROOT = os.environ.get(
    "SKRIPSI_DRIVE_ROOT", _DEFAULT_COLAB_ROOT if IN_COLAB else _DEFAULT_LOCAL_ROOT
)
LOCAL_ROOT = os.environ.get("SKRIPSI_LOCAL_ROOT", "/content/local_work" if IN_COLAB else os.path.join(_DEFAULT_LOCAL_ROOT, "_local_work"))

GITHUB_REPO_URL = "https://github.com/bestoism/corn-cl-indobert-rating-noise"

# ----------------------------------------------------------------------
# Penguncian korpus (Subbab 3.2 dan Batasan Masalah, butir pengendalian mutu)
# Nilai ini ditulis di kode, bukan dibaca dari manifest hasil run.
# ----------------------------------------------------------------------
EXPECTED_MASTER_SHA256 = "c589b92cbe2fbc34d67e43a0a866a281cb31603a190af4517ef7f157e54589f6"
EXPECTED_MASTER_ROWS = 16740

MIN_CORAL_PYTORCH_VERSION = "1.3.0"

# ----------------------------------------------------------------------
# Folder tingkat atas di Drive
# ----------------------------------------------------------------------
RAW_DIR = os.path.join(DRIVE_ROOT, "data", "raw")
MASTER_FILE = os.path.join(RAW_DIR, "all_reviews_master.csv")
LEXICON_DIR = os.path.join(DRIVE_ROOT, "lexicon")
SLANG_BASE_FILE = os.path.join(LEXICON_DIR, "slang_base.csv")
SLANG_DOMAIN_FILE = os.path.join(LEXICON_DIR, "slang_domain.csv")  # opsional

PRESERVED_DIR = os.path.join(DRIVE_ROOT, "preserved")
ANNOT_DIR = os.path.join(DRIVE_ROOT, "annotations")

SHARED_DIR = os.path.join(DRIVE_ROOT, "shared")
SHARED_PROCESSED_DIR = os.path.join(SHARED_DIR, "processed")
SHARED_RESULTS_DIR = os.path.join(SHARED_DIR, "results")
QUALITY_DIR = os.path.join(SHARED_RESULTS_DIR, "quality")
SENSITIVITY_SUMMARY_DIR = os.path.join(SHARED_RESULTS_DIR, "sensitivity_partisi")
LOGS_DIR = os.path.join(DRIVE_ROOT, "logs")
RUNS_DIR = os.path.join(DRIVE_ROOT, "runs")

CLEAN_TEXT_FILE = os.path.join(SHARED_PROCESSED_DIR, "reviews_clean.csv")

# Berkas yang dipertahankan (tidak dapat dibuat ulang dari raw dan lexicon)
PRESERVED_FILES = [
    os.path.join(PRESERVED_DIR, "scraping_summary.csv"),
    os.path.join(PRESERVED_DIR, "dataset_manifest_original.json"),
]
ANNOT_FILES = {
    "human_validation": os.path.join(ANNOT_DIR, "human_validation_sample.csv"),
    "human_validation_annotator2": os.path.join(ANNOT_DIR, "human_validation_sample_annotator2.csv"),
    "human_validation_retest": os.path.join(ANNOT_DIR, "human_validation_sample_retest.csv"),
    "gold": os.path.join(ANNOT_DIR, "gold_test_sample.csv"),
    "gold_annotator2": os.path.join(ANNOT_DIR, "gold_test_sample_annotator2.csv"),
}

# ----------------------------------------------------------------------
# Split partisi (Subbab 3.4)
# ----------------------------------------------------------------------
MAIN_SPLIT_SEED = 42
SPLIT_SEEDS = [42, 123, 2024]
TEST_SIZE = 0.20
VAL_SIZE = 0.10

# ----------------------------------------------------------------------
# Backbone dan hiperparameter (Tabel 3.3)
# ----------------------------------------------------------------------
PRETRAINED_MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
NUM_CLASSES = 5
EPOCHS = 10
PATIENCE = 3
DROPOUT = 0.3
WEIGHT_SEEDS = [42, 123, 2024]
WEIGHT_SEEDS_LIGHT = [42]  # opsi hemat, wajib dinyatakan sebagai keterbatasan

# Proxy classifier (Subbab 3.5, 3.9)
PROXY_CV_FOLDS = 5
PROXY_CV_SEED = 42
PROXY_FINETUNE_EPOCHS = 3
PROXY_FINETUNE_LR = 2e-5
PROXY_INNER_CALIB_FRACTION = 0.1
K_SENSITIVITY_VALUES = [3, 5, 10]

PROXY_REGISTRY = {
    0: {"name": "frozen_cls_lr", "label": "P1", "loss": None,
        "desc": "Embedding beku token [CLS] + Logistic Regression terkalibrasi"},
    1: {"name": "frozen_meanpool_lr", "label": "P2", "loss": None,
        "desc": "Embedding beku mean-pooling + Logistic Regression terkalibrasi"},
    2: {"name": "finetuned_ce", "label": "P3", "loss": "ce",
        "desc": "IndoBERT fine-tuned K-Fold, Cross-Entropy, temperature scaling"},
    3: {"name": "finetuned_corn", "label": "P4", "loss": "corn",
        "desc": "IndoBERT fine-tuned K-Fold, CORN, temperature scaling (proxy final)"},
}
FINAL_PROXY_ID = 3

# Confident learning (Subbab 3.6, 3.8)
CLEANLAB_FILTER_METHODS = ["confident_learning", "prune_by_noise_rate"]
MIN_EXAMPLES_PER_CLASS = 20
SEVERITY_THRESHOLD = 2

# Kalibrasi (Subbab 2.1.8, 3.6)
ECE_N_BINS = 10
PROB_FLOOR = 1e-8

# Validasi manusia (Subbab 3.7)
COCHRAN_Z = 1.96
COCHRAN_P = 0.5
COCHRAN_E = 0.10
HUMAN_VALIDATION_N = 100          # angka perencanaan (dibulatkan dari hasil Cochran)
MIN_PER_SEVERE_BIN = 15
SEVERE_BINS = (2, 3, 4)
SECOND_ANNOTATOR_FRACTION = 0.3
TEST_RETEST_FRACTION = 0.10
HUMAN_VERDICTS = ("noise", "not_noise", "ambiguous")

# Uji emas (Subbab 3.10)
GOLD_MARGIN_ERROR = 0.10
GOLD_SAMPLE_SEED = 42
GOLD_SECOND_ANNOTATOR_SEED = 99
UNDETERMINED_LABEL = "ND"

# Bootstrap (Subbab 3.12)
N_BOOT = 2000
BOOT_SEED = 42
ALPHA = 0.05

# ----------------------------------------------------------------------
# Enam skenario (Tabel 3.2) dan lima hipotesis (Tabel 3.4)
# ----------------------------------------------------------------------
SCENARIOS = [
    {"name": "M1_Baseline_CE", "variant": "raw", "loss": "ce"},
    {"name": "M2_CleanedHard_CE", "variant": "hard", "loss": "ce"},
    {"name": "M3_CleanedSevere_CE", "variant": "severe", "loss": "ce"},
    {"name": "M4_Baseline_CORN", "variant": "raw", "loss": "corn"},
    {"name": "M5_CleanedHard_CORN", "variant": "hard", "loss": "corn"},
    {"name": "M6_CleanedSevere_CORN", "variant": "severe", "loss": "corn"},
]
HYPOTHESES = [
    ("H1_CORN_vs_CE_raw", "M4_Baseline_CORN", "M1_Baseline_CE"),
    ("H2_SeverityAware_vs_Base", "M6_CleanedSevere_CORN", "M4_Baseline_CORN"),
    ("H3_HardPrune_vs_Base", "M5_CleanedHard_CORN", "M4_Baseline_CORN"),
    ("H4_SeverityAware_vs_Hard", "M6_CleanedSevere_CORN", "M5_CleanedHard_CORN"),
    ("H5_SeverityAware_vs_Base_CE", "M3_CleanedSevere_CE", "M1_Baseline_CE"),
]


class RunPaths:
    """Kumpulan path untuk satu split partisi. Semua folder dibuat sekali di awal."""

    SUBDIRS = ["data", "proxy", "ablation", "diagnostics", "noise", "cleaned",
               "training", "predictions", "significance", "validation", "gold",
               "k_sensitivity"]

    def __init__(self, split_seed):
        if split_seed not in SPLIT_SEEDS:
            raise ValueError(f"split_seed harus salah satu dari {SPLIT_SEEDS}, dapat {split_seed}")
        self.split_seed = split_seed
        self.is_main = split_seed == MAIN_SPLIT_SEED
        self.root = os.path.join(RUNS_DIR, f"split_seed{split_seed}")
        for sub in self.SUBDIRS:
            setattr(self, f"{sub}_dir", os.path.join(self.root, sub))
        self.local_dir = os.path.join(LOCAL_ROOT, f"split_seed{split_seed}")
        self.local_ckpt_dir = os.path.join(self.local_dir, "ckpt")
        self.local_cache_dir = os.path.join(self.local_dir, "cache")

    def all_dirs(self):
        return [self.root] + [getattr(self, f"{s}_dir") for s in self.SUBDIRS]

    def require_main(self, komponen):
        if not self.is_main:
            raise ValueError(
                f"[BERHENTI] Komponen '{komponen}' hanya boleh dijalankan pada split utama "
                f"(seed partisi {MAIN_SPLIT_SEED}), tetapi split aktif adalah seed {self.split_seed}. "
                f"Ablasi P1 sampai P3, sensitivitas K, validasi manusia, dan uji emas tidak diulang "
                f"per split karena anotasi manual tidak dapat diulang (Subbab 3.4)."
            )

    # berkas data
    def split_file(self, name):
        return os.path.join(self.data_dir, f"split_{name}.csv")

    def variant_file(self, variant):
        return os.path.join(self.cleaned_dir, f"train_{variant}.csv")

    def scenario_pred_file(self, scenario, seed):
        return os.path.join(self.predictions_dir, f"{scenario}__seed{seed}.csv")

    def scenario_metrics_file(self, scenario, seed):
        return os.path.join(self.training_dir, f"{scenario}__seed{seed}.json")

    def local_ckpt(self, scenario, seed):
        return os.path.join(self.local_ckpt_dir, f"{scenario}__seed{seed}_best.pt")


def all_shared_dirs():
    return [SHARED_DIR, SHARED_PROCESSED_DIR, SHARED_RESULTS_DIR, QUALITY_DIR,
            SENSITIVITY_SUMMARY_DIR, LOGS_DIR, RUNS_DIR, ANNOT_DIR, PRESERVED_DIR]


def ensure_all_dirs(split_seeds=None):
    """Membuat seluruh folder yang dibutuhkan, sekali di awal eksekusi."""
    split_seeds = split_seeds or SPLIT_SEEDS
    dirs = list(all_shared_dirs())
    for s in split_seeds:
        dirs += RunPaths(s).all_dirs()
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    for s in split_seeds:
        rp = RunPaths(s)
        os.makedirs(rp.local_ckpt_dir, exist_ok=True)
        os.makedirs(rp.local_cache_dir, exist_ok=True)
    return dirs


def scenario_by_name(name):
    for s in SCENARIOS:
        if s["name"] == name:
            return s
    raise KeyError(name)


def get_device():
    """Perangkat komputasi. Impor torch ditunda agar modul murni dapat diuji tanpa torch."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
