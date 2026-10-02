"""
proxy.py -- Proxy classifier P1-P4 untuk Confident Learning. Backbone tunggal IndoBERT.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from tqdm import tqdm

from src import config
from src.data import ReviewDataset
from src.models import IndoBERTStandard, IndoBERTCORN
from coral_pytorch.losses import corn_loss


# ---------------- cache helper ----------------
def _tagged(path, tag):
    if not tag:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}__{tag}{ext}"


def _load_cache_if_valid(cache_file, meta_file, texts):
    if os.path.exists(cache_file) and os.path.exists(meta_file):
        meta = pd.read_csv(meta_file)
        if meta["text"].tolist() == list(texts):
            print(f"⚡ Memuat cache proxy [{config.PROXY_NAME}] ({os.path.basename(cache_file)}) ...")
            return np.load(cache_file)
        print(f"⚠️ Cache proxy [{config.PROXY_NAME}] tidak cocok dgn data saat ini -> recompute.")
    return None


def _save_cache(cache_file, meta_file, texts, array):
    np.save(cache_file, array)
    pd.DataFrame({"text": texts}).to_csv(meta_file, index=False)


# ---------------- embedding beku (P1, P2) ----------------
# CATATAN: padding dinamis per-batch di sini vs padding="max_length" pada ReviewDataset (P3/P4)
# -> jalur tokenisasi P1/P2 vs P3/P4 tidak identik; sebutkan sebagai confound minor di Bab IV.
def _get_embeddings(texts, pooling="mean", batch_size=32):
    cache_file = config.EMBEDDING_CACHE_FILE_BASE.replace(".npy", f"_{pooling}.npy")
    meta_file = config.EMBEDDING_CACHE_META_FILE_BASE.replace(".csv", f"_{pooling}.csv")

    cached = _load_cache_if_valid(cache_file, meta_file, texts)
    if cached is not None:
        return cached

    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    model = AutoModel.from_pretrained(config.PRETRAINED_MODEL_NAME).to(config.DEVICE)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({pooling})"):
            batch_texts = texts[i: i + batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True,
                                max_length=config.MAX_LEN, return_tensors="pt").to(config.DEVICE)
            outputs = model(**encoded)
            if pooling == "cls":
                emb = outputs.last_hidden_state[:, 0, :]
            else:
                token_emb = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
                emb = torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings.extend(emb.cpu().numpy())

    embeddings = np.array(embeddings)
    _save_cache(cache_file, meta_file, texts, embeddings)
    return embeddings


def _proxy_frozen_lr(texts, labels, pooling):
    """StratifiedKFold(shuffle=True, random_state=42) eksplisit -- identik dengan P3/P4."""
    X = _get_embeddings(texts, pooling=pooling, batch_size=config.BATCH_SIZE)
    base_clf = LogisticRegression(max_iter=2000, random_state=42)
    calibrated = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")
    cv_splitter = StratifiedKFold(n_splits=config.PROXY_CV_FOLDS, shuffle=True, random_state=42)
    return cross_val_predict(calibrated, X, labels, cv=cv_splitter, method="predict_proba", n_jobs=-1)


# ---------------- konversi CORN -> probabilitas kelas (§3.6 butir 3) ----------------
def _corn_logits_to_probas(logits):
    probs_cond = torch.sigmoid(logits)
    cum_probs = torch.cumprod(probs_cond, dim=1)
    batch_size, K = logits.shape[0], logits.shape[1] + 1
    class_probs = torch.zeros(batch_size, K, device=logits.device, dtype=logits.dtype)
    class_probs[:, 0] = 1.0 - cum_probs[:, 0]
    for k in range(1, K - 1):
        class_probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
    class_probs[:, K - 1] = cum_probs[:, K - 2]
    class_probs = torch.clamp(class_probs, min=1e-8)
    return class_probs / class_probs.sum(dim=1, keepdim=True)


# ---------------- temperature scaling (§2.1.8, §3.6 butir 2) ----------------
def _calibrate_with_temperature(oof_logits, labels_arr, loss_type):
    """
    Satu skalar T (LBFGS, minimalkan NLL). CE: softmax(z/T). CORN: z/T sebelum sigmoid, satu T
    dibagikan ke K-1 logit. Fallback T=1.0 jika optimasi gagal.
    """
    logits_t = torch.tensor(oof_logits, dtype=torch.float32)
    labels_t = torch.tensor(labels_arr, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def _nll_loss(T):
        scaled = logits_t / torch.clamp(T, min=1e-2)
        if loss_type == "ce":
            log_probs = F.log_softmax(scaled, dim=1)
        else:
            log_probs = torch.log(torch.clamp(_corn_logits_to_probas(scaled), min=1e-8))
        return F.nll_loss(log_probs, labels_t)

    def closure():
        optimizer.zero_grad()
        loss = _nll_loss(temperature)
        loss.backward()
        return loss

    try:
        nll_before = _nll_loss(temperature).item()
        optimizer.step(closure)
        nll_after = _nll_loss(temperature).item()
        T_final = torch.clamp(temperature.detach(), min=1e-2).item()
        print(f"   🌡️  Temperature scaling [{config.PROXY_NAME}]: T={T_final:.4f} (NLL {nll_before:.4f} -> {nll_after:.4f})")
    except Exception as e:
        print(f"   ⚠️ Temperature scaling gagal konvergen ({e}) -- fallback T=1.0.")
        T_final = 1.0

    with torch.no_grad():
        scaled = logits_t / T_final
        probs = F.softmax(scaled, dim=1) if loss_type == "ce" else _corn_logits_to_probas(scaled)
    return probs.numpy(), T_final


# ---------------- fine-tune K-Fold generik (P3, P4) ----------------
# Tiap fold: 3 epoch tetap, tanpa early stopping/scheduler (beda dari train.py) -- batasan metodologis.
def _finetune_kfold_oof(texts, labels, loss_type, n_folds=None, cache_tag=""):
    n_folds = n_folds or config.PROXY_CV_FOLDS
    cache_file = _tagged(config.PROXY_PRED_PROBS_FILE, cache_tag)
    meta_file = _tagged(config.PROXY_PRED_PROBS_META_FILE, cache_tag)
    cached = _load_cache_if_valid(cache_file, meta_file, texts)
    if cached is not None:
        return cached

    torch.manual_seed(42)
    n = len(texts)
    oof_probs = np.zeros((n, config.NUM_CLASSES), dtype=np.float32)
    texts_arr = np.array(texts, dtype=object)
    labels_arr = np.array(labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_temperatures = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        print(f"   [Proxy {config.PROXY_NAME}{(' ' + cache_tag) if cache_tag else ''}] "
              f"Fold {fold + 1}/{n_folds} (train={len(train_idx)}, val={len(val_idx)})...")

        # validasi INTERNAL dalam bagian latih fold -> khusus mencari T (§3.4, §3.6)
        inner_train_idx, inner_calib_idx = train_test_split(
            train_idx, test_size=0.1, random_state=42, stratify=labels_arr[train_idx])

        model = (IndoBERTCORN() if loss_type == "corn" else IndoBERTStandard()).to(config.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.PROXY_FINETUNE_LR)
        criterion = nn.CrossEntropyLoss() if loss_type == "ce" else None

        train_ds = ReviewDataset(texts_arr[inner_train_idx].tolist(), (labels_arr[inner_train_idx] + 1).tolist())
        calib_ds = ReviewDataset(texts_arr[inner_calib_idx].tolist(), (labels_arr[inner_calib_idx] + 1).tolist())
        val_ds = ReviewDataset(texts_arr[val_idx].tolist(), (labels_arr[val_idx] + 1).tolist())

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        calib_loader = DataLoader(calib_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

        scaler = torch.cuda.amp.GradScaler()
        model.train()
        for epoch in range(config.PROXY_FINETUNE_EPOCHS):
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                lbl = batch["labels"].to(config.DEVICE)
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    loss = (criterion(logits, lbl) if loss_type == "ce"
                            else corn_loss(logits, lbl, num_classes=config.NUM_CLASSES))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            print(f"      Epoch {epoch+1}/{config.PROXY_FINETUNE_EPOCHS} - Loss: {epoch_loss/len(train_loader):.4f}")

        model.eval()

        calib_logits, calib_labels = [], []
        with torch.no_grad():
            for batch in calib_loader:
                logits = model(batch["input_ids"].to(config.DEVICE), batch["attention_mask"].to(config.DEVICE))
                calib_logits.append(logits.cpu().numpy())
                calib_labels.append(batch["labels"].numpy())
        calib_logits = np.concatenate(calib_logits, axis=0)
        calib_labels = np.concatenate(calib_labels, axis=0)

        _dummy, T_fold = _calibrate_with_temperature(calib_logits, calib_labels, loss_type)
        fold_temperatures.append(T_fold)

        fold_logits = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["input_ids"].to(config.DEVICE), batch["attention_mask"].to(config.DEVICE))
                fold_logits.append(logits.cpu().numpy())
        fold_logits = np.concatenate(fold_logits, axis=0)

        logits_t = torch.tensor(fold_logits, dtype=torch.float32) / T_fold
        with torch.no_grad():
            probs = (torch.nn.functional.softmax(logits_t, dim=1) if loss_type == "ce"
                     else _corn_logits_to_probas(logits_t))
        oof_probs[val_idx] = probs.numpy()

        del model, optimizer
        torch.cuda.empty_cache()

    print(f"   🌡️  Temperature per fold [{config.PROXY_NAME}]: {[f'{t:.3f}' for t in fold_temperatures]} "
          f"(mean={np.mean(fold_temperatures):.3f})")
    suffix = f"__{cache_tag}" if cache_tag else ""
    pd.DataFrame({"fold": range(1, n_folds + 1), "temperature": fold_temperatures}).to_csv(
        os.path.join(config.RESULTS_DIR, f"temperature_per_fold__{config.PROXY_NAME}{suffix}.csv"), index=False)

    _save_cache(cache_file, meta_file, texts, oof_probs)
    return oof_probs


# ---------------- dispatcher ----------------
def get_proxy_pred_probs(texts, labels):
    """OOF pred_probs (terkalibrasi untuk P3/P4) sesuai config.PROXY_ID (0-3)."""
    print(f"\n🧮 Menghitung OOF pred_probs — proxy [{config.PROXY_ID}] {config.PROXY_NAME}")
    if config.PROXY_ID == 0:
        return _proxy_frozen_lr(texts, labels, pooling="cls")
    elif config.PROXY_ID == 1:
        return _proxy_frozen_lr(texts, labels, pooling="mean")
    elif config.PROXY_ID == 2:
        return _finetune_kfold_oof(texts, labels, loss_type="ce")
    elif config.PROXY_ID == 3:
        return _finetune_kfold_oof(texts, labels, loss_type="corn")
    raise ValueError(f"PROXY_ID tidak dikenal: {config.PROXY_ID} (harus 0-3, Tabel 3.1).")