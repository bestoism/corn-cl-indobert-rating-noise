"""
proxy.py: proxy classifier P1 sampai P4 untuk Confident learning (Subbab 3.5, 3.6, 3.9).

Setiap fold P3 dan P4 menulis satu berkas npz unik (proxy_dir) sehingga eksekusi dapat
dilanjutkan bila runtime putus. Berkas fold hanya dipakai bila hash urutan review_id data
latih cocok dengan data saat ini.

CATATAN REPRODUSIBILITAS: torch.manual_seed(42) dipanggil satu kali di awal pemanggilan
get_oof_finetuned. Bila eksekusi dilanjutkan dari fold tengah, RNG dimulai ulang dari seed 42
pada fold pertama yang dihitung, sehingga hasil tidak identik bit per bit dengan eksekusi
tanpa putus. Nondeterminisme GPU (mixed precision) pada dasarnya juga mencegah identitas bit.
"""

import os
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import config, drive_io


def _ids_sha_ordered(df):
    return drive_io.ids_hash(df["review_id"].astype(str).tolist(), sort=False)


# ----------------------------------------------------------------------
# Konversi rantai CORN dan temperature scaling (memerlukan torch)
# ----------------------------------------------------------------------
def corn_chain(logits_t):
    """
    logits_t: tensor (n, K-1), sudah dibagi T. Mengembalikan:
      q_chain      : (n, K-1) kumulatif langsung rantai, P(y > k) = cumprod(sigmoid)
      class_probs  : (n, K) probabilitas kelas, dibatasi minimal 1e-8 lalu dinormalisasi ulang
    """
    import torch
    probs_cond = torch.sigmoid(logits_t)
    cum = torch.cumprod(probs_cond, dim=1)
    n, K = logits_t.shape[0], logits_t.shape[1] + 1
    cp = torch.zeros(n, K, device=logits_t.device, dtype=logits_t.dtype)
    cp[:, 0] = 1.0 - cum[:, 0]
    for k in range(1, K - 1):
        cp[:, k] = cum[:, k - 1] - cum[:, k]
    cp[:, K - 1] = cum[:, K - 2]
    cp = torch.clamp(cp, min=config.PROB_FLOOR)
    cp = cp / cp.sum(dim=1, keepdim=True)
    return cum, cp


def fit_temperature(logits, labels0, loss_type):
    """Temperature scaling (Guo dkk.) lewat LBFGS pada NLL. Fallback T = 1,0 bila gagal."""
    import torch
    import torch.nn.functional as F
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels0, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def nll(T):
        scaled = logits_t / torch.clamp(T, min=1e-2)
        if loss_type == "ce":
            log_probs = F.log_softmax(scaled, dim=1)
        else:
            _, probs = corn_chain(scaled)
            log_probs = torch.log(torch.clamp(probs, min=config.PROB_FLOOR))
        return F.nll_loss(log_probs, labels_t)

    def closure():
        optimizer.zero_grad()
        loss = nll(temperature)
        loss.backward()
        return loss

    try:
        before = nll(temperature).item()
        optimizer.step(closure)
        after = nll(temperature).item()
        T_final = torch.clamp(temperature.detach(), min=1e-2).item()
    except Exception as e:
        print(f"[PERINGATAN] Temperature scaling gagal konvergen ({e}); memakai T = 1,0.")
        T_final, before, after = 1.0, float("nan"), float("nan")
    return float(T_final), float(before), float(after)


# ----------------------------------------------------------------------
# P1 dan P2: embedding beku + Logistic Regression terkalibrasi
# ----------------------------------------------------------------------
def _get_embeddings(ctx, texts, ids_sha, pooling, batch_size):
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModel, AutoTokenizer

    cache = os.path.join(ctx.local_cache_dir, f"emb_{pooling}_{ids_sha[:12]}.npy")
    if os.path.exists(cache):
        return np.load(cache)

    device = config.get_device()
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    model = AutoModel.from_pretrained(config.PRETRAINED_MODEL_NAME).to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({pooling})", mininterval=20, leave=False):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                            max_length=config.MAX_LEN, return_tensors="pt").to(device)
            hs = model(**enc).last_hidden_state
            if pooling == "cls":
                emb = hs[:, 0, :]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).expand(hs.size()).float()
                emb = torch.sum(hs * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            out.extend(emb.cpu().numpy())
    arr = np.array(out)
    os.makedirs(ctx.local_cache_dir, exist_ok=True)
    np.save(cache, arr)
    del model
    torch.cuda.empty_cache()
    return arr


def _oof_frozen_lr(ctx, df_train, pooling):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    texts = df_train["cleaned_text"].tolist()
    labels0 = df_train["rating"].values - 1
    ids_sha = _ids_sha_ordered(df_train)
    X = _get_embeddings(ctx, texts, ids_sha, pooling, config.BATCH_SIZE)

    base = LogisticRegression(max_iter=2000, random_state=42)
    calibrated = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    splitter = StratifiedKFold(n_splits=config.PROXY_CV_FOLDS, shuffle=True, random_state=config.PROXY_CV_SEED)
    probs = cross_val_predict(calibrated, X, labels0, cv=splitter, method="predict_proba", n_jobs=-1)
    return probs.astype(np.float32)


# ----------------------------------------------------------------------
# P3 dan P4: fine-tuning K-Fold dengan temperature scaling per fold
# ----------------------------------------------------------------------
def _collect_logits(model, loader, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            out.append(logits.cpu().numpy())
    return np.concatenate(out, axis=0)


def _train_one_fold(texts_arr, labels_arr, train_idx, val_idx, loss_type, fold, K):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from coral_pytorch.losses import corn_loss
    from torch.utils.data import DataLoader

    from src.data import ReviewDataset
    from src.models import IndoBERTCORN, IndoBERTStandard

    device = config.get_device()
    inner_train_idx, inner_calib_idx = train_test_split(
        train_idx, test_size=config.PROXY_INNER_CALIB_FRACTION, random_state=config.PROXY_CV_SEED,
        stratify=labels_arr[train_idx])

    model = (IndoBERTCORN() if loss_type == "corn" else IndoBERTStandard()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.PROXY_FINETUNE_LR)
    criterion = nn.CrossEntropyLoss() if loss_type == "ce" else None

    def ds(idx):
        return ReviewDataset(texts_arr[idx].tolist(), (labels_arr[idx] + 1).tolist())

    train_loader = DataLoader(ds(inner_train_idx), batch_size=config.BATCH_SIZE, shuffle=True)
    calib_loader = DataLoader(ds(inner_calib_idx), batch_size=config.BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(ds(val_idx), batch_size=config.BATCH_SIZE, shuffle=False)

    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for epoch in range(config.PROXY_FINETUNE_EPOCHS):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)
            with torch.amp.autocast("cuda"):
                logits = model(input_ids, attention_mask)
                loss = (criterion(logits, lbl) if loss_type == "ce"
                        else corn_loss(logits, lbl, num_classes=config.NUM_CLASSES))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        print(f"      Epoch {epoch + 1}/{config.PROXY_FINETUNE_EPOCHS} - Loss: {epoch_loss / len(train_loader):.4f}")

    calib_logits = _collect_logits(model, calib_loader, device)
    calib_labels = labels_arr[inner_calib_idx]
    T, nll_before, nll_after = fit_temperature(calib_logits, calib_labels, loss_type)
    print(f"      Temperature scaling: T={T:.4f} (NLL {nll_before:.4f} -> {nll_after:.4f})")

    fold_logits = _collect_logits(model, val_loader, device)
    scaled = torch.tensor(fold_logits, dtype=torch.float32) / T
    with torch.no_grad():
        if loss_type == "ce":
            class_probs = F.softmax(scaled, dim=1).numpy()
            q_chain = None
        else:
            q, cp = corn_chain(scaled)
            class_probs, q_chain = cp.numpy(), q.numpy()

    del model, optimizer
    torch.cuda.empty_cache()
    return class_probs.astype(np.float32), (q_chain.astype(np.float32) if q_chain is not None else None), \
        T, nll_before, nll_after


def get_oof_finetuned(ctx, df_train, loss_type, K=None):
    """OOF probabilitas terkalibrasi untuk P3 (ce) atau P4 (corn). Lanjut dari berkas fold bila ada."""
    import torch
    K = K or config.PROXY_CV_FOLDS
    pid = 2 if loss_type == "ce" else 3
    name = config.PROXY_REGISTRY[pid]["name"]
    oof_path = os.path.join(ctx.proxy_dir, f"oof__{name}__K{K}.npz")
    ids_sha = _ids_sha_ordered(df_train)

    if drive_io.is_verified(oof_path):
        z = np.load(oof_path, allow_pickle=False)
        if str(z["ids_sha"]) == ids_sha:
            print(f"[OK] OOF {name} K={K} sudah ada, dimuat.")
            return {"class_probs": z["class_probs"],
                    "q_chain": z["q_chain"] if "q_chain" in z.files else None,
                    "temperatures": z["temperatures"]}
        raise RuntimeError(f"[BERHENTI] {oof_path} berasal dari data latih berbeda (hash tidak cocok).")

    torch.manual_seed(42)
    texts_arr = np.array(df_train["cleaned_text"].tolist(), dtype=object)
    labels_arr = df_train["rating"].values - 1
    n = len(texts_arr)
    class_probs = np.zeros((n, config.NUM_CLASSES), dtype=np.float32)
    q_chain = np.zeros((n, config.NUM_CLASSES - 1), dtype=np.float32) if loss_type == "corn" else None
    temps = []

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=config.PROXY_CV_SEED)
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        fold_path = os.path.join(ctx.proxy_dir, f"{name}__K{K}__fold{fold + 1}.npz")
        if drive_io.is_verified(fold_path):
            z = np.load(fold_path, allow_pickle=False)
            if str(z["ids_sha"]) != ids_sha or not np.array_equal(z["val_idx"], val_idx):
                raise RuntimeError(f"[BERHENTI] Berkas fold tidak cocok dengan data saat ini: {fold_path}")
            print(f"   [lewati] {name} K={K} fold {fold + 1}/{K} sudah terverifikasi.")
            cp, qc, T = z["class_probs"], (z["q_chain"] if "q_chain" in z.files else None), float(z["T"])
        else:
            print(f"   [{name}] K={K} fold {fold + 1}/{K} (train={len(train_idx)}, val={len(val_idx)})...")
            t0 = time.time()
            cp, qc, T, nb, na = _train_one_fold(texts_arr, labels_arr, train_idx, val_idx, loss_type, fold, K)
            arrays = dict(val_idx=val_idx, class_probs=cp, T=np.array(T), nll_before=np.array(nb),
                          nll_after=np.array(na), ids_sha=np.array(ids_sha),
                          detik=np.array(time.time() - t0))
            if qc is not None:
                arrays["q_chain"] = qc
            drive_io.write_once_npz(fold_path, **arrays)
        class_probs[val_idx] = cp
        if q_chain is not None:
            q_chain[val_idx] = qc
        temps.append(T)

    print(f"   Temperature per fold: {[f'{t:.3f}' for t in temps]} (rata-rata {np.mean(temps):.3f})")
    arrays = dict(class_probs=class_probs, temperatures=np.array(temps), ids_sha=np.array(ids_sha))
    if q_chain is not None:
        arrays["q_chain"] = q_chain
    drive_io.write_once_npz(oof_path, **arrays)
    return {"class_probs": class_probs, "q_chain": q_chain, "temperatures": np.array(temps)}


def get_oof(ctx, df_train, proxy_id, K=None):
    """Satu-satunya titik masuk dari ablation.py. Mengembalikan dict class_probs, q_chain, temperatures."""
    K = K or config.PROXY_CV_FOLDS
    if proxy_id in (0, 1):
        if K != config.PROXY_CV_FOLDS:
            raise ValueError("Sensitivitas K hanya untuk proxy P4.")
        name = config.PROXY_REGISTRY[proxy_id]["name"]
        oof_path = os.path.join(ctx.proxy_dir, f"oof__{name}__K{K}.npz")
        ids_sha = _ids_sha_ordered(df_train)
        if drive_io.is_verified(oof_path):
            z = np.load(oof_path, allow_pickle=False)
            if str(z["ids_sha"]) != ids_sha:
                raise RuntimeError(f"[BERHENTI] {oof_path} berasal dari data latih berbeda.")
            return {"class_probs": z["class_probs"], "q_chain": None, "temperatures": None}
        probs = _oof_frozen_lr(ctx, df_train, "cls" if proxy_id == 0 else "mean")
        drive_io.write_once_npz(oof_path, class_probs=probs, ids_sha=np.array(ids_sha))
        return {"class_probs": probs, "q_chain": None, "temperatures": None}
    if proxy_id == 2:
        if K != config.PROXY_CV_FOLDS:
            raise ValueError("Sensitivitas K hanya untuk proxy P4.")
        return get_oof_finetuned(ctx, df_train, "ce", K)
    if proxy_id == 3:
        return get_oof_finetuned(ctx, df_train, "corn", K)
    raise ValueError(f"proxy_id tidak dikenal: {proxy_id} (harus 0 sampai 3)")
