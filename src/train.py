"""
train.py: pelatihan enam skenario M1 sampai M6 dengan tiga seed bobot (Subbab 3.9, Tabel 3.3).

Unit kerja = satu (skenario, seed). Setiap unit menulis tepat dua berkas unik ke Drive:
  predictions/{skenario}__seed{seed}.csv  (review_id, y_true, y_pred pada data uji)
  training/{skenario}__seed{seed}.json    (metrik uji, jejak epoch, informasi checkpoint)
Checkpoint (sekitar 500 MB) ditulis ke disk lokal Colab. Uji signifikansi dan uji emas bekerja
dari berkas prediksi, tanpa memuat ulang checkpoint.
"""

import os
import shutil
import time

import numpy as np
import pandas as pd

from src import config, drive_io
from src.metrics import compute_metrics


def set_seed(seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _predict(model, loader, loss_type):
    import torch
    from coral_pytorch.dataset import corn_label_from_logits
    device = config.get_device()
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            p = (torch.argmax(logits, dim=1) if loss_type == "ce"
                 else corn_label_from_logits(logits)).cpu().numpy()
            preds.extend(p)
            labels.extend(batch["labels"].numpy())
    return np.array(preds), np.array(labels)


def unit_done(ctx, scenario, seed):
    return (drive_io.is_verified(ctx.scenario_pred_file(scenario, seed))
            and drive_io.is_verified(ctx.scenario_metrics_file(scenario, seed)))


def run_experiment(ctx, scenario, seed, copy_ckpt_to_drive=False):
    import torch
    import torch.nn as nn
    from coral_pytorch.losses import corn_loss
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from src.data import ReviewDataset
    from src.models import build_model

    torch.backends.cudnn.benchmark = True
    name, loss_type = scenario["name"], scenario["loss"]
    device = config.get_device()
    print(f"\n[MULAI] {name} | split {ctx.split_seed} | seed bobot {seed} | loss {loss_type.upper()}")
    t0 = time.time()
    set_seed(seed)

    df_train = drive_io.read_csv_verified(ctx.variant_file(scenario["variant"]))
    df_val = drive_io.read_csv_verified(ctx.split_file("val"))
    df_test = drive_io.read_csv_verified(ctx.split_file("test"))

    def loader(df, shuffle):
        return DataLoader(ReviewDataset(df["cleaned_text"].tolist(), df["rating"].tolist()),
                          batch_size=config.BATCH_SIZE, shuffle=shuffle)

    train_loader, val_loader, test_loader = loader(df_train, True), loader(df_val, False), loader(df_test, False)

    model = build_model(loss_type)
    criterion = nn.CrossEntropyLoss() if loss_type == "ce" else None
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(config.WARMUP_FRACTION * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda")

    best_val_mae = float("inf")          # hanya untuk early stopping
    best_qwk = -float("inf")             # kriteria checkpoint utama
    best_qwk_tiebreak_mae = float("inf")
    best_state, best_epoch = None, None
    patience_counter = 0
    epoch_log = []
    stopped_early = False

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda"):
                logits = model(input_ids, attention_mask)
                loss = (criterion(logits, labels) if loss_type == "ce"
                        else corn_loss(logits, labels, num_classes=config.NUM_CLASSES))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()

        vp, vl = _predict(model, val_loader, loss_type)
        vm = compute_metrics(vl, vp)
        val_mae, val_qwk = vm["mae"], vm["qwk"]
        mean_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{config.EPOCHS} - Loss: {mean_loss:.4f} - Val MAE: {val_mae:.4f} | "
              f"QWK: {val_qwk:.4f} | Off-by-1: {vm['off_by_one']:.4f} | Acc: {vm['accuracy']:.4f}")
        epoch_log.append({"epoch": epoch + 1, "train_loss": mean_loss, **{f"val_{k}": v for k, v in vm.items()}})

        improved = (val_qwk > best_qwk) or (val_qwk == best_qwk and val_mae < best_qwk_tiebreak_mae)
        if improved:
            best_qwk, best_qwk_tiebreak_mae, best_epoch = val_qwk, val_mae, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"   Early stopping di epoch {epoch + 1} (MAE validasi tidak membaik {config.PATIENCE} kali)")
                stopped_early = True
                break

    model.load_state_dict(best_state)

    os.makedirs(ctx.local_ckpt_dir, exist_ok=True)
    ckpt_local = ctx.local_ckpt(name, seed)
    torch.save(best_state, ckpt_local)
    ckpt_info = {"checkpoint_lokal": ckpt_local, "checkpoint_bytes": os.path.getsize(ckpt_local),
                 "disalin_ke_drive": False}
    if copy_ckpt_to_drive:
        dest = os.path.join(ctx.training_dir, f"{name}__seed{seed}_best.pt")
        shutil.copyfile(ckpt_local, dest)
        if os.path.getsize(dest) != os.path.getsize(ckpt_local):
            raise drive_io.DriveWriteError(f"[BERHENTI] Ukuran checkpoint di Drive tidak sama: {dest}")
        ckpt_info.update({"disalin_ke_drive": True, "checkpoint_drive": dest})

    tp, tl = _predict(model, test_loader, loss_type)
    test_metrics = compute_metrics(tl, tp)
    y_true = df_test["rating"].values
    y_pred = tp + 1
    if not np.array_equal(tl + 1, y_true):
        raise RuntimeError("[BERHENTI] Urutan label uji tidak selaras dengan data uji.")

    print(f"[SELESAI] Uji (checkpoint epoch {best_epoch}, QWK val {best_qwk:.4f}): "
          f"MAE={test_metrics['mae']:.4f} | RMSE={test_metrics['rmse']:.4f} | Acc={test_metrics['accuracy']:.4f} | "
          f"Off-by-1={test_metrics['off_by_one']:.4f} | QWK={test_metrics['qwk']:.4f}")

    pred_df = pd.DataFrame({"review_id": df_test["review_id"].values, "y_true": y_true, "y_pred": y_pred})
    drive_io.write_once_csv(pred_df, ctx.scenario_pred_file(name, seed))
    record = {
        "split_seed": ctx.split_seed, "scenario": name, "weight_seed": seed, "loss": loss_type,
        "variant": scenario["variant"], "n_train": len(df_train), "n_val": len(df_val),
        "n_test": len(df_test), "epochs_dijalankan": len(epoch_log), "epoch_checkpoint_terbaik": best_epoch,
        "early_stopping_aktif": stopped_early, "best_val_qwk": best_qwk,
        "best_val_mae_tiebreak": best_qwk_tiebreak_mae, "test": test_metrics,
        "jejak_epoch": epoch_log, "detik": round(time.time() - t0, 1), **ckpt_info,
    }
    drive_io.write_once_json(record, ctx.scenario_metrics_file(name, seed))

    del model, optimizer, best_state
    torch.cuda.empty_cache()
    return record


def run_all(ctx, seeds, copy_ckpt_to_drive=False):
    """Menjalankan seluruh unit kerja yang belum selesai (lanjut otomatis bila runtime putus)."""
    total = len(config.SCENARIOS) * len(seeds)
    done = 0
    for scenario in config.SCENARIOS:
        for seed in seeds:
            if unit_done(ctx, scenario["name"], seed):
                done += 1
                print(f"[lewati] {scenario['name']} | seed {seed} (terverifikasi)")
                continue
            run_experiment(ctx, scenario, seed, copy_ckpt_to_drive=copy_ckpt_to_drive)
            done += 1
            print(f"   Progres sesi pelatihan: {done}/{total}")
    return done


def completed_units(ctx, seeds):
    rows = []
    for s in config.SCENARIOS:
        for seed in seeds:
            rows.append({"scenario": s["name"], "weight_seed": seed, "selesai": unit_done(ctx, s["name"], seed)})
    return pd.DataFrame(rows)


def build_final_results_table(ctx, seeds):
    """Tabel rata-rata dan simpangan baku (populasi, ddof=0) dari berkas metrik unik per unit."""
    import json
    rows = []
    for s in config.SCENARIOS:
        vals = {k: [] for k in ("mae", "rmse", "accuracy", "off_by_one", "qwk")}
        for seed in seeds:
            with open(ctx.scenario_metrics_file(s["name"], seed), "r", encoding="utf-8") as f:
                rec = json.load(f)
            for k in vals:
                vals[k].append(rec["test"][k])
        row = {"Model": s["name"], "n_seed": len(seeds)}
        for k, label in (("mae", "MAE"), ("rmse", "RMSE"), ("accuracy", "Acc"),
                         ("off_by_one", "Off-by-1"), ("qwk", "QWK")):
            row[f"{label}_mean"] = float(np.mean(vals[k]))
            row[f"{label}_std"] = float(np.std(vals[k]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)


def run_scenario(ctx, scenario_name, seeds, copy_ckpt_to_drive=False):
    """Menjalankan semua seed untuk satu skenario (melewati unit yang sudah terverifikasi)."""
    scenario = config.scenario_by_name(scenario_name)
    for seed in seeds:
        if unit_done(ctx, scenario_name, seed):
            print(f"[lewati] {scenario_name} | seed {seed} (terverifikasi)")
            continue
        run_experiment(ctx, scenario, seed, copy_ckpt_to_drive=copy_ckpt_to_drive)
    return completed_units(ctx, seeds)
