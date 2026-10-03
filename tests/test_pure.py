"""
Uji fungsi murni dan alur penyimpanan dengan data SINTETIS (tanpa torch, tanpa GPU, tanpa HF Hub).
Menjalankan: python tests/test_pure.py
Yang TIDAK diuji di sini: fine-tuning IndoBERT, tokenizer, temperature scaling LBFGS, dan embedding beku
(membutuhkan torch, transformers, GPU, dan unduhan model).
"""

import json
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="skripsi_test_")
os.environ["SKRIPSI_DRIVE_ROOT"] = os.path.join(TMP, "drive")
os.environ["SKRIPSI_LOCAL_ROOT"] = os.path.join(TMP, "local")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src import (annotation, calibration, clean, config, consistency, data_split, diagnostics,
                 drive_io, gold_test, human_validation, metrics, pipeline, preprocess,
                 quality_report, sampling, selfcheck, sensitivity_k, significance, summary)

PASS = []


def ok(name):
    PASS.append(name)
    print(f"[LULUS] {name}")


rng = np.random.default_rng(0)
os.makedirs(config.DRIVE_ROOT, exist_ok=True)
config.ensure_all_dirs()


# ---------------------------------------------------------------- drive_io
def test_drive_io():
    p = os.path.join(config.LOGS_DIR, "a.csv")
    df = pd.DataFrame({"x": [1, 2, 3]})
    r = drive_io.write_once_csv(df, p)
    assert r["status"] == "ditulis"
    assert drive_io.write_once_csv(df, p)["status"] == "sudah_ada_identik"
    try:
        drive_io.write_once_csv(pd.DataFrame({"x": [9]}), p)
        raise AssertionError("seharusnya menolak menimpa")
    except drive_io.DriveWriteError:
        pass
    assert drive_io.is_verified(p, expected_rows=3)
    assert not drive_io.is_verified(p, expected_rows=4)
    pz = os.path.join(config.LOGS_DIR, "b.npz")
    drive_io.write_once_npz(pz, a=np.arange(5), s=np.array("abc"))
    assert drive_io.write_once_npz(pz, a=np.arange(5), s=np.array("abc"))["status"] == "sudah_ada_identik"
    try:
        drive_io.write_once_npz(pz, a=np.arange(6), s=np.array("abc"))
        raise AssertionError
    except drive_io.DriveWriteError:
        pass
    # deteksi salinan ganda di samping berkas baru
    folder = config.LOGS_DIR
    open(os.path.join(folder, "c (1).csv"), "w").write("x\n1\n")
    try:
        drive_io.write_once_csv(df, os.path.join(folder, "c.csv"))
        raise AssertionError
    except drive_io.DriveWriteError:
        pass
    os.remove(os.path.join(folder, "c (1).csv"))
    os.remove(os.path.join(folder, "c.csv"))
    assert drive_io.is_duplicate_name("M4_Baseline_CORN (1)")
    assert drive_io.is_duplicate_name("Salinan dari x.csv")
    assert not drive_io.is_duplicate_name("M4_Baseline_CORN__seed42.json")
    ok("drive_io: tulis sekali, tolak timpa, deteksi duplikat")


def test_audit():
    # keadaan awal: logs berisi berkas -> TIDAK DIHARAPKAN pada mode awal, boleh pada mode berjalan
    try:
        drive_io.audit_drive_state(mode="awal", verbose=False)
        raise AssertionError("audit awal seharusnya gagal karena logs berisi berkas")
    except drive_io.DriveAuditError:
        pass
    df, dups = drive_io.audit_drive_state(mode="berjalan", verbose=False)
    assert dups == []
    os.makedirs(os.path.join(config.LEXICON_DIR), exist_ok=True)
    assert drive_io.classify_path("lexicon/slang_base.csv", "awal") == drive_io.DIPERTAHANKAN
    assert drive_io.classify_path("data/raw/all_reviews_master.csv", "awal") == drive_io.DIPERTAHANKAN
    assert drive_io.classify_path("runs/split_seed42/x.csv", "awal") == drive_io.TIDAK_DIHARAPKAN
    assert drive_io.classify_path("runs/split_seed42/x.csv", "berjalan") == drive_io.DIPERBOLEHKAN
    # folder skenario ganda
    os.makedirs(os.path.join(config.RUNS_DIR, "M4_Baseline_CORN (1)"), exist_ok=True)
    try:
        drive_io.audit_drive_state(mode="berjalan", verbose=False)
        raise AssertionError
    except drive_io.DriveAuditError:
        pass
    os.rmdir(os.path.join(config.RUNS_DIR, "M4_Baseline_CORN (1)"))
    t = drive_io.bersihkan_artefak(dry_run=True)
    assert len(t) > 0
    ok("audit_drive_state, klasifikasi, dry run pembersihan")


# ---------------------------------------------------------------- metrik dan kalibrasi
def test_qwk_fast():
    y = rng.integers(1, 6, 2000)
    p = np.clip(y + rng.integers(-2, 3, 2000), 1, 5)
    assert abs(metrics.qwk_fast(y, p) - cohen_kappa_score(y, p, weights="quadratic")) < 1e-12
    assert abs(metrics.qwk_fast(y - 1, p - 1, offset=0) - metrics.qwk_fast(y, p)) < 1e-12
    ok("qwk_fast setara sklearn")


def test_calibration():
    n = 5000
    labels0 = rng.integers(0, 5, n)
    # kumulatif sempurna terkalibrasi: q_k = prob nyata 1 atau 0 (ECE nol)
    q_perfect = np.stack([(labels0 > k).astype(float) for k in range(4)], axis=1)
    m, each = calibration.ece_cumulative(q_perfect, labels0)
    assert m < 1e-12
    # ECE tidak sama untuk top-label vs kumulatif: dipanggil berdasarkan nama
    logits = rng.normal(size=(n, 4))
    cum = np.cumprod(1 / (1 + np.exp(-logits)), axis=1)
    cp = np.zeros((n, 5))
    cp[:, 0] = 1 - cum[:, 0]
    for k in range(1, 4):
        cp[:, k] = cum[:, k - 1] - cum[:, k]
    cp[:, 4] = cum[:, 3]
    cp = np.clip(cp, 1e-8, None)
    cp = cp / cp.sum(axis=1, keepdims=True)
    rep = calibration.calibration_report(cp, labels0, q_chain=cum)
    assert rep["monotonisitas_rantai"]["proporsi_monoton"] == 1.0
    assert rep["monotonisitas_turunan"]["proporsi_monoton"] == 1.0
    assert rep["selisih_rantai_vs_turunan"]["selisih_maks"] < 1e-6
    assert rep["ece_top_label"] != rep["ece_kumulatif_rantai_rata"]
    flat = calibration.flatten_report(rep)
    assert "ece_top_label" in flat and "ece_kumulatif_rantai_rata" in flat
    # q tidak monoton harus terdeteksi
    q_bad = cum.copy()
    q_bad[0, 2] = q_bad[0, 1] + 0.1
    assert calibration.monotonicity_report(q_bad)["n_baris_melanggar"] == 1
    # kumulatif turunan dari distribusi seragam
    uni = np.full((10, 5), 0.2)
    assert np.allclose(calibration.derived_cumulative(uni)[0], [0.8, 0.6, 0.4, 0.2])
    pc = calibration.per_class_quality(labels0, labels0)
    assert (pc["recall"] == 1).all() and (pc["precision"] == 1).all()
    ok("kalibrasi: ECE, monotonisitas, selisih rantai vs turunan, per kelas")


# ---------------------------------------------------------------- sampling
def test_sampling():
    c = sampling.cochran_n(2856, e=0.10)
    assert abs(c["n0"] - 96.04) < 0.01 and c["n_cochran"] == 93, c
    big = sampling.cochran_n(10 ** 6)
    assert big["n_cochran"] == 97
    plan = sampling.allocation_plan({1: 3000, 2: 1000, 3: 300, 4: 100}, n_total=100)
    assert plan["take"] == {2: 15, 3: 15, 4: 15, 1: 55} and plan["total"] == 100
    plan = sampling.allocation_plan({1: 3000, 2: 1000, 3: 300, 4: 7}, n_total=100)
    assert plan["take"][4] == 7 and plan["take"][1] == 63 and 4 in plan["shortfall"]
    df = pd.DataFrame({"review_id": np.arange(5000).astype(str), "rating_diff": rng.choice([1, 2, 3, 4], 5000, p=[.6, .25, .1, .05])})
    s, pl = sampling.stratified_human_sample(df, n=100, seed=42)
    vc = s["rating_diff"].value_counts().to_dict()
    assert len(s) == 100 and vc[2] == 15 and vc[3] == 15 and vc[4] == 15 and vc[1] == 55
    ok("Cochran (93 untuk N=2856, 97 untuk N besar), alokasi 15/15/15/55")


# ---------------------------------------------------------------- konsistensi
def _long(rows):
    return pd.DataFrame(rows, columns=["split_seed", "hypothesis", "metric", "diff", "ci_low", "ci_high"])


def test_consistency():
    # MAE: negatif = A lebih baik
    r = consistency.evaluate_consistency(_long([
        (42, "H1", "MAE", -0.06, -0.08, -0.04), (123, "H1", "MAE", -0.05, -0.07, -0.03),
        (2024, "H1", "MAE", -0.04, -0.09, 0.01)]))
    assert r.iloc[0]["putusan"] == "KONSISTEN" and r.iloc[0]["arah_dominan"] == "A lebih baik"
    # CI melewati nol pada dua split
    r = consistency.evaluate_consistency(_long([
        (42, "H4", "MAE", -0.001, -0.014, 0.013), (123, "H4", "MAE", -0.002, -0.01, 0.01),
        (2024, "H4", "MAE", -0.03, -0.05, -0.01)]))
    assert r.iloc[0]["putusan"] == "TIDAK KONSISTEN"
    # arah berbalik walau dua CI tidak melewati nol
    r = consistency.evaluate_consistency(_long([
        (42, "H2", "MAE", 0.02, 0.01, 0.03), (123, "H2", "MAE", 0.02, 0.01, 0.03),
        (2024, "H2", "MAE", -0.02, -0.03, -0.01)]))
    assert r.iloc[0]["putusan"] == "TIDAK KONSISTEN"
    # QWK: positif = A lebih baik. Selisih QWK negatif signifikan -> B lebih baik, tetap konsisten
    r = consistency.evaluate_consistency(_long([
        (42, "H2", "QWK_ensemble", -0.01, -0.02, -0.001), (123, "H2", "QWK_ensemble", -0.012, -0.02, -0.003),
        (2024, "H2", "QWK_ensemble", -0.009, -0.02, 0.002)]))
    assert r.iloc[0]["putusan"] == "KONSISTEN" and r.iloc[0]["arah_dominan"] == "B lebih baik"
    assert consistency.direction_label("MAE", -0.1) == "A lebih baik"
    assert consistency.direction_label("QWK_ensemble", -0.1) == "B lebih baik"
    r = consistency.evaluate_consistency(_long([(42, "H1", "MAE", -0.1, -0.2, -0.1), (123, "H1", "MAE", -0.1, -0.2, -0.1)]))
    assert r.iloc[0]["putusan"].startswith("TIDAK DAPAT DINILAI")
    ok("aturan konsistensi (arah MAE vs QWK, kasus batas)")


# ---------------------------------------------------------------- Confident learning
def synthetic_probs(n, noise=0.25):
    labels0 = rng.integers(0, 5, n)
    probs = np.full((n, 5), 0.02)
    pred = labels0.copy()
    flip = rng.random(n) < noise
    pred[flip] = np.clip(labels0[flip] + rng.choice([-2, -1, 1, 2], flip.sum()), 0, 4)
    probs[np.arange(n), pred] = 0.92
    probs = probs / probs.sum(axis=1, keepdims=True)
    return labels0, probs


def test_detect_noise():
    n = 3000
    labels0, probs = synthetic_probs(n)
    df = pd.DataFrame({"review_id": np.arange(n).astype(str), "rating": labels0 + 1,
                       "cleaned_text": [f"t{i}" for i in range(n)]})
    det = clean.detect_noise(df, probs, verbose=False)
    assert det["selected_method"] in config.CLEANLAB_FILTER_METHODS
    assert det["method_info"]["estimated_theoretical"] > 0
    d = det["df"]
    assert (d["is_noise_severe"] <= d["is_noise"]).all()
    assert (d.loc[d["is_noise_severe"], "rating_diff"] >= config.SEVERITY_THRESHOLD).all()
    q = calibration.per_class_quality(det["labels0"], det["preds0"])
    tab = diagnostics.noise_distance_by_class(d, q)
    assert len(tab) == 5 and tab["n_flag"].sum() == int(d["is_noise"].sum())
    assert set(tab["penanda_confound_recall"]) <= {"YA", "tidak"}
    ok("detect_noise (cleanlab 2.9), flag severe, diagnostik jarak per kelas")
    return df, det


# ---------------------------------------------------------------- preprocess
def test_preprocess():
    lex = {"gpp": "tidak apa-apa", "bgt": "banget"}
    t = preprocess.clean_text_for_bert("GPPPP, Bagusss bgt!!! http://x.co @user \U0001F44D", lex)
    assert "bagus" in t and "banget" in t and "http" not in t and "@" not in t and "bagus" in t
    assert preprocess.clean_text_for_bert("gpp,", lex).startswith("tidak apa-apa,")
    assert preprocess.clean_text_for_bert("\U0001F9FF mantap", lex) == "mantap"  # emoji tak dikenal dihapus
    # setelah duplikat murni dihapus, setiap grup konflik pasti seri dan dibuang seluruhnya
    df = pd.DataFrame({"cleaned_text": ["a", "a", "b", "b", "b", "c"], "rating": [1, 5, 2, 2, 4, 3]})
    df = df.drop_duplicates(subset=["cleaned_text", "rating"])
    out = preprocess.drop_text_rating_conflicts(df, verbose=False)
    assert sorted(out["cleaned_text"]) == ["c"], out
    ok("preprocess: normalisasi, emoji, resolusi konflik (semua seri setelah dedup murni)")


# ---------------------------------------------------------------- split dan kualitas data
def synthetic_clean(n_texts=1500):
    texts = [f"ulasan nomor {i}" for i in range(n_texts)]
    rows = []
    rid = 0
    for i, t in enumerate(texts):
        r = int(rng.integers(1, 6))
        rows.append((rid, t, r)); rid += 1
        if i % 15 == 0:  # teks identik dengan rating berbeda
            rows.append((rid, t, r % 5 + 1)); rid += 1
    df = pd.DataFrame(rows, columns=["review_id", "cleaned_text", "rating"])
    df["review_id"] = df["review_id"].astype(str)
    df["source_app"] = rng.choice(["A", "B", "C"], len(df))
    df["review_text"] = df["cleaned_text"]
    return df


def test_split():
    df = synthetic_clean()
    ctx = config.RunPaths(42)
    tr, va, te = data_split.create_splits(ctx, df_clean=df)
    assert len(tr) + len(va) + len(te) == len(df)
    s = pd.read_csv(os.path.join(ctx.data_dir, "split_summary.csv")).iloc[0]
    assert all(int(s[c]) == 0 for c in s.index if c.startswith("overlap_"))
    # ukuran sekitar 70:10:20
    assert abs(len(tr) / len(df) - 0.7) < 0.03 and abs(len(te) / len(df) - 0.2) < 0.03
    # dimuat ulang tanpa menulis ulang
    tr2, _, _ = data_split.create_splits(ctx, df_clean=df)
    assert len(tr2) == len(tr)
    # seed berbeda menghasilkan partisi berbeda tetapi tetap nol overlap
    ctx2 = config.RunPaths(123)
    tr3, va3, te3 = data_split.create_splits(ctx2, df_clean=df)
    assert set(te3["review_id"]) != set(te["review_id"])
    ok("split berkelompok 70:10:20, nol overlap, seed partisi berbeda")
    return df, ctx


def test_quality_report():
    n = 600
    df = pd.DataFrame({
        "source_app": rng.choice(["Gojek", "SeaBank"], n), "review_text": rng.choice(["bagus", "ok", "jelek sekali aplikasinya", "x y z w v"], n),
        "rating": rng.integers(1, 6, n), "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D"),
        "helpful_votes": 0, "review_id": np.arange(n).astype(str), "detected_lang": "id", "scraped_at": "2026-09-08"})
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    p = os.path.join(TMP, "master_sintetis.csv")
    df.to_csv(p, index=False)
    out = quality_report.run_quality_report(p, make_figures=True)
    assert out["summary"]["n_baris"] == n
    assert os.path.exists(os.path.join(config.QUALITY_DIR, "fig02_rentang_tanggal_per_rating.png"))
    ok("laporan kualitas data dari master (sintetis) tanpa scraping")


# ---------------------------------------------------------------- anotasi
def test_annotation_flow(df_clean, det):
    ctx = config.RunPaths(42)
    df_noise = det["df"][det["df"]["is_noise"]].copy()
    df_noise["source_app"] = "A"
    df_noise["review_text"] = df_noise["cleaned_text"]
    # alur B
    human_validation.create_sample(ctx, df_noise)
    try:
        human_validation.create_sample(ctx, df_noise)
        raise AssertionError("seharusnya menolak menimpa anotasi")
    except annotation.AnnotationExists:
        pass
    human_validation.create_second_annotator()
    # isi verdict (annotator 1 dan 2)
    p1 = config.ANNOT_FILES["human_validation"]
    d1 = pd.read_csv(p1)
    d1["human_verdict"] = rng.choice(list(config.HUMAN_VERDICTS), len(d1))
    d1.to_csv(p1, index=False)
    p2 = config.ANNOT_FILES["human_validation_annotator2"]
    d2 = pd.read_csv(p2)
    d2["human_verdict"] = rng.choice(list(config.HUMAN_VERDICTS), len(d2))
    d2.to_csv(p2, index=False)
    summary_df, breakdown = human_validation.compute_agreement(ctx, df_noise)
    assert summary_df.iloc[0]["n"] == config.HUMAN_VALIDATION_N
    kappa = human_validation.compute_interannotator_kappa(ctx)
    assert -1 <= kappa <= 1
    # hash tidak cocok harus berhenti
    df_other = df_noise.iloc[:-5]
    try:
        human_validation.verify_annotation(df_other)
        raise AssertionError
    except annotation.AnnotationMismatch:
        pass
    # komponen split non-utama harus ditolak
    try:
        human_validation.create_sample(config.RunPaths(123), df_noise)
        raise AssertionError
    except ValueError as e:
        assert "split utama" in str(e)
    ok("anotasi: alur B, tolak timpa, hash sidecar, agreement, kappa, tolak split non-utama")


# ---------------------------------------------------------------- pelatihan sintetis -> analisis
def fake_train_outputs(ctx, seeds, effect_shift):
    """Membuat berkas prediksi dan metrik seperti keluaran train.run_experiment (sintetis)."""
    te = pd.read_csv(ctx.split_file("test"))
    y = te["rating"].values
    base_noise = {"M1": 0.62, "M2": 0.60, "M3": 0.60, "M4": 0.58, "M5": 0.59, "M6": 0.59}
    for sc in config.SCENARIOS:
        for sd in seeds:
            prng = np.random.default_rng(abs(hash((sc["name"], sd, ctx.split_seed))) % (2 ** 32))
            p_wrong = base_noise[sc["name"][:2]] + effect_shift
            wrong = prng.random(len(y)) < p_wrong
            pred = y.copy()
            pred[wrong] = np.clip(y[wrong] + prng.choice([-2, -1, 1, 2], wrong.sum()), 1, 5)
            drive_io.write_once_csv(pd.DataFrame({"review_id": te["review_id"], "y_true": y, "y_pred": pred}),
                                    ctx.scenario_pred_file(sc["name"], sd))
            m = metrics.compute_metrics(y, pred)
            drive_io.write_once_json({"scenario": sc["name"], "weight_seed": sd, "test": m},
                                     ctx.scenario_metrics_file(sc["name"], sd))


def test_significance_and_gold(df_clean):
    ctx = config.RunPaths(42)
    seeds = config.WEIGHT_SEEDS
    pipeline.register_run_mode(ctx, seeds, False)
    fake_train_outputs(ctx, seeds, 0.0)
    out = significance.run_significance_suite(ctx, seeds)
    assert len(out["sig"]) == 5 and len(out["eff"]) == 5 and len(out["qwk_ens"]) == 5 and len(out["qwk_per"]) == 5
    assert set(out["long"]["metric"]) == {"MAE", "QWK_ensemble", "QWK_per_seed"} and len(out["long"]) == 15
    # QWK ensemble: titik taksiran sama dengan sklearn pada prediksi ensemble
    true, preds, _ = significance.load_predictions(ctx, seeds)
    pa = significance.ensemble_predictions(preds["M4_Baseline_CORN"])
    q = cohen_kappa_score(true, pa, weights="quadratic")
    assert abs(out["qwk_ens"].iloc[0]["qwk_a"] - q) < 1e-12
    # QWK per seed: rata-rata per seed sesuai sklearn
    qs = np.mean([cohen_kappa_score(true, preds["M4_Baseline_CORN"][s], weights="quadratic") for s in seeds])
    assert abs(out["qwk_per"].iloc[0]["qwk_a_rata_seed"] - qs) < 1e-12
    selfcheck.check_significance(ctx)
    selfcheck.check_training(ctx)

    # uji emas
    gold_test.create_sample(ctx)
    gold_test.create_second_annotator()
    g1 = pd.read_csv(config.ANNOT_FILES["gold"])
    te = pd.read_csv(ctx.split_file("test")).set_index("review_id")
    g1["human_gold_rating"] = [str(int(te.loc[r, "rating"])) for r in g1["review_id"]]
    g1.loc[g1.index[0], "human_gold_rating"] = "ND"
    g1.to_csv(config.ANNOT_FILES["gold"], index=False)
    g2 = pd.read_csv(config.ANNOT_FILES["gold_annotator2"])
    g2["human_gold_rating"] = [int(te.loc[r, "rating"]) for r in g2["review_id"]]
    g2.to_csv(config.ANNOT_FILES["gold_annotator2"], index=False)
    res, eff = gold_test.evaluate_on_gold(ctx, seeds)
    assert len(res) == 6 and res["n_gold"].iloc[0] == len(g1) - 1
    k = gold_test.compute_gold_kappa(ctx)
    assert k > 0.99
    ok("signifikansi (Wilcoxon, MAE, QWK ensemble, QWK per seed) dan uji emas dari prediksi tersimpan")


def test_partition_sensitivity(df_clean):
    for s, shift in ((123, 0.0), (2024, 0.01)):
        ctx = config.RunPaths(s)
        data_split.create_splits(ctx, df_clean=df_clean)
        seeds = pipeline.weight_seeds_for(s, hemat=False)
        pipeline.register_run_mode(ctx, seeds, False)
        fake_train_outputs(ctx, seeds, shift)
        significance.run_significance_suite(ctx, seeds)
    long, summ, wide = pipeline.consistency_summary()
    assert len(summ) == 15 and set(summ["putusan"]) <= {"KONSISTEN", "TIDAK KONSISTEN"}
    assert "wilcoxon_p_holm_split42" in wide.columns
    selfcheck.check_partition_sensitivity()
    # opsi hemat: satu seed
    assert pipeline.weight_seeds_for(123, hemat=True) == [42]
    assert pipeline.weight_seeds_for(42, hemat=True) == config.WEIGHT_SEEDS  # split utama tidak boleh hemat
    # mode tidak boleh berubah
    try:
        pipeline.register_run_mode(config.RunPaths(123), [42], True)
        raise AssertionError
    except RuntimeError:
        pass
    ok("sensitivitas partisi: tabel panjang, konsistensi, opsi hemat, mode terkunci")


def test_k_sensitivity_pure():
    a = [str(i) for i in range(100)]
    b = [str(i) for i in range(50, 150)]
    assert abs(sensitivity_k.jaccard(a, b) - 50 / 150) < 1e-12
    labels0 = rng.integers(0, 5, 1500)
    px = np.clip(labels0 + rng.integers(-1, 2, 1500), 0, 4)
    d, lo, hi = sensitivity_k.paired_qwk_diff_ci(labels0, px, px, n_boot=200)
    assert abs(d) < 1e-12 and lo == 0 and hi == 0
    ok("sensitivitas K: Jaccard dan CI berpasangan")


def test_summary_manifest():
    m = summary.build_output_map()
    assert {"berkas", "subbab_proposal", "ada"}.issubset(m.columns)
    df, res = summary.build_results_manifest()
    assert len(df) > 10 and df["sha256"].str.len().eq(64).all()
    z = drive_io.export_backup_zip(config.DRIVE_ROOT, "cadangan_uji.zip")
    assert z["n_files"] > 10
    ok("peta keluaran, manifest SHA-256, zip cadangan")


# ---------------------------------------------------------------- orkestrasi proxy dengan OOF tiruan
def test_orchestration_with_mock_oof():
    from src import proxy as proxy_mod
    ctx = config.RunPaths(42)
    df_train = pd.read_csv(ctx.split_file("train"))
    df_train["source_app"] = "A"
    df_train["review_text"] = df_train["cleaned_text"]
    n = len(df_train)
    labels0 = df_train["rating"].values - 1

    def mock_get_oof(ctx_, df, proxy_id, K=None):
        r = np.random.default_rng(100 + proxy_id + (K or 5))
        pred = np.where(r.random(n) < 0.55, labels0, np.clip(labels0 + r.choice([-2, -1, 1, 2], n), 0, 4))
        cp = np.full((n, 5), 0.03)
        cp[np.arange(n), pred] = 0.88
        cp = cp / cp.sum(axis=1, keepdims=True)
        q = None
        if proxy_id == 3:
            q = np.stack([cp[:, k + 1:].sum(axis=1) for k in range(4)], axis=1)
        temps = np.array([1.2] * 5) if proxy_id >= 2 else None
        return {"class_probs": cp.astype(np.float32), "q_chain": q, "temperatures": temps}

    original = proxy_mod.get_oof
    proxy_mod.get_oof = mock_get_oof
    try:
        out = pipeline.step_p4(ctx, df_train)
        for v in ("raw", "hard", "severe"):
            assert drive_io.is_verified(ctx.variant_file(v))
        n_raw, n_hard, n_sev = [len(pd.read_csv(ctx.variant_file(v))) for v in ("raw", "hard", "severe")]
        assert n_hard <= n_sev <= n_raw
        assert drive_io.is_verified(os.path.join(ctx.diagnostics_dir, "P4__jarak_ordinal_noise_per_kelas.csv"))
        res = out["eval"]["result"]
        assert "ece_kumulatif_rantai_rata" in res and "monotonisitas_rantai__proporsi_monoton" in res
        # idempoten: pemanggilan kedua tidak menulis ulang dan tidak error
        out2 = pipeline.step_p4(ctx, df_train)
        assert out2["variant_stats"]["n_raw"] == n_raw
        # P1 sampai P3 tidak boleh menghasilkan varian data
        table = pipeline.step_ablation_p1_p3(ctx, df_train, out["eval"])
        assert list(table["label"]) == ["P1", "P2", "P3", "P4"]
        assert sorted(f for f in os.listdir(ctx.cleaned_dir)) == ["train_hard.csv", "train_raw.csv", "train_severe.csv"]
        try:
            clean.export_p4_outputs(ctx, out["eval"]["detection"], 2, 5)
            raise AssertionError
        except ValueError:
            pass
        try:
            clean.export_p4_outputs(ctx, out["eval"]["detection"], 3, 10)
            raise AssertionError
        except ValueError:
            pass
        try:
            pipeline.step_ablation_p1_p3(config.RunPaths(123), df_train, out["eval"])
            raise AssertionError
        except ValueError:
            pass
        ktab, kpairs = sensitivity_k.run_k_sensitivity(ctx, df_train, out["eval"])
        assert sorted(ktab["K"]) == [3, 5, 10] and abs(ktab.loc[ktab["K"] == 5, "jaccard_vs_K5"].iloc[0] - 1.0) < 1e-12
        assert len(kpairs) == 3
        try:
            sensitivity_k.run_k_sensitivity(config.RunPaths(2024), df_train, out["eval"])
            raise AssertionError
        except ValueError:
            pass
        # berkas OOF P4 tiruan agar pemeriksaan mandiri dapat dijalankan
        ids = drive_io.ids_hash(df_train["review_id"].astype(str).tolist(), sort=False)
        mk = mock_get_oof(ctx, df_train, 3, 5)
        drive_io.write_once_npz(os.path.join(ctx.proxy_dir, "oof__finetuned_corn__K5.npz"),
                                class_probs=mk["class_probs"], q_chain=mk["q_chain"],
                                temperatures=np.array([1.2] * 5), ids_sha=np.array(ids))
        selfcheck.check_p4(ctx)
        selfcheck.check_ablation(ctx)
        selfcheck.check_k_sensitivity(ctx)
    finally:
        proxy_mod.get_oof = original
    ok("orkestrasi P4, ablasi, sensitivitas K dengan OOF tiruan; P1 sampai P3 tanpa varian; guard split non-utama")


if __name__ == "__main__":
    test_drive_io()
    test_audit()
    test_qwk_fast()
    test_calibration()
    test_sampling()
    test_consistency()
    df_syn, det = test_detect_noise()
    test_preprocess()
    df_clean, _ = test_split()
    test_quality_report()
    test_orchestration_with_mock_oof()
    test_annotation_flow(df_clean, det)
    test_significance_and_gold(df_clean)
    test_partition_sensitivity(df_clean)
    test_k_sensitivity_pure()
    test_summary_manifest()
    print(f"\nRINGKASAN: {len(PASS)} kelompok uji lulus.")
