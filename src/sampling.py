"""sampling.py: rumus Cochran dan alokasi sampel validasi manusia (Subbab 3.7, 3.10)."""

import math

import pandas as pd

from src import config


def cochran_n(N, e=None, z=None, p=None):
    """
    n0 = Z^2 p (1 - p) / e^2 ; n = n0 / (1 + (n0 - 1) / N). Pembulatan ke atas.
    Mengembalikan dict agar n0, n, dan parameter dapat dicetak dan disimpan.
    """
    e = config.COCHRAN_E if e is None else e
    z = config.COCHRAN_Z if z is None else z
    p = config.COCHRAN_P if p is None else p
    n0 = (z ** 2) * p * (1 - p) / (e ** 2)
    n = n0 / (1 + (n0 - 1) / N)
    return {"N": int(N), "z": z, "p": p, "e": e, "n0": n0, "n_cochran": math.ceil(n)}


def allocation_plan(pool_sizes, n_total=None, min_per_severe_bin=None, severe_bins=None):
    """
    Alokasi tidak proporsional: minimal 'min_per_severe_bin' baris untuk tiap bin jarak
    ordinal 2, 3, 4 (dibatasi ukuran kolam), sisa n_total ke bin jarak 1.
    pool_sizes: dict {bin: jumlah baris tersedia}, bin 1..4 (jarak >= 4 digabung ke 4).
    """
    n_total = n_total or config.HUMAN_VALIDATION_N
    min_per_severe_bin = config.MIN_PER_SEVERE_BIN if min_per_severe_bin is None else min_per_severe_bin
    severe_bins = severe_bins or config.SEVERE_BINS

    take, shortfall = {}, {}
    allocated_severe = 0
    for b in severe_bins:
        avail = int(pool_sizes.get(b, 0))
        t = min(min_per_severe_bin, avail)
        if t < min_per_severe_bin:
            shortfall[b] = {"target": min_per_severe_bin, "tersedia": avail}
        take[b] = t
        allocated_severe += t
    remaining = max(n_total - allocated_severe, 0)
    avail1 = int(pool_sizes.get(1, 0))
    t1 = min(remaining, avail1)
    if t1 < remaining:
        shortfall[1] = {"target": remaining, "tersedia": avail1}
    take[1] = t1
    return {"take": take, "shortfall": shortfall, "total": sum(take.values())}


def stratified_human_sample(df_noise, n=None, seed=42):
    """
    Mengambil sampel validasi manusia dari baris ter-flag. Urutan random_state mengikuti
    implementasi sebelumnya: bin 2, 3, 4 memakai seed, seed+1, seed+2, lalu bin 1 memakai seed+3,
    dan hasil diacak dengan random_state=seed.
    """
    n = n or config.HUMAN_VALIDATION_N
    df = df_noise.copy()
    df["_diff_bin"] = df["rating_diff"].clip(upper=4)
    pool_sizes = df["_diff_bin"].value_counts().to_dict()
    plan = allocation_plan(pool_sizes, n_total=n)

    parts, rng_seed = [], seed
    for b in config.SEVERE_BINS:
        take = plan["take"][b]
        if take > 0:
            parts.append(df[df["_diff_bin"] == b].sample(n=take, random_state=rng_seed))
        rng_seed += 1
    if plan["take"][1] > 0:
        parts.append(df[df["_diff_bin"] == 1].sample(n=plan["take"][1], random_state=rng_seed))

    sample = pd.concat(parts) if parts else df.iloc[0:0]
    sample = sample.drop(columns=["_diff_bin"]).sample(frac=1, random_state=seed)
    return sample, plan
