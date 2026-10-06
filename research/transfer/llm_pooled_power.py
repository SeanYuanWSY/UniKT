"""Batch40: pooled-metric plug-in replication power — dual of batch39.

batch39: per-student results are boundary-class (replication power <=0.48 at
N=400, structural: perm-rank robust to cohort enlargement). The pooled metric
(batch37: Junyi p=0.0149 ge=2, EdNet p=0.0249 ge=4, A17 p=0.1244 ge=24) sits
far from the alpha boundary for Junyi/EdNet — expected high plug-in
replication power. Same student-bootstrap design as batch39 but the statistic
is the pooled AUC of the drawn cohort's rows (tie-averaged rank method,
exactly Mann-Whitney with tie=0.5 — self-checked against batch37's values).

Per-draw p = (#{perm pooled AUC >= real} + 1)/201 over batch37's permutation
family (rng 500+p, zip-form mapping — must reproduce b37's full-cohort p);
power(N) = fraction of B=400 draws with p < 0.05; N in {30, 60, n_scored, 120,
200, 300, 400}. N80 on the grid; observed-effect plug-in caveat as batch39.

Implementation note: per-student (kc_index, y) arrays are precomputed once
(kc_index maps each row's KC into kcs_sorted, -1 for uncovered -> pred 0.5);
a draw is two np.concatenate calls, and each assignment's predictions are one
fancy-index gather — no Python-level row loops inside the bootstrap.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

CACHE = "/root/unikt-fork/research/transfer/results/k3_difficulty_cache.json"
N_PERM = 200
N_BOOT = 400
B37_POOLED_REF = {"assistments17": 0.5321, "ednet_kt1": 0.5893, "junyi2015": 0.6645}
B37_P_REF = {"assistments17": 0.1244, "ednet_kt1": 0.0249, "junyi2015": 0.0149}


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def rankdata_avg(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def auc_from_preds(preds, ys):
    pos = ys == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    r = rankdata_avg(preds)
    return (r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    diff = {ds: {int(k): v for k, v in d.items()} for ds, d in json.load(open(CACHE)).items()}
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval:]
        llm = diff.get(ds, {})

        # per-student (kc_index, y) arrays — row construction identical to b28/37/38/39
        kcs_sorted = sorted(llm)
        kc_pos = {k: t for t, k in enumerate(kcs_sorted)}
        stu_kci, stu_y = [], []
        for ws in eval_samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            kci, yy = [], []
            for i in hold:
                if i == 0:
                    continue
                if is_a1(ws, i):
                    continue
                kci.append(kc_pos.get(int(ws.sequence[i]), -1))
                yy.append(int(ws.response[i]))
            stu_kci.append(np.array(kci, dtype=np.int32))
            stu_y.append(np.array(yy, dtype=np.int8))

        vals = np.array([llm[k] for k in kcs_sorted], dtype=float)
        assert np.isfinite(vals).all()
        # batch37 permutation family: rng 500+p, zip-form mapping
        perm_val_arrs = []
        for p in range(N_PERM):
            r2 = np.random.default_rng(500 + p)
            perm_val_arrs.append(r2.permutation(vals))
        real_val_arr = vals.copy()

        def auc_of(kci, y, varr):
            preds = varr[kci]
            preds = np.where(kci < 0, 0.5, preds)
            return auc_from_preds(preds, y)

        def cohort(idx):
            kci = np.concatenate([stu_kci[s] for s in idx])
            y = np.concatenate([stu_y[s] for s in idx])
            return kci, y

        n_stu = len(stu_kci)
        kci_f, y_f = cohort(range(n_stu))
        real_full = auc_of(kci_f, y_f, real_val_arr)
        assert abs(real_full - B37_POOLED_REF[ds]) < 5e-5, f"{ds}: pooled {real_full:.4f} != b37 {B37_POOLED_REF[ds]}"
        ge_full = int(sum(auc_of(kci_f, y_f, a) >= real_full for a in perm_val_arrs))
        p_full = (ge_full + 1) / (N_PERM + 1)
        assert abs(p_full - B37_P_REF[ds]) < 0.5 / (N_PERM + 1), f"{ds}: p {p_full:.4f} != b37 {B37_P_REF[ds]}"

        rng = np.random.default_rng(950)
        curve = {}
        for N in sorted({30, 60, n_stu, 120, 200, 300, 400}):
            hits = 0
            for b in range(N_BOOT):
                idx = rng.integers(0, n_stu, size=N)
                kci, y = cohort(idx)
                stat = auc_of(kci, y, real_val_arr)
                ge = sum(auc_of(kci, y, a) >= stat for a in perm_val_arrs)
                if (ge + 1) / (N_PERM + 1) < 0.05:
                    hits += 1
            curve[N] = round(hits / N_BOOT, 3)
        n80 = next((N for N in sorted(curve) if curve[N] >= 0.8), None)

        out[ds] = {
            "n_students": n_stu,
            "pooled_full": round(real_full, 4),
            "ge_full": ge_full,
            "p_full_check": round(p_full, 4),
            "power_curve": curve,
            "N80": n80 if n80 else ">400",
        }
        print(ds, json.dumps(out[ds]), flush=True)

    with open("/root/unikt-fork/research/transfer/results/llm_pooled_power.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("POOLEDPOWER-DONE")


if __name__ == "__main__":
    main()
