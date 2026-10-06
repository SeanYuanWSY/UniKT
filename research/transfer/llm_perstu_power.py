"""Batch39: per-student power curve — is Junyi's per-student non-significance
absence of signal or absence of power?

batch28: per-student LLM table AUC A17 0.5246 / EdNet 0.5496 / Junyi 0.5456,
perm p 0.0995 / 0.0448 / 0.0647 (200 perms, rng 200+p). Junyi missed 0.05 by
one step with n=82 scored students. This batch estimates, per domain, the
cohort size N needed for 80% power at alpha=0.05 under the CURRENT effect,
using only existing rows:

- R[student]: per-student AUC under real assignment (row construction identical
  to batch28/37/38).
- P[perm][student]: per-student AUC under each of batch28's EXACT 200 permuted
  assignments (same rng 200+p and same permutation mapping form as
  llm_diff_formal.py — full-cohort p must reproduce batch28 exactly).
- Power at cohort size N: B=400 student-bootstrap draws (with replacement);
  for each draw, stat = mean R over drawn students; null for that draw = the
  200 perm means over the SAME drawn students; p_draw = (#{null>=stat}+1)/201;
  power = fraction of draws with p_draw < 0.05. N in {30, 60, n_scored, 120,
  200, 300, 400}; N80 = smallest N with power >= 0.8 (">400" if none).

Caveat (preregistered): N > n_scored is a bootstrap extrapolation — the n
scored students are treated as the sampling population; N80 beyond the cohort
is a model-based projection, flagged as such. Descriptive; no new significance
claims. Null-set alignment note: stratified_auc_rows returns None only when a
student's y set lacks a class — assignment-independent, so the scored-student
set is identical across real and all perms.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

CACHE = "/root/unikt-fork/research/transfer/results/k3_difficulty_cache.json"
N_PERM = 200
N_BOOT = 400
B28_MEAN_REF = {"assistments17": 0.5246, "ednet_kt1": 0.5496, "junyi2015": 0.5456}
B28_P_REF = {"assistments17": 0.0995, "ednet_kt1": 0.0448, "junyi2015": 0.0647}


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def main():
    diff = {ds: {int(k): v for k, v in d.items()} for ds, d in json.load(open(CACHE)).items()}
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval:]
        llm = diff.get(ds, {})

        # per-student (y, kc) row lists — identical construction to batch28/37/38
        stu_rows = []
        for ws in eval_samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            rows = []
            for i in hold:
                if i == 0:
                    continue
                if is_a1(ws, i):
                    continue
                rows.append((int(ws.response[i]), int(ws.sequence[i])))
            stu_rows.append(rows)

        def stu_auc(rows, assign):
            r = [{"y": y, "pred": assign.get(k, 0.5)} for (y, k) in rows]
            return stratified_auc_rows(r, "pred")

        R = np.array([a for a in (stu_auc(rows, llm) for rows in stu_rows) if a is not None])

        # batch28's exact permutation form: rng 200+p, sh = permutation(len(kcs)),
        # assign[kcs[sh[t]]] = vals[t]
        kcs = sorted(llm.keys())
        vals = [llm[k] for k in kcs]
        assert np.isfinite(np.array(vals)).all()  # None-alignment precondition: no NaN preds ever
        P = np.zeros((N_PERM, len(R)))
        for p in range(N_PERM):
            r2 = np.random.default_rng(200 + p)
            sh = r2.permutation(len(kcs))
            assign = {kcs[sh[t]]: vals[t] for t in range(len(kcs))}
            P[p] = [a for a in (stu_auc(rows, assign) for rows in stu_rows) if a is not None]

        # self-check 1: real mean matches batch28
        real_mean = float(R.mean())
        assert abs(real_mean - B28_MEAN_REF[ds]) < 5e-4, f"{ds}: mean {real_mean:.4f} != b28 {B28_MEAN_REF[ds]}"
        # self-check 2: full-cohort perm p reproduces batch28 exactly
        full_null = P.mean(axis=1)
        ge_full = int(np.sum(full_null >= real_mean))
        p_full = (ge_full + 1) / (N_PERM + 1)
        assert abs(p_full - B28_P_REF[ds]) < 0.5 / (N_PERM + 1), f"{ds}: p {p_full:.4f} != b28 {B28_P_REF[ds]}"

        rng = np.random.default_rng(900)
        n_stu = len(R)
        curve = {}
        for N in sorted({30, 60, n_stu, 120, 200, 300, 400}):
            hits = 0
            for b in range(N_BOOT):
                idx = rng.integers(0, n_stu, size=N)
                stat = R[idx].mean()
                null = P[:, idx].mean(axis=1)
                p_b = (np.sum(null >= stat) + 1) / (N_PERM + 1)
                if p_b < 0.05:
                    hits += 1
            curve[N] = round(hits / N_BOOT, 3)
        n80 = next((N for N in sorted(curve) if curve[N] >= 0.8), None)

        out[ds] = {
            "n_students": n_stu,
            "real_mean": round(real_mean, 4),
            "ge_full": ge_full,
            "p_full_check": round(p_full, 4),
            "power_curve": curve,
            "N80": n80 if n80 else ">400",
        }
        print(ds, json.dumps(out[ds]), flush=True)

    with open("/root/unikt-fork/research/transfer/results/llm_perstu_power.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("POWER-DONE")


if __name__ == "__main__":
    main()
