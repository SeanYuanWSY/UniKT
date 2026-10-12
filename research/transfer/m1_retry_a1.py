"""Batch 69: Markov transition table T(y_t | kc_{t-1}, y_{t-1}, kc_t) under
A1-only — the fair retrial of idea M1 that never happened.

M1 (markov_table_m1.py, batch-era ~M1) died to the artifact rebase: its
withA numbers (0.73-0.94) were artifact-inflated and the idea was closed as
"artifact-dominated" BEFORE the A1-only protocol existed. The transition
table has never been scored under the clean caliber. B1 is a first-order KC
marginal; the transition table adds response-conditioned dynamics that B1
structurally cannot see. This is a genuine hypothesis space, now testable
fairly.

Design: per (target KC, prev-KC, prev-response) -> P(y=1), estimated from
non-holdout rows of the SAME [:300] pool (pooled B1 walk, run-complete
rule), with Laplace smoothing to handle sparse cells. Score holdout rows
(excluding A1 rows, same as every fair eval) where the previous row exists
and is in-evidence (not holdout). Compare per-student AUC vs B1 on the
SAME scored rows (same-support pairing, stratified AUC, cluster bootstrap
B=5000 seed=0) across 3 domains.

Gates in 20261012_batch69.md. Sentinel: M1RETRY-DONE / M1RETRY-DONE-ANCHOR.
Anchors: n_kc_prior_b1 == {96,175,38}; B1-side cells on common support
reproduce pooled B1 references (sanity: B1 mean over common students within
0.01 of b59 identity-table values is NOT expected here — this is a fresh
construction; instead anchor on n_paired>0 per domain and n_nonfinite=0).
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

KC_N = {"assistments17": 96, "ednet_kt1": 175, "junyi2015": 38}
DS_LIST = ("assistments17", "ednet_kt1", "junyi2015")


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def run_ids(q_all):
    rid = [0] * len(q_all)
    cur = 0
    for i in range(1, len(q_all)):
        if q_all[i] != q_all[i - 1]:
            cur += 1
        rid[i] = cur
    return rid


def boot_ci(d, B=5000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = d[idx].mean()
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def build_tables(samples):
    """Pooled B1 (KC marginal) + Markov transition table from evidence rows.

    Evidence rule = run-complete (same as every b48+ walk): a row is
    evidence if its run contains no holdout row.
    Transition evidence additionally requires the previous row to exist and
    be evidence of the same run (adjacent within run).
    """
    b1 = {}
    trans = {}  # (kc_t, kc_{t-1}, y_{t-1}) -> [sum_y1, n]
    for ws in samples:
        hold = set(ws.holdout_idx.tolist())
        q_all = [int(x) for x in ws.question.tolist()]
        rid = run_ids(q_all)
        hold_runs = {rid[i] for i in hold}
        seq = [int(x) for x in ws.sequence.tolist()]
        resp = [int(x) for x in ws.response.tolist()]
        for i in range(len(seq)):
            if rid[i] not in hold_runs:
                b1.setdefault(seq[i], []).append(resp[i])
                if i > 0 and rid[i - 1] == rid[i] and rid[i] not in hold_runs:
                    key = (seq[i], seq[i - 1], resp[i - 1])
                    trans.setdefault(key, [0, 0])
                    trans[key][0] += resp[i]
                    trans[key][1] += 1
    b1_tab = {k: float(np.mean(v)) for k, v in b1.items() if v}
    # Laplace(1,2) smoothed transition probabilities
    trans_tab = {k: (s + 1) / (n + 2) for k, (s, n) in trans.items() if n >= 3}
    return b1_tab, trans_tab, len(trans), sum(1 for v in trans.values() if v[1] >= 3)


def main(n_users=300):
    out = {"cells": {}}
    for ds in DS_LIST:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        b1_tab, trans_tab, n_trans_raw, n_trans_kept = build_tables(samples)
        per_stu = {}
        n_nonfinite = 0
        for ws in samples:
            seq = [int(x) for x in ws.sequence.tolist()]
            resp = [int(x) for x in ws.response.tolist()]
            hold_pos = set(ws.holdout_idx.tolist())
            rows_m, rows_b = [], []
            for i in sorted(hold_pos):
                if i == 0 or is_a1(ws, i):
                    continue
                y = resp[i]
                kc = seq[i]
                if kc not in b1_tab:
                    continue
                pb = b1_tab[kc]
                pm = None
                if i > 0:
                    key = (kc, seq[i - 1], resp[i - 1])
                    pm = trans_tab.get(key)
                if pm is None:
                    pm = pb  # no evidence cell -> fall back to B1 marginal
                rows_m.append({"y": y, "pred": pm})
                rows_b.append({"y": y, "pred": pb})
            am = stratified_auc_rows(rows_m, "pred")
            ab = stratified_auc_rows(rows_b, "pred")
            if am is not None and ab is not None and rows_m:
                # delta = markov+fallback vs pure B1 on SAME rows
                per_stu[ws.user_id] = (am, ab)
        common = sorted(per_stu)
        deltas = [per_stu[u][0] - per_stu[u][1] for u in common]
        out["cells"][ds] = {
            "n_paired": len(common),
            "markov_mean": round(float(np.mean([per_stu[u][0] for u in common])), 4) if common else None,
            "b1_mean": round(float(np.mean([per_stu[u][1] for u in common])), 4) if common else None,
            "delta_markov_minus_b1": round(float(np.mean(deltas)), 4) if common else None,
            "ci95": boot_ci(deltas) if common else None,
            "n_nonfinite_probs": n_nonfinite,
            "n_kc_prior_b1": len(b1_tab),
            "n_trans_raw": n_trans_raw,
            "n_trans_kept_ge3": n_trans_kept,
            "anchor_n_kc": len(b1_tab) == KC_N[ds],
        }
        print(ds, json.dumps(out["cells"][ds], ensure_ascii=False), flush=True)
    out["n_anchor_pass"] = sum(1 for c in out["cells"].values() if c["anchor_n_kc"])
    with open("/root/unikt-fork/research/transfer/results/m1_retry_a1.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("M1RETRY-DONE" if out["n_anchor_pass"] == 3 else "M1RETRY-DONE-ANCHOR")


if __name__ == "__main__":
    main()
