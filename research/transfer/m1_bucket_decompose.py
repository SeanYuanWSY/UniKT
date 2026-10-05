"""M1 bucket decomposition (review-mandated decisive test).

Holdout rows bucketed by relation to the previous row:
  A) same question (q_{t-1}==q_t): label-copy artifact bucket (multi-KC explode)
  B) different question, same KC: genuine one-step recency bucket
  C) different KC: transition bucket
Per-bucket within-student AUC + row shares. If bucket A dominates and AUC~1,
the M1 headline is an artifact; if bucket B stays high, genuine recency.
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402


def run_domain(dataset, n_users=300, m=8.0):
    src = get_data_source(_make_rc(dataset))
    samples = load_user_samples(src, fold=0)[: n_users + 100]
    students = build_rows(samples, None)
    prior = fit_global_kc_prior(students)

    pair_pos, pair_tot = defaultdict(int), defaultdict(int)
    for ws in samples[: len(students)]:
        ev = ws.evidence_idx
        seq = [(int(ws.sequence[i]), int(ws.response[i])) for i in ev.tolist()]
        for (kc_prev, y_prev), (kc_t, y_t) in zip(seq[:-1], seq[1:]):
            pair_tot[(kc_t, kc_prev, y_prev)] += 1
            pair_pos[(kc_t, kc_prev, y_prev)] += y_t

    def m1(kc_t, kc_prev, y_prev):
        p_kc = prior.get(kc_t, 0.5)
        key = (kc_t, kc_prev, y_prev)
        c1, c0 = pair_pos.get(key, 0), pair_tot.get(key, 0)
        return (c1 + m * p_kc) / (c0 + m) if (c0 + m) > 0 else p_kc

    buckets = {"A_sameq": [], "B_samekc_diffq": [], "C_diffkc": []}
    shares = {"A_sameq": 0, "B_samekc_diffq": 0, "C_diffkc": 0}
    total = 0
    for ws in samples[: len(students)]:
        holdout_pos = set(ws.holdout_idx.tolist())
        rows_by_bucket = {"A_sameq": [], "B_samekc_diffq": [], "C_diffkc": []}
        for i in sorted(holdout_pos):
            if i == 0:
                continue
            kc_t, kc_prev = int(ws.sequence[i]), int(ws.sequence[i - 1])
            y_prev = int(ws.response[i - 1])
            p = m1(kc_t, kc_prev, y_prev)
            row = {"y": int(ws.response[i]), "pred": p}
            q_same = int(ws.question[i]) == int(ws.question[i - 1])
            if q_same:
                rows_by_bucket["A_sameq"].append(row)
            elif kc_t == kc_prev:
                rows_by_bucket["B_samekc_diffq"].append(row)
            else:
                rows_by_bucket["C_diffkc"].append(row)
            total += 1
        for b, rows in rows_by_bucket.items():
            shares[b] += len(rows)
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                buckets[b].append(a)

    out = {"total_rows": total}
    for b in buckets:
        out[b] = {
            "share": round(shares[b] / max(total, 1), 4),
            "within_student_auc": round(float(np.mean(buckets[b])), 4) if buckets[b] else None,
            "n_students": len(buckets[b]),
        }
    return out


res = {}
for ds in ["assistments17", "junyi2015", "ednet_kt1"]:
    res[ds] = run_domain(ds)
    print(ds, json.dumps(res[ds], ensure_ascii=False), flush=True)
with open("/root/unikt-fork/research/transfer/results/m1_bucket_decompose.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("M1-BUCKET-DONE")
