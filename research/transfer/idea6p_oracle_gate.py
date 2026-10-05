"""IDEA-6p oracle gate: reachable upper bound for low-attempt KC improvement.

For each domain: replace the B1 prior of KCs with pooled evidence count
n_kc <= 10 by their TRUE holdout mean (oracle), keep everything else, and
compute the within-student AUC delta vs vanilla B1 (full set).
Gate rule (preregistered): all three domains oracle-delta < +0.02 -> no-go.
Also reports holdout-row share by evidence-count bucket.
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402


def run_domain(dataset, n_users=300):
    src = get_data_source(_make_rc(dataset))
    samples = load_user_samples(src, fold=0)[:n_users]
    students = build_rows(samples, None)

    counts = defaultdict(int)
    sums = defaultdict(int)
    for st in students:
        for kc, hist in st["kc_hist"].items():
            counts[kc] += len(hist)
            sums[kc] += sum(hist)
    prior = {kc: sums[kc] / counts[kc] for kc in counts if counts[kc] > 0}

    # holdout true mean per KC (oracle target)
    hold_sums = defaultdict(int)
    hold_counts = defaultdict(int)
    for st in students:
        for r in st["rows"]:
            hold_sums[r["kc"]] += r["y"]
            hold_counts[r["kc"]] += 1
    oracle_p = {kc: hold_sums[kc] / hold_counts[kc] for kc in hold_counts if hold_counts[kc] >= 5}

    # bucket shares on holdout rows
    buckets = {"0": 0, "1-3": 0, "4-10": 0, "11-30": 0, ">30": 0}
    total_rows = 0
    for st in students:
        for r in st["rows"]:
            n = counts.get(r["kc"], 0)
            total_rows += 1
            key = "0" if n == 0 else "1-3" if n <= 3 else "4-10" if n <= 10 else "11-30" if n <= 30 else ">30"
            buckets[key] += 1
    shares = {k: round(v / max(total_rows, 1), 4) for k, v in buckets.items()}

    def auc_with(fn):
        aucs = []
        for st in students:
            rows = [{"y": r["y"], "pred": fn(r)} for r in st["rows"]]
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                aucs.append(a)
        return round(float(np.mean(aucs)), 4), len(aucs)

    base, n = auc_with(lambda r: prior.get(r["kc"], 0.5))

    results = {"n_students": n, "n_rows": total_rows, "bucket_share": shares, "b1": base}
    for thr in (3, 10):
        def patched(r, thr=thr):
            kc = r["kc"]
            if counts.get(kc, 0) <= thr and kc in oracle_p:
                return oracle_p[kc]
            return prior.get(kc, 0.5)

        a, _ = auc_with(patched)
        results[f"oracle_n<={thr}"] = a
        results[f"delta_n<={thr}"] = round(a - base, 4)
    return results


out = {}
for ds in ["assistments17", "junyi2015", "ednet_kt1"]:
    out[ds] = run_domain(ds)
    print(ds, json.dumps(out[ds], ensure_ascii=False), flush=True)

with open("/root/unikt-fork/research/transfer/results/idea6p_oracle_gate.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("IDEA6P-GATE-DONE")
