"""POOL-SIZE scan: how much target-domain data does the lookup table need?

Build the B1 prior from the FIRST n_pool students' evidence only
(n_pool in {10, 30, 100, 300}); evaluate on a FIXED evaluation set
(students 301..400) so the comparison is identical across pool sizes.
Reference lines: deep transfer best (0.532/0.466/0.509) and B2 recency
(per-student, pool-independent horizontal line).
Also reports KC-coverage of the prior (fraction of eval rows with n_kc>0).
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

DEEP_REF = {"assistments17": 0.532, "junyi2015": 0.466, "ednet_kt1": 0.509}
POOL_SIZES = [10, 30, 100, 300]


def run_domain(dataset, n_eval=100):
    src = get_data_source(_make_rc(dataset))
    samples = load_user_samples(src, fold=0)
    n_eval_actual = min(n_eval, max(10, len(samples) - 60))
    eval_samples = samples[len(samples) - n_eval_actual:]
    pool_samples = samples[: len(samples) - n_eval_actual]

    pool_students = build_rows(pool_samples, None)
    eval_students = build_rows(eval_samples, None)

    # accumulate raw counts per KC over pool students, incrementally by student order
    counts = defaultdict(int)
    sums = defaultdict(int)
    all_pool_rows = []
    for idx, st in enumerate(pool_students):
        for kc, hist in st["kc_hist"].items():
            counts[kc] += len(hist)
            sums[kc] += sum(hist)
        all_pool_rows.append((idx, {kc: (sums[kc], counts[kc]) for kc in counts}))

    # global prior at each cutoff: recompute cheaply by snapshotting at boundaries
    snapshots = {}
    snap_counts = defaultdict(int)
    snap_sums = defaultdict(int)
    n_pool_actual = len(pool_students)
    boundaries = sorted({min(s, n_pool_actual) - 1 for s in POOL_SIZES if s <= n_pool_actual} | {n_pool_actual - 1})
    idx_map = {idx: (dict(), dict()) for idx in boundaries}
    for idx, st in enumerate(pool_students):
        for kc, hist in st["kc_hist"].items():
            snap_counts[kc] += len(hist)
            snap_sums[kc] += sum(hist)
        if idx in idx_map:
            snapshots[idx + 1] = {kc: snap_sums[kc] / snap_counts[kc] for kc in snap_counts if snap_counts[kc] > 0}

    out = {"n_eval_students": len(eval_students), "deep_ref": DEEP_REF[dataset]}
    # B2 recency horizontal line (per-student, no pooling)
    b2_aucs = []
    for st in eval_students:
        rows = [{"y": r["y"], "pred": r["recent5"] if r["recent5"] is not None else 0.5} for r in st["rows"]]
        a = stratified_auc_rows(rows, "pred")
        if a is not None:
            b2_aucs.append(a)
    out["B2_recency"] = round(float(np.mean(b2_aucs)), 4) if b2_aucs else None

    for n_pool in POOL_SIZES:
        prior = snapshots.get(min(n_pool, len(pool_students)), {})
        aucs, covered, total = [], 0, 0
        for st in eval_students:
            rows = []
            for r in st["rows"]:
                p = prior.get(r["kc"])
                total += 1
                if p is not None:
                    covered += 1
                rows.append({"y": r["y"], "pred": p if p is not None else 0.5})
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                aucs.append(a)
        out[f"pool_{min(n_pool, len(pool_students))}"] = {
            "auc": round(float(np.mean(aucs)), 4) if aucs else None,
            "kc_coverage_of_rows": round(covered / max(total, 1), 4),
        }
    return out


res = {}
for ds in ["assistments17", "junyi2015", "ednet_kt1"]:
    res[ds] = run_domain(ds)
    print(ds, json.dumps(res[ds], ensure_ascii=False), flush=True)

with open("/root/unikt-fork/research/transfer/results/pool_size_scan.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("POOLSIZE-DONE")
