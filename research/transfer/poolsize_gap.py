"""Batch25: pool-size curve gap fill (n=1/3/5) + Junyi full curve.

POOL-SIZE (poolsize_a1.py) measured pool 10/30/100; PS-0 added n=0 (~chance).
Missing: the cold-start regime n=1/3/5 where the curve either jumps past
chance with ONE student's evidence or creeps. Protocol identical to
poolsize_a1.py: pool students = all minus tail-100 eval; n_pool random
draws (10 draws, rng 1000+seed, same as existing points); prior = pooled
KC marginals, 0.5 fallback; eval tail-100 A1-excluded per-student AUC.
New: draw_min/draw_max spread fields (n=1 draw variance = student
heterogeneity signal). A17+EdNet get 1/3/5 (gap fill); Junyi gets the
full 1/3/5/10/30/100 curve (best-alignment domain, complete cold-start
story). No deep reference rerun (measured: 0.4039/0.2723 A1-excluded).
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

DOMAINS = {
    "assistments17": [1, 3, 5],
    "ednet_kt1": [1, 3, 5],
    "junyi2015": [1, 3, 5, 10, 30, 100],
}
N0_REF = {"assistments17": 0.478, "ednet_kt1": 0.4988, "junyi2015": 0.5211}  # ps0_aligned_table.json


def main():
    out = {}
    for ds, sizes in DOMAINS.items():
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval_actual = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval_actual:]
        pool_samples = samples[: len(samples) - n_eval_actual]

        pool_hists = [st["kc_hist"] for st in build_rows(pool_samples, None)]

        res = {"n_eval": len(eval_samples), "n_pool_available": len(pool_hists), "pool_0_ref": N0_REF[ds]}
        for n_pool in sizes:
            draws = []
            for seed in range(10):
                r = np.random.default_rng(1000 + seed)
                idx = r.choice(len(pool_hists), size=min(n_pool, len(pool_hists)), replace=False)
                counts, sums = defaultdict(int), defaultdict(int)
                for i in idx:
                    for kc, hist in pool_hists[i].items():
                        counts[kc] += len(hist)
                        sums[kc] += sum(hist)
                prior = {kc: sums[kc] / counts[kc] for kc in counts if counts[kc] > 0}
                aucs = []
                for ws in eval_samples:
                    hold = sorted(set(ws.holdout_idx.tolist()))
                    rows = []
                    for i in hold:
                        if i == 0:
                            continue
                        isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                        if not isA1:
                            rows.append({"y": int(ws.response[i]), "pred": prior.get(int(ws.sequence[i]), 0.5)})
                    a = stratified_auc_rows(rows, "pred")
                    if a is not None:
                        aucs.append(a)
                draws.append(float(np.mean(aucs)) if aucs else float("nan"))
            res[f"pool_{min(n_pool, len(pool_hists))}"] = {
                "mean": round(float(np.mean(draws)), 4),
                "draw_min": round(float(np.min(draws)), 4),
                "draw_max": round(float(np.max(draws)), 4),
            }
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/poolsize_gap.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("POOLGAP-DONE")


if __name__ == "__main__":
    main()
