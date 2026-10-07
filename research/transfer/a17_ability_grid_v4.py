"""Batch 44: ability-decompose grid-fine + fallback count (Codex b42b P2-3/P3-2 closure).

Two reviewer-noted gaps from the v3.1 adjudication:
 (1) the alpha grid {0,.25,.5,.75,1} is coarse — the boundary selection
     alpha*=1.0 was never tested against 0.9/0.95 (near-boundary interior);
 (2) the alpha=0 endpoint arm falls back to the prior when the ability
     window is short, but the fallback count was never recorded.

Design (identical leak-free construction to v3.1, frozen):
 - prior_tuning: all students, holdout-run-complete + tuning-run-complete
   exclusion (run_ids maximal consecutive equal-q blocks);
 - selection grid extended to {0, .25, .5, .75, .9, .95, 1};
 - anchors: alpha*=1.0 on the coarse grid points must reproduce v3.1's
   tuning means (+-0.0005) and its alpha*=1.0 selection;
 - eval_alpha0 records n_fallback_rows / n_scored_rows.

Output: results/a17_ability_grid_v4.json
"""
import json
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
WIN, MIN_HIST, CAP = 20, 3, 100
# v3.1 anchors (a17_ability_decompose_v3.json, tag ability3b)
V31_ALPHA1_MEANS = {"assistments17": 0.537, "ednet_kt1": 0.5974, "junyi2015": 0.6098}


def ability_before(hist_deque, q_i):
    vals = [y for (qj, y) in list(hist_deque)[-CAP:] if qj != q_i][-WIN:]
    return float(np.mean(vals)) if len(vals) >= MIN_HIST else None


def is_a1(q_all, kc_all, i):
    return i > 0 and q_all[i] == q_all[i - 1] and kc_all[i] != kc_all[i - 1]


def run_ids(q_all):
    rid = [0] * len(q_all)
    r = 0
    for i in range(1, len(q_all)):
        if q_all[i] != q_all[i - 1]:
            r += 1
        rid[i] = r
    return rid


def main(n_users=300):
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]

        # leak-free prior_tuning — identical rule to v3.1
        prior_tuning = {}
        tuning_idx = {}
        for ws in samples:
            hold_set = set(ws.holdout_idx.tolist())
            q_all = [int(x) for x in ws.question.tolist()]
            kc_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            rid = run_ids(q_all)
            tset = [
                i
                for i in range(len(kc_all))
                if i not in hold_set and i > 0 and i % 5 == 0 and not is_a1(q_all, kc_all, i)
            ]
            tuning_idx[ws.user_id] = tset
            excl_runs = {rid[i] for i in hold_set} | {rid[i] for i in tset}
            for i in range(len(kc_all)):
                if rid[i] not in excl_runs:
                    prior_tuning.setdefault(kc_all[i], []).append(y_all[i])
        prior_tuning = {k: float(np.mean(v)) for k, v in prior_tuning.items() if v}

        # selection over the fine grid
        per_stu = {a: {} for a in ALPHAS}
        n_rows_scored = 0
        for ws in samples:
            q_all = [int(x) for x in ws.question.tolist()]
            kc_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            hold_set = set(ws.holdout_idx.tolist())
            hist = deque()
            rows_a = {a: [] for a in ALPHAS}
            tuning_here = set(tuning_idx[ws.user_id])
            for i in range(len(kc_all)):
                if i not in hold_set and i in tuning_here:
                    ab = ability_before(hist, q_all[i])
                    if ab is not None:
                        p1 = prior_tuning.get(kc_all[i], 0.5)
                        n_rows_scored += 1
                        for a in ALPHAS:
                            rows_a[a].append({"y": y_all[i], "pred": a * p1 + (1 - a) * ab})
                if i not in hold_set:
                    hist.append((q_all[i], y_all[i]))
            for a in ALPHAS:
                auc = stratified_auc_rows(rows_a[a], "pred")
                if auc is not None:
                    per_stu[a][ws.user_id] = auc
        means = {a: float(np.mean(list(v.values()))) if v else -1.0 for a, v in per_stu.items()}
        alpha_star = max(ALPHAS, key=lambda a: means[a])

        # alpha=0 endpoint arm on holdout: count prior-fallback rows (P3-2)
        n_fallback = 0
        n_scored = 0
        per_stu0 = {}
        for ws in samples:
            hold_set = set(ws.holdout_idx.tolist())
            q_all = [int(x) for x in ws.question.tolist()]
            kc_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            hist = deque()
            rows0 = []
            for i in range(len(kc_all)):
                if i in hold_set and i > 0 and not is_a1(q_all, kc_all, i):
                    ab = ability_before(hist, q_all[i])
                    n_scored += 1
                    if ab is None:
                        n_fallback += 1  # falls back to prior p1 (v3.1 behaviour)
                        rows0.append({"y": y_all[i], "pred": prior_tuning.get(kc_all[i], 0.5)})
                    else:
                        rows0.append({"y": y_all[i], "pred": ab})
                hist.append((q_all[i], y_all[i]))
            a0 = stratified_auc_rows(rows0, "pred")
            if a0 is not None:
                per_stu0[ws.user_id] = a0

        res = {
            "alpha_star_fine": alpha_star,
            "tuning_means_fine": {str(a): round(m, 4) for a, m in means.items()},
            "n_tuning_rows_scored": n_rows_scored,
            "n_students_valid_at_star": len(per_stu[alpha_star]),
            "selfcheck_v31_alpha1_mean": abs(means[1.0] - V31_ALPHA1_MEANS[ds]) <= 0.0005,
            "alpha0_holdout": {
                "mean_per_stu_auc": round(float(np.mean(list(per_stu0.values()))), 4) if per_stu0 else None,
                "n_students": len(per_stu0),
                "n_scored_rows": n_scored,
                "n_fallback_rows": n_fallback,
                "fallback_share": round(n_fallback / n_scored, 4) if n_scored else None,
            },
        }
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a17_ability_grid_v4.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("ABILITY-GRID-V4-DONE")


if __name__ == "__main__":
    main()
