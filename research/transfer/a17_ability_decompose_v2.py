"""A17 deep-advantage decomposition v2 (review-fixed).

v1 was degenerate: ability was constant within each student's holdout, so
within-student AUC (stratified) was invariant to the blend -> delta==0.
v2: ability is time-varying inside holdout — for each scored row i, the
mean of the last <=20 predecessor responses EXCLUDING same-question rows
(artifact channel cut, same discipline as sanitize_nodelete). Deep models
see holdout-prefix responses via teacher forcing (minus the copy channel);
this matches their information set. alpha selected on evidence-segment
pseudo-holdout (every 5th evidence row, A1 rows skipped), criterion =
mean of per-student AUCs. Eval: t1ci protocol (first-300, A1-excluded,
per-student AUC). rows1 must reproduce t1ci B1_mean (cross-validation).
"""
import json
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
WIN, MIN_HIST, CAP = 20, 3, 100


def ability_before(hist_deque, q_i):
    """Mean of last <=WIN predecessor responses excluding same-question rows."""
    vals = [y for (qj, y) in list(hist_deque)[-CAP:] if qj != q_i][-WIN:]
    return float(np.mean(vals)) if len(vals) >= MIN_HIST else None


def main(n_users=300):
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        students = build_rows(samples, None)
        prior = fit_global_kc_prior(students)

        per_student_auc = {a: {} for a in ALPHAS}  # alpha -> uid -> auc (selection)
        b1, b1ab = {}, {}
        for ws in samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            hold_set = set(hold)
            q_all = [int(x) for x in ws.question.tolist()]
            kc_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            L = len(kc_all)

            # --- selection pass: evidence-segment pseudo-holdout (every 5th evidence row, skip A1)
            hist = deque()
            sel_rows = {a: [] for a in ALPHAS}
            for i in range(L):
                if i not in hold_set and i > 0 and i % 5 == 0:
                    isA1 = q_all[i] == q_all[i - 1] and kc_all[i] != kc_all[i - 1]
                    if not isA1:
                        ab = ability_before(hist, q_all[i])
                        if ab is not None:
                            p1 = prior.get(kc_all[i], 0.5)
                            for a in ALPHAS:
                                sel_rows[a].append({"y": y_all[i], "pred": a * p1 + (1 - a) * ab})
                if i not in hold_set:
                    hist.append((q_all[i], y_all[i]))
            for a in ALPHAS:
                auc = stratified_auc_rows(sel_rows[a], "pred")
                if auc is not None:
                    per_student_auc[a][ws.user_id] = auc

        alpha_star = max(ALPHAS, key=lambda a: np.mean(list(per_student_auc[a].values())) if per_student_auc[a] else -1)

        # --- eval pass (separate loop; alpha frozen before touching any holdout row)
        for ws in samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            hold_set = set(hold)
            q_all = [int(x) for x in ws.question.tolist()]
            kc_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            L = len(kc_all)
            hist = deque()
            rows1, rows2 = [], []
            for i in range(L):
                if i in hold_set and i > 0:
                    isA1 = q_all[i] == q_all[i - 1] and kc_all[i] != kc_all[i - 1]
                    if not isA1:
                        p1 = prior.get(kc_all[i], 0.5)
                        rows1.append({"y": y_all[i], "pred": p1})
                        ab = ability_before(hist, q_all[i])  # holdout prefix in window, same-question rows cut inside
                        rows2.append({"y": y_all[i], "pred": alpha_star * p1 + (1 - alpha_star) * ab if ab is not None else p1})
                # every row enters the window (incl. holdout prefix and A1 rows):
                # same-question predecessors are excluded per-row inside ability_before,
                # which is what makes ability time-varying within the holdout segment
                hist.append((q_all[i], y_all[i]))
            a1 = stratified_auc_rows(rows1, "pred")
            a2 = stratified_auc_rows(rows2, "pred")
            if a1 is not None:
                b1[ws.user_id] = a1
            if a2 is not None:
                b1ab[ws.user_id] = a2

        res = {
            "alpha_star": alpha_star,
            "B1_A1excluded": round(float(np.mean(list(b1.values()))), 4),
            "B1_plus_ability": round(float(np.mean(list(b1ab.values()))), 4),
            "n": len(b1ab),
            "delta": round(float(np.mean(list(b1ab.values()))) - float(np.mean(list(b1.values()))), 4),
        }
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a17_ability_decompose_v2.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("ABILITY-DECOMPOSE-V2-DONE")


if __name__ == "__main__":
    main()
