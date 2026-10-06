"""A17 deep-advantage decomposition: is it student heterogeneity?

A17 is the only domain where native deep beats B1 (0.584/0.633 vs 0.548).
Test: add ONE global scalar per student (rolling mean correctness,
evidence-side only) to the table. alpha selected on within-history rows
(same discipline as b234's alpha_star), eval on holdout A1-excluded rows,
t1ci protocol. If B1+ability closes most of the gap on A17, the deep
advantage is student-level heterogeneity, not sequence structure.
Contrast domains: EdNet / Junyi. CPU-only.
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


def main(n_users=300):
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        students = build_rows(samples, None)
        prior = fit_global_kc_prior(students)

        b1, b1ab = {}, {}
        alpha_scores = defaultdict(list)
        for ws in samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            hold_set = set(hold)
            # evidence-side global rolling ability, reset per student
            hist = []
            ev_rows = []  # (kc, y, ability_before) within-history pseudo-holdout
            for i in range(len(ws.sequence)):
                y = int(ws.response[i])
                if i in hold_set and i > 0:
                    ab = float(np.mean(hist[-20:])) if len(hist) >= 3 else None
                    ev_rows.append((int(ws.sequence[i]), y, ab))
                else:
                    hist.append(y)

            # alpha selection on evidence-side rows (prior vs ability blend)
            for kc, y, ab in ev_rows:
                if ab is None:
                    continue
                for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                    alpha_scores[alpha].append({"y": y, "pred": alpha * prior.get(kc, 0.5) + (1 - alpha) * ab})
            alpha_star = max(
                (a for a in (0.0, 0.25, 0.5, 0.75, 1.0) if stratified_auc_rows(alpha_scores[a], "pred") is not None),
                key=lambda a: stratified_auc_rows(alpha_scores[a], "pred"),
            )

            rows1, rows2 = [], []
            hist = []
            for i in range(len(ws.sequence)):
                y = int(ws.response[i])
                if i in hold_set and i > 0:
                    isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                    if not isA1:
                        p1 = prior.get(int(ws.sequence[i]), 0.5)
                        rows1.append({"y": y, "pred": p1})
                        ab = float(np.mean(hist[-20:])) if len(hist) >= 3 else None
                        rows2.append({"y": y, "pred": alpha_star * p1 + (1 - alpha_star) * ab if ab is not None else p1})
                else:
                    hist.append(y)
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

    with open("/root/unikt-fork/research/transfer/results/a17_ability_decompose.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("ABILITY-DECOMPOSE-DONE")


if __name__ == "__main__":
    main()
