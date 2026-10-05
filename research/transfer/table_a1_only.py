"""B1 table under the A1-only-excluded primary scope (measured, not assumed).

Also B2 recency and the LLM-direct column? B2/LLM-direct condition on the
student's history (not the immediately-adjacent response), so they are
y_prev-unconditional in the copy sense but do use the student's own past.
Primary scope now = exclude A1 (same-question adjacent rows with different
KC — explode suspects). Measure B1 (+B2 for context) under this scope on
all three domains, full rows (table family has no keep constraint).
"""
import json
import sys

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
    for ds in ["assistments17", "junyi2015", "ednet_kt1"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[: n_users + 100]
        students = build_rows(samples, None)
        prior = fit_global_kc_prior(students)

        # bucket labels need raw sample context; build_rows preserves order
        # of surviving samples — recompute survivor order
        surviving = []
        for ws in samples:
            keep_all = len(ws.sequence) >= 1
            rows_n = len(ws.holdout_idx)
            if keep_all and rows_n > 0:
                surviving.append(ws)

        b1_all, b1_noA1, b2_noA1 = [], [], []
        for ws, st in zip(surviving, students):
            hold_pos = set(ws.holdout_idx.tolist())
            rows_all, rows_noA1, rows2_noA1 = [], [], []
            row_iter = [r for r in st["rows"]]  # ordered by holdout index? verify
            # build_rows iterates keep order == sequence order; rows follow holdout ascending
            # recompute per-row bucket directly from ws to stay safe:
            hold_sorted = sorted(hold_pos)
            if len(row_iter) != len(hold_sorted):
                continue
            for r, i in zip(row_iter, hold_sorted):
                isA1 = i > 0 and int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                row_b1 = {"y": r["y"], "pred": prior.get(r["kc"], 0.5)}
                row_b2 = {"y": r["y"], "pred": r["recent5"] if r["recent5"] is not None else prior.get(r["kc"], 0.5)}
                rows_all.append(row_b1)
                if not isA1:
                    rows_noA1.append(row_b1)
                    rows2_noA1.append(row_b2)
            a = stratified_auc_rows(rows_all, "pred")
            a1 = stratified_auc_rows(rows_noA1, "pred")
            a2 = stratified_auc_rows(rows2_noA1, "pred")
            if a is not None:
                b1_all.append(a)
            if a1 is not None:
                b1_noA1.append(a1)
            if a2 is not None:
                b2_noA1.append(a2)
        out[ds] = {
            "B1_withA": round(float(np.mean(b1_all)), 4) if b1_all else None,
            "B1_A1excluded": round(float(np.mean(b1_noA1)), 4) if b1_noA1 else None,
            "B2_A1excluded": round(float(np.mean(b2_noA1)), 4) if b2_noA1 else None,
        }
        print(ds, json.dumps(out[ds], ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/table_a1_scope.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("TABLE-A1-DONE")


if __name__ == "__main__":
    main()
