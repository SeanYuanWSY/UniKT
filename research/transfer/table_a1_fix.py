"""B1 table withA vs A1-excluded — independent traversal (kills zip misalign).

Replaces table_a1_scope.json (whose zip(surviving, students) /
zip(row_iter, hold_sorted) pairing was misalignment-prone and produced the
deprecated Junyi 0.513 with n=15). Protocol identical to t1_paired_ci.py's
B1 loop: per-student AUC over holdout rows, prior from same 300-student
pool, plus a withA pass. Reports n per scope.
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
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        students = build_rows(samples, None)
        prior = fit_global_kc_prior(students)

        b1_with, b1_noa1 = {}, {}
        for ws in samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            rows_all, rows_noa1 = [], []
            for i in hold:
                if i == 0:
                    continue
                pred = prior.get(int(ws.sequence[i]), 0.5)
                row = {"y": int(ws.response[i]), "pred": pred}
                rows_all.append(row)
                isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                if not isA1:
                    rows_noa1.append(row)
            a = stratified_auc_rows(rows_all, "pred")
            a1 = stratified_auc_rows(rows_noa1, "pred")
            if a is not None:
                b1_with[ws.user_id] = a
            if a1 is not None:
                b1_noa1[ws.user_id] = a1

        out[ds] = {
            "B1_withA": round(float(np.mean(list(b1_with.values()))), 4) if b1_with else None,
            "B1_withA_n": len(b1_with),
            "B1_A1excluded": round(float(np.mean(list(b1_noa1.values()))), 4) if b1_noa1 else None,
            "B1_A1excluded_n": len(b1_noa1),
        }
        print(ds, json.dumps(out[ds], ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/table_a1_fix.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("TABLE-A1-FIX-DONE")


if __name__ == "__main__":
    main()
