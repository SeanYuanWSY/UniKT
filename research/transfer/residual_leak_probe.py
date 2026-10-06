"""Residual-leakage degenerate predictor (review blocker #1).

On the A1-only-excluded surviving rows, a predictor that simply copies
y_{t-1} (the true previous response). Its within-student AUC quantifies
residual leakage beyond explode copies (junction pairs, autocorrelation).
Gate: >>0.5 means A1-only is insufficient; report the bound and consider
harder exclusion. Also: copy-y_{t-1} on non-A rows only (pure residual) vs
on A2 rows (true repeats, expected high for genuine autocorrelation).
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


def main(n_users=300):
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        aucs_noA1 = []  # copy predictor on A1-excluded rows
        aucs_nonA = []  # copy predictor on non-A rows (pure residual)
        aucs_A2 = []  # copy on A2 rows alone (genuine repeat autocorr)
        for ws in samples:
            seq = ws.sequence.tolist()
            resp = [int(x) for x in ws.response.tolist()]
            hold = sorted(i for i in set(ws.holdout_idx.tolist()) if i > 0)
            rows_noA1, rows_nonA, rows_A2 = [], [], []
            for i in hold:
                sameq = int(ws.question[i]) == int(ws.question[i - 1])
                a2 = sameq and seq[i] == seq[i - 1]
                a1 = sameq and not a2
                row = {"y": resp[i], "pred": float(resp[i - 1])}
                if not a1:
                    rows_noA1.append(row)
                if not (a1 or a2):
                    rows_nonA.append(row)
                if a2:
                    rows_A2.append(row)
            for rows, acc in ((rows_noA1, aucs_noA1), (rows_nonA, aucs_nonA), (rows_A2, aucs_A2)):
                a = stratified_auc_rows(rows, "pred")
                if a is not None:
                    acc.append(a)
        out[ds] = {
            "copy_y_prev_on_A1excluded": round(float(np.mean(aucs_noA1)), 4) if aucs_noA1 else None,
            "copy_y_prev_on_nonA": round(float(np.mean(aucs_nonA)), 4) if aucs_nonA else None,
            "copy_y_prev_on_A2_only": round(float(np.mean(aucs_A2)), 4) if aucs_A2 else None,
        }
        print(ds, json.dumps(out[ds], ensure_ascii=False), flush=True)
    with open("/root/unikt-fork/research/transfer/results/residual_leak_probe.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("RESIDUAL-DONE")


if __name__ == "__main__":
    main()
