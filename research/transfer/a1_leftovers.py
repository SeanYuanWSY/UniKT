"""A1-only leftovers: Junyi table scope unification + identity DKT columns.

1) Junyi B1 on the standard pool (300 users, same as earlier 0.562 runs) but
   scored A1-only (should equal withA since Junyi has no explode).
2) identity DKT (all three domains) under withA / A1-excluded / A-excluded —
   completes the T1 table's DKT rows.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

NATIVE_DKT = {
    "assistments17": "runs/normal/DKT_assistments17_20261003-144042_fold0_bs128",
    "ednet_kt1": "runs/normal/DKT_ednet_kt1_20261003-190327_fold0_bs128",
    "junyi2015": "runs/normal/DKT_junyi2015_20261004-195512_fold0_bs128",
}


def main(n_users=250):
    out = {}
    # 1) Junyi table standard-scope
    src = get_data_source(_make_rc("junyi2015"))
    samples = load_user_samples(src, fold=0)[:300]
    students = build_rows(samples, None)
    prior = fit_global_kc_prior(students)
    t_all, t_noA1 = [], []
    surviving = [ws for ws in samples if len(ws.holdout_idx) > 0][: len(students)]
    for ws, st in zip(surviving, students):
        hold = sorted(set(ws.holdout_idx.tolist()))
        if len(st["rows"]) != len(hold):
            continue
        ra, rn = [], []
        for r, i in zip(st["rows"], hold):
            isA1 = i > 0 and int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
            row = {"y": r["y"], "pred": prior.get(r["kc"], 0.5)}
            ra.append(row)
            if not isA1:
                rn.append(row)
        a = stratified_auc_rows(ra, "pred")
        a1 = stratified_auc_rows(rn, "pred")
        if a is not None:
            t_all.append(a)
        if a1 is not None:
            t_noA1.append(a1)
    out["junyi_table_std"] = {
        "withA": round(float(np.mean(t_all)), 4),
        "A1excluded": round(float(np.mean(t_noA1)), 4),
        "n": len(t_all),
    }

    # 2) identity DKT three-scope
    for ds, run in NATIVE_DKT.items():
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        rm = restore_any(run)
        auc_all, auc_noA1, auc_noA = [], [], []
        for ws in samples:
            if len(ws.sequence) < 6:
                continue
            seq = ws.sequence.tolist()
            resp = [int(x) for x in ws.response.tolist()]
            s = torch.tensor([seq], dtype=torch.long, device=rm.device)
            r_ = torch.tensor([resp], dtype=torch.long, device=rm.device)
            probs = _forward_probs(rm, s, r_, None)[0]
            ra, rn1, rn = [], [], []
            for i in sorted(set(ws.holdout_idx.tolist())):
                if i == 0:
                    continue
                sameq = int(ws.question[i]) == int(ws.question[i - 1])
                isA2 = sameq and seq[i] == seq[i - 1]
                isA1 = sameq and not isA2
                row = {"y": resp[i], "pred": float(probs[i])}
                ra.append(row)
                if not isA1:
                    rn1.append(row)
                if not (isA1 or isA2):
                    rn.append(row)
            a = stratified_auc_rows(ra, "pred")
            a1 = stratified_auc_rows(rn1, "pred")
            a2 = stratified_auc_rows(rn, "pred")
            if a is not None:
                auc_all.append(a)
            if a1 is not None:
                auc_noA1.append(a1)
            if a2 is not None:
                auc_noA.append(a2)
        out[f"identity_dkt_{ds}"] = {
            "withA": round(float(np.mean(auc_all)), 4),
            "A1excluded": round(float(np.mean(auc_noA1)), 4),
            "A_excluded": round(float(np.mean(auc_noA)), 4),
        }
        print(ds, out[f"identity_dkt_{ds}"], flush=True)

    print(json.dumps(out["junyi_table_std"], ensure_ascii=False), flush=True)
    with open("/root/unikt-fork/research/transfer/results/a1_leftovers.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("A1LEFT-DONE")


if __name__ == "__main__":
    main()
