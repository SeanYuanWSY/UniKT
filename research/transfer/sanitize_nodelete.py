"""No-delete sanitized variant (Kimi attack 3(b) defence).

Keep ALL holdout rows; for A1 rows (same-question adjacent, different KC =
explode suspects), score the row with ALL same-question predecessor rows
REMOVED from the input (the copy source excised) instead of deleting the
row. If deep models' A1-row AUC collapses (from ~0.9-1.0 toward ~0.5-0.6)
and the sanitized full-set ordering vs B1 matches the A1-only T1 ordering,
then row deletion == leak sanitization: the table advantage does not
depend on deletion itself.

Identity AKT + transfer AKT on A17 + EdNet (Junyi has no A1 rows).
Per-student AUC, first-300 cohort, same protocol as t1_paired_ci.py.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = {
    "assistments17": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
}
RUNS = {
    "identity_AKT": {
        "assistments17": "runs/normal/AKT_assistments17_20261003-145008_fold0_bs64",
        "ednet_kt1": "runs/normal/AKT_ednet_kt1_20261003-190537_fold0_bs64",
    },
    "transfer_AKT": {ds: "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64" for ds in ALIGN},
}


def fwd_probs(rm, s_seq, s_resp):
    s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
    r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
    return _forward_probs(rm, s, r_, None)[0]


def main(n_users=300):
    out = {}
    for ds, align_name in ALIGN.items():
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        students = build_rows(samples, None)
        prior = fit_global_kc_prior(students)
        alignment = {int(k): v for k, v in load_alignment(align_name).items()}

        # B1 full row set (withA), same cohort
        b1 = {}
        for ws in samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            rows = []
            for i in hold:
                if i == 0:
                    continue
                rows.append({"y": int(ws.response[i]), "pred": prior.get(int(ws.sequence[i]), 0.5)})
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                b1[ws.user_id] = a

        res = {"B1_fullset": round(float(np.mean(list(b1.values()))), 4), "B1_n": len(b1)}
        for mtag, runmap in RUNS.items():
            rm = restore_any(runmap[ds])
            full_aucs, a1_aucs = {}, {}
            for ws in samples:
                if mtag.startswith("identity"):
                    idxs = list(range(len(ws.sequence.tolist())))
                    seq_of = lambda i: int(ws.sequence[i])  # noqa: E731
                else:
                    mapping = {}
                    for kc in set(int(c) for c in ws.sequence.tolist()):
                        ms = alignment.get(kc, [])
                        mapping[kc] = ms[0]["source_kc"] if ms else None
                    idxs = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping.get(int(kc)) is not None]
                    if len(idxs) > 200:
                        idxs = idxs[-200:]
                    if len(idxs) < 6:
                        continue
                    seq_of = lambda i: mapping[int(ws.sequence[i])]  # noqa: E731
                s_seq = [seq_of(i) for i in idxs]
                s_resp = [int(ws.response[i]) for i in idxs]
                probs = fwd_probs(rm, s_seq, s_resp)
                hold_pos = set(ws.holdout_idx.tolist())

                rows_full, rows_a1 = [], []
                for m, i in enumerate(idxs):
                    if i not in hold_pos or m == 0:
                        continue
                    isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                    pred = float(probs[m])
                    if isA1:
                        # excise ALL same-question predecessors (the copy source)
                        keep_pos = [p for p in range(m) if int(ws.question[idxs[p]]) != int(ws.question[i])] + [m]
                        t_seq = [s_seq[p] for p in keep_pos]
                        t_resp = [s_resp[p] for p in keep_pos]
                        pred = float(fwd_probs(rm, t_seq, t_resp)[-1])
                        rows_a1.append({"y": int(ws.response[i]), "pred": pred})
                    row_full = {"y": int(ws.response[i]), "pred": pred}
                    rows_full.append(row_full)
                a = stratified_auc_rows(rows_full, "pred")
                if a is not None:
                    full_aucs[ws.user_id] = a
                a1 = stratified_auc_rows(rows_a1, "pred")
                if a1 is not None:
                    a1_aucs[ws.user_id] = a1
            res[mtag] = {
                "sanitized_fullset": round(float(np.mean(list(full_aucs.values()))), 4) if full_aucs else None,
                "n": len(full_aucs),
                "a1_rows_sanitized": round(float(np.mean(list(a1_aucs.values()))), 4) if a1_aucs else None,
                "a1_n": len(a1_aucs),
            }
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/sanitize_nodelete.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("SANITIZE-NODELETE-DONE")


if __name__ == "__main__":
    main()
