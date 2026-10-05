"""A-bucket purity check (review-mandated #1): KC-equality split.

Within same-question adjacent rows (bucket A):
  A1 = kc differs (explode suspects: explode groups have strictly distinct KCs)
  A2 = kc equal   (definite true repeats; explode cannot produce these)
Report shares and per-sub-bucket AUC for identity (AKT native) and frozen
transfer, plus "A1-only excluded" scores. Junyi code confirms one skill per
question (no explode) -> its A bucket should be ~all A2 (validation).
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
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

CFG = {
    "assistments17": {
        "align": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
        "native": "runs/normal/AKT_assistments17_20261003-145008_fold0_bs64",
    },
    "ednet_kt1": {
        "align": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
        "native": "runs/normal/AKT_ednet_kt1_20261003-190537_fold0_bs64",
    },
    "junyi2015": {
        "align": "assistments09__to__junyi2015__llm__k3-high__c25k3__rep0",
        "native": "runs/normal/AKT_junyi2015_20261004-195553_fold0_bs64",
    },
}
TRANSFER = "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64"


def main(n_users=250):
    out = {}
    for ds, cfg in CFG.items():
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        alignment = {int(k): v for k, v in load_alignment(cfg["align"]).items()}

        a1_cnt = a2_cnt = tot = 0
        res = {}
        for mtag, run in [("identity", cfg["native"]), ("transfer", TRANSFER)]:
            rm = restore_any(run)
            auc_all, auc_noA1, auc_noA = [], [], []
            a1_aucs, a2_aucs = [], []
            for ws in samples:
                if len(ws.sequence) < 6:
                    continue
                seq = ws.sequence.tolist()
                resp = [int(x) for x in ws.response.tolist()]
                if mtag == "identity":
                    s_seq, s_resp = seq, resp
                    hold_pos = set(ws.holdout_idx.tolist())
                    jmap = {i: i for i in range(len(seq))}
                else:
                    mapping = {}
                    for kc in set(int(c) for c in seq):
                        ms = alignment.get(kc, [])
                        mapping[kc] = ms[0]["source_kc"] if ms else None
                    keep = [i for i, kc in enumerate(seq) if mapping.get(int(kc)) is not None]
                    if len(keep) > 200:
                        keep = keep[-200:]
                    if len(keep) < 6:
                        continue
                    s_seq = [mapping[int(seq[i])] for i in keep]
                    s_resp = [resp[i] for i in keep]
                    hold_pos = {i for i in ws.holdout_idx.tolist() if i in set(keep)}
                    jmap = {i: j for j, i in enumerate(keep)}
                s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
                r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
                probs = _forward_probs(rm, s, r_, None)[0]
                rows_all, rows_noA1, rows_noA = [], [], []
                st_a1, st_a2 = [], []
                for i in sorted(hold_pos):
                    if i == 0:
                        continue
                    j = jmap[i]
                    row = {"y": resp[i], "pred": float(probs[j])}
                    rows_all.append(row)
                    sameq = int(ws.question[i]) == int(ws.question[i - 1])
                    if sameq:
                        if seq[i] == seq[i - 1]:
                            a2 = True
                        else:
                            a2 = False
                        if a2:
                            st_a2.append(row)
                            if mtag == "identity":
                                a2_cnt += 1
                        else:
                            st_a1.append(row)
                            if mtag == "identity":
                                a1_cnt += 1
                        tot += 1 if mtag == "identity" else 0
                    else:
                        pass
                # rebuild exclusion variants cleanly from labelled rows
                labelled = []
                for i in sorted(hold_pos):
                    if i == 0:
                        continue
                    j = jmap[i]
                    sameq = int(ws.question[i]) == int(ws.question[i - 1])
                    a2row = sameq and seq[i] == seq[i - 1]
                    a1row = sameq and not a2row
                    labelled.append((a1row, a2row, {"y": resp[i], "pred": float(probs[j])}))
                rows_noA = [r for a1, a2, r in labelled if not (a1 or a2)]
                rows_noA1 = [r for a1, a2, r in labelled if not a1]
                rows_all = [r for _, _, r in labelled]
                # fix: rows_noA1 should EXCLUDE A1 only (keep A2 and non-A)
                a_all = stratified_auc_rows(rows_all, "pred")
                a_noA = stratified_auc_rows(rows_noA, "pred")
                a_noA1 = stratified_auc_rows(rows_noA1, "pred")
                a1a = stratified_auc_rows(st_a1, "pred")
                a2a = stratified_auc_rows(st_a2, "pred")
                if a_all is not None:
                    auc_all.append(a_all)
                if a_noA is not None:
                    auc_noA.append(a_noA)
                if a_noA1 is not None:
                    auc_noA1.append(a_noA1)
                if a1a is not None:
                    a1_aucs.append(a1a)
                if a2a is not None:
                    a2_aucs.append(a2a)
            res[mtag] = {
                "withA": round(float(np.mean(auc_all)), 4) if auc_all else None,
                "A_excluded": round(float(np.mean(auc_noA)), 4) if auc_noA else None,
                "A1only_excluded": round(float(np.mean(auc_noA1)), 4) if auc_noA1 else None,
                "A1_explode_suspect_auc": round(float(np.mean(a1_aucs)), 4) if a1_aucs else None,
                "A2_true_repeat_auc": round(float(np.mean(a2_aucs)), 4) if a2_aucs else None,
            }
        res["A_bucket_split"] = {"A1_share": round(a1_cnt / max(tot, 1), 4), "A2_share": round(a2_cnt / max(tot, 1), 4)}
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a_bucket_kc_split.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("AKCSPLIT-DONE")


if __name__ == "__main__":
    main()
