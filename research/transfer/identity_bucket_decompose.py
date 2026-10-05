"""Identity same-question bucket decomposition (top priority, Batch 9 follow-up).

Does the label-copy artifact (multi-KC exploded rows sharing one label) also
inflate the NATIVE identity upper bounds? For each domain, run the native
DKT/AKT on its own students (mx_identity protocol) and decompose holdout
within-student AUC into the same three buckets (A same-question / B same-KC
diff-question / C diff-KC). If EdNet identity A-bucket ~0.99 at ~52% share,
the headline identity numbers are artifact-inflated and every deep number in
T1/T2 needs an artifact-corrected companion figure (report A-excluded AUC).
Also computes, for reference, the A-excluded AUC for the frozen-transfer deep
numbers (A09->A17 AKT) in the same pass where cheap.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from kc_align import load_alignment  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

NATIVE = {
    "assistments17": [
        ("DKT", "runs/normal/DKT_assistments17_20261003-144042_fold0_bs128"),
        ("AKT", "runs/normal/AKT_assistments17_20261003-145008_fold0_bs64"),
    ],
    "ednet_kt1": [
        ("DKT", "runs/normal/DKT_ednet_kt1_20261003-190327_fold0_bs128"),
        ("AKT", "runs/normal/AKT_ednet_kt1_20261003-190537_fold0_bs64"),
    ],
    "junyi2015": [
        ("DKT", "runs/normal/DKT_junyi2015_20261004-195512_fold0_bs128"),
        ("AKT", "runs/normal/AKT_junyi2015_20261004-195553_fold0_bs64"),
    ],
}


def decompose(rows_all):
    """rows_all: list per student of rows with y, pred, bucket."""
    per_bucket = {"A": [], "B": [], "C": []}
    noA = []
    for rows in rows_all:
        for b in "ABC":
            sub = [{"y": r["y"], "pred": r["pred"]} for r in rows if r["bucket"] == b]
            a = stratified_auc_rows(sub, "pred")
            if a is not None:
                per_bucket[b].append(a)
        sub = [{"y": r["y"], "pred": r["pred"]} for r in rows if r["bucket"] != "A"]
        a = stratified_auc_rows(sub, "pred")
        if a is not None:
            noA.append(a)
    out = {}
    for b in "ABC":
        out[f"bucket_{b}_auc"] = round(float(np.mean(per_bucket[b])), 4) if per_bucket[b] else None
        out[f"bucket_{b}_n"] = len(per_bucket[b])
    out["A_excluded_auc"] = round(float(np.mean(noA)), 4) if noA else None
    out["A_excluded_n"] = len(noA)
    return out


def run(dataset, n_users=250):
    src = get_data_source(_make_rc(dataset))
    samples = load_user_samples(src, fold=0)[:n_users]
    res = {}
    for tag, run_dir in NATIVE[dataset]:
        rm = restore_any(run_dir)
        rows_all = []
        shares = {"A": 0, "B": 0, "C": 0}
        for ws in samples:
            if len(ws.sequence) < 6:
                continue
            seq = ws.sequence.tolist()
            resp = [int(x) for x in ws.response.tolist()]
            s = torch.tensor([seq], dtype=torch.long, device=rm.device)
            r_ = torch.tensor([resp], dtype=torch.long, device=rm.device)
            probs = _forward_probs(rm, s, r_, None)[0]
            rows = []
            for i in set(ws.holdout_idx.tolist()):
                if i == 0:
                    continue
                b = (
                    "A"
                    if int(ws.question[i]) == int(ws.question[i - 1])
                    else ("B" if seq[i] == seq[i - 1] else "C")
                )
                shares[b] += 1
                rows.append({"y": resp[i], "pred": float(probs[i]), "bucket": b})
            if rows:
                rows_all.append(rows)
        total = sum(shares.values())
        d = decompose(rows_all)
        d["shares"] = {k: round(v / max(total, 1), 4) for k, v in shares.items()}
        res[tag] = d
        print(dataset, tag, json.dumps(d, ensure_ascii=False), flush=True)
    return res


out = {}
for ds in ["ednet_kt1", "assistments17", "junyi2015"]:
    out[ds] = run(ds)
with open("/root/unikt-fork/research/transfer/results/identity_bucket_decompose.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("IDENTITY-BUCKET-DONE")
