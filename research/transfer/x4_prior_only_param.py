"""X4: mechanism decomposition replicated on the k3-high alignment.

prior_only_test.py hardcoded the flash cache; this parameterised rerun checks
whether (priors-only > full transplant) holds at high coverage too.
"""
import collections
import sys

import numpy as np
import torch

ALIGN = sys.argv[1] if len(sys.argv) > 1 else "assistments09__to__assistments17__llm__k3-high__c25k3__rep0"
SRC_RUN = sys.argv[2] if len(sys.argv) > 2 else "runs/normal/DKT_assistments09_20261003-024700_fold0_bs128"

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

alignment = {int(k): v for k, v in load_alignment(ALIGN).items()}
rm = restore_any(SRC_RUN)
tgt = get_data_source(_make_rc("assistments17"))
samples = load_user_samples(tgt, fold=0)[:150]

per_kc_pred = collections.defaultdict(list)
per_student = []
for ws in samples:
    keep = [i for i, kc in enumerate(ws.sequence.tolist())
            if alignment.get(int(kc))]
    if len(keep) < 6:
        continue
    seq_m = [alignment[int(ws.sequence[i])][0]["source_kc"] for i in keep]
    resp_m = [int(ws.response[i]) for i in keep]
    s = torch.tensor([seq_m], dtype=torch.long, device=rm.device)
    r_ = torch.tensor([resp_m], dtype=torch.long, device=rm.device)
    probs = _forward_probs(rm, s, r_, None)[0]
    holdout_pos = set(ws.holdout_idx.tolist())
    rows = []
    for j, i in enumerate(keep):
        if i in holdout_pos:
            rows.append({"y": int(ws.response[i]), "pred": float(probs[j]), "kc": int(ws.sequence[i])})
            per_kc_pred[int(ws.sequence[i])].append(float(probs[j]))
    if rows and 0 < sum(x["y"] for x in rows) < len(rows):
        per_student.append(rows)

prior = {k: float(np.mean(v)) for k, v in per_kc_pred.items() if len(v) >= 5}

aucs_full, auc_prior = [], []
for rows in per_student:
    a = stratified_auc_rows(rows, "pred")
    if a is not None:
        aucs_full.append(a)
    rows2 = [{"y": x["y"], "pred": prior[x["kc"]]} for x in rows if x["kc"] in prior]
    if rows2 and 0 < sum(x["y"] for x in rows2) < len(rows2):
        a = stratified_auc_rows(rows2, "pred")
        if a is not None:
            auc_prior.append(a)

print(f"alignment: {ALIGN}")
print(f"full transplanted model : {np.mean(aucs_full):.4f} (n={len(aucs_full)})")
print(f"priors-only (no hidden) : {np.mean(auc_prior):.4f} (n={len(auc_prior)})")
print(f"delta (full - prior)    : {np.mean(aucs_full) - np.mean(auc_prior):+.4f}")
print("X4-PARAM-DONE")
