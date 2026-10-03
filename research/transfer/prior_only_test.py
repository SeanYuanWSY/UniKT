"""Decomposition test: priors-only vs full-model within-student AUC (A09->A17).

If priors-only (per-KC constant prediction, no hidden state) scores ABOVE the
full transplanted model, the recurrent student-conditional dynamics are what
anti-transfers; the vocabulary/KC-prior part transfers fine.
"""
import collections
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment
from kc_tables import _make_rc
from signals import _forward_probs
from transfer_eval import stratified_auc_rows
from llm_explain_restore_shim import restore_any
from restore import load_user_samples
from utils.data_process import get_data_source

alignment = {int(k): v for k, v in load_alignment("assistments09__to__assistments17__llm__glm__rep0").items()}
rm = restore_any("runs/normal/DKT_assistments09_20261003-024700_fold0_bs128")
tgt = get_data_source(_make_rc("assistments17"))
samples = load_user_samples(tgt, fold=0)[:150]

mapping = {}
all_kcs = set()
for ws in samples:
    all_kcs |= set(int(c) for c in ws.sequence.tolist())
for kc in all_kcs:
    ms = alignment.get(kc, [])
    mapping[kc] = ms[0]["source_kc"] if ms else None

# Pass 1: collect per-KC global mean prediction (prior) from holdout preds
rows_full = []  # (student_rows with pred)
per_kc_pred = collections.defaultdict(list)
per_student = []
for ws in samples:
    keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping[int(kc)] is not None]
    if len(keep) < 6:
        continue
    seq_m = [mapping[int(ws.sequence[i])] for i in keep]
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
    rows2 = [{"y": x["y"], "pred": prior.get(x["kc"], 0.5)} for x in rows if x["kc"] in prior]
    if rows2 and 0 < sum(x["y"] for x in rows2) < len(rows2):
        a = stratified_auc_rows(rows2, "pred")
        if a is not None:
            auc_prior.append(a)

print(f"full transplanted model : within-AUC = {np.mean(aucs_full):.4f} (n={len(aucs_full)})")
print(f"priors-only (no hidden) : within-AUC = {np.mean(auc_prior):.4f} (n={len(auc_prior)})")
print(f"delta (full - prior)    : {np.mean(aucs_full) - np.mean(auc_prior):+.4f}")
