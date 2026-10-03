"""Block-structure mismatch diagnostic.

Q1: How KC-blocked are the two datasets? (mean/max run length of consecutive
same-KC steps in evidence segments)
Q2: Does the transplant AUC depend on block exposure? Split students by the
mean same-KC run length of their kept sequence: long-block vs short-block ->
compare within-student AUC of the transplanted model.
"""
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


def run_stats(seq):
    runs, cur = [], 1
    for a, b in zip(seq[:-1], seq[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return float(np.mean(runs)), float(np.max(runs))


# Q1: block statistics per dataset
for name in ("assistments09", "assistments17"):
    src = get_data_source(_make_rc(name))
    samples = load_user_samples(src, fold=0)[:300]
    means, maxs = [], []
    for ws in samples:
        m, mx = run_stats(ws.sequence[ws.evidence_idx].tolist())
        means.append(m)
        maxs.append(mx)
    print(f"Q1 {name}: mean run len = {np.mean(means):.2f}, median max-run = {np.median(maxs):.0f}")

# Q2: AUC by block exposure (transplant)
alignment = {int(k): v for k, v in load_alignment("assistments09__to__assistments17__llm__glm__rep0").items()}
rm = restore_any("runs/normal/DKT_assistments09_20261003-024700_fold0_bs128")
tgt = get_data_source(_make_rc("assistments17"))
samples = load_user_samples(tgt, fold=0)[:150]
mapping = {}
for ws in samples:
    for kc in set(int(c) for c in ws.sequence.tolist()):
        ms = alignment.get(kc, [])
        mapping[kc] = ms[0]["source_kc"] if ms else None

short_auc, long_auc = [], []
runlens = []
for ws in samples:
    keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping.get(int(kc))]
    if len(keep) < 6:
        continue
    seq_m = [mapping[int(ws.sequence[i])] for i in keep]
    resp_m = [int(ws.response[i]) for i in keep]
    m_run, _ = run_stats(seq_m)
    runlens.append(m_run)
    s = torch.tensor([seq_m], dtype=torch.long, device=rm.device)
    r_ = torch.tensor([resp_m], dtype=torch.long, device=rm.device)
    probs = _forward_probs(rm, s, r_, None)[0]
    holdout_pos = set(ws.holdout_idx.tolist())
    rows = [{"y": int(ws.response[i]), "pred": float(probs[j])} for j, i in enumerate(keep) if i in holdout_pos]
    a = stratified_auc_rows(rows, "pred")
    if a is not None:
        (long_auc if m_run >= np.median(runlens) else short_auc).append(a)

print(f"Q2 transplant AUC: short-block students = {np.mean(short_auc):.4f} (n={len(short_auc)}), "
      f"long-block students = {np.mean(long_auc):.4f} (n={len(long_auc)})")
print(f"   median mapped run length = {np.median(runlens):.2f}")
