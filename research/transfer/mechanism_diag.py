"""Mechanism diagnostics for the anti-transfer phenomenon (A09-DKT -> A17).

Questions:
D1. Per-student AUC distribution: concentrated below 0.5 or bimodal?
D2. Prediction dispersion per student: are mapped preds nearly constant
    (ties -> 0.5) with a varying minority driving the signal?
D3. Cross-dataset KC prior correlation: A09 model's per-KC mean prediction
    (across students, under the mapping) vs A17's per-KC actual correctness.
    Negative correlation => the source model's difficulty priors are
    anti-aligned with the target's actual difficulty ordering.
D4. Evidence length / drop rate vs per-student AUC.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment
from kc_tables import _make_rc
from transfer_eval import evaluate_alignment, stratified_auc_rows
from llm_explain_restore_shim import restore_any
from restore import load_user_samples
from utils.data_process import get_data_source

SRC_RUN = "runs/normal/DKT_assistments09_20261003-024700_fold0_bs128"
ALIGN = load_alignment("assistments09__to__assistments17__llm__glm__rep0")
alignment = {int(k): v for k, v in ALIGN.items()}

rm = restore_any(SRC_RUN)
tgt = get_data_source(_make_rc("assistments17"))
samples = load_user_samples(tgt, fold=0)[:150]

res = evaluate_alignment(rm, samples, alignment)
rows_by_student = res["rows_by_student"]

# D1: per-student AUC distribution
aucs = np.array([stratified_auc_rows(r, "pred") for r in rows_by_student], dtype=object)
aucs = np.array([a for a in aucs if a is not None])
print(f"D1 per-student AUC: mean={aucs.mean():.3f} median={np.median(aucs):.3f} "
      f"<0.4: {(aucs<0.4).mean():.2%} 0.4-0.6: {((aucs>=0.4)&(aucs<0.6)).mean():.2%} >0.6: {(aucs>=0.6).mean():.2%}")

# D2: prediction dispersion per student (IQR of preds / fraction of tied pairs)
disp, tie_frac = [], []
for r in rows_by_student:
    p = np.array([x["pred"] for x in r])
    disp.append(np.percentile(p, 75) - np.percentile(p, 25))
    n = len(p)
    ties = sum(1 for i in range(n) for j in range(i + 1, n) if p[i] == p[j])
    tie_frac.append(ties / (n * (n - 1) / 2) if n > 1 else 1.0)
print(f"D2 pred IQR median={np.median(disp):.4f}; tied-pair fraction median={np.median(tie_frac):.2%}")

# D3: per-KC prior correlation (mapped source KC -> mean pred vs actual acc)
import collections

kc_pred = collections.defaultdict(list)
kc_actual = collections.defaultdict(list)
mapping = {}
for ws in samples:
    for kc in set(ws.sequence.tolist()):
        ms = alignment.get(int(kc), [])
        mapping[int(kc)] = ms[0]["source_kc"] if ms else None
# redo forward per student to attribute preds to target KCs (reuse evaluate but simpler: rerun)
import torch
from signals import _forward_probs

per_kc_pred, per_kc_act = collections.defaultdict(list), collections.defaultdict(list)
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
    for j, i in enumerate(keep):
        if i in holdout_pos:
            per_kc_pred[int(ws.sequence[i])].append(float(probs[j]))
            per_kc_act[int(ws.sequence[i])].append(int(ws.response[i]))

kcs = [k for k in per_kc_pred if len(per_kc_pred[k]) >= 20 and 0 < np.mean(per_kc_act[k]) < 1]
from scipy.stats import spearmanr

rho = spearmanr([np.mean(per_kc_pred[k]) for k in kcs], [np.mean(per_kc_act[k]) for k in kcs]).statistic
print(f"D3 KC-prior vs actual acc across {len(kcs)} KCs (n>=20): spearman={rho:.3f}")

# D4: evidence length vs AUC
print(f"D4 n_students={res['n_students']} drop_rate={res['drop_rate']}")
print(json.dumps({k: v for k, v in res.items() if k != "rows_by_student"}))
