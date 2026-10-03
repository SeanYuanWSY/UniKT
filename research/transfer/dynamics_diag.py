"""Platform-adaptivity hypothesis test for the anti-transfer mechanism.

H-A (adaptivity): within a student, later correctness correlates NEGATIVELY
with recent correctness on A17 (adaptive difficulty escalation) but NOT on A09.
H-B (momentum): the transplanted A09 model's prediction deviations from its KC
prior correlate POSITIVELY with recent correctness (learned momentum).
H-A & H-B together => transplanted dynamics anti-correlate with outcomes.

Runs on both: A09-native identity (DKT-A09 on A09) and A09->A17 transplant.
"""
import collections
import sys

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment
from kc_tables import _make_rc
from signals import _forward_probs
from llm_explain_restore_shim import restore_any
from restore import load_user_samples
from utils.data_process import get_data_source


def collect(rm, samples, mapping=None, k_recent=5):
    """Per holdout step: y, pred, kc, recent-k mean of prior responses."""
    per_kc_pred = collections.defaultdict(list)
    steps = []
    for ws in samples:
        if mapping is None:
            keep = list(range(len(ws.sequence)))
        else:
            keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping.get(int(kc))]
        if len(keep) < 6:
            continue
        seq_m = [mapping[int(ws.sequence[i])] if mapping else int(ws.sequence[i]) for i in keep]
        resp_m = [int(ws.response[i]) for i in keep]
        s = torch.tensor([seq_m], dtype=torch.long, device=rm.device)
        r_ = torch.tensor([resp_m], dtype=torch.long, device=rm.device)
        probs = _forward_probs(rm, s, r_, None)[0]
        holdout_pos = set(ws.holdout_idx.tolist())
        history = []
        for j, i in enumerate(keep):
            is_hold = i in holdout_pos
            if is_hold:
                rec = float(np.mean(history[-k_recent:])) if len(history) >= k_recent else None
                steps.append(
                    {
                        "y": int(ws.response[i]),
                        "pred": float(probs[j]),
                        "kc": int(ws.sequence[i]),
                        "recent": rec,
                    }
                )
                per_kc_pred[int(ws.sequence[i])].append(float(probs[j]))
            history.append(int(ws.response[i]))
    prior = {k: float(np.mean(v)) for k, v in per_kc_pred.items() if len(v) >= 5}
    return steps, prior


def pooled_corrs(steps, prior, label):
    rows = [s for s in steps if s["recent"] is not None and s["kc"] in prior]
    y = np.array([s["y"] for s in rows], dtype=float)
    rec = np.array([s["recent"] for s in rows])
    pred = np.array([s["pred"] for s in rows])
    pr = np.array([prior[s["kc"]] for s in rows])
    dyn = pred - pr  # dynamics component (deviation from KC prior)
    c1 = pearsonr(rec, y).statistic
    c2 = pearsonr(rec, dyn).statistic
    c3 = pearsonr(dyn, y).statistic
    print(f"[{label}] n={len(rows)}")
    print(f"  corr(recent_correctness, actual_y)        = {c1:+.3f}   (H-A adaptivity)")
    print(f"  corr(recent_correctness, pred_deviation)  = {c2:+.3f}   (H-B momentum)")
    print(f"  corr(pred_deviation, actual_y)            = {c3:+.3f}   (net dynamics signal)")
    return c1, c2, c3


# --- A09 native ---
rm9 = restore_any("runs/normal/DKT_assistments09_20261003-024700_fold0_bs128")
src9 = get_data_source(_make_rc("assistments09"))
s9 = load_user_samples(src9, fold=0)[:150]
steps9, prior9 = collect(rm9, s9, mapping=None)
pooled_corrs(steps9, prior9, "A09-native (DKT-A09 on A09)")

# --- A09 -> A17 transplant ---
alignment = {int(k): v for k, v in load_alignment("assistments09__to__assistments17__llm__glm__rep0").items()}
rm7 = restore_any("runs/normal/DKT_assistments09_20261003-024700_fold0_bs128")
tgt = get_data_source(_make_rc("assistments17"))
s17 = load_user_samples(tgt, fold=0)[:150]
mapping = {}
for ws in s17:
    for kc in set(int(c) for c in ws.sequence.tolist()):
        ms = alignment.get(kc, [])
        mapping[kc] = ms[0]["source_kc"] if ms else None
steps7, prior7 = collect(rm7, s17, mapping=mapping)
pooled_corrs(steps7, prior7, "A09->A17 transplant")

# --- A17 native (for the platform's own learned pattern) ---
rm17 = restore_any("runs/normal/DKT_assistments17_20261003-144042_fold0_bs128")
steps17n, prior17n = collect(rm17, s17, mapping=None)
pooled_corrs(steps17n, prior17n, "A17-native (DKT-A17 on A17)")
