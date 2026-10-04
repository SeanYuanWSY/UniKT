"""Junyi within-student anomaly diagnostics (K3 hypotheses H1/H2).

H1 (recency inversion via proficiency remediation):
  corr(y, recent5_sameKC) <=~0  while  corr(pred, recent5_sameKC) > 0.
H2 (repeat-question selection bias):
  AUC(first-seen holdout rows) >> AUC(repeated holdout rows); among repeats,
  evidence outcome of the same question skews wrong while holdout skews right.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "llm_explain"))
sys.path.insert(0, str(_HERE.parents[1]))

from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

RUN_DKT = "runs/normal/DKT_junyi2015_20261004-195512_fold0_bs128"
RUN_AKT = "runs/normal/AKT_junyi2015_20261004-195553_fold0_bs64"


def collect(rm, samples):
    """Per holdout row: y, pred, kc, repeat flag, evidence outcome of same q,
    recent5 correctness on the same KC (and overall) before the step."""
    rows = []
    per_kc_pred = defaultdict(list)
    students = []
    for ws in samples:
        if len(ws.sequence) < 6:
            continue
        seq = ws.sequence.tolist()
        resp = [int(x) for x in ws.response.tolist()]
        q = ws.question.tolist()
        s = torch.tensor([seq], dtype=torch.long, device=rm.device)
        r = torch.tensor([resp], dtype=torch.long, device=rm.device)
        probs = _forward_probs(rm, s, r, None)[0]
        holdout_pos = set(ws.holdout_idx.tolist())
        ev_q_out = {}
        hist_kc = defaultdict(list)
        hist_any = []
        st_rows = []
        for j in range(len(seq)):
            kc = int(seq[j])
            if j in holdout_pos:
                rec_kc = float(np.mean(hist_kc[kc][-5:])) if len(hist_kc[kc]) >= 3 else None
                row = {
                    "y": resp[j],
                    "pred": float(probs[j]),
                    "kc": kc,
                    "q": int(q[j]),
                    "repeat": int(q[j]) in ev_q_out,
                    "ev_q_out": ev_q_out.get(int(q[j])),
                    "recent_kc": rec_kc,
                }
                rows.append(row)
                st_rows.append(row)
                per_kc_pred[kc].append(float(probs[j]))
            else:
                ev_q_out[int(q[j])] = resp[j]
            hist_kc[kc].append(resp[j])
            hist_any.append(resp[j])
        if st_rows and 0 < sum(x["y"] for x in st_rows) < len(st_rows):
            students.append(st_rows)
    prior = {k: float(np.mean(v)) for k, v in per_kc_pred.items() if len(v) >= 5}
    return rows, students, prior


def analyse(rows, students, prior, label):
    print(f"\n===== {label} =====")
    y = np.array([r["y"] for r in rows], dtype=float)

    # H1: recency correlations (rows with recent_kc available)
    sub = [r for r in rows if r["recent_kc"] is not None]
    ys = np.array([r["y"] for r in sub], dtype=float)
    rec = np.array([r["recent_kc"] for r in sub])
    pred = np.array([r["pred"] for r in sub])
    print(f"H1 n={len(sub)}  corr(y, recent5_sameKC)={pearsonr(rec, ys).statistic:+.3f}"
          f"  corr(pred, recent5_sameKC)={pearsonr(rec, pred).statistic:+.3f}")

    # H2: repeat vs first AUC (within-student, subset rows)
    for flag, name in [(True, "repeat"), (False, "first-seen")]:
        aucs = []
        for st in students:
            sub_rows = [{"y": r["y"], "pred": r["pred"]} for r in st if r["repeat"] == flag]
            a = stratified_auc_rows(sub_rows, "pred")
            if a is not None:
                aucs.append(a)
        print(f"H2 {name:10s}: within-AUC = {np.mean(aucs):.4f} (n_students={len(aucs)})"
              if aucs else f"H2 {name:10s}: no valid students")

    # H2b: among repeats, evidence outcome vs holdout outcome / pred
    rep = [r for r in rows if r["repeat"] and r["ev_q_out"] is not None]
    if rep:
        ev = np.array([r["ev_q_out"] for r in rep], dtype=float)
        hy = np.array([r["y"] for r in rep], dtype=float)
        hp = np.array([r["pred"] for r in rep])
        print(f"H2b repeats n={len(rep)}: P(evidence correct)={ev.mean():.3f}  "
              f"P(holdout correct)={hy.mean():.3f}  corr(ev_out, y)={pearsonr(ev, hy).statistic:+.3f}  "
              f"corr(ev_out, pred)={pearsonr(ev, hp).statistic:+.3f}")

    # dynamics deviation vs recent (is the model's deviation driven by recency?)
    sub2 = [r for r in sub if r["kc"] in prior]
    if sub2:
        dyn = np.array([r["pred"] - prior[r["kc"]] for r in sub2])
        rec2 = np.array([r["recent_kc"] for r in sub2])
        y2 = np.array([r["y"] for r in sub2], dtype=float)
        print(f"dyn corr(rec, pred-prior)={pearsonr(rec2, dyn).statistic:+.3f}  "
              f"corr(pred-prior, y)={pearsonr(dyn, y2).statistic:+.3f}")


tgt = get_data_source(_make_rc("junyi2015"))
samples = load_user_samples(tgt, fold=0)[:300]
for run, name in [(RUN_DKT, "DKT-junyi native"), (RUN_AKT, "AKT-junyi native")]:
    rm = restore_any(run)
    rows, students, prior = collect(rm, samples)
    analyse(rows, students, prior, name)
print("\nJUNYI-DIAG-DONE")
