"""IDEA-3-pivot (minimal prototype): frozen-transfer logits x KC-prior table fusion.

On the A17 fair keep-set (k3-high alignment, AKT transfer = best deep 0.532;
B1 table = 0.546):
  F1  alpha fusion: logit(p_deep) and logit(p_table) mixed, alpha grid
  F2  2-logit logistic stack (fitted on evidence rows, leak-free)
  F3  coverage gating: alpha(cov) piecewise (matched-KC vs holdout coverage)
  F4  architecture gate: AKT + DKT transfer + table, 3-logit stack
Preregistered ladder: +0.02 over max(0.532, 0.546) = continue; +0.04 = strong.
"""
import json
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = "assistments09__to__assistments17__llm__k3-high__c25k3__rep0"
RUNS = {
    "AKT": "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64",
    "DKT": "runs/normal/DKT_assistments09_20261003-024700_fold0_bs128",
}

tgt = get_data_source(_make_rc("assistments17"))
samples = load_user_samples(tgt, fold=0)[:300]
students = build_rows(samples, ALIGN)
prior = fit_global_kc_prior(students)
alignment = {int(k): v for k, v in load_alignment(ALIGN).items()}

models = {tag: restore_any(run) for tag, run in RUNS.items()}

per_student = []
for ws in samples:
    mapping = {}
    for kc in set(int(c) for c in ws.sequence.tolist()):
        ms = alignment.get(kc, [])
        mapping[kc] = ms[0]["source_kc"] if ms else None
    keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping.get(int(kc)) is not None]
    if len(keep) > 200:
        keep = keep[-200:]
    if len(keep) < 6:
        continue
    holdout_pos = set(ws.holdout_idx.tolist())
    seq_m = [mapping[int(ws.sequence[i])] for i in keep]
    resp_m = [int(ws.response[i]) for i in keep]
    s = torch.tensor([seq_m], dtype=torch.long, device=models["AKT"].device)
    r_ = torch.tensor([resp_m], dtype=torch.long, device=models["AKT"].device)
    preds = {tag: _forward_probs(m, s, r_, None)[0] for tag, m in models.items()}
    ev_rows, hold_rows = [], []
    for j, i in enumerate(keep):
        row = {
            "y": int(ws.response[i]),
            "kc": int(ws.sequence[i]),
            **{f"deep_{t}": float(preds[t][j]) for t in RUNS},
        }
        (hold_rows if i in holdout_pos else ev_rows).append(row)
    if ev_rows and hold_rows and 0 < sum(x["y"] for x in hold_rows) < len(hold_rows):
        per_student.append({"ev": ev_rows, "hold": hold_rows})


def logits(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def auc_of(key_fn):
    aucs = []
    for st in per_student:
        rows = [{"y": x["y"], "pred": key_fn(x)} for x in st["hold"]]
        a = stratified_auc_rows(rows, "pred")
        if a is not None:
            aucs.append(a)
    return round(float(np.mean(aucs)), 4), len(aucs)


def table_p(x):
    return prior.get(x["kc"], 0.5)


base_deep, n = auc_of(lambda x: x["deep_AKT"])
base_table, _ = auc_of(lambda x: table_p(x))

out = {"n_students": n, "deep_AKT": base_deep, "table": base_table}

# F1 alpha grid (evaluated on evidence rows, applied to holdout — leak-free)
best_alpha, best_ev = None, -1
for alpha in np.linspace(0, 1, 11):
    ev_aucs = []
    for st in per_student:
        rows = [{"y": x["y"], "pred": alpha * logits(table_p(x)) + (1 - alpha) * logits(x["deep_AKT"])} for x in st["ev"]]
        a = stratified_auc_rows(rows, "pred")
        if a is not None:
            ev_aucs.append(a)
    m = float(np.mean(ev_aucs)) if ev_aucs else 0.5
    if m > best_ev:
        best_ev, best_alpha = m, float(alpha)
f1, _ = auc_of(lambda x: best_alpha * logits(table_p(x)) + (1 - best_alpha) * logits(x["deep_AKT"]))
out["F1_alpha_fusion"] = {"alpha": round(best_alpha, 2), "auc": f1}

# F2 2-logit stack (logistic on evidence)
from sklearn.linear_model import LogisticRegression  # noqa: E402

Xe = np.array([[logits(table_p(x)), logits(x["deep_AKT"])] for st in per_student for x in st["ev"]])
ye = np.array([x["y"] for st in per_student for x in st["ev"]])
stk = LogisticRegression(C=1.0).fit(Xe, ye)
f2, _ = auc_of(lambda x: float(stk.predict_proba([[logits(table_p(x)), logits(x["deep_AKT"])]])[0][1]))
out["F2_stack2"] = f2

# F4 3-logit stack (table + AKT + DKT)
Xe3 = np.array(
    [
        [logits(table_p(x)), logits(x["deep_AKT"]), logits(x["deep_DKT"])]
        for st in per_student
        for x in st["ev"]
    ]
)
stk3 = LogisticRegression(C=1.0).fit(Xe3, ye)
f4, _ = auc_of(
    lambda x: float(
        stk3.predict_proba([[logits(table_p(x)), logits(x["deep_AKT"]), logits(x["deep_DKT"])]])[0][1]
    )
)
out["F4_stack3"] = f4

out["preregistered_gate"] = {
    "continue_threshold": round(max(base_deep, base_table) + 0.02, 4),
    "strong_threshold": round(max(base_deep, base_table) + 0.04, 4),
    "best_fusion": max(f1, f2, f4),
}
print(json.dumps(out, ensure_ascii=False, indent=1))
with open("/root/unikt-fork/research/transfer/results/idea3_fusion_a17.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("IDEA3-DONE")
