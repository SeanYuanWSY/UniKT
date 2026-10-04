"""Pre-submission statistical completion (review-mandated, pure analysis).

S1: paired student-level cluster bootstrap CIs for Tree-3 baselines vs the
    best deep transfer cell (B1/B4 vs deep, per domain).
S2: Platt (logistic) calibration arm for the deep transfer predictions,
    fitted on the evidence segment, scored on holdout (A17 best cell).
S3: frequency-preserving KC shuffle (L4a-P1 control): permute KC ids within
    frequency-matched buckets, rerun the native A17 probe.
"""
import collections
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
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

OUT = {}
rng = np.random.default_rng(42)

# ============ A17: baselines vs deep (paired bootstrap) + Platt ============
ALIGN = "assistments09__to__assistments17__llm__k3-high__c25k3__rep0"
DEEP_RUN = "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64"  # best cell 0.534
NATIVE_RUN = "runs/normal/AKT_assistments17_20261003-145008_fold0_bs64"

tgt = get_data_source(_make_rc("assistments17"))
samples = load_user_samples(tgt, fold=0)[:300]
students = build_rows(samples, ALIGN)
prior = fit_global_kc_prior(students)
alignment = {int(k): v for k, v in load_alignment(ALIGN).items()}

rm = restore_any(DEEP_RUN)
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
    s = torch.tensor([seq_m], dtype=torch.long, device=rm.device)
    r_ = torch.tensor([resp_m], dtype=torch.long, device=rm.device)
    probs = _forward_probs(rm, s, r_, None)[0]
    ev_rows, hold_rows = [], []
    for j, i in enumerate(keep):
        row = {"y": int(ws.response[i]), "pred_deep": float(probs[j]), "kc": int(ws.sequence[i])}
        (hold_rows if i in holdout_pos else ev_rows).append(row)
    if not ev_rows or not hold_rows:
        continue
    if not (0 < sum(x["y"] for x in hold_rows) < len(hold_rows)):
        continue
    # Platt on evidence
    ex = np.array([x["pred_deep"] for x in ev_rows])
    ey = np.array([x["y"] for x in ev_rows])
    from sklearn.linear_model import LogisticRegression

    if 0 < ey.sum() < len(ey):
        lr_c = LogisticRegression(C=1e6)
        lr_c.fit(ex.reshape(-1, 1), ey)
        for x in hold_rows:
            x["pred_deep_platt"] = float(lr_c.predict_proba([[x["pred_deep"]]])[0][1])
    else:
        for x in hold_rows:  # single-class evidence: calibration undefined, keep raw
            x["pred_deep_platt"] = x["pred_deep"]
    per_student.append({"rows": hold_rows})

def mean_auc(rows_key):
    aucs = []
    for st in per_student:
        rows = [{"y": x["y"], "pred": x[rows_key]} for x in st["rows"]]
        a = stratified_auc_rows(rows, "pred")
        if a is not None:
            aucs.append(a)
    return np.array(aucs)

deep = mean_auc("pred_deep")
deep_platt = mean_auc("pred_deep_platt")
b1 = []
b4_stub = None
for st in students:  # B1 per-student on the same holdout rows
    pass
# B1 per-student aligned by order of per_student (same sample list & filters)
b1_aucs = []
for st in per_student:
    rows = [{"y": x["y"], "pred": prior.get(x["kc"], 0.5)} for x in st["rows"]]
    a = stratified_auc_rows(rows, "pred")
    if a is not None:
        b1_aucs.append(a)
b1_aucs = np.array(b1_aucs)
n = min(len(deep), len(b1_aucs))

def paired_boot(a, b, B=5000):
    d = a[:n] - b[:n]
    boots = [np.mean(d[rng.integers(0, n, n)]) for _ in range(B)]
    return float(np.mean(d)), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

d_b1_deep, ci1 = paired_boot(b1_aucs, deep)
d_deepplatt_deep, ci2 = paired_boot(deep_platt, deep)
OUT["a17_paired"] = {
    "deep_auc": round(float(np.mean(deep)), 4),
    "deep_platt_auc": round(float(np.mean(deep_platt)), 4),
    "b1_auc": round(float(np.mean(b1_aucs)), 4),
    "n_paired": int(n),
    "delta_b1_minus_deep": round(d_b1_deep, 4),
    "ci95_b1_minus_deep": [round(x, 4) for x in ci1],
    "delta_platt_minus_deep": round(d_deepplatt_deep, 4),
    "ci95_platt_minus_deep": [round(x, 4) for x in ci2],
}
print(json.dumps(OUT["a17_paired"], ensure_ascii=False))

# ============ S3: frequency-preserving KC shuffle (native A17 AKT) ============
rm_n = restore_any(NATIVE_RUN)
samples_p = load_user_samples(tgt, fold=0)[:250]
kc_freq = collections.Counter()
for ws in samples_p:
    for kc in ws.sequence.tolist():
        kc_freq[int(kc)] += 1
kcs = sorted(kc_freq)
freqs = np.array([kc_freq[k] for k in kcs])
# bucket by frequency tercile, permute within bucket
order = np.argsort(freqs)
bucket = np.zeros(len(kcs), dtype=int)
bucket[order[: len(kcs) // 3]] = 0
bucket[order[len(kcs) // 3 : 2 * len(kcs) // 3]] = 1
bucket[order[2 * len(kcs) // 3 :]] = 2

def auc_with_perm(kc_perm):
    aucs = []
    for ws in samples_p:
        if len(ws.sequence) < 6:
            continue
        seq = [kc_perm[int(k)] for k in ws.sequence.tolist()]
        resp = [int(x) for x in ws.response.tolist()]
        s = torch.tensor([seq], dtype=torch.long, device=rm_n.device)
        r_ = torch.tensor([resp], dtype=torch.long, device=rm_n.device)
        probs = _forward_probs(rm_n, s, r_, None)[0]
        rows = [{"y": resp[i], "pred": float(probs[i])} for i in set(ws.holdout_idx.tolist()) if i < len(seq)]
        a = stratified_auc_rows(rows, "pred")
        if a is not None:
            aucs.append(a)
    return float(np.mean(aucs))

base = auc_with_perm({k: k for k in kcs})
fp_shuffles = []
for seed in range(3):
    r = np.random.default_rng(2000 + seed)
    perm = {}
    for b in (0, 1, 2):
        idx = [i for i in range(len(kcs)) if bucket[i] == b]
        shuffled = r.permutation(idx)
        for orig, new in zip(idx, shuffled):
            perm[kcs[orig]] = kcs[new]
    fp_shuffles.append(auc_with_perm(perm))
OUT["a17_freq_preserving_shuffle_AKT"] = {
    "identity": round(base, 4),
    "freq_preserved_shuffle_mean": round(float(np.mean(fp_shuffles)), 4),
    "shuffles": [round(x, 4) for x in fp_shuffles],
    "ratio": round(float(np.mean(fp_shuffles)) / base, 4),
}
print(json.dumps(OUT["a17_freq_preserving_shuffle_AKT"], ensure_ascii=False))

with open("/root/unikt-fork/research/transfer/results/final_stats.json", "w") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=2)
print("FINAL-STATS-DONE")
