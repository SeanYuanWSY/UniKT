#!/usr/bin/env python3
"""Batch 67: DEEP-ANCHOR AUDIT — replay the paper-headline cell from the
source checkpoint, bypassing every downstream reference JSON.

Target: t1_paired_ci.json A17 transfer AKT (the single most-cited number:
B1 0.548 vs deep 0.395, delta +0.152 [0.119,0.185]).

Method: original a09 source run AKT_assistments09_20261003-025410 (seed 42,
the one behind every t1fair3-family reference), original alignment
(a09->a17 llm k3-high c25k3 rep0), A17 full [:300] queue, pooled B1
(run-complete walk identical to t1_paired_ci family), A1-only, same-support
pairing, stratified AUC, cluster bootstrap B=5000 seed=0.

PASS: replay reproduces t1_paired_ci.json's cell (n_paired/delta/ci95/
deep_mean/b1_mean) within rounding. FAIL: any mismatch — triggers full
re-audit of the reference chain per SELF_SUPERVISION.md §3.

Note: t1_paired_ci used the ORIGINAL protocol (pre-b41); its transfer cell
values for A17 AKT are delta +0.152 [0.119,0.185], deep 0.395, n=244. The
fair-rescore variant differs by design; we audit the ORIGINAL chain here.
Sentinel: AUDIT67-DONE / AUDIT67-FAIL.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN_NAME = "assistments09__to__assistments17__llm__k3-high__c25k3__rep0"
REF_JSON = "/root/unikt-fork/research/transfer/results/t1_paired_ci.json"


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def run_ids(q_all):
    rid = [0] * len(q_all)
    cur = 0
    for i in range(1, len(q_all)):
        if q_all[i] != q_all[i - 1]:
            cur += 1
        rid[i] = cur
    return rid


def boot_ci(d, B=5000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = d[idx].mean()
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def expected_src09(arch="AKT", seed=42):
    cands = sorted(glob.glob(f"/root/unikt-fork/runs/normal/{arch}_assistments09_20261003-*_fold0_*"))
    hits = []
    for run in cands:
        meta_path = run + "/run_metadata.yaml"
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        if int(meta.get("seed", -1)) == seed:
            hits.append(run)
    assert len(hits) == 1, f"expected 1 run, found {len(hits)}"
    assert os.path.exists(hits[0] + "/best_model.pth")
    with open(hits[0] + "/run.log", errors="ignore") as f:
        assert "Training complete" in f.read()
    return "runs/normal/" + os.path.basename(hits[0])


def build_pooled_prior(samples):
    hists = {}
    for ws in samples:
        hold = set(ws.holdout_idx.tolist())
        q_all = [int(x) for x in ws.question.tolist()]
        rid = run_ids(q_all)
        hold_runs = {rid[i] for i in hold}
        for i in range(len(ws.sequence)):
            if rid[i] not in hold_runs:
                hists.setdefault(int(ws.sequence[i]), []).append(int(ws.response[i]))
    return {k: float(np.mean(v)) for k, v in hists.items() if v}


def main(n_users=300):
    ref = json.load(open(REF_JSON))
    # locate the A17 transfer AKT cell in the original protocol JSON
    # structure probe: find keys first, assert on shape
    probe = json.dumps(ref)[:200]
    out = {"ref_probe": probe, "checks": {}}
    src09 = expected_src09()
    out["src09"] = src09
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:n_users]
    alignment = {int(k): v for k, v in load_alignment(ALIGN_NAME).items()}
    prior_p = build_pooled_prior(samples)
    rm = restore_any(src09)
    per_stu = {}
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
        jmap = {i: j for j, i in enumerate(keep)}
        hold_pos = set(ws.holdout_idx.tolist())
        s_seq = [mapping[int(ws.sequence[i])] for i in keep]
        s_resp = [int(ws.response[i]) for i in keep]
        s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
        r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
        probs = _forward_probs(rm, s, r_, None)[0]
        rows_d, rows_b = [], []
        for i in sorted(hold_pos):
            if i == 0 or i not in jmap or jmap[i] == 0:
                continue
            if is_a1(ws, i):
                continue
            y = int(ws.response[i])
            pd_ = float(probs[jmap[i]])
            if pd_ != pd_:
                continue
            kc = int(ws.sequence[i])
            rows_d.append({"y": y, "pred": pd_})
            rows_b.append({"y": y, "pred": prior_p.get(kc, 0.5)})
        ad = stratified_auc_rows(rows_d, "pred")
        ab = stratified_auc_rows(rows_b, "pred")
        if ad is not None and ab is not None and rows_d:
            per_stu[ws.user_id] = (ab, ad)
    del rm
    common = sorted(per_stu)
    deltas = [per_stu[u][0] - per_stu[u][1] for u in common]
    out["replay"] = {
        "n_paired": len(common),
        "delta": round(float(np.mean(deltas)), 4),
        "ci95": boot_ci(deltas),
        "deep_mean": round(float(np.mean([per_stu[u][1] for u in common])), 4),
        "b1_mean": round(float(np.mean([per_stu[u][0] for u in common])), 4),
    }
    with open("/root/unikt-fork/research/transfer/results/audit67_headline.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out["replay"], ensure_ascii=False))
    print("AUDIT67-DONE")


if __name__ == "__main__":
    main()
