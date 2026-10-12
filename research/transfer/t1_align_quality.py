"""Batch 63: EdNet transfer cells, ALIGNMENT-quality axis, 6 alignments x 4 archs x seed 42.

Every EdNet transfer number so far rests on ONE alignment configuration
(llm k3-high c25k3 rep0; n_paired=56 of ~23% coverage on the [:300] queue).
The cache holds five more a09->ednet alignments: two k3-high reruns (rep1/
rep2), a bge_m3 embedding baseline, a GLM edit-distance variant, and a
glm-5.3 c25k3 variant. Mapped-KC counts differ sharply (rep0 27 / rep1 19 /
rep2 43 / embed 26 / edit 72 / glm53 19 of 189 target KCs) and pairwise
agreement with rep0 on jointly-mapped targets is 11/17-11/19 for the LLM
variants and 13/20 for embed — the alignment is far from stable.

This batch re-scores the 4 EdNet transfer cells (seed 42 ONLY — the
headline-configuration seed) under each of the 6 alignments, everything
else frozen: same [:300] queue, same pooled B1 table, same models, same
protocol. Question: does "B1 beats frozen transfer" survive alignment
choice? Zero training; evaluation only.

Anchors per cell (3 points, 72 total): the rep0 arm MUST reproduce the b59
reference cell exactly (same script path, same alignment name -> n_paired
== 56, delta within 0.0001, deep_mean within 0.0001) — this doubles as the
frozen-pipeline self-check. Other arms have no reference (their n_paired
changes by design); for them only structural sanity is asserted inline
(n_nonfinite == 0 reported, counted in G1 by JSON fields).

Sentinel: T1ALIGN-DONE / T1ALIGN-DONE-ANCHOR-FAIL (dual-state).
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

ALIGN_NAMES = {
    "k3_rep0": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
    "k3_rep1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep1",
    "k3_rep2": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep2",
    "embed": "assistments09__to__ednet_kt1__embed__bge_m3__rep0",
    "edit": "assistments09__to__ednet_kt1__edit__glm__c25k3__rep0",
    "glm53": "assistments09__to__ednet_kt1__llm__glm-5.3__c25k3__rep0",
}
KC_N = 175  # frozen [:300] queue pooled B1 table (b59 construction)

with open("/root/unikt-fork/research/transfer/results/t1_seed_transfer_fair.json") as f:
    _b59 = json.load(f)
REF = {a: _b59["cells"][f"ednet_kt1|{a}|s1"]["old_run_seed42"] for a in ("DKT", "AKT", "SAKT", "DKVMN")}
# b59 old_run_seed42 = the seed-42 cell references (delta, deep_mean)
REF_N = {a: _b59["cells"][f"ednet_kt1|{a}|s1"]["n_paired"] for a in ("DKT", "AKT", "SAKT", "DKVMN")}


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


def mean_or_none(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def expected_src09(arch, seed=42):
    prefix = "20261003"
    cands = sorted(glob.glob(f"/root/unikt-fork/runs/normal/{arch}_assistments09_{prefix}-*_fold0_*"))
    assert cands, f"no {arch} a09 run dated {prefix}"
    hits = []
    for run in cands:
        meta_path = run + "/run_metadata.yaml"
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        if int(meta.get("seed", -1)) == seed:
            hits.append(run)
    assert len(hits) == 1, f"expected exactly 1 {arch} a09 run seed={seed} dated {prefix}, found {len(hits)}"
    assert os.path.exists(hits[0] + "/best_model.pth"), f"incomplete run: {hits[0]}"
    with open(hits[0] + "/run.log", errors="ignore") as f:
        assert "Training complete" in f.read(), f"run lacks completion marker: {hits[0]}"
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


def transfer_cell(samples, alignment, src09, prior_p):
    rm = restore_any(src09)
    per_stu = {}
    n_fb = 0
    n_nonfinite = 0
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
            if i == 0 or i not in jmap:
                continue
            if jmap[i] == 0:
                continue
            if is_a1(ws, i):
                continue
            y = int(ws.response[i])
            pd_ = float(probs[jmap[i]])
            if pd_ != pd_:
                n_nonfinite += 1
                continue
            kc = int(ws.sequence[i])
            rows_d.append({"y": y, "pred": pd_})
            if kc not in prior_p:
                n_fb += 1
            rows_b.append({"y": y, "pred": prior_p.get(kc, 0.5)})
        ad = stratified_auc_rows(rows_d, "pred")
        ab = stratified_auc_rows(rows_b, "pred")
        if ad is not None and ab is not None and rows_d:
            per_stu[ws.user_id] = (ab, ad)
    del rm
    return per_stu, n_fb, n_nonfinite


def main(n_users=300):
    out = {"cells": {}}
    src09 = {arch: expected_src09(arch) for arch in ("DKT", "AKT", "SAKT", "DKVMN")}
    out["src09"] = src09
    src = get_data_source(_make_rc("ednet_kt1"))
    samples = load_user_samples(src, fold=0)[:n_users]
    prior_p = build_pooled_prior(samples)
    assert len(prior_p) == KC_N, f"prior table {len(prior_p)} != {KC_N}"
    for arch in ("DKT", "AKT", "SAKT", "DKVMN"):
        for akey, aname in ALIGN_NAMES.items():
            alignment = {int(k): v for k, v in load_alignment(aname).items()}
            per_stu, n_fb, n_nonfinite = transfer_cell(samples, alignment, src09[arch], prior_p)
            common = sorted(per_stu)
            deltas = [per_stu[u][0] - per_stu[u][1] for u in common]
            b1_mean = mean_or_none([per_stu[u][0] for u in common])
            deep_mean = mean_or_none([per_stu[u][1] for u in common])
            cell = {
                "n_paired": len(common),
                "deep_mean": deep_mean,
                "b1_mean": b1_mean,
                "delta_b1_minus_deep": mean_or_none(deltas),
                "ci95": boot_ci(deltas),
                "n_nonfinite_probs": n_nonfinite,
                "n_fallback_rows": n_fb,
                "n_kc_prior": len(prior_p),
            }
            if akey == "k3_rep0":
                # anchor arm: must reproduce the b59 seed-42 reference cell
                cell["anchor_n_paired"] = len(common) == REF_N[arch]
                cell["anchor_delta"] = (
                    cell["delta_b1_minus_deep"] is not None
                    and abs(cell["delta_b1_minus_deep"] - REF[arch]["delta"]) <= 0.0001
                )
                cell["anchor_deep_mean"] = (
                    deep_mean is not None and abs(deep_mean - REF[arch]["deep_mean"]) <= 0.0001
                )
            out["cells"][f"ednet_kt1|{arch}|{akey}"] = cell
            print("ednet_kt1", arch, akey, json.dumps(cell, ensure_ascii=False), flush=True)

    out["n_anchor_pass"] = sum(
        3
        for k, c in out["cells"].items()
        if c.get("anchor_n_paired") and c.get("anchor_delta") and c.get("anchor_deep_mean")
    )
    with open("/root/unikt-fork/research/transfer/results/t1_align_quality.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1ALIGN-DONE" if out["n_anchor_pass"] == 4 * 3 else "T1ALIGN-DONE-ANCHOR-FAIL")


if __name__ == "__main__":
    main()
