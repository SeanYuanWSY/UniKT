"""Batch 53: multiplicity correction on the 12 transfer cells (4 archs).

The headline "lookup beats frozen transfer" currently reads "12/12 point
estimates B1-favored, 11/12 per-cell CIs exclude 0" with the standing
caveat: pointwise CIs, no family-wise error control. This batch closes it:

  - recompute per-student deltas for all 12 transfer cells
    (DKT/AKT/SAKT/DKVMN x assistments17/ednet_kt1/junyi2015) on the exact
    b41/b46/b48/b49 row construction (single code path — the structural
    anchors already proved the construction identical across archs);
  - per cell: two-sided bootstrap p (2*min(P(boot<=0), P(boot>=0)),
    B=5000, seed=0, same resampling as the CIs);
  - family = the 12 transfer cells; Holm step-down and Bonferroni verdicts
    at alpha=0.05.

Anchors: every cell's delta_auc + n_paired must reproduce its reference
JSON (t1_fair_rescore.json for DKT/AKT, t1_sakt_fair.json for SAKT,
t1_dkvmn_fair.json for DKVMN) within 0.0001 / equality. Identity cells are
NOT in the family (descriptive elsewhere); they are not recomputed here.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = {
    "assistments17": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
    "junyi2015": "assistments09__to__junyi2015__llm__k3-high__c25k3__rep0",
}
SRC09 = {
    "DKT": "runs/normal/DKT_assistments09_20261003-024700_fold0_bs128",
    "AKT": "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64",
    "SAKT": "runs/normal/SAKT_assistments09_20261003-025330_fold0_bs64",
    "DKVMN": "runs/normal/DKVMN_assistments09_20261003-024801_fold0_bs128",
}
REFS = {}
for _name, _path in (
    ("DKT", "t1_fair_rescore.json"),
    ("AKT", "t1_fair_rescore.json"),
    ("SAKT", "t1_sakt_fair.json"),
    ("DKVMN", "t1_dkvmn_fair.json"),
):
    with open(f"/root/unikt-fork/research/transfer/results/{_path}") as f:
        REFS[_name] = json.load(f)


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def run_ids(q_all):
    rid = [0] * len(q_all)
    r = 0
    for i in range(1, len(q_all)):
        if q_all[i] != q_all[i - 1]:
            r += 1
        rid[i] = r
    return rid


def boot_p_and_ci(d, B=5000, seed=0):
    """Two-sided bootstrap p + 95% CI from the same resample set."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return None, None
    means = np.array([np.mean(d[rng.integers(0, n, n)]) for _ in range(B)])
    p_le = float(np.mean(means <= 0.0))
    p_ge = float(np.mean(means >= 0.0))
    p = min(1.0, 2.0 * min(p_le, p_ge))
    ci = [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]
    return round(p, 6), ci


def transfer_cell(samples, alignment, arch):
    """Per-student (auc_b1 - auc_deep) for one transfer cell. Construction
    identical to b46/b48/b49: run-complete prior, A1-only, keep[-200:]/<6,
    jmap, NaN pair-drop, B1 = pooled per-KC evidence prior."""
    hold_runs_by_ws = []
    hists = {}
    for ws in samples:
        hold = set(ws.holdout_idx.tolist())
        q_all = [int(x) for x in ws.question.tolist()]
        rid = run_ids(q_all)
        hold_runs = {rid[i] for i in hold}
        hold_runs_by_ws.append(hold_runs)
        for i in range(len(ws.sequence)):
            if rid[i] not in hold_runs:
                hists.setdefault(int(ws.sequence[i]), []).append(int(ws.response[i]))
    prior_b1 = {k: float(np.mean(v)) for k, v in hists.items() if v}
    del hold_runs_by_ws  # prior support only; row loop scores hold_pos directly

    rm = restore_any(SRC09[arch])
    deltas = {}
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
                continue
            rows_d.append({"y": y, "pred": pd_})
            rows_b.append({"y": y, "pred": prior_b1.get(int(ws.sequence[i]), 0.5)})
        ad = stratified_auc_rows(rows_d, "pred")
        ab = stratified_auc_rows(rows_b, "pred")
        if ad is not None and ab is not None and rows_d:
            deltas[ws.user_id] = ab - ad
    del rm
    return deltas


def main(n_users=300):
    out = {"cells": {}, "family": {}}
    for ds in ALIGN:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        alignment = {int(k): v for k, v in load_alignment(ALIGN[ds]).items()}
        for arch in ("DKT", "AKT", "SAKT", "DKVMN"):
            deltas = transfer_cell(samples, alignment, arch)
            uids = sorted(deltas)
            vals = [deltas[u] for u in uids]
            mean = round(float(np.mean(vals)), 4) if vals else None
            p, ci = boot_p_and_ci(vals)
            ref = REFS[arch][ds][f"transfer_{arch}"]
            out["cells"][f"{ds}|{arch}"] = {
                "n_paired": len(uids),
                "delta_auc": mean,
                "p_boot_two_sided": p,
                "ci95": ci,
                "ref_n_paired": ref["n_paired"],
                "ref_delta": ref["delta_fair_B1_minus_deep_rc"],
                "anchor_matches_ref": (
                    len(uids) == ref["n_paired"]
                    and mean is not None
                    and ref["delta_fair_B1_minus_deep_rc"] is not None
                    and abs(mean - ref["delta_fair_B1_minus_deep_rc"]) <= 0.0001
                ),
            }
            print(ds, arch, json.dumps(out["cells"][f"{ds}|{arch}"], ensure_ascii=False), flush=True)

    # family: 12 transfer cells, Holm step-down + Bonferroni at alpha=0.05
    keys = sorted(out["cells"])
    raw = [(out["cells"][k]["p_boot_two_sided"], k) for k in keys]
    raw.sort()
    m = len(raw)
    holm = {}
    running_max = 0.0
    for rank, (p, k) in enumerate(raw):
        adj = (m - rank) * p
        running_max = max(running_max, adj)
        holm[k] = min(1.0, round(running_max, 6))
    alpha = 0.05
    out["family"] = {
        "n_family": m,
        "alpha": alpha,
        "holm_rejected": [k for k in keys if holm[k] < alpha],
        "holm_adjusted_p": holm,
        "bonferroni_rejected": [k for k in keys if (out["cells"][k]["p_boot_two_sided"] is not None and m * out["cells"][k]["p_boot_two_sided"] < alpha)],
        "n_anchor_pass": sum(1 for k in keys if out["cells"][k]["anchor_matches_ref"]),
    }
    with open("/root/unikt-fork/research/transfer/results/t1_multiplicity.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1MULT-DONE")


if __name__ == "__main__":
    main()
