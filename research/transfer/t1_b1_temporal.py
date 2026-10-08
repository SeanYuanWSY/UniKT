"""Batch 56: B1 temporal-split robustness on the 12 transfer cells.

The pooled B1 prior pools run-complete rows across students over each
student's whole non-holdout span. A deployment-realism question (and a
likely reviewer question) is whether the lookup win depends on pooling the
full timeline — including each student's late-course rows — rather than
early evidence only. This batch rebuilds the B1 table from:

  - pooled   (anchor arm: must reproduce the reference delta bit-for-bit)
  - firsthalf  (each student's first floor(n/2) non-holdout rows, pooled)
  - secondhalf (the remaining late rows, pooled — descriptive symmetry arm)

and scores the SAME holdout rows (construction otherwise identical to
b41/b46/b48/b49/b53: run-complete exclusion, A1-only, keep[-200:]/<6, jmap,
NaN pair-drop, fallback 0.5 for KCs missing from that arm's table).

Anchors: per cell, n_paired equality and the pooled-arm delta must
reproduce the reference JSON (t1_fair_rescore / t1_sakt_fair /
t1_dkvmn_fair) within 0.0001 — proving rows and deep side are the same
construction, so arm differences are attributable to the table alone.
"""
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


def boot_ci(d, B=5000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return None
    boots = [np.mean(d[rng.integers(0, n, n)]) for _ in range(B)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def mean_or_none(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def build_priors(samples):
    """Pooled / first-half / second-half run-complete priors.

    First/second split is per student over that student's non-holdout rows
    in position order: first floor(n/2) rows -> firsthalf, rest -> secondhalf.
    """
    pooled, first, second = {}, {}, {}
    for ws in samples:
        hold = set(ws.holdout_idx.tolist())
        q_all = [int(x) for x in ws.question.tolist()]
        rid = run_ids(q_all)
        hold_runs = {rid[i] for i in hold}
        rows = [i for i in range(len(ws.sequence)) if rid[i] not in hold_runs]
        half = len(rows) // 2
        for j, i in enumerate(rows):
            kc = int(ws.sequence[i])
            y = int(ws.response[i])
            pooled.setdefault(kc, []).append(y)
            (first if j < half else second).setdefault(kc, []).append(y)

    def means(d):
        return {k: float(np.mean(v)) for k, v in d.items() if v}

    return means(pooled), means(first), means(second)


def transfer_cell(samples, alignment, arch, priors):
    """Per-student (ab_pooled, ab_first, ab_second, ad) for one cell.

    One forward pass per student; the three B1 arms score the same rows
    with their own table (fallback 0.5 where that arm's table lacks the KC).
    """
    prior_p, prior_f, prior_s = priors
    rm = restore_any(SRC09[arch])
    per_stu = {}
    n_fb = {"pooled": 0, "first": 0, "second": 0}
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
        rows_d, rows_p, rows_f, rows_s = [], [], [], []
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
            kc = int(ws.sequence[i])
            rows_d.append({"y": y, "pred": pd_})
            for arm, prior, rows in (
                ("pooled", prior_p, rows_p),
                ("first", prior_f, rows_f),
                ("second", prior_s, rows_s),
            ):
                if kc not in prior:
                    n_fb[arm] += 1
                rows.append({"y": y, "pred": prior.get(kc, 0.5)})
        ad = stratified_auc_rows(rows_d, "pred")
        ap = stratified_auc_rows(rows_p, "pred")
        af = stratified_auc_rows(rows_f, "pred")
        as_ = stratified_auc_rows(rows_s, "pred")
        if ad is not None and ap is not None and af is not None and as_ is not None and rows_d:
            per_stu[ws.user_id] = (ap, af, as_, ad)
    del rm
    return per_stu, n_fb


def main(n_users=300):
    out = {"cells": {}}
    for ds in ALIGN:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        alignment = {int(k): v for k, v in load_alignment(ALIGN[ds]).items()}
        priors = build_priors(samples)
        for arch in ("DKT", "AKT", "SAKT", "DKVMN"):
            per_stu, n_fb = transfer_cell(samples, alignment, arch, priors)
            common = sorted(per_stu)
            dp = [per_stu[u][0] - per_stu[u][3] for u in common]
            df = [per_stu[u][1] - per_stu[u][3] for u in common]
            ds_ = [per_stu[u][2] - per_stu[u][3] for u in common]
            ref = REFS[arch][ds][f"transfer_{arch}"]
            mp = mean_or_none(dp)
            out["cells"][f"{ds}|{arch}"] = {
                "n_paired": len(common),
                "delta_pooled": mp,
                "anchor_matches_ref": (
                    len(common) == ref["n_paired"]
                    and mp is not None
                    and ref["delta_fair_B1_minus_deep_rc"] is not None
                    and abs(mp - ref["delta_fair_B1_minus_deep_rc"]) <= 0.0001
                ),
                "delta_first": mean_or_none(df),
                "ci95_first": boot_ci(df),
                "delta_second": mean_or_none(ds_),
                "ci95_second": boot_ci(ds_),
                "n_kc_prior": {"pooled": len(priors[0]), "first": len(priors[1]), "second": len(priors[2])},
                "n_fallback_rows": n_fb,
            }
            print(ds, arch, json.dumps(out["cells"][f"{ds}|{arch}"], ensure_ascii=False), flush=True)

    out["n_anchor_pass"] = sum(1 for c in out["cells"].values() if c["anchor_matches_ref"])
    with open("/root/unikt-fork/research/transfer/results/t1_b1_temporal.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1B1TMP-DONE")


if __name__ == "__main__":
    main()
