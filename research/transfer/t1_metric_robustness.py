"""Batch 46: metric robustness — calibration metrics on the exact fair rows.

Every conclusion in the main table rests on per-student AUC alone; whether
the table-vs-deep deltas survive proper-scoring metrics has never been
measured. This batch recomputes the batch41/43 fair rows (same samples,
same run-complete prior_fixed2, same same-support pairing) and adds, per
cell: per-student Brier and log-loss (clip 1e-7), paired delta B1-minus-deep
with cluster bootstrap CI, plus pooled row-level Brier/log-loss.

Deltas vs t1_fair_rescore.py are exactly three:
 (1) metric computation added (brier / logloss per student, paired bootstrap
     on the metric deltas);
 (2) AUC kept as the ANCHOR only — n_paired and delta_fair_B1_minus_deep_rc
     must reproduce t1_fair_rescore.json within rounding (anchor gate
     tolerance 0.0001), proving the row support is unchanged;
 (3) dropped: the old-path / sibling-drift prior diagnostics, B1 full-subset
     means, and the symmetric-closure attack (no new prediction surface —
     b36/t1fair3 already cover it). prior_old is not computed at all.

fork-review hardening (PASS-with-fixes): NaN-pred rows are dropped from
metric rows in PAIRS (deep+b1) with a counter so the anchor cannot pass
while metric rows differ; clip counts are split deep-side vs b1-side (B1
priors hit exact 0/1 on rare KCs — logloss asymmetry must be visible, not
inferred); the G2 sign-agreement tally is computed IN the script per cell.
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
RUNS = {
    "identity_AKT": {"assistments17": "runs/normal/AKT_assistments17_20261003-145008_fold0_bs64", "ednet_kt1": "runs/normal/AKT_ednet_kt1_20261003-190537_fold0_bs64", "junyi2015": "runs/normal/AKT_junyi2015_20261004-195553_fold0_bs64"},
    "identity_DKT": {"assistments17": "runs/normal/DKT_assistments17_20261003-144042_fold0_bs128", "ednet_kt1": "runs/normal/DKT_ednet_kt1_20261003-190327_fold0_bs128", "junyi2015": "runs/normal/DKT_junyi2015_20261004-195512_fold0_bs128"},
    "transfer_AKT": {ds: "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64" for ds in ALIGN},
    "transfer_DKT": {ds: "runs/normal/DKT_assistments09_20261003-024700_fold0_bs128" for ds in ALIGN},
}
# anchor: t1_fair_rescore.json (tag t1fair3) rc fields
with open("/root/unikt-fork/research/transfer/results/t1_fair_rescore.json") as f:
    T1FAIR_REF = json.load(f)

EPS = 1e-7


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
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return None
    boots = [np.mean(d[rng.integers(0, n, n)]) for _ in range(B)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def brier(rows):
    return float(np.mean([(r["pred"] - r["y"]) ** 2 for r in rows]))


def logloss(rows):
    n_clip = 0
    tot = 0.0
    for r in rows:
        p = r["pred"]
        if p < EPS or p > 1 - EPS:
            n_clip += 1
            p = min(max(p, EPS), 1 - EPS)
        tot += -(r["y"] * np.log(p) + (1 - r["y"]) * np.log(1 - p))
    return tot / len(rows), n_clip


def mean_or_none(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def main(n_users=300):
    out = {}
    for ds in ALIGN:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]

        # run-complete evidence-only prior (batch43 adjudicated construction)
        hist_all2 = {}
        for ws in samples:
            hold = set(ws.holdout_idx.tolist())
            q_all = [int(x) for x in ws.question.tolist()]
            rid = run_ids(q_all)
            hold_runs = {rid[i] for i in hold}
            for i in range(len(ws.sequence)):
                if rid[i] not in hold_runs:
                    hist_all2.setdefault(int(ws.sequence[i]), []).append(int(ws.response[i]))
        prior_b1 = {k: float(np.mean(v)) for k, v in hist_all2.items() if v}

        alignment = {int(k): v for k, v in load_alignment(ALIGN[ds]).items()}
        res = {"n_kc_prior_b1": len(prior_b1)}

        for mtag, runmap in RUNS.items():
            rm = restore_any(runmap[ds])
            per_stu = {}  # uid -> dict of metrics
            n_dummy_excluded = 0
            n_nonfinite_probs = 0
            for ws in samples:
                if mtag.startswith("identity"):
                    jmap = {i: i for i in range(len(ws.sequence))}
                else:
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
                s_seq = ws.sequence.tolist() if mtag.startswith("identity") else [mapping[int(ws.sequence[i])] for i in keep]
                s_resp = [int(x) for x in ws.response.tolist()] if mtag.startswith("identity") else [int(ws.response[i]) for i in keep]
                s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
                r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
                probs = _forward_probs(rm, s, r_, None)[0]
                rows_d, rows_b = [], []
                n_nan_dropped = 0
                for i in sorted(hold_pos):
                    if i == 0 or i not in jmap:
                        continue
                    if jmap[i] == 0:
                        n_dummy_excluded += 1
                        continue
                    if is_a1(ws, i):
                        continue
                    y = int(ws.response[i])
                    pd_ = float(probs[jmap[i]])
                    pb_ = prior_b1.get(int(ws.sequence[i]), 0.5)
                    if pd_ != pd_:  # NaN guard: drop the ROW PAIR so metric rows == AUC rows
                        n_nan_dropped += 1
                        continue
                    rows_d.append({"y": y, "pred": pd_})
                    rows_b.append({"y": y, "pred": pb_})
                n_nonfinite_probs += n_nan_dropped
                ad = stratified_auc_rows(rows_d, "pred")
                ab = stratified_auc_rows(rows_b, "pred")
                if ad is not None and ab is not None and rows_d:
                    ll_d, cd_ = logloss(rows_d)
                    ll_b, cb_ = logloss(rows_b)
                    per_stu[ws.user_id] = {
                        "auc_d": ad, "auc_b": ab,
                        "brier_d": brier(rows_d), "brier_b": brier(rows_b),
                        "ll_d": ll_d, "ll_b": ll_b,
                        "n_rows": len(rows_d), "n_clip_deep": cd_, "n_clip_b1": cb_,
                    }
            common = sorted(per_stu)
            ref = T1FAIR_REF[ds][mtag]
            if not common:
                res[mtag] = {
                    "n_paired": 0,
                    "anchor_delta_auc_rc": None,
                    "anchor_matches_t1fair": False,
                    "brier": None, "logloss": None, "pooled": None,
                    "g2_agree_brier": None, "g2_agree_logloss": None,
                    "n_clip_deep_rows": 0, "n_clip_b1_rows": 0,
                    "n_dummy_pos0_excluded": n_dummy_excluded,
                    "n_nonfinite_probs": n_nonfinite_probs,
                }
                del rm
                continue
            auc_diffs = np.array([per_stu[u]["auc_b"] - per_stu[u]["auc_d"] for u in common])
            brier_diffs = np.array([per_stu[u]["brier_b"] - per_stu[u]["brier_d"] for u in common])
            ll_diffs = np.array([per_stu[u]["ll_b"] - per_stu[u]["ll_d"] for u in common])
            auc_mean = mean_or_none(auc_diffs.tolist())
            brier_mean = mean_or_none(brier_diffs.tolist())
            ll_mean = mean_or_none(ll_diffs.tolist())
            # G2 tally computed in-script: agreement = B1 better on both or
            # worse on both (brier/ll deltas are NEGATIVE when B1 better).
            def agree(delta_auc, delta_m):
                if delta_auc is None or delta_m is None or delta_auc == 0 or delta_m == 0:
                    return None
                return (delta_auc > 0) == (delta_m < 0)
            # pooled: weight per-student means by their row counts
            pooled_rows = sum(per_stu[u]["n_rows"] for u in common)
            pooled_brier_d = sum(per_stu[u]["brier_d"] * per_stu[u]["n_rows"] for u in common) / pooled_rows
            pooled_brier_b = sum(per_stu[u]["brier_b"] * per_stu[u]["n_rows"] for u in common) / pooled_rows
            pooled_ll_d = sum(per_stu[u]["ll_d"] * per_stu[u]["n_rows"] for u in common) / pooled_rows
            pooled_ll_b = sum(per_stu[u]["ll_b"] * per_stu[u]["n_rows"] for u in common) / pooled_rows
            res[mtag] = {
                "n_paired": len(common),
                "anchor_delta_auc_rc": auc_mean,
                "anchor_matches_t1fair": (
                    len(common) == ref["n_paired"]
                    and auc_mean is not None
                    and ref["delta_fair_B1_minus_deep_rc"] is not None
                    and abs(auc_mean - ref["delta_fair_B1_minus_deep_rc"]) <= 0.0001
                ),
                "brier": {
                    "deep_mean": mean_or_none([per_stu[u]["brier_d"] for u in common]),
                    "b1_mean": mean_or_none([per_stu[u]["brier_b"] for u in common]),
                    "delta_b1_minus_deep": brier_mean,
                    "ci95": boot_ci(brier_diffs),
                },
                "logloss": {
                    "deep_mean": mean_or_none([per_stu[u]["ll_d"] for u in common]),
                    "b1_mean": mean_or_none([per_stu[u]["ll_b"] for u in common]),
                    "delta_b1_minus_deep": ll_mean,
                    "ci95": boot_ci(ll_diffs),
                },
                "pooled": {
                    "n_rows": pooled_rows,
                    "brier_deep": round(pooled_brier_d, 4), "brier_b1": round(pooled_brier_b, 4),
                    "ll_deep": round(pooled_ll_d, 4), "ll_b1": round(pooled_ll_b, 4),
                },
                "g2_agree_brier": agree(auc_mean, brier_mean),
                "g2_agree_logloss": agree(auc_mean, ll_mean),
                "n_clip_deep_rows": sum(per_stu[u]["n_clip_deep"] for u in common),
                "n_clip_b1_rows": sum(per_stu[u]["n_clip_b1"] for u in common),
                "n_dummy_pos0_excluded": n_dummy_excluded,
                "n_nonfinite_probs": n_nonfinite_probs,
            }
            del rm

        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/t1_metric_robustness.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1METRIC-ROBUST-DONE")


if __name__ == "__main__":
    main()
