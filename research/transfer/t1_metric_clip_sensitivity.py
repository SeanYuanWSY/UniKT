"""Batch 50: log-loss clipping (eps) sensitivity — closes Codex b46/48 F6.

Codex finding: the b46 log-loss conclusions are conditional on the eps=1e-7
clipping rule; deep-side clipped rows concentrate in ONE cell (48/48 in
A17 identity_AKT, 0.88% of that cell's rows), and each saturated wrong-side
prediction contributes ~16.1 loss units at 1e-7 — same order as that cell's
LL delta (-0.1452). This batch recomputes the exact b46 rows and reports:
 (1) per-student LL delta at eps in {1e-5, 1e-7, 1e-10} with CIs;
 (2) per-cell clip detail: counts split deep/b1, wrong-side vs correct-side
     (only wrong-side clips contribute ~|log eps|; correct-side ~0), and
     the number of students affected;
 (3) anchors: AUC delta + n_paired reproduce t1_fair_rescore.json, and the
     Brier delta (eps-free) reproduces t1_metric_robustness.json per cell.
Row construction identical to b46 (run-complete prior, A1-only, same-support,
NaN pair-drop).
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
EPS_LIST = (1e-5, 1e-7, 1e-10)
with open("/root/unikt-fork/research/transfer/results/t1_fair_rescore.json") as f:
    T1FAIR_REF = json.load(f)
with open("/root/unikt-fork/research/transfer/results/t1_metric_robustness.json") as f:
    TMET_REF = json.load(f)


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


def ll_eps(rows, eps):
    tot = 0.0
    for r in rows:
        p = r["pred"]
        p = min(max(p, eps), 1 - eps)
        tot += -(r["y"] * np.log(p) + (1 - r["y"]) * np.log(1 - p))
    return tot / len(rows)


def clip_detail(rows, eps):
    """wrong-side clipped rows are the only material contributors (~|log eps|)."""
    n_clip = n_wrong = 0
    for r in rows:
        p = r["pred"]
        if p < eps or p > 1 - eps:
            n_clip += 1
            if (p > 1 - eps and r["y"] == 0) or (p < eps and r["y"] == 1):
                n_wrong += 1
    return n_clip, n_wrong


def mean_or_none(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def main(n_users=300):
    out = {}
    for ds in ALIGN:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]

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
        res = {}
        for mtag, runmap in RUNS.items():
            rm = restore_any(runmap[ds])
            per_stu = {}
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
                        n_nonfinite_probs += 1
                        continue
                    rows_d.append({"y": y, "pred": pd_})
                    rows_b.append({"y": y, "pred": prior_b1.get(int(ws.sequence[i]), 0.5)})
                ad = stratified_auc_rows(rows_d, "pred")
                ab = stratified_auc_rows(rows_b, "pred")
                if ad is not None and ab is not None and rows_d:
                    per_stu[ws.user_id] = {
                        "auc_d": ad, "auc_b": ab,
                        "brier_d": float(np.mean([(r["pred"] - r["y"]) ** 2 for r in rows_d])),
                        "brier_b": float(np.mean([(r["pred"] - r["y"]) ** 2 for r in rows_b])),
                        "ll_d": {e: ll_eps(rows_d, e) for e in EPS_LIST},
                        "ll_b": {e: ll_eps(rows_b, e) for e in EPS_LIST},
                        "rows_d": rows_d, "rows_b": rows_b,
                    }
            common = sorted(per_stu)
            auc_diffs = np.array([per_stu[u]["auc_b"] - per_stu[u]["auc_d"] for u in common])
            brier_diffs = np.array([per_stu[u]["brier_b"] - per_stu[u]["brier_d"] for u in common])
            auc_mean = mean_or_none(auc_diffs.tolist())
            brier_mean = mean_or_none(brier_diffs.tolist())
            ref_auc = T1FAIR_REF[ds][mtag]["delta_fair_B1_minus_deep_rc"]
            ref_brier = TMET_REF[ds][mtag]["brier"]["delta_b1_minus_deep"]
            cell = {
                "n_paired": len(common),
                "anchor_auc_matches_t1fair": (
                    len(common) == T1FAIR_REF[ds][mtag]["n_paired"]
                    and auc_mean is not None and ref_auc is not None
                    and abs(auc_mean - ref_auc) <= 0.0001
                ),
                "anchor_brier_matches_b46": (
                    brier_mean is not None and ref_brier is not None
                    and abs(brier_mean - ref_brier) <= 0.0001
                ),
            }
            for e in EPS_LIST:
                d = np.array([per_stu[u]["ll_b"][e] - per_stu[u]["ll_d"][e] for u in common])
                cell[f"ll_delta_eps{e:g}"] = {"mean": mean_or_none(d.tolist()), "ci95": boot_ci(d)}
            # LL path anchor at the b46 eps (fork P2-2): proves the LL
            # construction is identical, not just the Brier one.
            ref_ll = TMET_REF[ds][mtag]["logloss"]["delta_b1_minus_deep"]
            ll7 = cell["ll_delta_eps1e-07"]["mean"]
            cell["anchor_ll_eps1e-07_matches_b46"] = (
                ll7 is not None and ref_ll is not None and abs(ll7 - ref_ll) <= 0.0001
            )
            # clip detail at all three eps (fork P2-3): 1e-7 doubles as a
            # reproduction of b46's n_clip counts.
            for tag in ("d", "b"):
                for e in EPS_LIST:
                    allrows = [r for u in common for r in per_stu[u][f"rows_{tag}"]]
                    nc, nw = clip_detail(allrows, e)
                    cell[f"clip_{tag}_eps{e:g}"] = {"n_clip": nc, "n_wrongside": nw}
            cell["clip_students_affected_deep_eps1e-10"] = sum(
                1 for u in common if any(r["pred"] < 1e-10 or r["pred"] > 1 - 1e-10 for r in per_stu[u]["rows_d"])
            )
            cell["n_nonfinite_probs"] = n_nonfinite_probs
            res[mtag] = cell
            del rm

        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/t1_metric_clip_sensitivity.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1CLIP-SENS-DONE")


if __name__ == "__main__":
    main()
