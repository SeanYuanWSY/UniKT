"""Batch 51: metric robustness (Brier/LL) on the SAKT + DKVMN fair rows.

Codex b46/48 F5: the architecture axis (b48 SAKT, b49 DKVMN) supplies AUC
evidence only. This batch closes that gap: per-student Brier and log-loss
(eps=1e-7) on the exact b48/b49 rows — same samples, run-complete prior,
A1-only, same-support pairing, NaN pair-drop.

Anchors: per cell, n_paired and the AUC delta must reproduce
t1_sakt_fair.json / t1_dkvmn_fair.json bit-identically (construction
unchanged). Identity run dirs are glob-discovered exactly as in b48/b49.
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
    "SAKT": "runs/normal/SAKT_assistments09_20261003-025330_fold0_bs64",
    "DKVMN": "runs/normal/DKVMN_assistments09_20261003-024801_fold0_bs128",
}
with open("/root/unikt-fork/research/transfer/results/t1_sakt_fair.json") as f:
    SAKT_REF = json.load(f)
with open("/root/unikt-fork/research/transfer/results/t1_dkvmn_fair.json") as f:
    DKVMN_REF = json.load(f)

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


def newest_run(arch, ds):
    cands = sorted(glob.glob(f"/root/unikt-fork/runs/normal/{arch}_{ds}_*_fold0_*"))
    assert cands, f"no {arch} run dir for {ds}"
    assert os.path.exists(cands[-1] + "/best_model.pth"), f"incomplete {arch} run: {cands[-1]}"
    return "runs/normal/" + os.path.basename(cands[-1])


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
        for arch in ("SAKT", "DKVMN"):
            ref_json = SAKT_REF if arch == "SAKT" else DKVMN_REF
            for mode in ("identity", "transfer"):
                mtag = f"{mode}_{arch}"
                run_dir = newest_run(arch, ds) if mode == "identity" else SRC09[arch]
                rm = restore_any(run_dir)
                per_stu = {}
                n_nonfinite_probs = 0
                for ws in samples:
                    if mode == "identity":
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
                    s_seq = ws.sequence.tolist() if mode == "identity" else [mapping[int(ws.sequence[i])] for i in keep]
                    s_resp = [int(x) for x in ws.response.tolist()] if mode == "identity" else [int(ws.response[i]) for i in keep]
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
                        ll_d, cd_ = logloss(rows_d)
                        ll_b, cb_ = logloss(rows_b)
                        per_stu[ws.user_id] = {
                            "auc_d": ad, "auc_b": ab,
                            "brier_d": brier(rows_d), "brier_b": brier(rows_b),
                            "ll_d": ll_d, "ll_b": ll_b,
                            "n_clip_deep": cd_, "n_clip_b1": cb_,
                        }
                common = sorted(per_stu)
                auc_diffs = np.array([per_stu[u]["auc_b"] - per_stu[u]["auc_d"] for u in common])
                brier_diffs = np.array([per_stu[u]["brier_b"] - per_stu[u]["brier_d"] for u in common])
                ll_diffs = np.array([per_stu[u]["ll_b"] - per_stu[u]["ll_d"] for u in common])
                auc_mean = mean_or_none(auc_diffs.tolist())
                brier_mean = mean_or_none(brier_diffs.tolist())
                ll_mean = mean_or_none(ll_diffs.tolist())
                ref = ref_json[ds][mtag]

                def agree(delta_auc, delta_m):
                    if delta_auc is None or delta_m is None or delta_auc == 0 or delta_m == 0:
                        return None
                    return (delta_auc > 0) == (delta_m < 0)

                cell = {
                    "run_dir": run_dir,
                    "n_paired": len(common),
                    "anchor_matches_ref": (
                        len(common) == ref["n_paired"]
                        and auc_mean is not None
                        and ref["delta_fair_B1_minus_deep_rc"] is not None
                        and abs(auc_mean - ref["delta_fair_B1_minus_deep_rc"]) <= 0.0001
                    ),
                    "delta_auc": auc_mean,
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
                    "g2_agree_brier": agree(auc_mean, brier_mean),
                    "g2_agree_logloss": agree(auc_mean, ll_mean),
                    "n_clip_deep_rows": sum(per_stu[u]["n_clip_deep"] for u in common),
                    "n_clip_b1_rows": sum(per_stu[u]["n_clip_b1"] for u in common),
                    "n_nonfinite_probs": n_nonfinite_probs,
                }
                res[mtag] = cell
                del rm

        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/t1_metric_robustness_arch.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1METRIC-ARCH-DONE")


if __name__ == "__main__":
    main()
