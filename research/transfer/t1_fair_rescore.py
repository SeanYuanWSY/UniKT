"""Batch 41: T1 fair recompute (Codex Run A P1-2/P1-4/P1-5, Run B F1; erratum #13/#18 record).

Three fixes over t1_paired_ci.py:
 (1) B1 prior membership fixed by evidence only — no holdout-biclass
     selection leak (build_rows' `0<sum(y)<len(rows)` filtered students by
     FUTURE holdout labels before the prior was fitted).
 (2) Per-cell B1 rescored on the deep cell's exact scored (student, row)
     support -> paired delta is a same-support comparison.
 (3) Transfer scoring excludes compressed position 0 (DKT/AKT padded dummy
     column); incidence counted as diagnostic.
Plus symmetric-closure copy attack (erratum #13): preds y_{t-1}, 1-y_{t-1},
y_{t-2}, 1-y_{t-2} on A1-only kept holdout rows; record per-domain sym_max.

Self-checks: old-path B1_mean reproduces t1_paired_ci.json (±0.002);
copy AUCs reproduce a1_copy_leakage.json (±0.002); closure identity.
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
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
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
B1_MEAN_REF = {"assistments17": 0.548, "ednet_kt1": 0.5252, "junyi2015": 0.5624}
COPY_REF = {  # a1_copy_leakage.json, per-student mean over students with valid AUC
    "assistments17": {"copy_prev": 0.4704, "copy_prev2": 0.4609},
    "ednet_kt1": {"copy_prev": 0.4557, "copy_prev2": 0.4430},
    "junyi2015": {"copy_prev": 0.5141, "copy_prev2": 0.4980},
}


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def boot_ci(d, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return None
    boots = [np.mean(d[rng.integers(0, n, n)]) for _ in range(B)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def mean_or_none(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def main(n_users=300):
    out = {}
    for ds in ALIGN:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]

        # ---- (1) prior with evidence-only membership: every student's
        # non-holdout evidence rows, no holdout-composition filtering.
        hist_all = {}
        for ws in samples:
            hold = set(ws.holdout_idx.tolist())
            for i in range(len(ws.sequence)):
                if i in hold:
                    continue
                hist_all.setdefault(int(ws.sequence[i]), []).append(int(ws.response[i]))
        prior_fixed = {k: float(np.mean(v)) for k, v in hist_all.items() if v}

        # old path (selection-leaked membership) for drift gate G3
        students_old = build_rows(samples, None)
        prior_old = fit_global_kc_prior(students_old)

        res = {
            "n_kc_prior_fixed": len(prior_fixed),
            "n_kc_prior_old": len(prior_old),
            "prior_drift_max_abs": round(
                float(max(abs(prior_fixed.get(k, 0.5) - prior_old.get(k, 0.5)) for k in set(prior_fixed) | set(prior_old))),
                4,
            ),
        }

        # B1 full-subset means — fork-review P1: reproduction check must use the
        # OLD prior (same membership path as t1_paired_ci); fixed-vs-old diff is
        # the G3 drift quantity. Two separate fields, two separate meanings.
        def b1_full_with(prior):
            vals = {}
            for ws in samples:
                hold = sorted(set(ws.holdout_idx.tolist()))
                rows = [
                    {"y": int(ws.response[i]), "pred": prior.get(int(ws.sequence[i]), 0.5)}
                    for i in hold
                    if i > 0 and not is_a1(ws, i)
                ]
                a = stratified_auc_rows(rows, "pred")
                if a is not None:
                    vals[ws.user_id] = a
            return vals

        b1_old = b1_full_with(prior_old)
        b1_full = b1_full_with(prior_fixed)
        res["B1_old_full_mean"] = mean_or_none(list(b1_old.values()))
        res["B1_fixed_full_mean"] = mean_or_none(list(b1_full.values()))
        res["B1_fixed_full_n"] = len(b1_full)
        res["selfcheck_B1_old_vs_t1"] = (
            res["B1_old_full_mean"] is not None
            and abs(res["B1_old_full_mean"] - B1_MEAN_REF[ds]) <= 0.002
        )
        res["prior_drift_g3"] = (
            round(res["B1_fixed_full_mean"] - res["B1_old_full_mean"], 4)
            if res["B1_fixed_full_mean"] is not None and res["B1_old_full_mean"] is not None
            else None
        )

        alignment = {int(k): v for k, v in load_alignment(ALIGN[ds]).items()}

        # ---- (2)+(3) per-cell same-support comparison
        for mtag, runmap in RUNS.items():
            rm = restore_any(runmap[ds])
            deep = {}
            b1c = {}
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
                for i in sorted(hold_pos):
                    if i == 0 or i not in jmap:
                        continue
                    if jmap[i] == 0:
                        # compressed position 0 = padded dummy column (DKT_model pads 0)
                        n_dummy_excluded += 1
                        continue
                    if is_a1(ws, i):
                        continue
                    y = int(ws.response[i])
                    rows_d.append({"y": y, "pred": float(probs[jmap[i]])})
                    rows_b.append({"y": y, "pred": prior_fixed.get(int(ws.sequence[i]), 0.5)})
                ad = stratified_auc_rows(rows_d, "pred")
                # stratified_auc_rows silently drops NaN preds (ok = vs == vs) —
                # count them so a same-support break is visible, not silent.
                n_nonfinite_probs += sum(1 for r in rows_d if r["pred"] != r["pred"])
                ab = stratified_auc_rows(rows_b, "pred")
                if ad is not None and ab is not None:
                    deep[ws.user_id] = ad
                    b1c[ws.user_id] = ab
            common = sorted(u for u in deep if u in b1c)
            diffs = np.array([b1c[u] - deep[u] for u in common])
            res[mtag] = {
                "deep_mean_samesupport": mean_or_none([deep[u] for u in common]),
                "B1_mean_samesupport": mean_or_none([b1c[u] for u in common]),
                "n_paired": len(common),
                "delta_fair_B1_minus_deep": mean_or_none(diffs.tolist()) if len(diffs) else None,
                "ci95_fair": boot_ci(diffs) if len(diffs) else None,
                "n_dummy_pos0_excluded": n_dummy_excluded,
                "n_nonfinite_probs": n_nonfinite_probs,
            }
            del rm

        # ---- symmetric-closure copy attack (erratum #13 record)
        # fork-review P1: b36's a1_copy_leakage.py used the FULL sample set
        # (no [:n_users] truncation; EdNet n=683) — run this model-free section
        # on the untruncated set or the ±0.002 selfcheck fails for sampling
        # reasons, not code reasons.
        samples_all = load_user_samples(src, fold=0)
        att = {k: [] for k in ("copy_prev", "neg_prev", "copy_prev2", "neg_prev2")}
        for ws in samples_all:
            hold = sorted(set(ws.holdout_idx.tolist()))
            kept = [i for i in hold if i > 0 and not is_a1(ws, i)]
            y = [int(v) for v in ws.response.tolist()]
            for name, lag, neg in (("copy_prev", 1, False), ("neg_prev", 1, True), ("copy_prev2", 2, False), ("neg_prev2", 2, True)):
                rows = [
                    {"y": y[i], "pred": (1 - y[i - lag]) if neg else y[i - lag]}
                    for i in kept
                    if i >= lag
                ]
                a = stratified_auc_rows(rows, "pred")
                if a is not None:
                    att[name].append(a)
        att_mean = {k: mean_or_none(v) for k, v in att.items()}
        if all(att_mean.get(k) is not None for k in ("copy_prev", "neg_prev", "copy_prev2", "neg_prev2")):
            # measured derivation first (P2-3): trust what this batch computed,
            # cross-check against the b36 identity-derived constant.
            # NOTE family max must include the POSITIVE attacks too — Junyi's
            # copy_prev (0.5141) dominates its negations (first run's
            # max(0.5, neg, neg2) omitted it; fixed via t1fair2 rerun).
            att_mean["sym_max_measured"] = round(
                max(0.5, att_mean["copy_prev"], att_mean["neg_prev"], att_mean["copy_prev2"], att_mean["neg_prev2"]), 4
            )
            att_mean["sym_max_ref_identity"] = round(
                max(
                    0.5,
                    COPY_REF[ds]["copy_prev"],
                    1 - COPY_REF[ds]["copy_prev"],
                    COPY_REF[ds]["copy_prev2"],
                    1 - COPY_REF[ds]["copy_prev2"],
                ),
                4,
            )
            att_mean["selfcheck_copy_vs_b36"] = (
                abs(att_mean["copy_prev"] - COPY_REF[ds]["copy_prev"]) <= 0.002
                and abs(att_mean["copy_prev2"] - COPY_REF[ds]["copy_prev2"]) <= 0.002
            )
            att_mean["selfcheck_neg_identity"] = (
                abs((1 - att_mean["copy_prev"]) - att_mean["neg_prev"]) <= 0.001
                and abs((1 - att_mean["copy_prev2"]) - att_mean["neg_prev2"]) <= 0.001
            )
        else:
            att_mean["selfcheck_copy_vs_b36"] = None
            att_mean["selfcheck_neg_identity"] = None
        res["sym_closure_attack"] = att_mean

        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/t1_fair_rescore.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1FAIR-DONE")


if __name__ == "__main__":
    main()
