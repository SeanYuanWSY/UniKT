"""Batch 42: ability-decompose v3 — leakage-free alpha tuning (Codex Run A P1-7 / erratum #19).

v2 leak: the global KC prior was fitted on ALL evidence rows, then every-5th
evidence rows were reused as the tuning pseudo-holdout — each tuning row's own
label re-entered its prediction through the prior. (ability_before itself is
clean: hist is appended after prediction.)

v3 design:
 (1) tuning rows = every-5th non-holdout non-A1 evidence row (same rule as v2);
     prior_tuning fitted from the OTHER evidence rows only (all students) —
     no tuning label can enter prior_tuning;
 (2) alpha* selected on tuning rows with prior_tuning + ability_before;
 (3) eval on real holdout (A1-excluded) with alpha frozen; prior_final refit on
     all evidence (standard refit-after-tuning; holdout untouched);
     sensitivity eval with prior_tuning also reported;
 (4) v2-repro anchor: rerun the selection with the LEAKY prior (all evidence)
     — must reproduce v2's alpha*=1.0 in all three domains.
"""
import json
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
WIN, MIN_HIST, CAP = 20, 3, 100
V2_ALPHA_REF = {"assistments17": 1.0, "ednet_kt1": 1.0, "junyi2015": 1.0}
V2_B1_REF = {"assistments17": 0.5481, "ednet_kt1": 0.5252, "junyi2015": 0.5624}


def ability_before(hist_deque, q_i):
    vals = [y for (qj, y) in list(hist_deque)[-CAP:] if qj != q_i][-WIN:]
    return float(np.mean(vals)) if len(vals) >= MIN_HIST else None


def is_a1(q_all, kc_all, i):
    return i > 0 and q_all[i] == q_all[i - 1] and kc_all[i] != kc_all[i - 1]


def main(n_users=300):
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]

        # v2-repro anchor: leaky prior (all evidence, incl. tuning rows)
        students = build_rows(samples, None)
        prior_leaky = fit_global_kc_prior(students)

        # ---- pass 0: collect tuning-row indices per student (v2 rule)
        tuning_idx = {}
        hist_evidence = {}  # uid -> non-tuning evidence (kc, y) pairs, for prior_tuning
        for ws in samples:
            hold_set = set(ws.holdout_idx.tolist())
            q_all = [int(x) for x in ws.question.tolist()]
            kc_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            tset = [
                i
                for i in range(len(kc_all))
                if i not in hold_set and i > 0 and i % 5 == 0 and not is_a1(q_all, kc_all, i)
            ]
            tuning_idx[ws.user_id] = tset
            tset_set = set(tset)
            hold_aware = hold_set | tset_set
            hist_evidence[ws.user_id] = [
                (kc_all[i], y_all[i]) for i in range(len(kc_all)) if i not in hold_aware
            ]

        # leak-free prior: non-tuning evidence rows only
        prior_tuning = {}
        for uid, pairs in hist_evidence.items():
            for kc, y in pairs:
                prior_tuning.setdefault(kc, []).append(y)
        prior_tuning = {k: float(np.mean(v)) for k, v in prior_tuning.items() if v}

        # ---- pass 1: alpha selection on tuning rows (two priors: leakfree + leaky-repro)
        def select_alpha(prior):
            per_stu = {a: {} for a in ALPHAS}
            for ws in samples:
                q_all = [int(x) for x in ws.question.tolist()]
                kc_all = [int(x) for x in ws.sequence.tolist()]
                y_all = [int(x) for x in ws.response.tolist()]
                hold_set = set(ws.holdout_idx.tolist())
                hist = deque()
                rows_a = {a: [] for a in ALPHAS}
                tuning_here = set(tuning_idx[ws.user_id])  # hoisted (review P3-1)
                for i in range(len(kc_all)):
                    if i not in hold_set and i in tuning_here:
                        ab = ability_before(hist, q_all[i])
                        if ab is not None:
                            p1 = prior.get(kc_all[i], 0.5)
                            for a in ALPHAS:
                                rows_a[a].append({"y": y_all[i], "pred": a * p1 + (1 - a) * ab})
                    if i not in hold_set:
                        hist.append((q_all[i], y_all[i]))
                for a in ALPHAS:
                    auc = stratified_auc_rows(rows_a[a], "pred")
                    if auc is not None:
                        per_stu[a][ws.user_id] = auc
            means = {a: float(np.mean(list(v.values()))) if v else -1.0 for a, v in per_stu.items()}
            alpha_star = max(ALPHAS, key=lambda a: means[a])
            return alpha_star, {str(a): round(m, 4) for a, m in means.items()}

        alpha_star, means_leakfree = select_alpha(prior_tuning)
        alpha_star_leaky, means_leaky = select_alpha(prior_leaky)

        # ---- pass 2: holdout eval, alpha frozen; prior_final = refit on all evidence
        prior_final = fit_global_kc_prior(build_rows(samples, None))  # == prior_leaky here
        res = {
            "alpha_star_leakfree": alpha_star,
            "tuning_means_leakfree": means_leakfree,
            "alpha_star_leaky_repro": alpha_star_leaky,
            "tuning_means_leaky": means_leaky,
            "selfcheck_v2_alpha": alpha_star_leaky == V2_ALPHA_REF[ds],
        }
        for tag, alpha, prior_ev in (
            ("eval_final", alpha_star, prior_final),
            ("eval_sens_prior_tuning", alpha_star, prior_tuning),
            ("eval_alpha0", 0.0, prior_final),
            ("eval_alpha1", 1.0, prior_final),
        ):
            b1, b1ab = {}, {}
            for ws in samples:
                hold = sorted(set(ws.holdout_idx.tolist()))
                hold_set = set(hold)
                q_all = [int(x) for x in ws.question.tolist()]
                kc_all = [int(x) for x in ws.sequence.tolist()]
                y_all = [int(x) for x in ws.response.tolist()]
                hist = deque()
                rows1, rows2 = [], []
                for i in range(len(kc_all)):
                    if i in hold_set and not is_a1(q_all, kc_all, i):
                        p1 = prior_ev.get(kc_all[i], 0.5)
                        ab = ability_before(hist, q_all[i])
                        # convention (review P1 fix): pred = alpha*p1 + (1-alpha)*ab
                        # -> alpha=1 is PURE PRIOR (fallback p1), alpha=0 is PURE
                        #    ABILITY (fallback p1 when no window). Endpoint arms
                        #    honour it; the old inversion is swapped.
                        if ab is None:
                            pred = p1
                        elif tag == "eval_alpha1":
                            pred = p1
                        elif tag == "eval_alpha0":
                            pred = ab
                        else:
                            pred = alpha * p1 + (1 - alpha) * ab
                        rows1.append({"y": y_all[i], "pred": p1})
                        rows2.append({"y": y_all[i], "pred": pred})
                    hist.append((q_all[i], y_all[i]))
                a1 = stratified_auc_rows(rows1, "pred")
                a2 = stratified_auc_rows(rows2, "pred")
                if a1 is not None:
                    b1[ws.user_id] = a1
                if a2 is not None:
                    b1ab[ws.user_id] = a2
            common = [u for u in b1 if u in b1ab]
            diffs = np.array([b1ab[u] - b1[u] for u in common])
            res[tag] = {
                "B1": round(float(np.mean([b1[u] for u in common])), 4) if common else None,
                "B1ab": round(float(np.mean([b1ab[u] for u in common])), 4) if common else None,
                "n": len(common),
                "delta": round(float(np.mean(diffs)), 4) if len(diffs) else None,
            }
        # review P3-4: eval_final.B1 must reproduce v2's B1_A1excluded (same
        # prior path, alpha only affects rows2) — hard anchor, not just alpha.
        res["selfcheck_v2_B1"] = (
            res["eval_final"]["B1"] is not None
            and abs(res["eval_final"]["B1"] - V2_B1_REF[ds]) <= 0.002
        )
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a17_ability_decompose_v3.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("ABILITY-DECOMPOSE-V3-DONE")


if __name__ == "__main__":
    main()
