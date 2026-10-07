"""Batch 42: ability-decompose v3 — leakage-free alpha tuning (Codex Run A P1-7 / erratum #19).

v3.1 (Codex batch42 review, 2026-10-07 FAIL -> fix): the first v3 run closed
only the index-level tuning leak. Two residual paths found and closed here:
 (P1-1) sibling expand rows — one exploded multi-KC answer is several
   consecutive rows sharing the same label; excluding only the tuning row's
   index left its siblings in prior_tuning. Fix: exclude the WHOLE question
   run (maximal consecutive equal-q block) of every tuning row, and of every
   holdout row in the clean prior.
 (P1-2) prior_final inherited build_rows' holdout-biclass MEMBERSHIP filter
   (future holdout labels decide whose evidence builds the table). Fix:
   prior_clean = all students' evidence, holdout-run-complete exclusion, no
   membership filter. The old construction is kept ONLY as the v2-repro arm.

v2 leak being repaired: the global KC prior was fitted on ALL evidence rows,
then every-5th evidence rows were reused as the tuning pseudo-holdout — each
tuning row's own label re-entered its prediction through the prior.
(ability_before itself is clean: hist is appended after prediction; note its
window is the last <=20 response ROWS excluding the current question, so a
multi-KC answer can be counted more than once.)

v3.1 design:
 (1) tuning rows = every-5th non-holdout non-A1 evidence row (same rule as v2);
     prior_tuning fitted from the OTHER evidence rows only (all students,
     interaction-complete: tuning question runs excluded) — no tuning label
     can enter prior_tuning, directly or via siblings;
 (1b) prior_leaky_isolated = same membership rule as prior_tuning but WITH
     tuning rows (index-level) included — isolates the tuning-label leak from
     the membership confound (review P2-4);
 (2) alpha* selected on tuning rows with prior_tuning + ability_before;
 (3) eval on real holdout (A1-excluded) with alpha frozen, prior_clean
     (all students, holdout-run-complete, no membership filter); sensitivity
     arm with prior_tuning; v2-repro arm with the OLD prior;
 (4) v2-repro anchors: leaky-prior selection must reproduce v2 alpha*=1.0 in
     all three domains; eval_v2repro.B1 must equal v2's B1_A1excluded.
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


def run_ids(q_all):
    """Maximal consecutive runs of identical question ids.

    One exploded multi-KC answer = one run; all rows in a run share the same
    answer label, so prior-construction exclusions must operate on runs
    (review P1-1: index-level exclusion leaks the label via sibling rows).
    """
    rid = [0] * len(q_all)
    r = 0
    for i in range(1, len(q_all)):
        if q_all[i] != q_all[i - 1]:
            r += 1
        rid[i] = r
    return rid


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
        # prior_tuning evidence: non-holdout rows EXCLUDING whole question
        # runs that contain a tuning row (P1-1: siblings carry the label)
        hist_evidence = {}
        # prior_clean evidence: all students, holdout-run-complete exclusion,
        # no membership filter (P1-2)
        clean_evidence = {}
        for ws in samples:
            hold_set = set(ws.holdout_idx.tolist())
            q_all = [int(x) for x in ws.question.tolist()]
            kc_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            rid = run_ids(q_all)
            tset = [
                i
                for i in range(len(kc_all))
                if i not in hold_set and i > 0 and i % 5 == 0 and not is_a1(q_all, kc_all, i)
            ]
            tuning_idx[ws.user_id] = tset
            hold_runs = {rid[i] for i in hold_set}
            tuning_runs = {rid[i] for i in tset}
            hist_evidence[ws.user_id] = [
                (kc_all[i], y_all[i]) for i in range(len(kc_all))
                if rid[i] not in hold_runs and rid[i] not in tuning_runs
            ]
            clean_evidence[ws.user_id] = [
                (kc_all[i], y_all[i]) for i in range(len(kc_all)) if rid[i] not in hold_runs
            ]

        # leak-free prior: non-tuning evidence rows only (interaction-complete)
        prior_tuning = {}
        for uid, pairs in hist_evidence.items():
            for kc, y in pairs:
                prior_tuning.setdefault(kc, []).append(y)
        prior_tuning = {k: float(np.mean(v)) for k, v in prior_tuning.items() if v}

        # clean prior for eval: all students, holdout-run-complete, no
        # membership filter (P1-2)
        prior_clean = {}
        for uid, pairs in clean_evidence.items():
            for kc, y in pairs:
                prior_clean.setdefault(kc, []).append(y)
        prior_clean = {k: float(np.mean(v)) for k, v in prior_clean.items() if v}

        # isolated tuning-label leak (P2-4): same all-students membership and
        # holdout-run exclusion as prior_tuning, but tuning rows INCLUDED
        # (index-level, exactly the v2 state minus the membership filter)
        prior_leaky_iso = {}
        for ws in samples:
            hold_set = set(ws.holdout_idx.tolist())
            q_all2 = [int(x) for x in ws.question.tolist()]
            kc_all2 = [int(x) for x in ws.sequence.tolist()]
            y_all2 = [int(x) for x in ws.response.tolist()]
            rid2 = run_ids(q_all2)
            hold_runs2 = {rid2[i] for i in hold_set}
            for i in range(len(kc_all2)):
                if rid2[i] not in hold_runs2:
                    prior_leaky_iso.setdefault(kc_all2[i], []).append(y_all2[i])
        prior_leaky_iso = {k: float(np.mean(v)) for k, v in prior_leaky_iso.items() if v}

        # ---- pass 1: alpha selection on tuning rows (leakfree + isolated-leak + leaky-repro)
        def select_alpha(prior):
            per_stu = {a: {} for a in ALPHAS}
            n_rows_scored = 0
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
                            n_rows_scored += 1
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
            coverage = {
                "n_tuning_rows_scored": n_rows_scored,
                "n_students_valid_at_star": len(per_stu[alpha_star]),
            }
            return alpha_star, {str(a): round(m, 4) for a, m in means.items()}, coverage

        alpha_star, means_leakfree, cov_leakfree = select_alpha(prior_tuning)
        alpha_star_iso, means_leaky_iso, _cov_iso = select_alpha(prior_leaky_iso)
        alpha_star_leaky, means_leaky, _cov_leaky = select_alpha(prior_leaky)

        # ---- pass 2: holdout eval, alpha frozen
        prior_old = prior_leaky  # v2 construction, kept ONLY as repro anchor (P1-2)
        res = {
            "alpha_star_leakfree": alpha_star,
            "tuning_means_leakfree": means_leakfree,
            "tuning_coverage_leakfree": cov_leakfree,
            "alpha_star_leaky_isolated": alpha_star_iso,
            "tuning_means_leaky_isolated": means_leaky_iso,
            "alpha_star_leaky_repro": alpha_star_leaky,
            "tuning_means_leaky": means_leaky,
            "selfcheck_v2_alpha": alpha_star_leaky == V2_ALPHA_REF[ds],
            "prior_drift_clean_vs_old_max_abs": round(
                float(max(abs(prior_clean.get(k, 0.5) - prior_old.get(k, 0.5)) for k in set(prior_clean) | set(prior_old))),
                4,
            ),
        }
        for tag, alpha, prior_ev in (
            ("eval_final", alpha_star, prior_clean),
            ("eval_sens_prior_tuning", alpha_star, prior_tuning),
            ("eval_v2repro", alpha_star, prior_old),
            ("eval_alpha0", 0.0, prior_clean),
            ("eval_alpha1", 1.0, prior_clean),
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
        # review P3-4: eval_v2repro.B1 must reproduce v2's B1_A1excluded (old
        # prior path; alpha only affects the ab arm) — hard anchor, not just alpha.
        res["selfcheck_v2_B1"] = (
            res["eval_v2repro"]["B1"] is not None
            and abs(res["eval_v2repro"]["B1"] - V2_B1_REF[ds]) <= 0.002
        )
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a17_ability_decompose_v3.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("ABILITY-DECOMPOSE-V3-DONE")


if __name__ == "__main__":
    main()
