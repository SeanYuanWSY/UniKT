"""Batch24: why is the A09-aligned prior anti-predictive on A17/EdNet?

PS-0 (ps0_aligned_table.json) found covered_only AUC 0.451 (A17) / 0.373
(EdNet) / 0.518 (Junyi): where the LLM alignment maps a target KC, the A09
source prior often ranks target correctness WORSE than chance on 2/3
domains. T2 found the same direction reversal in deep transfer. This
diagnostic separates three mechanisms over covered target KCs:
  (a) sign inversion — aligned prior anti-correlates with target marginal;
  (b) compression/noise — no correlation, priors near-constant;
  (c) subset bias — covered KCs are themselves atypical.
Measures per domain: Spearman rho(aligned_prior, target_B1_marginal) over
covered KCs, coverage stats, mean/abs-diff of the two marginal vectors,
and a sign-agreement rate. Pure table diagnostic, minutes, no training.
Target marginal = the authoritative B1 table (first-300 fold-0 evidence).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = {
    "assistments17": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
    "junyi2015": "assistments09__to__junyi2015__llm__k3-high__c25k3__rep0",
}


def rankdata_avg(x):
    """Tie-aware average ranks (scipy rankdata 'average' equivalent)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx = rankdata_avg(x)
    ry = rankdata_avg(y)
    if rx.std() == 0 or ry.std() == 0:
        return None  # constant vector: no meaningful rank corr
    return float(np.corrcoef(rx, ry)[0, 1])


def main(n_users=300):
    src09 = get_data_source(_make_rc("assistments09"))
    s09_prior = fit_global_kc_prior(build_rows(load_user_samples(src09, fold=0), None))

    out = {}
    for ds, align_name in ALIGN.items():
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        tgt_prior = fit_global_kc_prior(build_rows(samples[:n_users], None))  # B1 table (first-300 evidence)
        # tail-100 marginals: matches the ps0 eval cohort, for exact decomposition of its covered_only
        tgt_prior_tail = fit_global_kc_prior(build_rows(samples[-100:], None))
        alignment = {int(k): v for k, v in load_alignment(align_name).items()}

        covered = []  # (target_kc, aligned_a09_prior, target_marginal, target_tail_marginal)
        for kc, ms in alignment.items():
            if ms and kc in tgt_prior:
                ap = s09_prior.get(ms[0]["source_kc"])
                if ap is not None:
                    covered.append((kc, ap, tgt_prior[kc], tgt_prior_tail.get(kc)))

        if len(covered) < 5:
            out[ds] = {"n_covered_kcs": len(covered), "note": "too few covered KCs for diagnosis"}
            print(ds, json.dumps(out[ds]), flush=True)
            continue

        a = np.array([c[1] for c in covered])
        t = np.array([c[2] for c in covered])
        rho = spearman(a.tolist(), t.tolist())
        # exact ps0-cohort counterpart: only KCs present in the tail-100 marginal
        ctail = [(c[1], c[3]) for c in covered if c[3] is not None]
        rho_tail = spearman([c[0] for c in ctail], [c[1] for c in ctail]) if len(ctail) >= 5 else None
        # per-KC difficulty direction agreement: prior>0.5 iff marginal>0.5
        sign_agree = float(np.mean((a > 0.5) == (t > 0.5)))
        res = {
            "n_alignment_keys": len(alignment),
            "n_covered_kcs": len(covered),
            "n_tgt_kcs_total": len(tgt_prior),
            "n_tail_covered": len(ctail),
            "spearman_rho": round(rho, 4) if rho is not None else None,
            "spearman_rho_tail100": round(rho_tail, 4) if rho_tail is not None else None,
            "sign_agree": round(sign_agree, 4),
            "mean_aligned_prior": round(float(a.mean()), 4),
            "mean_tgt_marginal": round(float(t.mean()), 4),
            "std_aligned_prior": round(float(a.std()), 4),
            "std_tgt_marginal": round(float(t.std()), 4),
            "mean_abs_diff": round(float(np.abs(a - t).mean()), 4),
            "tgt_marginal_allmean": round(float(np.mean(list(tgt_prior.values()))), 4),
        }
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/align_anti_diag.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("ALIGN-DIAG-DONE")


if __name__ == "__main__":
    main()
