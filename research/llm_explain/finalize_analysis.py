"""Post-hoc unified analysis on archived records (freeze addendum compliant).

Recomputes all programmatic baselines offline (zero LLM cost), applies the
k>=4 inclusion rule and rho clipping, runs sensitivity analyses, and emits
the confirmatory + exploratory summary tables.

Usage:
  python research/llm_explain/finalize_analysis.py <results_dir> [--rho-col llm_holdout_rho]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from restore import load_user_samples, restore  # noqa: E402
from evidence import build_pack  # noqa: E402
from run_explain import spearman  # noqa: E402
from signals import holdout_actuals  # noqa: E402

CLIP = 0.99


def _clip(r: float) -> float:
    if r != r:
        return float("nan")
    return max(-CLIP, min(CLIP, r))


def fisher_z(rho: float) -> float:
    return math.atanh(_clip(rho))


def per_student_baselines(rm, ws) -> dict:
    """Programmatic baselines recomputed from the frozen model + data (no LLM)."""
    pack = build_pack(rm, ws)
    actuals = holdout_actuals(rm, ws)
    kcs = [k for k in pack.holdout_target_kcs if k in actuals]
    act = [actuals[k]["acc"] for k in kcs]
    rows = [pack.kc_row(k) for k in kcs]
    return {
        "n_target_kcs": len(kcs),
        "acc_copy_rho": spearman([r.acc for r in rows], act),
        "readiness_rho": spearman([r.readiness for r in rows], act),
        "predmean_rho": spearman([r.pred_mean for r in rows], act),
        "model_online_rho": spearman([actuals[k]["pred_mean"] for k in kcs], act),
    }


def weighted(records: list[dict], col: str) -> dict:
    pts = [
        (r[col], r["n_target_kcs"])
        for r in records
        if r.get("parse_ok") and r.get(col) == r.get(col)  # not NaN
    ]
    if not pts:
        return {"rho": None, "n": 0}
    zs = np.array([fisher_z(p) for p, _ in pts])
    ws = np.array([max(k - 3, 0) for _, k in pts], dtype=float)
    if ws.sum() == 0:
        ws = np.ones_like(ws)
    z = float((zs * ws).sum() / ws.sum())
    se = float(1 / math.sqrt(ws.sum()))
    lo_z = z - 1.645 * se  # one-sided 95% lower bound
    return {"rho": math.tanh(z), "one_sided95_lower": math.tanh(lo_z), "n": len(pts), "eff_k": float(ws.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--run-dir", required=True, help="model run dir for baselines")
    ap.add_argument("--min-k", type=int, default=4)
    args = ap.parse_args()

    rm = restore(args.run_dir, device="cuda")
    samples = {s.user_id: s for s in load_user_samples(rm.data_src, fold=0)}

    for d in args.dirs:
        rec_file = Path(d) / "records.jsonl"
        records = [json.loads(l) for l in open(rec_file)]
        base = {}
        for r in records:
            ws = samples.get(r["user_id"])
            if ws is not None:
                base[r["user_id"]] = per_student_baselines(rm, ws)
        for r in records:
            r["_base"] = base.get(r["user_id"], {})

        incl = [r for r in records if r.get("parse_ok") and r.get("_base", {}).get("n_target_kcs", 0) >= args.min_k]
        excl = len([r for r in records if r.get("parse_ok")]) - len(incl)

        llm = weighted(incl, "llm_holdout_rho")
        acc_copy = weighted(
            [{**r, "llm_holdout_rho": r["_base"]["acc_copy_rho"]} for r in incl if r["_base"].get("acc_copy_rho") == r["_base"].get("acc_copy_rho")],
            "llm_holdout_rho",
        )
        readiness = weighted(
            [{**r, "llm_holdout_rho": r["_base"]["readiness_rho"]} for r in incl if r["_base"].get("readiness_rho") == r["_base"].get("readiness_rho")],
            "llm_holdout_rho",
        )
        model_online = weighted(
            [{**r, "llm_holdout_rho": r["model_holdout_rho"]} for r in incl if r.get("model_holdout_rho") == r.get("model_holdout_rho")],
            "llm_holdout_rho",
        )

        rhos = [r["llm_holdout_rho"] for r in incl if r["llm_holdout_rho"] == r["llm_holdout_rho"]]
        tau_w = llm["rho"]
        print(
            json.dumps(
                {
                    "dir": d,
                    "n_total": len(records),
                    "parse_ok": sum(r.get("parse_ok", False) for r in records),
                    "n_included(k>=4)": len(incl),
                    "n_excluded_low_k": excl,
                    "LLM_weighted_rho": round(tau_w, 4) if tau_w == tau_w else None,
                    "LLM_one_sided95_lower": round(llm["one_sided95_lower"], 4) if llm["rho"] is not None else None,
                    "baseline_acc_copy": round(acc_copy["rho"], 4) if acc_copy["rho"] is not None else None,
                    "baseline_readiness": round(readiness["rho"], 4) if readiness["rho"] is not None else None,
                    "baseline_model_online": round(model_online["rho"], 4) if model_online["rho"] is not None else None,
                    "primary_endpoint_pass": (
                        llm["rho"] is not None
                        and acc_copy["rho"] is not None
                        and llm["one_sided95_lower"] > acc_copy["rho"] + 0.05
                    ),
                    "sensitivity_unweighted_mean_rho": round(float(np.mean(rhos)), 4) if rhos else None,
                    "sensitivity_median_rho": round(float(np.median(rhos)), 4) if rhos else None,
                    "sensitivity_k_ge5_rho": weighted([r for r in incl if r["_base"]["n_target_kcs"] >= 5], "llm_holdout_rho")["rho"],
                    "mean_violation_rate": round(float(np.mean([r["violation_rate"] for r in incl])), 4),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
