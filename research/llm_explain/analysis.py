"""Aggregate analysis: Fisher-z weighted rho, cluster bootstrap CI, paired tests.

Usage:
  python research/llm_explain/analysis.py results/dirA results/dirB [--compare]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_records(path: str) -> list[dict]:
    rec_file = Path(path) / "records.jsonl"
    if not rec_file.exists():
        # fall back to parsing run_explain stdout lines is NOT done: rerun instead
        raise FileNotFoundError(f"{rec_file} missing (records not persisted)")
    with open(rec_file) as f:
        return [json.loads(line) for line in f]


def fisher_z(rho: float) -> float:
    return math.atanh(max(-0.999, min(0.999, rho)))


def weighted_rho(records: list[dict]) -> dict:
    """Fisher-z average weighted by k-3 (k = number of target KCs)."""
    pairs = [
        (r["llm_holdout_rho"], r["n_target_kcs"])
        for r in records
        if r.get("parse_ok") and not math.isnan(r.get("llm_holdout_rho", float("nan")))
    ]
    if not pairs:
        return {"weighted_rho": None, "n": 0}
    zs = np.array([fisher_z(p) for p, _ in pairs])
    ws = np.array([max(k - 3, 1) for _, k in pairs], dtype=float)
    z_bar = float((zs * ws).sum() / ws.sum())
    return {"weighted_rho": math.tanh(z_bar), "fisher_z": z_bar, "n": len(pairs)}


def bootstrap_ci(records: list[dict], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Student-level cluster bootstrap CI of the weighted Fisher-z mean."""
    valid = [
        r
        for r in records
        if r.get("parse_ok") and not math.isnan(r.get("llm_holdout_rho", float("nan")))
    ]
    if len(valid) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    zs = []
    for _ in range(n_boot):
        sample = [valid[i] for i in rng.integers(0, len(valid), len(valid))]
        w = weighted_rho(sample)
        if w["weighted_rho"] is not None:
            zs.append(w["fisher_z"])
    lo, hi = np.percentile(zs, [2.5, 97.5])
    return (math.tanh(lo), math.tanh(hi))


def wilcoxon_paired(a: list[dict], b: list[dict]) -> dict:
    """Paired comparison of per-student Fisher-z between two conditions."""
    za = {r["user_id"]: fisher_z(r["llm_holdout_rho"]) for r in a if r.get("parse_ok") and not math.isnan(r.get("llm_holdout_rho", float("nan")))}
    zb = {r["user_id"]: fisher_z(r["llm_holdout_rho"]) for r in b if r.get("parse_ok") and not math.isnan(r.get("llm_holdout_rho", float("nan")))}
    common = sorted(set(za) & set(zb))
    if len(common) < 6:
        return {"n_pairs": len(common), "test": None}
    diffs = np.array([za[u] - zb[u] for u in common])
    # sign test p (exact binomial) -- distribution-free, tiny n friendly
    k = int((diffs > 0).sum())
    from math import comb

    n = len(diffs)
    p_two = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n * 2
    return {
        "n_pairs": n,
        "median_delta_z": float(np.median(diffs)),
        "pos_pairs": k,
        "sign_test_p": min(p_two, 1.0),
    }


def summarize(path: str) -> dict:
    records = load_records(path)
    ok = [r for r in records if r.get("parse_ok")]
    w = weighted_rho(records)
    lo, hi = bootstrap_ci(records)
    viol = [r["violation_rate"] for r in ok] or [0]
    cover = [r["holdout_coverage"] for r in ok] or [1]
    return {
        "dir": str(path),
        "n": len(records),
        "parse_ok": len(ok),
        "parse_rate": len(ok) / len(records) if records else 0,
        "weighted_rho": round(w["weighted_rho"], 4) if w["weighted_rho"] is not None else None,
        "rho_ci95": [round(lo, 4), round(hi, 4)],
        "mean_violation_rate": round(float(np.mean(viol)), 4),
        "mean_holdout_coverage": round(float(np.mean(cover)), 4),
        "model_rho_baseline": round(
            float(np.nanmean([r.get("model_holdout_rho", float("nan")) for r in ok])), 4
        ),
        "readiness_rho_baseline": round(
            float(np.nanmean([r.get("readiness_holdout_rho", float("nan")) for r in ok])), 4
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--compare", action="store_true", help="paired sign test dir1 vs dir2")
    args = ap.parse_args()

    results = [summarize(d) for d in args.dirs]
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    if args.compare and len(args.dirs) >= 2:
        a, b = load_records(args.dirs[0]), load_records(args.dirs[1])
        print(json.dumps({"paired": wilcoxon_paired(a, b)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
