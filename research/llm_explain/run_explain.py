"""CLI orchestration for the LLM explanation experiment.

Pipeline per student: signals -> evidence pack -> LLM -> grounding verify ->
holdout scoring. Conditions: full | behavior-only (readiness column removed)
| permuted (readiness shuffled, probe).

Usage (from repo root, pixi env):
  python research/llm_explain/run_explain.py --run-dir runs/normal/DKT_... \
      --provider glm --n 30 --split dev --condition full --out research/llm_explain/results/dev_glm_full
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence import EvidencePack, build_pack  # noqa: E402
from faithfulness import permute_readiness, verify_report  # noqa: E402
from llm_client import call_llm  # noqa: E402
from prompts import build_messages  # noqa: E402
from restore import RestoredModel, WindowSample, load_user_samples, restore  # noqa: E402
from signals import holdout_actuals  # noqa: E402


def stratified_sample(samples: list[WindowSample], n: int, seed: int = 42) -> list[WindowSample]:
    """Stratify by evidence length tercile x evidence accuracy median, sample n."""
    rng = np.random.default_rng(seed)
    lens = np.array([len(s.evidence_idx) for s in samples])
    accs = np.array([s.response[s.evidence_idx].mean() for s in samples])
    l_bins = np.digitize(lens, np.quantile(lens, [1 / 3, 2 / 3]))
    a_bins = (accs > np.median(accs)).astype(int)
    strata: dict[tuple[int, int], list[int]] = {}
    for i in range(len(samples)):
        strata.setdefault((int(l_bins[i]), int(a_bins[i])), []).append(i)

    picked: list[int] = []
    quota = {k: max(1, round(n * len(v) / len(samples))) for k, v in strata.items()}
    for k, idxs in strata.items():
        picked.extend(rng.choice(idxs, size=min(quota[k], len(idxs)), replace=False).tolist())
    rng.shuffle(picked)
    return [samples[i] for i in picked[:n]]


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman with proper midranks (ties averaged) via scipy; NaN-dropped."""
    from scipy.stats import spearmanr

    pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y]  # drop NaN
    if len(pairs) < 3:
        return float("nan")
    r = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    return float(r)


def score_student(parsed: dict, pack: EvidencePack, actuals: dict[int, dict[str, float]]) -> dict:
    """Holdout scoring: LLM predicted per-KC accuracy vs actual."""
    preds = {int(p["kc"]): float(p["pred_acc"]) for p in parsed.get("holdout_predictions", [])}
    kcs = [k for k in pack.holdout_target_kcs if k in preds and k in actuals]
    llm = [preds[k] for k in kcs]
    act = [actuals[k]["acc"] for k in kcs]
    model = [actuals[k]["pred_mean"] for k in kcs]
    rows = [pack.kc_row(k) for k in kcs]
    copy_baseline = [r.readiness for r in rows if r is not None]
    return {
        "n_target_kcs": len(kcs),
        "llm_holdout_rho": spearman(llm, act),
        "model_holdout_rho": spearman(model, act),
        "readiness_holdout_rho": spearman(copy_baseline, act) if copy_baseline else float("nan"),
        "llm_mae": float(np.mean(np.abs(np.array(llm) - np.array(act)))) if kcs else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--provider", default="glm", choices=["glm", "kimi"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--condition", default="full", choices=["full", "behavior-only", "permuted"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="skip users already in records.jsonl")
    args = ap.parse_args()

    rm: RestoredModel = restore(args.run_dir, device=args.device)
    fold = int(rm.rc.data.fold) if int(rm.rc.data.fold) >= 0 else 0
    samples = load_user_samples(rm.data_src, fold=fold)
    # Main-analysis filter: at least 10 distinct KCs learned in evidence
    samples = [
        s
        for s in samples
        if len({int(c) for c in s.sequence[s.evidence_idx]}) >= 10
    ]
    n_dev = min(30, len(samples) // 5)
    dev = stratified_sample(samples, n_dev, seed=7)
    holdout_pool = [s for s in samples if s.user_id not in {d.user_id for d in dev}]
    pool = dev if args.split == "dev" else stratified_sample(holdout_pool, args.n, seed=42)
    if args.split == "dev":
        pool = dev[: args.n]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    (out / "run_meta.json").write_text(
        json.dumps(
            {
                "run_dir": args.run_dir,
                "model": rm.model_name,
                "provider": args.provider,
                "condition": args.condition,
                "split": args.split,
                "seed": args.seed,
                "fold": fold,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    records_path = out / "records.jsonl"
    records: list[dict] = []
    if records_path.exists() and args.resume:
        records = [json.loads(l) for l in open(records_path)]
        done = {r["user_id"] for r in records}
        pool = [s for s in pool if s.user_id not in done]
        print(f"resume: {len(done)} done, {len(pool)} to go", flush=True)

    with open(records_path, "a") as rec_f:
        for i, ws in enumerate(pool):
            try:
                rec = _process_student(rm, ws, args, rng, out)
            except Exception as e:  # noqa: BLE001 - one bad report must not kill the run
                rec = {"user_id": ws.user_id, "parse_ok": False, "error": repr(e)[:200]}
            records.append(rec)
            rec_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rec_f.flush()
            rho = rec.get("llm_holdout_rho")
            rho_s = f"{rho:.2f}" if isinstance(rho, float) and rho == rho else "nan"
            print(f"[{i+1}/{len(pool)}] U{ws.user_id} rho={rho_s} viol={rec.get('violation_rate', 0):.2%}", flush=True)

    rhos = [r["llm_holdout_rho"] for r in records if r.get("parse_ok") and not math.isnan(r.get("llm_holdout_rho", float("nan")))]
    summary = {
        "model": rm.model_name,
        "provider": args.provider,
        "condition": args.condition,
        "split": args.split,
        "n": len(records),
        "parse_ok": sum(r.get("parse_ok", False) for r in records),
        "mean_rho": float(np.mean(rhos)) if rhos else None,
        "mean_violation_rate": float(np.mean([r["violation_rate"] for r in records if r.get("parse_ok")])),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


def _process_student(rm, ws, args, rng, out: Path) -> dict:
    """Build pack, call LLM, verify, score -- one student, exception-contained."""
    pack = build_pack(rm, ws)
    if args.condition == "behavior-only":
        for r in pack.kcs:
            r.readiness = float("nan")
    if args.condition == "permuted":
        pack = permute_readiness(pack, rng)
    md = pack.to_markdown()
    (out / f"pack_U{ws.user_id}.md").write_text(md)

    call = call_llm(
        args.provider,
        build_messages(md, has_readiness=args.condition != "behavior-only"),
        purpose="diagnosis",
        user_id=ws.user_id,
        condition=args.condition,
    )
    if not call.parse_ok:
        return {"user_id": ws.user_id, "parse_ok": False, "parse_error": True}

    ver = verify_report(call.parsed, pack)
    actuals = holdout_actuals(rm, ws)
    scores = score_student(call.parsed, pack, actuals)
    (out / f"report_U{ws.user_id}.json").write_text(
        json.dumps(call.parsed, ensure_ascii=False, indent=2)
    )
    return {
        "user_id": ws.user_id,
        "parse_ok": True,
        "violation_rate": ver.violation_rate,
        "violations": ver.violations,
        "holdout_coverage": ver.holdout_coverage,
        "weakest_valid": ver.weakest_valid,
        **scores,
    }


if __name__ == "__main__":
    main()
