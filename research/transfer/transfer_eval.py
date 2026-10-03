"""Zero-training transfer evaluation: frozen source model on target dataset.

Per target student: map the target KC sequence into the source KC space via an
alignment (hard top-1, no_match steps dropped), run the frozen source model,
score holdout responses with within-student AUC. Includes the
occupancy-matched random null with Monte-Carlo permutation (B configurable).

Reviews absorbed (2026-10-03, ZCode + Kimi cross-review):
- evaluate_alignment extracted; permutation at the ALIGNMENT layer (occupancy
  fixed, source assignment shuffled) with distinct seeds.
- matches sorted by score before top-1 (in kc_align).
- target data source built via RunDataConfig (kc_tables._make_rc).
- tail-truncation to the source run's max_seq_len budget.
- empty-student / missing-cache guards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "llm_explain"))

import torch  # noqa: E402

from kc_align import load_alignment, random_occupancy_matched  # noqa: E402
from kc_tables import _make_rc  # noqa: E402


def stratified_auc_rows(rows: list[dict], pred_key: str) -> float | None:
    """Within-student AUC over one student's holdout rows (ties = 0.5)."""
    ys = np.array([r["y"] for r in rows])
    vs = np.array([r[pred_key] for r in rows], dtype=float)
    ok = vs == vs
    ys, vs = ys[ok], vs[ok]
    if (ys == 1).sum() == 0 or (ys == 0).sum() == 0:
        return None
    num = den = 0.0
    for vp in vs[ys == 1]:
        for vn in vs[ys == 0]:
            num += 1.0 if vp > vn else (0.5 if vp == vn else 0.0)
            den += 1
    return num / den if den else None


def _forward_one(rm, seq_m: list[int], resp_m: list[int]) -> np.ndarray:
    from signals import _forward_probs  # noqa: PLC0415

    s = torch.tensor([seq_m], dtype=torch.long, device=rm.device)
    r = torch.tensor([resp_m], dtype=torch.long, device=rm.device)
    # question=None: AKT Rasch branch skipped == zero modulation (padding row),
    # by design — target question ids are meaningless to the source model.
    return _forward_probs(rm, s, r, None)[0]


def evaluate_alignment(
    rm,
    samples,
    alignment: dict[int, list[dict]],
    max_seq_len: int = 200,
    min_kept: int = 6,
) -> dict:
    """Map every student via one alignment, forward, collect holdout rows."""
    students = []
    n_dropped = n_total = 0
    n_holdout_total = n_holdout_retry = 0
    for ws in samples:
        mapping = {}
        for kc in set(ws.sequence.tolist()):
            ms = alignment.get(int(kc), [])
            mapping[int(kc)] = ms[0]["source_kc"] if ms else None
        keep = [
            i
            for i, kc in enumerate(ws.sequence.tolist())
            if mapping[int(kc)] is not None
        ]
        if len(keep) > max_seq_len:
            keep = keep[-max_seq_len:]
        n_total += len(ws.sequence)
        n_dropped += len(ws.sequence) - len(keep)
        if len(keep) < min_kept:
            continue
        seq_m = [mapping[int(ws.sequence[i])] for i in keep]
        resp_m = [int(ws.response[i]) for i in keep]
        holdout_pos = set(ws.holdout_idx.tolist())
        probs = _forward_one(rm, seq_m, resp_m)
        rows = []
        ev_len = 0
        ev_q = set()
        n_hold = n_hold_retry = 0
        for j, i in enumerate(keep):
            if i in holdout_pos:
                rows.append({"y": int(ws.response[i]), "pred": float(probs[j])})
                n_hold += 1
                if int(ws.question[i]) in ev_q:
                    n_hold_retry += 1
            else:
                ev_len += 1
                ev_q.add(int(ws.question[i]))
        n_holdout_total += n_hold
        n_holdout_retry += n_hold_retry
        if ev_len >= 3 and 0 < sum(r["y"] for r in rows) < len(rows):
            students.append(rows)
    aucs = [a for a in (stratified_auc_rows(s, "pred") for s in students) if a is not None]
    return {
        "rows_by_student": students,
        "auc": float(np.mean(aucs)) if aucs else None,
        "n_students": len(aucs),
        "drop_rate": round(n_dropped / max(n_total, 1), 4),
        "holdout_retry_rate": round(n_holdout_retry / max(n_holdout_total, 1), 4),
    }


def permutation_test(rm, samples, alignment: dict, B: int = 999, seed: int = 42) -> dict:
    """Occupancy-matched random null: permute source assignments (top-1) across
    matched target KCs; recompute the SAME students' AUC per permutation."""
    obs = evaluate_alignment(rm, samples, alignment)
    if obs["auc"] is None:
        return {"error": "no valid students under the observed alignment"}
    nulls = []
    for b in range(B):
        null_al = random_occupancy_matched(alignment, seed=seed * 1_000_003 + b)
        r = evaluate_alignment(rm, samples, null_al)
        if r["auc"] is not None:
            nulls.append(r["auc"])
    ge = sum(a >= obs["auc"] for a in nulls)
    return {
        "obs_auc": round(obs["auc"], 4),
        "null_mean": round(float(np.mean(nulls)), 4) if nulls else None,
        "null_std": round(float(np.std(nulls)), 4) if nulls else None,
        "perm_p": round((1 + ge) / (1 + len(nulls)), 4),
        "B": len(nulls),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--target-dataset", required=True)
    ap.add_argument("--alignment", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--b", type=int, default=0, help="0 = observed AUC only")
    ap.add_argument("--max-users", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--identity", action="store_true",
                    help="ignore --alignment; map every target KC to itself "
                    "(same-dataset sanity check of the eval wiring)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from llm_explain_restore_shim import restore_any  # noqa: PLC0415
    from restore import load_user_samples  # noqa: PLC0415
    from utils.data_process import get_data_source  # noqa: PLC0415

    rm = restore_any(args.source_run, device=args.device)
    if args.identity:
        from kc_tables import build_kc_table  # noqa: PLC0415

        tgt_table = build_kc_table(args.target_dataset)
        alignment = {k: [{"source_kc": k, "score": 1.0, "relation": "identity"}]
                     for k in tgt_table}
    else:
        raw = load_alignment(args.alignment)
        if raw is None:
            raise SystemExit(f"alignment cache not found: {args.alignment}")
        alignment = {int(k): v for k, v in raw.items()}

    tgt_src = get_data_source(_make_rc(args.target_dataset))
    samples = load_user_samples(tgt_src, fold=args.fold)
    if args.max_users:
        samples = samples[: args.max_users]

    max_len = int(getattr(rm.rc.data, "max_seq_len", 200) or 200)
    obs = evaluate_alignment(rm, samples, alignment, max_seq_len=max_len)
    result = {
        "source_run": args.source_run,
        "target": args.target_dataset,
        "alignment": args.alignment,
        "n_students": obs["n_students"],
        "drop_rate": obs["drop_rate"],
        "holdout_retry_rate": obs["holdout_retry_rate"],
        "within_student_auc": round(obs["auc"], 4) if obs["auc"] is not None else None,
    }
    if args.b:
        result.update(permutation_test(rm, samples, alignment, B=args.b))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
