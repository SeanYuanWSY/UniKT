"""Item-level (per-response) scoring of archived LLM predictions.

Re-anchors every archived KC-level LLM prediction to the individual holdout
responses it covers, then computes global AUC (tie-aware), per-student mean
AUC, and Brier for four prediction sources, plus the paired full-vs-behavior
Delta-AUC with student-level cluster bootstrap.

All inputs are already archived; no LLM calls. Run per model:

  python research/llm_explain/item_level_analysis.py \
      --full-dir  results/holdout_akt_glm_full \
      --behavior-dir results/holdout_akt_glm_behavior \
      --run-dir runs/normal/AKT_... [--paired]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from restore import load_user_samples, restore  # noqa: E402
from evidence import build_pack  # noqa: E402
from signals import holdout_actuals  # noqa: E402


def auc_tieaware(y: np.ndarray, p: np.ndarray) -> float:
    """AUC via tie-aware rank statistic (Mann-Whitney U with midranks)."""
    from scipy.stats import rankdata

    pos, neg = y == 1, y == 0
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(p)  # midranks: ties share the average rank
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified_auc(students: list) -> float:
    """Within-student (student-stratified) AUC: concordance pooled over
    within-student pairs only, weighted by each student's n+ * n- pairs.

    ``students``: list per student of (y, v) tuples.
    """
    num, den = 0.0, 0.0
    for s in students:
        ys = np.array([e[0] for e in s])
        vs = np.array([e[1] for e in s])
        for v_pos in vs[ys == 1]:
            for v_neg in vs[ys == 0]:
                if v_pos > v_neg:
                    num += 1
                elif v_pos == v_neg:
                    num += 0.5
                den += 1
    return num / den if den else float("nan")


def collect_responses(rm, samples_by_uid: dict) -> dict:
    """Per-student holdout responses with all programmatic predictors."""
    out = {}
    for uid, ws in samples_by_uid.items():
        pack = build_pack(rm, ws)
        actuals = holdout_actuals(rm, ws)
        rows = []
        for t in ws.holdout_idx:
            kc = int(ws.sequence[t])
            row = pack.kc_row(kc)
            if row is None:
                continue
            rows.append(
                {
                    "kc": kc,
                    "y": int(ws.response[t]),
                    "acc_copy": row.acc,
                    "readiness": row.readiness,
                }
            )
        # model-online per response: recompute with full-sequence forward
        seq = ws.sequence
        resp_full = ws.response
        import torch  # noqa: PLC0415

        from signals import _forward_probs  # noqa: PLC0415

        s = torch.tensor(seq, dtype=torch.long, device=rm.device).unsqueeze(0)
        r = torch.tensor(resp_full, dtype=torch.long, device=rm.device).unsqueeze(0)
        q = torch.tensor(ws.question, dtype=torch.long, device=rm.device).unsqueeze(0)
        probs = _forward_probs(rm, s, r, q)[0]
        for i, t in enumerate(ws.holdout_idx):
            if i < len(rows):
                rows[i]["model_online"] = float(probs[t])
        out[uid] = rows
    return out


def load_llm_preds(results_dir: str) -> dict[int, dict[int, float]]:
    """user_id -> {kc: pred_acc} from archived report files."""
    preds = {}
    for f in Path(results_dir).glob("report_U*.json"):
        uid = int(f.stem.split("_U")[1])
        d = json.loads(f.read_text())
        preds[uid] = {int(p["kc"]): float(p["pred_acc"]) for p in d.get("holdout_predictions", [])}
    return preds


def score_condition(
    responses: dict, llm_preds: dict[int, dict[int, float]]
) -> dict:
    """Global AUC / per-student mean AUC / Brier for each source."""
    per_source = {"llm": [], "acc_copy": [], "readiness": [], "model_online": []}
    per_student_auc = {"llm": [], "acc_copy": [], "readiness": [], "model_online": []}
    per_student_rows: dict[int, dict[str, list]] = {}
    for uid, rows in responses.items():
        lp = llm_preds.get(uid, {})
        enriched = []
        for r in rows:
            e = dict(r)
            e["llm"] = lp.get(r["kc"])
            if e["llm"] is None:
                continue  # LLM missed this KC: excluded from LLM source only
            enriched.append(e)
        if not enriched:
            continue
        per_student_rows[uid] = enriched
        for src in per_source:
            vals = np.array([e[src] for e in enriched], dtype=float)
            ys = np.array([e["y"] for e in enriched], dtype=int)
            per_source[src].append((ys, vals))
            if 0 < ys.sum() < len(ys):
                per_student_auc[src].append(auc_tieaware(ys, vals))

    result = {}
    for src, pairs in per_source.items():
        ys = np.concatenate([p[0] for p in pairs])
        vs = np.concatenate([p[1] for p in pairs])
        result[src] = {
            "global_auc": round(auc_tieaware(ys, vs), 4),
            "within_student_auc": round(
                stratified_auc([[(float(y), float(v)) for y, v in zip(p[0], p[1])] for p in pairs]), 4
            ),
            "student_mean_auc": round(float(np.nanmean(per_student_auc[src])), 4),
            "n_student": len(per_student_auc[src]),
            "brier": round(float(np.mean((vs - ys) ** 2)), 4),
            "n_resp": int(len(ys)),
        }
    result["_per_student"] = {u: {s: v for s, v in []} for u in []}  # placeholder
    result["_rows_by_student"] = per_student_rows
    return result


def paired_delta_auc(
    responses: dict,
    preds_full: dict[int, dict[int, float]],
    preds_behavior: dict[int, dict[int, float]],
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """Delta within-student AUC (full - behavior) on identical response sets,
    cluster bootstrap by student.

    Primary scale (per statistical review): student-stratified within-student
    AUC A_w -- cross-student pairs are excluded (they dominate global AUC by
    ~100:1 and dilute the within-student KC discrimination under test).
    Both conditions' A_w are recomputed on each student resample.
    """
    students = []
    for uid, rows in responses.items():
        pf, pb = preds_full.get(uid, {}), preds_behavior.get(uid, {})
        enriched = []
        for r in rows:
            vf, vb = pf.get(r["kc"]), pb.get(r["kc"])
            if vf is None or vb is None:
                continue
            enriched.append((r["y"], vf, vb))
        ys = [e[0] for e in enriched]
        if 0 < sum(ys) < len(ys):
            students.append(enriched)

    def aw(cond_idx: int, subset=None) -> float:
        ss = students if subset is None else subset
        return stratified_auc(
            [[(e[0], e[cond_idx]) for e in s] for s in ss]
        )

    d_obs = aw(1) - aw(2)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(students), len(students))
        sub = [students[i] for i in idx]
        d = aw(1, sub) - aw(2, sub)
        if d == d:
            boots.append(d)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_one = float((boots <= 0).mean())  # one-sided p for delta>0
    return {
        "scale": "within_student_stratified_auc",
        "n_students": len(students),
        "n_resp": int(sum(len(s) for s in students)),
        "aw_full": round(aw(1), 4),
        "aw_behavior": round(aw(2), 4),
        "delta_aw": round(d_obs, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "one_sided_p": round(min(p_one, 1.0), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", help="single condition: score its LLM vs baselines")
    ap.add_argument("--full-dir")
    ap.add_argument("--behavior-dir")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    rm = restore(args.run_dir, device="cuda")
    fold = int(rm.rc.data.fold) if int(rm.rc.data.fold) >= 0 else 0
    all_samples = {s.user_id: s for s in load_user_samples(rm.data_src, fold=fold)}

    uids = None
    if args.results_dir:
        uids = {int(f.stem.split("_U")[1]) for f in Path(args.results_dir).glob("report_U*.json")}
    elif args.full_dir and args.behavior_dir:
        uids = {int(f.stem.split("_U")[1]) for f in Path(args.full_dir).glob("report_U*.json")}
    samples = {u: all_samples[u] for u in (uids or []) if u in all_samples}

    print("computing per-response predictors (frozen model forwards)...", flush=True)
    responses = collect_responses(rm, samples)

    out = {"model": rm.model_name}
    if args.results_dir:
        preds = load_llm_preds(args.results_dir)
        sc = score_condition(responses, preds)
        sc.pop("_per_student")
        sc.pop("_rows_by_student")
        out["sources"] = sc
    if args.full_dir and args.behavior_dir:
        out["paired_full_vs_behavior"] = paired_delta_auc(
            responses,
            load_llm_preds(args.full_dir),
            load_llm_preds(args.behavior_dir),
            n_boot=args.n_boot,
        )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
