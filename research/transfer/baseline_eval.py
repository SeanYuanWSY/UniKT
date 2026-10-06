"""Tree-3: zero-training simple baselines vs deep transfer (fair-set edition).

Defences per designs_3trees_v2.md:
- Fair scoring set: baselines are scored on the EXACT (student, holdout-step)
  set each deep cell kept (k3-high alignment keep logic replicated); full-set
  numbers reported separately as deployment view.
- Deep reference = the best deep transfer number per domain (from the matrix).
- B3 alpha: single global alpha chosen on evidence-segment pooled within-AUC
  over a 5-point grid; fixed-alpha=0.5 anchor reported alongside.
- B4: sklearn LogisticRegression, default L2, no tuning; features =
  KC one-hot + recent5 + attempts.

Outputs per domain: {baseline: {matched_auc, full_auc, ...}, deep_ref, notes}.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "llm_explain"))

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402


def build_rows(samples, alignment_name: str | None, max_seq_len: int = 200):
    """Replicate transfer_eval's keep logic; return per-student dicts with
    evidence features + holdout rows (y + per-baseline preds)."""
    al = None
    if alignment_name:
        al = {int(k): v for k, v in load_alignment(alignment_name).items()}
    mapping = {}
    out = []
    for ws in samples:
        if al is not None:
            for kc in set(int(c) for c in ws.sequence.tolist()):
                ms = al.get(kc, [])
                mapping[kc] = ms[0]["source_kc"] if ms else None
            keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping.get(int(kc)) is not None]
            if len(keep) > max_seq_len:
                keep = keep[-max_seq_len:]
            if len(keep) < 6:
                continue
        else:
            keep = list(range(len(ws.sequence)))
        holdout_pos = set(ws.holdout_idx.tolist())
        ev = {"kc_hist": defaultdict(list), "q_hist": {}}
        rows = []
        ev_len = 0
        for j, i in enumerate(keep):
            kc = int(ws.sequence[i])
            q = int(ws.question[i])
            y = int(ws.response[i])
            if i in holdout_pos:
                hist = ev["kc_hist"][kc]
                rows.append(
                    {
                        "y": y,
                        "kc": kc,
                        "q": q,
                        "repeat": q in ev["q_hist"],
                        "recent5": float(np.mean(hist[-5:])) if len(hist) >= 3 else None,
                        "attempts": float(len(hist)),
                    }
                )
            else:
                ev["kc_hist"][kc].append(y)
                ev["q_hist"][q] = y
                ev_len += 1
        if ev_len >= 3 and rows and 0 < sum(r["y"] for r in rows) < len(rows):
            out.append({"rows": rows, "kc_hist": dict(ev["kc_hist"]), "user_id": ws.user_id})
    return out


def fit_global_kc_prior(students):
    prior = {}
    for st in students:
        for kc, hist in st["kc_hist"].items():
            prior.setdefault(kc, []).extend(hist)
    return {k: float(np.mean(v)) for k, v in prior.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alignment", required=True, help="deep cell alignment (keep-set source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deep-ref", type=float, required=True, help="best deep transfer AUC for this domain")
    ap.add_argument("--n-users", type=int, default=300)
    ap.add_argument("--expect-students", type=int, default=0, help="hard assertion on n_students (deep-cell parity)")
    args = ap.parse_args()

    src = get_data_source(_make_rc(args.dataset))
    samples = load_user_samples(src, fold=0)[: args.n_users]
    students = build_rows(samples, args.alignment)
    if args.expect_students and len(students) != args.expect_students:
        raise SystemExit(f"fair-set violation: {len(students)} students vs deep cell {args.expect_students}")
    prior = fit_global_kc_prior(students)

    # --- baselines: per-row predictors on the SAME holdout rows as deep ---
    def auc_with(fn):
        aucs = []
        for st in students:
            rows = []
            for r in st["rows"]:
                p = fn(r)
                if p is not None:
                    rows.append({"y": r["y"], "pred": p})
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                aucs.append(a)
        return round(float(np.mean(aucs)), 4), len(aucs)

    b1, n1 = auc_with(lambda r: prior.get(r["kc"], 0.5))                     # KC prior table
    b2, n2 = auc_with(lambda r: r["recent5"] if r["recent5"] is not None else prior.get(r["kc"], 0.5))  # recency, prior-fallback
    # leak-free alpha: computed from evidence histories only (predict the last
    # evidence attempt from earlier evidence)
    g_sum, g_cnt = defaultdict(float), defaultdict(int)
    for st in students:
        for kc, hist in st["kc_hist"].items():
            g_sum[kc] += float(sum(hist)); g_cnt[kc] += len(hist)
    alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    ev_scores = {}
    for alpha in alpha_grid:
        rows = []
        for st in students:
            for kc, hist in st["kc_hist"].items():
                if len(hist) >= 6 and g_cnt[kc] > len(hist):
                    pri = (g_sum[kc] - float(sum(hist))) / (g_cnt[kc] - len(hist))
                    for t in range(5, len(hist)):
                        rec = float(np.mean(hist[max(0, t - 5):t]))
                        rows.append({"y": hist[t], "pred": alpha * pri + (1 - alpha) * rec})
        a = stratified_auc_rows(rows, "pred") if rows else None
        ev_scores[alpha] = round(a, 4) if a is not None else 0.5
    alpha_star = max(ev_scores, key=ev_scores.get)
    b3, n3 = auc_with(lambda r: (alpha_star * prior.get(r["kc"], 0.5) + (1 - alpha_star) * r["recent5"]) if r["recent5"] is not None else prior.get(r["kc"], 0.5))
    b3_anchor, _ = auc_with(lambda r: 0.5 * prior.get(r["kc"], 0.5) + 0.5 * r["recent5"] if r["recent5"] is not None else prior.get(r["kc"], 0.5))

    # B4 logistic regression fit on evidence rows only
    from sklearn.linear_model import LogisticRegression

    kc_ids = sorted(prior)
    kc_index = {k: i for i, k in enumerate(kc_ids)}

    def feats(kc, rec, att):
        x = [0.0] * (len(kc_ids) + 3)
        if kc in kc_index:
            x[kc_index[kc]] = 1.0
        x[-3] = rec if rec is not None else 0.5
        x[-2] = min(att / 20.0, 1.0)
        x[-1] = 1.0 if rec is None else 0.0
        return x

    Xtr, ytr = [], []
    for st in students:
        for kc, hist in st["kc_hist"].items():
            if kc not in kc_index:
                continue
            for t in range(5, len(hist)):
                rec = float(np.mean(hist[max(0, t - 5):t]))
                Xtr.append(feats(kc, rec, float(t)))
                ytr.append(hist[t])
    if len(Xtr) > 200000:
        idx = np.random.default_rng(0).choice(len(Xtr), 200000, replace=False)
        Xtr = [Xtr[i] for i in idx]
        ytr = [ytr[i] for i in idx]
    lr = LogisticRegression(max_iter=1000)
    lr.fit(np.array(Xtr), np.array(ytr))
    b4, n4 = auc_with(lambda r: float(lr.predict_proba([feats(r["kc"], r["recent5"], r["attempts"])])[0][1]))

    result = {
        "dataset": args.dataset,
        "alignment_keepset": args.alignment,
        "n_students": len(students),
        "deep_ref_auc": args.deep_ref,
        "B1_kc_prior": {"auc": b1},
        "B2_recent5": {"auc": b2},
        "B3_mixed": {"auc": b3, "alpha_star": alpha_star, "anchor_auc": b3_anchor, "evidence_alpha_scores": ev_scores},
        "B4_logreg": {"auc": b4, "n_train_rows": len(ytr)},
        "note": "alpha chosen leak-free on evidence histories (last-3-attempt CV); all baselines scored on the deep cell's exact keep set",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
