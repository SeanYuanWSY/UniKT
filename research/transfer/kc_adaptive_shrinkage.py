"""Batch23: per-KC adaptive shrinkage (hierarchical alpha) vs fixed-alpha B2.

b234 showed global-alpha recency blend (B2) helps only EdNet (+0.011) and
hurts A17/Junyi (-0.022 each). Hypothesis: the blend weight should adapt to
evidence volume, not stay global. Here the per-student KC recency mean is
shrunk toward the global KC prior with a pseudo-count m (Beta-Binomial
posterior mean): pred = (n*rec + m*prior) / (n + m), n = capped evidence
attempts. Two variants: one global m; per-bucket m (KC global evidence
thin <50 / mid / rich >=500) by coordinate ascent. All selection on
evidence-side pseudo-holdout (b234 LOO discipline, no holdout labels);
eval is t1ci protocol (first-300, A1-excluded, per-student AUC) with
paired per-student bootstrap CI vs B1. Anchors must reproduce: B1 =
0.548/0.525/0.562 (t1ci); B3_fixed_anchor (alpha blend, >=3 evidence gate
on rec, matching baseline_eval recent5) ~ b234 B3 = 0.5276/0.534/0.5491.
Both shrink variants evaluated: global m_star and bucket mB.
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

MS = (1, 2, 4, 8, 16, 32)
NCAP = 20


def bucket(g):
    return 0 if g < 50 else (1 if g < 500 else 2)


def blend(rec, n, pri, m):
    """Beta-Binomial style posterior mean; falls back to prior with no recency."""
    if rec is None or n == 0:
        return pri
    return (n * rec + m * pri) / (n + m)


def main(n_users=300):
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        students = build_rows(samples, None)
        prior = fit_global_kc_prior(students)

        # evidence-side LOO priors and per-student KC histories (b234 discipline)
        g_sum, g_cnt = defaultdict(float), defaultdict(int)
        for st in students:
            for kc, hist in st["kc_hist"].items():
                g_sum[kc] += float(sum(hist))
                g_cnt[kc] += len(hist)

        # --- selection rows, ONE pass: (loo_prior, rec, n, bucket, y)
        pseudo = []
        for st in students:
            for kc, hist in st["kc_hist"].items():
                if len(hist) >= 6 and g_cnt[kc] > len(hist):
                    pri = (g_sum[kc] - float(sum(hist))) / (g_cnt[kc] - len(hist))
                    b = bucket(g_cnt[kc])
                    for t in range(5, len(hist)):
                        rec = float(np.mean(hist[max(0, t - 5):t]))
                        pseudo.append((pri, rec, min(t, NCAP), b, hist[t]))

        def sel_auc(m_of):
            rows = [{"y": y, "pred": blend(rec, n, pri, m_of(b))} for (pri, rec, n, b, y) in pseudo]
            a = stratified_auc_rows(rows, "pred")
            return a if a is not None else 0.5

        m_star = max(MS, key=lambda m: sel_auc(lambda b, m=m: m))
        mB = {b: m_star for b in (0, 1, 2)}
        for _ in range(2):  # coordinate ascent on bucket m values
            for b in (0, 1, 2):
                mB[b] = max(MS, key=lambda m: sel_auc(lambda bb, m=m, b=b: m if bb == b else mB[bb]))

        # --- eval: t1ci protocol (first-300, A1-excluded, per-student AUC)
        # per-student evidence-side recency: kc -> (rec, n)
        st_att = {}
        for st in students:
            st_att[st["user_id"]] = {
                kc: (float(np.mean(hist[-5:])) if hist else None, min(len(hist), NCAP))
                for kc, hist in st["kc_hist"].items()
            }
        alpha_star = {"assistments17": 0.25, "ednet_kt1": 0.5, "junyi2015": 0.25}[ds]  # b234 best

        b1_aucs, shr_aucs, b3_aucs, shrg_aucs = {}, {}, {}, {}
        for ws in samples:
            hold_set = set(ws.holdout_idx.tolist())
            q_all = [int(x) for x in ws.question.tolist()]
            kc_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            att = st_att.get(ws.user_id, {})
            r1, r2, r3, r4 = [], [], [], []
            for i in range(len(kc_all)):
                if i in hold_set and i > 0:
                    if q_all[i] == q_all[i - 1] and kc_all[i] != kc_all[i - 1]:
                        continue  # A1 excluded
                    kc = kc_all[i]
                    pri = prior.get(kc, 0.5)
                    rec_n = att.get(kc)
                    rec = rec_n[0] if rec_n else None
                    n = rec_n[1] if rec_n else 0
                    y = y_all[i]
                    r1.append({"y": y, "pred": pri})
                    r2.append({"y": y, "pred": blend(rec, n, pri, mB[bucket(g_cnt.get(kc, 0))])})
                    r4.append({"y": y, "pred": blend(rec, n, pri, m_star)})
                    # B3 anchor: alpha blend with the >=3 evidence gate of recent5
                    r3.append({"y": y, "pred": alpha_star * pri + (1 - alpha_star) * rec if (rec is not None and n >= 3) else pri})
            a1 = stratified_auc_rows(r1, "pred")
            a2 = stratified_auc_rows(r2, "pred")
            a3 = stratified_auc_rows(r3, "pred")
            if a1 is not None:
                b1_aucs[ws.user_id] = a1
            if a2 is not None:
                shr_aucs[ws.user_id] = a2
            if a3 is not None:
                b3_aucs[ws.user_id] = a3
            a4 = stratified_auc_rows(r4, "pred")
            if a4 is not None:
                shrg_aucs[ws.user_id] = a4

        # paired per-student bootstrap: shrink - B1 on common students
        common = sorted(set(b1_aucs) & set(shr_aucs))
        arr = np.array([shr_aucs[u] - b1_aucs[u] for u in common])
        rng = np.random.default_rng(0)
        boots = [float(np.mean(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(5000)]

        res = {
            "m_star": m_star,
            "m_bucket": {str(b): mB[b] for b in sorted(mB)},
            "B1_A1excluded": round(float(np.mean(list(b1_aucs.values()))), 4),
            "B3_fixed_anchor": round(float(np.mean(list(b3_aucs.values()))), 4),
            "shrink_global_m": round(float(np.mean(list(shrg_aucs.values()))), 4),
            "shrink_bucket_m": round(float(np.mean(list(shr_aucs.values()))), 4),
            "delta_vs_B1": round(float(np.mean(list(shr_aucs.values()))) - float(np.mean(list(b1_aucs.values()))), 4),
            "ci95_delta": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "n_paired": len(common),
        }
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/kc_adaptive_shrinkage.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("KC-SHRINK-DONE")


if __name__ == "__main__":
    main()
