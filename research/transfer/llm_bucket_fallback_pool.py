"""Batch30: quantized cold-KC injection — LLM tertile buckets as fallback anchors.

Design basis batch28/28b: LLM coarse partition is real and strong (Junyi
bucket-marginal spread 0.26, within-student coarse order 71.7%) while the
fine rank is noisy (bucket-internal rho ~0). So inject ONLY the coarse tier:
cold-KC (pool-unseen) pred = bucket mean. Bucket means come from either
  V1 pool-calibrated: the pool's own KC marginals grouped by LLM bucket
     (pure cold-start, no external data; empty bucket -> pool global mean)
  V2 cohort-anchor: first-300 evidence cohort marginals by bucket
     (historical anchor; for Junyi these are batch28b's 0.547/0.706/0.811)
Fallback trilogy so far: 0.5 (batch25) / aligned A09 (batch26, dead) /
continuous LLM k3 (batch27, EdNet conditional-pass). Same draws as
poolgap/llmfb (rng 1000+seed, 10 draws) for exact pairing. Primary gate:
Junyi V1 pool_1. Permutation test at n=1 (Junyi): shuffle the bucket
multiset across cache KCs (rng 300+p, 200 perms), recompute pool bucket
means per perm over the SAME 10 pools; p=(#{perm>=real}+1)/201.
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

SIZES = [1, 3, 5, 10, 30]
CACHE = "/root/unikt-fork/research/transfer/results/k3_difficulty_cache.json"
N_PERM = 200
PERM_DOMAINS = ["junyi2015"]


def eval_rows(eval_samples):
    """Precompute per-student scored (kc, y) row lists once."""
    per_stu = []
    for ws in eval_samples:
        hold = sorted(set(ws.holdout_idx.tolist()))
        rows = []
        for i in hold:
            if i == 0:
                continue
            isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
            if not isA1:
                rows.append((int(ws.sequence[i]), int(ws.response[i])))
        per_stu.append(rows)
    return per_stu


def score_with(per_stu, prior, fb_of):
    """Mean per-student AUC: warm->prior[kc], cold->fb_of(kc)=(value, is_bucket)."""
    aucs, cold_bucket_rows, cold_rows = [], 0, 0
    for rows in per_stu:
        rs = []
        for kc, y in rows:
            v = prior.get(kc)
            if v is not None:
                rs.append({"y": y, "pred": v})
            else:
                val, is_b = fb_of(kc)
                cold_rows += 1
                cold_bucket_rows += int(is_b)
                rs.append({"y": y, "pred": val})
        a = stratified_auc_rows(rs, "pred")
        if a is not None:
            aucs.append(a)
    return (float(np.mean(aucs)) if aucs else float("nan")), cold_bucket_rows, cold_rows


def pool_bucket_means(prior, bmap):
    """Bucket means over the pool's own KCs; empty bucket -> None; g = pool global mean."""
    m = {}
    for b in (0, 1, 2):
        ks = [k for k in prior if bmap.get(k, 1) == b]
        m[b] = float(np.mean([prior[k] for k in ks])) if ks else None
    g = float(np.mean(list(prior.values()))) if prior else 0.5
    return m, g


def make_fb(m1, g1, bmap):
    def fb(kc):
        b = bmap.get(kc)
        if b is not None and m1.get(b) is not None:
            return (m1[b], True)
        return (g1, False)

    return fb


def main():
    diff = {ds: {int(k): v for k, v in d.items()} for ds, d in json.load(open(CACHE)).items()}
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval_actual = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval_actual:]
        pool_samples = samples[: len(samples) - n_eval_actual]
        pool_hists = [st["kc_hist"] for st in build_rows(pool_samples, None)]
        llm = diff.get(ds, {})

        # LLM tertile buckets (llm_diff_formal.py protocol: 0=hardest)
        vals = np.array([llm[k] for k in sorted(llm)])
        qs = np.quantile(vals, [1 / 3, 2 / 3])
        bucket = {k: (0 if v <= qs[0] else (1 if v <= qs[1] else 2)) for k, v in llm.items()}

        # V2 anchors: first-300 evidence cohort marginals by bucket
        marg_f = fit_global_kc_prior(build_rows(samples[:300], None))
        anchor = {}
        for b in (0, 1, 2):
            ks = [k for k in marg_f if k in bucket and bucket[k] == b]  # cache-covered only, batch28b bmeans protocol
            if ks:
                anchor[b] = float(np.mean([marg_f[k] for k in ks]))
        v2_default = float(np.mean(list(marg_f.values())))

        def fb2(kc):
            if kc not in bucket:
                return (v2_default, False)  # cache-外 KC -> 队列全局均值, not bucket-1
            return (anchor.get(bucket[kc], v2_default), True)

        per_stu = eval_rows(eval_samples)

        # precompute the 10 pools per size (same rng discipline as poolgap)
        pools = {}
        for n_pool in SIZES:
            plist = []
            for seed in range(10):
                r = np.random.default_rng(1000 + seed)
                idx = r.choice(len(pool_hists), size=min(n_pool, len(pool_hists)), replace=False)
                counts, sums = defaultdict(int), defaultdict(int)
                for i in idx:
                    for kc, hist in pool_hists[i].items():
                        counts[kc] += len(hist)
                        sums[kc] += sum(hist)
                plist.append({kc: sums[kc] / counts[kc] for kc in counts if counts[kc] > 0})
            pools[n_pool] = plist

        res = {"n_cache_kcs": len(llm), "v2_anchor_means": {str(b): round(anchor.get(b, v2_default), 4) for b in (0, 1, 2)}}
        for n_pool in SIZES:
            draws1, draws2, cb, cr = [], [], 0, 0
            for prior in pools[n_pool]:
                m1, g1 = pool_bucket_means(prior, bucket)
                a1, b1n, r1n = score_with(per_stu, prior, make_fb(m1, g1, bucket))
                draws1.append(a1)
                cb += b1n
                cr += r1n
                a2, _, _ = score_with(per_stu, prior, fb2)
                draws2.append(a2)
            res[f"pool_{min(n_pool, len(pool_hists))}"] = {
                "v1_poolcal": {"mean": round(float(np.mean(draws1)), 4), "draw_min": round(float(np.min(draws1)), 4), "draw_max": round(float(np.max(draws1)), 4)},
                "v2_cohort_anchor": {"mean": round(float(np.mean(draws2)), 4)},
                "cold_bucket_row_frac": round(cb / max(cr, 1), 4),
            }

        if ds in PERM_DOMAINS:
            def stat_for(bmap):
                vals_ = []
                for prior in pools[1]:
                    m1, g1 = pool_bucket_means(prior, bmap)
                    a, _, _ = score_with(per_stu, prior, make_fb(m1, g1, bmap))
                    vals_.append(a)
                return float(np.mean(vals_))

            real = stat_for(bucket)
            kcs_sorted = sorted(llm)
            base_b = [bucket[k] for k in kcs_sorted]
            ge = 0
            for p in range(N_PERM):
                r2 = np.random.default_rng(300 + p)
                bmap = dict(zip(kcs_sorted, r2.permutation(base_b)))
                if stat_for(bmap) >= real:
                    ge += 1
            res["perm_pool_1"] = {"real": round(real, 4), "ge": ge, "n_perm": N_PERM, "p_perm": round((ge + 1) / (N_PERM + 1), 4)}

        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/llm_bucket_fallback_pool.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("BUCKETFB-DONE")


if __name__ == "__main__":
    main()
