"""Batch31: permutation test for batch30's V2 cohort-anchor injection at n=1.

Batch30 verdict: V1 pool-calibrated dead (perm p=0.473); V2 passed numerically
(Junyi pool_1 0.5369 = +0.0099 over 0.5-fallback, > continuous 0.5363) but had
no permutation test of its own — the batch28 lesson (assignment randomness
must be accounted) applies. Here: permute the LLM routing (KC->bucket multiset
shuffle over cache KCs, rng 400+p, 200 perms), recompute anchors from the
first-300 cohort marginals per perm (same cache-covered-only protocol as
batch30 V2), keep the same 10 n=1 pools (rng 1000+seed) for warm rows.
p=(#{perm>=real}+1)/201 per domain. A17 doubles as an artifact detector: its
V2 anchors are non-monotone and scale-mismatched — if A17's permutation ALSO
turns "significant", significance is a scale artifact, not LLM ordering, and
the Junyi read must be re-interpreted accordingly.
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

CACHE = "/root/unikt-fork/research/transfer/results/k3_difficulty_cache.json"
N_PERM = 200


def eval_rows(eval_samples):
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


def anchors_for(marg_f, bmap, v2_default):
    a = {}
    for b in (0, 1, 2):
        ks = [k for k in marg_f if k in bmap and bmap[k] == b]
        if ks:
            a[b] = float(np.mean([marg_f[k] for k in ks]))
    return a


def score_v2(per_stu, prior, bmap, anchor, v2_default):
    aucs = []
    for rows in per_stu:
        rs = []
        for kc, y in rows:
            v = prior.get(kc)
            if v is not None:
                rs.append({"y": y, "pred": v})
            elif kc in bmap:
                rs.append({"y": y, "pred": anchor.get(bmap[kc], v2_default)})
            else:
                rs.append({"y": y, "pred": v2_default})
        a = stratified_auc_rows(rs, "pred")
        if a is not None:
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


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

        vals = np.array([llm[k] for k in sorted(llm)])
        qs = np.quantile(vals, [1 / 3, 2 / 3])
        bucket = {k: (0 if v <= qs[0] else (1 if v <= qs[1] else 2)) for k, v in llm.items()}

        marg_f = fit_global_kc_prior(build_rows(samples[:300], None))
        v2_default = float(np.mean(list(marg_f.values())))

        per_stu = eval_rows(eval_samples)

        # the same 10 n=1 pools (rng 1000+seed, batch25/27/30 pairing)
        pools = []
        for seed in range(10):
            r = np.random.default_rng(1000 + seed)
            idx = r.choice(len(pool_hists), size=1, replace=False)
            counts, sums = defaultdict(int), defaultdict(int)
            for i in idx:
                for kc, hist in pool_hists[i].items():
                    counts[kc] += len(hist)
                    sums[kc] += sum(hist)
            pools.append({kc: sums[kc] / counts[kc] for kc in counts if counts[kc] > 0})

        def stat_for(bmap):
            anchor = anchors_for(marg_f, bmap, v2_default)
            return float(np.mean([score_v2(per_stu, p, bmap, anchor, v2_default) for p in pools]))

        real = stat_for(bucket)
        kcs_sorted = sorted(llm)
        base_b = [bucket[k] for k in kcs_sorted]
        ge = 0
        for p in range(N_PERM):
            r2 = np.random.default_rng(400 + p)
            bmap = dict(zip(kcs_sorted, r2.permutation(base_b)))
            if stat_for(bmap) >= real:
                ge += 1
        out[ds] = {"real": round(real, 4), "ge": ge, "n_perm": N_PERM, "p_perm": round((ge + 1) / (N_PERM + 1), 4)}
        print(ds, json.dumps(out[ds]), flush=True)

    with open("/root/unikt-fork/research/transfer/results/llm_bucket_v2_perm.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("V2PERM-DONE")


if __name__ == "__main__":
    main()
