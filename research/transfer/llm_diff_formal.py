"""Batch28: formalize the LLM difficulty zero-shot signal (batch27 follow-up).

Part A: 200-permutation null for the pool_0 LLM table (batch27 used 10
perms — EdNet real exceeded all; formalize p-values for all domains).
Part B: coarse-vs-fine decomposition. Cache values bucketed into tertiles
(LLM-easy/mid/hard); score (a) full-value table, (b) 3-level bucket-constant
table, (c) within-bucket Spearman rho of cache vs target marginal. If
bucket AUC ~ full AUC while within-bucket rho ~ 0, the LLM carries a coarse
tri-partition, not fine ranks ("粗对细错") — design input for the streaming
cold-KC protocol (quantized injection).
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

N_PERM = 200


def rankdata_avg(x):
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


def main():
    diff = {ds: {int(k): v for k, v in d.items()} for ds, d in json.load(open("/root/unikt-fork/research/transfer/results/k3_difficulty_cache.json")).items()}
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval:]
        llm = diff[ds]

        def pool0_auc(pred_of):
            aucs = []
            for ws in eval_samples:
                hold = sorted(set(ws.holdout_idx.tolist()))
                rows = []
                for i in hold:
                    if i == 0:
                        continue
                    if int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1]):
                        continue
                    kc = int(ws.sequence[i])
                    rows.append({"y": int(ws.response[i]), "pred": pred_of(kc)})
                a = stratified_auc_rows(rows, "pred")
                if a is not None:
                    aucs.append(a)
            return float(np.mean(aucs)), aucs

        real = pool0_auc(lambda kc: llm.get(kc, 0.5))[0]

        # Part A: 200-permutation null
        perms = []
        for p in range(N_PERM):
            rng = np.random.default_rng(200 + p)
            kcs = sorted(llm.keys())
            sh = rng.permutation(len(kcs))
            pllm = {kcs[sh[t]]: llm[kcs[t]] for t in range(len(kcs))}
            perms.append(pool0_auc(lambda kc: pllm.get(kc, 0.5))[0])
        # Codex Run A P1-6: with no valid students, real and every perm are NaN;
        # NaN >= NaN is False everywhere, yielding a bogus p = 1/(B+1). Guard.
        perm_arr = np.array(perms)
        if np.isfinite(real) and np.isfinite(perm_arr).all():
            pval = float((np.sum(perm_arr >= real) + 1) / (N_PERM + 1))
        else:
            pval = None

        # Part B: coarse vs fine
        vals = np.array([llm[k] for k in sorted(llm)])
        qs = np.quantile(vals, [1 / 3, 2 / 3])
        bucket = {k: (0 if v <= qs[0] else (1 if v <= qs[1] else 2)) for k, v in llm.items()}
        bpred = {k: float(b) for k, b in bucket.items()}  # 0=LLM-hardest .. 2=LLM-easiest
        auc_full = pool0_auc(lambda kc: llm.get(kc, 0.5))[0]
        auc_bucket = pool0_auc(lambda kc: bpred.get(kc, 1.0))[0]

        # within-bucket rank corr vs target marginal (first-300 evidence)
        tgt_prior = fit_global_kc_prior(build_rows(samples[:300], None))
        wbr = {}
        for b in (0, 1, 2):
            ks = [k for k in sorted(llm) if bucket[k] == b and k in tgt_prior]
            if len(ks) >= 5:
                rx, ry = rankdata_avg([llm[k] for k in ks]), rankdata_avg([tgt_prior[k] for k in ks])
                wbr[str(b)] = {"n": len(ks), "rho": round(float(np.corrcoef(rx, ry)[0, 1]), 4),
                               "mean_tgt_marginal": round(float(np.mean([tgt_prior[k] for k in ks])), 4)}
        # monotonicity check: target marginal should increase 0->2
        mono = [round(float(np.mean([tgt_prior[k] for k in sorted(llm) if bucket[k] == b and k in tgt_prior])), 4) for b in (0, 1, 2)]

        out[ds] = {
            "pool0_full": round(auc_full, 4),
            "perm200_mean": round(float(np.mean(perms)), 4),
            "perm200_max": round(float(np.max(perms)), 4),
            "perm200_pvalue": round(pval, 4),
            "pool0_bucket3": round(auc_bucket, 4),
            "bucket_marginals_0to2": mono,
            "within_bucket": wbr,
            "n_unique_values": len(set(llm.values())),
        }
        print(ds, json.dumps(out[ds], ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/llm_diff_formal.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("LLMFORMAL-DONE")


if __name__ == "__main__":
    main()
