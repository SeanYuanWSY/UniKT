"""Batch28b: same-cohort bucket marginals + within-student cross-bucket pair share.

Two fixes for llm_diff_formal part B:
(1) bucket marginals computed on the TAIL-100 eval cohort (same population
the AUC sees; first-300 vs tail-100 cohort mismatch left EdNet unresolved);
(2) for each eval student, the share of comparable row pairs whose two rows
fall in DIFFERENT LLM tertile buckets (the pairs where the coarse partition
could possibly matter) — tests the hypothesis that within-student AUC
structurally compresses the LLM coarse signal (Junyi spread 0.26 but AUC
0.546). Also reports bucket marginals for BOTH cohorts side by side.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402


def main():
    diff = {ds: {int(k): v for k, v in d.items()} for ds, d in json.load(open("/root/unikt-fork/research/transfer/results/k3_difficulty_cache.json")).items()}
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval:]
        llm = diff[ds]

        vals = np.array([llm[k] for k in sorted(llm)])
        qs = np.quantile(vals, [1 / 3, 2 / 3])
        bucket = {k: (0 if v <= qs[0] else (1 if v <= qs[1] else 2)) for k, v in llm.items()}

        marg_first = fit_global_kc_prior(build_rows(samples[:300], None))
        marg_tail = fit_global_kc_prior(build_rows(eval_samples, None))  # eval cohort's own evidence

        def bmeans(m):
            out = []
            for b in (0, 1, 2):
                ks = [k for k in sorted(llm) if bucket[k] == b and k in m]
                out.append(round(float(np.mean([m[k] for k in ks])), 4) if ks else None)
            return out

        # within-student comparable-pair stats on scored rows
        xbp_cross, pair_tot = 0, 0
        cross_disc, cross_ydiff = 0, 0  # concordant / label-differing among cross-bucket pairs
        for ws in eval_samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            bs, ys = [], []
            for i in hold:
                if i == 0:
                    continue
                if int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1]):
                    continue
                kc = int(ws.sequence[i])
                bs.append(bucket.get(kc, 1))
                ys.append(int(ws.response[i]))
            for a in range(len(bs)):
                for b in range(a + 1, len(bs)):
                    pair_tot += 1
                    if bs[a] != bs[b]:
                        xbp_cross += 1
                        if ys[a] != ys[b]:
                            cross_ydiff += 1
                            if (bs[a] < bs[b]) == (ys[a] < ys[b]):
                                cross_disc += 1
        out[ds] = {
            "bucket_marginals_first300": bmeans(marg_first),
            "bucket_marginals_tail100": bmeans(marg_tail),
            "cross_bucket_pair_share": round(xbp_cross / max(pair_tot, 1), 4),
            "coarse_pair_acc": round(cross_disc / max(cross_ydiff, 1), 4),
            "n_cross_label_pairs": cross_ydiff,
            "n_pairs": pair_tot,
        }
        print(ds, json.dumps(out[ds], ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/llm_diff_bucket_pairs.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("BUCKETPAIRS-DONE")


if __name__ == "__main__":
    main()
