"""Batch26: aligned A09 prior as pool-table fallback (PS-0 utilization recheck).

poolgap showed the cold-start curve (0.5-fallback for KCs unseen in the
pool). Here the fallback is swapped for the LLM-aligned A09 prior — same
pool draws (rng 1000+seed, 10 draws, identical code path) so pool_n points
pair exactly with poolsize_gap.json. Zero selection knobs. Expectation from
aligndiag: Junyi (correct alignment) gains; A17 (level inversion) / EdNet
(rank inversion) flat or worse. Also reports fallback usage per point.
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = {
    "assistments17": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
    "junyi2015": "assistments09__to__junyi2015__llm__k3-high__c25k3__rep0",
}
SIZES = [1, 3, 5, 10]


def main():
    src09 = get_data_source(_make_rc("assistments09"))
    s09_prior = fit_global_kc_prior(build_rows(load_user_samples(src09, fold=0), None))

    out = {}
    for ds, align_name in ALIGN.items():
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval_actual = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval_actual:]
        pool_samples = samples[: len(samples) - n_eval_actual]
        pool_hists = [st["kc_hist"] for st in build_rows(pool_samples, None)]
        alignment = {int(k): v for k, v in load_alignment(align_name).items()}

        # aligned fallback: target KC -> A09 source prior (None where unmapped)
        afb = {}
        for kc, ms in alignment.items():
            if ms:
                p = s09_prior.get(ms[0]["source_kc"])
                if p is not None:
                    afb[kc] = p

        res = {"n_eval": len(eval_samples), "n_aligned_fallback_kcs": len(afb)}
        for n_pool in SIZES:
            draws = []
            fb_rows = 0
            aligned_fb_rows = 0  # rows where the aligned prior was actually used
            tot_rows = 0
            for seed in range(10):
                r = np.random.default_rng(1000 + seed)
                idx = r.choice(len(pool_hists), size=min(n_pool, len(pool_hists)), replace=False)
                counts, sums = defaultdict(int), defaultdict(int)
                for i in idx:
                    for kc, hist in pool_hists[i].items():
                        counts[kc] += len(hist)
                        sums[kc] += sum(hist)
                prior = {kc: sums[kc] / counts[kc] for kc in counts if counts[kc] > 0}
                aucs = []
                for ws in eval_samples:
                    hold = sorted(set(ws.holdout_idx.tolist()))
                    rows = []
                    for i in hold:
                        if i == 0:
                            continue
                        isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                        if not isA1:
                            kc = int(ws.sequence[i])
                            tot_rows += 1
                            if kc in prior:
                                rows.append({"y": int(ws.response[i]), "pred": prior[kc]})
                            else:
                                fb_rows += 1
                                if kc in afb:
                                    aligned_fb_rows += 1
                                rows.append({"y": int(ws.response[i]), "pred": afb.get(kc, 0.5)})  # aligned fallback, 0.5 if also unmapped
                    a = stratified_auc_rows(rows, "pred")
                    if a is not None:
                        aucs.append(a)
                draws.append(float(np.mean(aucs)) if aucs else float("nan"))
            res[f"pool_{min(n_pool, len(pool_hists))}"] = {
                "mean": round(float(np.mean(draws)), 4),
                "draw_min": round(float(np.min(draws)), 4),
                "draw_max": round(float(np.max(draws)), 4),
                "fallback_row_frac": round(fb_rows / max(tot_rows, 1), 4),
                "aligned_fallback_row_frac": round(aligned_fb_rows / max(tot_rows, 1), 4),
            }
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/aligned_fallback_pool.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("ALIGNEDFB-DONE")


if __name__ == "__main__":
    main()
