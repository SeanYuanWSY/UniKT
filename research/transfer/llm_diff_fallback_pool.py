"""Batch27: LLM difficulty as pool-table fallback + pure-LLM zero-shot table.

Fallback trilogy: 0.5 (poolsize_gap.json) / aligned A09 (aligned_fallback_pool.json,
dead) / LLM k3 difficulty estimates (this file). CACHE PROVENANCE: the cache
was written by streaming_gate.py keyed by REAL KC id (str(id)); do NOT
regenerate it from idea7_difficulty_gate.py, whose estimates are keyed by
position in a filtered list — keys would silently misalign. Values are
predicted P(correct) (high = easy), idea7 gate: Junyi rho 0.61 / A17 0.15 /
EdNet weak). Same pool draws as poolgap (rng 1000+seed, 10 draws) for exact
pairing. Also scores pool_0: pure LLM difficulty table, zero target data
(PS-0 analogue with target-native LLM knowledge instead of cross-domain
alignment).
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

SIZES = [1, 3, 5, 10]
CACHE = "/root/unikt-fork/research/transfer/results/k3_difficulty_cache.json"


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

        def score(fallback_of):
            """Returns per-(n_pool) draw stats using a fallback fn for pool-unseen KCs."""
            res = {}
            for n_pool in SIZES:
                draws = []
                llm_rows = 0
                tot = 0
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
                                tot += 1
                                if kc in prior:
                                    rows.append({"y": int(ws.response[i]), "pred": prior[kc]})
                                else:
                                    v = fallback_of(kc)
                                    if v[1]:
                                        llm_rows += 1
                                    rows.append({"y": int(ws.response[i]), "pred": v[0]})
                        a = stratified_auc_rows(rows, "pred")
                        if a is not None:
                            aucs.append(a)
                    draws.append(float(np.mean(aucs)) if aucs else float("nan"))
                res[f"pool_{min(n_pool, len(pool_hists))}"] = {
                    "mean": round(float(np.mean(draws)), 4),
                    "draw_min": round(float(np.min(draws)), 4),
                    "draw_max": round(float(np.max(draws)), 4),
                    "llm_fallback_row_frac": round(llm_rows / max(tot, 1), 4),
                }
            return res

        def llm_fb(kc):
            v = llm.get(kc)
            return (v, True) if v is not None else (0.5, False)

        # pool_0: pure LLM difficulty table (zero target data), same eval protocol
        p0_rows, p0_llm, p0_tot = [], 0, 0
        for ws in eval_samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            rows = []
            for i in hold:
                if i == 0:
                    continue
                isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                if not isA1:
                    kc = int(ws.sequence[i])
                    p0_tot += 1
                    if kc in llm:
                        p0_llm += 1
                    rows.append({"y": int(ws.response[i]), "pred": llm.get(kc, 0.5)})
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                p0_rows.append(a)
        rng = np.random.default_rng(0)
        arr = np.array(p0_rows)
        boots = [float(np.mean(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(5000)]

        out[ds] = {
            "n_llm_kcs": len(llm),
            "pool_0_llm_table": {
                "mean": round(float(np.mean(arr)), 4),
                "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                "n_scored": len(arr),
                "llm_row_frac": round(p0_llm / max(p0_tot, 1), 4),
            },
            "pools": score(llm_fb),
        }
        print(ds, json.dumps(out[ds], ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/llm_diff_fallback_pool.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("LLMFB-DONE")


if __name__ == "__main__":
    main()
