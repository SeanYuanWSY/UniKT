"""Streaming gate robustness: 10 random arrival orders (review P1).

The single timestamp order = one draw. Rerun the cold-KC gate over 10 random
permutations of student arrival (keeping honest default arm + LLM arm +
random-difficulty control); report mean +- std of the LLM delta per domain.
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc, build_kc_table  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

N_COLD = 20
N_ORDERS = 10
CACHE = "/root/unikt-fork/research/transfer/results/k3_difficulty_cache.json"
cache = json.load(open(CACHE))


def auc_pairs(scores, rows_meta):
    num = den = 0.0
    by_student = defaultdict(list)
    for sc, (y, uid, is_cold) in zip(scores, rows_meta):
        by_student[uid].append((sc, y, is_cold))
    for rs in by_student.values():
        for sc_c, y_c, _ in [r for r in rs if r[2]]:
            for sc_o, y_o, _ in rs:
                if y_c != y_o and (sc_o, y_o) != (sc_c, y_c):
                    den += 1
                    if (sc_c > sc_o) == (y_c > y_o):
                        num += 1
                    elif sc_c == sc_o:
                        num += 0.5
    return num / den if den else None


def run_domain(dataset, n_users=300):
    src = get_data_source(_make_rc(dataset))
    samples = load_user_samples(src, fold=0)[: n_users + 150]
    llm_p = {int(k): v for k, v in cache[dataset].items()}

    deltas_llm, deltas_rand = [], []
    for order_seed in range(N_ORDERS):
        rng = np.random.default_rng(5000 + order_seed)
        order = rng.permutation(len(samples))
        counts, sums = defaultdict(int), defaultdict(int)
        rows_meta, s_def, s_llm, s_rnd = [], [], [], []
        n_eval = 0
        rr = np.random.default_rng(99)
        for idx in order:
            if n_eval >= n_users:
                break
            ws = samples[idx]
            n_eval += 1
            for i in ws.holdout_idx.tolist():
                kc = int(ws.sequence[i])
                n_kc = counts.get(kc, 0)
                raw = (sums[kc] / counts[kc]) if n_kc > 0 else 0.5
                cold = n_kc < N_COLD
                rows_meta.append((int(ws.response[i]), ws.user_id, cold))
                s_def.append(raw)
                s_llm.append(llm_p.get(kc, 0.5) if n_kc == 0 else raw)
                s_rnd.append(float(rr.uniform(0.3, 0.8)) if n_kc == 0 else raw)
            for i in ws.evidence_idx.tolist():
                kc = int(ws.sequence[i])
                counts[kc] += 1
                sums[kc] += int(ws.response[i])
        a_def = auc_pairs(s_def, rows_meta)
        a_llm = auc_pairs(s_llm, rows_meta)
        a_rnd = auc_pairs(s_rnd, rows_meta)
        if a_def and a_llm:
            deltas_llm.append(a_llm - a_def)
        if a_def and a_rnd:
            deltas_rand.append(a_rnd - a_def)
    return {
        "n_orders": len(deltas_llm),
        "llm_delta_mean": round(float(np.mean(deltas_llm)), 4),
        "llm_delta_std": round(float(np.std(deltas_llm)), 4),
        "random_delta_mean": round(float(np.mean(deltas_rand)), 4),
        "random_delta_std": round(float(np.std(deltas_rand)), 4),
    }


res = {}
for ds in ["junyi2015", "assistments17", "ednet_kt1"]:
    res[ds] = run_domain(ds)
    print(ds, json.dumps(res[ds], ensure_ascii=False), flush=True)
with open("/root/unikt-fork/research/transfer/results/streaming_order_robust.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("ORDER-ROBUST-DONE")
