"""Batch33: complete the 2x2 factorial + noise-seed robustness of batch32.

batch32 decomposed the b29 history channel into evidence-history (0.0134/0.0373)
and holdout-rolling-context (0.0178/0.0560) via (ev=F,hold=T)=k0 and
(ev=F,hold=F)=b29-all-scramble. Missing cell: (ev=T, hold=F) — evidence labels
true, holdout labels scrambled (noise base 1000, bit-compatible with b32).
Additivity check: loss_hold(measured here) vs b32's implied 0.0178/0.0560;
residual = loss_hold - implied; |residual|<=0.003 additive, >=0.005 synergy/
redundancy. Part B: rerun the k-curve with noise bases {1000,2000,3000}
(rng default_rng(base+uid%100000)) — base 1000 must reproduce a17_hist_recency
.json per-k (deterministic self-check); report per-k spread across bases to
test dip/shape robustness (k=1 dip, no >=50% recovery by k=10).
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from signals import _forward_probs  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

MODELS = {
    "idAKT": "runs/normal/AKT_assistments17_20261003-145008_fold0_bs64",
    "idDKT": "runs/normal/DKT_assistments17_20261003-144042_fold0_bs128",
}
KS = [0, 1, 3, 10, 20, 50, 10 ** 9]
NOISE_BASES = [1000, 2000, 3000]
B32_IMPLIED_HOLD_LOSS = {"idAKT": 0.0178, "idDKT": 0.0560}


def main(n_users=300):
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:n_users]

    out = {"b32_implied_hold_loss": B32_IMPLIED_HOLD_LOSS}
    for mtag, run in MODELS.items():
        rm = restore_any(run)
        partA = {}
        curve = {b: {k: {} for k in KS} for b in NOISE_BASES}
        for ws in samples:
            s_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            hold = sorted(set(ws.holdout_idx.tolist()))
            hold_pos = set(hold)
            L = len(s_all)
            first_hold = min(hold_pos) if hold_pos else L
            uid = ws.user_id

            noises = {}
            for base in NOISE_BASES:
                rng = np.random.default_rng(base + uid % 100000)
                noises[base] = [int(v) for v in rng.binomial(1, 0.5, size=L)]

            # Part A: (ev=T, hold=F) with base-1000 noise
            yA = list(y_all)
            for i in range(first_hold, L):
                yA[i] = noises[1000][i]

            # Part B k-curve per base (cut==0 conditions are noise-free: one forward, all bases)
            for k in KS:
                cut = max(first_hold - k, 0)
                for base in (NOISE_BASES if cut > 0 else [NOISE_BASES[0]]):
                    y_cond = list(y_all)
                    for i in range(cut):
                        y_cond[i] = noises[base][i]
                    s = torch.tensor([s_all], dtype=torch.long, device=rm.device)
                    r_ = torch.tensor([y_cond], dtype=torch.long, device=rm.device)
                    probs = _forward_probs(rm, s, r_, None)[0]
                    rows = []
                    for i in hold:
                        if i == 0:
                            continue
                        isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and s_all[i] != s_all[i - 1]
                        if not isA1:
                            rows.append({"y": y_all[i], "pred": float(probs[i])})
                    a = stratified_auc_rows(rows, "pred")
                    if a is not None:
                        if cut > 0:
                            curve[base][k][uid] = a
                        else:
                            for b2 in NOISE_BASES:
                                curve[b2][k][uid] = a

            # Part A forward (separate label vector from all k conditions)
            s = torch.tensor([s_all], dtype=torch.long, device=rm.device)
            r_ = torch.tensor([yA], dtype=torch.long, device=rm.device)
            probs = _forward_probs(rm, s, r_, None)[0]
            rows = []
            for i in hold:
                if i == 0:
                    continue
                isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and s_all[i] != s_all[i - 1]
                if not isA1:
                    rows.append({"y": y_all[i], "pred": float(probs[i])})
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                partA[uid] = a

        base_auc = float(np.mean(list(curve[1000][10 ** 9].values())))
        aA = float(np.mean(list(partA.values())))
        res = {
            "partA_evT_holdF": {"mean": round(aA, 4), "n": len(partA)},
            "base_kinf": round(base_auc, 4),
            "measured_hold_loss": round(base_auc - aA, 4),
            "additivity_residual": round((base_auc - aA) - B32_IMPLIED_HOLD_LOSS[mtag], 4),
        }
        kres = {}
        for k in KS:
            key = "inf" if k == 10 ** 9 else str(k)
            means = [float(np.mean(list(curve[b][k].values()))) for b in NOISE_BASES]
            kres[key] = {"base1000": round(means[0], 4), "base2000": round(means[1], 4), "base3000": round(means[2], 4)}
        res["kcurve_by_noise"] = kres
        out[mtag] = res
        print(mtag, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a17_factorial_noise.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("FACTORIAL-DONE")


if __name__ == "__main__":
    main()
