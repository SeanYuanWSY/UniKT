"""Batch32: history recency curve — graded prefix scramble on frozen A17 models.

batch29 established the response-history channel carries the whole native-deep
advantage (hist-scramble erases 86%/109%, models fall back to table level).
This batch grades WHERE in the history the interaction lives: for k in
{0,1,3,10,20,50,inf}, evidence labels at positions < first_hold-k are replaced
by Bernoulli(0.5) draws (same per-student noise array across k — conditions
are strictly nested); the most recent k evidence labels and ALL holdout labels
stay true. k=inf must reproduce the base anchors 0.584/0.633 exactly.
Decomposition readout:
  base(k=inf) - ev_scramble(k=0)  = evidence-history contribution
  ev_scramble(k=0) - b29_all_scramble(0.5529/0.5398) = holdout rolling context
  recovery midpoint k* (gap recovered to 50%) = recency vs accumulation
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
B29_REF = {"idAKT": 0.5529, "idDKT": 0.5398}  # all-scramble reference (evidence+holdout)


def main(n_users=300):
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:n_users]

    out = {"b29_all_scramble_ref": B29_REF}
    for mtag, run in MODELS.items():
        rm = restore_any(run)
        cond_aucs = {k: {} for k in KS}
        for ws in samples:
            s_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            hold = sorted(set(ws.holdout_idx.tolist()))
            hold_pos = set(hold)
            L = len(s_all)
            first_hold = min(hold_pos) if hold_pos else L
            rng = np.random.default_rng(1000 + ws.user_id % 100000)
            y_noise = [int(v) for v in rng.binomial(1, 0.5, size=L)]  # one noise array, shared across k

            for k in KS:
                cut = max(first_hold - k, 0)
                y_cond = list(y_all)
                for i in range(cut):  # scramble only the OLD evidence prefix
                    y_cond[i] = y_noise[i]
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
                    cond_aucs[k][ws.user_id] = a

        res = {}
        for k in KS:
            m = float(np.mean(list(cond_aucs[k].values())))
            key = "inf" if k == 10 ** 9 else str(k)
            res[key] = {"mean": round(m, 4), "n": len(cond_aucs[k])}
        out[mtag] = res
        print(mtag, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a17_hist_recency.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("RECENCY-DONE")


if __name__ == "__main__":
    main()
