"""Batch29: channel ablation of the A17 native-deep advantage (idAKT 0.584 / idDKT 0.633 vs B1 0.548).

batch22 closed "not global ability"; the remaining question is WHICH input
channel carries the +0.036/+0.085 advantage. Four forward-pass conditions
on frozen native models, t1ci protocol (first-300, A1-excluded, per-student
AUC), rng fixed per student:
  base      — original inputs (must reproduce t1ci anchors 0.584/0.633)
  hist-scramble — ALL response labels replaced by Bernoulli(0.5) draw
                  (kills the response-history channel)
  kc-relabel   — global permutation of KC ids, same map for all students
                  (kills KC content/embeddings, keeps co-occurrence+labels)
  order-shuffle— evidence-segment (kc,y) pairs jointly shuffled, holdout
                  tail kept in order (kills temporal/recency structure)
Attribution readout: gap_after = auc_abl - 0.548 vs base gap; the ablation
that erases >=2/3 of the gap names the dominant channel.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from signals import _forward_probs  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

MODELS = {
    "idAKT": "runs/normal/AKT_assistments17_20261003-145008_fold0_bs64",
    "idDKT": "runs/normal/DKT_assistments17_20261003-144042_fold0_bs128",
}
B1_REF = 0.548


def main(n_users=300):
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:n_users]
    students = build_rows(samples, None)
    prior = fit_global_kc_prior(students)

    # global KC permutation (fixed)
    max_kc = max(int(c) for ws in samples for c in ws.sequence.tolist())
    perm_rng = np.random.default_rng(7)
    kc_perm = perm_rng.permutation(max_kc + 1)

    out = {"prior_mean": round(float(np.mean([p for p in prior.values()])), 4)}
    for mtag, run in MODELS.items():
        rm = restore_any(run)
        cond_aucs = {c: {} for c in ("base", "hist-scramble", "kc-relabel", "order-shuffle")}
        for ws in samples:
            s_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            hold = sorted(set(ws.holdout_idx.tolist()))
            hold_pos = set(hold)
            L = len(s_all)
            rng = np.random.default_rng(1000 + ws.user_id % 100000)

            # condition variants
            y_scram = [int(v) for v in rng.binomial(1, 0.5, size=L)]
            s_relab = [int(kc_perm[k]) for k in s_all]
            # order-shuffle: evidence segment (positions before first holdout) jointly shuffled
            first_hold = min(hold_pos) if hold_pos else L
            ev_idx = list(range(first_hold))
            ev_perm = rng.permutation(first_hold)
            s_shuf = list(s_all)
            y_shuf = list(y_all)
            for new, old in enumerate(ev_perm):
                s_shuf[new] = s_all[old]
                y_shuf[new] = y_all[old]

            conds = {
                "base": (s_all, y_all),
                "hist-scramble": (s_all, y_scram),
                "kc-relabel": (s_relab, y_all),
                "order-shuffle": (s_shuf, y_shuf),
            }
            for cname, (s_seq, s_resp) in conds.items():
                s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
                r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
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
                    cond_aucs[cname][ws.user_id] = a

        res = {}
        for cname in cond_aucs:
            m = float(np.mean(list(cond_aucs[cname].values())))
            res[cname] = {"mean": round(m, 4), "n": len(cond_aucs[cname]), "gap_vs_B1": round(m - B1_REF, 4)}
        out[mtag] = res
        print(mtag, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a17_channel_ablation.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("CHANNEL-ABLATION-DONE")


if __name__ == "__main__":
    main()
