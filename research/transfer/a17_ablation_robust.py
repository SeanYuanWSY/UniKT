"""Batch35: robustness of batch29's remaining ablation conditions (kc-relabel, order-shuffle).

b34 hardened hist-scramble (FF cell x 3 bases). Two b29 conditions still rest
on single permutation draws: kc-relabel (global KC permutation rng(7)) and
order-shuffle (per-student evidence permutation, drawn after the binomial in
rng(1000+uid)). Here each is rerun with 3 draw seeds; base1000/rng(7) must
reproduce b29 exactly (0.4858/0.4846 kc-relabel; 0.5730/0.6147 order-shuffle).
Seeds: kc_perm via default_rng(7/8/9); order via per-student
default_rng(base+uid%100000) with the same consumption order as b29 (binomial
first — discarded — then permutation), base in {1000,2000,3000}.
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
KC_PERM_SEEDS = [7, 8, 9]
ORDER_BASES = [1000, 2000, 3000]
B29_REF = {"kc-relabel": {"idAKT": 0.4858, "idDKT": 0.4846}, "order-shuffle": {"idAKT": 0.5730, "idDKT": 0.6147}}


def main(n_users=300):
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:n_users]
    max_kc = max(int(c) for ws in samples for c in ws.sequence.tolist())
    kc_perms = {s: np.random.default_rng(s).permutation(max_kc + 1).tolist() for s in KC_PERM_SEEDS}

    out = {"b29_ref": B29_REF}
    for mtag, run in MODELS.items():
        rm = restore_any(run)
        res = {"kc-relabel": {}, "order-shuffle": {}}
        accs = {"kc-relabel": {s: {} for s in KC_PERM_SEEDS}, "order-shuffle": {b: {} for b in ORDER_BASES}}
        for ws in samples:
            s_all = [int(x) for x in ws.sequence.tolist()]
            y_all = [int(x) for x in ws.response.tolist()]
            hold = sorted(set(ws.holdout_idx.tolist()))
            hold_pos = set(hold)
            L = len(s_all)
            first_hold = min(hold_pos) if hold_pos else L
            uid = ws.user_id

            variants = {}
            for s in KC_PERM_SEEDS:
                p = kc_perms[s]
                variants[("kc-relabel", s)] = ([p[k] for k in s_all], list(y_all))
            for base in ORDER_BASES:
                rng = np.random.default_rng(base + uid % 100000)
                _ = rng.binomial(1, 0.5, size=L)  # same consumption order as b29 (y_scram), discarded
                ev_perm = rng.permutation(first_hold)
                s_shuf = list(s_all)
                y_shuf = list(y_all)
                for new, old in enumerate(ev_perm):  # joint (kc,y) shuffle, b29 semantics
                    s_shuf[new] = s_all[old]
                    y_shuf[new] = y_all[old]
                variants[("order-shuffle", base)] = (s_shuf, y_shuf)

            for (cname, tag), (s_seq, s_resp) in variants.items():
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
                    accs[cname][tag][uid] = a

        for cname in ("kc-relabel", "order-shuffle"):
            for tag in accs[cname]:
                m = float(np.mean(list(accs[cname][tag].values())))
                key = f"seed{tag}" if cname == "kc-relabel" else f"base{tag}"
                res[cname][key] = {"mean": round(m, 4), "n": len(accs[cname][tag])}
        out[mtag] = res
        print(mtag, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a17_ablation_robust.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("ABLROBUST-DONE")


if __name__ == "__main__":
    main()
