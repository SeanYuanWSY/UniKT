"""Batch34: full 2x2 x noise-base robustness (hardens batch33's single-draw numbers).

batch33 left two numbers on base-1000 single draws: DKT holdout-rolling loss
0.0560 and synergy +0.0274. Here all four factorial cells are computed for
each noise base {1000,2000,3000} (rng default_rng(base+uid%100000), binomial
first draw — bit-compatible with b29 y_scram at base 1000):
  (T,T) base        deterministic, must reproduce 0.5841/0.6331
  (F,T) evidence scrambled
  (T,F) holdout scrambled
  (F,F) both scrambled — base 1000 must reproduce b29 hist-scramble 0.5529/0.5398
Per-base: ev_loss = base-(F,T); hold_loss = base-(T,F); total = base-(F,F);
synergy = total - ev_loss - hold_loss (= batch33 residual per base).
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
NOISE_BASES = [1000, 2000, 3000]
B29_HIST_SCRAMBLE = {"idAKT": 0.5529, "idDKT": 0.5398}


def main(n_users=300):
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:n_users]

    out = {"b29_hist_scramble_ref": B29_HIST_SCRAMBLE}
    for mtag, run in MODELS.items():
        rm = restore_any(run)
        cells = {c: {b: {} for b in NOISE_BASES} for c in ("TT", "FT", "TF", "FF")}
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

            variants = {"TT": {b: list(y_all) for b in NOISE_BASES}}
            variants["FT"] = {b: [noises[b][i] if i < first_hold else y_all[i] for i in range(L)] for b in NOISE_BASES}
            variants["TF"] = {b: [y_all[i] if i < first_hold else noises[b][i] for i in range(L)] for b in NOISE_BASES}
            variants["FF"] = {b: list(noises[b]) for b in NOISE_BASES}

            for cname in ("TT", "FT", "TF", "FF"):
                bases_here = [NOISE_BASES[0]] if cname == "TT" else NOISE_BASES
                for base in bases_here:
                    y_cond = variants[cname][base]
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
                        if cname == "TT":
                            for b2 in NOISE_BASES:
                                cells[cname][b2][uid] = a
                        else:
                            cells[cname][base][uid] = a

        res = {}
        for b in NOISE_BASES:
            m = {c: float(np.mean(list(cells[c][b].values()))) for c in cells}
            res[f"base{b}"] = {
                "TT": round(m["TT"], 4),
                "FT": round(m["FT"], 4),
                "TF": round(m["TF"], 4),
                "FF": round(m["FF"], 4),
                "ev_loss": round(m["TT"] - m["FT"], 4),
                "hold_loss": round(m["TT"] - m["TF"], 4),
                "total_loss": round(m["TT"] - m["FF"], 4),
                "synergy": round((m["TT"] - m["FF"]) - (m["TT"] - m["FT"]) - (m["TT"] - m["TF"]), 4),
            }
        res["n"] = len(cells["TT"][1000])
        out[mtag] = res
        print(mtag, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a17_factorial_noise2.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("FACTORIAL2-DONE")


if __name__ == "__main__":
    main()
