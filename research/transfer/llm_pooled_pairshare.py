"""Batch38: pooled pair-share decomposition — WHERE does the pooled metric see
the LLM signal?

batch37: same Junyi LLM table is n.s. per-student (0.5456, p=0.065) but
significant pooled (0.6645, p=0.0149). batch28b explained per-student
compression via pair-share (cross-bucket pairs only 20% of comparable pairs).
This batch decomposes the POOLED AUC into within-student and between-student
pair contributions:

  pooled AUC = share_within * acc_within + share_between * acc_between  (exact)

where pairs are (pos row, neg row) over the global pool; "within" = both rows
from the same student. Deterministic (no permutation): the b37 null already
covers significance; this is the source decomposition for the T5 methodology
paragraph. Self-checks: pooled AUC must reproduce batch37 values exactly
(0.5321/0.5893/0.6645) and the identity must close to 1e-9.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

CACHE = "/root/unikt-fork/research/transfer/results/k3_difficulty_cache.json"
B37_POOLED_REF = {"assistments17": 0.5321, "ednet_kt1": 0.5893, "junyi2015": 0.6645}


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def main():
    diff = {ds: {int(k): v for k, v in d.items()} for ds, d in json.load(open(CACHE)).items()}
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval:]
        llm = diff.get(ds, {})

        # pooled rows with student id (same row construction as batch37)
        preds, ys, sids = [], [], []
        for s_i, ws in enumerate(eval_samples):
            hold = sorted(set(ws.holdout_idx.tolist()))
            for i in hold:
                if i == 0:
                    continue
                if is_a1(ws, i):
                    continue
                preds.append(llm.get(int(ws.sequence[i]), 0.5))
                ys.append(int(ws.response[i]))
                sids.append(s_i)
        preds = np.array(preds)
        ys = np.array(ys)
        sids = np.array(sids)

        pos_i = np.where(ys == 1)[0]
        neg_i = np.where(ys == 0)[0]
        P = preds[pos_i][:, None]
        N = preds[neg_i][None, :]
        S_P = sids[pos_i][:, None]
        S_N = sids[neg_i][None, :]
        gt = P > N
        eq = P == N
        win = gt + 0.5 * eq
        same = S_P == S_N

        n_pairs = same.size
        pooled = float(win.sum() / n_pairs)
        n_w, n_b = int(same.sum()), int((~same).sum())
        acc_w = float(win[same].sum() / n_w)
        acc_b = float(win[~same].sum() / n_b)
        share_w = n_w / n_pairs
        identity = share_w * acc_w + (1 - share_w) * acc_b

        assert abs(pooled - B37_POOLED_REF[ds]) < 5e-5, f"{ds}: pooled {pooled:.4f} != b37 {B37_POOLED_REF[ds]}"
        assert abs(pooled - identity) < 1e-9, f"{ds}: identity gap {pooled - identity}"

        out[ds] = {
            "n_rows": len(ys),
            "n_pairs": n_pairs,
            "pooled_auc": round(pooled, 4),
            "within_share": round(share_w, 4),
            "within_acc": round(acc_w, 4),
            "between_share": round(1 - share_w, 4),
            "between_acc": round(acc_b, 4),
        }
        print(ds, json.dumps(out[ds]), flush=True)

    with open("/root/unikt-fork/research/transfer/results/llm_pooled_pairshare.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("POOLEDPAIRSHARE-DONE")


if __name__ == "__main__":
    main()
