"""Batch36: degenerate copy predictor on the A1-only set — residual teacher-forcing leakage bound.

Kimi final review (kimi_full_review_20261006.md line 70) demands: train/eval a
"copy y_{t-1}" degenerate predictor on the A1-only subset and report its AUC;
if significantly >0.5, either delete boundary pairs and re-report T1, or bound
the residual leakage contribution. This file runs the attack empirically per
domain: for every KEPT (A1-only) row i, pred = y[i-1] from the RAW expanded
sequence (teacher forcing sees the physically preceding label regardless of
scoring-set membership — the honest version of the attack).
Predictors: copy_prev (main attack), copy_prev2 (decay), copy_prev_kept
(table-side variant). Positivity controls: same attack on the withA set
(artifact direction) and accuracy on the A1-removed rows (expansion blocks,
should be ~1.0). Binary preds; stratified per-student AUC, same protocol.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def main():
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)

        kept_aucs = {"copy_prev": [], "copy_prev2": [], "copy_prev_kept": []}
        witha_aucs = []
        kept_eq, kept_n = 0, 0
        removed_eq, removed_n = 0, 0
        ys_all = []
        for ws in samples:
            y = [int(v) for v in ws.response.tolist()]
            hold = sorted(set(ws.holdout_idx.tolist()))
            ys_all.extend(y)
            kept_idx = [i for i in hold if i > 0 and not is_a1(ws, i)]

            # main attack: pred = y[i-1] (raw preceding label), scored on kept rows
            for pname, lag in (("copy_prev", 1), ("copy_prev2", 2)):
                rows = [{"y": y[i], "pred": y[i - lag]} for i in kept_idx if i >= lag]
                a = stratified_auc_rows(rows, "pred")
                if a is not None:
                    kept_aucs[pname].append(a)
            rows = [{"y": y[i], "pred": y[prev_kept]} for prev_kept, i in zip([None] + kept_idx[:-1], kept_idx) if prev_kept is not None and i > prev_kept]
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                kept_aucs["copy_prev_kept"].append(a)

            # positivity control: same attack scored on ALL holdout rows (withA)
            rows = [{"y": y[i], "pred": y[i - 1]} for i in hold if i > 0]
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                witha_aucs.append(a)

            # accuracies: kept adjacency equality vs removed-row equality
            for i in hold:
                if i == 0:
                    continue
                if is_a1(ws, i):
                    removed_n += 1
                    removed_eq += int(y[i] == y[i - 1])
                else:
                    kept_n += 1
                    kept_eq += int(y[i] == y[i - 1])

        base = max(np.mean(ys_all), 1 - np.mean(ys_all))
        out[ds] = {
            "n_students": len(samples),
            "copy_prev_auc_a1only": round(float(np.mean(kept_aucs["copy_prev"])), 4),
            "copy_prev2_auc_a1only": round(float(np.mean(kept_aucs["copy_prev2"])), 4),
            "copy_prev_kept_auc_a1only": round(float(np.mean(kept_aucs["copy_prev_kept"])), 4),
            "copy_prev_auc_withA": round(float(np.mean(witha_aucs)), 4),
            "kept_adj_eq_rate": round(kept_eq / max(kept_n, 1), 4),
            "removed_adj_eq_rate": round(removed_eq / max(removed_n, 1), 4),
            "base_rate_maxp": round(float(base), 4),
        }
        print(ds, json.dumps(out[ds]), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a1_copy_leakage.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("COPYLEAK-DONE")


if __name__ == "__main__":
    main()
