"""L4a: within-domain KC-identity shuffle probe (mechanism layer, tree 4).

Question: does a frozen native KT model actually use KC semantics, or is it
a recency machine? Shuffle KC ids (frequency-preserving random re-labeling)
on the NATIVE dataset and feed the NATIVE frozen model.

Pre-registered readout (designs_3trees_v2.md):
- AUC(shuffled) >= 0.95 * AUC(identity)  -> model ignores KC semantics ->
  C2 (alignment is second-order) gains a mechanistic explanation.
- AUC(shuffled) < 0.85 * AUC(identity)   -> model uses semantics; story pivots.
- Sensitivity control: response shuffle must collapse AUC to ~0.5.
5 permutation seeds; A17 + EdNet + Junyi native DKT/AKT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "llm_explain"))

from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402


def auc_under_perm(rm, samples, kc_perm: dict[int, int] | None, resp_perm=None) -> float:
    aucs = []
    for ws in samples:
        if len(ws.sequence) < 6:
            continue
        seq = ws.sequence.tolist()
        resp = [int(x) for x in ws.response.tolist()]
        if kc_perm:
            seq = [kc_perm[int(k)] for k in seq]
        if resp_perm is not None:
            resp = [resp_perm[i % len(resp_perm)] for i in range(len(resp))]
        s = torch.tensor([seq], dtype=torch.long, device=rm.device)
        r = torch.tensor([resp], dtype=torch.long, device=rm.device)
        probs = _forward_probs(rm, s, r, None)[0]
        holdout_pos = set(ws.holdout_idx.tolist())
        rows = [
            {"y": resp[i], "pred": float(probs[i])} for i in range(len(seq)) if i in holdout_pos
        ]
        a = stratified_auc_rows(rows, "pred")
        if a is not None:
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-dkt", required=True)
    ap.add_argument("--run-akt", required=True)
    ap.add_argument("--n-users", type=int, default=250)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = get_data_source(_make_rc(args.dataset))
    samples = load_user_samples(src, fold=0)[: args.n_users]
    all_kcs = sorted({int(c) for ws in samples for c in ws.sequence})
    rng_global = np.random.default_rng(42)

    results = {"dataset": args.dataset, "n_users": len(samples), "n_kc": len(all_kcs)}
    for run, tag in [(args.run_dkt, "DKT"), (args.run_akt, "AKT")]:
        rm = restore_any(run)
        base = auc_under_perm(rm, samples, None)
        shuffles = []
        for seed in range(5):
            perm_ids = rng_global.permutation(len(all_kcs))
            kc_perm = {kc: all_kcs[perm_ids[i]] for i, kc in enumerate(all_kcs)}
            shuffles.append(auc_under_perm(rm, samples, kc_perm))
        # sensitivity control: response shuffle (fixed seed 1)
        resp_perm = rng_global.permutation([0, 1] * 4000).tolist()
        resp_ctrl = auc_under_perm(rm, samples, None, resp_perm)
        results[tag] = {
            "identity_auc": round(base, 4),
            "kc_shuffle_auc_mean": round(float(np.mean(shuffles)), 4),
            "kc_shuffle_auc_std": round(float(np.std(shuffles)), 4),
            "shuffle_shuffles": [round(x, 4) for x in shuffles],
            "response_shuffle_auc": round(resp_ctrl, 4),
            "ratio": round(float(np.mean(shuffles)) / base, 4) if base else None,
        }
        print(tag, json.dumps(results[tag]), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("KC-SHUFFLE-PROBE-DONE")


if __name__ == "__main__":
    main()
