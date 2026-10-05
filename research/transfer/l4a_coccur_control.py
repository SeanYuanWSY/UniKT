"""L4a-P2 control: degree-preserving KC permutation (co-occurrence channel).

Buckets KCs by transition-graph degree (total in+out edge weight over student
evidence sequences), permutes labels WITHIN buckets: each KC keeps its
connectivity magnitude but connects to different neighbours -> semantic
correspondence destroyed, first-order co-occurrence statistics approx kept.
If AUC recovers toward identity -> L4a damage is mostly co-occurrence;
if it stays ~0.6 -> semantic correspondence itself dominates.
Run on native A17 AKT (identity 0.7465, unconstrained 0.612, freq-kept 0.599).
"""
import json
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

RUN = "runs/normal/AKT_assistments17_20261003-145008_fold0_bs64"


def main(n_users=250, n_buckets=10, n_seeds=3):
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:n_users]
    rm = restore_any(RUN)

    # transition degree per KC (evidence segments)
    degree = defaultdict(float)
    for ws in samples:
        ev = ws.evidence_idx
        seq = ws.sequence[ev].tolist()
        for a, b in zip(seq[:-1], seq[1:]):
            if a != b:
                degree[int(a)] += 1
                degree[int(b)] += 1
    kcs = sorted(degree)
    degs = np.array([degree[k] for k in kcs])
    order = np.argsort(degs)
    bucket = np.zeros(len(kcs), dtype=int)
    edges = np.array_split(order, n_buckets)
    for b, idxs in enumerate(edges):
        bucket[idxs] = b

    def auc_with_perm(perm):
        aucs = []
        for ws in samples:
            if len(ws.sequence) < 6:
                continue
            seq = [perm[int(k)] for k in ws.sequence.tolist()]
            resp = [int(x) for x in ws.response.tolist()]
            s = torch.tensor([seq], dtype=torch.long, device=rm.device)
            r_ = torch.tensor([resp], dtype=torch.long, device=rm.device)
            probs = _forward_probs(rm, s, r_, None)[0]
            rows = [{"y": resp[i], "pred": float(probs[i])} for i in set(ws.holdout_idx.tolist()) if i < len(seq)]
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                aucs.append(a)
        return float(np.mean(aucs))

    base = auc_with_perm({k: k for k in kcs})
    outs = []
    for seed in range(n_seeds):
        r = np.random.default_rng(3000 + seed)
        perm = {}
        for b in range(n_buckets):
            idx = [i for i in range(len(kcs)) if bucket[i] == b]
            sh = r.permutation(idx)
            for o, nw in zip(idx, sh):
                perm[kcs[o]] = kcs[nw]
        outs.append(auc_with_perm(perm))

    res = {
        "n_kc": len(kcs),
        "n_buckets": n_buckets,
        "identity": round(base, 4),
        "degree_preserving_mean": round(float(np.mean(outs)), 4),
        "degree_preserving_shuffles": [round(x, 4) for x in outs],
        "ratio": round(float(np.mean(outs)) / base, 4),
        "reference": {"unconstrained": 0.612, "freq_preserved": 0.599},
    }
    print(json.dumps(res, ensure_ascii=False), flush=True)
    with open("/root/unikt-fork/research/transfer/results/l4a_coccur_control.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("L4A-COCCUR-DONE")


if __name__ == "__main__":
    main()
