"""IDEA-1 gate: does the frozen transfer model's representation support retrieval?

Embed each target student by the frozen A09->A17 AKT's final evidence-step
hidden state (approximated by the model's output prob vector over the mapped
sequence — a cheap proxy; if the gate fails here we upgrade to true hidden
state). k=5 nearest neighbours vs random students: compare mean Jaccard of
KC sets and mean |accuracy difference|. Gate: neighbour-Jaccard > random by
>= 0.05 AND neighbour |acc diff| < random |acc diff| (similar students are
retrievable) -> retrieval augmentation has a basis.
"""
import json
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = "assistments09__to__assistments17__llm__k3-high__c25k3__rep0"
RUN = "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64"


def main(n_users=250, k=5):
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:n_users]
    alignment = {int(kk): v for kk, v in load_alignment(ALIGN).items()}
    rm = restore_any(RUN)

    embs, kc_sets, accs = [], [], []
    for ws in samples:
        mapping = {}
        for kc in set(int(c) for c in ws.sequence.tolist()):
            ms = alignment.get(kc, [])
            mapping[kc] = ms[0]["source_kc"] if ms else None
        keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping.get(int(kc)) is not None]
        if len(keep) < 6:
            continue
        ev = [i for i in keep if i not in set(ws.holdout_idx.tolist())]
        if not ev:
            continue
        seq_m = [mapping[int(ws.sequence[i])] for i in ev]
        resp_m = [int(ws.response[i]) for i in ev]
        s = torch.tensor([seq_m], dtype=torch.long, device=rm.device)
        r_ = torch.tensor([resp_m], dtype=torch.long, device=rm.device)
        # embedding proxy: mean output prob over the last 10 evidence steps
        probs = _forward_probs(rm, s, r_, None)[0][-10:]
        embs.append(float(np.mean(probs)))
        kc_sets.append(frozenset(int(c) for c in ws.sequence[ws.evidence_idx].tolist()))
        accs.append(float(np.mean(ws.response[ws.evidence_idx])))

    embs = np.array(embs)
    n = len(embs)
    print(f"students embedded: {n}", flush=True)

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a | b else 0.0

    rng = np.random.default_rng(7)
    nb_j, nb_d, rd_j, rd_d = [], [], [], []
    order = np.argsort(embs)
    for i in range(n):
        # neighbours: closest embeddings (1-D proxy -> nearest by |e_i - e_j|)
        pos = int(np.searchsorted(embs[order], embs[i]))
        cand = [order[j] for j in range(max(0, pos - k), min(n, pos + k + 1)) if order[j] != i][:k]
        for j in cand:
            nb_j.append(jaccard(kc_sets[i], kc_sets[j]))
            nb_d.append(abs(accs[i] - accs[j]))
        for j in rng.integers(0, n, 3):
            if j != i:
                rd_j.append(jaccard(kc_sets[i], kc_sets[int(j)]))
                rd_d.append(abs(accs[i] - accs[int(j)]))

    out = {
        "n": n,
        "k": k,
        "neighbour_kc_jaccard": round(float(np.mean(nb_j)), 4),
        "random_kc_jaccard": round(float(np.mean(rd_j)), 4),
        "neighbour_accdiff": round(float(np.mean(nb_d)), 4),
        "random_accdiff": round(float(np.mean(rd_d)), 4),
        "gate_jaccard_lift": round(float(np.mean(nb_j) - np.mean(rd_j)), 4),
        "gate_accdiff_drop": round(float(np.mean(rd_d) - np.mean(nb_d)), 4),
        "note": "1-D embedding proxy (mean prob); upgrade to true hidden state if borderline",
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    with open("/root/unikt-fork/research/transfer/results/idea1_gate.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("IDEA1-GATE-DONE")


if __name__ == "__main__":
    main()
