"""IDEA-1 gate v2: retrieval on the model's functional state vector.

Embed each student by their full readiness scan over source KCs (the frozen
AKT's predicted correctness for every source KC given the student's mapped
evidence — a 123-dim functional state, no hooks needed). k=5 NN by cosine.
Gate (same as v1): neighbour KC-set Jaccard exceeds random by >= 0.05 AND
neighbour accuracy-gap smaller than random.
"""
import json
import sys

import numpy as np
import torch

np_float = np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from signals import readiness_scan  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = "assistments09__to__assistments17__llm__k3-high__c25k3__rep0"
RUN = "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64"


def main(n_users=200, k=5):
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:n_users]
    alignment = {int(kk): v for kk, v in load_alignment(ALIGN).items()}
    rm = restore_any(RUN)

    all_src_kcs = sorted({v[0]["source_kc"] for v in alignment.values() if v})
    kc_index = {kc: i for i, kc in enumerate(all_src_kcs)}
    D = len(all_src_kcs)

    from restore import WindowSample  # noqa: PLC0415

    embs, kc_sets, accs = [], [], []
    for ws in samples:
        mapping = {}
        for kc in set(int(c) for c in ws.sequence.tolist()):
            ms = alignment.get(kc, [])
            mapping[kc] = ms[0]["source_kc"] if ms else None
        keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping.get(int(kc)) is not None]
        if len(keep) < 6:
            continue
        holdout_pos = set(ws.holdout_idx.tolist())
        ev = [i for i in keep if i not in holdout_pos]
        # mapped pseudo-sample: evidence side mapped to source KC space
        pseudo = WindowSample(
            user_id=ws.user_id,
            sequence=np.array([mapping[int(ws.sequence[i])] for i in ev] + [all_src_kcs[0]]),
            response=np.array([int(ws.response[i]) for i in ev] + [0]),
            question=np.zeros(len(ev) + 1, dtype=np.int64),
            holdout_idx=np.array([len(ev)]),
        )
        try:
            read = readiness_scan(rm, pseudo, all_src_kcs, batch_size=64)
        except Exception:  # noqa: BLE001
            continue
        vec = np.array([read.get(kc, 0.5) for kc in all_src_kcs])
        embs.append(vec)
        kc_sets.append(frozenset(int(c) for c in ws.sequence[ws.evidence_idx].tolist()))
        accs.append(float(np.mean(ws.response[ws.evidence_idx])))

    E = np.array(embs)
    E = E - E.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.clip(norms, 1e-8, None)
    n = len(E)
    print(f"students embedded: {n} (dim {D})", flush=True)

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a | b else 0.0

    sim = E @ E.T
    rng = np.random.default_rng(7)
    nb_j, nb_d, rd_j, rd_d = [], [], [], []
    for i in range(n):
        order = np.argsort(-sim[i])
        for j in order[1 : k + 1]:
            nb_j.append(jaccard(kc_sets[i], kc_sets[j]))
            nb_d.append(abs(accs[i] - accs[j]))
        for j in rng.integers(0, n, 3):
            if j != i:
                rd_j.append(jaccard(kc_sets[i], kc_sets[int(j)]))
                rd_d.append(abs(accs[i] - accs[int(j)]))

    out = {
        "n": n,
        "dim": D,
        "k": k,
        "neighbour_kc_jaccard": round(float(np.mean(nb_j)), 4),
        "random_kc_jaccard": round(float(np.mean(rd_j)), 4),
        "neighbour_accdiff": round(float(np.mean(nb_d)), 4),
        "random_accdiff": round(float(np.mean(rd_d)), 4),
        "gate_jaccard_lift": round(float(np.mean(nb_j) - np.mean(rd_j)), 4),
        "gate_accdiff_drop": round(float(np.mean(rd_d) - np.mean(nb_d)), 4),
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    with open("/root/unikt-fork/research/transfer/results/idea1_hidden_gate.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("IDEA1-HIDDEN-GATE-DONE")


if __name__ == "__main__":
    main()
