"""Tree-1: artificial history-deletion dosage curve (pure causal coverage probe).

On the A17 hard-mapping (k3-high) cell: randomly delete X% of EVIDENCE steps
(keep all holdout steps) before the frozen forward, X in {0, 20, 50, 80}.
X=0 must reproduce the existing cell numbers (0.486/0.532) — built-in parity
check. 3 deletion seeds per level.
"""
import sys

import numpy as np
import torch

ALIGN = "assistments09__to__assistments17__llm__k3-high__c25k3__rep0"
sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

alignment = {int(k): v for k, v in load_alignment(ALIGN).items()}
tgt = get_data_source(_make_rc("assistments17"))
samples = load_user_samples(tgt, fold=0)[:264]

RUNS = [
    ("DKT", "runs/normal/DKT_assistments09_20261003-024700_fold0_bs128"),
    ("AKT", "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64"),
]

for tag, run in RUNS:
    rm = restore_any(run)
    for X in (0, 20, 50, 80):
        aucs_all = []
        for seed in (0, 1, 2):
            rng = np.random.default_rng(1000 + seed)
            aucs = []
            for ws in samples:
                mapping = {}
                for kc in set(int(c) for c in ws.sequence.tolist()):
                    ms = alignment.get(kc, [])
                    mapping[kc] = ms[0]["source_kc"] if ms else None
                keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping.get(int(kc)) is not None]
                if len(keep) > 200:
                    keep = keep[-200:]
                if len(keep) < 6:
                    continue
                holdout_pos = set(ws.holdout_idx.tolist())
                ev_steps = [i for i in keep if i not in holdout_pos]
                if X > 0:
                    n_del = int(len(ev_steps) * X / 100)
                    dropped = set(rng.choice(ev_steps, size=n_del, replace=False).tolist())
                else:
                    dropped = set()
                kept = [i for i in keep if i not in dropped]
                ev_len = sum(1 for i in kept if i not in holdout_pos)
                if ev_len < 3:
                    continue
                seq_m = [mapping[int(ws.sequence[i])] for i in kept]
                resp_m = [int(ws.response[i]) for i in kept]
                s = torch.tensor([seq_m], dtype=torch.long, device=rm.device)
                r_ = torch.tensor([resp_m], dtype=torch.long, device=rm.device)
                probs = _forward_probs(rm, s, r_, None)[0]
                rows = [
                    {"y": int(ws.response[i]), "pred": float(probs[j])}
                    for j, i in enumerate(kept)
                    if i in holdout_pos
                ]
                if rows and 0 < sum(x["y"] for x in rows) < len(rows):
                    a = stratified_auc_rows(rows, "pred")
                    if a is not None:
                        aucs.append(a)
            aucs_all.append(float(np.mean(aucs)) if aucs else float("nan"))
        print(f"{tag} X={X:2d}%: auc={np.mean(aucs_all):.4f} (seeds {[round(a,4) for a in aucs_all]})", flush=True)
print("DOSAGE-DONE")
