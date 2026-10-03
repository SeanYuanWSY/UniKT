"""Within-run label autocorrelation: genuine A09 blocks vs re-blocked A17 runs.

If genuine same-KC runs have highly autocorrelated labels (adjacent attempts,
stable ability) while re-blocked runs (assembled by dropping interleaved rows)
have low/negative autocorrelation, the transplanted model's within-run
momentum modulation inverts -- the final mechanistic link.
"""
import sys

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment
from kc_tables import _make_rc
from restore import load_user_samples
from utils.data_process import get_data_source


def run_label_autocorr(seq, resp):
    """Mean lag-1 label agreement within same-KC runs (weighted by run length-1)."""
    agree, total = 0, 0
    cur_start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[cur_start]:
            run = resp[cur_start:i]
            for a, b in zip(run[:-1], run[1:]):
                agree += int(a == b)
                total += 1
            cur_start = i
    return agree / total if total else None


alignment = {int(k): v for k, v in load_alignment("assistments09__to__assistments17__llm__glm__rep0").items()}

# A09 genuine runs (evidence segment)
src9 = get_data_source(_make_rc("assistments09"))
s9 = load_user_samples(src9, fold=0)[:300]
a9 = [run_label_autocorr(ws.sequence[ws.evidence_idx].tolist(), ws.response[ws.evidence_idx].tolist()) for ws in s9]
a9 = [x for x in a9 if x is not None]
print(f"A09 genuine runs   : within-run label agreement = {np.mean(a9):.3f} (n={len(a9)})")

# A17 native evidence (genuine, interleaved -> few runs but real)
tgt = get_data_source(_make_rc("assistments17"))
s17 = load_user_samples(tgt, fold=0)[:300]
a17n = [run_label_autocorr(ws.sequence[ws.evidence_idx].tolist(), ws.response[ws.evidence_idx].tolist()) for ws in s17]
a17n = [x for x in a17n if x is not None]
print(f"A17 genuine runs   : within-run label agreement = {np.mean(a17n):.3f} (n={len(a17n)})")

# A17 re-blocked (kept steps after mapping, evidence-side approximation: full kept seq)
mapping = {}
for ws in s17:
    for kc in set(int(c) for c in ws.sequence.tolist()):
        ms = alignment.get(kc, [])
        mapping[kc] = ms[0]["source_kc"] if ms else None
vals = []
for ws in s17:
    keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping.get(int(kc))]
    if len(keep) < 6:
        continue
    seq_m = [mapping[int(ws.sequence[i])] for i in keep]
    resp_m = [int(ws.response[i]) for i in keep]
    v = run_label_autocorr(seq_m, resp_m)
    if v is not None:
        vals.append(v)
print(f"A17 re-blocked runs: within-run label agreement = {np.mean(vals):.3f} (n={len(vals)})")
