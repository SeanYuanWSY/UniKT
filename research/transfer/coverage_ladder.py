"""Coverage-controlled alignment experiments (review-mandated).

Ladder A (quality constant, coverage varies): restrict the k3-high alignment
to its top-K scored target KCs, K in {44, 60, 75} (+full 93 as existing cell)
-> isolates the COVERAGE effect with alignment quality held constant.

Ladder B (coverage constant = flash's 44-KC set, quality varies):
  B1 k3-high restricted to exactly flash's matched set (best assignments)
  B2 flash's own assignments (existing cell)
  B3 random within the same set (occupancy null, B=399)
-> isolates the QUALITY effect with coverage held constant.

Writes restricted alignment caches, then the caller runs transfer_eval on each.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from kc_align import load_alignment, save_alignment  # noqa: E402

CACHE = _HERE / "align_cache"

k3 = {int(k): v for k, v in load_alignment("assistments09__to__assistments17__llm__k3-high__c25k3__rep0").items()}
flash = {int(k): v for k, v in load_alignment("assistments09__to__assistments17__llm__glm__rep0").items()}

# --- Ladder A: restrict k3-high to top-K by score ---
matched = [(t, ms[0]["score"]) for t, ms in k3.items() if ms]
matched.sort(key=lambda x: -x[1])
for K in (44, 60, 75):
    keep = {t for t, _ in matched[:K]}
    restricted = {t: (k3[t] if t in keep else []) for t in k3}
    name = f"assistments09__to__assistments17__ladderA_k3high_top{K}"
    save_alignment(name, restricted, meta={"ladder": "A", "K": K})
    m = sum(1 for v in restricted.values() if v)
    print(f"[{name}] matched {m}/{len(restricted)}")

# --- Ladder B: coverage = flash's 44-set, quality varies ---
flash_set = {t for t, ms in flash.items() if ms}
b1 = {t: (k3[t] if t in flash_set else []) for t in k3}
save_alignment("assistments09__to__assistments17__ladderB_k3high_on_flashset", b1, meta={"ladder": "B", "set": "flash44"})
print(f"[ladderB k3-on-flashset] matched {sum(1 for v in b1.values() if v)}/{len(b1)}")

# B3: random assignment within the same set (seeded; transfer_eval's own
# permutation can also serve, but a fixed random cell documents the null)
rng = random.Random(4242)
sources = [b1[t][0]["source_kc"] for t in sorted(flash_set) if b1.get(t)]
rng.shuffle(sources)
b3 = {t: [] for t in k3}
for t, s in zip(sorted(flash_set), sources):
    b3[t] = [{"source_kc": s, "score": 1.0, "relation": "random"}]
save_alignment("assistments09__to__assistments17__ladderB_random_on_flashset", b3, meta={"ladder": "B-random", "seed": 4242})
print(f"[ladderB random-on-flashset] matched {sum(1 for v in b3.values() if v)}/{len(b3)}")
print("LADDER-CACHES-DONE")
