"""Alignment-module model comparison: which LLM tier to use long-term.

Runs one A09->A17 alignment per (model, effort) variant, then scores each
against the flash anchor: match rate, top-1 overlap, disagreement samples.
"""
import sys
import time

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import align_llm, load_alignment, save_alignment
from kc_tables import build_kc_table

src = build_kc_table("assistments09")
tgt = build_kc_table("assistments17")
print(f"tables: src={len(src)} tgt={len(tgt)}", flush=True)

VARIANTS = [
    ("glm", "glm-5.3", None),
    ("glm", "glm-5.3-flashx", None),
    ("kimi", "kimi-for-coding", "low"),
    ("kimi", "kimi-for-coding", "high"),
    ("kimi", "k3", "low"),
    ("kimi", "k3", "high"),
]

for provider, model, effort in VARIANTS:
    tag = f"{model}" + (f"-{effort}" if effort else "")
    name = f"assistments09__to__assistments17__llm__{tag}__c25k3__rep0"
    if load_alignment(name) is not None:
        print(f"cache hit {tag}")
        continue
    t0 = time.time()
    al, failed = align_llm(src, tgt, provider, 0, model_override=model, kimi_effort=effort)
    dt = time.time() - t0
    save_alignment(name, al, meta={"model": model, "effort": effort, "seconds": round(dt, 1), "failed_chunks": failed})
    matched = sum(1 for v in al.values() if v)
    print(f"[{tag}] matched {matched}/{len(al)} failed={len(failed)} {dt:.0f}s", flush=True)

# embedding baseline
from kc_align import align_embedding

name = "assistments09__to__assistments17__embed__bge_m3__rep0"
if load_alignment(name) is None:
    t0 = time.time()
    al = align_embedding(src, tgt)
    save_alignment(name, al, meta={"seconds": round(time.time() - t0, 1)})
matched = sum(1 for v in load_alignment(name).values() if v)
print(f"[bge_m3-embed] matched {matched}/{len(load_alignment(name))}", flush=True)
print("VARIANTS-DONE")
