"""Agreement analysis across alignment methods (model comparison follow-up)."""
import itertools
import json
import sys

sys.path.insert(0, "/root/unikt-fork/research/transfer")

from kc_align import load_alignment
from kc_tables import build_kc_table

tgt = build_kc_table("assistments17")
src = build_kc_table("assistments09")

NAMES = {
    "flash": "assistments09__to__assistments17__llm__glm__rep0",
    "glm-5.3": "assistments09__to__assistments17__llm__glm-5.3__c25k3__rep0",
    "flashx": "assistments09__to__assistments17__llm__glm-5.3-flashx__c25k3__rep0",
    "k2.8-low": "assistments09__to__assistments17__llm__kimi-for-coding-low__c25k3__rep0",
    "k2.8-high": "assistments09__to__assistments17__llm__kimi-for-coding-high__c25k3__rep0",
    "k3-low": "assistments09__to__assistments17__llm__k3-low__c25k3__rep0",
    "k3-high": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
    "embed": "assistments09__to__assistments17__embed__bge_m3__rep0",
    "edit": "assistments09__to__assistments17__edit__c25k3__rep0",
}

als = {}
for k, n in NAMES.items():
    a = load_alignment(n)
    if a is not None:
        als[k] = {int(t): v for t, v in a.items()}
print("loaded:", list(als))

print(f"\n{'pair':28s} {'|matched∩|':>10s} {'top1 agree':>10s} {'jaccard':>8s}")
for a, b in itertools.combinations(als, 2):
    A, B = als[a], als[b]
    ma = {t for t, v in A.items() if v}
    mb = {t for t, v in B.items() if v}
    common = ma & mb
    agree = sum(1 for t in common if A[t][0]["source_kc"] == B[t][0]["source_kc"])
    jac = len(ma & mb) / len(ma | mb) if ma | mb else 0
    print(f"{a:14s} x {b:13s} {len(common):>10d} {agree/max(len(common),1):>10.2%} {jac:>8.2f}")

# disagreement samples between the two biggest matchers
a, b = "k3-high", "glm-5.3"
if a in als and b in als:
    A, B = als[a], als[b]
    n = 0
    print(f"\n--- {a} vs {b} top-1 disagreements ---")
    for t in sorted(A):
        if A.get(t) and B.get(t) and A[t][0]["source_kc"] != B[t][0]["source_kc"]:
            print(f"  KC{t}({tgt[t][:28]}): {a}->{src[A[t][0]['source_kc']][:24]} vs {b}->{src[B[t][0]['source_kc']][:24]}")
            n += 1
            if n >= 10:
                break
