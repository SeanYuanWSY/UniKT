"""A17 final verdict: table vs deep transfer, FULL sample, both scoring scopes.

Scopes (preregistered, both reported):
- keep-scope: only steps the deep model can score (alignment keep set) — the
  fair-to-deep comparison (Tree-3 protocol).
- full-scope: all holdout steps — the deployment view (deep missing steps get
  its global mean prediction, i.e. no information).
Paired per-student AUC differences + cluster bootstrap CIs (B=5000).
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
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = "assistments09__to__assistments17__llm__k3-high__c25k3__rep0"
RUN = "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64"


def paired_boot(d, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = [np.mean(d[rng.integers(0, n, n)]) for _ in range(B)]
    return round(float(np.mean(d)), 4), [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def main():
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)  # FULL sample
    print(f"full sample: {len(samples)} students", flush=True)
    alignment = {int(k): v for k, v in load_alignment(ALIGN).items()}
    rm = restore_any(RUN)

    students_full = build_rows(samples, None)
    prior = fit_global_kc_prior(students_full)

    per_student = {}
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
        seq_m = [mapping[int(ws.sequence[i])] for i in keep]
        resp_m = [int(ws.response[i]) for i in keep]
        s = torch.tensor([seq_m], dtype=torch.long, device=rm.device)
        r_ = torch.tensor([resp_m], dtype=torch.long, device=rm.device)
        probs = _forward_probs(rm, s, r_, None)[0]
        pmap = {i: float(probs[j]) for j, i in enumerate(keep)}
        all_hold = [i for i in range(len(ws.sequence)) if i in holdout_pos]
        keep_rows = [{"y": int(ws.response[i]), "pred": pmap[i]} for i in all_hold if i in pmap]
        full_rows = [
            {"y": int(ws.response[i]), "pred": pmap.get(i, 0.5)} for i in all_hold
        ]  # deep missing -> 0.5 (no info)
        tab_rows = [{"y": int(ws.response[i]), "pred": prior.get(int(ws.sequence[i]), 0.5)} for i in all_hold]
        rec = {}
        for name, rows in (("keep", keep_rows), ("full", full_rows)):
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                rec[f"deep_{name}"] = a
        a = stratified_auc_rows(tab_rows, "pred")
        if a is not None:
            rec["table_full"] = a
            # table restricted to keep rows (fair-to-deep scope)
            tr = [{"y": int(ws.response[i]), "pred": prior.get(int(ws.sequence[i]), 0.5)} for i in all_hold if i in pmap]
            a2 = stratified_auc_rows(tr, "pred")
            if a2 is not None:
                rec["table_keep"] = a2
        if "deep_keep" in rec and "table_keep" in rec or "deep_full" in rec and "table_full" in rec:
            per_student[ws.user_id] = rec

    out = {"n": len(per_student)}
    for scope in ("keep", "full"):
        d = np.array(
            [r[f"table_{scope}"] - r[f"deep_{scope}"] for r in per_student.values() if f"table_{scope}" in r and f"deep_{scope}" in r]
        )
        m = np.mean([r[f"table_{scope}"] for r in per_student.values() if f"table_{scope}" in r])
        md = np.mean([r[f"deep_{scope}"] for r in per_student.values() if f"deep_{scope}" in r])
        point, ci = paired_boot(d)
        out[f"{scope}_scope"] = {
            "table_mean": round(float(m), 4),
            "deep_mean": round(float(md), 4),
            "paired_delta(table-deep)": point,
            "ci95": ci,
            "verdict": "table" if ci[0] > 0 else ("deep" if ci[1] < 0 else "tie"),
        }
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    with open("/root/unikt-fork/research/transfer/results/a17_final_verdict.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("A17-VERDICT-DONE")


if __name__ == "__main__":
    main()
