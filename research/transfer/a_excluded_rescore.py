"""A-excluded rescoring: frozen-transfer deep numbers without label-copy rows.

Recomputes the headline A09->A17 deep transfer (AKT, k3-high, seeds 42/43/44)
and the B1 table on the keep scope, both WITH and WITHOUT bucket-A rows
(same-question adjacent rows from multi-KC explode). This yields the
artifact-corrected deployment comparison the paper must report.
"""
import glob
import json
import sys

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


def seed_of(run):
    for line in open(f"{run}/run_config.yaml"):
        if line.strip().startswith("seed:"):
            return int(line.split(":")[1].strip())
    return 42


def main(n_users=264):
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:n_users]
    students = build_rows(samples, ALIGN)
    prior = fit_global_kc_prior(students)
    alignment = {int(k): v for k, v in load_alignment(ALIGN).items()}

    # table rows with buckets (rows carry kc; bucket needs question context)
    rows_by_student = []
    for st, ws in zip(students, [s for s in samples if True]):
        pass  # students built from samples in order; rebuild mapping below

    # simpler: iterate samples once, keep rows aligned to build_rows order
    table_all, table_noA = [], []
    runs = sorted(glob.glob("/root/unikt-fork/runs/normal/AKT_assistments09_*_fold0_bs64"))
    out = {"runs": {}}
    for run in runs:
        seed = seed_of(run)
        rm = restore_any(run)
        deep_all, deep_noA = [], []
        a_share = 0
        total = 0
        idx = 0
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
            rows = []
            rows_noA = []
            for j, i in enumerate(keep):
                if i in holdout_pos:
                    total += 1
                    isA = i > 0 and int(ws.question[i]) == int(ws.question[i - 1])
                    if isA:
                        a_share += 1
                    row = {"y": int(ws.response[i]), "pred": float(probs[j])}
                    rows.append(row)
                    if not isA:
                        rows_noA.append(row)
            a1 = stratified_auc_rows(rows, "pred")
            a2 = stratified_auc_rows(rows_noA, "pred")
            if a1 is not None:
                deep_all.append(a1)
            if a2 is not None:
                deep_noA.append(a2)
        out["runs"][f"seed{seed}"] = {
            "deep_keep_auc": round(float(np.mean(deep_all)), 4),
            "deep_keep_Aexcluded": round(float(np.mean(deep_noA)), 4),
        }
        print(seed, out["runs"][f"seed{seed}"], flush=True)

    # table A-excluded (B1: same-KC constant predictions tie within A; recompute)
    t_all, t_noA = [], []
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
        rows, rows_noA = [], []
        for i in keep:
            if i in holdout_pos:
                isA = i > 0 and int(ws.question[i]) == int(ws.question[i - 1])
                row = {"y": int(ws.response[i]), "pred": prior.get(int(ws.sequence[i]), 0.5)}
                rows.append(row)
                if not isA:
                    rows_noA.append(row)
        a1 = stratified_auc_rows(rows, "pred")
        a2 = stratified_auc_rows(rows_noA, "pred")
        if a1 is not None:
            t_all.append(a1)
        if a2 is not None:
            t_noA.append(a2)
    out["table"] = {
        "keep_auc": round(float(np.mean(t_all)), 4),
        "keep_Aexcluded": round(float(np.mean(t_noA)), 4),
    }
    out["A_share_of_keep_holdout"] = round(a_share / max(total, 1), 4)
    print(json.dumps(out, ensure_ascii=False), flush=True)
    with open("/root/unikt-fork/research/transfer/results/a_excluded_rescore.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("AEXCLUDED-DONE")


if __name__ == "__main__":
    main()
