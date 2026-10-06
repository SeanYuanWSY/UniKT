"""Decompose the T3b-vs-T1 deep-reference gap (0.272 vs 0.428 on EdNet).

T1 (t1_paired_ci.json) transfer_AKT = single seed-42 checkpoint, first-300
cohort, paired with B1 (EdNet n=56). T3b (poolsize_a1.json) deep ref =
3-seed mean (42/43/44), last-100 tail cohort. Same model family (AKT
trained on assistments09), so the gap must come from seed variance and/or
cohort selection. This script measures transfer AKT per-seed on BOTH
cohorts for A17 + EdNet, A1-excluded per-student AUC, and reproduces the
t1ci pairing (B1 intersection) on the first-300 cohort.
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

ALIGN = {
    "assistments17": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
}
RUNS = sorted(glob.glob("/root/unikt-fork/runs/normal/AKT_assistments09_*_fold0_bs64"))


def seed_of(run):
    for line in open(f"{run}/run_config.yaml"):
        if line.strip().startswith("seed:"):
            return int(line.split(":")[1].strip())
    return 42


def transfer_auc(rm, samples_eval, alignment):
    """Per-student A1-excluded AUC for frozen transfer; returns {uid: auc}."""
    out = {}
    for ws in samples_eval:
        mapping = {}
        for kc in set(int(c) for c in ws.sequence.tolist()):
            ms = alignment.get(kc, [])
            mapping[kc] = ms[0]["source_kc"] if ms else None
        keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping.get(int(kc)) is not None]
        if len(keep) > 200:
            keep = keep[-200:]
        if len(keep) < 6:
            continue
        s_seq = [mapping[int(ws.sequence[i])] for i in keep]
        s_resp = [int(ws.response[i]) for i in keep]
        jmap = {i: j for j, i in enumerate(keep)}
        hold_pos = set(ws.holdout_idx.tolist())
        s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
        r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
        probs = _forward_probs(rm, s, r_, None)[0]
        rows = []
        for i in sorted(hold_pos):
            if i in jmap and i > 0:
                isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                if not isA1:
                    rows.append({"y": int(ws.response[i]), "pred": float(probs[jmap[i]])})
        a = stratified_auc_rows(rows, "pred")
        if a is not None:
            out[ws.user_id] = a
    return out


def main(n_users=300):
    models = [(seed_of(r), restore_any(r)) for r in RUNS]
    print("seeds:", [s for s, _ in models], flush=True)
    out = {}
    for ds, align_name in ALIGN.items():
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        first300 = samples[:n_users]
        n_eval = min(100, max(10, len(samples) - 60))
        tail100 = samples[len(samples) - n_eval:]
        alignment = {int(k): v for k, v in load_alignment(align_name).items()}

        # B1 on first-300 for t1ci-style pairing
        students = build_rows(first300, None)
        prior = fit_global_kc_prior(students)
        b1 = {}
        for ws in first300:
            hold = sorted(set(ws.holdout_idx.tolist()))
            rows = []
            for i in hold:
                if i == 0:
                    continue
                isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                if not isA1:
                    rows.append({"y": int(ws.response[i]), "pred": prior.get(int(ws.sequence[i]), 0.5)})
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                b1[ws.user_id] = a

        res = {"tail100_n_students": len(tail100), "B1_first300": round(float(np.mean(list(b1.values()))), 4), "B1_n": len(b1)}
        for cohort_name, cohort in [("first300_paired", first300), ("tail100", tail100)]:
            per_seed = {}
            for seed, rm in models:
                aucs = transfer_auc(rm, cohort, alignment)
                if cohort_name == "first300_paired":
                    common = [u for u in b1 if u in aucs]
                    mean = round(float(np.mean([aucs[u] for u in common])), 4) if common else None
                    per_seed[f"seed{seed}"] = {"auc": mean, "n": len(common)}
                else:
                    mean = round(float(np.mean(list(aucs.values()))), 4) if aucs else None
                    per_seed[f"seed{seed}"] = {"auc": mean, "n": len(aucs)}
            seed_means = [v["auc"] for v in per_seed.values() if v["auc"] is not None]
            per_seed["mean_over_seeds"] = round(float(np.mean(seed_means)), 4) if seed_means else None
            res[cohort_name] = per_seed
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/gap_decompose.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("GAP-DECOMPOSE-DONE")


if __name__ == "__main__":
    main()
