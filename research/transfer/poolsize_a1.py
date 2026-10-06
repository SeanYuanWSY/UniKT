"""POOL-SIZE under the A1-only-excluded scope (artifact-corrected crossover).

Rerun the pool-size scan with A1 rows (same-question adjacent, KC differs)
excluded from scoring. Table prior still built from full evidence (table
immune; only scoring rows filtered). Deep reference = A1-excluded transfer
(AKT seeds 42/43/44, from a_bucket_kc_split seed rows) recomputed here on
the same eval tail. Domains: A17 + EdNet (Junyi unaffected: no explode).
"""
import glob
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
from baseline_eval import build_rows  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = {
    "assistments17": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
}
RUNS = sorted(glob.glob("/root/unikt-fork/runs/normal/AKT_assistments09_*_fold0_bs64"))
POOL_SIZES = [10, 30, 100]


def seed_of(run):
    for line in open(f"{run}/run_config.yaml"):
        if line.strip().startswith("seed:"):
            return int(line.split(":")[1].strip())
    return 42


def main():
    out = {}
    for ds, align_name in ALIGN.items():
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval_actual = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval_actual:]
        pool_samples = samples[: len(samples) - n_eval_actual]
        alignment = {int(k): v for k, v in load_alignment(align_name).items()}

        pool_students = build_rows(pool_samples, None)
        pool_hists = [st["kc_hist"] for st in pool_students]

        # deep reference (A1-excluded) on eval tail, mean over seeds
        seed_aucs = []
        models = [(seed_of(r), restore_any(r)) for r in RUNS]
        for seed, rm in models:
            aucs = []
            for ws in eval_samples:
                mapping = {}
                for kc in set(int(c) for c in ws.sequence.tolist()):
                    ms = alignment.get(kc, [])
                    mapping[kc] = ms[0]["source_kc"] if ms else None
                keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping.get(int(kc)) is not None]
                if len(keep) > 200:
                    keep = keep[-200:]
                if len(keep) < 6:
                    continue
                hold_pos = set(ws.holdout_idx.tolist())
                s_seq = [mapping[int(ws.sequence[i])] for i in keep]
                s_resp = [int(ws.response[i]) for i in keep]
                s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
                r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
                probs = _forward_probs(rm, s, r_, None)[0]
                jmap = {i: j for j, i in enumerate(keep)}
                rows = []
                for i in sorted(hold_pos):
                    if i in jmap and i > 0:
                        isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                        if not isA1:
                            rows.append({"y": int(ws.response[i]), "pred": float(probs[jmap[i]])})
                a = stratified_auc_rows(rows, "pred")
                if a is not None:
                    aucs.append(a)
            if aucs:
                seed_aucs.append(float(np.mean(aucs)))
        deep_ref = round(float(np.mean(seed_aucs)), 4) if seed_aucs else None

        res = {"n_eval": len(eval_samples), "deep_A1excluded_mean": deep_ref}
        for n_pool in POOL_SIZES:
            draws = []
            for seed in range(10):
                r = np.random.default_rng(1000 + seed)
                idx = r.choice(len(pool_hists), size=min(n_pool, len(pool_hists)), replace=False)
                counts, sums = defaultdict(int), defaultdict(int)
                for i in idx:
                    for kc, hist in pool_hists[i].items():
                        counts[kc] += len(hist)
                        sums[kc] += sum(hist)
                prior = {kc: sums[kc] / counts[kc] for kc in counts if counts[kc] > 0}
                aucs = []
                for ws in eval_samples:
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
                        aucs.append(a)
                draws.append(float(np.mean(aucs)) if aucs else float("nan"))
            res[f"pool_{min(n_pool, len(pool_hists))}"] = {
                "mean": round(float(np.mean(draws)), 4),
            }
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/poolsize_a1.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("POOLSIZE-A1-DONE")


if __name__ == "__main__":
    main()
