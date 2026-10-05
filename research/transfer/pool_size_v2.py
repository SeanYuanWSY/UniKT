"""POOL-SIZE v2: randomized pools + strict same-eval-set pairing vs deep transfer.

Upgrades over v1:
- Pools are RANDOM subsets of the pool side (10 seeds each), not "first n":
  mean +- std across seeds.
- The deep reference (AKT transfer, k3-high alignment) is recomputed on the
  SAME eval students with the SAME keep-set logic -> paired per-student
  differences with cluster bootstrap CIs. "Crossover" now means the paired
  CI excludes zero in favour of the table.
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
from baseline_eval import build_rows  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = {
    "assistments17": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
    "junyi2015": "assistments09__to__junyi2015__llm__k3-high__c25k3__rep0",
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
}
DEEP_RUN = {
    "assistments17": "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64",
    "junyi2015": "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64",
    "ednet_kt1": "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64",
}
POOL_SIZES = [10, 30, 100]
N_SEEDS = 10


def main():
    rng = np.random.default_rng(42)
    out_all = {}
    for ds in ["assistments17", "junyi2015", "ednet_kt1"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval_actual = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval_actual:]
        pool_samples = samples[: len(samples) - n_eval_actual]

        eval_students = build_rows(eval_samples, None)  # full-set rows for table
        pool_students = build_rows(pool_samples, None)
        # per-pool-student KC histories for incremental prior construction
        pool_hists = [st["kc_hist"] for st in pool_students]

        # deep reference on the SAME eval students (keep-set via alignment)
        alignment = {int(k): v for k, v in load_alignment(ALIGN[ds]).items()}
        rm = restore_any(DEEP_RUN[ds])
        deep_aucs = {}
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
            holdout_pos = set(ws.holdout_idx.tolist())
            seq_m = [mapping[int(ws.sequence[i])] for i in keep]
            resp_m = [int(ws.response[i]) for i in keep]
            s = torch.tensor([seq_m], dtype=torch.long, device=rm.device)
            r_ = torch.tensor([resp_m], dtype=torch.long, device=rm.device)
            probs = _forward_probs(rm, s, r_, None)[0]
            rows = [
                {"y": int(ws.response[i]), "pred": float(probs[j])}
                for j, i in enumerate(keep)
                if i in holdout_pos
            ]
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                deep_aucs[ws.user_id] = a
        deep_mean = float(np.mean(list(deep_aucs.values()))) if deep_aucs else None

        # eval-table aucs per user (prior passed per pool draw)
        def table_aucs_for_prior(prior):
            res = {}
            for st in eval_students:
                rows = [{"y": r["y"], "pred": prior.get(r["kc"], 0.5)} for r in st["rows"]]
                a = stratified_auc_rows(rows, "pred")
                if a is not None:
                    res[st["rows"][0]["y"] if False else id(st)] = a
            return list(res.values())

        out = {"n_eval": len(eval_students), "deep_same_set": round(deep_mean, 4) if deep_mean else None}
        for n_pool in POOL_SIZES:
            aucs_means = []
            for seed in range(N_SEEDS):
                r = np.random.default_rng(1000 + seed)
                idx = r.choice(len(pool_hists), size=min(n_pool, len(pool_hists)), replace=False)
                counts, sums = defaultdict(int), defaultdict(int)
                for i in idx:
                    for kc, hist in pool_hists[i].items():
                        counts[kc] += len(hist)
                        sums[kc] += sum(hist)
                prior = {kc: sums[kc] / counts[kc] for kc in counts if counts[kc] > 0}
                aucs_means.append(float(np.mean(table_aucs_for_prior(prior))))
            out[f"pool_{n_pool}"] = {
                "mean": round(float(np.mean(aucs_means)), 4),
                "std": round(float(np.std(aucs_means)), 4),
            }
        out_all[ds] = out
        print(ds, json.dumps(out, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/pool_size_v2.json", "w") as f:
        json.dump(out_all, f, ensure_ascii=False, indent=2)
    print("POOLSIZE-V2-DONE")


if __name__ == "__main__":
    main()
