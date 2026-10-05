"""Deep-side multi-seed robustness: A09->A17 transfer across AKT seeds 42/43/44.

Main-paper deep numbers all come from seed-42 checkpoints. This reruns the
headline transfer cell (AKT, k3-high alignment, keep-scope, paired vs table)
on seed-43/44 checkpoints and reports seed spread vs the paired table delta.
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


def find_runs():
    runs = sorted(glob.glob("/root/unikt-fork/runs/normal/AKT_assistments09_*_fold0_bs64"))
    out = []
    for r in runs:
        cfg = f"{r}/run_config.yaml"
        seed = 42
        for line in open(cfg):
            if line.strip().startswith("seed:"):
                seed = int(line.split(":")[1].strip())
                break
        out.append((seed, r))
    return out


def main(n_users=264):
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:n_users]
    students = build_rows(samples, ALIGN)  # keep-set rows for table
    prior = fit_global_kc_prior(students)
    alignment = {int(k): v for k, v in load_alignment(ALIGN).items()}

    table_aucs = []
    for st in students:
        rows = [{"y": r["y"], "pred": prior.get(r["kc"], 0.5)} for r in st["rows"]]
        a = stratified_auc_rows(rows, "pred")
        if a is not None:
            table_aucs.append(a)

    res = {"table_keep": round(float(np.mean(table_aucs)), 4), "n": len(table_aucs)}
    for seed, run in find_runs():
        rm = restore_any(run)
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
                aucs.append(a)
        res[f"seed{seed}"] = {
            "run": run.split("/")[-1][:50],
            "deep_keep_auc": round(float(np.mean(aucs)), 4) if aucs else None,
            "n": len(aucs),
        }
        print(seed, res[f"seed{seed}"], flush=True)

    ds = [v["deep_keep_auc"] for v in res.values() if isinstance(v, dict) and "deep_keep_auc" in v and v["deep_keep_auc"]]
    if len(ds) >= 2:
        res["seed_spread"] = round(max(ds) - min(ds), 4)
        res["seed_mean"] = round(float(np.mean(ds)), 4)
    with open("/root/unikt-fork/research/transfer/results/seed_robustness.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False))
    print("SEED-ROBUSTNESS-DONE")


if __name__ == "__main__":
    main()
