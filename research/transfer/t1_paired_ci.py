"""T1 full-grid paired bootstrap CIs (review blocker #3).

Every cell of the T1 primary table (A1-only scope): per-student AUCs for
B1 table vs deep (identity + transfer), paired difference, cluster
bootstrap 95% CI, n. Domains x {AKT, DKT} x {identity, transfer} x B1.
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
    "junyi2015": "assistments09__to__junyi2015__llm__k3-high__c25k3__rep0",
}
RUNS = {
    "identity_AKT": {"assistments17": "runs/normal/AKT_assistments17_20261003-145008_fold0_bs64", "ednet_kt1": "runs/normal/AKT_ednet_kt1_20261003-190537_fold0_bs64", "junyi2015": "runs/normal/AKT_junyi2015_20261004-195553_fold0_bs64"},
    "identity_DKT": {"assistments17": "runs/normal/DKT_assistments17_20261003-144042_fold0_bs128", "ednet_kt1": "runs/normal/DKT_ednet_kt1_20261003-190327_fold0_bs128", "junyi2015": "runs/normal/DKT_junyi2015_20261004-195512_fold0_bs128"},
    "transfer_AKT": {ds: "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64" for ds in ALIGN},
    "transfer_DKT": {ds: "runs/normal/DKT_assistments09_20261003-024700_fold0_bs128" for ds in ALIGN},
}


def boot_ci(d, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return None
    boots = [np.mean(d[rng.integers(0, n, n)]) for _ in range(B)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def main(n_users=300):
    out = {}
    for ds in ALIGN:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        students = build_rows(samples, None)
        prior = fit_global_kc_prior(students)
        alignment = {int(k): v for k, v in load_alignment(ALIGN[ds]).items()}

        # B1 per-student (A1-excluded rows) — independent traversal, no zip
        b1 = {}
        for ws in samples:
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

        res = {"B1_mean": round(float(np.mean(list(b1.values()))), 4), "B1_n": len(b1)}
        for mtag, runmap in RUNS.items():
            rm = restore_any(runmap[ds])
            deep = {}
            for ws in samples:
                if mtag.startswith("identity"):
                    s_seq = ws.sequence.tolist()
                    s_resp = [int(x) for x in ws.response.tolist()]
                    jmap = {i: i for i in range(len(s_seq))}
                else:
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
                    if i == 0 or i not in jmap:
                        continue
                    isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                    if not isA1:
                        rows.append({"y": int(ws.response[i]), "pred": float(probs[jmap[i]])})
                a = stratified_auc_rows(rows, "pred")
                if a is not None:
                    deep[ws.user_id] = a
            common = [u for u in b1 if u in deep]
            diffs = np.array([b1[u] - deep[u] for u in common])
            res[mtag] = {
                "deep_mean": round(float(np.mean([deep[u] for u in common])), 4) if common else None,
                "n_paired": len(common),
                "delta_B1_minus_deep": round(float(np.mean(diffs)), 4) if len(diffs) else None,
                "ci95": boot_ci(diffs) if len(diffs) else None,
            }
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/t1_paired_ci.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1CI-DONE")


if __name__ == "__main__":
    main()
