"""Batch 49: DKVMN through the fair-recompute pipeline (fourth architecture).

All headline conclusions rest on DKT+AKT only. This batch scores DKVMN
(default-config identity runs trained in batch49 + the existing DKVMN
assistments09 source run) on the exact batch41/43 fair rows: same samples,
same run-complete prior_fixed2, same A1-only protocol, same same-support
pairing.

Structural anchor (preregistered): the scored support is model-independent
— it is determined by the mapping/keep filter, A1 exclusion, holdout labels
and per-student label diversity, never by the model (given finite probs).
So n_paired must EQUAL t1fair_rescore.json's counterpart cells (both the
AKT and DKT cell of the same domain/mode), and n_kc_prior_b1 must equal
the reference. Any mismatch = construction drift, not a DKVMN finding.

Deltas vs t1_fair_rescore.py:
 (1) RUNS: DKVMN only, identity run dirs discovered by glob (newest
     DKVMN_<ds>_*) and recorded in the JSON; transfer = existing
     DKVMN_assistments09_20261003-025330 run;
 (2) support anchor instead of value anchor (no external DKVMN reference
     exists under this protocol);
 (3) dropped old-prior diagnostics / full-subset B1 means / sym attack
     (covered by b36/t1fair3 on this row surface).
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = {
    "assistments17": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
    "junyi2015": "assistments09__to__junyi2015__llm__k3-high__c25k3__rep0",
}
DKVMN_A09 = "runs/normal/DKVMN_assistments09_20261003-024801_fold0_bs128"
with open("/root/unikt-fork/research/transfer/results/t1_fair_rescore.json") as f:
    T1FAIR_REF = json.load(f)


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def run_ids(q_all):
    rid = [0] * len(q_all)
    r = 0
    for i in range(1, len(q_all)):
        if q_all[i] != q_all[i - 1]:
            r += 1
        rid[i] = r
    return rid


def boot_ci(d, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return None
    boots = [np.mean(d[rng.integers(0, n, n)]) for _ in range(B)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def mean_or_none(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def newest_dkvmn_run(ds):
    cands = sorted(glob.glob(f"/root/unikt-fork/runs/normal/DKVMN_{ds}_*_fold0_*"))
    assert cands, f"no DKVMN run dir for {ds}"
    # restore_any resolves run_dir relative to CWD (/root/unikt-fork) —
    # return the runs/normal/<name> form, not the bare basename (fork P1).
    assert os.path.exists(cands[-1] + "/best_model.pth"), f"incomplete DKVMN run: {cands[-1]}"
    return "runs/normal/" + os.path.basename(cands[-1])


def main(n_users=300):
    out = {}
    for ds in ALIGN:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]

        # run-complete evidence-only prior (batch43 adjudicated construction)
        hist_all2 = {}
        for ws in samples:
            hold = set(ws.holdout_idx.tolist())
            q_all = [int(x) for x in ws.question.tolist()]
            rid = run_ids(q_all)
            hold_runs = {rid[i] for i in hold}
            for i in range(len(ws.sequence)):
                if rid[i] not in hold_runs:
                    hist_all2.setdefault(int(ws.sequence[i]), []).append(int(ws.response[i]))
        prior_b1 = {k: float(np.mean(v)) for k, v in hist_all2.items() if v}

        alignment = {int(k): v for k, v in load_alignment(ALIGN[ds]).items()}
        res = {
            "n_kc_prior_b1": len(prior_b1),
            "anchor_n_kc_prior_matches": len(prior_b1) == T1FAIR_REF[ds]["n_kc_prior_fixed2"],
        }

        for mtag in ("identity_DKVMN", "transfer_DKVMN"):
            run_dir = newest_dkvmn_run(ds) if mtag.startswith("identity") else DKVMN_A09
            rm = restore_any(run_dir)
            deep, b1c = {}, {}
            n_dummy_excluded = 0
            n_nonfinite_probs = 0
            for ws in samples:
                if mtag.startswith("identity"):
                    jmap = {i: i for i in range(len(ws.sequence))}
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
                    jmap = {i: j for j, i in enumerate(keep)}
                hold_pos = set(ws.holdout_idx.tolist())
                s_seq = ws.sequence.tolist() if mtag.startswith("identity") else [mapping[int(ws.sequence[i])] for i in keep]
                s_resp = [int(x) for x in ws.response.tolist()] if mtag.startswith("identity") else [int(ws.response[i]) for i in keep]
                s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
                r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
                probs = _forward_probs(rm, s, r_, None)[0]
                rows_d, rows_b = [], []
                for i in sorted(hold_pos):
                    if i == 0 or i not in jmap:
                        continue
                    if jmap[i] == 0:
                        n_dummy_excluded += 1
                        continue
                    if is_a1(ws, i):
                        continue
                    y = int(ws.response[i])
                    pd_ = float(probs[jmap[i]])
                    if pd_ != pd_:  # NaN rows dropped in pairs (batch46 hardening)
                        n_nonfinite_probs += 1
                        continue
                    rows_d.append({"y": y, "pred": pd_})
                    rows_b.append({"y": y, "pred": prior_b1.get(int(ws.sequence[i]), 0.5)})
                ad = stratified_auc_rows(rows_d, "pred")
                ab = stratified_auc_rows(rows_b, "pred")
                if ad is not None and ab is not None:
                    deep[ws.user_id] = ad
                    b1c[ws.user_id] = ab
            common = sorted(u for u in deep if u in b1c)
            diffs = np.array([b1c[u] - deep[u] for u in common])
            mode, arch = mtag.split("_")
            ref = T1FAIR_REF[ds][f"{mode}_AKT"]
            ref_dkt = T1FAIR_REF[ds][f"{mode}_DKT"]
            res[mtag] = {
                "run_dir": run_dir,
                "deep_mean_samesupport": mean_or_none([deep[u] for u in common]),
                "B1_mean_samesupport_rc": mean_or_none([b1c[u] for u in common]),
                "n_paired": len(common),
                "anchor_n_paired_model_independent": (
                    len(common) == ref["n_paired"] == ref_dkt["n_paired"]
                ),
                "delta_fair_B1_minus_deep_rc": mean_or_none(diffs.tolist()) if len(diffs) else None,
                "ci95_fair_rc": boot_ci(diffs) if len(diffs) else None,
                "n_dummy_pos0_excluded": n_dummy_excluded,
                "n_nonfinite_probs": n_nonfinite_probs,
            }
            del rm

        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/t1_dkvmn_fair.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1DKVMN-FAIR-DONE")


if __name__ == "__main__":
    main()
