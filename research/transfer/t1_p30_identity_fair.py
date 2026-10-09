"""Batch 57: patience-30 identity-cell sweep — fair identity re-eval, 12 cells.

b55 closed the Junyi SAKT/DKVMN identity cells (patience 30 does not flip
the lookup win). This batch completes the symmetric sweep: 10 fresh p30
retrains (AKT/DKT x a17/ednet/junyi + SAKT/DKVMN x a17/ednet) join the two
b55 Junyi runs, and all 12 identity cells are re-scored on the exact
b48/b49/b55 construction (run-complete prior, A1-only, same-support
pairing, NaN pair-drop, identity jmap).

Run selection is a WHITELIST: the newest run for each (arch, ds) must be
exactly the expected p30 run (10 sweep dirs + 2 b55 dirs). Any other
newest run (e.g. the pre-p30 originals) fails loudly — a wrong-run
re-score would silently reproduce the old numbers.

Anchors per cell: n_paired == reference, B1-side mean AUC reproduces the
reference identity cell within 0.0001 (row set + B1 side identical), and
the ds-level prior size n_kc_prior == {a17: 96, ednet: 175, junyi: 38}
(structural, model-independent; b55 verified 38, b54/b56 the others).
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from signals import _forward_probs  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

DS_LIST = ("assistments17", "ednet_kt1", "junyi2015")
KC_N = {"assistments17": 96, "ednet_kt1": 175, "junyi2015": 38}

# exact expected p30 run basenames (10 sweep + 2 b55); anything else = wrong run
EXPECTED = {
    ("AKT", "assistments17"): "AKT_assistments17_20261008-141724_fold0_bs64",
    ("AKT", "ednet_kt1"): "AKT_ednet_kt1_20261008-193334_fold0_bs64",
    ("AKT", "junyi2015"): "AKT_junyi2015_20261009-050043_fold0_bs64",
    ("DKT", "assistments17"): "DKT_assistments17_20261009-073914_fold0_bs128",
    ("DKT", "ednet_kt1"): "DKT_ednet_kt1_20261009-074122_fold0_bs128",
    ("DKT", "junyi2015"): "DKT_junyi2015_20261009-074511_fold0_bs128",
    ("SAKT", "assistments17"): "SAKT_assistments17_20261009-074614_fold0_bs64",
    ("SAKT", "ednet_kt1"): "SAKT_ednet_kt1_20261009-075011_fold0_bs64",
    ("SAKT", "junyi2015"): "SAKT_junyi2015_20261008-133328_fold0_bs64",
    ("DKVMN", "assistments17"): "DKVMN_assistments17_20261009-075724_fold0_bs128",
    ("DKVMN", "ednet_kt1"): "DKVMN_ednet_kt1_20261009-081704_fold0_bs128",
    ("DKVMN", "junyi2015"): "DKVMN_junyi2015_20261008-133445_fold0_bs128",
}
REF_PATH = {
    ("DKT", None): "t1_fair_rescore.json",
    ("AKT", None): "t1_fair_rescore.json",
    ("SAKT", None): "t1_sakt_fair.json",
    ("DKVMN", None): "t1_dkvmn_fair.json",
}
REFS = {}
for (_a, _), _path in REF_PATH.items():
    if _a not in REFS:
        with open(f"/root/unikt-fork/research/transfer/results/{_path}") as f:
            REFS[_a] = json.load(f)


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
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return None
    boots = [np.mean(d[rng.integers(0, n, n)]) for _ in range(B)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def mean_or_none(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def expected_run(arch, ds):
    cands = sorted(glob.glob(f"/root/unikt-fork/runs/normal/{arch}_{ds}_*_fold0_*"))
    assert cands, f"no {arch} run for {ds}"
    run = cands[-1]
    base = os.path.basename(run)
    assert base == EXPECTED[(arch, ds)], (
        f"newest {arch}/{ds} run is not the expected p30 run: got {base}, "
        f"expected {EXPECTED[(arch, ds)]}"
    )
    assert os.path.exists(run + "/best_model.pth"), f"incomplete {arch} run: {run}"
    return "runs/normal/" + base


def main(n_users=300):
    out = {}
    for ds in DS_LIST:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]

        # run-complete pooled prior (identical walk to b48/b49/b55)
        hists = {}
        for ws in samples:
            hold = set(ws.holdout_idx.tolist())
            q_all = [int(x) for x in ws.question.tolist()]
            rid = run_ids(q_all)
            hold_runs = {rid[i] for i in hold}
            for i in range(len(ws.sequence)):
                if rid[i] not in hold_runs:
                    hists.setdefault(int(ws.sequence[i]), []).append(int(ws.response[i]))
        prior_b1 = {k: float(np.mean(v)) for k, v in hists.items() if v}

        for arch in ("DKT", "AKT", "SAKT", "DKVMN"):
            run_rel = expected_run(arch, ds)
            rm = restore_any(run_rel)
            per_stu = {}
            n_nonfinite = 0
            n_dummy = 0
            for ws in samples:
                jmap = {i: i for i in range(len(ws.sequence))}
                hold_pos = set(ws.holdout_idx.tolist())
                s_seq = ws.sequence.tolist()
                s_resp = [int(x) for x in ws.response.tolist()]
                s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
                r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
                probs = _forward_probs(rm, s, r_, None)[0]
                rows_d, rows_b = [], []
                for i in sorted(hold_pos):
                    if i == 0 or i not in jmap:
                        continue
                    if jmap[i] == 0:
                        n_dummy += 1
                        continue
                    if is_a1(ws, i):
                        continue
                    y = int(ws.response[i])
                    pd_ = float(probs[jmap[i]])
                    if pd_ != pd_:
                        n_nonfinite += 1
                        continue
                    rows_d.append({"y": y, "pred": pd_})
                    rows_b.append({"y": y, "pred": prior_b1.get(int(ws.sequence[i]), 0.5)})
                ad = stratified_auc_rows(rows_d, "pred")
                ab = stratified_auc_rows(rows_b, "pred")
                if ad is not None and ab is not None and rows_d:
                    per_stu[ws.user_id] = (ab, ad)
            del rm

            common = sorted(per_stu)
            deltas = [per_stu[u][0] - per_stu[u][1] for u in common]
            ref = REFS[arch][ds][f"identity_{arch}"]
            b1_mean = mean_or_none([per_stu[u][0] for u in common])
            deep_mean = mean_or_none([per_stu[u][1] for u in common])
            out[f"identity_{arch}_{ds}"] = {
                "run_dir": run_rel,
                "n_paired": len(common),
                "deep_mean": deep_mean,
                "b1_mean": b1_mean,
                "delta_b1_minus_deep": mean_or_none(deltas),
                "ci95": boot_ci(deltas),
                "n_nonfinite_probs": n_nonfinite,
                "n_dummy_pos0_excluded": n_dummy,
                "n_kc_prior_b1": len(prior_b1),
                "anchor_n_paired": len(common) == ref["n_paired"],
                "anchor_b1_mean_matches_ref": (
                    b1_mean is not None
                    and abs(b1_mean - ref["B1_mean_samesupport_rc"]) <= 0.0001
                ),
                "anchor_n_kc_prior": len(prior_b1) == KC_N[ds],
                "old_run": {
                    "delta": ref.get("delta_fair_B1_minus_deep_rc"),
                    "deep_mean": ref.get("deep_mean_samesupport"),
                },
            }
            print(ds, arch, json.dumps(out[f"identity_{arch}_{ds}"], ensure_ascii=False), flush=True)

    out["n_anchor_pass"] = sum(
        3
        for c in (v for k, v in out.items() if k != "n_anchor_pass")
        if c["anchor_n_paired"] and c["anchor_b1_mean_matches_ref"] and c["anchor_n_kc_prior"]
    )
    with open("/root/unikt-fork/research/transfer/results/t1_p30_identity_fair.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1P30ID-DONE")


if __name__ == "__main__":
    main()
