"""Batch 55: Junyi training-sufficiency recheck (SAKT/DKVMN, patience 30).

b48/b49 left "training sufficiency unresolved" on the Junyi identity cells
(SAKT 16ep id AUC 0.4059, DKVMN 30ep 0.4326, both B1-favored with CI
excluding 0). This batch retrains both with --early_stopping.patience 30
(default 10, all else default) and re-runs the fair identity evaluation on
the exact b48/b49 construction (run-complete prior, A1-only, same-support
pairing, NaN pair-drop, identity jmap).

Anchors: n_paired == 213 (structural, model-independent) and the B1-side
mean AUC must reproduce t1_sakt_fair.json / t1_dkvmn_fair.json identity
cells within 0.0001 — proving the row set and B1 side are identical, so
any delta change is attributable to the new deep model.

Reference constants (report-side, not anchors): old deltas +0.1609/+0.1342,
old deep means 0.4059/0.4326, old epochs 16/30.
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

DS = "junyi2015"
REFS = {}
for _name, _path in (
    ("SAKT", "t1_sakt_fair.json"),
    ("DKVMN", "t1_dkvmn_fair.json"),
):
    with open(f"/root/unikt-fork/research/transfer/results/{_path}") as f:
        REFS[_name] = json.load(f)[DS][f"identity_{_name}"]

OLD = {"SAKT": {"delta": 0.1609, "deep_mean": 0.4059, "epochs": 16},
       "DKVMN": {"delta": 0.1342, "deep_mean": 0.4326, "epochs": 30}}

# b47/b49 original runs: if newest_run returns one of these, the p30 training
# died before creating its run dir and we would silently re-score the OLD
# checkpoint (fork review P2-1) — fail loudly instead.
OLD_RUN_BASES = {
    "SAKT_junyi2015_20261008-092533_fold0_bs64",
    "DKVMN_junyi2015_20261008-100906_fold0_bs128",
}


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


def newest_run(arch):
    cands = sorted(glob.glob(f"/root/unikt-fork/runs/normal/{arch}_{DS}_*_fold0_*"))
    assert cands, f"no {arch} run for {DS}"
    run = cands[-1]
    assert os.path.basename(run) not in OLD_RUN_BASES, (
        f"newest {arch} run is the OLD pre-p30 run — p30 training missing: {run}"
    )
    assert os.path.exists(run + "/best_model.pth"), f"incomplete {arch} run: {run}"
    return "runs/normal/" + os.path.basename(run), run


def main(n_users=300):
    src = get_data_source(_make_rc(DS))
    samples = load_user_samples(src, fold=0)[:n_users]

    # run-complete pooled prior (identical walk to b48/b49)
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

    out = {}
    for arch in ("SAKT", "DKVMN"):
        run_rel, run_abs = newest_run(arch)
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
        ref = REFS[arch]
        b1_mean = mean_or_none([per_stu[u][0] for u in common])
        deep_mean = mean_or_none([per_stu[u][1] for u in common])
        out[f"identity_{arch}_p30"] = {
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
            "anchor_n_kc_prior": len(prior_b1) == 38,
            "anchor_b1_mean_matches_ref": (
                b1_mean is not None
                and abs(b1_mean - ref["B1_mean_samesupport_rc"]) <= 0.0001
            ),
            "old_run": {"delta": OLD[arch]["delta"], "deep_mean": OLD[arch]["deep_mean"], "epochs": OLD[arch]["epochs"]},
        }
        print(arch, json.dumps(out[f"identity_{arch}_p30"], ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/t1_junyi_retrain_fair.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1JYP30-DONE")


if __name__ == "__main__":
    main()
