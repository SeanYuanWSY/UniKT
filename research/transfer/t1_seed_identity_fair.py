"""Batch 60: multi-seed identity-cell re-eval, 12 cells x 2 seeds.

b59 retrained the a09 SOURCE models at seeds 1/2 (transfer side: point-pos
24/24, CI-pos 19/24, mixed band with a clean domain split — A17/Junyi
robust, EdNet significance seed-sensitive). The identity cells still rest
on seed 42 only. This batch retrains each (arch, ds) IN-DOMAIN model at
seeds 1 and 2 (DEFAULT patience 10, the exact original protocol; the only
varied factor is the seed) and re-scores all 12 identity cells per seed.

Run selection: glob every {arch}_{ds} run (date prefix unknown at
script-writing time), read run_metadata.yaml, require an exact seed match
and UNIQUENESS per (arch, ds, seed), plus best_model.pth and the
"Training complete" run.log marker (b59 Kimi P1: metadata is written
pre-training and best_model.pth exists after the first improving epoch —
neither proves completion). All seed-42 runs (originals, b55 p30, b57 p30)
fail the seed match and cannot shadow the selection; the b59 a09 seed runs
fail the dataset glob.

Anchors per cell (per seed): n_paired == reference identity cell (261/245/
213), B1-side mean AUC reproduces the reference within 0.0001 (B1 side is
seed-invariant), and n_kc_prior == {a17: 96, ednet: 175, junyi: 38}.
The deep side changes by design; the seed-42 comparison lives in old_run.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

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
SEEDS = (1, 2)
REFS = {}
for _name, _path in (
    ("DKT", "t1_fair_rescore.json"),
    ("AKT", "t1_fair_rescore.json"),
    ("SAKT", "t1_sakt_fair.json"),
    ("DKVMN", "t1_dkvmn_fair.json"),
):
    with open(f"/root/unikt-fork/research/transfer/results/{_path}") as f:
        REFS[_name] = json.load(f)


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


def expected_run(arch, ds, seed):
    cands = sorted(glob.glob(f"/root/unikt-fork/runs/normal/{arch}_{ds}_*_fold0_*"))
    assert cands, f"no {arch} run for {ds}"
    hits = []
    for run in cands:
        meta_path = run + "/run_metadata.yaml"
        # Kimi b60 P1: the glob spans ALL historical {arch,ds} runs, and a
        # pre-metadata-era run would crash the assert below. A run without
        # metadata cannot prove its seed, so skipping it never loosens the
        # whitelist; a b60 run missing metadata still fails loudly via
        # len(hits) == 0.
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        if int(meta.get("seed", -1)) == seed:
            hits.append(run)
    assert len(hits) == 1, (
        f"expected exactly 1 {arch}/{ds} run with seed={seed}, found {len(hits)}: "
        + ", ".join(os.path.basename(h) for h in hits)
    )
    assert os.path.exists(hits[0] + "/best_model.pth"), f"incomplete run: {hits[0]}"
    # b59 Kimi P1: neither metadata nor best_model.pth proves completion;
    # run.log receives exactly one "Training complete" line on a finished run.
    with open(hits[0] + "/run.log", errors="ignore") as f:
        assert "Training complete" in f.read(), f"run lacks completion marker: {hits[0]}"
    return "runs/normal/" + os.path.basename(hits[0])


def main(n_users=300):
    out = {"cells": {}}
    runs = {}
    for ds in DS_LIST:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]

        # run-complete pooled prior (identical walk to b48/b49/b55/b57)
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
            for seed in SEEDS:
                run_rel = expected_run(arch, ds, seed)
                runs[f"{arch}|{ds}|s{seed}"] = run_rel
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
                out["cells"][f"identity_{arch}_{ds}|s{seed}"] = {
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
                    "old_run_seed42": {
                        "delta": ref.get("delta_fair_B1_minus_deep_rc"),
                        "deep_mean": ref.get("deep_mean_samesupport"),
                    },
                }
                print(ds, arch, f"s{seed}", json.dumps(out["cells"][f"identity_{arch}_{ds}|s{seed}"], ensure_ascii=False), flush=True)

    out["runs_by_seed"] = runs
    out["n_anchor_pass"] = sum(
        3
        for c in out["cells"].values()
        if c["anchor_n_paired"] and c["anchor_b1_mean_matches_ref"] and c["anchor_n_kc_prior"]
    )
    with open("/root/unikt-fork/research/transfer/results/t1_seed_identity_fair.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # b59 Kimi P2: dual sentinel so a queue-side grep cannot mistake an anchor
    # failure for success.
    print("T1SEEDID-DONE" if out["n_anchor_pass"] == len(out["cells"]) * 3 else "T1SEEDID-DONE-ANCHOR-FAIL")


if __name__ == "__main__":
    main()
