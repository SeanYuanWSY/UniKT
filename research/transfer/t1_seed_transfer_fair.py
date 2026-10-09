"""Batch 59: multi-seed source-side transfer re-eval, 12 cells x 2 seeds.

b57/b58 closed the patience axis (zero changes, via deterministic replay —
same seed, same order). All evidence so far is single-seed (42). This batch
retrains the four a09 source models at seed 1 and seed 2 (DEFAULT patience
10, i.e. the exact original protocol; the only varied factor is the seed)
and re-scores all 12 transfer cells per seed.

Run selection: run dir names carry no seed, so the WHITELIST reads
run_metadata.yaml and requires an exact seed match, a 20261009 date prefix,
uniqueness per (arch, seed), and best_model.pth. The seed-42 p30 runs from
b58 (AKT 085404 / DKT 101243 / SAKT 101331 / DKVMN 101442) also match the
date prefix but not the seed, so they cannot shadow the selection.

Anchors per cell (per seed): n_paired == reference, B1-side mean AUC
reproduces the reference transfer cell within 0.0001, and n_kc_prior ==
{96, 175, 38}. The deep side changes by design (different seed); the
seed-42 comparison lives in old_run.
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


def expected_src09(arch, seed):
    cands = sorted(glob.glob(f"/root/unikt-fork/runs/normal/{arch}_assistments09_20261009-*_fold0_*"))
    assert cands, f"no {arch} a09 run dated 20261009"
    hits = []
    for run in cands:
        meta_path = run + "/run_metadata.yaml"
        assert os.path.exists(meta_path), f"missing run_metadata.yaml in {run}"
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        if int(meta.get("seed", -1)) == seed:
            hits.append(run)
    assert len(hits) == 1, (
        f"expected exactly 1 {arch} a09 run with seed={seed} dated 20261009, found {len(hits)}: "
        + ", ".join(os.path.basename(h) for h in hits)
    )
    assert os.path.exists(hits[0] + "/best_model.pth"), f"incomplete run: {hits[0]}"
    # Kimi P1: best_model.pth exists after the FIRST improving epoch and metadata is
    # written before training starts — neither proves completion. run.log receives
    # exactly one "Training complete" line (base_trainer.py:626) on a finished run.
    with open(hits[0] + "/run.log", errors="ignore") as f:
        assert "Training complete" in f.read(), f"run lacks completion marker: {hits[0]}"
    return "runs/normal/" + os.path.basename(hits[0])


def build_pooled_prior(samples):
    hists = {}
    for ws in samples:
        hold = set(ws.holdout_idx.tolist())
        q_all = [int(x) for x in ws.question.tolist()]
        rid = run_ids(q_all)
        hold_runs = {rid[i] for i in hold}
        for i in range(len(ws.sequence)):
            if rid[i] not in hold_runs:
                hists.setdefault(int(ws.sequence[i]), []).append(int(ws.response[i]))
    return {k: float(np.mean(v)) for k, v in hists.items() if v}


def transfer_cell(samples, alignment, src09, prior_p):
    rm = restore_any(src09)
    per_stu = {}
    n_fb = 0
    n_nonfinite = 0
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
        jmap = {i: j for j, i in enumerate(keep)}
        hold_pos = set(ws.holdout_idx.tolist())
        s_seq = [mapping[int(ws.sequence[i])] for i in keep]
        s_resp = [int(ws.response[i]) for i in keep]
        s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
        r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
        probs = _forward_probs(rm, s, r_, None)[0]
        rows_d, rows_b = [], []
        for i in sorted(hold_pos):
            if i == 0 or i not in jmap:
                continue
            if jmap[i] == 0:
                continue
            if is_a1(ws, i):
                continue
            y = int(ws.response[i])
            pd_ = float(probs[jmap[i]])
            if pd_ != pd_:
                n_nonfinite += 1
                continue
            kc = int(ws.sequence[i])
            rows_d.append({"y": y, "pred": pd_})
            if kc not in prior_p:
                n_fb += 1
            rows_b.append({"y": y, "pred": prior_p.get(kc, 0.5)})
        ad = stratified_auc_rows(rows_d, "pred")
        ab = stratified_auc_rows(rows_b, "pred")
        if ad is not None and ab is not None and rows_d:
            per_stu[ws.user_id] = (ab, ad)
    del rm
    return per_stu, n_fb, n_nonfinite


def main(n_users=300):
    out = {"cells": {}}
    src09 = {(arch, seed): expected_src09(arch, seed) for arch in ("DKT", "AKT", "SAKT", "DKVMN") for seed in SEEDS}
    out["src09_by_seed"] = {f"{a}|s{s}": r for (a, s), r in src09.items()}
    for ds in ALIGN:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        alignment = {int(k): v for k, v in load_alignment(ALIGN[ds]).items()}
        prior_p = build_pooled_prior(samples)
        for arch in ("DKT", "AKT", "SAKT", "DKVMN"):
            for seed in SEEDS:
                per_stu, n_fb, n_nonfinite = transfer_cell(samples, alignment, src09[(arch, seed)], prior_p)
                common = sorted(per_stu)
                deltas = [per_stu[u][0] - per_stu[u][1] for u in common]
                ref = REFS[arch][ds][f"transfer_{arch}"]
                b1_mean = mean_or_none([per_stu[u][0] for u in common])
                deep_mean = mean_or_none([per_stu[u][1] for u in common])
                out["cells"][f"{ds}|{arch}|s{seed}"] = {
                    "n_paired": len(common),
                    "deep_mean": deep_mean,
                    "b1_mean": b1_mean,
                    "delta_b1_minus_deep": mean_or_none(deltas),
                    "ci95": boot_ci(deltas),
                    "n_nonfinite_probs": n_nonfinite,
                    "n_fallback_rows": n_fb,
                    "n_kc_prior": len(prior_p),
                    "anchor_n_paired": len(common) == ref["n_paired"],
                    "anchor_b1_mean_matches_ref": (
                        b1_mean is not None
                        and abs(b1_mean - ref["B1_mean_samesupport_rc"]) <= 0.0001
                    ),
                    "anchor_n_kc_prior": len(prior_p) == KC_N[ds],
                    "old_run_seed42": {
                        "delta": ref.get("delta_fair_B1_minus_deep_rc"),
                        "deep_mean": ref.get("deep_mean_samesupport"),
                    },
                }
                print(ds, arch, f"s{seed}", json.dumps(out["cells"][f"{ds}|{arch}|s{seed}"], ensure_ascii=False), flush=True)

    out["n_anchor_pass"] = sum(
        3
        for c in out["cells"].values()
        if c["anchor_n_paired"] and c["anchor_b1_mean_matches_ref"] and c["anchor_n_kc_prior"]
    )
    with open("/root/unikt-fork/research/transfer/results/t1_seed_transfer_fair.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # Kimi P2: sentinel variant so a queue-side grep cannot mistake an anchor
    # failure for success.
    print("T1SEEDTR-DONE" if out["n_anchor_pass"] == len(out["cells"]) * 3 else "T1SEEDTR-DONE-ANCHOR-FAIL")


if __name__ == "__main__":
    main()
