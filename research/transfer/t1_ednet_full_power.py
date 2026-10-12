"""Batch 61: EdNet transfer cells, FULL sample (n=683) x 3 source seeds.

b59 left an unresolved gap: EdNet transfer CI significance is seed-sensitive
(3/8 CI-pos across seeds 1/2; DKVMN both seeds -> cross-zero, SAKT s2 ->
cross-zero) while point estimates stayed positive everywhere. With n=56 (the
[:300] paired queue) the CI half-width runs 0.080-0.107 and the cause cannot
be separated: small support vs effect-size fluctuation.

This batch re-scores the 4 EdNet transfer cells at the FULL ednet_kt1 sample
(683 students) for all three source seeds (42 = original a09 runs, 1/2 = the
b59 retrained runs), same-support pooled B1, same alignment, same protocol.
Zero training; evaluation only. Expect n_paired to grow from 56 toward the
mid-100s (alignment coverage ~23% of holdout rows is structural, see b45).

Run selection (per arch x seed): identical whitelist to b59 — 20261009 date
prefix + run_metadata.yaml exact seed match + uniqueness + best_model.pth +
run.log "Training complete". Seed 42 uses the ORIGINAL a09 runs dated
20261008 (pre-b59 protocol runs, same date rule as the t1fair3 reference
evaluations: the 20261008-193334 family). Seed 1/2 use the 20261009 runs.

Anchors per cell: n_kc_prior == 175; B1-side mean AUC reproduces the b59
[:300] cell within 0.001 (the B1 side is seed-invariant AND queue-invariant
on the paired common set — the common students are a subset of the [:300]
queue, so the full-sample B1 mean over the SAME students must match; the
tolerance is wider than b59's 0.0001 because rounding is at 4 decimals and
the common set can differ slightly as per-student AUC needs >=2 pos/neg
rows in the larger queue). The deep side changes by design (bigger queue +
all three seeds reported side by side); no old_run embedding needed — the
[:300] references live in the b59/b48/b49 JSONs.

Run selection (per arch x seed): identical whitelist to b59 (run_metadata.yaml
exact seed match + uniqueness + best_model.pth + run.log "Training
complete") with a per-seed date rule. Seed 42 = the ORIGINAL 20261003 a09
runs (the ones behind every t1fair3/b41/b48/b49 reference; the 20261009
seed-42 runs are the b57 p30 RETRAINS and must not be selected — the date
rule plus the seed match enforces this). Seed 1/2 = the 20261009 b59
retrained runs. The 20261006 seed 43/44 runs fail the seed match everywhere.
"""
import glob
import json

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

ALIGN_NAME = "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0"
KC_N = 175
SEEDS = (42, 1, 2)
# (arch, seed) -> run date-prefix rule: seed42 = ORIGINAL 20261003 runs
# (t1fair3/b41/b48/b49 reference source; NOT the 20261009 p30 retrain family),
# seed 1/2 = the 20261009 b59 retrained runs.
DATE_PREFIX = {42: "20261003", 1: "20261009", 2: "20261009"}

REF_B1 = {}
with open("/root/unikt-fork/research/transfer/results/t1_seed_transfer_fair.json") as f:
    _b59 = json.load(f)
for _arch in ("DKT", "AKT", "SAKT", "DKVMN"):
    REF_B1[_arch] = _b59["cells"][f"ednet_kt1|{_arch}|s1"]["b1_mean"]


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def run_ids(q_all):
    rid = [0] * len(q_all)
    cur = 0
    for i in range(1, len(q_all)):
        if q_all[i] != q_all[i - 1]:
            cur += 1
        rid[i] = cur
    return rid


def boot_ci(d, B=5000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = d[idx].mean()
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def mean_or_none(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def expected_src09(arch, seed):
    prefix = DATE_PREFIX[seed]
    cands = sorted(glob.glob(f"/root/unikt-fork/runs/normal/{arch}_assistments09_{prefix}-*_fold0_*"))
    assert cands, f"no {arch} a09 run dated {prefix}"
    hits = []
    for run in cands:
        meta_path = run + "/run_metadata.yaml"
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        if int(meta.get("seed", -1)) == seed:
            hits.append(run)
    assert len(hits) == 1, (
        f"expected exactly 1 {arch} a09 run seed={seed} dated {prefix}, found {len(hits)}: "
        + ", ".join(os.path.basename(h) for h in hits)
    )
    assert os.path.exists(hits[0] + "/best_model.pth"), f"incomplete run: {hits[0]}"
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


def main():
    out = {"cells": {}}
    src09 = {(arch, seed): expected_src09(arch, seed) for arch in ("DKT", "AKT", "SAKT", "DKVMN") for seed in SEEDS}
    out["src09_by_seed"] = {f"{a}|s{s}": r for (a, s), r in src09.items()}
    src = get_data_source(_make_rc("ednet_kt1"))
    samples = load_user_samples(src, fold=0)  # FULL sample, no [:300]
    out["n_total_students"] = len(samples)
    alignment = {int(k): v for k, v in load_alignment(ALIGN_NAME).items()}
    prior_p = build_pooled_prior(samples)
    for arch in ("DKT", "AKT", "SAKT", "DKVMN"):
        for seed in SEEDS:
            per_stu, n_fb, n_nonfinite = transfer_cell(samples, alignment, src09[(arch, seed)], prior_p)
            common = sorted(per_stu)
            deltas = [per_stu[u][0] - per_stu[u][1] for u in common]
            b1_mean = mean_or_none([per_stu[u][0] for u in common])
            deep_mean = mean_or_none([per_stu[u][1] for u in common])
            out["cells"][f"ednet_kt1|{arch}|s{seed}"] = {
                "n_paired": len(common),
                "deep_mean": deep_mean,
                "b1_mean": b1_mean,
                "delta_b1_minus_deep": mean_or_none(deltas),
                "ci95": boot_ci(deltas),
                "n_nonfinite_probs": n_nonfinite,
                "n_fallback_rows": n_fb,
                "n_kc_prior": len(prior_p),
                "anchor_n_kc_prior": len(prior_p) == KC_N,
                "anchor_b1_matches_b59": (
                    b1_mean is not None and abs(b1_mean - REF_B1[arch]) <= 0.001
                ),
                "b59_queue300_ref": {
                    "n_paired": 56,
                    "delta_s1": _b59["cells"][f"ednet_kt1|{arch}|s1"]["delta_b1_minus_deep"],
                    "ci95_s1": _b59["cells"][f"ednet_kt1|{arch}|s1"]["ci95"],
                    "delta_s2": _b59["cells"][f"ednet_kt1|{arch}|s2"]["delta_b1_minus_deep"],
                    "ci95_s2": _b59["cells"][f"ednet_kt1|{arch}|s2"]["ci95"],
                },
            }
            print("ednet_kt1", arch, f"s{seed}", json.dumps(out["cells"][f"ednet_kt1|{arch}|s{seed}"], ensure_ascii=False), flush=True)

    out["n_anchor_pass"] = sum(
        2
        for c in out["cells"].values()
        if c["anchor_n_kc_prior"] and c["anchor_b1_matches_b59"]
    )
    with open("/root/unikt-fork/research/transfer/results/t1_ednet_full_power.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1EDFULL-DONE" if out["n_anchor_pass"] == len(out["cells"]) * 2 else "T1EDFULL-DONE-ANCHOR-FAIL")


if __name__ == "__main__":
    main()
