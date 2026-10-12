"""Batch 71: model+table FUSION retrial under the fair caliber — revival
candidate #1 from the b68 sweep.

idea3 (batch-era) died as "fusion zero increment" on A17 single-domain,
single-architecture (AKT), seed 42, BEFORE the A1-only rebase and before
the b41 same-support fair caliber. Its numbers (fusion 0.5433 vs table
0.5473) were also computed on the old mixed-support protocol. The claim
"fusion adds nothing" has never been tested fairly:

- fair caliber (same-support, evidence-only B1)
- four architectures (DKT/AKT/SAKT/DKVMN)
- three domains
- seed 42 (single, per the original)

Fusion rule (preregistered, mirroring idea3's F1): per-student AUC-max?
No — alpha-fusion at the ROW level is the honest version: fused_pred =
alpha*deep + (1-alpha)*B1 for alpha in {0.25, 0.5, 0.75}, alpha chosen on
EVIDENCE rows only (per-domain, per-arch best alpha by pooled evidence-row
AUC), then scored on holdout. This avoids holdout tuning while allowing
complementarity. Compare vs max(B1, deep) per cell (the deployment
baseline) on the same holdout rows.

Gates in 20261012_batch71.md. Sentinel: FUSION71-DONE / -ANCHOR.
Anchors: per-cell n_paired equals the b48/b49 reference identity cells
(261/245/213) and B1-side mean matches the b57 identity references within
0.0001 (same frozen table construction as b57/b60).
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
NP_REF = {"assistments17": 261, "ednet_kt1": 245, "junyi2015": 213}
B1_REF = {"assistments17": 0.5485, "ednet_kt1": 0.5263, "junyi2015": 0.5668}  # b57 identity b1_mean (run-complete prior)
ARCHS = ("DKT", "AKT", "SAKT", "DKVMN")
ALPHAS = (0.25, 0.5, 0.75)


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


def expected_run(arch, ds, seed=42):
    # whitelist: seed match + completion + CHECKPOINT-DIMENSION sanity
    # (b71 crash: orphan runs from old preprocessing carry mismatched item
    # counts, e.g. DKT junyi 014541 = 40 items vs current 38-KC+pad=39).
    # v3: per-arch item-embedding key (advisory: v2 used the DKT-only
    # out_layer key, which would skip ALL AKT/SAKT/DKVMN candidates).
    # Direct n_items embeddings measured from real checkpoints:
    #   DKT out_layer.weight | AKT q_embed.weight | SAKT exercise_emb.weight
    #   DKVMN k_emb_layer.weight   (each [0] == n_items; interaction/qa
    #   embeddings are 2n(+1) and are NOT used, to avoid emb/out conflation).
    ITEM_KEY = {"DKT": "out_layer.weight", "AKT": "q_embed.weight",
                "SAKT": "exercise_emb.weight", "DKVMN": "k_emb_layer.weight"}
    EXPECT_ITEMS = {"assistments17": {102}, "ednet_kt1": {189}, "junyi2015": {38, 39}}
    cands = sorted(glob.glob(f"/root/unikt-fork/runs/normal/{arch}_{ds}_*_fold0_*"))
    ok_runs = []
    for run in cands:
        meta_path = run + "/run_metadata.yaml"
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        if int(meta.get("seed", -1)) != seed:
            continue
        if not os.path.exists(run + "/best_model.pth"):
            continue
        with open(run + "/run.log", errors="ignore") as f:
            if "Training complete" not in f.read():
                continue
        try:
            sd = torch.load(run + "/best_model.pth", map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            key = ITEM_KEY[arch]
            item_w = [v for k, v in sd.items() if k.endswith(key)]
            if item_w and item_w[0].shape[0] in EXPECT_ITEMS[ds]:
                ok_runs.append(run)  # dimension-compatible original
            else:
                print(f"[whitelist] skip dim-mismatch {run}: {key}={tuple(item_w[0].shape) if item_w else 'absent'}", flush=True)
        except Exception as e:
            print(f"[whitelist] skip unreadable {run}: {e}", flush=True)
    assert ok_runs, f"no dimension-valid seed-42 run for {arch}/{ds}"
    ok_runs.sort()  # oldest first (original protocol)
    chosen = ok_runs[0]
    return "runs/normal/" + os.path.basename(chosen), len(ok_runs)


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


def main(n_users=300):
    out = {"cells": {}}
    for ds in DS_LIST:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        prior_b1 = build_pooled_prior(samples)
        for arch in ARCHS:
            run_rel, n_hits = expected_run(arch, ds)
            rm = restore_any(run_rel)
            # collect per-student rows: evidence rows (alpha selection) + holdout rows (scoring)
            per_stu = {}
            n_nonfinite = 0
            for ws in samples:
                jmap = {i: i for i in range(len(ws.sequence))}
                hold_pos = set(ws.holdout_idx.tolist())
                ev_pos = set(range(len(ws.sequence))) - hold_pos
                s_seq = ws.sequence.tolist()
                s_resp = [int(x) for x in ws.response.tolist()]
                s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
                r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
                probs = _forward_probs(rm, s, r_, None)[0]
                ev_rows, ho_rows = [], []
                for i in sorted(ev_pos):
                    if i == 0 or is_a1(ws, i):
                        continue
                    pd_ = float(probs[jmap[i]])
                    if pd_ != pd_:
                        continue
                    ev_rows.append({"y": int(ws.response[i]), "pred": pd_, "b1": prior_b1.get(int(ws.sequence[i]), 0.5)})
                for i in sorted(hold_pos):
                    if i == 0 or is_a1(ws, i):
                        continue
                    pd_ = float(probs[jmap[i]])
                    if pd_ != pd_:
                        n_nonfinite += 1
                        continue
                    ho_rows.append({"y": int(ws.response[i]), "pred": pd_, "b1": prior_b1.get(int(ws.sequence[i]), 0.5)})
                if not ho_rows:
                    continue
                # alpha selection on evidence rows (pooled AUC)
                best_alpha, best_auc = None, -1.0
                for a in ALPHAS:
                    rows = [{"y": r["y"], "pred": a * r["pred"] + (1 - a) * r["b1"]} for r in ev_rows]
                    auc = stratified_auc_rows(rows, "pred")
                    if auc is not None and auc > best_auc:
                        best_auc, best_alpha = auc, a
                if best_alpha is None:
                    best_alpha = 0.5
                # holdout scoring: fusion / deep / b1 / max-baseline
                fus = [{"y": r["y"], "pred": best_alpha * r["pred"] + (1 - best_alpha) * r["b1"]} for r in ho_rows]
                dpr = [{"y": r["y"], "pred": r["pred"]} for r in ho_rows]
                b1r = [{"y": r["y"], "pred": r["b1"]} for r in ho_rows]
                mxr = [{"y": r["y"], "pred": max(r["pred"], r["b1"])} for r in ho_rows]
                af = stratified_auc_rows(fus, "pred")
                ad = stratified_auc_rows(dpr, "pred")
                ab = stratified_auc_rows(b1r, "pred")
                am = stratified_auc_rows(mxr, "pred")
                if None in (af, ad, ab, am):
                    continue
                per_stu[ws.user_id] = (af, ad, ab, am, best_alpha)
            del rm
            common = sorted(per_stu)
            d_fus_vs_max = [per_stu[u][0] - per_stu[u][3] for u in common]
            d_fus_vs_b1 = [per_stu[u][0] - per_stu[u][2] for u in common]
            d_fus_vs_deep = [per_stu[u][0] - per_stu[u][1] for u in common]
            out["cells"][f"{ds}|{arch}"] = {
                "run_dir": run_rel,
                "n_seed42_runs_seen": n_hits,
                "alpha_choices": {str(a): sum(1 for u in common if per_stu[u][4] == a) for a in ALPHAS},
                "n_paired": len(common),
                "fus_mean": round(float(np.mean([per_stu[u][0] for u in common])), 4),
                "deep_mean": round(float(np.mean([per_stu[u][1] for u in common])), 4),
                "b1_mean": round(float(np.mean([per_stu[u][2] for u in common])), 4),
                "max_mean": round(float(np.mean([per_stu[u][3] for u in common])), 4),
                "delta_fus_minus_max": round(float(np.mean(d_fus_vs_max)), 4),
                "ci95_max": boot_ci(d_fus_vs_max),
                "delta_fus_minus_b1": round(float(np.mean(d_fus_vs_b1)), 4),
                "ci95_vs_b1": boot_ci(d_fus_vs_b1),
                "delta_fus_minus_deep": round(float(np.mean(d_fus_vs_deep)), 4),
                "ci95_vs_deep": boot_ci(d_fus_vs_deep),
                "n_nonfinite_probs": n_nonfinite,
                "anchor_n_paired": len(common) == NP_REF[ds],
                "anchor_b1_mean": abs(round(float(np.mean([per_stu[u][2] for u in common])), 4) - B1_REF[ds]) <= 0.0001,
                "anchor_n_kc": len(prior_b1) == KC_N[ds],
            }
            print(ds, arch, json.dumps(out["cells"][f"{ds}|{arch}"], ensure_ascii=False), flush=True)
    out["n_anchor_pass"] = sum(3 for c in out["cells"].values() if c["anchor_n_paired"] and c["anchor_b1_mean"] and c["anchor_n_kc"])
    with open("/root/unikt-fork/research/transfer/results/fusion71_fair.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("FUSION71-DONE" if out["n_anchor_pass"] == len(out["cells"]) * 3 else "FUSION71-DONE-ANCHOR")


if __name__ == "__main__":
    main()
