"""Batch 52: B1 lookup-prior construction robustness (anti-self-leakage).

Motivation (pre-registered 2026-10-08): the headline "lookup beats frozen
transfer" uses B1 = per-KC mean over ALL students' non-holdout-run rows —
including the scored student's OWN earlier attempts of the same KC. A
reviewer can argue the lookup wins via self-history leakage, not KC
difficulty. This batch rebuilds the B1 side under four stricter variants:

  pooled     V0: b46 construction (anchor vs t1_fair_rescore.json)
  loso       V1: per-KC mean EXCLUDING the scored student's own rows
  laplace    V2: (sum_kc + 5*global) / (n_kc + 5) shrinkage toward global
  first      V3: per-KC mean over first non-holdout-run occurrence per student
  loso_first V4: V1 support restricted to V3 (others' first attempts only)

Row construction, deep side, runs, and pairing are IDENTICAL to b46
(t1_metric_robustness.py): run-complete prior support, A1-only, same-support
pairing, NaN pair-drop. The deep forward runs ONCE per cell; variants only
change the scalar prior on the B1 rows. Fallback when a variant's support
for a KC is empty: 0.5 (b46 convention), counted per cell.

Anchors: per cell, pooled-variant delta_auc must reproduce
t1_fair_rescore.json delta_fair_B1_minus_deep_rc within 0.0001 and n_paired
must equal the ref — construction-identity gate before any variant is read.
"""
import json
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
RUNS = {
    "identity_AKT": {"assistments17": "runs/normal/AKT_assistments17_20261003-145008_fold0_bs64", "ednet_kt1": "runs/normal/AKT_ednet_kt1_20261003-190537_fold0_bs64", "junyi2015": "runs/normal/AKT_junyi2015_20261004-195553_fold0_bs64"},
    "identity_DKT": {"assistments17": "runs/normal/DKT_assistments17_20261003-144042_fold0_bs128", "ednet_kt1": "runs/normal/DKT_ednet_kt1_20261003-190327_fold0_bs128", "junyi2015": "runs/normal/DKT_junyi2015_20261004-195512_fold0_bs128"},
    "transfer_AKT": {ds: "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64" for ds in ALIGN},
    "transfer_DKT": {ds: "runs/normal/DKT_assistments09_20261003-024700_fold0_bs128" for ds in ALIGN},
}
# anchor: t1_fair_rescore.json (tag t1fair3) rc fields — same as b46
with open("/root/unikt-fork/research/transfer/results/t1_fair_rescore.json") as f:
    T1FAIR_REF = json.load(f)

VARIANTS = ("pooled", "loso", "laplace", "first", "loso_first")
LAPLACE_M = 5.0


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
    d = np.asarray(d, dtype=float)  # accept plain lists (b46 callers passed np.array)
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return None
    boots = [np.mean(d[rng.integers(0, n, n)]) for _ in range(B)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def brier(rows):
    return float(np.mean([(r["pred"] - r["y"]) ** 2 for r in rows]))


def mean_or_none(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def build_prior_tables(samples):
    """Five per-KC prior tables + per-student LOSO views (domain-level).

    Support = positions outside holdout runs (run-complete prior, b46).
    LOSO needs per-student access, so we keep (uid, y) lists and build
    per-student means on demand. First-attempt tables keep each student's
    FIRST non-holdout-run occurrence per KC only.
    """
    hist = {}        # kc -> list[(uid, y)]   (pooled / loso base)
    hist_first = {}  # kc -> list[(uid, y)]   (first / loso_first base)
    seen_first = set()  # (uid, kc) already contributed a first attempt
    for ws in samples:
        uid = ws.user_id
        hold = set(ws.holdout_idx.tolist())
        q_all = [int(x) for x in ws.question.tolist()]
        rid = run_ids(q_all)
        hold_runs = {rid[i] for i in hold}
        for i in range(len(ws.sequence)):
            if rid[i] in hold_runs:
                continue
            kc = int(ws.sequence[i])
            y = int(ws.response[i])
            hist.setdefault(kc, []).append((uid, y))
            if (uid, kc) not in seen_first:
                seen_first.add((uid, kc))
                hist_first.setdefault(kc, []).append((uid, y))
    ys_all = [y for rows in hist.values() for _, y in rows]
    global_mean = float(np.mean(ys_all)) if ys_all else 0.5

    def mean_excluding(rows, uid):
        vals = [y for u, y in rows if u != uid]
        return float(np.mean(vals)) if vals else None

    pooled = {kc: float(np.mean([y for _, y in rows])) for kc, rows in hist.items() if rows}
    laplace = {
        kc: (sum(y for _, y in rows) + LAPLACE_M * global_mean) / (len(rows) + LAPLACE_M)
        for kc, rows in hist.items()
    }
    first = {kc: float(np.mean([y for _, y in rows])) for kc, rows in hist_first.items() if rows}

    def prior_for(variant, kc, uid):
        if variant == "pooled":
            return pooled.get(kc)
        if variant == "laplace":
            return laplace.get(kc)
        if variant == "first":
            return first.get(kc)
        if variant == "loso":
            return mean_excluding(hist.get(kc, []), uid) if kc in hist else None
        if variant == "loso_first":
            return mean_excluding(hist_first.get(kc, []), uid) if kc in hist_first else None
        raise ValueError(variant)

    return prior_for


def main(n_users=300):
    out = {}
    for ds in ALIGN:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        prior_for = build_prior_tables(samples)
        alignment = {int(k): v for k, v in load_alignment(ALIGN[ds]).items()}
        res = {}

        for mtag, runmap in RUNS.items():
            rm = restore_any(runmap[ds])
            per_stu = {}   # uid -> {variant: {"auc": a, "brier": b, "n_fb": k}}
            deep = {}      # uid -> {"auc": a, "brier": b}  (shared across variants)
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
                rows_d, base_rows = [], []
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
                        n_nonfinite_probs += 1
                        continue
                    rows_d.append({"y": y, "pred": pd_})
                    base_rows.append({"y": y, "kc": int(ws.sequence[i])})
                ad = stratified_auc_rows(rows_d, "pred")
                if ad is None or not rows_d:
                    continue
                deep[ws.user_id] = {"auc": ad, "brier": brier(rows_d)}
                entry = {}
                for variant in VARIANTS:
                    rows_v, n_fb = [], 0
                    for b in base_rows:
                        pv = prior_for(variant, b["kc"], ws.user_id)
                        if pv is None:
                            n_fb += 1
                            pv = 0.5  # b46 fallback convention
                        rows_v.append({"y": b["y"], "pred": pv})
                    entry[variant] = {
                        "auc": stratified_auc_rows(rows_v, "pred"),
                        "brier": brier(rows_v),
                        "n_fallback": n_fb,
                    }
                per_stu[ws.user_id] = entry
            del rm

            common = sorted(set(per_stu) & set(deep))
            ref = T1FAIR_REF[ds][mtag]
            cell = {
                "run_dir": runmap[ds],
                "n_paired": len(common),
                "n_nonfinite_probs": n_nonfinite_probs,
                "deep_mean_auc": mean_or_none([deep[u]["auc"] for u in common]),
                "variants": {},
            }
            for variant in VARIANTS:
                auc_pair = [
                    (per_stu[u][variant]["auc"], deep[u]["auc"])
                    for u in common if per_stu[u][variant]["auc"] is not None
                ]
                auc_deltas = [a - d for a, d in auc_pair]
                brier_deltas = [per_stu[u][variant]["brier"] - deep[u]["brier"] for u in common]
                cell["variants"][variant] = {
                    "delta_auc_mean": mean_or_none(auc_deltas),
                    "ci95_auc": boot_ci(auc_deltas),
                    "n_students_with_auc": len(auc_deltas),
                    "delta_brier_mean": mean_or_none(brier_deltas),
                    "ci95_brier": boot_ci(brier_deltas),
                    "n_fallback_rows": sum(per_stu[u][variant]["n_fallback"] for u in common),
                }
            # anchor: pooled variant must reproduce t1fair3 rc delta + n_paired
            cell["anchor_matches_t1fair"] = (
                len(common) == ref["n_paired"]
                and cell["variants"]["pooled"]["delta_auc_mean"] is not None
                and ref["delta_fair_B1_minus_deep_rc"] is not None
                and abs(cell["variants"]["pooled"]["delta_auc_mean"] - ref["delta_fair_B1_minus_deep_rc"]) <= 0.0001
            )
            res[mtag] = cell

        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/t1_b1_variants.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1B1VAR-DONE")


if __name__ == "__main__":
    main()
