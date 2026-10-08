"""Batch 54: B1 lookup-table sample-size curve (how many students suffice?).

Pair axis to b52: b52 varied WHO feeds the table (self vs others vs first
attempts); this batch varies HOW MANY students feed it. Reviewer-facing
question: "the lookup uses all ~300 students' history — what is its sample
complexity?" If a table built from the first N=25 students keeps most of
the transfer advantage, the lookup result is a cold-start-deployable fact,
not a big-data artifact.

  B1_N = pooled per-KC evidence prior built from samples[:N] only,
         N in {10, 25, 50, 100, 200, 300}; scored on ALL students' rows
         (contributors included — b52 showed self-inclusion moves deltas
         by <=0.003; N=300 anchors to the pooled reference).

Row construction: the single t1_multiplicity.py transfer path (proven
identical to b41/b46/b48/b49 by 12/12 anchors there): run-complete prior
support, A1-only, keep[-200:]/<6, jmap, NaN pair-drop.

Anchors: per cell (12 transfer cells), N=300 delta_auc must reproduce the
same reference JSONs as b53 (t1_fair_rescore / t1_sakt_fair /
t1_dkvmn_fair) within 0.0001 and n_paired equal.
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
SRC09 = {
    "DKT": "runs/normal/DKT_assistments09_20261003-024700_fold0_bs128",
    "AKT": "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64",
    "SAKT": "runs/normal/SAKT_assistments09_20261003-025330_fold0_bs64",
    "DKVMN": "runs/normal/DKVMN_assistments09_20261003-024801_fold0_bs128",
}
REFS = {}
for _name, _path in (
    ("DKT", "t1_fair_rescore.json"),
    ("AKT", "t1_fair_rescore.json"),
    ("SAKT", "t1_sakt_fair.json"),
    ("DKVMN", "t1_dkvmn_fair.json"),
):
    with open(f"/root/unikt-fork/research/transfer/results/{_path}") as f:
        REFS[_name] = json.load(f)

N_LIST = (10, 25, 50, 100, 200, 300)


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


def prior_from(subset):
    """Pooled per-KC evidence prior from a student subset (run-complete support)."""
    hists = {}
    for ws in subset:
        hold = set(ws.holdout_idx.tolist())
        q_all = [int(x) for x in ws.question.tolist()]
        rid = run_ids(q_all)
        hold_runs = {rid[i] for i in hold}
        for i in range(len(ws.sequence)):
            if rid[i] not in hold_runs:
                hists.setdefault(int(ws.sequence[i]), []).append(int(ws.response[i]))
    return {k: float(np.mean(v)) for k, v in hists.items() if v}, len(hists)


def main(n_users=300):
    out = {}
    for ds in ALIGN:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        alignment = {int(k): v for k, v in load_alignment(ALIGN[ds]).items()}

        priors = {}
        n_kcs = {}
        for n in N_LIST:
            priors[n], n_kcs[n] = prior_from(samples[:n])

        for arch in ("DKT", "AKT", "SAKT", "DKVMN"):
            rm = restore_any(SRC09[arch])
            per_stu = {n: {} for n in N_LIST}  # n -> uid -> auc_b1N
            deep_auc = {}
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
                        continue
                    rows_d.append({"y": y, "pred": pd_})
                    base_rows.append({"y": y, "kc": int(ws.sequence[i])})
                ad = stratified_auc_rows(rows_d, "pred")
                if ad is None or not rows_d:
                    continue
                deep_auc[ws.user_id] = ad
                for n in N_LIST:
                    p = priors[n]
                    rows_v = [{"y": b["y"], "pred": p.get(b["kc"], 0.5)} for b in base_rows]
                    av = stratified_auc_rows(rows_v, "pred")
                    if av is not None:
                        per_stu[n][ws.user_id] = av
            del rm

            common = sorted(set(deep_auc))
            ref = REFS[arch][ds][f"transfer_{arch}"]
            cell = {
                "run_dir": SRC09[arch],
                "n_paired": len(common),
                "deep_mean_auc": mean_or_none([deep_auc[u] for u in common]),
                "n_kc_by_N": {str(n): n_kcs[n] for n in N_LIST},
                "by_N": {},
            }
            for n in N_LIST:
                pair_u = sorted(set(per_stu[n]) & set(deep_auc))
                deltas = [per_stu[n][u] - deep_auc[u] for u in pair_u]
                cell["by_N"][str(n)] = {
                    "delta_auc_mean": mean_or_none(deltas),
                    "ci95": boot_ci(deltas),
                    "n_students": len(pair_u),
                }
            full = cell["by_N"][str(N_LIST[-1])]
            cell["anchor_matches_ref"] = (
                len(common) == ref["n_paired"]
                and full["delta_auc_mean"] is not None
                and ref["delta_fair_B1_minus_deep_rc"] is not None
                and abs(full["delta_auc_mean"] - ref["delta_fair_B1_minus_deep_rc"]) <= 0.0001
            )
            key = f"{ds}|{arch}"
            out.setdefault("cells", {})[key] = cell
            print(key, json.dumps({k: v for k, v in cell.items() if k != "n_kc_by_N"}, ensure_ascii=False), flush=True)

    # summary: smallest N per cell with point+CI B1-favored
    summary = {}
    for key, cell in out["cells"].items():
        ok_n = [n for n in N_LIST
                if (lambda c: c["delta_auc_mean"] is not None and c["delta_auc_mean"] > 0 and c["ci95"] and c["ci95"][0] > 0)(cell["by_N"][str(n)])]
        summary[key] = {"min_N_significant": min(ok_n) if ok_n else None,
                        "all_N_point_positive": all(cell["by_N"][str(n)]["delta_auc_mean"] is not None and cell["by_N"][str(n)]["delta_auc_mean"] > 0 for n in N_LIST)}
    out["summary"] = summary
    out["n_anchor_pass"] = sum(1 for c in out["cells"].values() if c["anchor_matches_ref"])

    with open("/root/unikt-fork/research/transfer/results/t1_b1_samplesize.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1B1N-DONE")


if __name__ == "__main__":
    main()
