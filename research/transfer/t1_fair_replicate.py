"""Batch 45: EdNet disjoint-subsample replication of the T1 fair recompute.

The batch41/43 fair recompute used samples[:300]. Per-domain availability:
A17 = 274 and Junyi = 252 students — [:300] already IS the full available
set there (no sampling question). EdNet has 683, so its headline cells rest
on one 300-student subsample. This batch reruns the same fair-recompute
logic on the DISJOINT EdNet slice [300:683] (383 students).

Deltas vs t1_fair_rescore.py are exactly four, nothing else:
 (1) datasets restricted to ednet_kt1;
 (2) samples = load_user_samples(...)[300:] (disjoint from [:300]);
 (3) output -> results/t1_fair_replicate.json;
 (4) the [:300]-referencing selfchecks (B1_MEAN_REF vs t1_paired_ci,
     COPY_REF vs b36) are dropped — no external reference exists for a new
     sample; the deterministic neg-identity closure (neg == 1-copy within
     0.001) is kept as the in-batch anchor and works on any sample.
B1 uses prior_fixed2 (run-complete) — the adjudicated-clean construction
from batch43.
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
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = {
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
}
RUNS = {
    "identity_AKT": {"ednet_kt1": "runs/normal/AKT_ednet_kt1_20261003-190537_fold0_bs64"},
    "identity_DKT": {"ednet_kt1": "runs/normal/DKT_ednet_kt1_20261003-190327_fold0_bs128"},
    "transfer_AKT": {"ednet_kt1": "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64"},
    "transfer_DKT": {"ednet_kt1": "runs/normal/DKT_assistments09_20261003-024700_fold0_bs128"},
}
SLICE_START = 300  # disjoint from the batch41/43 [:300] subsample


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def run_ids(q_all):
    """Maximal consecutive runs of identical question ids (see t1_fair_rescore)."""
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


def main():
    out = {}
    for ds in ALIGN:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[SLICE_START:]
        assert samples, "empty disjoint slice"

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

        # old-membership prior kept only as a drift diagnostic
        prior_old = fit_global_kc_prior(build_rows(samples, None))
        res = {
            "n_students": len(samples),
            "n_kc_prior_b1": len(prior_b1),
            "n_kc_prior_old": len(prior_old),
            "prior_drift_max_abs": round(
                float(max(abs(prior_b1.get(k, 0.5) - prior_old.get(k, 0.5)) for k in set(prior_b1) | set(prior_old))),
                4,
            ),
        }

        alignment = {int(k): v for k, v in load_alignment(ALIGN[ds]).items()}

        for mtag, runmap in RUNS.items():
            rm = restore_any(runmap[ds])
            deep = {}
            b1c = {}
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
                    rows_d.append({"y": y, "pred": float(probs[jmap[i]])})
                    rows_b.append({"y": y, "pred": prior_b1.get(int(ws.sequence[i]), 0.5)})
                ad = stratified_auc_rows(rows_d, "pred")
                n_nonfinite_probs += sum(1 for r in rows_d if r["pred"] != r["pred"])
                ab = stratified_auc_rows(rows_b, "pred")
                if ad is not None and ab is not None:
                    deep[ws.user_id] = ad
                    b1c[ws.user_id] = ab
            common = sorted(u for u in deep if u in b1c)
            diffs = np.array([b1c[u] - deep[u] for u in common])
            res[mtag] = {
                "deep_mean_samesupport": mean_or_none([deep[u] for u in common]),
                "B1_mean_samesupport": mean_or_none([b1c[u] for u in common]),
                "n_paired": len(common),
                "delta_fair_B1_minus_deep": mean_or_none(diffs.tolist()) if len(diffs) else None,
                "ci95_fair": boot_ci(diffs) if len(diffs) else None,
                "n_dummy_pos0_excluded": n_dummy_excluded,
                "n_nonfinite_probs": n_nonfinite_probs,
            }
            del rm

        # symmetric-closure attack on the same slice (neg-identity = anchor)
        att = {k: [] for k in ("copy_prev", "neg_prev", "copy_prev2", "neg_prev2")}
        for ws in samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            kept = [i for i in hold if i > 0 and not is_a1(ws, i)]
            y = [int(v) for v in ws.response.tolist()]
            for name, lag, neg in (("copy_prev", 1, False), ("neg_prev", 1, True), ("copy_prev2", 2, False), ("neg_prev2", 2, True)):
                rows = [
                    {"y": y[i], "pred": (1 - y[i - lag]) if neg else y[i - lag]}
                    for i in kept
                    if i >= lag
                ]
                a = stratified_auc_rows(rows, "pred")
                if a is not None:
                    att[name].append(a)
        att_mean = {k: mean_or_none(v) for k, v in att.items()}
        att_mean["sym_max_measured"] = round(
            max(0.5, att_mean.get("copy_prev") or 0.5, att_mean.get("neg_prev") or 0.5, att_mean.get("copy_prev2") or 0.5, att_mean.get("neg_prev2") or 0.5),
            4,
        )
        att_mean["selfcheck_neg_identity"] = (
            att_mean.get("copy_prev") is not None
            and abs((1 - att_mean["copy_prev"]) - att_mean["neg_prev"]) <= 0.001
            and abs((1 - att_mean["copy_prev2"]) - att_mean["neg_prev2"]) <= 0.001
        )
        res["sym_closure_attack"] = att_mean

        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/t1_fair_replicate.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("T1FAIR-REPLICATE-DONE")


if __name__ == "__main__":
    main()
