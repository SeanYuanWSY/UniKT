"""PS-0: aligned source-only table transfer (the missing n=0 point).

POOL-SIZE showed pooled-target tables crush frozen deep transfer at n>=10
target students. The curve lacks n=0: build the B1 table on assistments09
(source) only, map source priors to target KCs through the LLM alignment,
and score target students with ZERO target evidence. Same eval protocol
as poolsize_a1.py (A1-excluded, per-student AUC, 100-student tail) so the
point slots into the existing curve. All three target domains.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_align import load_alignment  # noqa: E402
from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

ALIGN = {
    "assistments17": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
    "junyi2015": "assistments09__to__junyi2015__llm__k3-high__c25k3__rep0",
}


def main():
    # source table from A09 (full fold-0 evidence, no target data)
    src09 = get_data_source(_make_rc("assistments09"))
    s09_samples = load_user_samples(src09, fold=0)
    s09_prior = fit_global_kc_prior(build_rows(s09_samples, None))
    print("a09 kcs with prior:", len(s09_prior), flush=True)

    out = {}
    for ds, align_name in ALIGN.items():
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval:]
        alignment = {int(k): v for k, v in load_alignment(align_name).items()}

        # aligned prior: target KC -> source KC prior
        aligned_prior, unmapped_kcs = {}, set()
        for kc in set(int(c) for ws in eval_samples for c in ws.sequence.tolist()):
            ms = alignment.get(kc, [])
            if ms:
                aligned_prior[kc] = s09_prior.get(ms[0]["source_kc"], 0.5)
            else:
                unmapped_kcs.add(kc)

        aucs, aucs_cov, rows_total, rows_covered = [], [], 0, 0
        for ws in eval_samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            rows, rows_cov = [], []
            for i in hold:
                if i == 0:
                    continue
                isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])
                if isA1:
                    continue
                rows_total += 1
                kc = int(ws.sequence[i])
                # matched scope: 0.5 fallback on unmapped KCs (poolsize-comparable)
                rows.append({"y": int(ws.response[i]), "pred": aligned_prior.get(kc, 0.5)})
                if kc in aligned_prior:
                    rows_covered += 1
                    rows_cov.append({"y": int(ws.response[i]), "pred": aligned_prior[kc]})
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                aucs.append(a)
            a_cov = stratified_auc_rows(rows_cov, "pred")
            if a_cov is not None:
                aucs_cov.append(a_cov)

        if not aucs:
            continue
        arr = np.array(aucs)
        rng = np.random.default_rng(0)
        boots = [float(np.mean(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(5000)]
        res = {
            "source": "assistments09_full_fold0",
            "n_eval": len(eval_samples),
            "n_scored_students": len(aucs),
            "pool_0_mean": round(float(np.mean(arr)), 4),
            "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "pool_0_covered_only": round(float(np.mean(aucs_cov)), 4) if aucs_cov else None,
            "n_covered_students": len(aucs_cov),
            "row_coverage": round(rows_covered / max(rows_total, 1), 4),
            "target_kcs_unmapped": len(unmapped_kcs),
        }
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/ps0_aligned_table.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("PS0-DONE")


if __name__ == "__main__":
    main()
