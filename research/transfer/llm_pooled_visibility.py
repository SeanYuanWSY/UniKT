"""Batch37: pooled-AUC visibility of the LLM zero-shot table — is Junyi's signal
metric-limited?

batch28b established: Junyi's LLM coarse partition is real and strong (bucket
marginal spread 0.26, within-student coarse order 71.7%) but per-student AUC
structurally compresses it (cross-bucket pairs only 20% of comparable pairs),
leaving pool_0 p=0.0647. Flip side test: under a POOLED AUC (all eval rows in
one Mann-Whitney pool — counts cross-KC ordering across students), does the
pure-LLM table become significant? Same pool_0 rows/protocol as llm_diff_formal
(tail-100, A1-excluded); permutation null identical in spirit (permute the
KC->LLM-value assignment, rng 500+p, 200 perms; p=(#{perm>=real}+1)/201).
Reports pooled and per-student side by side for all 3 domains.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

CACHE = "/root/unikt-fork/research/transfer/results/k3_difficulty_cache.json"
N_PERM = 200


B28_POOL0_REF = {"assistments17": 0.5246, "ednet_kt1": 0.5496, "junyi2015": 0.5456}


def is_a1(ws, i):
    return int(ws.question[i]) == int(ws.question[i - 1]) and int(ws.sequence[i]) != int(ws.sequence[i - 1])


def main():
    diff = {ds: {int(k): v for k, v in d.items()} for ds, d in json.load(open(CACHE)).items()}
    out = {}
    for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)
        n_eval = min(100, max(10, len(samples) - 60))
        eval_samples = samples[len(samples) - n_eval:]
        llm = diff.get(ds, {})

        # pool_0 rows: (kc, y) per student; per-student AUC for contrast
        per_stu = []
        kc_list = []
        y_list = []
        for ws in eval_samples:
            hold = sorted(set(ws.holdout_idx.tolist()))
            rows = []
            for i in hold:
                if i == 0:
                    continue
                if is_a1(ws, i):
                    continue
                kc_list.append(int(ws.sequence[i]))
                y_list.append(int(ws.response[i]))
                rows.append({"y": int(ws.response[i]), "pred": llm.get(int(ws.sequence[i]), 0.5)})
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                per_stu.append(a)
        kc_arr = np.array(kc_list)
        y_arr = np.array(y_list)

        def pooled_for(assign):
            preds = np.array([assign.get(int(k), 0.5) for k in kc_arr])
            pos, neg = preds[y_arr == 1], preds[y_arr == 0]
            gt = (pos[:, None] > neg[None, :]).sum()
            eq = (pos[:, None] == neg[None, :]).sum()
            return float((gt + 0.5 * eq) / (len(pos) * len(neg)))

        real_pooled = pooled_for(llm)
        kcs_sorted = sorted(llm)
        vals = [llm[k] for k in kcs_sorted]
        ge = 0
        for p in range(N_PERM):
            r2 = np.random.default_rng(500 + p)
            assign = dict(zip(kcs_sorted, r2.permutation(vals)))
            if pooled_for(assign) >= real_pooled:
                ge += 1

        ps_mean = float(np.mean(per_stu))
        # reproduction self-check vs batch28 pool_0 (per-student mean, same rows)
        assert abs(ps_mean - B28_POOL0_REF[ds]) < 5e-4, f"{ds}: per-student {ps_mean:.4f} != b28 {B28_POOL0_REF[ds]}"
        out[ds] = {
            "n_rows": len(kc_arr),
            "n_students_scored": len(per_stu),
            "pooled_auc_llm": round(real_pooled, 4),
            "pooled_p_perm": round((ge + 1) / (N_PERM + 1), 4),
            "ge": ge,
            "per_student_auc_mean": round(ps_mean, 4),
            "per_student_p_perm_ref": {"junyi2015": 0.0647, "ednet_kt1": 0.0448, "assistments17": 0.0995}[ds],
        }
        print(ds, json.dumps(out[ds]), flush=True)

    with open("/root/unikt-fork/research/transfer/results/llm_pooled_visibility.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("POOLEDVIS-DONE")


if __name__ == "__main__":
    main()
