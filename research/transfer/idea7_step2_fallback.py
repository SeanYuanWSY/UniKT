"""IDEA-7 step 2: LLM difficulty estimates as the table's blind-spot fallback.

B1 currently predicts 0.5 for KCs with no evidence history (blind spots).
Replace that fallback with the LLM zero-shot difficulty estimate (from the
gate run) and measure the within-student AUC delta on each domain's fair
keep-set. A17 also serves as the gate's negative-control prediction.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import build_kc_table  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402
from kc_tables import _make_rc  # noqa: E402

# LLM estimates saved from the gate run: rebuild per-dataset from raw log is
# not needed if we re-ask cheaply; instead re-run estimation inline via cache.
GATE = "/root/unikt-fork/research/transfer/results/idea7_gate.json"
gate = json.load(open(GATE)) if __import__("os").path.exists(GATE) else {}

ALIGN = {
    "assistments17": "assistments09__to__assistments17__llm__k3-high__c25k3__rep0",
    "junyi2015": "assistments09__to__junyi2015__llm__k3-high__c25k3__rep0",
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
}
N_USERS = {"assistments17": 300, "junyi2015": 300, "ednet_kt1": 300}

# Re-derive LLM estimates deterministically via a fixed-seed re-ask is
# unnecessary: re-estimate here by loading the gate's raw estimates if
# persisted; otherwise approximate using KC-name order statistics is NOT
# acceptable — so we re-ask the LLM once per domain (cheap) inside this script.
from idea7_difficulty_gate import ask_llm, true_kc_acc, SUBJECT  # noqa: E402


def main():
    out = {}
    for dataset in ["assistments17", "junyi2015", "ednet_kt1"]:
        src = get_data_source(_make_rc(dataset))
        samples = load_user_samples(src, fold=0)[: N_USERS[dataset]]
        students = build_rows(samples, None)  # FULL set: blind-spot KCs are exactly what we must score
        prior = fit_global_kc_prior(students)
        table = build_kc_table(dataset)
        truth = true_kc_acc(dataset)
        kcs = [k for k in sorted(truth) if k in table]
        names = [table[k] for k in kcs]
        try:
            est = ask_llm(names, SUBJECT[dataset], "kimi", "k3")
        except Exception as e:  # noqa: BLE001
            out[dataset] = {"error": repr(e)[:120]}
            continue
        llm_p = {kcs[i]: est.get(i, 0.5) for i in range(len(kcs))}

        def auc_with(fallback):
            aucs = []
            for st in students:
                rows = [{"y": r["y"], "pred": prior.get(r["kc"], fallback(r["kc"]))} for r in st["rows"]]
                a = stratified_auc_rows(rows, "pred")
                if a is not None:
                    aucs.append(a)
            return round(float(np.mean(aucs)), 4), len(aucs)

        base_auc, n = auc_with(lambda kc: 0.5)
        llm_auc, _ = auc_with(lambda kc: llm_p.get(kc, 0.5))
        out[dataset] = {
            "n_students": n,
            "b1_with_0.5_fallback": base_auc,
            "b1_with_llm_fallback": llm_auc,
            "delta": round(llm_auc - base_auc, 4),
            "gate_rho_k3": gate.get(dataset, {}).get("k3", {}).get("rho"),
        }
        print(dataset, json.dumps(out[dataset], ensure_ascii=False), flush=True)
    with open("/root/unikt-fork/research/transfer/results/idea7_step2.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("IDEA7-STEP2-DONE")


if __name__ == "__main__":
    main()
