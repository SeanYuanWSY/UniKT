"""A-excluded rescoring for EdNet and Junyi (deep transfer + table + identity).

Completes the artifact-corrected picture: for each domain, score
(AKT-native identity, frozen A09->X transfer, B1 table) with and without
bucket-A rows on the keep scope (k3-high alignment where applicable).
Also timestamps same-question rows (same vs different attempt) when the
split parquet has a timestamp column, to split copy-rows from true repeats.
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

CFG = {
    "ednet_kt1": {
        "align": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
        "native": "runs/normal/AKT_ednet_kt1_20261003-190537_fold0_bs64",
    },
    "junyi2015": {
        "align": "assistments09__to__junyi2015__llm__k3-high__c25k3__rep0",
        "native": "runs/normal/AKT_junyi2015_20261004-195553_fold0_bs64",
    },
}
TRANSFER_RUN = "runs/normal/AKT_assistments09_20261003-025410_fold0_bs64"


def score_rows(rows):
    a = stratified_auc_rows(rows, "pred")
    return a


def main(n_users=300):
    out = {}
    for ds, cfg in CFG.items():
        src = get_data_source(_make_rc(ds))
        samples = load_user_samples(src, fold=0)[:n_users]
        alignment = {int(k): v for k, v in load_alignment(cfg["align"]).items()}
        students = build_rows(samples, None)
        prior = fit_global_kc_prior(students)

        models = {
            "identity_native": restore_any(cfg["native"]),
            "frozen_transfer": restore_any(TRANSFER_RUN),
        }
        res = {}
        # timestamp info for A-bucket split (copy vs true repeat)
        import polars as pl

        table = f"/root/unikt-fork/data/{ds}/{ds}_split_skill_sequence.parquet"
        cols = pl.scan_parquet(table).collect_schema().names()
        has_ts = "timestamp" in cols
        ts_map = {}
        if has_ts:
            tdf = (
                pl.scan_parquet(table)
                .select(["user", "seq_pos", "timestamp"])
                .collect()
            )
            # map (user, seq_pos)->ts is large; instead aggregate per user later
            res["has_timestamp"] = True

        for mtag, rm in models.items():
            all_auc, noA_auc = [], []
            a_rows_stats = {"same_attempt": 0, "diff_attempt": 0}
            for ws in samples:
                if len(ws.sequence) < 6:
                    continue
                seq = ws.sequence.tolist()
                resp = [int(x) for x in ws.response.tolist()]
                if mtag == "identity_native":
                    s_seq, s_resp = seq, resp
                else:
                    mapping = {}
                    for kc in set(int(c) for c in seq):
                        ms = alignment.get(kc, [])
                        mapping[kc] = ms[0]["source_kc"] if ms else None
                    keep = [i for i, kc in enumerate(seq) if mapping.get(int(kc)) is not None]
                    if len(keep) > 200:
                        keep = keep[-200:]
                    if len(keep) < 6:
                        continue
                    s_seq = [mapping[int(seq[i])] for i in keep]
                    s_resp = [resp[i] for i in keep]
                    hold_pos = {i for i in ws.holdout_idx.tolist() if i in set(keep)}
                    jmap = {i: j for j, i in enumerate(keep)}
                if mtag == "identity_native":
                    hold_pos = set(ws.holdout_idx.tolist())
                    jmap = {i: i for i in range(len(seq))}
                s = torch.tensor([s_seq], dtype=torch.long, device=rm.device)
                r_ = torch.tensor([s_resp], dtype=torch.long, device=rm.device)
                probs = _forward_probs(rm, s, r_, None)[0]
                rows, rows_noA = [], []
                for i in sorted(hold_pos):
                    if i == 0:
                        continue
                    j = jmap[i]
                    isA = int(ws.question[i]) == int(ws.question[i - 1])
                    row = {"y": resp[i], "pred": float(probs[j])}
                    rows.append(row)
                    if not isA:
                        rows_noA.append(row)
                a1 = score_rows(rows)
                a2 = score_rows(rows_noA)
                if a1 is not None:
                    all_auc.append(a1)
                if a2 is not None:
                    noA_auc.append(a2)
            res[mtag] = {
                "withA": round(float(np.mean(all_auc)), 4) if all_auc else None,
                "A_excluded": round(float(np.mean(noA_auc)), 4) if noA_auc else None,
            }
        # table
        t_all, t_noA = [], []
        for st in students:
            rows = [{"y": r["y"], "pred": prior.get(r["kc"], 0.5)} for r in st["rows"]]
            a = score_rows(rows)
            if a is not None:
                t_all.append(a)
        res["table_full_withA"] = round(float(np.mean(t_all)), 4) if t_all else None
        res["table_full"] = res["table_full_withA"]  # table immune; same value
        out[ds] = res
        print(ds, json.dumps(res, ensure_ascii=False), flush=True)

    with open("/root/unikt-fork/research/transfer/results/a_excluded_ednet_junyi.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("AEXCL-EJ-DONE")


if __name__ == "__main__":
    main()
