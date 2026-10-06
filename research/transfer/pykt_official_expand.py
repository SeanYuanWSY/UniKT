"""batch20: run pyKT's OFFICIAL KC-expansion on our EdNet 5000-student sample.

Feeds our sample (question->multi-skill relation + interactions) through
pykt.preprocess.split_datasets.extend_multi_concepts — the official
function that expands multi-concept interactions into repeated rows — and
measures the label-copy fingerprint on pyKT's OWN output. Upgrades the
published-pipeline audit from rule-based inference to executed-official-
code evidence.
"""
import importlib.util

import json
import sys

import numpy as np
import pandas as pd
import polars as pl

# direct module load: pykt package __init__ pulls compiled extensions that
# fail under pixi python (CXXABI); split_datasets.py itself needs only
# os/sys/pandas/numpy/json/copy (verified by reviewer smoke test)
_spec = importlib.util.spec_from_file_location(
    "sd", "/root/pykt_lib/pykt/preprocess/split_datasets.py"
)
_sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sd)
extend_multi_concepts = _sd.extend_multi_concepts

DATA = "/root/unikt-fork/data/ednet_kt1"


def main():
    inter = pl.read_parquet(f"{DATA}/ednet_kt1_sequence.parquet")
    qs = pl.read_parquet(f"{DATA}/ednet_kt1_relation_question_skill.parquet")

    # question -> composite concept string "a_b_c" (pyKT: ';' replaced by '_')
    q2tags = qs.group_by("question").agg(pl.col("skill").sort().cast(str))
    q2c = {
        r["question"]: "_".join(r["skill"])
        for r in q2tags.to_dicts()
    }
    n_questions = len(q2c)
    multi_share = sum(1 for c in q2c.values() if "_" in c) / n_questions

    # build pyKT-format df: one row per student, comma-joined sequences
    rows_q, rows_c, rows_r, uids = [], [], [], []
    for (uid,), grp in inter.group_by(["user"], maintain_order=True):
        g = grp.sort("timestamp", maintain_order=True)
        ql = [int(x) for x in g["question"]]
        rows_q.append(",".join(str(x) for x in ql))
        rows_c.append(",".join(q2c.get(x, "-1") for x in ql))
        rows_r.append(",".join(str(int(x)) for x in g["label"]))
        uids.append(int(uid))
    df = pd.DataFrame(
        {
            "uid": uids,
            "questions": rows_q,
            "concepts": rows_c,
            "responses": rows_r,
        }
    )
    effective_keys = {"questions", "concepts", "responses"}
    n_in = sum(len(x.split(",")) for x in df["questions"])

    out_df, out_keys = extend_multi_concepts(df, effective_keys)
    n_out = sum(len(x.split(",")) for x in out_df["questions"])

    # fingerprint on pyKT's OWN output
    sameq_pairs, sameq_same_label = 0, 0
    is_repeat_rows = 0
    for _, row in out_df.iterrows():
        q = [int(x) for x in row["questions"].split(",")]
        r = [int(x) for x in row["responses"].split(",")]
        rep = row["is_repeat"].split(",") if "is_repeat" in row else []
        is_repeat_rows += sum(1 for x in rep if x == "1")
        for i in range(1, len(q)):
            if q[i] == q[i - 1]:
                sameq_pairs += 1
                if r[i] == r[i - 1]:
                    sameq_same_label += 1

    res = {
        "source": "pykt-toolkit 0.0.38 official extend_multi_concepts",
        "students": len(df),
        "interactions_in": n_in,
        "rows_out_after_expand": n_out,
        "expansion_factor": round(n_out / n_in, 4),
        "questions_total": n_questions,
        "multi_concept_question_share": round(multi_share, 4),
        "adjacent_sameq_pairs": sameq_pairs,
        "adjacent_sameq_share_of_rows": round(sameq_pairs / max(n_out - len(df), 1), 4),
        "sameq_samelabel_rate": round(sameq_same_label / max(sameq_pairs, 1), 4),
        "pykt_is_repeat_rows": is_repeat_rows,
        "is_repeat_share": round(is_repeat_rows / max(n_out, 1), 4),
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open("/root/unikt-fork/research/transfer/results/pykt_official_expand.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("PYKT-OFFICIAL-EXPAND-DONE")


if __name__ == "__main__":
    main()
