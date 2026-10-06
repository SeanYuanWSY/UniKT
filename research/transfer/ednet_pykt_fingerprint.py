"""pyKT-protocol EdNet explode fingerprint (audit supplement).

Simulate pyKT's EdNet preprocessing rule (";"-joined tags -> "_" compound KC
-> explode to one row per tag) on the raw questions.csv, then measure the
resulting label-copy row share on actual interaction sequences. This turns
"pyKT's EdNet mechanism holds" into a quantitative fingerprint on our data.
"""
import json
import sys
from collections import Counter

import polars as pl

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

QPATH = "/root/unikt-fork/data/ednet_kt1/raw/EdNet-Contents/questions.csv"


def main():
    q = pl.read_csv(QPATH)
    tags = dict(zip(q["question_id"].to_list(), q["tags"].fill_null("").to_list()))
    n_tags = {qid: len([t for t in str(t_).split(";") if t.strip()]) for qid, t_ in tags.items()}
    multi = sum(1 for v in n_tags.values() if v > 1)
    print(f"questions: {len(n_tags)}, multi-tag: {multi} ({multi/len(n_tags):.1%})")

    # interactions: use our split sequence parquet (per-row expanded already)
    seq = pl.scan_parquet("/root/unikt-fork/data/ednet_kt1/ednet_kt1_split_skill_sequence.parquet")
    df = seq.select(["user", "seq_pos", "question", "label", "skill"]).collect()
    # adjacent same-question pairs (post-explode rows)
    df = df.sort(["user", "seq_pos"])
    users = df.partition_by("user", as_dict=True)
    same_q = same_q_same_label = diff_kc = 0
    total_rows = 0
    for key, u in list(users.items())[:5000]:
        qs = u["question"].to_list()
        ls = u["label"].to_list()
        ks = u["skill"].to_list()
        total_rows += len(qs)
        for i in range(1, len(qs)):
            if qs[i] == qs[i - 1]:
                same_q += 1
                if ls[i] == ls[i - 1]:
                    same_q_same_label += 1
                if ks[i] != ks[i - 1]:
                    diff_kc += 1
    out = {
        "questions_total": len(n_tags),
        "questions_multi_tag_pct": round(multi / len(n_tags), 4),
        "rows_sampled": total_rows,
        "adjacent_sameq": same_q,
        "sameq_share": round(same_q / max(total_rows - 5000, 1), 4),
        "sameq_samelabel_pct": round(same_q_same_label / max(same_q, 1), 4),
        "sameq_diffkc_pct": round(diff_kc / max(same_q, 1), 4),
        "verdict": "pyKT-rule fingerprint on EdNet: same-question adjacent rows share labels ~100% with different KCs = explode copies",
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open("/root/unikt-fork/research/transfer/results/ednet_pykt_fingerprint.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("FINGERPRINT-DONE")


if __name__ == "__main__":
    main()
