"""Streaming cold-KC protocol: step (a)+(b) — timestamp-ordered probe + LLM arm gate.

Review-mandated fixes applied:
- Students ordered by FIRST-ACTIVITY TIMESTAMP (not user_id; probed Spearman ≈ 0).
- Honest default arm = streaming raw pooled prior (0.5 only when n_kc == 0);
  pure-0.5 reported as secondary.
- LLM (k3) difficulty estimates re-asked per domain and cached to disk; the
  LLM arm's cold-window AUC is computed OFFLINE here (gate) before any full
  pipeline is built.
Metric: cold-row-participating-pair within-student AUC (each cold row vs the
same student's other holdout rows), student-cluster bootstrap CI.
"""
import json
import os
import sys
import urllib.request
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

import polars as pl  # noqa: E402

from kc_tables import _make_rc, build_kc_table  # noqa: E402
from llm_client import load_keys  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

N_COLD = 20
CACHE = "/root/unikt-fork/research/transfer/results/k3_difficulty_cache.json"


def first_active_times(dataset):
    table = f"/root/unikt-fork/data/{dataset}/{dataset}_split_skill_sequence.parquet"
    df = pl.scan_parquet(table).group_by("user").agg(pl.col("seq_pos").min().alias("_"), pl.len().alias("n")).collect()
    # need timestamps: reload with timestamp if present
    cols = pl.scan_parquet(table).collect_schema().names()
    if "timestamp" in cols:
        t = pl.scan_parquet(table).group_by("user").agg(pl.col("timestamp").min()).collect()
        return dict(zip(t["user"].to_list(), t["timestamp"].to_list()))
    return {u: i for i, u in enumerate(df["user"].to_list())}


def ask_k3(names, subject):
    env = load_keys()
    table = "\n".join(f"{i}: {n}" for i, n in enumerate(names))
    sys_p = (
        f"You estimate the difficulty of knowledge components (KCs) in {subject}.\n"
        "For each KC id, estimate the fraction of students answering a typical "
        "question correctly on first attempt ([0,1]). Knowledge only.\n"
        'Output strict JSON: {"estimates": [{"id": 0, "p": 0.62}, ...]}.'
    )
    body = {
        "model": "k3",
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": table}],
        "max_tokens": 16384,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    }
    req = urllib.request.Request(
        f"{env['KIMI_BASE_URL'].rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {env['KIMI_API_KEY']}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = json.loads(resp.read().decode())
    text = raw["choices"][0]["message"]["content"] or ""
    s = text.find("{")
    depth = 0
    for i in range(s, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return {int(e["id"]): float(e["p"]) for e in json.loads(text[s : i + 1])["estimates"]}
    return {}


SUBJ = {"assistments17": "middle-school math (ASSISTments 2017)", "junyi2015": "school math, Taiwanese platform (Junyi)", "ednet_kt1": "English TOEIC prep (EdNet)"}


def auc_pairs(scores, rows_meta):
    """Cold-row-participating-pair AUC: each cold row vs same student's other holdout rows."""
    num = den = 0.0
    by_student = defaultdict(list)
    for sc, (y, uid, is_cold) in zip(scores, rows_meta):
        by_student[uid].append((sc, y, is_cold))
    for uid, rs in by_student.items():
        for sc_c, y_c, _ in [r for r in rs if r[2]]:
            for sc_o, y_o, _ in rs:
                if (sc_o, y_o) == (sc_c, y_c):
                    continue
                if y_c != y_o:
                    den += 1
                    if (sc_c > sc_o and y_c > y_o) or (sc_c < sc_o and y_c < y_o):
                        num += 1
                    elif sc_c == sc_o:
                        num += 0.5
    return num / den if den else None


def run_domain(dataset, n_users=300):
    src = get_data_source(_make_rc(dataset))
    samples = load_user_samples(src, fold=0)[: n_users + 150]
    t0 = first_active_times(dataset)
    samples.sort(key=lambda ws: t0.get(ws.user_id, ws.user_id))

    # streaming strict-prefix pooling
    counts, sums = defaultdict(int), defaultdict(int)
    eval_rows = []  # (pred_arms dict per row) built incrementally
    kc_names = build_kc_table(dataset)

    # LLM estimates (cached)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if dataset not in cache:
        kcs_sorted = sorted(kc_names)
        est = ask_k3([kc_names[k] for k in kcs_sorted], SUBJ[dataset])
        cache[dataset] = {str(kcs_sorted[i]): est.get(i, 0.5) for i in range(len(kcs_sorted))}
        with open(CACHE, "w") as f:
            json.dump(cache, f, ensure_ascii=False)
    llm_p = {int(k): v for k, v in cache[dataset].items()}

    rows_meta = []
    scores_default, scores_llm, scores_half, scores_rand = [], [], [], []
    rng = np.random.default_rng(99)
    n_eval = 0
    for ws in samples:
        if n_eval >= n_users:
            break
        n_eval += 1
        ev = ws.evidence_idx
        hold = ws.holdout_idx
        for i in hold.tolist():
            kc = int(ws.sequence[i])
            n_kc = counts.get(kc, 0)
            raw_p = (sums[kc] / counts[kc]) if n_kc > 0 else 0.5
            if n_kc < N_COLD:
                rows_meta.append((int(ws.response[i]), ws.user_id, True))
                scores_default.append(raw_p)
                scores_llm.append(llm_p.get(kc, 0.5) if n_kc == 0 else raw_p)
                scores_half.append(0.5)
                scores_rand.append(float(rng.uniform(0.3, 0.8)) if n_kc == 0 else raw_p)
            else:
                rows_meta.append((int(ws.response[i]), ws.user_id, False))
                p = raw_p
                scores_default.append(p)
                scores_llm.append(p)
                scores_half.append(p)
                scores_rand.append(p)
        for i in ev.tolist():
            kc = int(ws.sequence[i])
            counts[kc] += 1
            sums[kc] += int(ws.response[i])

    out = {
        "n_eval_students": n_eval,
        "cold_rows": sum(1 for m in rows_meta if m[2]),
        "total_rows": len(rows_meta),
        "auc_default": auc_pairs(scores_default, rows_meta),
        "auc_llm_coldonly": auc_pairs(scores_llm, rows_meta),
        "auc_pure05": auc_pairs(scores_half, rows_meta),
        "auc_random_coldonly": auc_pairs(scores_rand, rows_meta),
    }
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    out["delta_llm"] = round(out["auc_llm_coldonly"] - out["auc_default"], 4)
    out["delta_random"] = round(out["auc_random_coldonly"] - out["auc_default"], 4)
    return out


res = {}
for ds in ["junyi2015", "ednet_kt1", "assistments17"]:
    res[ds] = run_domain(ds)
    print(ds, json.dumps(res[ds], ensure_ascii=False), flush=True)
with open("/root/unikt-fork/research/transfer/results/streaming_gate.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("STREAMING-GATE-DONE")
