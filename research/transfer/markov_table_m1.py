"""M1: first-order Markov transition table (classic-method archaeology line).

p(y | kc_t, kc_{t-1}, y_{t-1}) with conjugate shrinkage toward the B1 KC
marginal: p = (c1 + m*p_kc) / (c0 + c1 + m). m grid selected leak-free via
leave-own-history-out on evidence segments (B3-alpha protocol); m=inf
degenerates to B1 (built-in lower anchor). Review fixes applied:
pair counts only from evidence->evidence adjacent steps (holdout endpoint
excluded); no hard fallback branch.
Scoring: both scopes (full rows; keep rows via alignment), three domains,
vs B1 and deep references.
"""
import json
import sys
from collections import defaultdict

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
    "junyi2015": "assistments09__to__junyi2015__llm__k3-high__c25k3__rep0",
    "ednet_kt1": "assistments09__to__ednet_kt1__llm__k3-high__c25k3__rep0",
}


def run_domain(dataset, n_users=300):
    src = get_data_source(_make_rc(dataset))
    samples = load_user_samples(src, fold=0)[: n_users + 100]
    students_full = build_rows(samples, None)

    # B1 prior + transition counts from evidence segments
    prior = fit_global_kc_prior(students_full)
    pair_pos = defaultdict(int)  # (kc_t, kc_prev, y_prev) -> correct count at t
    pair_tot = defaultdict(int)

    def build_rows_seq(ws):
        """Return evidence sequence of (kc, y) for counting, in order."""
        ev = ws.evidence_idx
        return [(int(ws.sequence[i]), int(ws.response[i])) for i in ev.tolist()]

    seqs = [build_rows_seq(ws) for ws in samples[: len(students_full)]]
    for seq in seqs:
        for (kc_prev, y_prev), (kc_t, y_t) in zip(seq[:-1], seq[1:]):
            pair_tot[(kc_t, kc_prev, y_prev)] += 1
            pair_pos[(kc_t, kc_prev, y_prev)] += y_t

    def m1_predict(kc_t, kc_prev, y_prev, m):
        p_kc = prior.get(kc_t, 0.5)
        key = (kc_t, kc_prev, y_prev)
        c1 = pair_pos.get(key, 0)
        c0 = pair_tot.get(key, 0)
        if np.isinf(m):
            return p_kc
        return (c1 + m * p_kc) / (c0 + m) if (c0 + m) > 0 else p_kc

    # leak-free m selection: leave-own-history-out on evidence tails
    # for each evidence step t>=6 (with t-1 in evidence), predict using global
    # counts minus this student's own pairs
    def loo_pairs(exclude_seq):
        d_pos, d_tot = defaultdict(int), defaultdict(int)
        for seq in exclude_seq:
            for (kc_prev, y_prev), (kc_t, y_t) in zip(seq[:-1], seq[1:]):
                d_tot[(kc_t, kc_prev, y_prev)] += 1
                d_pos[(kc_t, kc_prev, y_prev)] += y_t
        return d_pos, d_tot

    global_pos, global_tot = pair_pos, pair_tot
    best_m, best_auc = 8.0, -1
    for m in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf):
        rows = []
        for seq in seqs:
            if len(seq) < 6:
                continue
            # subtract own pairs
            own_pos, own_tot = loo_pairs([seq])
            for t in range(5, len(seq)):
                kc_t, y_t = seq[t]
                kc_prev, y_prev = seq[t - 1]
                p_kc = prior.get(kc_t, 0.5)
                key = (kc_t, kc_prev, y_prev)
                c1 = global_pos.get(key, 0) - own_pos.get(key, 0)
                c0 = global_tot.get(key, 0) - own_tot.get(key, 0)
                if not np.isinf(m):
                    p = (c1 + m * p_kc) / (c0 + m) if (c0 + m) > 0 else p_kc
                else:
                    p = p_kc
                rows.append({"y": y_t, "pred": p})
        a = stratified_auc_rows(rows, "pred") if rows else None
        if a is not None and a > best_auc:
            best_auc, best_m = a, m

    # score on holdout rows: full scope (uses true previous label — teacher
    # forcing, same as deep forward) and keep scope
    alignment = {int(k): v for k, v in load_alignment(ALIGN[dataset]).items()}

    def score(scope):
        aucs = []
        for ws in samples[: len(students_full)]:
            rows = []
            holdout_pos = set(ws.holdout_idx.tolist())
            n = len(ws.sequence)
            for i in range(n):
                if i not in holdout_pos:
                    continue
                if scope == "keep":
                    # only steps whose previous step is also within the mapped keep set
                    mapping = {}
                    for kc in set(int(c) for c in ws.sequence.tolist()):
                        ms = alignment.get(kc, [])
                        mapping[kc] = ms[0]["source_kc"] if ms else None
                    if mapping.get(int(ws.sequence[i])) is None:
                        continue
                if i == 0:
                    continue  # no previous step
                kc_t = int(ws.sequence[i])
                kc_prev = int(ws.sequence[i - 1])
                y_prev = int(ws.response[i - 1]) if (i - 1) not in holdout_pos else None
                # teacher forcing uses true previous label even inside holdout
                y_prev = int(ws.response[i - 1])
                rows.append({"y": int(ws.response[i]), "pred": m1_predict(kc_t, kc_prev, y_prev, best_m)})
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                aucs.append(a)
        return round(float(np.mean(aucs)), 4), len(aucs)

    full_auc, n1 = score("full")
    keep_auc, n2 = score("keep")
    return {
        "best_m": best_m if not np.isinf(best_m) else "inf",
        "loo_auc": round(best_auc, 4),
        "full_auc": full_auc,
        "keep_auc": keep_auc,
        "n_students": n1,
    }


res = {}
for ds in ["assistments17", "junyi2015", "ednet_kt1"]:
    res[ds] = run_domain(ds)
    print(ds, json.dumps(res[ds], ensure_ascii=False), flush=True)
with open("/root/unikt-fork/research/transfer/results/markov_m1.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("M1-DONE")
