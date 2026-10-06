"""B2-B4 under the A1-only scope (review item #7: baseline-choice defence)."""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from baseline_eval import build_rows, fit_global_kc_prior  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402


def run_domain(dataset, n_users=300):
    src = get_data_source(_make_rc(dataset))
    samples = load_user_samples(src, fold=0)[:n_users]
    students = build_rows(samples, None)
    prior = fit_global_kc_prior(students)

    # leak-free alpha (LOO, evidence-side; same as before)
    g_sum, g_cnt = defaultdict(float), defaultdict(int)
    for st in students:
        for kc, hist in st["kc_hist"].items():
            g_sum[kc] += float(sum(hist))
            g_cnt[kc] += len(hist)
    ev_scores = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows = []
        for st in students:
            for kc, hist in st["kc_hist"].items():
                if len(hist) >= 6 and g_cnt[kc] > len(hist):
                    pri = (g_sum[kc] - float(sum(hist))) / (g_cnt[kc] - len(hist))
                    for t in range(5, len(hist)):
                        rec = float(np.mean(hist[max(0, t - 5):t]))
                        rows.append({"y": hist[t], "pred": alpha * pri + (1 - alpha) * rec})
        a = stratified_auc_rows(rows, "pred") if rows else None
        ev_scores[alpha] = a if a is not None else 0.5
    alpha_star = max(ev_scores, key=ev_scores.get)

    from sklearn.linear_model import LogisticRegression

    kc_ids = sorted(prior)
    kc_index = {k: i for i, k in enumerate(kc_ids)}

    def feats(kc, rec, att):
        x = [0.0] * (len(kc_ids) + 3)
        if kc in kc_index:
            x[kc_index[kc]] = 1.0
        x[-3] = rec if rec is not None else 0.5
        x[-2] = min(att / 20.0, 1.0)
        x[-1] = 1.0 if rec is None else 0.0
        return x

    Xtr, ytr = [], []
    for st in students:
        for kc, hist in st["kc_hist"].items():
            if kc not in kc_index:
                continue
            for t in range(5, len(hist)):
                Xtr.append(feats(kc, float(np.mean(hist[max(0, t - 5):t])), float(t)))
                ytr.append(hist[t])
    lr = LogisticRegression(max_iter=1000).fit(np.array(Xtr), np.array(ytr))

    b2a, b3a, b4a = [], [], []
    for ws in samples:
        hold = sorted(set(ws.holdout_idx.tolist()))
        # rows with recency features require build_rows row alignment — rebuild
        rows = []
        # evidence-side recent history per KC up to each holdout step
        hist_kc = defaultdict(list)
        hold_set = set(hold)
        feats_rows = []
        for i in range(len(ws.sequence)):
            kc = int(ws.sequence[i])
            y = int(ws.response[i])
            if i in hold_set and i > 0:
                isA1 = int(ws.question[i]) == int(ws.question[i - 1]) and kc != int(ws.sequence[i - 1])
                rec = float(np.mean(hist_kc[kc][-5:])) if len(hist_kc[kc]) >= 3 else None
                feats_rows.append((y, kc, rec, float(len(hist_kc[kc])), isA1))
            else:
                hist_kc[kc].append(y)
        r2, r3, r4 = [], [], []
        for y, kc, rec, att, isA1 in feats_rows:
            if isA1:
                continue
            p1 = prior.get(kc, 0.5)
            p2 = rec if rec is not None else p1
            r2.append({"y": y, "pred": p2})
            r3.append({"y": y, "pred": alpha_star * p1 + (1 - alpha_star) * p2})
            r4.append({"y": y, "pred": float(lr.predict_proba([feats(kc, rec, att)])[0][1])})
        for acc, rr in ((b2a, r2), (b3a, r3), (b4a, r4)):
            a = stratified_auc_rows(rr, "pred")
            if a is not None:
                acc.append(a)
    return {
        "alpha_star": alpha_star,
        "B2_A1excluded": round(float(np.mean(b2a)), 4) if b2a else None,
        "B3_A1excluded": round(float(np.mean(b3a)), 4) if b3a else None,
        "B4_A1excluded": round(float(np.mean(b4a)), 4) if b4a else None,
    }


res = {}
for ds in ["assistments17", "ednet_kt1", "junyi2015"]:
    res[ds] = run_domain(ds)
    print(ds, json.dumps(res[ds], ensure_ascii=False), flush=True)
with open("/root/unikt-fork/research/transfer/results/b234_a1_scope.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("B234-A1-DONE")
