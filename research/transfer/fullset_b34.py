"""Tree-3 full-set appendix: B3/B4 on the FULL scoring set (deployment view).

POOL-SIZE v1 already produced full-set B1/B2 (0.540/0.562/0.545 and
0.506/0.541/0.522). This completes B3 (global-alpha mix) and B4 (logreg)
on the same full sets, reusing baseline_eval helpers.
"""
import json
import sys

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

    def auc_with(fn):
        aucs = []
        for st in students:
            rows = [{"y": r["y"], "pred": fn(r)} for r in st["rows"]]
            a = stratified_auc_rows(rows, "pred")
            if a is not None:
                aucs.append(a)
        return round(float(np.mean(aucs)), 4), len(aucs)

    # leak-free alpha via student-LOO evidence CV (same as fair-set version)
    from collections import defaultdict

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
        ev_scores[alpha] = round(a, 4) if a is not None else 0.5
    alpha_star = max(ev_scores, key=ev_scores.get)

    b3, n = auc_with(
        lambda r: (alpha_star * prior.get(r["kc"], 0.5) + (1 - alpha_star) * r["recent5"])
        if r["recent5"] is not None
        else prior.get(r["kc"], 0.5)
    )

    # B4 logreg (evidence rows, no tuning)
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
                rec = float(np.mean(hist[max(0, t - 5):t]))
                Xtr.append(feats(kc, rec, float(t)))
                ytr.append(hist[t])
    if len(Xtr) > 200000:
        idx = np.random.default_rng(0).choice(len(Xtr), 200000, replace=False)
        Xtr = [Xtr[i] for i in idx]
        ytr = [ytr[i] for i in idx]
    lr = LogisticRegression(max_iter=1000).fit(np.array(Xtr), np.array(ytr))
    b4, _ = auc_with(
        lambda r: float(lr.predict_proba([feats(r["kc"], r["recent5"], r["attempts"])])[0][1])
    )
    return {"n": n, "B3_alpha": alpha_star, "B3": b3, "B4": b4}


out = {}
for ds in ["assistments17", "junyi2015", "ednet_kt1"]:
    out[ds] = run_domain(ds)
    print(ds, json.dumps(out[ds], ensure_ascii=False), flush=True)
with open("/root/unikt-fork/research/transfer/results/fullset_b34.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("FULLSET-B34-DONE")
