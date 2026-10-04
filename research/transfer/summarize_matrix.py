"""Aggregate all transfer-matrix cells into one table."""
import glob
import json
import os

os.chdir("/root/unikt-fork/research/transfer/results")
files = sorted(glob.glob("mx_*.json")) + sorted(glob.glob("k3high_*.json"))
print(f"{'cell':46s} {'AUC':>6s} {'drop':>6s} {'p_perm':>7s} {'null':>6s} {'n':>4s}")
for f in files:
    d = json.load(open(f))
    n = f.replace(".json", "")
    a = d.get("within_student_auc")
    dr = d.get("drop_rate")
    p = d.get("perm_p")
    nm = d.get("null_mean")
    ns = d.get("n_students")
    ps = f"{p:.3f}" if p is not None else "-"
    nms = f"{nm:.3f}" if nm is not None else "-"
    print(f"{n:46s} {a:6.4f} {dr:6.3f} {ps:>7s} {nms:>6s} {ns:4d}")
