"""LLM-direct column: zero-shot LLM predicts A17 holdout from evidence text.

Per student: prompt = evidence trail (KC names with correct/wrong marks,
compressed) + list of KCs to predict. glm-5.3-flash outputs per-KC next-attempt
correctness. Scored within-student AUC on the same rows as the table/deep.
Reference lines: table 0.546, deep 0.533 (keep-scope, A17).
"""
import json
import sys
import urllib.request

import numpy as np

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc, build_kc_table  # noqa: E402
from baseline_eval import build_rows  # noqa: E402
from transfer_eval import stratified_auc_rows  # noqa: E402
from llm_client import load_keys  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

MAX_STUDENTS = 100
MODEL = "glm-5.3-flash"

SYS = """You are a student-model predictor. Given a student's recent practice trail
(knowledge-component names with correct/wrong marks), predict for each listed
knowledge component the probability (in [0,1]) that this student answers their
NEXT question on it correctly. Base it on the pattern of right/wrong and the
components' relationship. Output strict JSON: {"predictions": {"<KC name>": 0.62, ...}}
covering every listed component. No other text."""


def ask(env, trail_text, targets):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"Trail:\n{trail_text}\n\nPredict for: {targets}"},
        ],
        "max_tokens": 4096,
        "temperature": 0,
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(
        f"{env['GLM_BASE_URL'].rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {env['GLM_API_KEY']}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
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
                d = json.loads(text[s : i + 1])
                return d.get("predictions", {})
    return {}


def main():
    env = load_keys()
    kc_names = build_kc_table("assistments17")
    src = get_data_source(_make_rc("assistments17"))
    samples = load_user_samples(src, fold=0)[:MAX_STUDENTS]
    students = build_rows(samples, None)

    aucs = []
    n_rows = 0
    for st in students[:MAX_STUDENTS]:
        # evidence trail from kc_hist (order lost; use aggregated per-KC marks)
        trail_parts = []
        for kc, hist in st["kc_hist"].items():
            name = kc_names.get(kc, f"KC{kc}")[:40]
            marks = "".join("+" if h else "-" for h in hist[-12:])
            trail_parts.append(f"{name}: {marks}")
        trail = "; ".join(trail_parts[-40:])  # cap prompt size
        target_kcs = sorted({r["kc"] for r in st["rows"]})
        targets = json.dumps({str(kc): kc_names.get(kc, f"KC{kc}") for kc in target_kcs}, ensure_ascii=False)
        try:
            preds = ask(env, trail, targets)
        except Exception as e:  # noqa: BLE001
            print("skip student:", repr(e)[:80], flush=True)
            continue
        rows = []
        for r in st["rows"]:
            name = kc_names.get(r["kc"], f"KC{r['kc']}")
            p = preds.get(name) or preds.get(str(r["kc"]))
            if p is None:
                # try partial key match (LLM may alter casing/spacing)
                for k, v in preds.items():
                    if k.lower().replace(" ", "") == name.lower().replace(" ", ""):
                        p = v
                        break
            if p is not None:
                rows.append({"y": r["y"], "pred": float(p)})
                n_rows += 1
        a = stratified_auc_rows(rows, "pred")
        if a is not None:
            aucs.append(a)

    out = {
        "model": MODEL,
        "n_students_scored": len(aucs),
        "n_rows_scored": n_rows,
        "within_student_auc": round(float(np.mean(aucs)), 4) if aucs else None,
        "references": {"table_keep": 0.546, "deep_keep": 0.533},
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    with open("/root/unikt-fork/research/transfer/results/llm_direct_column.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("LLM-DIRECT-DONE")


if __name__ == "__main__":
    main()
