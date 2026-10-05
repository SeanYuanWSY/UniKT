"""IDEA-7 gate: can an LLM estimate KC difficulty from the KC NAME alone?

Zero-shot: give the model only the KC name (plus dataset subject context), ask
for a first-attempt correctness probability. Correlate (Spearman) with the
dataset's evidence-segment true KC accuracy. Preregistered gate:
  rho >= 0.3  -> text channel alive, build blind-spot table
  rho < 0.15  -> close the text-difficulty channel
Run on all four datasets with two model tiers (glm-5.3, k3-low).
"""
import json
import sys
import urllib.request

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc, build_kc_table  # noqa: E402
from llm_client import load_keys  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402

SUBJECT = {
    "assistments09": "middle-school math (ASSISTments 2009-10)",
    "assistments17": "middle-school math (ASSISTments 2017)",
    "junyi2015": "school math in Taiwanese (Junyi Academy)",
    "ednet_kt1": "English TOEIC-style test prep (EdNet)",
}


def true_kc_acc(dataset, n_users=400):
    src = get_data_source(_make_rc(dataset))
    samples = load_user_samples(src, fold=0)[:n_users]
    agg = {}
    for ws in samples:
        ev = ws.evidence_idx
        for kc, y in zip(ws.sequence[ev].tolist(), ws.response[ev].tolist()):
            agg.setdefault(int(kc), []).append(int(y))
    return {k: float(np.mean(v)) for k, v in agg.items() if len(v) >= 20}


def ask_llm(kc_names: list[str], subject: str, provider: str, model: str) -> dict:
    env = load_keys()
    if provider == "glm":
        base, key, mdl = env["GLM_BASE_URL"], env["GLM_API_KEY"], model
    else:
        base, key, mdl = env["KIMI_BASE_URL"], env["KIMI_API_KEY"], model
    table = "\n".join(f"{i}: {n}" for i, n in enumerate(kc_names))
    sys_prompt = (
        f"You estimate the difficulty of knowledge components (KCs) in {subject}.\n"
        "For each KC id, estimate the fraction of students who answer a typical "
        "question on it correctly on their first attempt (a number in [0,1]).\n"
        "Base it only on your knowledge of the subject and typical curricula.\n"
        'Output strict JSON: {"estimates": [{"id": 0, "p": 0.62}, ...]} covering every id. No other text.'
    )
    body = {
        "model": mdl,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": table},
        ],
        "max_tokens": 16384,
    }
    if provider == "glm":
        body["temperature"] = 0
        body["thinking"] = {"type": "disabled"}
    else:
        body["thinking"] = {"type": "enabled", "budget_tokens": 2048}
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
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


def main():
    out = {}
    for dataset in ["assistments17", "assistments09", "junyi2015", "ednet_kt1"]:
        table = build_kc_table(dataset)
        truth = true_kc_acc(dataset)
        kcs = [k for k in sorted(truth) if k in table]
        names = [table[k] for k in kcs]
        row = {"n_kc": len(kcs)}
        for provider, model in [("glm", "glm-5.3"), ("kimi", "k3")]:
            try:
                est = ask_llm(names, SUBJECT[dataset], provider, model)
            except Exception as e:  # noqa: BLE001
                row[model] = {"error": repr(e)[:120]}
                continue
            xs = [est.get(i) for i in range(len(kcs))]
            pairs = [(x, truth[k]) for x, k in zip(xs, kcs) if x is not None]
            if len(pairs) >= 10:
                rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
                row[model] = {"rho": round(float(rho), 3), "n": len(pairs)}
            else:
                row[model] = {"error": "parse", "got": len(pairs)}
        out[dataset] = row
        print(dataset, json.dumps(row, ensure_ascii=False), flush=True)
    with open("/root/unikt-fork/research/transfer/results/idea7_gate.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("IDEA7-GATE-DONE")


if __name__ == "__main__":
    main()
