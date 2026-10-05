"""IDEA-5 probe: LLM zero-shot ITEM difficulty from question text (EdNet).

Sample questions with >= 30 evidence attempts, ask glm-5.3 and k3 for a
first-attempt correctness estimate from the question text alone, correlate
(Spearman) with the true evidence accuracy. Gate: rho >= 0.3 continue.
"""
import json
import sys
import urllib.request
from collections import defaultdict

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")

from kc_tables import _make_rc  # noqa: E402
from llm_client import load_keys  # noqa: E402
from restore import load_user_samples  # noqa: E402
from utils.data_process import get_data_source  # noqa: E402


def true_item_acc(min_attempts=30, cap=300):
    src = get_data_source(_make_rc("ednet_kt1"))
    samples = load_user_samples(src, fold=0)[:300]
    agg = defaultdict(list)
    for ws in samples:
        ev = ws.evidence_idx
        for q, y in zip(ws.question[ev].tolist(), ws.response[ev].tolist()):
            agg[int(q)].append(int(y))
    items = {q: float(np.mean(v)) for q, v in agg.items() if len(v) >= min_attempts}
    items = dict(sorted(items.items())[:cap])
    return items


def ask_llm(items_text: list[str], provider: str, model: str) -> list[float]:
    env = load_keys()
    if provider == "glm":
        base, key = env["GLM_BASE_URL"], env["GLM_API_KEY"]
    else:
        base, key = env["KIMI_BASE_URL"], env["KIMI_API_KEY"]
    out = []
    B = 40
    for i in range(0, len(items_text), B):
        chunk = items_text[i : i + B]
        table = "\n".join(f"{j}: {t[:400]}" for j, t in enumerate(chunk))
        sys_prompt = (
            "You estimate the difficulty of English test questions (TOEIC-style) from their metadata (part + tags).\n"
            "For each item id, estimate the fraction of learners who answer it "
            "correctly on the first attempt (a number in [0,1]) from the question "
            "text alone (difficulty of vocabulary, reasoning, traps).\n"
            'Output strict JSON: {"estimates": [{"id": 0, "p": 0.62}, ...]} covering every id.'
        )
        body = {
            "model": model,
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
        with urllib.request.urlopen(req, timeout=240) as resp:
            raw = json.loads(resp.read().decode())
        text = raw["choices"][0]["message"]["content"] or ""
        s = text.find("{")
        depth = 0
        parsed = {}
        for k in range(s, len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    parsed = {int(e["id"]): float(e["p"]) for e in json.loads(text[s : k + 1]).get("estimates", [])}
                    break
        out.extend(parsed.get(j, None) for j in range(len(chunk)))
    return out


def main():
    qdf = pl.read_csv("/root/unikt-fork/data/ednet_kt1/raw/EdNet-Contents/questions.csv")
    q_text = {
        q: f"part {p_}; tags: {t}"
        for q, p_, t in zip(
            qdf["question_id"].to_list(),
            qdf["part"].fill_null("").to_list(),
            qdf["tags"].fill_null("").to_list(),
        )
    }
    items = true_item_acc()
    print(f"items with >=30 evidence attempts: {len(items)}", flush=True)
    qids = sorted(items)
    texts = [q_text.get(str(q), "") or q_text.get(f"q{q}", "") or "" for q in qids]
    keep = [i for i, t in enumerate(texts) if t.strip()]
    print(f"items with question text: {len(keep)}/{len(qids)}", flush=True)
    qids = [qids[i] for i in keep]
    texts = [texts[i] for i in keep]
    truth = [items[q] for q in qids]

    out = {"n_items": len(qids)}
    for provider, model in [("glm", "glm-5.3"), ("kimi", "k3")]:
        est = ask_llm(texts, provider, model)
        pairs = [(e, t) for e, t in zip(est, truth) if e is not None]
        rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
        out[model] = {"rho": round(float(rho), 3), "n": len(pairs)}
        print(model, out[model], flush=True)
    with open("/root/unikt-fork/research/transfer/results/idea5_gate.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("IDEA5-GATE-DONE")


if __name__ == "__main__":
    main()
