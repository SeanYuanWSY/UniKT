"""LLM KC alignment between two datasets' knowledge-component tables.

For each target KC: top-k source matches with similarity score, relation type
(equivalent/broader/narrower/related), or no_match. Cached per (source,
target, method, replicate). Baselines: edit-distance string similarity,
popularity (question-count) alignment, occupancy-matched random permutation.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_explain"))

from kc_tables import build_kc_table, kc_table_markdown  # noqa: E402
from llm_client import call_llm  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / "align_cache"

ALIGN_SYSTEM = """你是教育领域的知识点对齐专家。给你两个数据集的知识点（KC）表：源表和目标表。
对目标表的每一个 KC，判断它与源表中哪些 KC 语义对应，输出 JSON：

{"alignments": [
  {"target_kc": 12, "matches": [{"source_kc": 3, "score": 0.9, "relation": "equivalent|broader|narrower|related"}], "no_match": false},
  {"target_kc": 15, "matches": [], "no_match": true}
]}

规则：
- score ∈ [0,1]：1=完全等价（同一知识点），0.7+≈同主题不同侧重，<0.5 不算匹配；
- relation：equivalent=基本同一知识点；broader=目标 KC 涵盖源 KC（目标更粗）；narrower=目标是源的一部分；related=相关但不等同；
- 只有当存在 score≥0.5 的匹配时才给 matches，否则 no_match=true（宁缺毋滥）；
- 必须覆盖目标表的每一个 KC；
- 只输出 JSON，不输出其他文字。"""


def align_chunk_llm(
    source_md: str, target_md: str, provider: str = "glm", replicate: int = 0,
    temperature: float = 0.0, model_override: str | None = None, kimi_effort: str | None = None,
) -> dict:
    messages = [
        {"role": "system", "content": ALIGN_SYSTEM},
        {
            "role": "user",
            "content": f"源表（{source_md.count(chr(10))-2} 个 KC）：\n{source_md}\n\n目标表：\n{target_md}",
        },
    ]
    call = call_llm(
        provider,
        messages,
        purpose="kc_align",
        user_id=replicate,
        condition=f"rep{replicate}",
        max_tokens=16384,
        temperature=temperature,
        model_override=model_override,
        kimi_effort=kimi_effort,
    )
    if not call.parse_ok:
        return {"error": True, "attempts": len(call.attempts)}
    return call.parsed


def align_embedding(
    source_table: dict[int, str],
    target_table: dict[int, str],
    keys_file: str = "/root/.llm-keys.env",
    threshold: float = 0.6,
    top_k: int = 3,
) -> dict[int, list[dict]]:
    """Baseline: Kimi bge_m3 embeddings + cosine similarity (free on plan)."""
    import urllib.request

    from llm_client import load_keys

    env = load_keys(keys_file)
    base, key = env["KIMI_BASE_URL"], env["KIMI_API_KEY"]

    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 64):
            body = json.dumps({"model": "bge_m3_embed", "input": texts[i : i + 64]}).encode()
            req = urllib.request.Request(
                f"{base.rstrip('/')}/embeddings",
                data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                d = json.loads(resp.read().decode())
            out.extend(x["embedding"] for x in d["data"])
        return out

    src_ids = sorted(source_table)
    tgt_ids = sorted(target_table)
    src_emb = embed([source_table[k] for k in src_ids])
    tgt_emb = embed([target_table[k] for k in tgt_ids])

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

    out: dict[int, list[dict]] = {}
    for tk, te in zip(tgt_ids, tgt_emb):
        sims = sorted(((cos(te, se), sk) for se, sk in zip(src_emb, src_ids)), reverse=True)
        out[tk] = [
            {"source_kc": sk, "score": round(s, 4), "relation": "embedding"}
            for s, sk in sims[:top_k]
            if s >= threshold
        ]
    return out


def align_llm(
    source_table: dict[int, str],
    target_table: dict[int, str],
    provider: str = "glm",
    replicate: int = 0,
    chunk_size: int = 25,
    top_k: int = 3,
    temperature: float = 0.0,
    model_override: str | None = None,
    kimi_effort: str | None = None,
) -> tuple[dict[int, list[dict]], list[list[int]]]:
    """LLM alignment: target_kc -> list of {source_kc, score, relation} (sorted,
    filtered, top-k). Returns (alignment, failed_chunks)."""
    source_md = kc_table_markdown(source_table)
    targets = sorted(target_table)
    out: dict[int, list[dict]] = {}
    failed_chunks: list[list[int]] = []
    for i in range(0, len(targets), chunk_size):
        chunk = targets[i : i + chunk_size]
        target_md = kc_table_markdown({k: target_table[k] for k in chunk})
        parsed = align_chunk_llm(source_md, target_md, provider, replicate, temperature, model_override, kimi_effort)
        if parsed.get("error"):  # chunk-level retry once (review P1)
            parsed = align_chunk_llm(source_md, target_md, provider, replicate, temperature, model_override, kimi_effort)
        if parsed.get("error"):
            failed_chunks.append(chunk)
            for k in chunk:
                out[k] = []
            continue
        for row in parsed.get("alignments", []):
            try:
                tk = int(row["target_kc"])
            except (KeyError, TypeError, ValueError):
                continue
            if tk in chunk and not row.get("no_match"):
                ms = []
                for m in row.get("matches", []):
                    try:
                        ms.append(
                            {
                                "source_kc": int(m["source_kc"]),
                                "score": float(m["score"]),
                                "relation": str(m.get("relation", "related")),
                            }
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                # sort by score first, then filter, then slice (review P1)
                ms.sort(key=lambda m: -m["score"])
                out[tk] = [
                    m
                    for m in ms[:top_k]
                    if m["source_kc"] in source_table and m["score"] >= 0.5
                ]
            elif tk in chunk:
                out[tk] = []
        for k in chunk:
            out.setdefault(k, [])
        time.sleep(0.5)
    return out, failed_chunks


def align_edit_distance(
    source_table: dict[int, str], target_table: dict[int, str], top_k: int = 3
) -> dict[int, list[dict]]:
    """Baseline: difflib similarity over names; keeps same no_match threshold."""
    out: dict[int, list[dict]] = {}
    for tk, tname in target_table.items():
        sims = []
        for sk, sname in source_table.items():
            r = difflib.SequenceMatcher(None, tname.lower(), sname.lower()).ratio()
            sims.append((r, sk))
        sims.sort(reverse=True)
        out[tk] = [
            {"source_kc": sk, "score": float(r), "relation": "string"}
            for r, sk in sims[:top_k]
            if r >= 0.5
        ]
    return out


def align_popularity(
    source_pop: dict[int, int], target_pop: dict[int, int]
) -> dict[int, list[dict]]:
    """Baseline: rank KCs by interaction count in each dataset, align by rank."""
    s_rank = sorted(source_pop, key=lambda k: -source_pop[k])
    t_rank = sorted(target_pop, key=lambda k: -target_pop[k])
    out: dict[int, list[dict]] = {}
    for i, tk in enumerate(t_rank):
        out[tk] = (
            [{"source_kc": s_rank[i], "score": 1.0, "relation": "rank"}]
            if i < len(s_rank)
            else []
        )
    return out


def random_occupancy_matched(
    llm_alignment: dict[int, list[dict]], seed: int
) -> dict[int, list[dict]]:
    """Null: permute the LLM's matched source KCs across matched target KCs.

    Same matched/no_match occupancy, same values, shuffled assignment —
    isolates 'what got mapped' from 'how accurately'.
    """
    matched = [tk for tk, ms in llm_alignment.items() if ms]
    sources = [ms[0]["source_kc"] for tk in matched for ms in [llm_alignment[tk]]]
    rng = random.Random(seed)
    rng.shuffle(sources)
    out: dict[int, list[dict]] = {tk: [] for tk in llm_alignment}
    for tk, sk in zip(matched, sources):
        out[tk] = [{"source_kc": sk, "score": 1.0, "relation": "random"}]
    return out


def save_alignment(name: str, alignment: dict, meta: dict | None = None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{name}.json").write_text(
        json.dumps({"alignment": alignment, "meta": meta or {}}, ensure_ascii=False, indent=1)
    )


def load_alignment(name: str) -> dict | None:
    p = CACHE_DIR / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["alignment"] if "alignment" in d else d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source dataset name")
    ap.add_argument("--target", required=True, help="target dataset name")
    ap.add_argument("--method", default="llm", choices=["llm", "edit", "embed"])
    ap.add_argument("--provider", default="glm")
    ap.add_argument("--model", default=None, help="model id override (e.g. k3, glm-5.3)")
    ap.add_argument("--effort", default=None, choices=["low", "high", "max"], help="kimi thinking effort")
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help=">0 for replicate variation (glm only; kimi is temp=1)")
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    src_t = build_kc_table(args.source)
    tgt_t = build_kc_table(args.target)
    print(f"source {args.source}: {len(src_t)} KCs; target {args.target}: {len(tgt_t)} KCs", flush=True)

    if args.method == "embed":
        name = f"{args.source}__to__{args.target}__embed__bge_m3__rep0"
        if load_alignment(name) is None:
            al = align_embedding(src_t, tgt_t)
            save_alignment(name, al)
        matched = sum(1 for v in load_alignment(name).values() if v)
        print(f"[{name}] matched {matched}/{len(load_alignment(name))}", flush=True)
        return

    tag = (args.model or args.provider) + (f"-{args.effort}" if args.effort else "")
    for rep in range(args.replicates):
        name = (
            f"{args.source}__to__{args.target}__{args.method}__{tag}"
            f"__c{args.chunk_size}k{args.top_k}__rep{rep}"
        )
        if load_alignment(name) is not None:
            print(f"cache hit {name}")
            continue
        if args.method == "llm":
            al, failed = align_llm(
                src_t, tgt_t, args.provider, rep,
                chunk_size=args.chunk_size, top_k=args.top_k,
                temperature=args.temperature if rep > 0 else 0.0,
                model_override=args.model, kimi_effort=args.effort,
            )
            save_alignment(name, al, meta={"failed_chunks": failed, "model": args.model, "effort": args.effort})
            matched = sum(1 for v in al.values() if v)
            print(f"[{name}] matched {matched}/{len(al)} failed_chunks={len(failed)}", flush=True)
        else:
            al = align_edit_distance(src_t, tgt_t, top_k=args.top_k)
            save_alignment(name, al)
            matched = sum(1 for v in al.values() if v)
            print(f"[{name}] matched {matched}/{len(al)}", flush=True)


if __name__ == "__main__":
    main()
