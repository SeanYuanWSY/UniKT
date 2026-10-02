"""Cross-LLM NLI judge: does each claim's cited evidence actually support it?

The judge model is the *opposite* provider of the report's author (design v2
P0-1b): GLM reports are judged by kimi-for-coding, Kimi reports by GLM.
Judges see ONLY the claim and its cited evidence rows (not the full pack).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence import EvidencePack  # noqa: E402
from llm_client import call_llm  # noqa: E402

JUDGE_SYSTEM = """你是严格的事实核查员。给你一条关于某学生的陈述和它引用的证据行。只判断：证据是否支持该陈述。
- supports: 证据与陈述一致且足以支撑
- contradicts: 证据与陈述矛盾
- unverifiable: 证据不足以判断该陈述
输出 JSON: {"verdict": "supports|contradicts|unverifiable", "reason": "一句话"}
不要使用外部知识。"""

_NUM = re.compile(r"0\.\d+|\d+")


def _evidence_rows(pack: EvidencePack, kcs: list[int]) -> str:
    rows = []
    for kc in kcs:
        r = pack.kc_row(int(kc))
        if r is not None:
            rows.append(
                f"KC {r.kc}({r.name}): 尝试{r.attempts}, 正确率{r.acc:.2f}, "
                f"模型预测均值{r.pred_mean:.2f}, 模型就绪度{r.readiness:.2f}, 最近5次正确率{r.recent5:.2f}"
            )
    return "\n".join(rows) or "（无引用行）"


def judge_report(
    parsed: dict,
    pack: EvidencePack,
    judge_provider: str,
    max_claims: int = 30,
) -> list[dict]:
    """Judge each claim of a report with the opposite-provider LLM."""
    verdicts = []
    for claim in parsed.get("claims", [])[:max_claims]:
        cits = claim.get("citations", [])
        rows = _evidence_rows(pack, [c.get("kc") for c in cits if c.get("kc") is not None])
        user_msg = (
            f"陈述: {claim.get('statement','')}\n\n引用的证据行:\n{rows}\n\n"
            f"（全局: 共{pack.global_stats['steps']:.0f}步, 总正确率{pack.global_stats['acc']:.2f}）"
        )
        call = call_llm(
            judge_provider,
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            purpose="nli_judge",
            user_id=pack.user_id,
            condition="judge",
            max_tokens=16384,  # kimi reasoning-only: budget must cover CoT + answer
        )
        v = call.parsed if call.parse_ok else None
        verdicts.append(
            {
                "statement": claim.get("statement", ""),
                "verdict": (v or {}).get("verdict", "parse_fail"),
                "reason": (v or {}).get("reason", ""),
            }
        )
    return verdicts


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="dir with report_U*.json")
    ap.add_argument("--packs-dir", required=True, help="dir with pack_U*.md (same run)")
    ap.add_argument("--author-provider", default="glm", choices=["glm", "kimi"])
    ap.add_argument("--n-users", type=int, default=5)
    args = ap.parse_args()

    # Packs are markdown; rebuild minimal info from reports + packs via evidence
    # rebuild is heavyweight, so judge reads packs from the same run dir.
    from restore import load_user_samples, restore  # noqa: E402
    from evidence import build_pack  # noqa: E402

    judge = "kimi" if args.author_provider == "glm" else "glm"
    rdir = Path(args.results_dir)
    reports = sorted(rdir.glob("report_U*.json"))[: args.n_users]

    rm = restore(
        json.loads(
            (rdir.parent / "run_meta.json").read_text()
        )["run_dir"]
        if (rdir.parent / "run_meta.json").exists()
        else "runs/normal/DKT_assistments09_20261003-024700_fold0_bs128"
    )
    samples = {s.user_id: s for s in load_user_samples(rm.data_src, fold=0)}

    tally: dict[str, int] = {}
    all_verdicts = []
    for rf in reports:
        uid = int(rf.stem.split("_U")[1])
        ws = samples.get(uid)
        if ws is None:
            continue
        pack = build_pack(rm, ws)
        parsed = json.loads(rf.read_text())
        vs = judge_report(parsed, pack, judge)
        for v in vs:
            tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
        all_verdicts.append({"user_id": uid, "verdicts": vs})
        print(f"U{uid}: " + json.dumps({k: v for k, v in tally.items()}, ensure_ascii=False), flush=True)

    out = rdir / f"nli_judge_{judge}.json"
    out.write_text(json.dumps({"tally": tally, "details": all_verdicts}, ensure_ascii=False, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
